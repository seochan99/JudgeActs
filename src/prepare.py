"""Validate image/annotation joins, decode every image, freeze prompt-level splits."""
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from .common import ROOT, SEED, digest, orders, save_json, save_jsonl

def aggregate(item):
    anns = item['human_annotations']
    specs = {'utility': ('prompt_alignment', 'score'),
             'missing_explicit': ('prompt_alignment', 'missing_explicit'),
             'missing_implicit': ('prompt_alignment', 'missing_implicit'),
             'stereotype': ('stereotype', 'is_stereotypical'),
             'image_quality': ('image_quality', 'score'), 'overall': ('overall_score',)}
    out = {'n_annotations': len(anns)}
    for name, keys in specs.items():
        vals = []
        for a in anns:
            v = a
            for key in keys:
                v = v.get(key) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and np.isfinite(v):
                val = float(v)
                if name in ['utility', 'image_quality'] and val not in [0, .5, 1]:
                    raise ValueError(f'Unexpected {name} score: {val}')
                if name == 'overall':
                    if not 1 <= val <= 5:
                        raise ValueError(f'Unexpected overall score: {val}')
                    val = (val-1)/4
                vals.append(val)
        out[name] = float(np.mean(vals)) if vals else None
        out[f'n_{name}'] = len(vals)
        if name == 'utility':
            out['alignment_ratings'] = vals
    return out

def build():
    raw = ROOT/'data/raw/culturalframes'
    anns = json.loads((raw/'culturalframes_human_annotations.json').read_text())
    counts = Counter(a['id'] for a in anns)
    assert max(counts.values()) == 1, 'duplicate annotation IDs'
    by_id = {a['id']: a for a in anns}
    groups = defaultdict(list)
    image_ids = set()
    hashes = defaultdict(list)
    audit = {'annotation_records': len(anns), 'annotation_count': sum(len(a['human_annotations']) for a in anns),
             'rows': 0, 'missing_annotation_rows': 0, 'broken_rows': [], 'join_mismatches': [], 'duplicate_ids': []}
    images = ROOT/'data/images'
    images.mkdir(parents=True, exist_ok=True)
    for shard in sorted((raw/'data').glob('*.parquet')):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=24):
            for row in batch.to_pylist():
                audit['rows'] += 1
                rid = row['id']
                if rid in image_ids:
                    audit['duplicate_ids'].append(rid)
                    continue
                image_ids.add(rid)
                blob = row['image']['bytes']
                hashes[hashlib.sha256(blob).hexdigest()].append(rid)
                try:
                    image = Image.open(io.BytesIO(blob))
                    image.load()
                    # Preserve original encoded bytes; only inference resizes decoded pixels.
                    (images/f'{rid}.img').write_bytes(blob)
                except Exception as exc:
                    audit['broken_rows'].append({'id': rid, 'error': str(exc)})
                    continue
                a = by_id.get(rid)
                if a is None:
                    audit['missing_annotation_rows'] += 1
                    continue
                if a['prompt'] != row['prompt'] or a['country'] != row['country'] or not rid.endswith('_'+a['model_name']):
                    audit['join_mismatches'].append(rid)
                    continue
                scores = aggregate(a)
                if scores['utility'] is None:
                    audit.setdefault('missing_utility_rows', []).append(rid)
                    continue
                pid = rid.rsplit('_', 1)[0]
                groups[pid].append({'id': rid, 'source_model': row['model'], 'prompt': row['prompt'],
                                    'country': row['country'], 'category': a['category'],
                                    'image': str((images/f'{rid}.img').relative_to(ROOT)), **scores})
        print('validated', shard.name, audit['rows'], flush=True)
    assert not audit['join_mismatches'], audit['join_mismatches']
    assert not audit['duplicate_ids'], audit['duplicate_ids']
    eligible = []
    for pid, candidates in sorted(groups.items()):
        if len(candidates) < 3:
            continue
        for key in ['prompt', 'country', 'category']:
            assert len({c[key] for c in candidates}) == 1, (pid, key)
        candidates.sort(key=lambda c: c['id'])
        eligible.append({'prompt_id': pid, 'prompt': candidates[0]['prompt'], 'country': candidates[0]['country'],
                         'category': candidates[0]['category'],
                         'candidates': [{k:v for k,v in c.items() if k not in ['prompt','country','category']} for c in candidates]})
    audit['eligible_before_prompt_deduplication'] = len(eligible)
    seen_prompts = set()
    deduplicated = []
    audit['excluded_duplicate_prompt_ids'] = []
    for g in eligible:
        key = (g['country'], g['prompt'])
        if key in seen_prompts:
            audit['excluded_duplicate_prompt_ids'].append(g['prompt_id'])
            continue
        seen_prompts.add(key)
        deduplicated.append(g)
    eligible = deduplicated
    audit.update({'eligible_prompts': len(eligible), 'country_counts': dict(Counter(g['country'] for g in eligible)),
                  'candidate_count_distribution': dict(Counter(len(g['candidates']) for g in eligible)),
                  'annotation_missing_rate': audit['missing_annotation_rows']/audit['rows'],
                  'duplicate_image_content': [v for v in hashes.values() if len(v)>1],
                  'unjoined_annotation_ids': sorted(set(by_id)-image_ids),
                  'constant_utility_sets': int(sum(np.ptp([c['utility'] for c in g['candidates']])==0 for g in eligible)),
                  'unique_prompt_texts': len({g['prompt'] for g in eligible})})
    save_json(ROOT/'results/data_audit.json', audit)
    save_jsonl(ROOT/'data/derived/groups.jsonl', eligible)
    # Freeze split identities without consulting any judge output or score.
    dev, main, reserve = [], [], []
    country_groups = defaultdict(list)
    for g in eligible:
        country_groups[g['country']].append(g)
    assert len(country_groups) == 10
    for country, gs in sorted(country_groups.items()):
        gs.sort(key=lambda g: digest(f'{SEED}:split:{g["prompt_id"]}'))
        assert len(gs) >= 33, (country, len(gs))
        dev.extend(gs[:3]); main.extend(gs[3:33]); reserve.extend(gs[33:])
    for name, selected in [('dev',dev), ('main',main), ('reserve',reserve)]:
        save_jsonl(ROOT/f'data/derived/{name}.jsonl', selected)
        manifest = {'seed': SEED, 'n': len(selected), 'ids': [g['prompt_id'] for g in selected],
                    'orders': {g['prompt_id']: orders(g) for g in selected}}
        target = ROOT/f'data/manifests/{name}.json'
        if target.exists():
            assert json.loads(target.read_text()) == manifest, 'refusing split mutation'
        save_json(target, manifest)
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == '__main__':
    build()
