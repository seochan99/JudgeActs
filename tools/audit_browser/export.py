"""Export audit data for the offline Selection Audit Browser.

Builds tools/audit_browser/data.js (window.AUDIT = {...}) from the frozen
data/run/result files and writes ~256px center-cropped JPEG thumbnails to
tools/audit_browser/thumbs/. All values are copied or recomputed from the
repository's files; nothing is hand-entered.

Usage:  uv run python tools/audit_browser/export.py [--no-thumbs]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THUMBS = HERE / 'thumbs'
EPS = 1e-12


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def thumb(src, dst, size=256):
    from PIL import Image
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    with Image.open(src) as im:
        im = im.convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((size, size), Image.LANCZOS)
        im.save(dst, 'JPEG', quality=82, optimize=True, progressive=True)


def pick_metrics(scores, i):
    best, mean = max(scores), sum(scores) / len(scores)
    s = scores[i]
    return {'utility': s, 'regret': best - s, 'gain': s - mean,
            'below_mean': s < mean - EPS, 'best': s >= best - EPS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-thumbs', action='store_true')
    args = ap.parse_args()

    groups = load_jsonl(ROOT / 'data/derived/main.jsonl')
    runs = {}
    for name in ['qwen', 'smol']:
        runs[name] = {(r['prompt_id'], r['permutation']): r
                      for r in load_jsonl(ROOT / f'runs/{name}_main.jsonl')}
    summary = json.loads((ROOT / 'results/main/summary.json').read_text())
    position = json.loads((ROOT / 'results/main/position.json').read_text())
    country = json.loads((ROOT / 'results/main/country.json').read_text())

    THUMBS.mkdir(exist_ok=True)
    prompts = []
    for g in groups:
        cs = g['candidates']
        ids = [c['id'] for c in cs]
        scores = [c['utility'] for c in cs]
        cands = []
        for c in cs:
            short = c['id'][len(g['prompt_id']) + 1:]
            name = f"{g['prompt_id'][:16]}_{short}.jpg"
            if not args.no_thumbs:
                thumb(ROOT / c['image'], THUMBS / name)
            cands.append({'id': c['id'], 'gen': short, 'source': c['source_model'],
                          'thumb': f'thumbs/{name}', 'utility': c['utility'],
                          'n_utility': c.get('n_utility'), 'ratings': c.get('alignment_ratings'),
                          'stereotype': c.get('stereotype'), 'missing_explicit': c.get('missing_explicit'),
                          'missing_implicit': c.get('missing_implicit'),
                          'image_quality': c.get('image_quality')})
        judges = {}
        for name, run in runs.items():
            orders = []
            for k in range(3):
                r = run.get((g['prompt_id'], k))
                if r is None:
                    orders.append(None)
                    continue
                labels = sorted(r['mapping'])
                orders.append({'perm': k, 'labels': labels,
                               'order': [ids.index(r['mapping'][l]) for l in labels],
                               'parse_ok': bool(r['parse_ok']),
                               'pick': ids.index(r['selected_id']) if r['parse_ok'] and r.get('selected_id') in ids else None})
            picks = [o['pick'] if o else None for o in orders]
            valid = all(p is not None for p in picks)
            j = {'orders': orders, 'pick0': picks[0],
                 'all_valid': valid, 'unanimous': valid and len(set(picks)) == 1,
                 'orders_disagree': valid and len(set(picks)) > 1}
            if picks[0] is not None:
                j['m0'] = pick_metrics(scores, picks[0])
            judges[name] = j
        q0, s0 = judges['qwen']['pick0'], judges['smol']['pick0']
        mean = sum(scores) / len(scores)
        prompts.append({
            'id': g['prompt_id'], 'prompt': g['prompt'], 'country': g['country'].replace('_', ' '),
            'category': g['category'], 'candidates': cands,
            'random_utility': mean, 'oracle_utility': max(scores),
            'best': [i for i, s in enumerate(scores) if s >= max(scores) - EPS],
            'qwen': judges['qwen'], 'smol': judges['smol'],
            'judges_agree': q0 is not None and s0 is not None and q0 == s0,
            'gates': {'unanimity': judges['qwen']['unanimous'],
                      'cross_model': q0 is not None and s0 is not None and q0 == s0},
        })

    keep = ['policy', 'n', 'coverage', 'utility', 'regret', 'regret_ci', 'gain', 'gain_ci',
            'hsr', 'hsr_ci', 'random_hsr', 'human_best', 'bottom_half', 'random_utility', 'oracle_utility']
    audit = {
        'meta': {'n_prompts': len(prompts),
                 'sources': ['data/derived/main.jsonl', 'runs/qwen_main.jsonl', 'runs/smol_main.jsonl',
                             'results/main/summary.json', 'results/main/position.json',
                             'results/main/country.json']},
        'summary': {p['policy']: {k: p.get(k) for k in keep} for p in summary},
        'position': {m: {'calls': position[m]['calls'], 'chi2': position[m]['chi2'],
                         'p_value': position[m]['p_value'], 'slots': position[m]['slots']}
                     for m in ['qwen', 'smol']},
        'country': [{'country': r['country'].replace('_', ' '), 'policy': r['policy'], 'n': r['n'],
                     'regret': r['regret'], 'regret_ci': r.get('regret_ci'), 'gain': r['gain']}
                    for r in country if r['policy'] in ('Qwen', 'Smol', 'Random', 'Qwen unanimous')],
        'prompts': prompts,
    }
    (HERE / 'data.js').write_text('window.AUDIT = ' + json.dumps(audit, separators=(',', ':')) + ';\n')
    print(f'wrote data.js ({len(prompts)} prompts), thumbs: {len(list(THUMBS.glob("*.jpg")))}')


if __name__ == '__main__':
    main()
