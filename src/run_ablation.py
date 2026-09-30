"""POST-HOC position-bias ablations for the primary Qwen judge (added after the frozen main run).

Frozen inputs (runs/*.jsonl, data/, configs/, prompts/, src/run_judge.py) are only read, never written.
Every ablation reuses the frozen input builder / decoding settings and changes exactly one factor:
  rotation4      4-candidate pools only: the missing 4th cyclic rotation (permutation 3), frozen protocol.
  choice_only    Smol's choice-only output contract (no ranking, no "choice = first of ranking" sentence).
  opaque_labels  frozen instruction/schema, but position-free 2-character labels instead of A/B/C/D.
Outputs: runs/ablations/<name>.jsonl (+ .meta.json); resumable by (prompt_id, permutation).
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .common import ROOT, SEED, digest, load_jsonl, orders, strict_parse, save_json
from .run_judge import messages_for
from .run_smol import parse_selection

OUT = ROOT/'runs/ablations'
IMAGE_SIZE, MAX_TOKENS = 448, 64
# No A-D (letter-order meaning), no I/O (confusable with digits).
OPAQUE_LETTERS = 'EFGHJKLMNPQRSTUVWXYZ'
OPAQUE_DIGITS = '23456789'

def rotation4_order(group):
    ids = [c['id'] for c in group['candidates']]
    used = {tuple(o) for o in orders(group)}
    missing = [ids[s:]+ids[:s] for s in range(len(ids)) if tuple(ids[s:]+ids[:s]) not in used]
    assert len(missing) == 1
    return missing[0]

def opaque_labels(prompt_id, k, n):
    rng = random.Random(int(digest(f'{SEED}:{prompt_id}:{k}:opaque')[:16], 16))
    letters = rng.sample(OPAQUE_LETTERS, n)
    labels = [l+rng.choice(OPAQUE_DIGITS) for l in letters]
    assert len(set(labels)) == n
    return labels

def messages_with_labels(group, order, instruction, size, labels):
    """Byte-for-byte copy of run_judge.messages_for (ranking contract) with the label strings swapped."""
    by_id = {c['id']:c for c in group['candidates']}
    content = [{'type':'text', 'text':instruction+'\n\nUser prompt:\n'+group['prompt']}]
    images = []
    for label, cid in zip(labels, order):
        im = Image.open(ROOT/by_id[cid]['image']).convert('RGB')
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        images.append(im)
        content.extend([{'type':'text','text':f'Candidate {label}:'}, {'type':'image','image':im}])
    contract = 'Return exactly two JSON keys: choice (your selected label) and ranking (all labels ordered from best to worst). The first ranking label must equal choice. Rank by the images, not their presentation order.'
    content.append({'type':'text','text':f'Allowed candidate labels: {", ".join(labels)}. '+contract})
    return [{'role':'user','content':content}], images

def strict_choice_only(raw, labels):
    data = json.loads(raw.strip())
    if not isinstance(data, dict) or set(data) != {'choice'} or data['choice'] not in labels:
        raise ValueError('expected one valid choice key')
    return data

def jobs(name, groups, primary, choice_instruction):
    """Yield (group, permutation, labels, order, messages builder, parser)."""
    for g in groups:
        if name == 'rotation4':
            if len(g['candidates']) != 4:
                continue
            order = rotation4_order(g)
            yield g, 3, list('ABCD'), order, lambda g=g,o=order: messages_for(g, o, primary, IMAGE_SIZE), strict_parse
            continue
        for k, order in enumerate(orders(g)[:3]):
            if name == 'choice_only':
                labels = list('ABCD'[:len(order)])
                yield g, k, labels, order, lambda g=g,o=order: messages_for(g, o, choice_instruction, IMAGE_SIZE, choice_only=True), strict_choice_only
            elif name == 'opaque_labels':
                labels = opaque_labels(g['prompt_id'], k, len(order))
                yield g, k, labels, order, lambda g=g,o=order,l=labels: messages_with_labels(g, o, primary, IMAGE_SIZE, l), strict_parse

def self_check(groups, primary):
    """The copied builder with A-D labels must reproduce the frozen builder exactly (text parts)."""
    g = groups[0]; o = orders(g)[0]
    a, _ = messages_for(g, o, primary, IMAGE_SIZE)
    b, _ = messages_with_labels(g, o, primary, IMAGE_SIZE, list('ABCD'[:len(o)]))
    strip = lambda m: [c.get('text') for c in m[0]['content']]
    assert strip(a) == strip(b), 'opaque builder diverges from frozen builder'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ablations', nargs='+', default=['rotation4','choice_only','opaque_labels'],
                   choices=['rotation4','choice_only','opaque_labels'])
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    groups = load_jsonl(ROOT/'data/derived/main.jsonl')
    if args.limit:
        groups = groups[:args.limit]
    primary = (ROOT/'prompts/judge_primary.txt').read_text()
    # Identical to the Smol choice-only instruction (src/run_smol.py / run_judge --model smol).
    choice_instruction = primary.split('The candidate labels')[0] + 'The candidate labels are supplied next to each image. Return compact JSON only, with exactly one key choice containing your selected label. Do not explain your decision.\n'
    self_check(groups, primary)
    sources = json.loads((ROOT/'provenance/sources.json').read_text())
    torch.manual_seed(SEED); torch.set_num_threads(8)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('loading qwen', device, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(ROOT/'models/qwen', dtype=torch.bfloat16,
                                                      attn_implementation='sdpa', local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(ROOT/'models/qwen', local_files_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.ablations:
        out = OUT/f'{name}.jsonl'
        instruction = choice_instruction if name == 'choice_only' else primary
        metadata = {'ablation':name, 'post_hoc':True, 'model':sources['models']['qwen'], 'split':'main',
                    'runner_sha256':digest(Path(__file__).read_text()),
                    'input_builder_sha256':digest((ROOT/'src/run_judge.py').read_text()),
                    'parser_sha256':digest((ROOT/'src/common.py').read_text()),
                    'prompt_sha256':digest(instruction),
                    'split_sha256':digest((ROOT/'data/manifests/main.json').read_text()),
                    'image_size':IMAGE_SIZE, 'max_tokens':MAX_TOKENS, 'dtype':'bfloat16', 'backend':'transformers-mps',
                    'output_schema':'choice_only' if name=='choice_only' else 'choice_and_ranking',
                    'labels':'opaque seeded 2-char codes' if name=='opaque_labels' else 'ABCD',
                    'permutations':[3] if name=='rotation4' else [0,1,2],
                    'torch':torch.__version__, 'do_sample':False}
        meta_path = out.with_suffix('.meta.json')
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            # Runner edits that do not change the call protocol are allowed on resume.
            assert {k:v for k,v in old.items() if k!='runner_sha256'} == {k:v for k,v in metadata.items() if k!='runner_sha256'}, 'incompatible resume'
        save_json(meta_path, metadata)
        done = {(r['prompt_id'], r['permutation']) for r in load_jsonl(out)} if out.exists() else set()
        todo = [j for j in jobs(name, groups, primary, choice_instruction) if (j[0]['prompt_id'], j[1]) not in done]
        print(f'[{name}] done={len(done)} todo={len(todo)}', flush=True)
        started = time.monotonic()
        for n, (g, k, labels, order, build, parse) in enumerate(todo, 1):
            mapping = dict(zip(labels, order))
            record = {'prompt_id':g['prompt_id'], 'model':'qwen', 'ablation':name, 'permutation':k,
                      'mapping':mapping, 'positions':{l:i for i,l in enumerate(labels)}, 'parse_ok':False,
                      'utc':datetime.now(timezone.utc).isoformat(), 'attempts':[]}
            messages, images = build()
            record['image_count'] = len(images)
            tick = time.monotonic()
            for attempt in range(2):
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=images, return_tensors='pt')
                    record['input_shapes'] = {key:list(v.shape) for key,v in inputs.items() if hasattr(v,'shape')}
                    record['input_text_sha256'] = digest(text)
                    inputs = inputs.to(device)
                    for key in inputs:
                        if torch.is_floating_point(inputs[key]):
                            inputs[key] = inputs[key].to(torch.bfloat16)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_TOKENS, use_cache=True)
                    raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                    record['raw_output'] = raw; record['raw_output_sha256'] = digest(raw)
                    try:
                        parsed = parse(raw, labels)
                        record.update({'parse_ok':True, 'parsed':parsed, 'selected_id':mapping[parsed['choice']]})
                    except (ValueError, TypeError) as exc:
                        record['parse_error'] = str(exc)
                        if name == 'choice_only':
                            # Diagnostic only; never used as a selection.
                            try: record['lenient_choice'] = parse_selection(raw, labels)[0]['choice']
                            except ValueError: pass
                    break
                except Exception as exc:
                    record['attempts'].append({'attempt':attempt, 'error':repr(exc)})
                    if device == 'mps': torch.mps.empty_cache()
            record['seconds'] = time.monotonic()-tick
            with out.open('a') as f:
                f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
            elapsed = time.monotonic()-started
            eta = elapsed/n*(len(todo)-n)
            print(f'[{name}] {n}/{len(todo)} {g["prompt_id"][:10]} p{k} ok={record["parse_ok"]} '
                  f'{record.get("raw_output", record["attempts"])!r} {record["seconds"]:.1f}s eta={eta/60:.1f}min', flush=True)
            if device == 'mps': torch.mps.empty_cache()
        print(f'[{name}] finished {time.monotonic()-started:.0f}s', flush=True)

if __name__ == '__main__':
    main()
