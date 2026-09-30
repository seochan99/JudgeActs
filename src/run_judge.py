"""Local greedy multi-image inference. Metadata and human scores never enter messages."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .common import ROOT, digest, load_jsonl, orders, strict_parse, save_json

def messages_for(group, order, instruction, size):
    by_id = {c['id']:c for c in group['candidates']}
    content = [{'type':'text', 'text':instruction+'\n\nUser prompt:\n'+group['prompt']}]
    images = []
    for label, cid in zip('ABCD', order):
        im = Image.open(ROOT/by_id[cid]['image']).convert('RGB')
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        images.append(im)
        content.extend([{'type':'text','text':f'Candidate {label}:'}, {'type':'image','image':im}])
    labels = list('ABCD'[:len(order)])
    content.append({'type':'text','text':f'Return {{"choice":"{labels[0]}","ranking":{json.dumps(labels)}}} with your chosen label and complete ranking.'})
    return [{'role':'user','content':content}], images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen','smol'], default='qwen')
    parser.add_argument('--split', choices=['dev','main'], default='dev')
    parser.add_argument('--permutations', type=int, default=3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--max-tokens', type=int, default=64)
    args = parser.parse_args()
    out = ROOT/f'runs/{args.model}_{args.split}.jsonl'
    out.parent.mkdir(exist_ok=True)
    existing = load_jsonl(out) if out.exists() else []
    done = {(r['prompt_id'],r['permutation']) for r in existing}
    groups = load_jsonl(ROOT/f'data/derived/{args.split}.jsonl')
    if args.limit:
        groups = groups[:args.limit]
    instruction = (ROOT/'prompts/judge_primary.txt').read_text()
    sources = json.loads((ROOT/'provenance/sources.json').read_text())
    metadata = {'model':sources['models'][args.model], 'split':args.split,
                'prompt_sha256':digest(instruction), 'split_sha256':digest((ROOT/f'data/manifests/{args.split}.json').read_text()),
                'image_size':args.image_size, 'max_tokens':args.max_tokens, 'dtype':'bfloat16', 'backend':'transformers-mps',
                'torch':torch.__version__, 'do_sample':False}
    meta_path = out.with_suffix('.meta.json')
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == metadata, 'incompatible resume configuration'
    save_json(meta_path, metadata)
    torch.manual_seed(20261002)
    torch.set_num_threads(8)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('loading', args.model, device, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(ROOT/'models'/args.model, dtype=torch.bfloat16,
                                                      attn_implementation='sdpa', local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(ROOT/'models'/args.model, local_files_only=True)
    started = time.monotonic()
    for group in groups:
        for k, order in enumerate(orders(group)[:args.permutations]):
            if (group['prompt_id'], k) in done:
                continue
            record = {'prompt_id':group['prompt_id'],'model':args.model,'permutation':k,
                      'mapping':dict(zip('ABCD', order)), 'parse_ok':False,
                      'utc':datetime.now(timezone.utc).isoformat(), 'attempts':[]}
            messages, images = messages_for(group, order, instruction, args.image_size)
            tick = time.monotonic()
            for attempt in range(2):
                try:
                    # Explicit images + rendered template make image count auditable.
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=images, return_tensors='pt')
                    record['input_shapes'] = {key:list(value.shape) for key,value in inputs.items() if hasattr(value,'shape')}
                    record['image_count'] = len(images)
                    record['input_text_sha256'] = digest(text)
                    inputs = inputs.to(device)
                    for key in inputs:
                        if torch.is_floating_point(inputs[key]):
                            inputs[key] = inputs[key].to(torch.bfloat16)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens,
                                                   use_cache=True)
                    raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                    record['raw_output'] = raw
                    record['raw_output_sha256'] = digest(raw)
                    try:
                        parsed = strict_parse(raw, list(record['mapping']))
                        record.update({'parse_ok':True,'parsed':parsed, 'selected_id':record['mapping'][parsed['choice']]})
                    except (ValueError, TypeError) as exc:
                        record['parse_error'] = str(exc)
                    # Invalid JSON is retained as an abstention; no repeated sampling.
                    break
                except Exception as exc:
                    record['attempts'].append({'attempt':attempt, 'error':repr(exc)})
                    if device == 'mps':
                        torch.mps.empty_cache()
            record['seconds'] = time.monotonic()-tick
            with out.open('a') as f:
                f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
            print(args.split, group['prompt_id'][:10], k, record['parse_ok'],
                  record.get('raw_output', record['attempts']), f'{record["seconds"]:.1f}s', flush=True)
            if device == 'mps':
                torch.mps.empty_cache()
    print('finished seconds', time.monotonic()-started, flush=True)

if __name__ == '__main__':
    main()
