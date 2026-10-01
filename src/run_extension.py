"""POST-HOC scale extension: Qwen3-VL-8B-Instruct under the frozen main Qwen protocol.

Frozen inputs (runs/*.jsonl, data/, configs/, prompts/, src/run_judge.py, provenance/sources.json) are only read.
The input builder (messages_for), cyclic orders, strict parser, instruction, 448-px bounding box, greedy decoding
and 64 new tokens are imported / copied unchanged from the frozen runner; only the checkpoint differs.
Output: runs/extension/qwen8b_main.jsonl (+ .meta.json), resumable by (prompt_id, permutation).
"""
import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from .common import ROOT, SEED, digest, load_jsonl, orders, strict_parse, save_json
from .run_judge import messages_for

NAME = 'qwen8b'
MODEL = {'id': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
         'license': 'apache-2.0'}
OUT = ROOT/'runs/extension'/f'{NAME}_main.jsonl'
IMAGE_SIZE, MAX_TOKENS, PERMUTATIONS = 448, 64, 3

def versions():
    import PIL, tokenizers, huggingface_hub, transformers
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
            'tokenizers': tokenizers.__version__, 'huggingface_hub': huggingface_hub.__version__,
            'pillow': PIL.__version__, 'platform': platform.platform()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--max-pixels', type=int, help='last-resort processor max_pixels override (recorded)')
    p.add_argument('--limit', type=int, help='number of calls to run in this invocation (smoke test)')
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)
    groups = load_jsonl(ROOT/'data/derived/main.jsonl')
    instruction = (ROOT/'prompts/judge_primary.txt').read_text()
    model_dir = ROOT/'models'/NAME
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    metadata = {'post_hoc': True, 'extension': 'scale', 'model': MODEL, 'split': 'main',
                'runner_sha256': digest(Path(__file__).read_text()),
                'input_builder_sha256': digest((ROOT/'src/run_judge.py').read_text()),
                'parser_sha256': digest((ROOT/'src/common.py').read_text()),
                'prompt_sha256': digest(instruction),
                'split_sha256': digest((ROOT/'data/manifests/main.json').read_text()),
                'chat_template_sha256': digest((model_dir/'chat_template.json').read_text()),
                'image_size': IMAGE_SIZE, 'max_tokens': MAX_TOKENS, 'permutations': list(range(PERMUTATIONS)),
                'dtype': args.dtype, 'device': device, 'backend': 'transformers-' + device,
                'max_pixels_override': args.max_pixels,
                'output_schema': 'choice_and_ranking', 'do_sample': False, 'seed': SEED,
                'versions': versions()}
    meta_path = OUT.with_suffix('.meta.json')
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        # Runner edits that leave the call protocol unchanged (logging etc.) are allowed on resume.
        skip = {'runner_sha256', 'versions'}
        assert {k: v for k, v in old.items() if k not in skip} == {k: v for k, v in metadata.items() if k not in skip}, \
            'incompatible resume configuration'
        assert old['versions']['torch'] == metadata['versions']['torch'] and \
            old['versions']['transformers'] == metadata['versions']['transformers'], 'library change on resume'
        metadata['started_utc'] = old.get('started_utc')
    else:
        metadata['started_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(meta_path, metadata)

    done = {(r['prompt_id'], r['permutation']) for r in load_jsonl(OUT)} if OUT.exists() else set()
    todo = [(g, k, o) for g in groups for k, o in enumerate(orders(g)[:PERMUTATIONS]) if (g['prompt_id'], k) not in done]
    total = len(groups)*PERMUTATIONS
    if args.limit:
        todo = todo[:args.limit]
    print(f'[{NAME}] done={len(done)}/{total} todo={len(todo)} dtype={args.dtype} device={device}', flush=True)
    if not todo:
        print(f'[{NAME}] COMPLETE', flush=True)
        return

    torch.manual_seed(SEED); torch.set_num_threads(8)
    t0 = time.monotonic()
    # Load straight onto the device to avoid a transient second copy of the 17 GB weights in RAM.
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype, attn_implementation='sdpa',
                                                        local_files_only=True, device_map={'': device}).eval()
    pkw = {'max_pixels': args.max_pixels} if args.max_pixels else {}
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, **pkw)
    print(f'[{NAME}] loaded in {time.monotonic()-t0:.0f}s', flush=True)

    recent = []
    started = time.monotonic()
    for n, (g, k, order) in enumerate(todo, 1):
        record = {'prompt_id': g['prompt_id'], 'model': NAME, 'permutation': k,
                  'mapping': dict(zip('ABCD', order)), 'parse_ok': False,
                  'utc': datetime.now(timezone.utc).isoformat(), 'attempts': []}
        messages, images = messages_for(g, order, instruction, IMAGE_SIZE)
        record['image_count'] = len(images)
        tick = time.monotonic()
        for attempt in range(2):
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=images, return_tensors='pt')
                record['input_shapes'] = {key: list(v.shape) for key, v in inputs.items() if hasattr(v, 'shape')}
                record['image_count'] = len(images)
                record['input_text_sha256'] = digest(text)
                inputs = inputs.to(device)
                for key in inputs:
                    if torch.is_floating_point(inputs[key]):
                        inputs[key] = inputs[key].to(dtype)
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_TOKENS, use_cache=True)
                raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                record['raw_output'] = raw
                record['raw_output_sha256'] = digest(raw)
                try:
                    parsed = strict_parse(raw, list(record['mapping']))
                    record.update({'parse_ok': True, 'parsed': parsed, 'selected_id': record['mapping'][parsed['choice']]})
                except (ValueError, TypeError) as exc:
                    record['parse_error'] = str(exc)
                break
            except Exception as exc:
                record['attempts'].append({'attempt': attempt, 'error': repr(exc)})
                print(f'[{NAME}] runtime error attempt {attempt}: {exc!r}', flush=True)
                if device == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
        record['seconds'] = time.monotonic()-tick
        with OUT.open('a') as f:
            f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
        recent = (recent + [record['seconds']])[-50:]
        remaining = total - len(done) - n
        eta = statistics.median(recent)*remaining
        print(f'[{NAME}] {len(done)+n}/{total} {g["prompt_id"][:10]} p{k} ok={record["parse_ok"]} '
              f'{record.get("raw_output", record["attempts"])!r} {record["seconds"]:.1f}s '
              f'eta={eta/3600:.2f}h', flush=True)
        if device == 'mps':
            torch.mps.empty_cache()
        if n % 20 == 0:
            gc.collect()
    print(f'[{NAME}] session finished {time.monotonic()-started:.0f}s', flush=True)
    if len(done)+len(todo) == total and not args.limit:
        print(f'[{NAME}] COMPLETE', flush=True)

if __name__ == '__main__':
    main()
