"""POST-HOC scale extension on Apple MLX (8-bit LM weights): Qwen3-VL-4B (quantization/framework control) and 8B.

Frozen inputs (runs/*.jsonl, data/, configs/, prompts/, src/run_judge.py) are only read. The message builder
(messages_for), instruction, 448-px aspect-preserving bounding box, strict parser and 64 new tokens are imported from
the frozen code; the label mapping of each call is copied from runs/qwen_main.jsonl for the same (prompt_id,
permutation) and checked against common.orders. Inputs are tokenised with the transformers Qwen3-VL processor exactly
as in the frozen runner (rendered chat text + images -> input_ids, pixel_values, image_grid_thw); only the forward pass
and greedy decoding run in mlx-vlm. Each record stores whether its rendered text hash and tensor shapes equal the frozen
4B record for the same key (the frozen
processor additionally returned mm_token_type_ids, which only marks image-token positions and is not compared).
Output: runs/extension/{name}_main.jsonl (+ .meta.json), resumable by (prompt_id, permutation).
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
from .common import ROOT, SEED, digest, load_jsonl, orders, strict_parse, save_json
from .run_judge import messages_for

MODELS = {
    'qwen4b_mlx': {'id': 'mlx-community/Qwen3-VL-4B-Instruct-8bit', 'revision': '0943db6e15185b86be368d3cf0704aec740b142b',
                   'base': 'Qwen/Qwen3-VL-4B-Instruct', 'dir': 'qwen4b_mlx8', 'license': 'apache-2.0'},
    'qwen8b_mlx': {'id': 'mlx-community/Qwen3-VL-8B-Instruct-8bit', 'revision': 'a0093b9b5fda6f76ddd4a462c6830ae7c4fe47ec',
                   'base': 'Qwen/Qwen3-VL-8B-Instruct', 'dir': 'qwen8b_mlx8', 'license': 'apache-2.0'},
}
FROZEN = ROOT/'runs/qwen_main.jsonl'
IMAGE_SIZE, MAX_TOKENS, PERMUTATIONS = 448, 64, 3
EOS = [151645, 151643]  # generation_config.json eos_token_id (same as the frozen HF generate stop set)

def versions():
    import mlx.core as mx, mlx_vlm, transformers, tokenizers, PIL, numpy
    from importlib.metadata import version
    return {'python': sys.version.split()[0], 'mlx': version('mlx'), 'mlx_metal': version('mlx-metal'),
            'mlx_vlm': version('mlx-vlm'), 'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
            'pillow': PIL.__version__, 'numpy': numpy.__version__, 'platform': platform.platform()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', choices=list(MODELS), required=True)
    p.add_argument('--limit', type=int, help='number of calls to run in this invocation (smoke test)')
    args = p.parse_args()
    spec = MODELS[args.name]
    model_dir = ROOT/'models'/spec['dir']
    out = ROOT/'runs/extension'/f'{args.name}_main.jsonl'
    cfg = json.loads((model_dir/'config.json').read_text())
    quant = cfg.get('quantization') or cfg.get('quantization_config')
    groups = load_jsonl(ROOT/'data/derived/main.jsonl')
    instruction = (ROOT/'prompts/judge_primary.txt').read_text()
    frozen = {(r['prompt_id'], r['permutation']): r for r in load_jsonl(FROZEN)}

    metadata = {'post_hoc': True, 'extension': 'scale-mlx', 'name': args.name,
                'model': {k: spec[k] for k in ('id', 'revision', 'base', 'license')},
                'quant': {'bits': quant['bits'], 'group_size': quant['group_size'], 'mode': quant.get('mode'),
                          'scope': 'language model linear layers; vision tower bf16 (as shipped)'},
                'split': 'main', 'backend': 'mlx',
                'runner_sha256': digest(Path(__file__).read_text()),
                'input_builder_sha256': digest((ROOT/'src/run_judge.py').read_text()),
                'parser_sha256': digest((ROOT/'src/common.py').read_text()),
                'prompt_sha256': digest(instruction),
                'split_sha256': digest((ROOT/'data/manifests/main.json').read_text()),
                'chat_template_sha256': digest((model_dir/'chat_template.json').read_text()),
                'mapping_source': 'runs/qwen_main.jsonl',
                'image_size': IMAGE_SIZE, 'max_tokens': MAX_TOKENS, 'permutations': list(range(PERMUTATIONS)),
                'temperature': 0.0, 'do_sample': False, 'eos_token_ids': EOS, 'seed': SEED,
                'output_schema': 'choice_and_ranking', 'versions': versions()}
    meta_path = out.with_suffix('.meta.json')
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        skip = {'runner_sha256', 'started_utc'}
        assert {k: v for k, v in old.items() if k not in skip} == {k: v for k, v in metadata.items() if k not in skip}, \
            'incompatible resume configuration'
        metadata['started_utc'] = old.get('started_utc')
    else:
        metadata['started_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(meta_path, metadata)

    done = {(r['prompt_id'], r['permutation']) for r in load_jsonl(out)} if out.exists() else set()
    todo = []
    for g in groups:
        for k, order in enumerate(orders(g)[:PERMUTATIONS]):
            if (g['prompt_id'], k) in done:
                continue
            mapping = frozen[(g['prompt_id'], k)]['mapping']
            assert mapping == dict(zip('ABCD', order)), 'mapping differs from frozen run'
            todo.append((g, k, [mapping[l] for l in sorted(mapping)]))
    total = len(groups)*PERMUTATIONS
    if args.limit:
        todo = todo[:args.limit]
    print(f'[{args.name}] done={len(done)}/{total} todo={len(todo)}', flush=True)
    if not todo:
        print(f'[{args.name}] COMPLETE', flush=True)
        return

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate import stream_generate
    from transformers import AutoProcessor
    mx.random.seed(SEED)
    mx.set_cache_limit(512*1024**2)
    t0 = time.monotonic()
    model, mlx_processor = load(str(model_dir))
    mlx_processor.tokenizer.stopping_criteria.reset(EOS)
    hf_processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    print(f'[{args.name}] loaded in {time.monotonic()-t0:.0f}s active={mx.get_active_memory()/1e9:.2f}GB', flush=True)

    recent = []
    started = time.monotonic()
    for n, (g, k, order) in enumerate(todo, 1):
        ref = frozen[(g['prompt_id'], k)]
        record = {'prompt_id': g['prompt_id'], 'model': args.name, 'backend': 'mlx', 'quant': f"{quant['bits']}bit",
                  'permutation': k, 'mapping': dict(zip('ABCD', order)), 'parse_ok': False,
                  'utc': datetime.now(timezone.utc).isoformat(), 'attempts': []}
        messages, images = messages_for(g, order, instruction, IMAGE_SIZE)
        record['image_count'] = len(images)
        tick = time.monotonic()
        for attempt in range(2):
            try:
                text = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = hf_processor(text=[text], images=images, return_tensors='np')
                record['input_shapes'] = {key: list(v.shape) for key, v in inputs.items() if hasattr(v, 'shape')}
                record['input_text_sha256'] = digest(text)
                record['input_matches_frozen'] = (record['input_text_sha256'] == ref.get('input_text_sha256')
                                                  and record['input_shapes'] == {key: v for key, v in ref.get('input_shapes', {}).items()
                                                                                 if key != 'mm_token_type_ids'})
                tokens = []
                for resp in stream_generate(model, mlx_processor, '', input_ids=mx.array(inputs['input_ids']),
                                            pixel_values=mx.array(inputs['pixel_values']),
                                            mask=mx.array(inputs['attention_mask']),
                                            image_grid_thw=mx.array(inputs['image_grid_thw']),
                                            max_tokens=MAX_TOKENS, temperature=0.0):
                    if resp.token is not None:
                        tokens.append(int(resp.token))
                raw = hf_processor.batch_decode([tokens], skip_special_tokens=True)[0]
                record['generated_tokens'] = len(tokens)
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
                print(f'[{args.name}] runtime error attempt {attempt}: {exc!r}', flush=True)
                mx.clear_cache(); gc.collect()
        record['seconds'] = time.monotonic()-tick
        record['agrees_with_frozen_4b'] = (record.get('selected_id') == ref.get('selected_id')
                                           if record['parse_ok'] and ref['parse_ok'] else None)
        with out.open('a') as f:
            f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
        recent = (recent + [record['seconds']])[-50:]
        remaining = total - len(done) - n
        eta = statistics.median(recent)*remaining
        print(f'[{args.name}] {len(done)+n}/{total} {g["prompt_id"][:10]} p{k} ok={record["parse_ok"]} '
              f'match_in={record.get("input_matches_frozen")} agree4b={record["agrees_with_frozen_4b"]} '
              f'{record.get("raw_output", record["attempts"])!r} {record["seconds"]:.1f}s '
              f'peak={mx.get_peak_memory()/1e9:.1f}GB eta={eta/3600:.2f}h', flush=True)
        mx.clear_cache()
        if n % 20 == 0:
            gc.collect()
    print(f'[{args.name}] session finished {time.monotonic()-started:.0f}s', flush=True)
    if len(done)+len(todo) == total and not args.limit:
        print(f'[{args.name}] COMPLETE', flush=True)

if __name__ == '__main__':
    main()
