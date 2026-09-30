"""Secondary judge adapter: bounded native visual input and explicit schema normalization.

Raw JSON/schema validity is preserved independently of recovered selection validity.
No choice is inferred from scores, incomplete rankings, or explanatory prose.
"""
import argparse
import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from .common import ROOT, digest, load_jsonl, orders, save_json
from .run_judge import messages_for

def parse_selection(raw, labels):
    bare=re.fullmatch(r'(?:Candidate\s+)?([A-D])',raw.strip())
    if bare and bare[1] in labels:
        return {'choice':bare[1]}, {'json_valid':False,'schema_valid':False,'recovered':True}
    json_ok=True
    try:
        data=json.loads(raw.strip())
    except ValueError:
        json_ok=False
        # Safely parse a Python-style literal object, never arbitrary code or prose.
        try:data=ast.literal_eval(raw.strip())
        except (ValueError,SyntaxError):raise ValueError('no structured selection object')
    if not isinstance(data,dict) or len(data)!=1:
        raise ValueError('expected a single selection field')
    key=next(iter(data));value=data[key]
    if key not in ['choice','image','selected','selected_image','candidate'] or not isinstance(value,str):
        raise ValueError('unsupported selection field')
    match=re.fullmatch(r'(?:Candidate\s+)?([A-D])',value.strip())
    if not match or match[1] not in labels:raise ValueError('not exactly one supplied label')
    strict=json_ok and key=='choice' and value==match[1]
    return {'choice':match[1]}, {'json_valid':json_ok,'schema_valid':strict,'recovered':not strict}

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','main'],default='dev')
    p.add_argument('--permutations',type=int,default=3);p.add_argument('--limit',type=int);args=p.parse_args()
    path=ROOT/f'runs/smol_{args.split}.jsonl';path.parent.mkdir(exist_ok=True)
    done={(r['prompt_id'],r['permutation']) for r in load_jsonl(path)} if path.exists() else set()
    groups=load_jsonl(ROOT/f'data/derived/{args.split}.jsonl')
    if args.limit:groups=groups[:args.limit]
    instruction=(ROOT/'prompts/judge_primary.txt').read_text().split('The candidate labels')[0]
    instruction+='The candidate labels are supplied next to each image. Return compact JSON only, with exactly one key choice containing your selected label. Do not explain your decision.\n'
    sources=json.loads((ROOT/'provenance/sources.json').read_text())
    metadata={'model':sources['models']['smol'],'split':args.split,'runner_sha256':digest(Path(__file__).read_text()),
              'input_builder_sha256':digest((ROOT/'src/run_judge.py').read_text()), 'prompt_sha256':digest(instruction),
              'split_sha256':digest((ROOT/f'data/manifests/{args.split}.json').read_text()),
              'backend':'transformers-mps','dtype':'bfloat16','image_size':384,'do_image_splitting':False,
              'max_tokens':64,'do_sample':False,'torch':torch.__version__,
              'parser':'one explicit bare label or one-field structured object; preserve schema/json validity'}
    meta=path.with_suffix('.meta.json')
    if meta.exists():assert json.loads(meta.read_text())==metadata,'incompatible resume'
    save_json(meta,metadata)
    torch.manual_seed(20261002);torch.set_num_threads(8)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    processor=AutoProcessor.from_pretrained(ROOT/'models/smol',local_files_only=True)
    processor.image_processor.do_image_splitting=False
    processor.image_processor.size={'longest_edge':384}
    model=AutoModelForImageTextToText.from_pretrained(ROOT/'models/smol',local_files_only=True,
                                                   dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    model.generation_config.pad_token_id=processor.tokenizer.pad_token_id
    for g in groups:
        for k,order in enumerate(orders(g)[:args.permutations]):
            if (g['prompt_id'],k) in done:continue
            record={'prompt_id':g['prompt_id'],'model':'smol','permutation':k,'mapping':dict(zip('ABCD',order)),
                    'parse_ok':False,'utc':datetime.now(timezone.utc).isoformat(),'attempts':[],'image_count':len(order)}
            messages,images=messages_for(g,order,instruction,384,choice_only=True)
            tick=time.monotonic()
            for attempt in range(2):
                try:
                    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    inputs=processor(text=[text],images=images,return_tensors='pt')
                    record['input_shapes']={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
                    record['input_text_sha256']=digest(text)
                    inputs=inputs.to(device)
                    for key in inputs:
                        if torch.is_floating_point(inputs[key]):inputs[key]=inputs[key].to(torch.bfloat16)
                    with torch.inference_mode():out=model.generate(**inputs,do_sample=False,max_new_tokens=64,use_cache=True)
                    raw=processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
                    record['raw_output']=raw;record['raw_output_sha256']=digest(raw)
                    try:
                        parsed,flags=parse_selection(raw,list(record['mapping']))
                        record.update({'parse_ok':True,'parsed':parsed,'selected_id':record['mapping'][parsed['choice']],**flags})
                    except ValueError as exc:
                        record['parse_error']=str(exc)
                        try:json.loads(raw.strip());record['json_valid']=True
                        except ValueError:record['json_valid']=False
                        record['schema_valid']=False;record['recovered']=False
                    break
                except Exception as exc:
                    record['attempts'].append({'attempt':attempt,'error':repr(exc)})
                    if device=='mps':torch.mps.empty_cache()
            record['seconds']=time.monotonic()-tick
            with path.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            print(args.split,g['prompt_id'][:10],k,record['parse_ok'],repr(record.get('raw_output')),f'{record["seconds"]:.1f}s',flush=True)
            if device=='mps':torch.mps.empty_cache()

if __name__=='__main__':main()
