"""Check completed scientific artifacts and fail if their denominators drift."""
import argparse
import hashlib
import json
from pathlib import Path
from .common import ROOT, load_jsonl, orders, save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=['dev','main']);args=p.parse_args()
    groups=load_jsonl(ROOT/f'data/derived/{args.split}.jsonl')
    manifest=json.loads((ROOT/f'data/manifests/{args.split}.json').read_text())
    assert manifest['ids']==[g['prompt_id'] for g in groups]
    dev=load_jsonl(ROOT/'data/derived/dev.jsonl');main=load_jsonl(ROOT/'data/derived/main.jsonl')
    assert not set(g['prompt_id'] for g in dev)&set(g['prompt_id'] for g in main)
    assert not set(g['prompt'] for g in dev)&set(g['prompt'] for g in main)
    data={g['prompt_id']:g for g in groups}
    report={'split':args.split,'prompt_sets':len(groups),'checks':[]}
    for model in ['qwen','smol']:
        path=ROOT/f'runs/{model}_{args.split}.jsonl'
        if not path.exists():continue
        rows=load_jsonl(path)
        keys={(r['prompt_id'],r['permutation']) for r in rows}
        assert len(keys)==len(rows)
        for r in rows:
            g=data[r['prompt_id']]
            assert list(r['mapping'].values())==orders(g)[r['permutation']]
            assert r['image_count']==len(g['candidates'])
            if 'image_grid_thw' in r['input_shapes']:
                assert r['input_shapes']['image_grid_thw'][0]==len(g['candidates'])
            if r['parse_ok']:
                assert r['selected_id']==r['mapping'][r['parsed']['choice']]
                c=next(c for c in g['candidates'] if c['id']==r['selected_id'])
                assert 0<=c['utility']<=1
        report['checks'].append({'model':model,'calls':len(rows),'valid':sum(r['parse_ok'] for r in rows),
                                 'complete_3_orders':len(keys)==len(groups)*3})
    save_json(ROOT/f'results/{args.split}/validation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
