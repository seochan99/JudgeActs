import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT, load_jsonl, save_json
from .metrics import candidate_metrics, random_metrics, bootstrap_mean, wilson, choice_consistency

def build_rows(groups, paths):
    by_run = {}
    validation = {}
    for path in paths:
        rows = load_jsonl(path)
        unique = {(r['prompt_id'],r['permutation']):r for r in rows}
        assert len(unique)==len(rows), f'duplicate inference keys: {path}'
        name = rows[0]['model'] if rows else path.name.split('_')[0]
        by_run[name] = unique
        validation[name] = {'calls':len(rows),'valid':sum(r['parse_ok'] for r in rows),
                            'invalid':sum(not r['parse_ok'] for r in rows),
                            'seconds':sum(r['seconds'] for r in rows)}
    records = []
    order_records = []
    for g in groups:
        cs=g['candidates']; ids=[c['id'] for c in cs]; scores=[c['utility'] for c in cs]
        base={'prompt_id':g['prompt_id'],'country':g['country'],'category':g['category'],
              'random_utility':float(np.mean(scores)), 'oracle_utility':float(max(scores)),
              'score_range':float(np.ptp(scores))}
        for axis in ['missing_explicit','missing_implicit','stereotype']:
            vals=[c[axis] for c in cs]
            base[axis+'_sensitive']=all(v is not None for v in vals) and np.ptp(vals)>1e-12
        def add(policy, selected=None):
            row={**base,'policy':policy}
            if selected is None:
                row.update(random_metrics(scores))
                for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
                    vals=[c[axis] for c in cs]
                    row[axis]=float(np.mean(vals)) if all(v is not None for v in vals) else np.nan
            else:
                i=ids.index(selected)
                row.update(candidate_metrics(scores,i))
                row['selected_id']=selected
                for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
                    row[axis]=cs[i][axis]
            records.append(row)
        add('Random')
        # Ties are handled symmetrically for auxiliary outcomes, rather than arbitrarily choosing a source model.
        best_ix=[i for i,s in enumerate(scores) if s >= max(scores)-1e-12]
        add('Oracle',ids[best_ix[0]])
        for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
            vals=[cs[i][axis] for i in best_ix]
            records[-1][axis]=float(np.mean(vals)) if all(v is not None for v in vals) else np.nan
        primary = {}
        for name, run in by_run.items():
            rs=[run.get((g['prompt_id'], k)) for k in range(3)]
            label='Qwen' if name=='qwen' else 'Smol'
            if rs[0] and rs[0]['parse_ok']:
                primary[name]=rs[0]['selected_id']; add(label,rs[0]['selected_id'])
            if all(r and r['parse_ok'] for r in rs):
                choices=[r['selected_id'] for r in rs]
                cons=choice_consistency(choices,ids)
                order_records.append({**base,'model':name,**cons})
                add(label+' majority',cons['majority_id'])
                if cons['agreement']>=2/3-1e-12:
                    add(label+' agree 2/3',cons['majority_id'])
                if cons['agreement']==1:
                    add(label+' unanimous',cons['majority_id'])
        if len(primary)==2 and primary['qwen']==primary['smol']:
            add('Cross-model',primary['qwen'])
    return pd.DataFrame(records),pd.DataFrame(order_records),validation

def summarize(df, total):
    rows=[]
    for policy, sub in df.groupby('policy',sort=False):
        row={'policy':policy,'n':len(sub),'coverage':len(sub)/total}
        for key in ['utility','regret','gain','human_best','near_best','bottom_half','stereotype','missing_explicit','missing_implicit','image_quality','overall']:
            vals=sub[key].dropna().to_numpy()
            row[key]=float(np.mean(vals)) if len(vals) else None
            row[key+'_ci']=bootstrap_mean(vals)
        for key in ['hsr','hsr05']:
            row[key]=float(sub[key].mean())
            # Wilson only for deterministic selected policies (Bernoulli prompt outcomes).
            row[key+'_ci']=bootstrap_mean(sub[key]) if policy=='Random' else wilson(float(sub[key].sum()),len(sub))
        row['regret_median']=float(sub['regret'].median())
        group_means=sub.groupby('country')['regret'].mean()
        row['worst_group_regret']=float(group_means.max())
        row['disparity']=float(group_means.max()-group_means.min())
        # Recompute the maximum in every stratified bootstrap replicate.
        rng=np.random.default_rng(20261002)
        sample_means=[]
        for country, country_sub in sub.groupby('country'):
            values=country_sub['regret'].to_numpy()
            sample_means.append(values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1))
        maxima=np.max(np.stack(sample_means),axis=0)
        row['worst_group_regret_ci']=np.quantile(maxima,[.025,.975]).tolist()
        row['countries_covered']=len(group_means)
        row['missing_countries']=sorted(set(df['country'])-set(group_means.index))
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='main',choices=['main','dev']); args=p.parse_args()
    groups=load_jsonl(ROOT/f'data/derived/{args.split}.jsonl')
    paths=[ROOT/f'runs/{name}_{args.split}.jsonl' for name in ['qwen','smol']]
    paths=[p for p in paths if p.exists()]
    df,order,validation=build_rows(groups,paths)
    dest=ROOT/'results'/args.split;dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'prompt_metrics.csv',index=False)
    order.to_csv(dest/'order_metrics.csv',index=False)
    summary=summarize(df,len(groups))
    save_json(dest/'summary.json',summary)
    subgroup=[]
    for (country,policy),sub in df.groupby(['country','policy']):
        subgroup.append({'country':country,**summarize(sub,len([g for g in groups if g['country']==country]))[0]})
    save_json(dest/'country.json',subgroup)
    decomp=[]
    for name,axis in [('Explicit','missing_explicit_sensitive'),('Implicit','missing_implicit_sensitive'),('Stereotype','stereotype_sensitive')]:
        for _,row in df[df[axis]].groupby('policy'):
            if len(row):
                decomp.append({'condition':name,**summarize(row,len(groups))[0]})
    save_json(dest/'decomposition.json',decomp)
    expected={name:len(groups)*3 for name in validation}
    status={'split':args.split,'n_manifest':len(groups),'validation':validation,
            'complete_primary':validation.get('qwen',{}).get('calls')==len(groups)*3,
            'complete_secondary':validation.get('smol',{}).get('calls')==len(groups)*3}
    save_json(dest/'status.json',status)
    print(json.dumps({'status':status,'summary':summary},indent=2))

if __name__=='__main__':main()
