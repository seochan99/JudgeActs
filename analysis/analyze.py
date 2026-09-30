import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT, load_jsonl, save_json
from .metrics import candidate_metrics, random_metrics, bootstrap_mean, wilson, choice_consistency

def is_json(raw):
    try:
        json.loads(raw)
        return True
    except (ValueError, TypeError):
        return False

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
                            'strict_schema_valid':sum(r.get('schema_valid',r['parse_ok']) for r in rows),
                            'recovered':sum(r.get('recovered',False) for r in rows),
                            'json_invalid':sum(not r.get('json_valid',is_json(r.get('raw_output'))) for r in rows),
                            'runtime_failed':sum('raw_output' not in r for r in rows),
                            'parse_failed':sum('raw_output' in r and not r['parse_ok'] for r in rows),
                            'seconds':sum(r['seconds'] for r in rows)}
    records = []
    order_records = []
    for g in groups:
        cs=g['candidates']; ids=[c['id'] for c in cs]; scores=[c['utility'] for c in cs]
        base={'prompt_id':g['prompt_id'],'country':g['country'],'category':g['category'],
              'random_utility':float(np.mean(scores)), 'oracle_utility':float(max(scores)),
              'score_range':float(np.ptp(scores))}
        random_reference=random_metrics(scores)
        for key in ['regret','hsr','hsr05']:
            base['random_'+key]=random_reference[key]
        for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
            vals=[c[axis] for c in cs]
            base['random_'+axis]=float(np.mean(vals)) if all(v is not None for v in vals) else np.nan
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
                ov=[c['overall'] for c in cs]
                if all(v is not None for v in ov):
                    row['overall_regret']=max(ov)-float(np.mean(ov))
                    row['overall_gain']=0.0
            else:
                i=ids.index(selected)
                row.update(candidate_metrics(scores,i))
                row['selected_id']=selected
                for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
                    row[axis]=cs[i][axis]
                ov=[c['overall'] for c in cs]
                if all(v is not None for v in ov):
                    row['overall_regret']=max(ov)-ov[i]
                    row['overall_gain']=ov[i]-float(np.mean(ov))
            records.append(row)
        add('Random')
        # Ties are handled symmetrically for auxiliary outcomes, rather than arbitrarily choosing a source model.
        best_ix=[i for i,s in enumerate(scores) if s >= max(scores)-1e-12]
        add('Oracle',ids[best_ix[0]])
        for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
            vals=[cs[i][axis] for i in best_ix]
            records[-1][axis]=float(np.mean(vals)) if all(v is not None for v in vals) else np.nan
        if all(c['overall'] is not None for c in cs):
            ov=[c['overall'] for c in cs]
            records[-1]['overall_regret']=max(ov)-records[-1]['overall']
            records[-1]['overall_gain']=records[-1]['overall']-float(np.mean(ov))
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
        # Exploratory complements (added after main inference): Qwen's original-order choice on the
        # prompts each agreement gate rejects, so gates can be compared with what they discard.
        qrun=by_run.get('qwen',{})
        qrs=[qrun.get((g['prompt_id'],k)) for k in range(3)]
        if 'qwen' in primary:
            if all(r and r['parse_ok'] for r in qrs) and len({r['selected_id'] for r in qrs})>1:
                add('Qwen, not unanimous',primary['qwen'])
            if primary.get('smol')!=primary['qwen']:
                add('Qwen, judges disagree',primary['qwen'])
    return pd.DataFrame(records),pd.DataFrame(order_records),validation

def position_bias(paths):
    """Presentation-slot choice rates versus the uniform rate implied by each call's pool size."""
    from scipy.stats import chisquare
    out={}
    for path in paths:
        rows=[r for r in load_jsonl(path) if r['parse_ok']]
        name=rows[0]['model']
        observed=defaultdict(float); expected=defaultdict(float)
        for r in rows:
            slot=next(k for k,v in r['mapping'].items() if v==r['selected_id'])
            observed[slot]+=1
            for k in r['mapping']:
                expected[k]+=1/len(r['mapping'])
        slots=sorted(expected)
        test=chisquare([observed[k] for k in slots],[expected[k] for k in slots])
        # Generator share over all orders: rotations decouple generator from slot.
        gens=defaultdict(float)
        for r in rows:
            gens[r['selected_id'].rsplit('_',1)[1]]+=1/len(rows)
        out.setdefault('_generator_uniform',{})
        out[name+'_generators']=dict(gens)
        out[name]={'calls':len(rows),
                   'slots':{k:{'count':int(observed[k]),'rate':observed[k]/len(rows),
                               'uniform_rate':expected[k]/len(rows)} for k in slots},
                   'chi2':float(test.statistic),'df':len(slots)-1,'p_value':float(test.pvalue)}
    return out

def summarize(df, total):
    rows=[]
    for policy, sub in df.groupby('policy',sort=False):
        row={'policy':policy,'n':len(sub),'coverage':len(sub)/total}
        for key in ['utility','regret','gain','human_best','near_best','bottom_half','stereotype','missing_explicit','missing_implicit','image_quality','overall','overall_regret','overall_gain','random_utility','oracle_utility','random_regret','random_hsr','random_hsr05','score_range']:
            vals=sub[key].dropna().to_numpy()
            row[key]=float(np.mean(vals)) if len(vals) else None
            row[key+'_ci']=bootstrap_mean(vals)
            row[key+'_n']=len(vals)
        for axis in ['stereotype','missing_explicit','missing_implicit','image_quality','overall']:
            paired=sub[[axis,'random_'+axis]].dropna()
            row[axis+'_comparison_n']=len(paired)
            row['matched_'+axis]=float(paired[axis].mean()) if len(paired) else None
            row['matched_random_'+axis]=float(paired['random_'+axis].mean()) if len(paired) else None
            difference=(paired[axis]-paired['random_'+axis]).to_numpy()
            row[axis+'_delta']=float(difference.mean()) if len(difference) else None
            row[axis+'_delta_ci']=bootstrap_mean(difference)
        # Bonferroni over the three cultural-error outcomes (exploratory adjustment).
        for axis in ['stereotype','missing_explicit','missing_implicit']:
            paired=sub[[axis,'random_'+axis]].dropna()
            row[axis+'_delta_ci_bonf3']=bootstrap_mean((paired[axis]-paired['random_'+axis]).to_numpy(),alpha=.05/3)
        # Paired below-mean difference against the exact random event probability on the same prompts.
        hsr_diff=(sub['hsr']-sub['random_hsr']).to_numpy()
        row['hsr_delta']=float(hsr_diff.mean())
        row['hsr_delta_ci']=bootstrap_mean(hsr_diff)
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
    categories=[]
    for (category,policy),sub in df.groupby(['category','policy']):
        categories.append({'category':category,**summarize(sub,len([g for g in groups if g['category']==category]))[0]})
    save_json(dest/'category.json',categories)
    # Include explicit zero-acceptance cells without assigning them zero risk.
    coverage=[]
    for axis in ['country','category']:
        for group in sorted({g[axis] for g in groups}):
            total=sum(g[axis]==group for g in groups)
            for policy in df['policy'].unique():
                accepted=int(((df[axis]==group)&(df['policy']==policy)).sum())
                coverage.append({'axis':axis,'group':group,'policy':policy,
                                 'accepted_n':accepted,'manifest_n':total,'coverage':accepted/total})
    save_json(dest/'coverage_by_group.json',coverage)
    decomp=[]
    for name,axis in [('Explicit','missing_explicit_sensitive'),('Implicit','missing_implicit_sensitive'),('Stereotype','stereotype_sensitive')]:
        for _,row in df[df[axis]].groupby('policy'):
            if len(row):
                decomp.append({'condition':name,**summarize(row,len(groups))[0]})
    save_json(dest/'decomposition.json',decomp)
    position=position_bias(paths)
    uniform=defaultdict(float)
    for g in groups:
        for c in g['candidates']:
            uniform[c['id'].rsplit('_',1)[1]]+=1/len(g['candidates'])/len(groups)
    position['_generator_uniform']=dict(uniform)
    save_json(dest/'position.json',position)
    expected_keys={(g['prompt_id'],k) for g in groups for k in range(3)}
    complete={}
    for name in validation:
        rows=load_jsonl(ROOT/f'runs/{name}_{args.split}.jsonl')
        observed={(r['prompt_id'],r['permutation']) for r in rows}
        validation[name]['unattempted']=len(expected_keys-observed)
        complete[name]=observed==expected_keys and len(rows)==len(expected_keys)
    status={'split':args.split,'n_manifest':len(groups),'validation':validation,
            'complete_primary':complete.get('qwen',False),
            'complete_secondary':complete.get('smol',False)}
    save_json(dest/'status.json',status)
    print(json.dumps({'status':status,'summary':summary},indent=2))

if __name__=='__main__':main()
