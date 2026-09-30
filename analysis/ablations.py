"""POST-HOC position-bias ablations (runs/ablations/*.jsonl) -> results/main/ablations.json.

Every number is computed from the run files and human scores; the frozen Qwen main run is recomputed
with the same code as a reference on the same prompts.
"""
from collections import Counter
import numpy as np
from scipy.stats import chisquare
from src.common import ROOT, load_jsonl, save_json
from .metrics import candidate_metrics, bootstrap_mean, choice_consistency

SEED = 20261002
ABL = ROOT/'runs/ablations'

def load_run(path):
    rows = load_jsonl(path) if path.exists() else []
    unique = {(r['prompt_id'], r['permutation']):r for r in rows}
    assert len(unique) == len(rows), f'duplicate keys in {path}'
    return unique

def slot_of(r):
    """0-based presentation position of the selected candidate."""
    label = next(l for l, cid in r['mapping'].items() if cid == r['selected_id'])
    return r['positions'][label] if 'positions' in r else list(r['mapping']).index(label)

def slot_shares(rows):
    rows = [r for r in rows if r['parse_ok']]
    if not rows:
        return None
    obs, exp = Counter(), Counter()
    for r in rows:
        obs[slot_of(r)] += 1
        for i in range(len(r['mapping'])):
            exp[i] += 1/len(r['mapping'])
    slots = sorted(exp)
    test = chisquare([obs[s] for s in slots], [exp[s] for s in slots])
    return {'valid_calls':len(rows),
            'slots':{'ABCD'[s]+'_position':{'count':obs[s], 'share':obs[s]/len(rows), 'uniform':exp[s]/len(rows)} for s in slots},
            'chi2':float(test.statistic), 'df':len(slots)-1, 'p_value':float(test.pvalue)}

def policy(groups, pick):
    """pick(group) -> selected id or None (abstain). Paired against exact random on the same prompt."""
    vals = []
    for g in groups:
        sel = pick(g)
        if sel is None:
            continue
        ids = [c['id'] for c in g['candidates']]; scores = [c['utility'] for c in g['candidates']]
        vals.append(candidate_metrics(scores, ids.index(sel)))
    if not vals:
        return {'n':0, 'coverage':0.0}
    gain = np.array([v['gain'] for v in vals])
    return {'n':len(vals), 'coverage':len(vals)/len(groups),
            'utility':float(np.mean([v['utility'] for v in vals])),
            'regret':float(np.mean([v['regret'] for v in vals])),
            'regret_ci':bootstrap_mean([v['regret'] for v in vals], seed=SEED),
            'gain_vs_random':float(gain.mean()), 'gain_ci':bootstrap_mean(gain, seed=SEED),
            'human_best':float(np.mean([v['human_best'] for v in vals]))}

def order_stats(groups, run, perms):
    """Flip rate / unanimity over the given permutations; requires all of them valid."""
    complete, flips, unanimous = 0, 0, {}
    for g in groups:
        rs = [run.get((g['prompt_id'], k)) for k in perms]
        if not all(r and r['parse_ok'] for r in rs):
            continue
        complete += 1
        ch = {r['selected_id'] for r in rs}
        flips += len(ch) > 1
        if len(ch) == 1:
            unanimous[g['prompt_id']] = ch.pop()
    return {'complete_prompts':complete, 'flip_rate':flips/complete if complete else None,
            'unanimity_coverage':len(unanimous)/len(groups),
            'unanimous':policy(groups, lambda g: unanimous.get(g['prompt_id']))}

def majority_pick(run, perms):
    def pick(g):
        rs = [run.get((g['prompt_id'], k)) for k in perms]
        if not all(r and r['parse_ok'] for r in rs):
            return None
        # Plurality; ties broken by canonical candidate order (frozen selective-policy rule).
        return choice_consistency([r['selected_id'] for r in rs], [c['id'] for c in g['candidates']])['majority_id']
    return pick

def original(run):
    def pick(g):
        r = run.get((g['prompt_id'], 0))
        return r['selected_id'] if r and r['parse_ok'] else None
    return pick

def agreement(run, ref, keys):
    both = [k for k in keys if k in run and k in ref and run[k]['parse_ok'] and ref[k]['parse_ok']]
    return {'n':len(both), 'same_selection':float(np.mean([run[k]['selected_id']==ref[k]['selected_id'] for k in both])) if both else None}

def validity(run):
    rows = list(run.values())
    return {'calls':len(rows), 'valid':sum(r['parse_ok'] for r in rows),
            'invalid':sum(not r['parse_ok'] for r in rows),
            'runtime_failed':sum('raw_output' not in r for r in rows),
            'seconds_median':float(np.median([r['seconds'] for r in rows])) if rows else None}

def three_order_block(name, groups, run, main):
    block = {**validity(run), 'slot_choice':slot_shares(run.values()),
             'orders':order_stats(groups, run, [0,1,2]),
             'original_order':policy(groups, original(run)),
             'agreement_with_main_same_permutation':agreement(run, main, list(run))}
    if name == 'choice_only':
        block['order_majority'] = policy(groups, majority_pick(run, [0,1,2]))
        block['lenient_recoverable_invalid'] = sum('lenient_choice' in r for r in run.values())
    if name == 'opaque_labels':
        # Label-string check: choice share by the label's first character (position-free by design).
        valid = [r for r in run.values() if r['parse_ok']]
        chosen = Counter(next(l for l,c in r['mapping'].items() if c==r['selected_id'])[0] for r in valid)
        shown = Counter(l[0] for r in valid for l in r['mapping'])
        block['choice_rate_by_label_letter'] = {k:{'chosen':chosen[k], 'shown':shown[k], 'rate':chosen[k]/shown[k]} for k in sorted(shown)}
    return block

def flat_three(b):
    first = b['slot_choice']['slots']['A_position']
    u = b['orders']['unanimous']
    flat = {'slot_first_rate':first['share'], 'slot_first_uniform':first['uniform'],
            'flip_rate':b['orders']['flip_rate'], 'gain':b['original_order']['gain_vs_random'],
            'gain_ci':b['original_order']['gain_ci'], 'unanimous_coverage':b['orders']['unanimity_coverage'],
            'unanimous_gain':u.get('gain_vs_random'), 'unanimous_gain_ci':u.get('gain_ci')}
    if 'order_majority' in b:
        flat.update({'majority_gain':b['order_majority']['gain_vs_random'], 'majority_gain_ci':b['order_majority']['gain_ci']})
    return flat

def flat_rot(b):
    sc = b['combined_slot_choice']['slots']; o = b['orders_over_4']
    return {'n_prompts':b['pools_with_all_4_orders'],
            'slot_rates':{k[0]:v['share'] for k,v in sc.items()},
            'slot_first_rate':sc['A_position']['share'], 'slot_first_uniform':sc['A_position']['uniform'],
            'flip_rate':o['flip_rate'], 'unanimous_coverage':o['unanimity_coverage'],
            'unanimous_regret':o['unanimous'].get('regret'), 'unanimous_gain':o['unanimous'].get('gain_vs_random'),
            'unanimous_gain_ci':o['unanimous'].get('gain_ci'),
            'gain':b['original_order_on_4pools']['gain_vs_random'], 'gain_ci':b['original_order_on_4pools']['gain_ci'],
            'plurality_gain':b['order_averaged_plurality_over_4']['gain_vs_random'],
            'plurality_gain_ci':b['order_averaged_plurality_over_4']['gain_ci']}

def main():
    groups = load_jsonl(ROOT/'data/derived/main.jsonl')
    main_run = load_run(ROOT/'runs/qwen_main.jsonl')
    out = {'post_hoc':True, 'bootstrap':{'resamples':10000, 'seed':SEED, 'unit':'prompt'},
           'n_prompts':len(groups),
           'reference_main':{**validity(main_run), 'slot_choice':slot_shares(main_run.values()),
                             'orders':order_stats(groups, main_run, [0,1,2]),
                             'original_order':policy(groups, original(main_run)),
                             'order_majority':policy(groups, majority_pick(main_run, [0,1,2]))}}
    for name in ['choice_only', 'opaque_labels']:
        run = load_run(ABL/f'{name}.jsonl')
        if run:
            out[name] = three_order_block(name, groups, run, main_run)
    rot = load_run(ABL/'rotation4.jsonl')
    if rot:
        four = [g for g in groups if len(g['candidates']) == 4]
        combined = {**{k:v for k,v in main_run.items() if k[1] in (0,1,2)}, **rot}
        combined = {k:v for k,v in combined.items() if k[0] in {g['prompt_id'] for g in four}}
        full = [g for g in four if all((g['prompt_id'],k) in combined for k in range(4))]
        # Balanced design check: every image in every slot exactly once.
        balanced = all(len({tuple(combined[(g['prompt_id'],k)]['mapping'].values())[s] for k in range(4)}) == 4
                       for g in full for s in range(4))
        # Pure position-following: the same slot chosen in all 4 rotations.
        same_slot = Counter()
        for g in full:
            rs = [combined[(g['prompt_id'],k)] for k in range(4)]
            if all(r['parse_ok'] for r in rs):
                slots = {slot_of(r) for r in rs}
                if len(slots) == 1:
                    same_slot['ABCD'[slots.pop()]] += 1
        out['rotation4'] = {**validity(rot), 'four_candidate_pools':len(four), 'pools_with_all_4_orders':len(full),
                            'balanced_latin_rotation':balanced,
                            'permutation3_only_slot_choice':slot_shares(rot.values()),
                            'combined_slot_choice':slot_shares([combined[(g['prompt_id'],k)] for g in full for k in range(4)]),
                            'orders_over_4':order_stats(full, combined, [0,1,2,3]),
                            'same_slot_all_4_orders':dict(same_slot),
                            'original_order_on_4pools':policy(full, original(combined)),
                            'order_averaged_plurality_over_4':policy(full, majority_pick(combined, [0,1,2,3])),
                            'main_majority_over_3_on_4pools':policy(full, majority_pick(combined, [0,1,2]))}
    for key in ['reference_main','choice_only','opaque_labels']:
        if key in out:
            out[key].update(flat_three(out[key]))
    if 'rotation4' in out:
        out['rotation4'].update(flat_rot(out['rotation4']))
    save_json(ROOT/'results/main/ablations.json', out)
    print('wrote results/main/ablations.json')

if __name__ == '__main__':
    main()
