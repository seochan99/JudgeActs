"""POST-HOC scale extension (MLX, 8-bit LM weights): bf16 4B (frozen) vs MLX 4B (quantization/framework control)
vs MLX 8B, all under the frozen main protocol. Writes results/main/extension.json.

All estimands reuse analysis/analyze.py (build_rows, summarize, position_bias) and analysis/metrics.py
(10k prompt bootstrap, seed 20261002). Random is the exact uniform-choice expectation on the same prompts.
"""
import json
import statistics
import numpy as np
import pandas as pd
from src.common import ROOT, load_jsonl, save_json
from .analyze import build_rows, summarize, position_bias
from .metrics import bootstrap_mean

JUDGES = {'bf16_4b': ROOT/'runs/qwen_main.jsonl',
          'mlx_4b': ROOT/'runs/extension/qwen4b_mlx_main.jsonl',
          'mlx_8b': ROOT/'runs/extension/qwen8b_mlx_main.jsonl'}
KEYS = ['n', 'coverage', 'utility', 'gain', 'gain_ci', 'regret', 'regret_ci', 'random_regret',
        'hsr', 'hsr_ci', 'random_hsr', 'hsr_delta', 'hsr_delta_ci', 'human_best', 'human_best_ci',
        'stereotype_delta', 'stereotype_delta_ci', 'missing_explicit_delta', 'missing_explicit_delta_ci',
        'missing_implicit_delta', 'missing_implicit_delta_ci', 'stereotype_comparison_n',
        'missing_explicit_comparison_n', 'missing_implicit_comparison_n']

def pick(row):
    return {k: row.get(k) for k in KEYS}

def judge_block(name, path, groups, total):
    df, order, val = build_rows(groups, [path])
    label = 'Qwen' if name == 'bf16_4b' else 'Smol'  # build_rows labels non-'qwen' runs 'Smol'
    df = df[df.policy != 'Oracle'].copy()
    df['policy'] = df['policy'].replace({label: 'original', f'{label} majority': 'majority',
                                         f'{label} unanimous': 'unanimous', f'{label} agree 2/3': 'agree_2of3'})
    df = df[df.policy.isin(['Random', 'original', 'majority', 'unanimous'])]
    # Prompts the unanimity gate rejects (complete orders, not unanimous): the original-order choice there.
    rejected_ids = set(order.loc[order.agreement < 1, 'prompt_id'])
    rej = df[(df.policy == 'original') & df.prompt_id.isin(rejected_ids)].assign(policy='rejected')
    df = pd.concat([df, rej], ignore_index=True)
    summ = {r['policy']: pick(r) for r in summarize(df, total)}
    pos = position_bias([path])
    p = pos[next(k for k in pos if not k.startswith('_') and not k.endswith('_generators'))]
    fl = order['flip'].to_numpy()
    rows = load_jsonl(path)
    v = next(iter(val.values()))
    expected = {(g['prompt_id'], k) for g in groups for k in range(3)}
    return df, order, {
        'calls': v['calls'], 'valid_calls': v['valid'], 'invalid_calls': v['invalid'],
        'complete': {(r['prompt_id'], r['permutation']) for r in rows} == expected,
        'first_slot_share': p['slots']['A']['rate'], 'first_slot_uniform': p['slots']['A']['uniform_rate'],
        'slot_rates': {k: s['rate'] for k, s in p['slots'].items()},
        'slot_uniform': {k: s['uniform_rate'] for k, s in p['slots'].items()},
        'position_chi2': p['chi2'], 'position_df': p['df'], 'position_p': p['p_value'],
        'complete_order_prompts': int(len(order)),
        'flip_rate': float(fl.mean()), 'flip_ci': bootstrap_mean(fl),
        'unanimous_prompts': int((order.agreement == 1).sum()),
        'unanimity_coverage': float((order.agreement == 1).sum()/total),
        'original_order': summ['original'], 'majority_vote': summ['majority'],
        'unanimous': summ.get('unanimous'), 'rejected_by_unanimity': summ.get('rejected'),
        'median_seconds_per_call': statistics.median(r['seconds'] for r in rows),
    }

def main():
    groups = load_jsonl(ROOT/'data/derived/main.jsonl')
    total = len(groups)
    out = {'post_hoc': True, 'note': 'Scale extension on Apple MLX with 8-bit language-model weights; the MLX 4B run is '
           'a quantization/framework control for the frozen bf16 transformers-MPS 4B run.', 'judges': {}, 'meta': {}}
    dfs, orders_ = {}, {}
    for name, path in JUDGES.items():
        dfs[name], orders_[name], out['judges'][name] = judge_block(name, path, groups, total)
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        out['meta'][name] = {k: meta.get(k) for k in ('model', 'quant', 'backend', 'dtype', 'versions', 'torch')}

    # Per-call choice agreement with the frozen bf16 4B judge.
    ref = {(r['prompt_id'], r['permutation']): r for r in load_jsonl(JUDGES['bf16_4b'])}
    chance = float(np.mean([1/len(g['candidates']) for g in groups]))
    for name in ('mlx_4b', 'mlx_8b'):
        recs = {(r['prompt_id'], r['permutation']): r for r in load_jsonl(JUDGES[name])}
        block = {}
        for label, ks in [('all', (0, 1, 2)), ('original_order', (0,))]:
            same = np.array([recs[key]['selected_id'] == ref[key]['selected_id'] for key in recs
                             if key[1] in ks and key in ref and recs[key]['parse_ok'] and ref[key]['parse_ok']], float)
            block[label] = {'n': len(same), 'rate': float(same.mean()) if len(same) else None,
                            'ci': bootstrap_mean(same)}
        # Same answer in the same slot is the position-only explanation of agreement; report it for context.
        block['input_text_and_shapes_match_frozen'] = sum(bool(r.get('input_matches_frozen')) for r in recs.values())
        block['chance_rate'] = chance
        out['judges'][name]['agreement_with_bf16_4b'] = block

    # Cross-model gate on original-order choices: MLX 8B and frozen bf16 4B agree -> keep.
    p8 = dfs['mlx_8b'][dfs['mlx_8b'].policy == 'original'].set_index('prompt_id')
    p4 = dfs['bf16_4b'][dfs['bf16_4b'].policy == 'original'].set_index('prompt_id')
    both = p8.index.intersection(p4.index)
    agree = [pid for pid in both if p8.at[pid, 'selected_id'] == p4.at[pid, 'selected_id']]
    disagree = [pid for pid in both if pid not in set(agree)]
    gate = pd.concat([p8.loc[agree].reset_index().assign(policy='kept'),
                      p8.loc[disagree].reset_index().assign(policy='rejected_8b_choice'),
                      p4.loc[disagree].reset_index().assign(policy='rejected_4b_choice')], ignore_index=True)
    out['cross_model_gate_mlx8b_bf16_4b'] = {'kept_n': len(agree), 'rejected_n': len(disagree),
                                             'kept_coverage': len(agree)/total,
                                             **{r['policy']: pick(r) for r in summarize(gate, total)}}
    save_json(ROOT/'results/main/extension.json', out)

    f = lambda x: 'NA' if x is None else f'{x:+.4f}'
    ci = lambda c: '[NA]' if c[0] is None else f'[{c[0]:+.4f}, {c[1]:+.4f}]'
    for name, j in out['judges'].items():
        o = j['original_order']
        print(f"\n== {name}: valid {j['valid_calls']}/{j['calls']} complete={j['complete']} "
              f"median {j['median_seconds_per_call']:.1f}s/call")
        print(f"  first slot {j['first_slot_share']:.3f} (uniform {j['first_slot_uniform']:.3f}) chi2={j['position_chi2']:.1f} "
              f"df={j['position_df']} p={j['position_p']:.2g}; flip {j['flip_rate']:.3f} {ci(j['flip_ci'])}")
        print(f"  original gain {f(o['gain'])} {ci(o['gain_ci'])} regret {o['regret']:.4f} (rand {o['random_regret']:.4f}) "
              f"BMR {o['hsr']:.3f} (rand {o['random_hsr']:.3f})")
        for k in ('majority_vote', 'unanimous', 'rejected_by_unanimity'):
            r = j[k]
            if r:
                print(f"  {k:22s} n={r['n']:3d} gain {f(r['gain'])} {ci(r['gain_ci'])}")
        for ax in ('stereotype', 'missing_explicit', 'missing_implicit'):
            print(f"  {ax}_delta {f(o[ax+'_delta'])} {ci(o[ax+'_delta_ci'])} (n={o[ax+'_comparison_n']})")
        if 'agreement_with_bf16_4b' in j:
            a = j['agreement_with_bf16_4b']
            print(f"  agreement w/ bf16 4B: all {a['all']['rate']:.3f} {ci(a['all']['ci'])} n={a['all']['n']}; "
                  f"orig {a['original_order']['rate']:.3f}; chance {a['chance_rate']:.3f}; inputs match {a['input_text_and_shapes_match_frozen']}")
    g = out['cross_model_gate_mlx8b_bf16_4b']
    print(f"\n== gate mlx8B∧bf16 4B: kept {g['kept_n']} gain {f(g['kept']['gain'])} {ci(g['kept']['gain_ci'])}; "
          f"rejected {g['rejected_n']}: 8B {f(g['rejected_8b_choice']['gain'])} {ci(g['rejected_8b_choice']['gain_ci'])}, "
          f"4B {f(g['rejected_4b_choice']['gain'])} {ci(g['rejected_4b_choice']['gain_ci'])}")

if __name__ == '__main__':
    main()
