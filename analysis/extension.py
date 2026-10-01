"""POST-HOC scale extension: Qwen3-VL-8B judge under the frozen main protocol, compared with the 4B judge.

Reads runs/extension/qwen8b_main.jsonl and the frozen runs/qwen_main.jsonl; writes results/main/extension.json.
All estimands reuse analysis/analyze.py (build_rows, summarize, position_bias) and analysis/metrics.py.
"""
import json
import statistics
import numpy as np
import pandas as pd
from src.common import ROOT, load_jsonl, save_json
from .analyze import build_rows, summarize, position_bias
from .metrics import bootstrap_mean

PATH8 = ROOT/'runs/extension/qwen8b_main.jsonl'
PATH4 = ROOT/'runs/qwen_main.jsonl'
# build_rows labels any non-'qwen' run as 'Smol'; relabel for the 8B run.
RENAME = {'Smol': 'Qwen8B', 'Smol majority': 'Qwen8B majority', 'Smol agree 2/3': 'Qwen8B agree 2/3',
          'Smol unanimous': 'Qwen8B unanimous', 'Qwen': 'Qwen4B', 'Qwen majority': 'Qwen4B majority',
          'Qwen agree 2/3': 'Qwen4B agree 2/3', 'Qwen unanimous': 'Qwen4B unanimous'}
KEYS = ['n', 'coverage', 'utility', 'gain', 'gain_ci', 'regret', 'regret_ci', 'random_regret',
        'hsr', 'hsr_ci', 'random_hsr', 'hsr_delta', 'hsr_delta_ci', 'human_best', 'human_best_ci',
        'stereotype_delta', 'stereotype_delta_ci', 'missing_explicit_delta', 'missing_explicit_delta_ci',
        'missing_implicit_delta', 'missing_implicit_delta_ci', 'stereotype_comparison_n',
        'missing_explicit_comparison_n', 'missing_implicit_comparison_n']

def main():
    groups = load_jsonl(ROOT/'data/derived/main.jsonl')
    total = len(groups)
    df8, order8, val8 = build_rows(groups, [PATH8])
    df4, order4, val4 = build_rows(groups, [PATH4])
    df8['policy'] = df8['policy'].replace(RENAME)
    df4['policy'] = df4['policy'].replace(RENAME)
    rand = df8[df8.policy == 'Random'].set_index('prompt_id')

    # Cross-model gate on original-order choices (frozen cross_model_policy, with 8B in place of Smol).
    p8 = df8[df8.policy == 'Qwen8B'].set_index('prompt_id')
    p4 = df4[df4.policy == 'Qwen4B'].set_index('prompt_id')
    both = p8.index.intersection(p4.index)
    agree = [pid for pid in both if p8.at[pid, 'selected_id'] == p4.at[pid, 'selected_id']]
    disagree = [pid for pid in both if pid not in set(agree)]
    gate = [p8.loc[agree].reset_index().assign(policy='Gate 8B∧4B agree (kept)'),
            p8.loc[disagree].reset_index().assign(policy='Gate rejected: 8B choice'),
            p4.loc[disagree].reset_index().assign(policy='Gate rejected: 4B choice')]
    keep4 = ['Qwen4B', 'Qwen4B majority', 'Qwen4B unanimous']
    combined = pd.concat([df8[df8.policy != 'Oracle'], df4[df4.policy.isin(keep4)], *gate], ignore_index=True)
    summary = {r['policy']: r for r in summarize(combined, total)}
    policies = {}
    for name, r in summary.items():
        row = {k: r.get(k) for k in KEYS}
        ids = combined[combined.policy == name]['prompt_id']
        row['random_best_rate'] = float(rand.loc[ids, 'human_best'].mean())
        policies[name] = row

    # Order sensitivity.
    def orders_block(order):
        fl = order['flip'].to_numpy()
        return {'complete_prompts': int(len(order)), 'flip_rate': float(fl.mean()), 'flip_ci': bootstrap_mean(fl),
                'unanimity_coverage': float((order['agreement'] == 1).sum()/total)}
    pos = position_bias([PATH8, PATH4])
    def slot_block(p):
        a = p['slots']['A']
        return {'first_slot_share': a['rate'], 'first_slot_uniform': a['uniform_rate'],
                'slots': {k: v['rate'] for k, v in p['slots'].items()}, 'chi2': p['chi2'], 'df': p['df'],
                'p_value': p['p_value'], 'calls': p['calls']}

    # Choice agreement with the 4B judge, per permutation and for original order.
    r8 = {(r['prompt_id'], r['permutation']): r for r in load_jsonl(PATH8)}
    r4 = {(r['prompt_id'], r['permutation']): r for r in load_jsonl(PATH4)}
    agreement = {}
    for k in range(3):
        pairs = [(r8[key], r4[key]) for key in r8 if key[1] == k and key in r4
                 and r8[key]['parse_ok'] and r4[key]['parse_ok']]
        same = np.array([a['selected_id'] == b['selected_id'] for a, b in pairs], dtype=float)
        agreement[f'permutation_{k}'] = {'n': len(pairs), 'rate': float(same.mean()) if len(same) else None,
                                         'ci': bootstrap_mean(same)}
    u8 = order8[order8.agreement == 1].set_index('prompt_id')['majority_id']
    u4 = order4[order4.agreement == 1].set_index('prompt_id')['majority_id']
    common = u8.index.intersection(u4.index)
    agreement['both_unanimous'] = {'n': int(len(common)),
                                   'rate': float((u8.loc[common] == u4.loc[common]).mean()) if len(common) else None}
    # Exact random-chance agreement for two independent uniform pickers on the same pools.
    agreement['chance_rate'] = float(np.mean([1/len(g['candidates']) for g in groups]))

    # Input identity check: same rendered chat text as the 4B run for the same keys.
    shared = [key for key in r8 if key in r4 and 'input_text_sha256' in r8[key]]
    text_match = sum(r8[key]['input_text_sha256'] == r4[key]['input_text_sha256'] for key in shared)
    tok_match = sum(r8[key].get('input_shapes') == r4[key].get('input_shapes') for key in shared)
    meta = json.loads(PATH8.with_suffix('.meta.json').read_text())
    expected = {(g['prompt_id'], k) for g in groups for k in range(3)}
    out = {
        'post_hoc': True,
        'model': meta['model'], 'dtype': meta['dtype'], 'device': meta['device'],
        'max_pixels_override': meta.get('max_pixels_override'), 'versions': meta['versions'],
        'validation': {'qwen8b': {**val8['qwen8b'], 'unattempted': len(expected-set(r8)),
                                  'complete': set(r8) == expected},
                       'qwen4b': val4['qwen']},
        'median_seconds_per_call': {'qwen8b': statistics.median(r['seconds'] for r in r8.values()),
                                    'qwen4b': statistics.median(r['seconds'] for r in r4.values())},
        'input_identity_vs_4b': {'keys': len(shared), 'input_text_sha256_match': text_match,
                                 'input_shapes_match': tok_match},
        'policies': policies,
        'orders': {'qwen8b': orders_block(order8), 'qwen4b': orders_block(order4)},
        'position': {'qwen8b': slot_block(pos['qwen8b']), 'qwen4b': slot_block(pos['qwen'])},
        'agreement_with_4b': agreement,
        'cross_model_gate': {'kept_n': len(agree), 'rejected_n': len(disagree), 'kept_coverage': len(agree)/total},
    }
    save_json(ROOT/'results/main/extension.json', out)
    show = ['Qwen8B', 'Qwen4B', 'Qwen8B majority', 'Qwen8B unanimous', 'Gate 8B∧4B agree (kept)',
            'Gate rejected: 8B choice', 'Gate rejected: 4B choice']
    for name in show:
        p = policies.get(name)
        if p:
            print(f"{name:28s} n={p['n']:3d} gain={p['gain']:+.4f} {np.round(p['gain_ci'], 4)} "
                  f"regret={p['regret']:.4f} below={p['hsr']:.3f} (rand {p['random_hsr']:.3f}) best={p['human_best']:.3f}")
    print(json.dumps({k: out[k] for k in ['validation', 'median_seconds_per_call', 'input_identity_vs_4b',
                                          'orders', 'position', 'agreement_with_4b', 'cross_model_gate']}, indent=1))

if __name__ == '__main__':
    main()
