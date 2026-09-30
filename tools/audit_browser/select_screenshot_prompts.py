"""Pick appendix detail prompts by fixed rules; writes screenshot_prompts.json.

Rules (ties broken by prompt_id ascending):
  detail_2: 4-candidate prompt with the largest Qwen original-order regret.
  detail_3: prompt where Qwen and Smol original-order picks agree (cross-model gate accepts)
            with the largest regret of the shared pick, excluding the detail_2 prompt (the
            same prompt tops both rules, so the next one is taken to show a second example).
Usage:  uv run python tools/audit_browser/select_screenshot_prompts.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / 'data.js').read_text()[len('window.AUDIT = '):].rstrip().rstrip(';'))
P = d['prompts']
four = [p for p in P if len(p['candidates']) == 4 and 'm0' in p['qwen']]
cross = [p for p in P if p['gates']['cross_model']]
pick = lambda xs: sorted(xs, key=lambda p: (-p['qwen']['m0']['regret'], p['id']))[0]
a = pick(four)
b = pick([p for p in cross if p['id'] != a['id']])  # distinct from detail_2
out = {
    'detail': {'prompt_id': '7c51b0579f2a01ed23dbf4a6a21134431808d94f57d4ab74f5e93d3a275d2a23',
               'rule': 'fixed example (Chile, pastel de choclo)', 'file': 'paper/figures/browser_detail.png'},
    'detail_2': {'prompt_id': a['id'], 'prompt': a['prompt'], 'country': a['country'],
                 'rule': '4-candidate prompt with largest Qwen original-order regret (ties by prompt_id)',
                 'qwen_regret': a['qwen']['m0']['regret'], 'file': 'paper/figures/browser_detail_2.png'},
    'detail_3': {'prompt_id': b['id'], 'prompt': b['prompt'], 'country': b['country'],
                 'rule': 'cross-model agreement (Qwen pick == Smol pick, original order) with largest regret (ties by prompt_id), excluding the detail_2 prompt, which the rule would otherwise select again',
                 'regret': b['qwen']['m0']['regret'], 'file': 'paper/figures/browser_detail_3.png'},
    'explore_below': {'route': '#explore?outcome=below&sort=regret', 'file': 'paper/figures/browser_explore_below.png'},
    'explore_unanimous': {'route': '#explore?outcome=unanimous', 'file': 'paper/figures/browser_explore_unanimous.png'},
}
(HERE / 'screenshot_prompts.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
# show ties for transparency
for name, xs in [('four', four), ('cross', cross)]:
    top = max(p['qwen']['m0']['regret'] for p in xs)
    print(name, 'tied at max:', sum(abs(p['qwen']['m0']['regret'] - top) < 1e-12 for p in xs))
