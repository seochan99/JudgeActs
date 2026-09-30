"""Bar-chart statistical figures for the manuscript (AAAI two-column layout).

Every plotted value is read from the frozen main-run outputs in ``results/main``.
Outputs (PDF + 300 dpi PNG) in ``paper/figures``:

* ``results_main``     full width: regret, gate gains, presentation-slot preference
* ``country_bars``     full width: per-country regret for Random / Qwen / Smol
* ``cultural_deltas``  column width: Qwen minus matched random error rates
* ``coverage_bars``    column width: selective regret vs. matched random by gate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis.metrics import bootstrap_mean
from analysis.style import (AGREE, BAD, GOOD, GRID, INK, MUTED, ORACLE, QWEN,
                            RANDOM, SMOL, apply)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "main"
OUT = ROOT / "paper" / "figures"
COL, FULL = 3.33, 7.0
ERR = dict(ecolor=INK, elinewidth=.7, capsize=2.2, capthick=.7)
NEUTRAL = "#B9BEC4"


# --------------------------------------------------------------------- data
def load():
    summary = {r["policy"]: r for r in json.loads((RESULTS / "summary.json").read_text())}
    country = json.loads((RESULTS / "country.json").read_text())
    position = json.loads((RESULTS / "position.json").read_text())
    prompts = pd.read_csv(RESULTS / "prompt_metrics.csv")
    return summary, country, position, prompts


def _yerr(values, cis):
    values = np.asarray(values, float)
    cis = np.asarray(cis, float)
    return np.vstack([values - cis[:, 0], cis[:, 1] - values])


def _pct(x):
    return f"{100 * x:.0f}%"


def _signed(v):
    return f"{v:+.3f}".replace("-", "\u2212")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)


def _refline(ax, y, label, x=None, ha="right", color=MUTED, horizontal=True):
    if horizontal:
        ax.axhline(y, color=color, ls=(0, (3, 2)), lw=.7, zorder=1)
        if label:
            ax.text(ax.get_xlim()[1] if x is None else x, y, label, ha=ha, va="bottom",
                    fontsize=6, color=color)
    else:
        ax.axvline(y, color=color, ls=(0, (3, 2)), lw=.7, zorder=1)


def matched_random_regret(prompts, policy):
    """Random regret on the prompts a policy/gate accepted (oracle - random utility)."""
    rows = prompts[prompts.policy == policy]
    vals = (rows.oracle_utility - rows.random_utility).to_numpy(float)
    return float(vals.mean()), bootstrap_mean(vals), len(vals)


# ----------------------------------------------------------- results_main
def _panel_regret(ax, summary):
    names = ["Random", "Qwen", "Smol"]
    colors = [RANDOM, QWEN, SMOL]
    vals = [summary[n]["regret"] for n in names]
    cis = [summary[n]["regret_ci"] for n in names]
    x = np.arange(len(names))
    ax.bar(x, vals, .62, color=colors, zorder=2, **{})
    ax.errorbar(x, vals, yerr=_yerr(vals, cis), fmt="none", zorder=3, **ERR)
    for xi, v, ci in zip(x, vals, cis):
        ax.text(xi, ci[1] + .008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, names, fontsize=6.8)
    ax.set_xlim(-.55, len(names) - .45)
    top = max(c[1] for c in cis)
    ax.set_ylim(0, top * 1.3)
    ax.text(.03, .985, f"Oracle = {summary['Oracle']['regret']:.0f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.3, color=ORACLE)
    ax.set_ylabel("Mean regret (lower is better)")
    ax.set_title("(a) Selection regret")
    _grid(ax)


GATE_ROWS = [  # (policy key, short label, group)
    ("Qwen", "All prompts", None),
    ("Qwen unanimous", "3/3 orders agree", "Order gate"),
    ("Qwen, not unanimous", "Orders disagree", "Order gate"),
    ("Cross-model", "Judges agree", "Cross-model gate"),
    ("Qwen, judges disagree", "Judges disagree", "Cross-model gate"),
]


def _gain_color(ci):
    if ci[0] > 0:
        return GOOD
    if ci[1] < 0:
        return BAD
    return NEUTRAL


def _panel_gates(ax, summary):
    ys = np.array([0, 1.6, 2.6, 4.2, 5.2])
    vals = [summary[k]["gain"] for k, _, _ in GATE_ROWS]
    cis = [summary[k]["gain_ci"] for k, _, _ in GATE_ROWS]
    colors = [_gain_color(c) for c in cis]
    ax.barh(ys, vals, .7, color=colors, zorder=2)
    ax.errorbar(vals, ys, xerr=_yerr(vals, cis), fmt="none", zorder=3, **ERR)
    ax.axvline(0, color=INK, lw=.7, zorder=2)
    lo = min(c[0] for c in cis)
    hi = max(c[1] for c in cis)
    for y, (k, _, _), v, ci in zip(ys, GATE_ROWS, vals, cis):
        r = summary[k]
        ax.text(max(ci[1], 0) + .007, y, _signed(v) + f"  (n={r['n']})", ha="left", va="center",
                fontsize=6)
    ax.set_xlim(lo - .012, hi + .1)
    ax.set_yticks(ys, [lab for _, lab, _ in GATE_ROWS], fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(5.75, -.5)
    tr = ax.get_yaxis_transform()
    for y, g in ((ys[1] - .78, "Order gate"),
                 (ys[3] - .78, "Cross-model gate")):
        ax.text(.015, y, g, transform=tr, ha="left", va="center", fontsize=6.3,
                color=AGREE, fontweight="bold")
    ax.set_xlabel("Gain over random (regret reduction)")
    ax.set_title("(b) Gain over random: kept vs. rejected")
    _grid(ax, "x")


def _panel_slots(ax, position):
    slots = ["A", "B", "C", "D"]
    x = np.arange(len(slots))
    w = .36
    for off, key, name, color in ((-w / 2, "qwen", "Qwen", QWEN), (w / 2, "smol", "Smol", SMOL)):
        s = position[key]["slots"]
        rates = [s[k]["rate"] for k in slots]
        ax.bar(x + off, rates, w, color=color, label=name, zorder=2)
        for xi, u in zip(x + off, [s[k]["uniform_rate"] for k in slots]):
            ax.plot([xi - w / 2, xi + w / 2], [u, u], color=INK, ls=(0, (2, 1.2)), lw=.8, zorder=3)
    ax.set_xticks(x, slots)
    ax.set_xlabel("Presentation slot")
    ax.set_ylabel("Share of picks")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    top = max(position[k]["slots"][s]["rate"] for k in ("qwen", "smol") for s in slots)
    ax.set_ylim(0, top * 1.42)
    ax.set_yticks([t / 10 for t in range(int(top * 10) + 2)])
    handles = [Patch(color=QWEN, label="Qwen"), Patch(color=SMOL, label="Smol"),
               Line2D([], [], color=INK, ls=(0, (2, 1.2)), lw=.8, label="Uniform")]
    ax.legend(handles=handles, loc="upper right", ncol=3, handlelength=1.2, borderpad=.3, columnspacing=.8,
              labelspacing=.25)
    ax.set_title("(c) Presentation-slot preference")
    _grid(ax)


def results_main(summary, position):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3),
                             gridspec_kw=dict(width_ratios=[.86, 1.3, 1.0], wspace=.62))
    _panel_regret(axes[0], summary)
    _panel_gates(axes[1], summary)
    _panel_slots(axes[2], position)
    _save(fig, "results_main")


# ------------------------------------------------------------ country_bars
def country_bars(country):
    rows = {(r["country"], r["policy"]): r for r in country}
    names = sorted({r["country"] for r in country},
                   key=lambda c: -rows[(c, "Random")]["regret"])
    pols = [("Random", RANDOM), ("Qwen", QWEN), ("Smol", SMOL)]
    x = np.arange(len(names))
    w = .26
    fig, ax = plt.subplots(figsize=(FULL, 2.0))
    for i, (p, color) in enumerate(pols):
        vals = [rows[(c, p)]["regret"] for c in names]
        cis = [rows[(c, p)]["regret_ci"] for c in names]
        xs = x + (i - 1) * w
        ax.bar(xs, vals, w, color=color, label=p, zorder=2)
        ax.errorbar(xs, vals, yerr=_yerr(vals, cis), fmt="none", zorder=3,
                    **{**ERR, "capsize": 1.4, "elinewidth": .55, "capthick": .55})
    ax.set_xticks(x, [c.replace("_", " ") for c in names])
    ax.set_xlim(-.55, len(names) - .45)
    ax.set_ylabel("Mean regret (lower is better)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", ncol=3, handlelength=1.2, columnspacing=1, borderpad=.35)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    _grid(ax)
    _save(fig, "country_bars")


# --------------------------------------------------------- cultural_deltas
def cultural_deltas(summary):
    r = summary["Qwen"]
    items = [("stereotype", "Stereotype"), ("missing_explicit", "Missing explicit\ncue"),
             ("missing_implicit", "Missing implicit\ncue")]
    ys = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    for y, (k, _) in zip(ys, items):
        v = r[f"{k}_delta"]
        lo, hi = r[f"{k}_delta_ci"]
        color = BAD if lo > 0 else GOOD if hi < 0 else NEUTRAL
        bonf = r.get(f"{k}_delta_ci_bonf3")
        if bonf:
            ax.plot(bonf, [y, y], color=color, alpha=.35, lw=6, solid_capstyle="butt", zorder=2)
        ax.plot([lo, hi], [y, y], color=color, lw=6, solid_capstyle="butt", zorder=3)
        ax.plot(v, y, "o", ms=4.2, color="white", mec=INK, mew=.8, zorder=4)
        ends = bonf or (lo, hi)
        if v >= 0:
            ax.text(ends[1] + .004, y, _signed(v), ha="left", va="center", fontsize=6.3)
        else:
            ax.text(ends[0] - .004, y, _signed(v), ha="right", va="center", fontsize=6.3)
    ax.axvline(0, color=INK, lw=.7, zorder=1)
    ax.set_yticks(ys, [lab for _, lab in items])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(items) - .45, -.75)
    lim = max(abs(x) for k, _ in items
              for x in (r.get(f"{k}_delta_ci_bonf3") or r[f"{k}_delta_ci"]))
    ax.set_xlim(-lim * 1.4, lim * 1.02)
    ax.set_xlabel("Qwen $-$ matched random (error rate)")
    ax.text(ax.get_xlim()[0], -.62, "← fewer errors", ha="left", va="center",
            fontsize=6, color=GOOD)
    ax.text(ax.get_xlim()[1], -.62, "more errors →", ha="right", va="center",
            fontsize=6, color=BAD)
    handles = [Patch(color=MUTED, label="95% CI"),
               Patch(color=MUTED, alpha=.35, label="Bonferroni (3 tests)")]
    ax.legend(handles=handles, loc="lower right", fontsize=6, handlelength=1.2,
              borderpad=.3, labelspacing=.2)
    _grid(ax, "x")
    _save(fig, "cultural_deltas")


# ----------------------------------------------------------- coverage_bars
GATES = [("Qwen majority", "Majority"), ("Qwen agree 2/3", "≥2/3"),
         ("Qwen unanimous", "3/3"), ("Cross-model", "Judges")]


def coverage_bars(summary, prompts):
    x = np.arange(len(GATES))
    w = .36
    q = [summary[k]["regret"] for k, _ in GATES]
    qci = [summary[k]["regret_ci"] for k, _ in GATES]
    rnd = [matched_random_regret(prompts, k) for k, _ in GATES]
    rv = [m for m, _, _ in rnd]
    rci = [ci for _, ci, _ in rnd]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    colors = [AGREE if k == "Cross-model" else QWEN for k, _ in GATES]
    ax.bar(x - w / 2, rv, w, color=RANDOM, zorder=2)
    ax.bar(x + w / 2, q, w, color=colors, zorder=2)
    ax.errorbar(x - w / 2, rv, yerr=_yerr(rv, rci), fmt="none", zorder=3, **ERR)
    ax.errorbar(x + w / 2, q, yerr=_yerr(q, qci), fmt="none", zorder=3, **ERR)
    for xi, v, ci in zip(x + w / 2, q, qci):
        ax.text(xi, ci[1] + .006, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for xi, v, ci in zip(x - w / 2, rv, rci):
        ax.text(xi, ci[1] + .006, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(x, [f"{lab}\n{_pct(summary[k]['coverage'])}, n={summary[k]['n']}"
                      for k, lab in GATES])
    ax.set_ylabel("Mean regret on kept prompts")
    top = max(max(c[1] for c in qci), max(c[1] for c in rci))
    ax.set_ylim(0, top * 1.28)
    handles = [Patch(color=RANDOM, label="Random (matched)"),
               Patch(color=QWEN, label="Qwen, order gate"),
               Patch(color=AGREE, label="Qwen, cross-model gate")]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=6, handlelength=1.1,
              columnspacing=.8, borderpad=.3, labelspacing=.2)
    ax.set_xlabel("Gate (coverage of 300 prompts)", labelpad=2)
    _grid(ax)
    _save(fig, "coverage_bars")
    return rnd


def main():
    apply()
    summary, country, position, prompts = load()
    results_main(summary, position)
    country_bars(country)
    cultural_deltas(summary)
    rnd = coverage_bars(summary, prompts)
    for (k, _), (m, ci, n) in zip(GATES, rnd):
        print(f"matched random regret {k}: {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] n={n}"
              f" (summary random_regret={summary[k]['random_regret']:.3f})")


if __name__ == "__main__":
    main()
