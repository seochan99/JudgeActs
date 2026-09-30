"""Ablation summary figure and appendix table, read from results/main/ablations.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import ROOT
from . import style

RES = ROOT / "results/main"
DEST = ROOT / "paper/figures"
GEN = ROOT / "paper/generated"
CONDITIONS = [("reference_main", "Main"), ("choice_only", "No\nranking"),
              ("opaque_labels", "Opaque\ncodes"), ("rotation4", "4\norders")]
LONG = {"Main": "Main (3 orders)", "No\nranking": "No ranking (choice-only format)", "Opaque\ncodes": "Opaque codes (no letters)",
        "4\norders": "Four orders (4-image pools)"}


def flat(a, key):
    if key != "reference_main":
        return a[key]
    m = a[key]
    slots = m["slot_choice"]["slots"]
    first = next(v for k, v in slots.items() if k.startswith("A"))
    un = m["orders"]["unanimous"]
    return {"slot_first_rate": first["share"], "slot_first_uniform": first["uniform"],
            "gain": m["original_order"]["gain_vs_random"], "gain_ci": m["original_order"]["gain_ci"],
            "unanimous_gain": un["gain_vs_random"], "unanimous_gain_ci": un["gain_ci"],
            "unanimous_coverage": un["coverage"], "flip_rate": m["orders"]["flip_rate"], "calls": m["calls"]}


def figure(a):
    style.apply()
    rows = [(label, flat(a, key)) for key, label in CONDITIONS if key in a]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.33, 2.0), gridspec_kw={"wspace": .5})
    x = np.arange(len(rows))
    ax1.bar(x, [r["slot_first_rate"] for _, r in rows], width=.62, color=style.QWEN)
    for i, (_, r) in enumerate(rows):
        ax1.plot([i - .36, i + .36], [r["slot_first_uniform"]] * 2, color=style.INK, lw=.9, ls=(0, (2, 1.4)))
        ax1.text(i, r["slot_first_rate"] + .015, f"{100 * r['slot_first_rate']:.0f}%", ha="center", fontsize=6.2)
    ax1.set_ylim(0, .66)
    ax1.set_yticks([0, .2, .4, .6])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax1.set_title("(a) First slot chosen", loc="left", fontsize=7.5)
    for ax in (ax1, ax2):
        ax.set_xticks(x, [label for label, _ in rows], fontsize=5.5)
        ax.grid(axis="y", color=style.GRID, lw=.5)
        ax.set_axisbelow(True)
    w = .32
    for off, key, color, name in [(-w / 2, "gain", style.QWEN, "all prompts"),
                                  (w / 2, "unanimous_gain", style.AGREE, "unanimous")]:
        vals = [r[key] for _, r in rows]
        err = np.array([[r[key] - r[key + "_ci"][0], r[key + "_ci"][1] - r[key]] for _, r in rows]).T
        ax2.bar(x + off, vals, width=w, color=color, label=name, yerr=err, ecolor=style.INK,
                error_kw={"lw": .6, "capsize": 1.5})
    ax2.axhline(0, color=style.INK, lw=.6)
    ax2.set_title("(b) Gain over random", loc="left", fontsize=7.5)
    ax2.legend(fontsize=5.6, loc="upper left", frameon=False, handlelength=1, borderaxespad=.1)
    ax2.set_ylim(-.02, .21)
    fig.savefig(DEST / "ablations.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(DEST / "ablations.png", dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    return rows


def table(rows):
    def f(x):
        return f"{x:.3f}".replace("-", "$-$")
    lines = [r"\begin{tabular}{lrrrll}", r"\toprule",
             r"Condition & First slot (\%) & Uniform (\%) & Choice changes (\%) & Gain [95\% CI] & Unanimous: coverage, gain [95\% CI]\\",
             r"\midrule"]
    for label, r in rows:
        lines.append(" & ".join([LONG[label], f"{100 * r['slot_first_rate']:.1f}",
                                 f"{100 * r['slot_first_uniform']:.1f}", f"{100 * r['flip_rate']:.1f}",
                                 f"{f(r['gain'])} [{f(r['gain_ci'][0])}, {f(r['gain_ci'][1])}]",
                                 f"{100 * r['unanimous_coverage']:.1f}\\%, {f(r['unanimous_gain'])} "
                                 f"[{f(r['unanimous_gain_ci'][0])}, {f(r['unanimous_gain_ci'][1])}]"]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "appendix_ablations.tex").write_text("\n".join(lines) + "\n")


def main():
    path = RES / "ablations.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    rows = figure(a)
    table(rows)
    from .appendix_tables import normalize
    p = GEN / "appendix_ablations.tex"
    p.write_text(normalize(p.read_text()))


if __name__ == "__main__":
    main()
