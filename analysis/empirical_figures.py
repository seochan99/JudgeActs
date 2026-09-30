"""Column-width statistical figures exported as PDF, editable SVG and PNG.

The manuscript exports are gated on the frozen main run.  Development plots can
be rendered explicitly with ``render(..., preview=True)`` into a separate folder.
No estimated or illustrative observations are used in pending panels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from analysis.metrics import bootstrap_mean
from src.common import ROOT

WIDTH = 3.35
INK = "#263238"
GRAY = "#7A8087"
TEAL = "#147D92"
ORANGE = "#B87540"
RULE = "#DCE1E4"
COLORS = {"Random": GRAY, "Qwen": TEAL, "Smol": ORANGE, "Oracle": INK}
MARKERS = {"Random": "s", "Qwen": "o", "Smol": "D", "Oracle": "|"}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": RULE, "axes.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": True,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": None,
}


def _save(fig, dest: Path, name: str, preview: bool = False):
    dest.mkdir(parents=True, exist_ok=True)
    if preview:
        fig.text(.99, .992, "DEV PREVIEW · NOT MAIN RESULTS", ha="right", va="top",
                 fontsize=6.5, color=ORANGE)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(dest / f"{name}.{suffix}", dpi=300)
    plt.close(fig)


def _interval(row):
    value = float(row["regret"])
    low, high = map(float, row["regret_ci"])
    if not all(np.isfinite([value, low, high])):
        raise ValueError(f"Non-finite regret interval for {row.get('policy')}")
    if low > high:
        raise ValueError("The reported CI endpoints are reversed")
    return value, low, high


def _point(ax, row, y, policy, size=4.8):
    value, low, high = _interval(row)
    color = COLORS[policy]
    ax.plot([low, high], [y, y], color=color, lw=1.25, zorder=2,
            solid_capstyle="butt")
    ax.plot([low, high], [y, y], "|", color=color, ms=4, mew=.8, zorder=2)
    ax.plot(value, y, MARKERS[policy], color=color, ms=size,
            mec="white" if policy != "Oracle" else color,
            mew=.6 if policy != "Oracle" else 1.4, zorder=3)


def _vertical_point(ax, x, value, low, high, color, marker, size):
    """Keep the reported interval intact even for asymmetric percentile CIs."""
    ax.plot([x, x], [low, high], color=color, lw=1, zorder=2)
    ax.plot([x, x], [low, high], "_", color=color, ms=4, mew=.8, zorder=2)
    ax.plot(x, value, marker, color=color, ms=size,
            mec="white", mew=.6, zorder=3)


def _horizontal_axis(ax, rows):
    largest = max(_interval(row)[2] for row in rows)
    ax.set_xlim(-.007, max(.10, largest * 1.14))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=.5)
    ax.axvline(0, color=GRAY, lw=.6, zorder=1)
    ax.set_xlabel("Selection regret", labelpad=7)


def policy_regret(summary, dest, preview=False):
    by = {row["policy"]: row for row in summary}
    policies = [p for p in ("Random", "Qwen", "Smol", "Oracle") if p in by]
    rows = [by[p] for p in policies]
    fig = plt.figure(figsize=(WIDTH, 2.65))
    ax = fig.add_axes([.275, .29, .565, .51])
    for y, policy in enumerate(policies):
        row = by[policy]
        _point(ax, row, y, policy)
        value = _interval(row)[0]
        ax.text(1.055, y, f"{value:.3f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.5, color=COLORS[policy])
    ax.set_yticks(range(len(policies)), policies)
    ax.tick_params(axis="y", pad=8)
    ax.set_ylim(len(policies) - .45, -.55)
    _horizontal_axis(ax, rows)
    fig.text(.045, .90, "Policy comparison", fontsize=10, weight="bold")
    fig.text(.045, .825, "Mean and 95% bootstrap CI", fontsize=7.5, color=GRAY)
    fig.text(.895, .825, "Mean", fontsize=7.5, ha="center", color=GRAY)
    total = int(by["Random"]["n"])
    supports = " · ".join(f"{p} {int(by[p]['n'])}/{total}" for p in ("Qwen", "Smol") if p in by)
    fig.text(.045, .075, "Valid original-order selections", fontsize=7.2, color=GRAY)
    fig.text(.045, .033, supports, fontsize=7.2, color=GRAY)
    _save(fig, dest, "main_regret", preview)


def country_regret(country, summary, dest, preview=False):
    by = {row["policy"]: row for row in summary}
    policies = [p for p in ("Random", "Qwen", "Smol") if p in by]
    relevant = [row for row in country if row["policy"] in policies]
    labels = sorted({row["country"] for row in relevant})
    index = {(row["country"], row["policy"]): row for row in relevant}
    fig = plt.figure(figsize=(WIDTH, 3.95))
    ax = fig.add_axes([.29, .20, .65, .60])
    for y, label in enumerate(labels):
        ax.axhline(y, color=RULE, lw=.35, zorder=0)
        for j, policy in enumerate(policies):
            row = index.get((label, policy))
            if row is None:
                continue
            offset = (j - (len(policies)-1)/2) * .20
            _point(ax, row, y + offset, policy, size=3.6 if policy == "Random" else 4.2)
    ax.set_yticks(range(len(labels)), [s.replace("_", " ") for s in labels])
    ax.tick_params(axis="y", pad=7)
    ax.set_ylim(len(labels)-.45, -.55)
    _horizontal_axis(ax, relevant)
    fig.text(.045, .925, "Country-group risk", fontsize=10, weight="bold")
    handles = [Line2D([], [], color=COLORS[p], marker=MARKERS[p], lw=1,
                      ms=4, label=p) for p in policies]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.025, .88),
               ncol=3, frameon=False, handlelength=1.3, columnspacing=1.1,
               fontsize=8, borderaxespad=0)
    fig.text(.045, .035, "Mean and 95% bootstrap CI · alphabetical groups", fontsize=7.1, color=GRAY)
    _save(fig, dest, "country_regret", preview)


def coverage_risk(summary, prompt_metrics, dest, preview=False):
    by = {row["policy"]: row for row in summary}
    points = [(p, label) for p, label in (
        ("Qwen unanimous", "3/3"), ("Qwen agree 2/3", "≥2/3"),
        ("Qwen majority", "All")) if p in by]
    fig = plt.figure(figsize=(WIDTH, 3.30))
    ax = fig.add_axes([.19, .21, .75, .54])
    tops = []
    for policy, label in points:
        row = by[policy]
        sub = prompt_metrics[prompt_metrics.policy == policy]
        if sub.empty:
            raise ValueError(f"No matched-subset rows for {policy}")
        if len(sub) != int(row["n"]) or sub.prompt_id.duplicated().any():
            raise ValueError(f"Expected one row per accepted prompt for {policy}")
        x = float(row["coverage"])
        value, low, high = _interval(row)
        random_values = (sub.oracle_utility - sub.random_utility).to_numpy()
        rv = float(random_values.mean())
        rlow, rhigh = bootstrap_mean(random_values)
        tops.extend([high, rhigh])
        # Unconnected points: only the three attainable agreement gates exist.
        _vertical_point(ax, x, rv, rlow, rhigh, GRAY, "s", 4)
        _vertical_point(ax, x, value, low, high, TEAL, "o", 5)
        # Direct gate labels refer to the observed operating points, not a curve.
        ax.annotate(label, (x, value), xytext=((-6, 5) if x > .96 else (6, 5)),
                    textcoords="offset points", ha="right" if x > .96 else "left",
                    fontsize=7.5, color=TEAL)
    if "Cross-model" in by:
        row = by["Cross-model"]
        value, low, high = _interval(row)
        tops.append(high)
        _vertical_point(ax, row["coverage"], value, low, high, ORANGE, "D", 4.8)
    ax.set_xlim(-.035, 1.045)
    ax.set_ylim(-.007, max(.12, max(tops, default=.12)*1.14))
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.set_xlabel("Automation coverage", labelpad=7)
    ax.set_ylabel("Selective regret", labelpad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, linewidth=.5)
    fig.text(.045, .93, "Selective automation", fontsize=10, weight="bold")
    handles = [Line2D([], [], marker="o", color=TEAL, linestyle="none", ms=4,
                      label="Qwen majority"),
               Line2D([], [], marker="s", color=GRAY, linestyle="none", ms=4,
                      label="Random, matched subset")]
    if "Cross-model" in by:
        handles.append(Line2D([], [], marker="D", color=ORANGE, linestyle="none",
                              ms=4, label="Cross-model agreement"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.025, .89),
               frameon=False, fontsize=7.6, handlelength=1.2, labelspacing=.35,
               borderaxespad=0)
    fig.text(.045, .035, "Gates: 3/3, ≥2/3, All · bars: 95% bootstrap CI", fontsize=7, color=GRAY)
    _save(fig, dest, "coverage_risk", preview)


def pending(dest):
    specs = (
        ("main_regret", "Policy comparison", ["Random", "Qwen", "Smol", "Oracle"]),
        ("country_regret", "Country-group risk", ["Country means", "95% bootstrap intervals"]),
        ("coverage_risk", "Selective automation", ["Automation coverage", "Regret on accepted prompts"]),
    )
    for name, title, labels in specs:
        fig = plt.figure(figsize=(WIDTH, 2.35))
        fig.text(.045, .91, title, fontsize=10, weight="bold")
        fig.text(.045, .78, "Frozen main evaluation in progress", fontsize=8, color=GRAY)
        for j, label in enumerate(labels):
            y = .61 - j*.105
            fig.text(.06, y, label, fontsize=9, color=INK)
            fig.text(.925, y, "Pending", ha="right", fontsize=8, color=GRAY)
            fig.add_artist(Line2D([.045, .95], [y-.038, y-.038],
                                  transform=fig.transFigure, color=RULE, lw=.5))
        fig.text(.045, .065, "No empirical observations displayed", fontsize=7.2, color=GRAY)
        _save(fig, dest, name)


def render(result_dir: Path, dest: Path, preview: bool = False):
    """Render only the requested result directory; never mix dev and main."""
    summary = json.loads((result_dir / "summary.json").read_text())
    country = json.loads((result_dir / "country.json").read_text())
    prompt_metrics = pd.read_csv(result_dir / "prompt_metrics.csv")
    with plt.rc_context(STYLE):
        policy_regret(summary, dest, preview)
        country_regret(country, summary, dest, preview)
        coverage_risk(summary, prompt_metrics, dest, preview)


def main():
    result_dir = ROOT / "results/main"
    dest = ROOT / "paper/figures"
    status_path = result_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    with plt.rc_context(STYLE):
        if not (status.get("complete_primary") and status.get("complete_secondary")):
            pending(dest)
            return
    render(result_dir, dest)


if __name__ == "__main__":
    main()
