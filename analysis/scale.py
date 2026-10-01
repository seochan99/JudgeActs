"""Scale extension: figure, appendix table, and manuscript text from results/main/extension.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import ROOT
from . import style
from .extra_text import v, p, ci

RES = ROOT / "results/main"
GEN = ROOT / "paper/generated"
DEST = ROOT / "paper/figures"
JUDGES = [("bf16_4b", "4B"), ("mlx_4b", "4B, 8-bit"), ("mlx_8b", "8B, 8-bit")]


def figure(e):
    style.apply()
    J = e["judges"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.33, 2.0), gridspec_kw={"wspace": .5, "width_ratios": [1, 1.4]})
    x = np.arange(len(JUDGES))
    colors = [style.QWEN, "#8DB8E4", style.ORACLE]
    for i, (k, _) in enumerate(JUDGES):
        r = J[k]
        ax1.bar(i, r["first_slot_share"], width=.6, color=colors[i])
        ax1.plot([i - .34, i + .34], [r["first_slot_uniform"]] * 2, color=style.INK, lw=.9, ls=(0, (2, 1.4)))
        ax1.text(i, r["first_slot_share"] + .015, f"{100 * r['first_slot_share']:.0f}%", ha="center", fontsize=6.2)
    ax1.set_xticks(x, [n.replace(", ", "\n") for _, n in JUDGES], fontsize=5.8)
    ax1.set_ylim(0, .62)
    ax1.set_yticks([0, .2, .4, .6])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax1.set_title("(a) First slot chosen", loc="left", fontsize=7.5)
    w = .26
    parts = [("original_order", "all prompts", style.QWEN), ("unanimous", "orders agree", style.AGREE),
             ("rejected_by_unanimity", "orders disagree", style.RANDOM)]
    for j, (key, name, color) in enumerate(parts):
        vals = [J[k][key]["gain"] for k, _ in JUDGES]
        err = np.array([[J[k][key]["gain"] - J[k][key]["gain_ci"][0], J[k][key]["gain_ci"][1] - J[k][key]["gain"]]
                        for k, _ in JUDGES]).T
        ax2.bar(x + (j - 1) * w, vals, width=w, color=color, label=name, yerr=err, ecolor=style.INK,
                error_kw={"lw": .6, "capsize": 1.4})
    ax2.axhline(0, color=style.INK, lw=.6)
    ax2.set_xticks(x, [n.replace(", ", "\n") for _, n in JUDGES], fontsize=5.8)
    ax2.set_title("(b) Gain over random", loc="left", fontsize=7.5)
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.6, loc="lower center", ncol=3, frameon=False, handlelength=1,
               bbox_to_anchor=(.62, -.1), columnspacing=.9)
    ax2.set_ylim(-.07, .19)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=style.GRID, lw=.5)
        ax.set_axisbelow(True)
    fig.savefig(DEST / "scale.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(DEST / "scale.png", dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def table(e):
    J = e["judges"]
    head = (r"Judge & First slot (\%) & Choice changes (\%) & Gain [95\% CI] & Regret & BMR (\%) & "
            r"Unanimous: cov.\ (\%), gain & Rejected: gain [95\% CI] & $\Delta$Stereo.\ [95\% CI]")
    lines = [r"\begin{tabular}{lrrlrrlll}", r"\toprule", head + r"\\", r"\midrule"]
    names = {"bf16_4b": "Qwen3-VL-4B (main, bf16)", "mlx_4b": "Qwen3-VL-4B (8-bit control)",
             "mlx_8b": "Qwen3-VL-8B (8-bit)"}
    for k, _ in JUDGES:
        r = J[k]
        o, u, rj = r["original_order"], r["unanimous"], r["rejected_by_unanimity"]
        lines.append(" & ".join([names[k], f"{100 * r['first_slot_share']:.1f}", f"{100 * r['flip_rate']:.1f}",
                                 f"{v(o['gain'])} {ci(o['gain_ci'])}", v(o["regret"]), f"{100 * o['hsr']:.1f}",
                                 f"{100 * u['coverage']:.1f}, {v(u['gain'])}", f"{v(rj['gain'])} {ci(rj['gain_ci'])}",
                                 f"{v(o['stereotype_delta'])} {ci(o['stereotype_delta_ci'])}"]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "appendix_scale.tex").write_text("\n".join(lines) + "\n")


def text(e):
    J = e["judges"]
    b, c, g = J["bf16_4b"], J["mlx_4b"], J["mlx_8b"]
    cm = e["cross_model_gate_mlx8b_bf16_4b"]
    clip = json.loads((RES / "baselines.json").read_text())["policies"]["CLIP"] if (RES / "baselines.json").exists() else None
    diff = paired_vs_clip()
    t = [
        "Is position bias a property of small judges? We ran the larger Qwen3-VL-8B-Instruct on the same 300 prompts and "
        "three orders. To fit it in memory we used 8-bit weights, so we also reran the 4B judge with the same quantization "
        f"as a control. The control agrees with the main 4B run on {p(c['agreement_with_bf16_4b']['all']['rate'])} of calls "
        f"and reproduces its first-slot share ({p(c['first_slot_share'])}) and gain ({v(c['original_order']['gain'])}), so "
        "neither quantization nor the MLX runtime explains the differences below (Figure~\\ref{fig:scale}, Appendix Table~\\ref{tab:scale}).",
        f"At 8B the position bias largely disappears: the first slot is chosen in {p(g['first_slot_share'])} of calls, close "
        f"to the uniform {p(g['first_slot_uniform'])} ($p={g['position_p']:.2f}$), although the choice still changes across "
        f"orders on {p(g['flip_rate'])} of prompts. The 8B judge is also much better: its gain over random is "
        f"{v(g['original_order']['gain'])} {ci(g['original_order']['gain_ci'])}"
        + (f", above the CLIP baseline ({v(clip['gain'])}; paired difference {v(diff[0])} {ci(diff[1])})" if clip and diff else "")
        + f", and its stereotype difference is no longer detectable ({v(g['original_order']['stereotype_delta'])} "
        f"{ci(g['original_order']['stereotype_delta_ci'])}).",
        "The value of the unanimity gate changes with it. For the 4B judge, the prompts unanimity rejects fall below random; "
        f"for the 8B judge they stay above random ({v(g['rejected_by_unanimity']['gain'])} "
        f"{ci(g['rejected_by_unanimity']['gain_ci'])}), so the gate mostly discards good decisions. Agreement between the 8B and "
        f"4B judges shows the same pattern: on the {cm['rejected_n']} prompts where they disagree, the 8B choice has gain "
        f"{v(cm['rejected_8b_choice']['gain'])} and the 4B choice {v(cm['rejected_4b_choice']['gain'])}. A gate is useful only "
        "while it filters the failure mode of the judge in front of it, and must be re-audited when the judge changes.",
    ]
    (GEN / "scale_text.tex").write_text("\n\n".join(t) + "\n")


def paired_vs_clip():
    """Paired per-prompt difference in returned utility, 8B original order minus CLIP ViT-L/14."""
    from .content_baselines import select
    from .metrics import bootstrap_mean
    from src.common import load_jsonl
    path = RES / "clip_scores.json"
    run = ROOT / "runs/extension/qwen8b_mlx_main.jsonl"
    if not (path.exists() and run.exists()):
        return None
    scores = json.loads(path.read_text())["scores"]
    picks = {r["prompt_id"]: r["selected_id"] for r in load_jsonl(run) if r["permutation"] == 0 and r["parse_ok"]}
    d = []
    for g in load_jsonl(ROOT / "data/derived/main.jsonl"):
        if g["prompt_id"] in picks:
            u = {c["id"]: c["utility"] for c in g["candidates"]}
            d.append(u[picks[g["prompt_id"]]] - u[select(g, scores)])
    d = np.array(d)
    return float(d.mean()), bootstrap_mean(d)


def main():
    path = RES / "extension.json"
    if not path.exists():
        return
    e = json.loads(path.read_text())
    figure(e)
    table(e)
    text(e)


if __name__ == "__main__":
    main()
