"""Local-only candidate boards; source-image reuse rights are not established.

Examples are the first frozen main-manifest set of each pool size, selected by
manifest position alone, never by model correctness or an attractive outcome.
Outputs are deliberately outside paper/figures and excluded from submission/git.
"""
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.common import ROOT, load_jsonl, save_json
from .empirical_figures import STYLE, INK, TEAL, ORANGE, GRAY, RULE


def main():
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    records = {}
    for model in ["qwen", "smol"]:
        path = ROOT / f"runs/{model}_main.jsonl"
        records[model] = {(r["prompt_id"], r["permutation"]): r for r in load_jsonl(path)}
    examples = [next(g for g in groups if len(g["candidates"]) == size) for size in [3, 4]]
    dest = ROOT / "results/visual_review"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    with plt.rc_context(STYLE):
        for group in examples:
            cs = group["candidates"]
            ids = [c["id"] for c in cs]
            labels = list("ABCD"[:len(cs)])
            selections = {}
            for model in ["qwen", "smol"]:
                r = records[model].get((group["prompt_id"], 0))
                selections[model] = r.get("selected_id") if r and r["parse_ok"] else None
            fig = plt.figure(figsize=(7.1, 4.9), facecolor="white")
            fig.text(.035, .965, "Recorded image-selection example", fontsize=11, weight="bold")
            fig.text(.035, .913, "\n".join(textwrap.wrap(group["prompt"], 112)), fontsize=9,
                     va="top", linespacing=1.25)
            left, width, gap = .035, .218, .019
            if len(cs) == 3:
                left = .145
            for i, candidate in enumerate(cs):
                x = left + i * (width + gap)
                ax = fig.add_axes([x, .385, width, .40])
                with Image.open(ROOT / candidate["image"]) as source:
                    ax.imshow(source.convert("RGB"))
                ax.set_axis_off()
                fig.text(x, .807, labels[i], weight="bold", fontsize=10)
                marks = []
                if selections["qwen"] == candidate["id"]:
                    marks.append((TEAL, "Qwen"))
                if selections["smol"] == candidate["id"]:
                    marks.append((ORANGE, "Smol"))
                for j, (color, name) in enumerate(marks):
                    fig.add_artist(Rectangle((x-.004-j*.004, .38-j*.004), width+.008+j*.008,
                        .41+j*.008, transform=fig.transFigure, fill=False,
                        edgecolor=color, linewidth=1.2))
                fig.text(x, .345, f"Alignment  {candidate['utility']:.3f}", fontsize=9)
                fig.add_artist(Rectangle((x, .302), width, .019, transform=fig.transFigure,
                                        color=RULE, linewidth=0))
                fig.add_artist(Rectangle((x, .302), width*candidate["utility"], .019,
                                        transform=fig.transFigure, color=GRAY, linewidth=0))
                chosen = ", ".join(name for _, name in marks) or "Not selected"
                fig.text(x, .261, chosen, fontsize=8, color=marks[0][0] if marks else GRAY)
            score = {c["id"]: c["utility"] for c in cs}
            random = float(np.mean(list(score.values())))
            oracle = max(score.values())
            lines = [f"Random expected alignment: {random:.3f}    Annotation oracle: {oracle:.3f}"]
            for model, selected in selections.items():
                if selected:
                    choices = []
                    for k in range(3):
                        r = records[model].get((group["prompt_id"], k))
                        choices.append(labels[ids.index(r["selected_id"])] if r and r["parse_ok"] else "unresolved")
                    lines.append(f"{model.capitalize()}: regret {oracle-score[selected]:.3f}; "
                                 f"identities across orders: {' / '.join(choices)}")
                else:
                    lines.append(f"{model.capitalize()}: original-order choice unresolved")
            fig.text(.035, .203, "\n".join(lines), fontsize=8.2, va="top", linespacing=1.35)
            fig.text(.035, .045, "LOCAL REVIEW ONLY: source-image reuse permission not established.",
                     fontsize=8, color=ORANGE, weight="bold")
            fig.text(.035, .015, "Source: CulturalFrames. First main-manifest set per pool size; not a representative outcome claim.",
                     fontsize=7, color=GRAY)
            name = f"candidate_example_{len(cs)}"
            for suffix in ["pdf", "svg", "png"]:
                fig.savefig(dest / f"{name}.{suffix}", dpi=250)
            plt.close(fig)
            manifest.append({"prompt_id": group["prompt_id"], "candidate_ids": ids,
                             "selected_ids": selections, "pool_size": len(cs),
                             "selection_rule": "first main-manifest set for each candidate count"})
    save_json(dest / "example_manifest.json", manifest)


if __name__ == "__main__":
    main()
