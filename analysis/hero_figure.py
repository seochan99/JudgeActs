"""Figure 1 (recorded example + audit pipeline) and the presentation-slot figure.

The example is chosen by a fixed rule, never by inspecting images: among main-split
three-candidate sets where Qwen picks presentation slot A in every order and each order
yields a different image, take the set with the widest alignment range (manifest order
breaks ties). Images are CulturalFrames outputs, reproduced with citation in the caption.
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

from src.common import ROOT, load_jsonl, save_json
from .empirical_figures import STYLE, INK, TEAL, ORANGE, GRAY, RULE, WIDTH, _save

PALE = "#EAF3F5"
DEST = ROOT / "paper" / "figures"


def select_example():
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    runs = {(r["prompt_id"], r["permutation"]): r for r in load_jsonl(ROOT / "runs/qwen_main.jsonl")}
    best = None
    for g in groups:
        rs = [runs.get((g["prompt_id"], k)) for k in range(3)]
        if len(g["candidates"]) != 3 or not all(r and r["parse_ok"] for r in rs):
            continue
        slots = [next(k for k, v in r["mapping"].items() if v == r["selected_id"]) for r in rs]
        if slots != ["A", "A", "A"] or len({r["selected_id"] for r in rs}) != 3:
            continue
        spread = float(np.ptp([c["utility"] for c in g["candidates"]]))
        if best is None or spread > best[0] + 1e-12:
            best = (spread, g, rs)
    if best is None:
        raise ValueError("No set satisfies the example rule")
    return best[1], best[2]


def hero():
    group, runs = select_example()
    position = json.loads((ROOT / "results/main/position.json").read_text())
    summary = {r["policy"]: r for r in json.loads((ROOT / "results/main/summary.json").read_text())}
    cand = {c["id"]: c for c in group["candidates"]}
    images = {}
    for cid, c in cand.items():
        with Image.open(ROOT / c["image"]) as im:
            im = im.convert("RGB")
            side = min(im.size)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            images[cid] = np.asarray(im.crop((left, top, left + side, top + side)).resize((360, 360)))
    W, H = 7.0, 3.55
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    def text(x, y, s, size=8, color=INK, weight="normal", ha="left", va="center", style="normal"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                fontstyle=style, linespacing=1.25, zorder=6)

    def box(x, y, w, h, edge=RULE, face="white", lw=.8, pad=.04, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=.07",
                                    ec=edge, fc=face, lw=lw, ls=ls))

    def arrow(p, q, color=INK, lw=.9, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color,
                                     lw=lw, ls=ls, shrinkA=0, shrinkB=0))

    # Panel a: one recorded request under three candidate orders.
    text(.08, 3.37, "a", 10, weight="bold")
    text(.28, 3.37, "One request, three presentation orders", 9, weight="bold")
    quote = "\n".join(textwrap.wrap(f"\u201c{group['prompt']}\u201d", 60))
    text(.28, 3.06, quote, 7.2, color=GRAY, style="italic")
    size, gap, x0 = .66, .07, .62
    text(x0 + 1.5 * size + gap, 2.8, "Candidates in presented order", 6.8, color=GRAY, ha="center")
    text(3.28, 2.8, "Pick\u2019s human rating", 6.8, color=GRAY, ha="center")
    rows = [2.05, 1.18, .31]
    for k, (r, y) in enumerate(zip(runs, rows)):
        text(.08, y + size / 2, f"Order {k + 1}", 7.2, color=GRAY)
        for j, slot in enumerate("ABC"):
            cid = r["mapping"][slot]
            x = x0 + j * (size + gap)
            ax.imshow(images[cid], extent=(x, x + size, y, y + size), zorder=2)
            chosen = cid == r["selected_id"]
            ax.add_patch(Rectangle((x, y), size, size, fill=False, zorder=3,
                                   ec=TEAL if chosen else "white", lw=2.2 if chosen else .6))
            ax.add_patch(FancyBboxPatch((x + .035, y + size - .165), .13, .13,
                                        boxstyle="round,pad=0,rounding_size=.03", zorder=4,
                                        fc=TEAL if chosen else "white", ec="none", alpha=.95))
            text(x + .1, y + size - .1, slot, 6.5, color="white" if chosen else INK,
                 weight="bold", ha="center")
        pick = cand[r["selected_id"]]["utility"]
        best = max(c["utility"] for c in cand.values())
        bx, bw = 2.83, .9

        ax.add_patch(Rectangle((bx, y + .25), bw, .13, fc=RULE, ec="none"))
        ax.add_patch(Rectangle((bx, y + .25), bw * pick, .13, fc=TEAL, ec="none"))
        ax.plot([bx + bw * best] * 2, [y + .21, y + .42], color=INK, lw=.8)
        text(bx, y + .52, f"{pick:.2f}", 8.5, color=TEAL, weight="bold")
        text(bx + bw, y + .52, f"regret {best - pick:.2f}", 6.8, color=GRAY, ha="right")
    text(2.83, .13, "Tick: best available rating", 6.2, color=GRAY)

    # Panel b: the audit pipeline.
    px = 4.12
    ax.plot([px - .12, px - .12], [.12, 3.3], color=RULE, lw=.7)
    text(px, 3.37, "b", 10, weight="bold")
    text(px + .2, 3.37, "Selection and retrospective audit", 9, weight="bold")
    cx, bw = px + .1, 2.62
    box(cx, 2.63, bw, .36)
    text(cx + .1, 2.88, "Prompt + 3–4 candidate images", 7.6, weight="bold")
    text(cx + .1, 2.7, "no generator, country, or rating fields", 6.6, color=GRAY)
    arrow((cx + bw / 2, 2.58), (cx + bw / 2, 2.43))
    grad = LinearSegmentedColormap.from_list("judge", ["#DFF0F3", "#F5FAFB"])
    ax.imshow(np.linspace(0, 1, 64).reshape(1, -1), cmap=grad, aspect="auto",
              extent=(cx, cx + bw, 1.93, 2.37), zorder=0)
    box(cx, 1.93, bw, .44, edge=TEAL, face="none", lw=1.1)
    text(cx + .1, 2.24, "VLM judge × 3 cyclic orders", 7.8, weight="bold", color=TEAL)
    text(cx + .1, 2.05, "Qwen3-VL-4B (primary) · SmolVLM2-2.2B", 6.6, color=INK)
    arrow((cx + bw / 2, 1.88), (cx + bw / 2, 1.73))
    box(cx, 1.34, 1.52, .33)
    text(cx + .1, 1.58, "Agreement gate", 7.6, weight="bold")
    text(cx + .1, 1.42, "3/3 orders, or both judges", 6.4, color=GRAY)
    arrow((cx + 1.6, 1.5), (cx + 1.84, 1.5), ORANGE)
    text(cx + 1.9, 1.58, "Abstain", 7.6, color=ORANGE, weight="bold")
    text(cx + 1.9, 1.42, "no fallback", 6.4, color=GRAY)
    arrow((cx + .76, 1.29), (cx + .76, 1.14), TEAL)
    text(cx + .84, 1.21, "act", 6.4, color=TEAL)
    box(cx, .2, bw, .88, edge=INK, lw=.9)
    text(cx + .1, .96, "Audit with released human ratings", 7.6, weight="bold")
    text(cx + .1, .8, "joined by image ID after selection", 6.4, color=GRAY)
    q, u = summary["Qwen"], summary["Qwen unanimous"]
    slot_a = position["qwen"]["slots"]["A"]
    stats = [("regret vs. pool best", f"{q['regret']:.3f}"),
             ("gain vs. exact random", f"+{q['gain']:.3f}"),
             ("regret if 3/3 orders agree", f"{u['regret']:.3f}"),
             ("slot A chosen (uniform)", f"{100 * slot_a['rate']:.0f}% ({100 * slot_a['uniform_rate']:.0f}%)")]
    for i, (name, value) in enumerate(stats):
        y = .63 - i * .13
        text(cx + .1, y, name, 6.7)
        text(cx + bw - .1, y, value, 7, weight="bold", color=TEAL, ha="right")
    fig.savefig(DEST / "hero.pdf")
    fig.savefig(DEST / "hero.png", dpi=300)
    plt.close(fig)
    save_json(ROOT / "results/main/hero_example.json",
              {"prompt_id": group["prompt_id"], "prompt": group["prompt"],
               "rule": "3-candidate main set; Qwen chooses slot A in all 3 orders with 3 distinct images; widest alignment range",
               "orders": [{"mapping": r["mapping"], "selected_id": r["selected_id"]} for r in runs]})


def slot_preference():
    position = json.loads((ROOT / "results/main/position.json").read_text())
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 2.35))
        ax = fig.add_axes([.17, .2, .78, .56])
        slots = ["A", "B", "C", "D"]
        colors = {"qwen": TEAL, "smol": ORANGE}
        for j, model in enumerate(["qwen", "smol"]):
            data = position[model]["slots"]
            xs = np.arange(4) + (j - .5) * .36
            ax.bar(xs, [data[s]["rate"] for s in slots], width=.34, color=colors[model],
                   label=f"{'Qwen' if model == 'qwen' else 'Smol'} ({position[model]['calls']} calls)")
        uniform = position["qwen"]["slots"]
        for i, s in enumerate(slots):
            ax.plot([i - .4, i + .4], [uniform[s]["uniform_rate"]] * 2, color=INK, lw=1, ls=(0, (2, 1.5)))
        ax.set_xticks(range(4), [f"Slot {s}" for s in slots])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_ylim(0, .62)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=RULE, linewidth=.5)
        ax.set_ylabel("Share of choices")
        fig.text(.045, .92, "Presentation-slot preference", fontsize=10, weight="bold")
        ax.legend(loc="upper right", frameon=False, fontsize=7.4, handlelength=1, borderaxespad=0, ncol=2)
        fig.text(.045, .045, "Dashed: uniform rate given pool sizes (slot D only in 4-image sets)",
                 fontsize=6.8, color=GRAY)
        _save(fig, DEST, "slot_preference")


def main():
    with plt.rc_context({**STYLE, "savefig.bbox": None}):
        hero()
    slot_preference()
    with plt.rc_context({"savefig.bbox": None}):
        order_example()
        hero_column()


if __name__ == "__main__":
    main()


def order_example():
    """Column-width version of panel (a): the same recorded request under three orders."""
    from . import style
    style.apply()
    group, runs = select_example()
    cand = {c["id"]: c for c in group["candidates"]}
    images = {}
    for cid, c in cand.items():
        with Image.open(ROOT / c["image"]) as im:
            im = im.convert("RGB")
            side = min(im.size)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            images[cid] = np.asarray(im.crop((left, top, left + side, top + side)).resize((300, 300)))
    W, H = 3.33, 2.72
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    def text(x, y, s, size=7, color=style.INK, weight="normal", ha="left", style_="normal"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="center",
                fontstyle=style_, linespacing=1.2, zorder=6)

    quote = "\n".join(textwrap.wrap(f"“{group['prompt']}”", 58))
    text(.05, H - .17, quote, 6.8, color=style.MUTED, style_="italic")
    size, gap, x0 = .56, .045, .36
    best = max(c["utility"] for c in cand.values())
    text(x0 + 1.5 * size + gap, H - .47, "Candidates in presented order", 6.3, color=style.MUTED, ha="center")
    text(2.78, H - .47, "Human rating of pick", 6.3, color=style.MUTED, ha="center")
    rows = [H - 1.1, H - 1.73, H - 2.36]
    for k, (r, y) in enumerate(zip(runs, rows)):
        ax.text(.14, y + size / 2, f"Order {k + 1}", fontsize=6.5, color=style.MUTED, rotation=90,
                ha="center", va="center")
        for j, slot in enumerate(sorted(r["mapping"])):
            cid = r["mapping"][slot]
            x = x0 + j * (size + gap)
            ax.imshow(images[cid], extent=(x, x + size, y, y + size), zorder=2)
            chosen = cid == r["selected_id"]
            ax.add_patch(Rectangle((x, y), size, size, fill=False, zorder=3,
                                   ec=style.QWEN if chosen else "white", lw=2.0 if chosen else .5))
            ax.add_patch(FancyBboxPatch((x + .03, y + size - .15), .12, .12,
                                        boxstyle="round,pad=0,rounding_size=.03", zorder=4,
                                        fc=style.QWEN if chosen else "white", ec="none", alpha=.95))
            text(x + .09, y + size - .09, slot, 6, color="white" if chosen else style.INK,
                 weight="bold", ha="center")
        pick = cand[r["selected_id"]]["utility"]
        bx, bw = 2.38, .82
        color = style.GOOD if pick >= best - 1e-12 else style.BAD
        ax.add_patch(Rectangle((bx, y + .17), bw, .11, fc=style.GRID, ec="none"))
        ax.add_patch(Rectangle((bx, y + .17), bw * pick, .11, fc=color, ec="none"))
        ax.plot([bx + bw * best] * 2, [y + .13, y + .32], color=style.INK, lw=.8)
        text(bx, y + .43, f"{pick:.2f}", 8, color=color, weight="bold")
        text(bx + bw, y + .43, f"regret {best - pick:.2f}", 6, color=style.MUTED, ha="right")
    text(2.38, .2, "tick = best available rating", 5.8, color=style.MUTED)
    fig.savefig(DEST / "order_example.pdf")
    fig.savefig(DEST / "order_example.png", dpi=300)
    plt.close(fig)


def hero_column():
    """Page-1 figure in the style of a qualitative grid: serif labels, square frames, status icons.

    Rows are the three cyclic orders; columns are presentation slots. Ratings are printed on the images.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from . import style
    style.apply()
    serif = {"family": "Times New Roman"}
    group, runs = select_example()
    cand = {c["id"]: c for c in group["candidates"]}
    images = {}
    for cid, c in cand.items():
        with Image.open(ROOT / c["image"]) as im:
            im = im.convert("RGB")
            side = min(im.size)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            images[cid] = np.asarray(im.crop((left, top, left + side, top + side)).resize((640, 640), Image.LANCZOS))
    best = max(c["utility"] for c in cand.values())
    W, H = 3.33, 3.5
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    def icon(cx, cy, r, good):
        color = style.GOOD if good else style.BAD
        ax.add_patch(Circle((cx, cy), r, fc=color, ec="white", lw=.8, zorder=8))
        if good:
            ax.add_line(Line2D(np.array([-.48, -.12, .5]) * r + cx, np.array([.02, -.36, .38]) * r + cy,
                               color="white", lw=1.3, solid_capstyle="round", zorder=9))
        else:
            d = .38 * r
            for sgn in (1, -1):
                ax.add_line(Line2D([cx - d, cx + d], [cy - sgn * d, cy + sgn * d], color="white", lw=1.3,
                                   solid_capstyle="round", zorder=9))

    size, gap, x0 = .9, .045, .33
    top = H - .42
    ax.text(W / 2 + .1, H - .12, f"\u201c{group['prompt']}\u201d", fontsize=6.6, ha="center", va="center",
            style="italic", **serif)
    for j, slot in enumerate("ABC"):
        ax.text(x0 + j * (size + gap) + size / 2, top + .1, f"Slot {slot}", fontsize=8.5, weight="bold",
                ha="center", va="center", **serif)
    for k, r in enumerate(runs):
        y = top - (k + 1) * size - k * gap
        ax.text(.15, y + size / 2, f"Order {k + 1}", fontsize=8.5, weight="bold", rotation=90,
                ha="center", va="center", **serif)
        for j, slot in enumerate(sorted(r["mapping"])):
            cid = r["mapping"][slot]
            x = x0 + j * (size + gap)
            ax.imshow(images[cid], extent=(x, x + size, y, y + size), zorder=2, interpolation="lanczos")
            u = cand[cid]["utility"]
            picked = cid == r["selected_id"]
            ax.text(x + .05, y + .07, f"{u:.2f}", fontsize=7, color="white", weight="bold", va="bottom", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.18", fc=(0, 0, 0, .55), ec="none"))
            if picked:
                good = u >= best - 1e-12
                ax.add_patch(Rectangle((x - .018, y - .018), size + .036, size + .036, fill=False, zorder=6,
                                       ec=style.GOOD if good else style.BAD, lw=2.2, joinstyle="miter"))
                icon(x + size - .11, y + size - .11, .075, good)
    ly = .13
    icon(.36, ly, .055, True)
    ax.text(.45, ly, "judge's pick is best-rated", fontsize=6.8, va="center", **serif)
    icon(1.78, ly, .055, False)
    ax.text(1.87, ly, "judge's pick is not best-rated", fontsize=6.8, va="center", **serif)
    fig.savefig(DEST / "hero_column.pdf", dpi=400)
    fig.savefig(DEST / "hero_column.png", dpi=300)
    plt.close(fig)
