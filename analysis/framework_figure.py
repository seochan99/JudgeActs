"""Framework / protocol overview figure: (a) frozen pools -> (b) selection policy -> (c) audit.

Counts (prompts, countries, categories, generators, candidates per prompt) are read from
data/derived/main.jsonl; judge names, order count and gate names are protocol constants.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from analysis.style import *  # noqa: F401,F403
from analysis.style import (AGREE, BAD, GOOD, INK, MUTED, PANEL_BLUE, PANEL_GREEN,
                            PANEL_PURPLE, QWEN, SMOL, apply)
from src.common import ROOT

W, H = 7.0, 2.72
DEST = ROOT / "paper" / "figures"
DATA = ROOT / "data" / "derived" / "main.jsonl"
ABSTAIN = "#E08A3C"
JUDGES = [("Qwen3-VL-4B", QWEN), ("SmolVLM2-2.2B", SMOL)]
GATES = ["3/3 orders", "≥2/3 orders", "both judges"]
N_ORDERS = 3


def load_facts():
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    gens = []
    for r in rows:
        for c in r["candidates"]:
            if c["source_model"] not in gens:
                gens.append(c["source_model"])
    sizes = [len(r["candidates"]) for r in rows]
    return {
        "n": len(rows),
        "countries": sorted({r["country"].replace("_", " ") for r in rows}),
        "categories": sorted({r["category"] for r in rows}),
        "generators": gens,
        "kmin": min(sizes), "kmax": max(sizes),
    }


def pretty_category(c):
    s = c.replace("-", " ")
    return s[0].upper() + s[1:]


class Canvas:
    """Axes in physical inches with text measurement."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W, H), dpi=300)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()

    def tw(self, s, size, weight="bold"):
        t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(self.r).width / self.fig.dpi
        t.remove()
        return w

    def text(self, x, y, s, size=6.5, weight="bold", color=INK, ha="center", va="center", **kw):
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                            ha=ha, va=va, **{"zorder": 6, **kw})

    def panel(self, x0, y0, x1, y1, colors, title):
        fill, edge = colors
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle="round,pad=0,rounding_size=0.09",
                                         fc=fill, ec=edge, lw=0.9, zorder=1))
        self.text(x0 + 0.09, y1 - 0.13, title, size=7.6, ha="left")

    def pill(self, cx, cy, s, size=6.2, color=INK, fc="white", h=0.15, pad=0.06, w=None,
             lw=0.8, weight="bold", tcolor=INK, ha="center"):
        w = w if w is not None else self.tw(s, size, weight) + 2 * pad
        x0 = cx - w / 2 if ha == "center" else cx
        self.ax.add_patch(FancyBboxPatch((x0, cy - h / 2), w, h,
                                         boxstyle=f"round,pad=0,rounding_size={h / 2:.4f}",
                                         fc=fc, ec=color, lw=lw, zorder=4))
        self.text(x0 + w / 2, cy, s, size=size, weight=weight, color=tcolor)
        return x0, x0 + w

    def flow(self, x0, x1, ytop, items, size=6.0, color=INK, h=0.14, gap=0.035, vgap=0.04,
             fc="white"):
        """Wrap pills left-to-right; return y of last row bottom."""
        x, y = x0, ytop - h / 2
        for s in items:
            w = self.tw(s, size) + 0.1
            if x + w > x1 + 1e-6 and x > x0:
                x, y = x0, y - h - vgap
            self.pill(x, y, s, size=size, color=color, h=h, pad=0.05, fc=fc, ha="left")
            x += w + gap
        return y - h / 2

    def step(self, x, y, n, color=INK):
        self.ax.add_patch(Circle((x, y), 0.058, fc="white", ec=color, lw=0.7, zorder=7))
        self.text(x, y - 0.003, str(n), size=6, color=color, zorder=8)

    def arrow(self, p0, p1, color=INK, lw=1.0, ms=7, ls="-", style="-|>", zorder=5, **kw):
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                          color=color, lw=lw, ls=ls, zorder=zorder,
                                          shrinkA=0, shrinkB=0, **kw))

    def dotted(self, p0, p1, color=MUTED):
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ls=(0, (1, 1.3)), lw=0.8,
                     color=color, zorder=3)


def draw():
    apply()
    f = load_facts()
    cv = Canvas()
    ax = cv.ax
    ytop, ybot = 2.68, 0.30
    fh, fv = 0.135, 0.032          # flow pill height / row gap

    # ---------------- (a) Frozen candidate pools ----------------
    ax0, ax1 = 0.03, 2.53
    cv.panel(ax0, ybot, ax1, ytop, PANEL_GREEN, "(a) Frozen candidate pools")
    green = PANEL_GREEN[1]
    gdark = "#3E8E50"
    sx, sy, sw = 0.45, 1.66, 0.74
    ax.add_patch(FancyBboxPatch((sx - sw / 2, sy - 0.17), sw, 0.34,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=gdark, lw=1.0, zorder=4))
    cv.text(sx, sy + 0.06, "CulturalFrames", size=6.6)
    cv.text(sx, sy - 0.08, f"{f['n']} prompts", size=6.2, weight="normal", color=MUTED)
    cv.text(sx, sy + 0.29, "$p$", size=9, weight="normal")
    # Candidate stack C_p.
    cy = 1.08
    for k in range(3):
        ax.add_patch(Rectangle((sx - 0.19 + 0.07 * k, cy - 0.02 - 0.05 * k), 0.2, 0.15,
                               fc="#D6EEDB" if k == 2 else "white", ec=gdark, lw=0.7,
                               zorder=4 + k * 0.1))
    cv.text(sx + 0.19, cy + 0.02, "$C_p$", size=9, weight="normal", ha="left")
    kr = f"{f['kmin']}\u2013{f['kmax']}" if f["kmin"] != f["kmax"] else f"{f['kmin']}"
    cv.text(sx, cy - 0.31, f"{kr} candidates\nper prompt", size=6.0, weight="normal",
            linespacing=1.1)
    cv.arrow((sx, sy - 0.17), (sx, cy + 0.15), color=gdark, lw=0.8, ms=6)
    cv.pill(sx, 0.46, "ratings withheld", size=6.0, color=MUTED, fc="#F4F5F6",
            tcolor=MUTED, h=0.15)

    gx0, gx1 = 0.99, ax1 - 0.06
    groups = [
        (f"{len(f['countries'])} countries", f["countries"]),
        (f"{len(f['categories'])} categories", [pretty_category(c) for c in f["categories"]]),
        (f"{len(f['generators'])} generators", f["generators"]),
    ]
    y = 2.33
    for title, items in groups:
        cv.text(gx0, y, title, size=6.2, ha="left", color=gdark)
        top = y - 0.085
        bottom = cv.flow(gx0, gx1, top, items, color=green, h=0.145, vgap=0.035)
        cv.dotted((sx + sw / 2, sy), (gx0 - 0.045, (top + bottom) / 2))
        ax.plot([gx0 - 0.045] * 2, [top - 0.01, bottom + 0.01], color=MUTED, lw=0.6, zorder=3)
        y = bottom - 0.17

    # ---------------- (b) Selection policy ----------------
    bx0, bx1 = 2.71, 4.71
    cv.panel(bx0, ybot, bx1, ytop, PANEL_BLUE, "(b) Selection policy")
    blue_d = "#3F6FA8"
    sxc = bx0 + 0.13
    # (1) cyclic orders.
    y1 = 2.27
    cv.step(sxc, y1, 1, blue_d)
    cv.text(bx0 + 0.22, y1, "cyclic orders", size=6.2, ha="left", color=blue_d)
    letters, fills = ["A", "B", "C"], ["#FFFFFF", "#D5E4F6", "#AFCBEE"]
    sq, sgap, ogap = 0.105, 0.01, 0.06
    strip_w = 3 * sq + 2 * sgap
    ox0 = bx1 - 0.08 - (N_ORDERS * strip_w + (N_ORDERS - 1) * ogap)
    for o in range(N_ORDERS):
        ox = ox0 + o * (strip_w + ogap)
        for j in range(3):
            li = (j + o) % 3
            ax.add_patch(FancyBboxPatch((ox + j * (sq + sgap), y1 - sq / 2), sq, sq,
                                        boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=fills[li], ec=blue_d, lw=0.6, zorder=4))
            cv.text(ox + j * (sq + sgap) + sq / 2, y1, letters[li], size=6)
    strips_mid = ox0 + (N_ORDERS * strip_w + (N_ORDERS - 1) * ogap) / 2
    # (2) judges.
    y2 = 1.84
    cv.step(sxc, y2, 2, blue_d)
    cv.text(bx0 + 0.22, y2 + 0.005, r"$\pi$", size=9, weight="normal", ha="left")
    jl, jr, jg = bx0 + 0.36, bx1 - 0.07, 0.07
    jw = (jr - jl - jg) / 2
    jx = [jl + jw / 2, jr - jw / 2]
    for (name, col), x in zip(JUDGES, jx):
        cv.pill(x, y2, name, size=6.2, color=col, w=jw, h=0.18, lw=1.1)
    for x in jx:
        cv.arrow((strips_mid, y1 - 0.075), (x, y2 + 0.1), color=blue_d, lw=0.8, ms=6)
    # (3) choice.
    bmid = (jl + jr) / 2
    y3 = 1.42
    cv.step(sxc, y3, 3, blue_d)
    cl, cr = cv.pill(bmid, y3, "one pick per order", size=6.2, color=blue_d, h=0.17)
    cv.text(cr + 0.05, y3, r"$\pi(p)$", size=8.5, weight="normal", ha="left")
    for x in jx:
        cv.arrow((x, y2 - 0.095), (bmid, y3 + 0.095), color=blue_d, lw=0.8, ms=6)
    # (4) agreement gates.
    y4 = 0.99
    cv.step(sxc, y4, 4, AGREE)
    gws = [cv.tw(g, 6.0) + 0.1 for g in GATES]
    ggap = 0.045
    total = sum(gws) + ggap * (len(GATES) - 1)
    gx = (bx0 + 0.25 + bx1 - 0.05) / 2 - total / 2
    gmid = gx + total / 2
    for g, w in zip(GATES, gws):
        cv.pill(gx, y4, g, size=6.0, color=AGREE, w=w, h=0.16, ha="left", fc="#F6F2FC")
        gx += w + ggap
    cv.arrow((bmid, y3 - 0.085), (gmid, y4 + 0.085), color=blue_d, lw=0.8, ms=6)
    cv.text(max(bmid, gmid) + 0.05, (y3 + y4) / 2 - 0.005, "agreement gates", size=6.0, weight="normal",
            color=AGREE, ha="left")
    # act / abstain.
    y5 = 0.56
    actx, absx = gmid - 0.42, gmid + 0.42
    cv.pill(actx, y5, "act", size=6.4, color=GOOD, h=0.17, w=0.5, lw=1.1, fc="#EAF7EF")
    cv.pill(absx, y5, "abstain", size=6.4, color=ABSTAIN, h=0.17, w=0.62, lw=1.1,
            fc="#FDF1E6", tcolor="#9A5A22")
    cv.arrow((gmid, y4 - 0.08), (actx + 0.08, y5 + 0.09), color=GOOD, lw=0.8, ms=6)
    cv.arrow((gmid, y4 - 0.08), (absx - 0.08, y5 + 0.09), color=ABSTAIN, lw=0.8, ms=6)
    cv.text(actx - 0.02, (y4 + y5) / 2 + 0.01, "agree", size=6, weight="normal",
            color=GOOD, ha="right")
    cv.text(absx + 0.02, (y4 + y5) / 2 + 0.01, "else", size=6, weight="normal",
            color="#9A5A22", ha="left")

    # ---------------- (c) Retrospective audit ----------------
    cx0, cx1 = 4.89, 6.97
    cv.panel(cx0, ybot, cx1, ytop, PANEL_PURPLE, "(c) Retrospective audit")
    pdark = "#6B4FA3"
    ppale = PANEL_PURPLE[1]
    scx = cx0 + 0.13
    y6 = 2.27
    cv.step(scx, y6, 5, pdark)
    _, x1p = cv.pill(cx0 + 0.23, y6, "join human ratings by image ID", size=6.0,
                     color=pdark, h=0.17, ha="left")
    cv.text(x1p + 0.04, y6, r"$h_{pi}$", size=8.5, weight="normal", ha="left")
    ogroups = [
        ("Outcomes", ["Primary: selection regret", "Paired gain vs. exact random",
                      "Below-mean rate (BMR)"], pdark),
        ("Cultural errors", ["Stereotype", "Missing explicit", "Missing implicit"], BAD),
        ("Diagnostics", ["Country groups", "Presentation-slot bias",
                         "Matched-subset baselines for gates"], ppale),
    ]
    ox0, ox1 = cx0 + 0.25, cx1 - 0.05
    y = 1.99
    cv.step(scx, y, 6, pdark)
    spine_top, mids = y - 0.065, []
    for title, items, col in ogroups:
        cv.text(ox0, y, title, size=6.2, ha="left", color=pdark)
        top = y - 0.085
        bottom = cv.flow(ox0, ox1, top, items, color=col, h=fh, vgap=fv)
        mids.append((top + bottom) / 2)
        cv.dotted((scx, mids[-1]), (ox0 - 0.045, mids[-1]))
        ax.plot([ox0 - 0.045] * 2, [top - 0.01, bottom + 0.01], color=MUTED, lw=0.6, zorder=3)
        y = bottom - 0.13
    ax.plot([scx, scx], [spine_top, mids[-1]], ls=(0, (1, 1.3)), lw=0.8, color=MUTED, zorder=3)
    assert y + 0.13 > ybot + 0.03, f"(c) overflows: {y}"

    # ---------------- inter-panel arrows ----------------
    amid = 1.66
    cv.arrow((ax1 + 0.02, amid), (bx0 - 0.02, amid), lw=1.6, ms=9)
    cv.arrow((bx1 + 0.02, amid), (cx0 - 0.02, amid), lw=1.6, ms=9)
    # Ratings bypass the judge (dashed, beneath panel b).
    yb, ex = 0.14, cx0 + 0.10
    dash = (0, (3, 2))
    ax.plot([sx, sx], [0.46 - 0.075, yb], color=MUTED, lw=0.8, ls=dash, zorder=2)
    ax.plot([sx, ex], [yb, yb], color=MUTED, lw=0.8, ls=dash, zorder=2)
    cv.arrow((ex, yb), (ex, ybot + 0.005), color=MUTED, lw=0.8, ms=6, ls=dash)
    cv.text((bx0 + bx1) / 2, yb, "  released ratings: withheld from judge, joined only in audit  ",
            size=6.0, weight="normal", color=MUTED, bbox=dict(fc="white", ec="none", pad=0.4))
    return cv.fig


def main():
    fig = draw()
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / "framework.pdf")
    fig.savefig(DEST / "framework.png", dpi=300)
    plt.close(fig)
    print("wrote", DEST / "framework.pdf", DEST / "framework.png", f"{W}x{H} in")


if __name__ == "__main__":
    main()
