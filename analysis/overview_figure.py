"""Publication-sized, editable schematic of selection authority and outcome audit.

All symbols are schematic. This figure uses no source images or empirical scores.
The independent reference lane deliberately has no connection into the judge.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from src.common import ROOT

INK = "#263238"
TEAL = "#147D92"
ORANGE = "#B87540"
GRAY = "#7A8087"
PALE = "#EAF3F5"
RULE = "#D8DDE0"
DEST = ROOT / "paper" / "figures"


def overview():
    """Render Figure 1; coordinates are physical inches for predictable typography."""
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": None,
    }):
        fig = plt.figure(figsize=(7.1, 3.35), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 7.1), ylim=(0, 3.35))
        ax.axis("off")

        def text(x, y, label, size=9, color=INK, weight="normal", ha="left", va="center"):
            return ax.text(x, y, label, fontsize=size, color=color,
                           fontweight=weight, ha=ha, va=va, linespacing=1.32)

        def rect(x, y, w, h, color=RULE, fill="white", lw=.8):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                                   edgecolor=color, linewidth=lw))

        def arrow(points, color=INK, dashed=False):
            vertices = list(points)
            path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices)-1))
            ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
                mutation_scale=8, linewidth=.9, color=color,
                linestyle=(0, (3.4, 2.4)) if dashed else "-",
                joinstyle="miter", capstyle="butt"))

        def line(points, color=GRAY, lw=.8):
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color=color, linewidth=lw, solid_capstyle="butt",
                    solid_joinstyle="miter")

        def document_glyph(x, y):
            """An original folded sheet, with rules encoding text rather than art."""
            w, h, fold = .18, .24, .055
            vertices = [(x, y), (x+w, y), (x+w, y+h-fold),
                        (x+w-fold, y+h), (x, y+h), (x, y)]
            path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.LINETO] * 5)
            ax.add_patch(PathPatch(path, facecolor="white", edgecolor=GRAY, linewidth=.8))
            line([(x+w-fold, y+h), (x+w-fold, y+h-fold), (x+w, y+h-fold)])
            for offset in (.065, .105, .145):
                line([(x+.035, y+offset), (x+w-.035, y+offset)], lw=.65)

        def decision_glyph(x, y):
            """One selected branch and one unselected branch; no accuracy checkmark."""
            line([(x, y), (x+.07, y)], color=TEAL)
            line([(x+.07, y), (x+.17, y+.075)], color=TEAL)
            line([(x+.07, y), (x+.17, y-.075)], color=GRAY)
            ax.add_patch(Circle((x+.19, y+.075), .023, facecolor=TEAL, edgecolor=TEAL, lw=.6))
            ax.add_patch(Circle((x+.19, y-.075), .023, facecolor=PALE, edgecolor=GRAY, lw=.65))

        def image_glyph(x, y):
            """A symbolic image canvas, not a depicted experimental candidate."""
            rect(x, y, .24, .20, color=TEAL, fill="white", lw=.8)
            line([(x+.025, y+.035), (x+.085, y+.105),
                  (x+.14, y+.050), (x+.185, y+.085), (x+.22, y+.035)], color=TEAL, lw=.7)
            ax.add_patch(Circle((x+.18, y+.145), .020, facecolor="white", edgecolor=TEAL, lw=.65))

        # Section labels describe authority, rather than decorating individual boxes.
        text(.06, 3.18, "a", size=10, weight="bold")
        text(.25, 3.18, "Selection policy", size=10, weight="bold")
        text(6.93, 3.18, "Prompt + images only", color=GRAY, ha="right")

        centers = [.61, 2.22, 4.15, 6.03]
        for x, title in zip(centers, ["Prompt", "Candidate set", "Judge", "Returned image"]):
            text(x, 2.92, title, color=GRAY, ha="center")

        # Each node shares the same horizontal midline and visual anchor boundaries.
        rect(.06, 2.02, 1.10, .70)
        document_glyph(.39, 2.32)
        text(.73, 2.43, "$p$", size=13, ha="center")
        text(.61, 2.18, "Original text", ha="center")

        # Small typed image cards convey a candidate pool without fabricated imagery.
        for label, x, y in [("A", 1.46, 2.40), ("B", 2.28, 2.40),
                            ("C", 1.46, 1.96), ("D", 2.28, 1.96)]:
            rect(x, y, .70, .36, color="#B4BEC4")
            text(x+.13, y+.18, label, weight="bold", color=TEAL)
            text(x+.47, y+.18, "$x_{p" + label + "}$", ha="center")
        text(2.22, 1.78, "3–4 candidates", color=GRAY, ha="center")

        rect(3.53, 2.02, 1.24, .70, color=TEAL, fill=PALE, lw=1)
        decision_glyph(3.67, 2.48)
        text(3.96, 2.48, "VLM judge", weight="bold")
        text(4.15, 2.21, "$\\pi(p)$: choose one", ha="center")

        rect(5.41, 2.02, 1.24, .70, color=TEAL, lw=1)
        image_glyph(5.60, 2.36)
        text(6.19, 2.47, "$x_{p,\\pi(p)}$", size=13, color=TEAL, ha="center")
        text(6.03, 2.18, "Selected identity", ha="center")

        mid = 2.37
        arrow([(1.20, mid), (1.41, mid)])
        arrow([(3.02, mid), (3.48, mid)])
        arrow([(4.82, mid), (5.36, mid)], color=TEAL)

        # Optional rejection is an orthogonal branch, never an invented human fallback.
        arrow([(4.15, 1.98), (4.15, 1.52), (5.36, 1.52)], color=ORANGE)
        text(5.42, 1.60, "Abstain", color=ORANGE, weight="bold")
        text(5.42, 1.38, "Decision unresolved", color=GRAY)
        text(4.19, 1.77, "if gate rejects", color=GRAY)

        # Independent public human reference. No reference data enters the judge.
        ax.plot([.06, 6.65], [1.13, 1.13], color=RULE, lw=.7)
        text(.06, .96, "b", size=10, weight="bold")
        text(.25, .96, "Retrospective audit", size=10, weight="bold")
        text(.06, .63, "Released human annotations", weight="bold")
        text(.06, .38, "$h_{pA},\\ h_{pB},\\ h_{pC},\\ h_{pD}$", size=10)
        text(.06, .15, "Image-ID match; withheld from judge", color=GRAY)

        # A quiet rectangular frame gives both evidence connectors an unambiguous
        # endpoint. It has no fill or shadow and is secondary to the action nodes.
        rect(3.09, .10, 3.56, .79, color=RULE, lw=.7)
        arrow([(2.54, .51), (3.05, .51)], color=GRAY, dashed=True)
        text(3.23, .67, "$R_p = \\max_i\\,h_{pi} - h_{p,\\pi(p)}$", size=11)
        text(3.23, .31, "$\\mathrm{HSR} = \\Pr\\!\\left[h_{p,\\pi(p)} < \\frac{1}{m_p}\\sum_i h_{pi}\\right]$", size=10.5)

        # Route the observed selected identity around the outer edge; its path never
        # crosses a node, an abstention label, or an empirical formula.
        arrow([(6.69, mid), (6.98, mid), (6.98, .64), (6.69, .64)],
              color=GRAY, dashed=True)

        DEST.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(DEST / f"overview.{suffix}", dpi=350, facecolor="white", bbox_inches=None)
        plt.close(fig)


def main():
    overview()


if __name__ == "__main__":
    main()
