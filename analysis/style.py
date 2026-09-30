"""Shared visual language for all manuscript figures (diagram, galleries, charts).

Semantic colors are fixed across figures: a policy always has the same color.
"""
import matplotlib

FONT = "Arial"

# Policies and outcomes.
RANDOM = "#9AA0A6"
QWEN = "#3B8ED0"
SMOL = "#E4572E"
ORACLE = "#2F3B45"
AGREE = "#8E6CC7"      # agreement gates / cross-model
GOOD = "#2EAD5B"       # positive gain, annotation-best pick
BAD = "#E04848"        # negative gain, below-mean pick
INK = "#1F2A33"
MUTED = "#6B7680"
GRID = "#E3E7EA"

# Pastel panels: (fill, edge).
PANEL_BLUE = ("#EAF2FC", "#7FA8D8")
PANEL_GREEN = ("#EAF6EC", "#6DBF7E")
PANEL_PURPLE = ("#F3EEFA", "#A48BD1")
PANEL_YELLOW = ("#FFF5CF", "#E5C55A")
PANEL_RED = ("#FDECEC", "#E58C8C")

POLICY_COLORS = {"Random": RANDOM, "Qwen": QWEN, "Smol": SMOL, "Oracle": ORACLE,
                 "Cross-model": AGREE}

RC = {
    "font.family": FONT, "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.fontsize": 7, "legend.frameon": True, "legend.edgecolor": GRID,
    "legend.fancybox": False, "legend.framealpha": 1,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
}


def apply():
    matplotlib.rcParams.update(RC)
