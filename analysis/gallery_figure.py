"""Qualitative image-grid figures: page-1 teaser and appendix gallery.

Both selections are deterministic rules over the frozen data and recorded judge
outputs; images are never inspected when choosing examples.

* Teaser: prompts with 3 candidates where Qwen's original-order pick
  (permutation 0) is below the pool mean and the pool holds a candidate at
  least 0.25 above the mean. Per country keep the largest regret (max utility -
  chosen utility; ties -> manifest order); then the 4 countries with the largest
  regret (ties -> country name). Prompt 7c51b057... is excluded (used elsewhere).
* Appendix gallery: per country (alphabetical), the first prompt in manifest
  order with 4 candidates (fallback 3). Unfiltered, representative sample.

Run: uv run python -m analysis.gallery_figure
"""
import json
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image

from src.common import ROOT, load_jsonl, save_json
from analysis.style import *  # noqa: F401,F403
from analysis.style import apply, QWEN, SMOL, GOOD, BAD, INK, MUTED, GRID

EXCLUDE_PREFIX = "7c51b057"
MIN_GAP = 0.25
N_TEASER = 4
FIG_DIR = ROOT / "paper/figures"
JSON_OUT = ROOT / "results/main/gallery_examples.json"
EPS = 1e-9


# ----------------------------------------------------------------------------- data
def load():
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    picks = {}
    for model in ["qwen", "smol"]:
        picks[model] = {r["prompt_id"]: r for r in load_jsonl(ROOT / f"runs/{model}_main.jsonl")
                        if r["permutation"] == 0}
    return groups, picks


def presented(group, rec):
    """Candidates in the original (permutation-0) presented order A, B, C(, D)."""
    by_id = {c["id"]: c for c in group["candidates"]}
    return [by_id[rec["mapping"][k]] for k in sorted(rec["mapping"])]


def pick(rec):
    return rec["selected_id"] if rec and rec.get("parse_ok") else None


def country_name(c):
    return c.replace("_", " ")


def select_teaser(groups, picks):
    per_country = {}
    for g in groups:
        if len(g["candidates"]) != 3 or g["prompt_id"].startswith(EXCLUDE_PREFIX):
            continue
        chosen = pick(picks["qwen"].get(g["prompt_id"]))
        if chosen is None:
            continue
        u = {c["id"]: c["utility"] for c in g["candidates"]}
        mean, top = float(np.mean(list(u.values()))), max(u.values())
        if not (u[chosen] < mean - EPS and top >= mean + MIN_GAP - EPS):
            continue
        regret = top - u[chosen]
        cur = per_country.get(g["country"])
        if cur is None or regret > cur["regret"] + EPS:  # ties keep manifest order
            per_country[g["country"]] = dict(group=g, regret=regret, mean=mean)
    ranked = sorted(per_country.items(), key=lambda kv: (-round(kv[1]["regret"], 9), kv[0]))
    return ranked[:N_TEASER], ranked


def select_gallery(groups):
    out = []
    for country in sorted({g["country"] for g in groups}):
        gs = [g for g in groups if g["country"] == country]
        g = next((g for g in gs if len(g["candidates"]) == 4), None) or \
            next(g for g in gs if len(g["candidates"]) == 3)
        out.append(g)
    return out


# ----------------------------------------------------------------------------- drawing
_THUMBS = {}


def thumb(path, px):
    key = (path, px)
    if key not in _THUMBS:
        with Image.open(ROOT / path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            _THUMBS[key] = np.asarray(im.resize((px, px), Image.LANCZOS))
    return _THUMBS[key]


class Canvas:
    """Figure addressed in inches from the top-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.tr = self.fig.dpi_scale_trans

    def y(self, top):
        return self.h - top

    def image(self, arr, x, top, s):
        ax = self.fig.add_axes([x / self.w, (self.h - top - s) / self.h, s / self.w, s / self.h])
        ax.imshow(arr, interpolation="none")  # embed native pixels in PDF
        ax.set_axis_off()
        return ax

    def frame(self, x, top, s, color, lw=2.0, pad=0.02, rad=0.0, z=5):
        self.fig.add_artist(FancyBboxPatch(
            (x - pad, self.y(top + s) - pad), s + 2 * pad, s + 2 * pad,
            boxstyle="square,pad=0", transform=self.tr, joinstyle="miter",
            fill=False, edgecolor=color, linewidth=lw, zorder=z))

    def text(self, x, top, s, **kw):
        kw.setdefault("color", INK)
        return self.fig.text(x, self.y(top), s, transform=self.tr, **kw)

    def icon(self, cx, cy_top, r, kind, color, letter=None):
        """Circular status icon: kind in {'check','cross','letter'}; centre in inches."""
        cy = self.y(cy_top)
        self.fig.add_artist(Circle((cx, cy), r, transform=self.tr, facecolor=color,
                                   edgecolor="white", linewidth=0.8, zorder=10))
        lw = max(r * 72 * 0.28, 0.9)
        if kind == "check":
            xs = np.array([-0.48, -0.12, 0.50]) * r + cx
            ys = np.array([0.02, -0.36, 0.38]) * r + cy
            self.fig.add_artist(Line2D(xs, ys, transform=self.tr, color="white", lw=lw,
                                       solid_capstyle="round", solid_joinstyle="round", zorder=11))
        elif kind == "cross":
            d = 0.40 * r
            for sx in (1, -1):
                self.fig.add_artist(Line2D([cx - d, cx + d], [cy - sx * d, cy + sx * d],
                                           transform=self.tr, color="white", lw=lw,
                                           solid_capstyle="round", zorder=11))
        elif kind == "letter":
            self.fig.text(cx, cy - 0.004, letter, transform=self.tr, ha="center", va="center",
                          color="white", fontsize=r * 72 * 1.15, weight="bold", zorder=11)

    def pill(self, x, top, s, fontsize, color="white", bg=(0, 0, 0, 0.62), ha="left", weight="bold"):
        return self.text(x, top, s, fontsize=fontsize, color=color, ha=ha, va="center",
                         weight=weight, zorder=9,
                         bbox=dict(boxstyle="square,pad=0.2", fc=bg, ec="none"))

    def legend(self, top, items, fontsize=6.7, r=0.055, gap=0.28, family=None):
        """Centered row of (kind, color, letter, label) icon-legend entries."""
        fam = {"family": family} if family else {}
        rend = self.fig.canvas.get_renderer()
        widths = []
        for kind, _, _, label in items:
            t = self.fig.text(0, 0, label, fontsize=fontsize, **fam)
            bb = t.get_window_extent(renderer=rend)
            widths.append((bb.width / self.fig.dpi) + (2 * r + 0.05 if kind else 0))
            t.remove()
        x = (self.w - (sum(widths) + gap * (len(items) - 1))) / 2
        for (kind, color, letter, label), wd in zip(items, widths):
            tx = x
            if kind == "frame":
                self.fig.add_artist(FancyBboxPatch(
                    (x, self.y(top) - r), 2 * r, 2 * r, transform=self.tr,
                    boxstyle="square,pad=0", fill=False,
                    edgecolor=color, linewidth=1.6))
                tx = x + 2 * r + 0.05
            elif kind:
                self.icon(x + r, top, r, kind, color, letter)
                tx = x + 2 * r + 0.05
            self.text(tx, top, label, fontsize=fontsize, va="center", color=INK, **fam)
            x += wd + gap

    def save(self, stem):
        for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
            self.fig.savefig(FIG_DIR / f"{stem}.{ext}", **kw)
        plt.close(self.fig)
        jpeg_pdf(FIG_DIR / f"{stem}.pdf")


def jpeg_pdf(path, quality=90):
    """Re-encode embedded raster thumbnails as JPEG (matplotlib writes lossless Flate)."""
    import io
    import pymupdf
    doc = pymupdf.open(path)
    for page in doc:
        for info in page.get_images(full=True):
            xref = info[0]
            pix = pymupdf.Pixmap(doc, xref)
            if pix.alpha or pix.n != 3 or min(pix.width, pix.height) < 32:
                continue
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
            page.replace_image(xref, stream=buf.getvalue())
    tmp = path.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(path)


def fit_lines(text, width_chars, max_lines):
    lines = textwrap.wrap(text, width_chars)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" ,.;") + "…"
    return lines


# ----------------------------------------------------------------------------- figure A
def teaser(selected, picks):
    W = 7.0
    s, g, col_gap = 0.87, 0.105, 0.36      # thumb, gap between thumbs, gap between groups
    gw = 3 * s + 2 * g
    left = (W - (2 * gw + col_gap)) / 2
    head, bar_h, row_gap = 0.30, 0.17, 0.08
    top0 = 0.03
    ly = top0 + 2 * (head + s + bar_h) + row_gap + 0.09
    H = ly + 0.11
    cv = Canvas(W, H)
    records = []
    for k, (country, info) in enumerate(selected):
        grp = info["group"]
        rec = picks["qwen"][grp["prompt_id"]]
        cands = presented(grp, rec)
        chosen = pick(rec)
        us = [c["utility"] for c in cands]
        top_u, mean = max(us), info["mean"]
        gx = left + (k % 2) * (gw + col_gap)
        gy = top0 + (k // 2) * (head + s + bar_h + row_gap)
        cx = gx + gw / 2
        cv.text(cx, gy + 0.07, country_name(country), fontsize=8.2, weight="bold",
                ha="center", va="center")
        lines = fit_lines(f"“{grp['prompt']}”", 78, 1)
        cv.text(cx, gy + 0.205, lines[0], fontsize=6.6, style="italic", ha="center",
                va="center", color=MUTED)
        it = gy + head
        for i, c in enumerate(cands):
            x = gx + i * (s + g)
            cv.image(thumb(c["image"], 420), x, it, s)
            best = c["utility"] >= top_u - EPS
            is_q = c["id"] == chosen
            if is_q:
                cv.frame(x, it, s, BAD, lw=2.4)
                cv.icon(x + 0.10, it + 0.10, 0.068, "cross", BAD)
                cv.pill(x + s - 0.05, it + s - 0.085, "Qwen’s pick", 5.8, ha="right",
                        bg=QWEN + "E6")
            elif best:
                cv.frame(x, it, s, GOOD, lw=2.4)
                cv.icon(x + 0.10, it + 0.10, 0.068, "check", GOOD)
            # alignment bar beneath the thumbnail
            by = it + s + 0.095
            bx0, bw = x + 0.0, s - 0.28
            col = BAD if is_q else (GOOD if best else MUTED)
            cv.fig.add_artist(Rectangle((bx0, cv.y(by) - 0.025), bw, 0.05, transform=cv.tr,
                                        facecolor=GRID, edgecolor="none"))
            cv.fig.add_artist(Rectangle((bx0, cv.y(by) - 0.025), bw * c["utility"], 0.05,
                                        transform=cv.tr, facecolor=col, edgecolor="none"))
            mx = bx0 + bw * mean
            cv.fig.add_artist(Line2D([mx, mx], [cv.y(by) - 0.05, cv.y(by) + 0.05],
                                     transform=cv.tr, color=INK, lw=0.8))
            cv.text(x + s, by, f"{c['utility']:.2f}", fontsize=6.8, weight="bold",
                    color=col, ha="right", va="center")
        records.append(dict(
            country=country, prompt_id=grp["prompt_id"], prompt=grp["prompt"],
            category=grp["category"], pool_mean=mean, qwen_pick=chosen,
            qwen_pick_utility=next(c["utility"] for c in cands if c["id"] == chosen),
            max_utility=top_u, regret=info["regret"],
            candidates_in_presented_order=[dict(
                id=c["id"], source_model=c["source_model"], utility=c["utility"],
                qwen_pick=c["id"] == chosen, annotation_best=c["utility"] >= top_u - EPS)
                for c in cands]))
    cv.legend(ly, [
        ("check", GOOD, None, "highest human rating in pool"),
        ("cross", BAD, None, "Qwen’s pick (original order), below pool mean"),
        (None, None, None, "bar/number: mean human prompt-alignment; tick = pool mean"),
    ])
    cv.save("teaser")
    return records, (W, H)


# ----------------------------------------------------------------------------- figure B
def gallery(groups, picks):
    """Two columns of five countries (alphabetical, top-to-bottom then left-to-right)."""
    W, s, g, col_gap, lab_w = 7.0, 0.715, 0.07, 0.30, 0.20
    half_w = lab_w + 4 * s + 3 * g
    left = (W - (2 * half_w + col_gap)) / 2
    head, after = 0.37, 0.09
    pitch = head + s + after
    n_rows = (len(groups) + 1) // 2
    top0 = 0.02
    grid_bottom = top0 + n_rows * pitch - after
    H = grid_bottom + 0.30
    cv = Canvas(W, H)
    records = []
    for i, grp in enumerate(groups):
        col, r = divmod(i, n_rows)
        x0 = left + col * (half_w + col_gap)
        ty = top0 + r * pitch
        iy = ty + head
        rq, rs = picks["qwen"].get(grp["prompt_id"]), picks["smol"].get(grp["prompt_id"])
        cands = presented(grp, rq)
        q, sm = pick(rq), pick(rs)
        us = [c["utility"] for c in cands]
        top_u, mean = max(us), float(np.mean(us))
        if r:
            cv.fig.add_artist(Line2D([x0, x0 + half_w], [cv.y(ty - after / 2 - 0.005)] * 2,
                                     transform=cv.tr, color=GRID, lw=0.5))
        cv.text(x0 + 0.085, iy + s / 2, country_name(grp["country"]), fontsize=8.5,
                weight="bold", rotation=90, ha="center", va="center", **SERIF)
        tx = x0 + lab_w
        cv.text(tx, iy - 0.165, "\n".join(fit_lines(f"“{grp['prompt']}”", 70, 2)),
                fontsize=6.9, style="italic", va="bottom", linespacing=1.12, **SERIF)
        cv.text(tx, iy - 0.045,
                f"{grp['category'].replace('-', ' ').replace('_', ' ')}  ·  pool mean {mean:.2f}",
                fontsize=6.4, color=MUTED, va="bottom", **SERIF)
        for k, c in enumerate(cands):
            x = tx + k * (s + g)
            badges = []
            if c["id"] == q:
                badges.append(("letter", QWEN, "Q"))
            if c["id"] == sm:
                badges.append(("letter", SMOL, "S"))
            frame = QWEN if c["id"] == q else (SMOL if c["id"] == sm else None)
            tile(cv, thumb(c["image"], 360), x, iy, s, f"{c['utility']:.2f}", frame=frame,
                 icon_tr=badges or None,
                 icon_tl=("check", GOOD, None) if c["utility"] >= top_u - EPS else None)
            if c["id"] == q and c["id"] == sm:  # nested: outer blue (Qwen), inner orange (Smol)
                p = 0.014
                cv.fig.add_artist(Rectangle((x + p, cv.y(iy + s) + p), s - 2 * p, s - 2 * p,
                                            transform=cv.tr, fill=False, edgecolor=SMOL,
                                            lw=1.8, joinstyle="miter", zorder=6))
        for k in range(len(cands), 4):
            placeholder(cv, tx + k * (s + g), iy, s, len(cands))
        records.append(dict(
            country=grp["country"], prompt_id=grp["prompt_id"], prompt=grp["prompt"],
            category=grp["category"], n_candidates=len(cands), pool_mean=mean,
            qwen_pick=q, smol_pick=sm,
            candidates_in_presented_order=[dict(
                id=c["id"], source_model=c["source_model"], utility=c["utility"],
                qwen_pick=c["id"] == q, smol_pick=c["id"] == sm,
                annotation_best=c["utility"] >= top_u - EPS) for c in cands]))
    cv.fig.add_artist(Line2D([W / 2] * 2, [cv.y(top0), cv.y(grid_bottom)], transform=cv.tr,
                             color=GRID, lw=0.7))
    cv.legend(H - 0.12, [
        ("check", GOOD, None, "highest human rating"),
        ("letter", QWEN, "Q", "Qwen’s pick (original order)"),
        ("letter", SMOL, "S", "Smol’s pick (original order)"),
        (None, None, None, "numbers: mean human prompt-alignment (0–1)"),
    ], fontsize=7.4, family="Times New Roman")
    cv.save("gallery_appendix")
    return records, (W, H)

# ----------------------------------------------------------------------------- serif appendix galleries
# Style of analysis/hero_figure.py::hero_column: serif headers, square frames, circular icons,
# rating printed on the image in white on a small dark square box.
SERIF = {"family": "Times New Roman"}
CLIP_C = "#8E6CC7"
N_ORDERS, N_CLIP_EACH, N_STEREO = 5, 3, 4


def rkey(v):
    return round(v, 9)


def load_all_perms():
    return {(r["prompt_id"], r["permutation"]): r for r in load_jsonl(ROOT / "runs/qwen_main.jsonl")}


def select_orders(groups, qall):
    rows = []
    for g in groups:
        pid = g["prompt_id"]
        if pid.startswith(EXCLUDE_PREFIX) or len(g["candidates"]) not in (3, 4):
            continue
        rs = [qall.get((pid, k)) for k in range(3)]
        if any(r is None or not r["parse_ok"] for r in rs):
            continue
        slots = [{v: k for k, v in r["mapping"].items()}[r["selected_id"]] for r in rs]
        ids = [r["selected_id"] for r in rs]
        if slots != ["A"] * 3 or len(set(ids)) < 2:
            continue
        u = {c["id"]: c["utility"] for c in g["candidates"]}
        spread = max(u[i] for i in ids) - min(u[i] for i in ids)
        rows.append(dict(group=g, runs=rs, spread=spread))
    rows.sort(key=lambda r: (-rkey(r["spread"]), r["group"]["prompt_id"]))
    return rows[:N_ORDERS], len(rows)


def select_clip(groups, picks):
    from analysis.content_baselines import select as clip_select
    scores = json.loads((ROOT / "results/main/clip_scores.json").read_text())["scores"]
    diff = []
    for g in groups:
        q = pick(picks["qwen"].get(g["prompt_id"]))
        c = clip_select(g, scores)
        if q is None or c == q:
            continue
        u = {x["id"]: x["utility"] for x in g["candidates"]}
        diff.append(dict(group=g, clip=c, qwen=q, delta=u[c] - u[q]))
    key = lambda r: (-rkey(abs(r["delta"])), r["group"]["prompt_id"])
    clip_better = sorted([r for r in diff if r["delta"] > EPS], key=key)[:N_CLIP_EACH]
    qwen_better = sorted([r for r in diff if r["delta"] < -EPS], key=key)[:N_CLIP_EACH]
    return clip_better, qwen_better, len(diff)


def select_stereo(groups, picks):
    rows = []
    for g in groups:
        q = pick(picks["qwen"].get(g["prompt_id"]))
        cs = g["candidates"]
        if q is None or any(c["stereotype"] is None for c in cs):
            continue
        qc = next(c for c in cs if c["id"] == q)
        if qc["stereotype"] <= 0 or qc["stereotype"] < max(c["stereotype"] for c in cs) - EPS:
            continue
        alts = [c for c in cs if c["id"] != q and c["utility"] >= qc["utility"] - EPS
                and c["stereotype"] < qc["stereotype"] - EPS]
        if not alts:
            continue
        gap = qc["stereotype"] - min(c["stereotype"] for c in alts)
        rows.append(dict(group=g, qwen=q, alts=[c["id"] for c in alts], gap=gap))
    rows.sort(key=lambda r: (-rkey(r["gap"]), r["group"]["prompt_id"]))
    return rows[:N_STEREO], len(rows)


def tile(cv, arr, x, top, s, label, frame=None, icon_tr=None, icon_tl=None, fs=6.3):
    """Image tile: square frame, icons (kind, color, letter) at top corners, rating box bottom-left."""
    cv.image(arr, x, top, s)
    if frame:
        p = 0.018
        cv.fig.add_artist(Rectangle((x - p, cv.y(top + s) - p), s + 2 * p, s + 2 * p,
                                    transform=cv.tr, fill=False, edgecolor=frame, lw=2.0,
                                    joinstyle="miter", zorder=6))
    r = 0.058
    for spec, cx in ((icon_tr, x + s - 0.09), (icon_tl, x + 0.09)):
        if spec:
            specs = spec if isinstance(spec, list) else [spec]
            for k, (kind, color, letter) in enumerate(specs):
                cv.icon(cx - k * 0.135 * (1 if cx > x + s / 2 else -1), top + 0.09, r, kind, color, letter)
    cv.text(x + 0.045, top + s - 0.05, label, fontsize=fs, color="white", weight="bold",
            va="bottom", zorder=9,
            bbox=dict(boxstyle="square,pad=0.18", fc=(0, 0, 0, .55), ec="none"))


def placeholder(cv, x, top, s, n):
    """Pools smaller than the widest row leave the cell empty; captions note three-image pools."""
    return


def cand_records(cands, **flags):
    top_u = max(c["utility"] for c in cands)
    return [dict(id=c["id"], source_model=c["source_model"], utility=c["utility"],
                 stereotype=c["stereotype"], annotation_best=c["utility"] >= top_u - EPS,
                 **{k: c["id"] == v if isinstance(v, str) else c["id"] in v
                    for k, v in flags.items()}) for c in cands]


def fig_orders(rows):
    W = 7.0
    s, g, sep = 0.71, 0.09, 0.20
    lab_w, txt_w = 0.24, 1.95
    grid_w = lab_w + txt_w + 4 * s + 2 * g + sep
    left = (W - grid_w) / 2
    top0, head = 0.03, 0.20
    pitch = s + 0.09
    H = top0 + head + len(rows) * pitch + 0.24
    cv = Canvas(W, H)
    ix0 = left + lab_w + txt_w
    xs = [ix0 + k * (s + g) for k in range(3)] + [ix0 + 2 * (s + g) + s + sep]
    for x, h in zip(xs, ["Order 1", "Order 2", "Order 3", "Best available"]):
        cv.text(x + s / 2, top0 + 0.09, h, fontsize=8.5, weight="bold", ha="center",
                va="center", **SERIF)
    rx = ix0 + 3 * s + 2 * g + sep / 2
    records = []
    for r, row in enumerate(rows):
        grp = row["group"]
        cand = {c["id"]: c for c in grp["candidates"]}
        best = max(cand.values(), key=lambda c: (rkey(c["utility"]), ""))  # value only
        top_u = best["utility"]
        best_c = sorted([c for c in grp["candidates"] if c["utility"] >= top_u - EPS],
                        key=lambda c: c["id"])[0]
        ty = top0 + head + r * pitch
        if r:
            cv.fig.add_artist(Line2D([left, xs[-1] + s], [cv.y(ty - 0.045)] * 2,
                                     transform=cv.tr, color=GRID, lw=0.6))
        cv.text(left + 0.08, ty + s / 2, country_name(grp["country"]), fontsize=8.5,
                weight="bold", rotation=90, ha="center", va="center", **SERIF)
        tx = left + lab_w
        cv.text(tx, ty + 0.02, "\n".join(fit_lines(f"“{grp['prompt']}”", 40, 3)),
                fontsize=7.2, style="italic", va="top", linespacing=1.2, **SERIF)
        n_distinct = len({x["selected_id"] for x in row["runs"]})
        cv.text(tx, ty + s - 0.02,
                f"Qwen picks slot A in all 3 orders,\n{n_distinct} different images",
                fontsize=6.4, color=MUTED, va="bottom", **SERIF)
        for k, run in enumerate(row["runs"]):
            c = cand[run["selected_id"]]
            good = c["utility"] >= top_u - EPS
            tile(cv, thumb(c["image"], 240), xs[k], ty, s, f"{c['utility']:.2f}",
                 frame=GOOD if good else BAD,
                 icon_tr=("check" if good else "cross", GOOD if good else BAD, None))
        tile(cv, thumb(best_c["image"], 240), xs[3], ty, s, f"{best_c['utility']:.2f}")
        cv.fig.add_artist(Line2D([rx, rx], [cv.y(ty), cv.y(ty + s)], transform=cv.tr,
                                 color=MUTED, lw=0.6))
        records.append(dict(
            country=grp["country"], prompt_id=grp["prompt_id"], prompt=grp["prompt"],
            n_candidates=len(cand), utility_range_of_picks=row["spread"],
            qwen_picks_by_order=[dict(permutation=x["permutation"], slot="A",
                                      selected_id=x["selected_id"],
                                      utility=cand[x["selected_id"]]["utility"])
                                 for x in row["runs"]],
            best_available=dict(id=best_c["id"], utility=top_u)))
    cv.legend(H - 0.12, [
        ("check", GOOD, None, "Qwen’s pick is best-rated"),
        ("cross", BAD, None, "Qwen’s pick is not best-rated"),
        (None, None, None, "numbers: mean human prompt-alignment (0–1)"),
    ], fontsize=7.4, family="Times New Roman")
    cv.save("gallery_orders")
    return records, (W, H)


def paired_rows(cv, items, left, top0, s, g, half_w, n_rows, draw_row):
    """Lay out items in two side-by-side columns of n_rows rows (prompt above each row)."""
    head_h = 0.28
    pitch = head_h + s + 0.11
    for i, item in enumerate(items):
        col, r = divmod(i, n_rows)
        x0 = left + col * (half_w + 0.30)
        ty = top0 + r * pitch
        grp = item["group"]
        iy = ty + head_h
        cv.text(x0 + 0.22, iy - 0.05, "\n".join(fit_lines(f"“{grp['prompt']}”", 68, 2)),
                fontsize=6.9, style="italic", va="bottom", linespacing=1.15, **SERIF)
        cv.text(x0 + 0.08, iy + s / 2, country_name(grp["country"]), fontsize=8.2, weight="bold",
                rotation=90, ha="center", va="center", **SERIF)
        draw_row(item, x0 + 0.22, iy)
    return pitch


def fig_clip(clip_better, qwen_better, picks):
    W, s, g = 7.0, 0.70, 0.07
    half_w = 0.22 + 4 * s + 3 * g
    left = (W - (2 * half_w + 0.30)) / 2
    top0 = 0.25
    records = []

    def draw(item, x, iy):
        grp = item["group"]
        cands = presented(grp, picks["qwen"][grp["prompt_id"]])
        top_u = max(c["utility"] for c in cands)
        for i, c in enumerate(cands):
            badge, frame = [], None
            if c["id"] == item["qwen"]:
                badge, frame = [("letter", QWEN, "Q")], QWEN
            if c["id"] == item["clip"]:
                badge, frame = [("letter", CLIP_C, "C")], CLIP_C
            tile(cv, thumb(c["image"], 240), x + i * (s + g), iy, s, f"{c['utility']:.2f}",
                 frame=frame, icon_tr=badge or None,
                 icon_tl=("check", GOOD, None) if c["utility"] >= top_u - EPS else None)
        for i in range(len(cands), 4):
            placeholder(cv, x + i * (s + g), iy, s, len(cands))
        records.append(dict(
            country=grp["country"], prompt_id=grp["prompt_id"], prompt=grp["prompt"],
            clip_pick=item["clip"], qwen_pick=item["qwen"],
            utility_clip_minus_qwen=item["delta"],
            candidates_in_presented_order=cand_records(cands, qwen_pick=item["qwen"],
                                                       clip_pick=item["clip"])))

    pitch = 0.28 + s + 0.11
    H = top0 + N_CLIP_EACH * pitch + 0.20
    cv = Canvas(W, H)
    for col, h in enumerate(["CLIP’s pick rated higher", "Qwen’s pick rated higher"]):
        cv.text(left + col * (half_w + 0.30) + half_w / 2 + 0.11, 0.11, h, fontsize=8.8,
                weight="bold", ha="center", va="center", **SERIF)
    paired_rows(cv, clip_better + qwen_better, left, top0, s, g, half_w, N_CLIP_EACH, draw)
    cv.fig.add_artist(Line2D([W / 2] * 2, [cv.y(0.02), cv.y(H - 0.24)], transform=cv.tr,
                             color=GRID, lw=0.7))
    cv.legend(H - 0.11, [
        ("letter", QWEN, "Q", "Qwen’s pick (original order)"),
        ("letter", CLIP_C, "C", "CLIP-score pick"),
        ("check", GOOD, None, "highest human rating"),
        (None, None, None, "numbers: mean human prompt-alignment (0–1)"),
    ], fontsize=7.4, family="Times New Roman")
    cv.save("gallery_clip")
    return records, (W, H)


def fig_stereo(rows, picks):
    W, s, g = 7.0, 0.70, 0.07
    half_w = 0.22 + 4 * s + 3 * g
    left = (W - (2 * half_w + 0.30)) / 2
    top0 = 0.04
    n_rows = (len(rows) + 1) // 2
    pitch = 0.28 + s + 0.11
    H = top0 + n_rows * pitch + 0.20
    cv = Canvas(W, H)
    records = []

    def draw(item, x, iy):
        grp = item["group"]
        cands = presented(grp, picks["qwen"][grp["prompt_id"]])
        for i, c in enumerate(cands):
            is_q = c["id"] == item["qwen"]
            tile(cv, thumb(c["image"], 240), x + i * (s + g), iy, s,
                 f"A {c['utility']:.2f} · S {c['stereotype']:.2f}", fs=5.9,
                 frame=BAD if is_q else None,
                 icon_tr=("cross", BAD, None) if is_q else None,
                 icon_tl=("check", GOOD, None) if c["id"] in item["alts"] else None)
        for i in range(len(cands), 4):
            placeholder(cv, x + i * (s + g), iy, s, len(cands))
        records.append(dict(
            country=grp["country"], prompt_id=grp["prompt_id"], prompt=grp["prompt"],
            qwen_pick=item["qwen"], stereotype_gap=item["gap"],
            candidates_in_presented_order=cand_records(cands, qwen_pick=item["qwen"],
                                                       dominating_alternative=item["alts"])))

    paired_rows(cv, rows, left, top0, s, g, half_w, n_rows, draw)
    cv.fig.add_artist(Line2D([W / 2] * 2, [cv.y(0.02), cv.y(H - 0.24)], transform=cv.tr,
                             color=GRID, lw=0.7))
    cv.legend(H - 0.11, [
        ("cross", BAD, None, "Qwen’s pick: highest stereotype rate in pool"),
        ("check", GOOD, None, "alignment ≥ Qwen’s pick, lower stereotype rate"),
        (None, None, None, "A: mean prompt-alignment; S: stereotype rate"),
    ], fontsize=7.4, family="Times New Roman")
    cv.save("gallery_stereotype")
    return records, (W, H)


def main():
    apply()
    groups, picks = load()
    selected, ranked = select_teaser(groups, picks)
    teaser_records, teaser_size = teaser(selected, picks)
    gal_records, gal_size = gallery(select_gallery(groups), picks)
    ord_rows, n_ord = select_orders(groups, load_all_perms())
    ord_records, ord_size = fig_orders(ord_rows)
    clip_b, qwen_b, n_clip = select_clip(groups, picks)
    clip_records, clip_size = fig_clip(clip_b, qwen_b, picks)
    st_rows, n_st = select_stereo(groups, picks)
    st_records, st_size = fig_stereo(st_rows, picks)
    size = lambda v: [round(x, 3) for x in v]
    save_json(JSON_OUT, dict(
        teaser=dict(
            figure="paper/figures/teaser.pdf", size_in=[round(v, 3) for v in teaser_size],
            rule=("Among main prompts with 3 candidates (excluding prompt_id prefix "
                  f"{EXCLUDE_PREFIX}), keep those where Qwen's permutation-0 pick has utility "
                  f"below the pool mean and the pool has a candidate with utility >= mean + "
                  f"{MIN_GAP}. Per country keep the prompt with the largest regret (max utility "
                  "- chosen utility; ties -> first in manifest order); take the "
                  f"{N_TEASER} countries with the largest regret (ties -> country name). "
                  "Selection uses only data/derived/main.jsonl and runs/qwen_main.jsonl; "
                  "images were not viewed during selection. Candidates shown in "
                  "permutation-0 presented order."),
            eligible_per_country=[dict(country=c, prompt_id=i["group"]["prompt_id"],
                                       regret=i["regret"]) for c, i in ranked],
            selected=teaser_records),
        gallery_appendix=dict(
            figure="paper/figures/gallery_appendix.pdf", size_in=[round(v, 3) for v in gal_size],
            rule=("Per country (alphabetical), the first prompt in data/derived/main.jsonl "
                  "order with 4 candidates (fallback 3). Representative, unfiltered sample: "
                  "chosen by manifest position only, not by judge outcome or image content. "
                  "Picks are permutation-0 (original order) selections from "
                  "runs/qwen_main.jsonl and runs/smol_main.jsonl."),
            rows=gal_records),
        gallery_orders=dict(
            figure="paper/figures/gallery_orders.pdf", size_in=size(ord_size),
            rule=("Main prompts with 3 or 4 candidates (excluding prompt_id prefix "
                  f"{EXCLUDE_PREFIX}) where Qwen (runs/qwen_main.jsonl) parses and selects "
                  "presentation slot A under all three orders (permutations 0-2) and the three "
                  "picks contain >= 2 distinct images. Rank by the range (max - min) of utility "
                  f"among the three picks, descending (ties -> prompt_id); take {N_ORDERS}. "
                  "'Best available' = highest-utility candidate (ties -> smallest id). "
                  "Images were not viewed during selection."),
            n_eligible=n_ord, rows=ord_records),
        gallery_clip=dict(
            figure="paper/figures/gallery_clip.pdf", size_in=size(clip_size),
            rule=("CLIP pick = analysis.content_baselines.select (argmax of "
                  "results/main/clip_scores.json cosine score; ties -> canonical sorted id). "
                  "Among prompts where the CLIP pick differs from Qwen's permutation-0 pick, "
                  f"take the {N_CLIP_EACH} with the largest utility(CLIP) - utility(Qwen) > 0 and "
                  f"the {N_CLIP_EACH} with the largest utility(Qwen) - utility(CLIP) > 0 "
                  "(ties -> prompt_id). Candidates shown in permutation-0 presented order. "
                  "Images were not viewed during selection."),
            n_disagreements=n_clip, rows=clip_records),
        gallery_stereotype=dict(
            figure="paper/figures/gallery_stereotype.pdf", size_in=size(st_size),
            rule=("Prompts where Qwen's permutation-0 pick has the highest stereotype rate in "
                  "the pool (ties allowed) and stereotype > 0, and at least one other candidate "
                  "has utility >= the pick's utility and strictly lower stereotype "
                  "('dominating alternative'). Stereotype gap = pick's stereotype - lowest "
                  f"stereotype among dominating alternatives; take the {N_STEREO} largest "
                  "(ties -> prompt_id). Images were not viewed during selection."),
            n_eligible=n_st, rows=st_records),
    ))


if __name__ == "__main__":
    main()
