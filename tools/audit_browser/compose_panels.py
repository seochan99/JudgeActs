"""Compose the three browser screenshots into paper/figures/browser_panels.pdf.

Layout: (a) overview on top at full width (7.0 in), (b) explorer and (c) detail below.
Screenshots are embedded as JPEG (DCT) streams to keep the PDF small; labels are vector text.

Usage:  uv run python tools/audit_browser/compose_panels.py
"""
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'paper/figures'
PANELS = [('browser_overview.png', '(a) Overview dashboard', 3200),
          ('browser_explore.png', '(b) Prompt explorer (country = Japan)', 1800),
          ('browser_detail.png', '(c) Prompt detail: three orders, ratings, gates', 1800)]
W = 7.0 * 72          # page width in pt
GAP = 8               # gap between the two bottom panels
LAB = 11              # label band height
INK = (0x1F / 255, 0x2A / 255, 0x33 / 255)
EDGE = (0.82, 0.85, 0.88)


def jpeg(path, width, quality=84):
    im = Image.open(path).convert('RGB')
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality, optimize=True, progressive=False)
    return buf.getvalue(), im.width / im.height


def main():
    imgs = [jpeg(FIG / f, w) for f, _, w in PANELS]
    w_top, w_bot = W, (W - GAP) / 2
    h_top, h_bot = w_top / imgs[0][1], w_bot / imgs[1][1]
    H = LAB + h_top + 6 + LAB + h_bot
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    boxes = [fitz.Rect(0, LAB, w_top, LAB + h_top)]
    y2 = LAB + h_top + 6 + LAB
    boxes += [fitz.Rect(0, y2, w_bot, y2 + h_bot), fitz.Rect(w_bot + GAP, y2, W, y2 + h_bot)]
    for (data, _), (_, label, _), r in zip(imgs, PANELS, boxes):
        page.insert_image(r, stream=data, keep_proportion=False)
        page.draw_rect(r, color=EDGE, width=0.5)
        page.insert_text((r.x0, r.y0 - 3), label, fontname='hebo', fontsize=7.5, color=INK)
    out = FIG / 'browser_panels.pdf'
    doc.save(out, garbage=4, deflate=True)
    print(f'wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {W / 72:.2f} x {H / 72:.2f} in)')


if __name__ == '__main__':
    main()
