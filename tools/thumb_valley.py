"""Valley & Tide thumbnail maker (1280x720). Same colours and font every time, layout changes per video.
usage: thumb_valley.py <scene.png> <out.png> <layout: left|right|bottom|top> <icon: rain|wave|stream|storm|moon> "<LINE 1>" "<LINE 2>" "<small line>"
Font: Outfit (SIL Open Font Licence, assets/fonts). Colours: Valley & Tide brand palette."""
import sys, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = lambda w, s: ImageFont.truetype(f"{ROOT}/assets/fonts/outfit-latin-{w}-normal.woff", s)
MIDNIGHT, TEAL, MIST, MOON, LILAC = (11, 20, 38), (15, 76, 92), (143, 184, 201), (232, 238, 242), (156, 137, 224)
W, H = 1280, 720


def scrim(img, layout):
    """soft midnight-blue fade behind the text so it stays readable on a phone."""
    g = Image.new("L", (W, H), 0); d = ImageDraw.Draw(g)
    for i in range(256):
        a = int(215 * (1 - i / 255) ** 1.6)
        if layout == "left":   d.line([(int(i * W * 0.62 / 255), 0), (int(i * W * 0.62 / 255), H)], fill=a, width=4)
        if layout == "right":  d.line([(W - int(i * W * 0.62 / 255), 0), (W - int(i * W * 0.62 / 255), H)], fill=a, width=4)
        if layout == "bottom": d.line([(0, H - int(i * H * 0.6 / 255)), (W, H - int(i * H * 0.6 / 255))], fill=a, width=4)
        if layout == "top":    d.line([(0, int(i * H * 0.6 / 255)), (W, int(i * H * 0.6 / 255))], fill=a, width=4)
    g = g.filter(ImageFilter.GaussianBlur(30))
    return Image.composite(Image.new("RGB", (W, H), MIDNIGHT), img, g)


def icon(d, kind, x, y, s, col):
    w = max(3, s // 14)
    if kind == "rain":
        for i, dx in enumerate((0, 0.38, 0.76)):
            cx, cy = x + dx * s, y + (0.15 if i == 1 else 0) * s
            d.line([(cx + 0.12 * s, cy), (cx, cy + 0.55 * s)], fill=col, width=w)
    elif kind == "wave":
        import math
        for row in (0, 0.32):
            pts = [(x + t * s, y + row * s + 0.12 * s * math.sin(t * 2 * math.pi * 1.5)) for t in [k / 40 for k in range(41)]]
            d.line(pts, fill=col, width=w, joint="curve")
    elif kind == "stream":
        import math
        for row in (0, 0.25, 0.5):
            pts = [(x + t * s, y + row * s + 0.07 * s * math.sin(t * 2 * math.pi * 2 + row * 4)) for t in [k / 40 for k in range(41)]]
            d.line(pts, fill=col, width=w, joint="curve")
    elif kind == "moon":
        d.arc([x, y, x + 0.8 * s, y + 0.8 * s], 60, 300, fill=col, width=w)
        d.arc([x + 0.22 * s, y + 0.02 * s, x + 0.9 * s, y + 0.7 * s], 75, 285, fill=col, width=w)
        d.ellipse([x + 0.85 * s, y - 0.05 * s, x + 0.85 * s + w * 1.6, y - 0.05 * s + w * 1.6], fill=col)
    elif kind == "storm":
        d.arc([x, y, x + 0.55 * s, y + 0.45 * s], 160, 360, fill=col, width=w)
        d.arc([x + 0.35 * s, y - 0.12 * s, x + s, y + 0.45 * s], 190, 30, fill=col, width=w)
        d.line([(x + 0.05 * s, y + 0.45 * s), (x + 0.95 * s, y + 0.45 * s)], fill=col, width=w)
        d.line([(x + 0.55 * s, y + 0.5 * s), (x + 0.42 * s, y + 0.75 * s), (x + 0.56 * s, y + 0.75 * s), (x + 0.44 * s, y + s)], fill=col, width=w, joint="curve")


def main():
    scene, out, layout, kind, l1, l2, small = sys.argv[1:8]
    img = Image.open(scene).convert("RGB")
    r = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * r), round(img.height * r)), Image.LANCZOS)
    img = img.crop(((img.width - W) // 2, (img.height - H) // 2, (img.width - W) // 2 + W, (img.height - H) // 2 + H))
    img = scrim(img, layout); d = ImageDraw.Draw(img)
    f1, f2, f3, fw = F(700, 104), F(600, 64), F(400, 38), F(600, 24)
    maxw = W * (0.80 if layout in ("bottom", "top") else 0.60)
    while d.textlength(l1, font=f1) > maxw and f1.size > 60: f1 = F(700, f1.size - 4)
    while d.textlength(l2, font=f2) > maxw and f2.size > 40: f2 = F(600, f2.size - 3)
    lines = [(l1, f1, MOON), (l2, f2, MIST)]
    widths = [d.textlength(t, font=f) for t, f, _ in lines]
    block_w = max(max(widths), d.textlength(small, font=f3))
    pad = 64
    if layout == "left":   x0, y0, align = pad, 190, "l"
    if layout == "right":  x0, y0, align = W - pad - block_w, 190, "r"
    if layout == "bottom": x0, y0, align = (W - block_w) / 2, H - 330, "c"
    if layout == "top":    x0, y0, align = (W - block_w) / 2, 110, "c"
    xs = lambda tw: x0 if align == "l" else (x0 + block_w - tw if align == "r" else (W - tw) / 2)
    iy = y0 - 62
    icon(d, kind, xs(64), iy - 8, 64, LILAC)
    y = y0 + 10
    for (t, f, c), tw in zip(lines, widths):
        d.text((xs(tw) + 3, y + 3), t, font=f, fill=(5, 10, 20))
        d.text((xs(tw), y), t, font=f, fill=c)
        y += f.size + 18
    lw = 120; d.line([(xs(lw), y + 6), (xs(lw) + lw, y + 6)], fill=LILAC, width=5)
    sw = d.textlength(small, font=f3); d.text((xs(sw), y + 26), small, font=f3, fill=MOON)
    mark = "VALLEY & TIDE"; mw = d.textlength(mark, font=fw) + 4 * len(mark)
    mx = W - pad - mw if layout != "right" else pad
    cx, my = mx, (40 if layout == "bottom" else H - 52)
    for ch in mark:
        d.text((cx + 2, my + 2), ch, font=fw, fill=(5, 10, 20)); d.text((cx, my), ch, font=fw, fill=MOON); cx += d.textlength(ch, font=fw) + 4
    img.save(out, quality=95)


if __name__ == "__main__":
    main()
