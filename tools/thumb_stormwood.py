"""Stormwood Cabin thumbnail maker (1280x720). Same colours and font every time, layout changes per video.
usage: thumb_stormwood.py <scene.png> <out.jpg> <layout: left|right|bottom|top> "<LINE 1>" "<LINE 2>" "<small line>"
Font: Bitter (SIL Open Font Licence, assets/fonts/BITTER-OFL-LICENSE.txt).
Colours: Stormwood palette - Charred Walnut #2B1D14 fade, Lantern Gold #F2B544 headline,
Smoke Cream #EDE3D3 second line, Ember Orange #E07A2E accent rule."""
import sys, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = lambda w, s: ImageFont.truetype(f"{ROOT}/assets/fonts/bitter-latin-{w}-normal.woff", s)
WALNUT, GOLD, CREAM, EMBER = (43, 29, 20), (242, 181, 68), (237, 227, 211), (224, 122, 46)
W, H = 1280, 720


def scrim(img, layout):
    """soft charred-walnut fade behind the text so it stays readable on a phone."""
    g = Image.new("L", (W, H), 0); d = ImageDraw.Draw(g)
    for i in range(256):
        a = int(225 * (1 - i / 255) ** 1.6)
        if layout == "left":   d.line([(int(i * W * 0.62 / 255), 0), (int(i * W * 0.62 / 255), H)], fill=a, width=4)
        if layout == "right":  d.line([(W - int(i * W * 0.62 / 255), 0), (W - int(i * W * 0.62 / 255), H)], fill=a, width=4)
        if layout == "bottom": d.line([(0, H - int(i * H * 0.6 / 255)), (W, H - int(i * H * 0.6 / 255))], fill=a, width=4)
        if layout == "top":    d.line([(0, int(i * H * 0.6 / 255)), (W, int(i * H * 0.6 / 255))], fill=a, width=4)
    g = g.filter(ImageFilter.GaussianBlur(30))
    return Image.composite(Image.new("RGB", (W, H), WALNUT), img, g)


def main():
    scene, out, layout, l1, l2, small = sys.argv[1:7]
    img = Image.open(scene).convert("RGB")
    r = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * r), round(img.height * r)), Image.LANCZOS)
    img = img.crop(((img.width - W) // 2, (img.height - H) // 2, (img.width - W) // 2 + W, (img.height - H) // 2 + H))
    img = scrim(img, layout); d = ImageDraw.Draw(img)
    f1, f2, f3 = F(700, 100), F(500, 58), F(500, 30)
    maxw = W * (0.80 if layout in ("bottom", "top") else 0.58)
    while d.textlength(l1, font=f1) > maxw and f1.size > 56: f1 = F(700, f1.size - 4)
    while d.textlength(l2, font=f2) > maxw and f2.size > 36: f2 = F(500, f2.size - 3)
    h1, h2, h3 = f1.size * 1.12, f2.size * 1.25, f3.size * 1.3
    block = h1 + 22 + h2 + 18 + h3
    pad = 70
    if layout == "left":     x, y, anchor = pad, (H - block) / 2, "l"
    elif layout == "right":  x, y, anchor = W - pad, (H - block) / 2, "r"
    elif layout == "bottom": x, y, anchor = W / 2, H - pad - block, "m"
    else:                    x, y, anchor = W / 2, pad, "m"
    def put(text, font, col, yy):
        tw = d.textlength(text, font=font)
        xx = x if anchor == "l" else (x - tw if anchor == "r" else x - tw / 2)
        d.text((xx + 2, yy + 3), text, font=font, fill=(20, 12, 8))      # soft shadow
        d.text((xx, yy), text, font=font, fill=col)
        return xx, tw
    put(l1, f1, GOLD, y)
    yy = y + h1 + 4
    rule = 120
    rx = x if anchor == "l" else (x - rule if anchor == "r" else x - rule / 2)
    d.line([(rx, yy + 6), (rx + rule, yy + 6)], fill=EMBER, width=4)
    put(l2, f2, CREAM, yy + 22)
    put(small, f3, CREAM, yy + 22 + h2 + 14)
    img.save(out, quality=92)


if __name__ == "__main__":
    main()
