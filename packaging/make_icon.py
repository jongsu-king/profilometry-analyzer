"""Generate the application icon (PNG, ICO, ICNS) with Pillow.

A dark rounded square with a stylus tracing a film step (substrate, wall, film top)
in the accent colour - the palette of the application.  Run from the repository root:

    python packaging/make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZE = 1024
BG = (26, 26, 25, 255)
BORDER = (56, 56, 53, 255)
ACCENT = (57, 135, 229, 255)
ACCENT_FILL = (57, 135, 229, 90)
SECOND = (217, 89, 38, 255)
INK = (240, 239, 236, 255)


def render(size: int = SIZE) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=size * 0.22,
        fill=BG,
        outline=BORDER,
        width=max(2, size // 128),
    )
    left, right = size * 0.10, size * 0.90
    low, high = size * 0.74, size * 0.46
    x_edge = size * 0.52
    n = 400
    pts = []
    for i in range(n + 1):
        x = left + (right - left) * i / n
        s = 0.5 * (1 + math.tanh((x - x_edge) / (size * 0.012)))
        pts.append((x, low + (high - low) * s))
    poly = [(left, size * 0.84)] + pts + [(right, size * 0.84)]
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(poly, fill=ACCENT_FILL)
    img.alpha_composite(layer)
    d.line(pts, fill=ACCENT, width=max(3, size // 36), joint="curve")
    # the measured height: a double arrow between the two levels
    xa = size * 0.70
    w = max(3, size // 64)
    d.line([(xa, high + size * 0.04), (xa, low - size * 0.04)], fill=INK, width=w)
    for y, sgn in ((high + size * 0.035, 1), (low - size * 0.035, -1)):
        d.polygon(
            [
                (xa, y - sgn * size * 0.035),
                (xa - size * 0.03, y + sgn * size * 0.01),
                (xa + size * 0.03, y + sgn * size * 0.01),
            ],
            fill=INK,
        )
    # stylus: a cone with a round tip on the film top
    tip_x, tip_y = size * 0.30, low - size * 0.035
    r = size * 0.035
    d.polygon(
        [(tip_x - size * 0.09, size * 0.16), (tip_x + size * 0.09, size * 0.16), (tip_x, tip_y)],
        fill=SECOND,
    )
    d.ellipse((tip_x - r, tip_y - r, tip_x + r, tip_y + r), fill=SECOND)
    return img


def main() -> None:
    img = render()
    img.save(HERE / "icon.png")
    img.resize((256, 256), Image.LANCZOS).save(HERE / "icon_256.png")
    img.save(
        HERE / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )
    img.save(HERE / "icon.icns")
    for name in ("icon.png", "icon_256.png", "icon.ico", "icon.icns"):
        print(name, (HERE / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
