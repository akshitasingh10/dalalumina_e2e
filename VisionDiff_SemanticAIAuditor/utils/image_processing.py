"""Draw bounding boxes from Gemini-style normalized coordinates on screenshots."""

from __future__ import annotations

import io
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# Coordinates are [ymin, xmin, ymax, xmax] on a 0–1000 scale (inclusive).
NormBox = Sequence[float]


def normalized_box_to_pixels(
    box: NormBox, width: int, height: int, scale: float = 1000.0
) -> tuple[int, int, int, int]:
    """Map normalized [ymin, xmin, ymax, xmax] to pixel (left, top, right, bottom)."""
    if len(box) != 4:
        raise ValueError("box must have four values [ymin, xmin, ymax, xmax]")
    ymin, xmin, ymax, xmax = (float(v) for v in box)
    sx = width / scale
    sy = height / scale
    left = int(max(0, xmin * sx))
    top = int(max(0, ymin * sy))
    right = int(min(width, xmax * sx))
    bottom = int(min(height, ymax * sy))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def draw_boxes_on_image(
    image_bytes: bytes,
    boxes: list[tuple[NormBox, str]],
    *,
    outline: str = "#FF3366",
    width_px: int = 3,
    label_fill: str = "#FFFFFF",
    label_bg: str = "#FF3366",
) -> bytes:
    """Overlay rectangles and optional labels; returns PNG bytes.

    ``boxes`` is a list of (normalized_box, label) pairs. Labels may be empty.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    w, h = img.size
    for box, label in boxes:
        l, t, r, b = normalized_box_to_pixels(box, w, h)
        draw.rectangle([l, t, r, b], outline=outline, width=width_px)
        if label:
            text = label[:80]
            if font:
                tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            else:
                tw, th = len(text) * 6, 12
            pad = 2
            bg = [l, t - th - 2 * pad, l + tw + 2 * pad, t]
            if bg[1] < 0:
                bg = [l, b, l + tw + 2 * pad, b + th + 2 * pad]
            draw.rectangle(bg, fill=label_bg)
            draw.text((bg[0] + pad, bg[1] + pad), text, fill=label_fill, font=font)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
