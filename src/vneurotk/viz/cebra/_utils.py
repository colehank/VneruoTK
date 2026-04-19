from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

ImageSource = Union[str, Path, Image.Image]


def get_img(img_id: str, images: dict[str, ImageSource]) -> Image.Image:
    """Return a PIL Image for *img_id*, or a gray placeholder if absent."""
    img_id = str(img_id)
    if img_id not in images:
        return Image.new("RGB", (224, 224), (128, 128, 128))
    val = images[img_id]
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    return Image.open(val).convert("RGB")


def get_images(
    img_ids,
    images: dict[str, ImageSource],
    padding: int = 5,
    bg_color: tuple = (255, 255, 255),
) -> Image.Image:
    """Tile multiple images into a square grid."""
    img_ids = np.asarray(img_ids).ravel()
    if img_ids.size == 1:
        return get_img(str(img_ids[0]), images)

    imgs = [get_img(str(iid), images) for iid in img_ids]
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    canvas = Image.new(
        "RGB",
        (cols * w + (cols + 1) * padding, rows * h + (rows + 1) * padding),
        bg_color,
    )
    for i, im in enumerate(imgs):
        x = (i % cols) * w + (i % cols + 1) * padding + (w - im.width) // 2
        y = (i // cols) * h + (i // cols + 1) * padding + (h - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def fmt_time(samples: int, sfreq: Optional[float]) -> str:
    if sfreq is not None:
        return f"{samples / sfreq * 1000:.0f} ms"
    return str(samples)
