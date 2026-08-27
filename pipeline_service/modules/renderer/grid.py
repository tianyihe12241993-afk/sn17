"""Grid collage composition — Python owns the canonical orbit and composes
the 2x2 collage the way the browser canvas used to."""
from __future__ import annotations

import io
import math
from typing import Sequence

from PIL import Image

# Canonical grid orbit (theta, phi) in degrees.
GRID_VIEWS: tuple[tuple[float, float], ...] = (
    (24.0, -15.0),
    (120.0, -15.0),
    (216.0, -15.0),
    (312.0, -15.0),
)
GRID_KEYS: tuple[str, ...] = tuple(f"grid_{i}" for i in range(len(GRID_VIEWS)))


def compose_grid(tiles: Sequence[bytes], *, img_size: int, gap: int) -> bytes:
    """Compose PNG tiles row-major onto opaque black with ``gap`` px gutters;
    alpha_composite mirrors drawImage's source-over for alpha < 255 tiles."""
    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    width = cols * img_size + (cols - 1) * gap
    height = rows * img_size + (rows - 1) * gap

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    for i, tile_bytes in enumerate(tiles):
        tile = Image.open(io.BytesIO(tile_bytes))
        if tile.size != (img_size, img_size):
            raise ValueError(
                f"grid tile {i} is {tile.size[0]}x{tile.size[1]}, expected {img_size}x{img_size}"
            )
        row, col = divmod(i, cols)
        canvas.alpha_composite(tile.convert("RGBA"), (col * (img_size + gap), row * (img_size + gap)))

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="PNG")
    return out.getvalue()
