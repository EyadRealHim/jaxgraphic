"""Rasterization basics: colors, blank canvases, pixel coordinates, coverage and blending.

Images are ``float32`` arrays in planar layout ``(C, H, W)`` (channels first) with values
in ``[0, 1]``. Planar keeps the width axis innermost, which vectorizes on CPU and avoids
padding a size-3 minor dimension on accelerators. Use :func:`jaxgraphic.to_hwc` /
:func:`jaxgraphic.from_hwc` to convert to and from ``(H, W, C)``.
"""
from __future__ import annotations

import jax.numpy as jnp

from ._types import Array, Color, Image, Scalar
from ._util import F32


def rgb(color: Color) -> Array:
    """Convert ``"#rrggbb"`` / ``"#rgb"`` or an ``(r, g, b)`` array-like in [0, 1] to ``float32[3]``.

    Raises:
        ValueError: If a string is not a valid 3- or 6-digit hex color.
    """
    if isinstance(color, str):
        digits = color.lstrip("#")
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) != 6:
            raise ValueError(f"expected '#rgb' or '#rrggbb', got {color!r}")
        return jnp.asarray([int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=F32)
    return jnp.asarray(color, dtype=F32)


def new_canvas(height: int, width: int, bg: Color = "#ffffff") -> Image:
    """A solid-color ``float32[3, height, width]`` image (sizes are static ints)."""
    return jnp.broadcast_to(rgb(bg)[:, None, None], (3, height, width))


def coords(height: int, width: int) -> tuple[Array, Array]:
    """Pixel-center coordinates as separable arrays ``x: (1, W)`` and ``y: (H, 1)``.

    They broadcast against each other, so shapes can do per-axis work on small arrays
    and only pay O(H*W) for the final combination. ``x[0, j] = j + 0.5``, ``y[i, 0] = i + 0.5``.
    """
    x = (jnp.arange(width, dtype=F32) + 0.5)[None, :]
    y = (jnp.arange(height, dtype=F32) + 0.5)[:, None]
    return x, y


def coverage(sd: Array, softness: Scalar = 1.0) -> Array:
    """Signed distance to anti-aliased coverage in ``[0, 1]``: ``clip(0.5 - sd / softness)``.

    ``softness`` is the edge width in pixels (> 0). ``1.0`` gives a crisp one-pixel edge.
    """
    return jnp.clip(0.5 - sd * (1.0 / softness), 0.0, 1.0)


def blend(img: Array, alpha: Array, color: Array) -> Array:
    """Source-over blend of a solid ``color`` using per-pixel ``alpha``.

    ``img + alpha * (color - img)``: the lerp form needs one fewer multiply per channel
    than ``img * (1 - alpha) + color * alpha``.

    Args:
        img: ``(C, H, W)`` destination.
        alpha: ``(H, W)`` opacity.
        color: ``(C,)`` or ``(C, H, W)`` source color.
    """
    color = color[:, None, None] if color.ndim == 1 else color
    return img + alpha[None] * (color - img)
