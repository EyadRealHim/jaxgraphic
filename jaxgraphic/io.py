"""Image output. A dependency-free PNG writer (only NumPy and the standard library)."""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
from numpy.typing import NDArray

from ._types import ArrayLike


def to_uint8(img: ArrayLike) -> NDArray[np.uint8]:
    """Convert a float image in ``[0, 1]`` to 8-bit.

    Values are clipped to ``[0, 1]`` first, then scaled to ``[0, 255]`` and
    rounded to nearest.

    Args:
        img: Float image of any shape.

    Returns:
        A NumPy ``uint8`` array of the same shape. This forces device
        computation, so call it outside ``jit``.
    """
    return np.asarray(np.clip(np.asarray(img), 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def save_png(
    img: ArrayLike, path: str | os.PathLike[str], scale: int = 1
) -> str | os.PathLike[str]:
    """Save an RGB image as a PNG file.

    Args:
        img: Float image of shape ``(H, W, 3)`` with values in ``[0, 1]``
            (out-of-range values are clipped).
        path: Output file path.
        scale: Integer nearest-neighbor upscale factor (``2`` writes a file
            twice as wide and tall). Handy for viewing small renders.

    Returns:
        ``path``, for convenience.

    Raises:
        ValueError: If ``img`` is not shaped ``(H, W, 3)`` or ``scale < 1``.
    """
    a = to_uint8(img)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {a.shape}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale > 1:
        a = a.repeat(scale, axis=0).repeat(scale, axis=1)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))  # filter type 0

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)
    return path
