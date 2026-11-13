"""Conversions between jaxgraphic's planar float images and other layouts and dtypes."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._types import Array, ArrayLike


def to_uint8(img: ArrayLike) -> Array:
    """Convert a float image in ``[0, 1]`` to 8-bit: clip, scale to 0..255, round to nearest.

    This is plain ``jax.numpy``, so it works inside ``jit``. Converting on the device
    *before* copying to the host moves 4x fewer bytes, which matters when you are
    producing frames at a high rate.
    """
    return (jnp.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(jnp.uint8)


def to_hwc(img: ArrayLike):
    """``(C, H, W)`` -> ``(H, W, C)`` (e.g. for ``matplotlib.imshow``); works on NumPy and JAX arrays."""
    return img.transpose(1, 2, 0) if hasattr(img, "transpose") else np.transpose(img, (1, 2, 0))


def from_hwc(img: ArrayLike) -> NDArray[np.float32]:
    """Convert PIL/matplotlib-style data to the channels-first float32 layout.

    Accepts ``(H, W)`` grayscale or ``(H, W, C)``, floats in [0, 1] or integers 0..255
    (for example ``from_hwc(np.asarray(PIL.Image.open(path)))``). Returns
    ``float32[C, H, W]`` (``[1, H, W]`` for grayscale).
    """
    a = np.asarray(img)
    a = a.astype(np.float32) / 255.0 if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3:
        raise ValueError(f"expected (H, W) or (H, W, C), got shape {a.shape}")
    return np.ascontiguousarray(a.transpose(2, 0, 1))
