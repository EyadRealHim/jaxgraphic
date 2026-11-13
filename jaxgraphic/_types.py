"""Type aliases shared across jaxgraphic."""
from __future__ import annotations

from typing import Union

from jax import Array
from jax.typing import ArrayLike

#: A scalar or array-like value (Python number, NumPy/JAX array, tracer, ...).
Scalar = ArrayLike

#: An ``(x, y)`` position in pixels; any array-like of shape ``(2,)``.
Point = ArrayLike

#: ``"#rrggbb"`` / ``"#rgb"`` or an array-like ``(r, g, b)`` with channels in ``[0, 1]``.
Color = Union[str, ArrayLike]

#: A float image in planar layout ``(C, H, W)`` (channels first), values in ``[0, 1]``.
Image = Array

__all__ = ["Array", "ArrayLike", "Color", "Image", "Point", "Scalar"]
