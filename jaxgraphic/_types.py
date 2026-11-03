"""Type aliases shared across jaxgraphic.

These are re-exported from :mod:`jaxgraphic` so user code can annotate with them:

    import jaxgraphic as jg

    def my_scene(t: jg.Array) -> jg.Array: ...
    def my_shape(c: jg.Point) -> jg.Shape: ...
"""

from __future__ import annotations

from collections.abc import Callable

from jax import Array
from jax.typing import ArrayLike

#: A scalar or array-like value (Python number, NumPy array, JAX array, ...).
#: Shape parameters accept this, so they may be traced values under ``jit``/``grad``.
Scalar = ArrayLike

#: An (x, y) position in pixel coordinates; any array-like of shape ``(2,)``.
Point = ArrayLike

#: A color: a ``"#rrggbb"`` / ``"#rgb"`` string, or an array-like ``(r, g, b)``
#: with channels in ``[0, 1]``.
Color = str | ArrayLike

#: A float image of shape ``(H, W, 3)`` with values in ``[0, 1]``.
Image = Array

#: A shape: a signed distance function mapping pixel coordinates ``(..., 2)``
#: to signed distances ``(...)`` (negative inside, positive outside).
Shape = Callable[[Array], Array]

__all__ = ["Array", "ArrayLike", "Color", "Image", "Point", "Scalar", "Shape"]
