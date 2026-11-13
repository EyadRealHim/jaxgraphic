"""Painting: ``fill``, ``stroke``, ``linear_gradient`` and ``fill_many``.

Every public function here is ``jax.jit``-compiled over *pytrees*. Shapes are pytrees, so
the compiled code is cached by shape structure and image size, and eager calls run as one
fused kernel instead of dozens of tiny dispatches. Inside your own ``jit`` these inline,
and XLA can fuse consecutive fills into a single pass over the image.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ._types import Array, ArrayLike, Color, Image, Point, Scalar
from ._util import F32, as_f32
from .raster import blend, coords, coverage, rgb
from .shapes import Outline, Shape, ShapeLike, as_shape

Window = tuple[int, int]     # static (width, height) in pixels


# --------------------------------------------------------------------------
# One shape, whole canvas
# --------------------------------------------------------------------------
def _fill_whole(img, shape, color, alpha, softness):
    """Blend ``color`` through ``shape``'s coverage over every pixel."""
    _, h, w = img.shape
    x, y = coords(h, w)
    a = coverage(shape.sdf(x, y), softness) * alpha
    return blend(img, jnp.broadcast_to(a, (h, w)), color)


# --------------------------------------------------------------------------
# One shape, small window
# --------------------------------------------------------------------------
def _window_origin(center, window_size: int, limit: int):
    """Integer start of a window of ``window_size`` pixels centered on ``center``, kept inside ``[0, limit)``."""
    return jnp.clip(jnp.round(center).astype(jnp.int32) - window_size // 2, 0, limit - window_size)


def _fill_in_window(img, shape, color, alpha, softness, at, window: Window):
    """Like :func:`_fill_whole`, but only evaluate and write a window around ``at``.

    Cost is O(window) instead of O(H*W). Pixels outside the window are left untouched.
    """
    channels, h, w = img.shape
    win_w, win_h = min(window[0], w), min(window[1], h)
    x0 = _window_origin(at[0], win_w, w)
    y0 = _window_origin(at[1], win_h, h)

    # Pixel-center coordinates of the window, in full-canvas space.
    xs = (x0.astype(F32) + jnp.arange(win_w, dtype=F32) + 0.5)[None, :]
    ys = (y0.astype(F32) + jnp.arange(win_h, dtype=F32) + 0.5)[:, None]
    a = coverage(shape.sdf(xs, ys), softness) * alpha

    zero = jnp.zeros((), jnp.int32)
    start = (zero, y0, x0)
    region = lax.dynamic_slice(img, start, (channels, win_h, win_w))
    painted = blend(region, jnp.broadcast_to(a, (win_h, win_w)), color)
    return lax.dynamic_update_slice(img, painted, start)


_fill_whole_jit = jax.jit(_fill_whole)
_fill_in_window_jit = jax.jit(_fill_in_window, static_argnames=("window",))


def _window_args(at, window) -> tuple:
    if at is None:
        raise ValueError("window= needs at=(x, y): the point the window is centered on")
    return as_f32(at), (int(window[0]), int(window[1]))


# --------------------------------------------------------------------------
# Public painting functions
# --------------------------------------------------------------------------
def fill(img: Image, shape: ShapeLike, color: Color, alpha: Scalar = 1.0,
         softness: Scalar = 1.0, *, at: Point | None = None,
         window: Window | None = None) -> Image:
    """Paint the inside of ``shape`` over ``img`` (source-over blending).

    Args:
        img: ``float32[3, H, W]`` image. Not modified.
        shape: A :class:`~jaxgraphic.Shape` (or a function ``(x, y) -> distance``).
        color: Fill color (see :func:`rgb`).
        alpha: Overall opacity in ``[0, 1]``.
        softness: Edge width in pixels.
        at: With ``window``: the pixel position the window is centered on.
        window: Optional static ``(width, height)``. Only a window of that size around
            ``at`` is evaluated and written (cost O(window) instead of O(H*W)); it is
            clamped to stay inside the image. It must contain the shape's visible extent,
            including the ``softness`` fringe, or the shape is cut off at the window edge.

    Returns:
        A new image.
    """
    shape, color = as_shape(shape), rgb(color)
    if window is None:
        return _fill_whole_jit(img, shape, color, alpha, softness)
    at, window = _window_args(at, window)
    return _fill_in_window_jit(img, shape, color, alpha, softness, at, window=window)


def stroke(img: Image, shape: ShapeLike, color: Color, width: Scalar = 2.0,
           alpha: Scalar = 1.0, softness: Scalar = 1.0, *, at: Point | None = None,
           window: Window | None = None) -> Image:
    """Paint the outline of ``shape``: a band ``width`` pixels wide centered on its edge.

    Takes the same ``at`` / ``window`` options as :func:`fill`.
    """
    return fill(img, Outline(shape, width), color, alpha, softness, at=at, window=window)


@jax.jit
def _gradient_jit(img, shape, start, end, color0, color1, alpha, softness):
    _, h, w = img.shape
    x, y = coords(h, w)
    dx, dy = end[0] - start[0], end[1] - start[1]
    inv_length_sq = 1.0 / (dx * dx + dy * dy + 1e-9)
    t = jnp.clip(((x - start[0]) * dx + (y - start[1]) * dy) * inv_length_sq, 0.0, 1.0)
    color = color0[:, None, None] + t[None] * (color1 - color0)[:, None, None]
    a = coverage(shape.sdf(x, y), softness) * alpha
    return blend(img, jnp.broadcast_to(a, (h, w)), color)


def linear_gradient(img: Image, shape: ShapeLike, start: Point, end: Point, color0: Color,
                    color1: Color, alpha: Scalar = 1.0, softness: Scalar = 1.0) -> Image:
    """Fill ``shape`` with a linear gradient: ``color0`` at ``start``, ``color1`` at ``end``, clamped beyond.

    Example:
        ``linear_gradient(img, Box((w/2, h/2), (w, h)), (0, 0), (0, h), "#ffb86b", "#5ec3ff")``
    """
    return _gradient_jit(img, as_shape(shape), as_f32(start), as_f32(end),
                         rgb(color0), rgb(color1), alpha, softness)


# --------------------------------------------------------------------------
# Many shapes in one pass
# --------------------------------------------------------------------------
def _as_color_table(colors) -> Array:
    if isinstance(colors, (list, tuple)):
        return jnp.stack([rgb(c) for c in colors])
    colors = jnp.asarray(colors, dtype=F32)
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError(f"colors must be (N, 3) or a list of color specs, got shape {colors.shape}")
    return colors


@partial(jax.jit, static_argnames=("window",))
def _fill_many_jit(img, shapes, colors, alphas, softness, at, window):
    alphas = jnp.broadcast_to(alphas, (colors.shape[0],))

    def paint_one(canvas, item):
        shape, color, alpha, position = item
        if window is None:
            return _fill_whole(canvas, shape, color, alpha, softness), None
        return _fill_in_window(canvas, shape, color, alpha, softness, position, window), None

    out, _ = lax.scan(paint_one, img, (shapes, colors, alphas, at))
    return out


def fill_many(img: Image, shapes: Shape, colors, alphas: ArrayLike = 1.0,
              softness: Scalar = 1.0, *, at: ArrayLike | None = None,
              window: Window | None = None) -> Image:
    """Paint N shapes in order (later ones on top) with one ``lax.scan``.

    Compile time and program size stay constant in N, where a Python loop of ``fill``
    calls unrolls into N copies of the shape code (XLA can take minutes to compile that).

    Args:
        img: Image to draw on.
        shapes: ONE shape object whose leaves carry a leading axis of length N, e.g.
            ``Circle(centers, radii)`` with ``centers: (N, 2)`` and ``radii: (N,)``.
        colors: ``(N, 3)`` array, or a list of N color specs.
        alphas: Scalar or ``(N,)`` opacities.
        softness: Edge width in pixels.
        at: ``(N, 2)`` window centers; required with ``window``.
        window: Static ``(width, height)`` per-shape window, as in :func:`fill`. Each step
            then touches O(window) pixels instead of the whole canvas.

    Returns:
        A new image.
    """
    colors = _as_color_table(colors)
    if window is None:
        at = None
    else:
        at, window = _window_args(at, window)
    return _fill_many_jit(img, shapes, colors, as_f32(alphas), softness, at, window=window)
