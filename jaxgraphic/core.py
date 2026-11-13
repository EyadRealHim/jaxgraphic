"""Canvas creation and compositing.

**Images** are ``float32`` arrays in *planar* layout ``(C, H, W)`` (channels
first), values in ``[0, 1]``. Planar keeps the width axis innermost, which
vectorizes on CPU and avoids padding a size-3 minor dimension on accelerators.
Use :func:`jaxgraphic.to_hwc` / :func:`jaxgraphic.from_hwc` to talk to
matplotlib/PIL-style ``(H, W, C)`` arrays.

Every drawing function is pure (returns a new image) and is itself ``jax.jit``-ed
over *pytrees*: shapes are pytrees (see :mod:`jaxgraphic.shapes`), so compiled code
is cached by shape structure and image size, and eager calls (no outer ``jit``)
run as one fused kernel instead of dozens of tiny dispatches. Inside your own
``jit`` the inner calls are inlined and XLA fuses consecutive fills into one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ._types import Array, ArrayLike, Color, Image, Point, Scalar
from .io import save_png, to_hwc
from .shapes import F32, Outline, Shape, ShapeLike, _f32, _hyp, as_shape


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------
def rgb(color: Color) -> Array:
    """Convert ``"#rrggbb"`` / ``"#rgb"`` or an ``(r, g, b)`` array-like in [0, 1] to ``float32[3]``.

    Raises:
        ValueError: If a string is not a valid 3- or 6-digit hex color.
    """
    if isinstance(color, str):
        h = color.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) != 6:
            raise ValueError(f"expected '#rgb' or '#rrggbb', got {color!r}")
        return jnp.asarray(
            [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=F32
        )
    return jnp.asarray(color, dtype=F32)


def new_canvas(height: int, width: int, bg: Color = "#ffffff") -> Image:
    """A solid-color ``float32[3, height, width]`` image (sizes are static ints)."""
    return jnp.broadcast_to(rgb(bg)[:, None, None], (3, height, width))


def coords(height: int, width: int) -> tuple[Array, Array]:
    """Pixel-center coordinates as separable arrays ``x: (1, W)`` and ``y: (H, 1)``.

    These broadcast against each other, so shapes can do per-axis work on small
    arrays. ``x[0, j] = j + 0.5`` and ``y[i, 0] = i + 0.5``.
    """
    x = (jnp.arange(width, dtype=F32) + 0.5)[None, :]
    y = (jnp.arange(height, dtype=F32) + 0.5)[:, None]
    return x, y


def coverage(sd: Array, softness: Scalar = 1.0) -> Array:
    """Signed distance to anti-aliased coverage in ``[0, 1]``: ``clip(0.5 - sd / softness)``.

    ``softness`` is the edge width in pixels (> 0). Larger values blur the edge,
    letting gradients reach further from a shape: useful while optimizing, then
    anneal toward ``1.0``.
    """
    return jnp.clip(0.5 - sd * (1.0 / softness), 0.0, 1.0)


def _blend(img: Array, a: Array, color: Array) -> Array:
    """``img + a * (color - img)``: lerp form (3 flops/channel instead of 4)."""
    return img + a[None] * (color[:, None, None] - img)


# --------------------------------------------------------------------------
# fill / stroke / gradient
# --------------------------------------------------------------------------
def _fill_core(img, shape, color, alpha, softness):
    _, h, w = img.shape
    x, y = coords(h, w)
    a = coverage(shape.sdf(x, y), softness) * alpha
    return _blend(img, jnp.broadcast_to(a, (h, w)), color)


def _fill_window_core(img, shape, color, alpha, softness, at, window):
    c, h, w = img.shape
    ww, wh = min(window[0], w), min(window[1], h)
    x0 = jnp.clip(jnp.round(at[0]).astype(jnp.int32) - ww // 2, 0, w - ww)
    y0 = jnp.clip(jnp.round(at[1]).astype(jnp.int32) - wh // 2, 0, h - wh)
    xs = (x0.astype(F32) + jnp.arange(ww, dtype=F32) + 0.5)[None, :]
    ys = (y0.astype(F32) + jnp.arange(wh, dtype=F32) + 0.5)[:, None]
    a = jnp.broadcast_to(coverage(shape.sdf(xs, ys), softness) * alpha, (wh, ww))
    zero = jnp.zeros((), jnp.int32)
    region = lax.dynamic_slice(img, (zero, y0, x0), (c, wh, ww))
    return lax.dynamic_update_slice(img, _blend(region, a, color), (zero, y0, x0))


_fill_jit = jax.jit(_fill_core)
_fill_window_jit = jax.jit(_fill_window_core, static_argnames=("window",))


def _window_args(at, window):
    if at is None:
        raise ValueError("window= needs at=(x, y): the point the window is centered on")
    return _f32(at), (int(window[0]), int(window[1]))


def fill(
    img: Image,
    shape: ShapeLike,
    color: Color,
    alpha: Scalar = 1.0,
    softness: Scalar = 1.0,
    *,
    at: Point | None = None,
    window: tuple[int, int] | None = None,
) -> Image:
    """Paint the inside of ``shape`` over ``img`` (source-over blending).

    Args:
        img: ``float32[3, H, W]`` image. Not modified.
        shape: A :class:`~jaxgraphic.Shape` (or a function ``(x, y) -> distance``).
        color: Fill color (see :func:`rgb`).
        alpha: Overall opacity in ``[0, 1]``.
        softness: Edge width in pixels (see :func:`coverage`).
        at: With ``window``: the pixel position the window is centered on.
        window: Optional static ``(width, height)`` in pixels. Only a window of that
            size around ``at`` is evaluated and written, costing O(window) instead of
            O(H*W); the window is clamped to stay inside the image. It must contain
            the shape's visible extent (including the ``softness`` fringe), or the
            shape is cut off at the window edge. ``at`` only chooses the window and
            gets no gradient; the shape's own parameters do.

    Returns:
        A new image.
    """
    shape, color = as_shape(shape), rgb(color)
    if window is None:
        return _fill_jit(img, shape, color, alpha, softness)
    at, window = _window_args(at, window)
    return _fill_window_jit(img, shape, color, alpha, softness, at, window=window)


def stroke(
    img: Image,
    shape: ShapeLike,
    color: Color,
    width: Scalar = 2.0,
    alpha: Scalar = 1.0,
    softness: Scalar = 1.0,
    *,
    at: Point | None = None,
    window: tuple[int, int] | None = None,
) -> Image:
    """Paint the outline of ``shape``: a band ``width`` pixels wide centered on its edge.

    Takes the same ``at`` / ``window`` options as :func:`fill`.
    """
    return fill(
        img, Outline(shape, width), color, alpha, softness, at=at, window=window
    )


@jax.jit
def _gradient_jit(img, shape, start, end, c0, c1, alpha, softness):
    _, h, w = img.shape
    x, y = coords(h, w)
    dx, dy = end[0] - start[0], end[1] - start[1]
    t = jnp.clip(
        ((x - start[0]) * dx + (y - start[1]) * dy)
        * (1.0 / (dx * dx + dy * dy + 1e-9)),
        0.0,
        1.0,
    )
    col = c0[:, None, None] + t[None] * (c1 - c0)[:, None, None]
    a = jnp.broadcast_to(coverage(shape.sdf(x, y), softness) * alpha, (h, w))
    return img + a[None] * (col - img)


def linear_gradient(
    img: Image,
    shape: ShapeLike,
    start: Point,
    end: Point,
    c0: Color,
    c1: Color,
    alpha: Scalar = 1.0,
    softness: Scalar = 1.0,
) -> Image:
    """Fill ``shape`` with a linear gradient: ``c0`` at ``start``, ``c1`` at ``end``, clamped beyond.

    Example:
        ``linear_gradient(img, Box((w/2, h/2), (w, h)), (0, 0), (0, h), "#ffb86b", "#5ec3ff")``
    """
    return _gradient_jit(
        img, as_shape(shape), _f32(start), _f32(end), rgb(c0), rgb(c1), alpha, softness
    )


# --------------------------------------------------------------------------
# Many shapes at once
# --------------------------------------------------------------------------
def _as_colors(colors) -> Array:
    if isinstance(colors, (list, tuple)):
        return jnp.stack([rgb(c) for c in colors])
    colors = jnp.asarray(colors, dtype=F32)
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError(
            f"colors must be (N, 3) or a list of color specs, got shape {colors.shape}"
        )
    return colors


@partial(jax.jit, static_argnames=("window",))
def _fill_many_jit(img, shapes, colors, alphas, softness, at, window):
    alphas = jnp.broadcast_to(alphas, (colors.shape[0],))

    def body(canvas, item):
        shp, col, al, pos = item
        if window is None:
            return _fill_core(canvas, shp, col, al, softness), None
        return _fill_window_core(canvas, shp, col, al, softness, pos, window), None

    out, _ = lax.scan(body, img, (shapes, colors, alphas, at))
    return out


def fill_many(
    img: Image,
    shapes: Shape,
    colors,
    alphas: ArrayLike = 1.0,
    softness: Scalar = 1.0,
    *,
    at: ArrayLike | None = None,
    window: tuple[int, int] | None = None,
) -> Image:
    """Paint N shapes in order (later ones on top) with one ``lax.scan``.

    Compile time and program size stay constant in N, where a Python loop of
    ``fill`` calls unrolls into N copies of the shape code.

    Args:
        img: Image to draw on.
        shapes: ONE shape object whose leaves carry a leading axis of length N,
            e.g. ``Circle(centers, radii)`` with ``centers: (N, 2)``, ``radii: (N,)``.
            (For heterogeneous shapes, loop over :func:`fill` or ``vmap`` per type.)
        colors: ``(N, 3)`` array, or a list of N color specs.
        alphas: Scalar or ``(N,)`` opacities.
        softness: Edge width in pixels.
        at: ``(N, 2)`` window centers; required with ``window``.
        window: Static ``(width, height)`` per-shape window, as in :func:`fill`.
            With a window each step touches O(window) pixels, and reverse-mode
            keeps window-sized residuals per step instead of a whole canvas, so
            use it when N is large and you need gradients.

    Returns:
        A new image.
    """
    colors = _as_colors(colors)
    if window is None:
        at = None
    else:
        at, window = _window_args(at, window)
    return _fill_many_jit(
        img, shapes, colors, _f32(alphas), softness, at, window=window
    )


# --------------------------------------------------------------------------
# Drawing images
# --------------------------------------------------------------------------
def _as_chw(src: ArrayLike) -> Array:
    """Normalize a source image to ``float32[C, H, W]`` with C in {3, 4}.

    Accepts float in [0, 1] or integer 0..255; ``(H, W)`` grayscale and
    ``(1|2|3|4, H, W)`` channels-first arrays.
    """
    a = jnp.asarray(src)
    a = (
        (a.astype(F32) / 255.0)
        if jnp.issubdtype(a.dtype, jnp.integer)
        else a.astype(F32)
    )
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] not in (1, 2, 3, 4):
        raise ValueError(
            f"expected a channels-first (C, H, W) image with C in 1..4 "
            f"(use jaxgraphic.from_hwc for (H, W, C) data), got shape {a.shape}"
        )
    if a.shape[0] == 1:
        a = jnp.repeat(a, 3, axis=0)
    elif a.shape[0] == 2:
        a = jnp.concatenate([jnp.repeat(a[:1], 3, axis=0), a[1:]], axis=0)
    return a


def _draw_image_core(img, src, center, size, scale, angle, alpha, clip, softness):
    _, h, w = img.shape
    s = _as_chw(src)
    cp, sh, sw = s.shape
    x, y = coords(h, w)

    # Rotate pixel coords into the image's frame (skipped when angle is None).
    dx, dy = x - center[0], y - center[1]
    if angle is None:
        qx, qy = dx, dy
    else:
        co, si = jnp.cos(angle), jnp.sin(angle)
        qx, qy = co * dx + si * dy, co * dy - si * dx

    disp = size if size is not None else jnp.asarray([sw, sh], dtype=F32) * scale
    # Anti-aliased rectangle coverage (same math as Box, sharing the rotated coords).
    ax, ay = jnp.abs(qx) - disp[0] * 0.5, jnp.abs(qy) - disp[1] * 0.5
    sd = _hyp(jnp.maximum(ax, 0.0), jnp.maximum(ay, 0.0)) + jnp.minimum(
        jnp.maximum(ax, ay), 0.0
    )
    cov = coverage(sd, softness)
    if clip is not None:
        cov = cov * coverage(clip.sdf(x, y), softness)
    k = jnp.broadcast_to(cov * alpha, (h, w))

    # Source texel coordinates (pixel-centre convention), clamped to the image so
    # edge clamping falls out of the clamp and the +1 neighbour comes from padding.
    u = jnp.clip(qx * (sw / disp[0]) + (sw * 0.5 - 0.5), 0.0, sw - 1.0)
    v = jnp.clip(qy * (sh / disp[1]) + (sh * 0.5 - 0.5), 0.0, sh - 1.0)
    u0, v0 = jnp.floor(u), jnp.floor(v)
    fx = jnp.broadcast_to(u - u0, (h, w))
    fy = jnp.broadcast_to(v - v0, (h, w))

    # Premultiply once (O(source), not O(canvas)); edge-pad by one texel so the
    # right/bottom neighbours never need clamping; one flat row table for gathers.
    if cp == 4:
        prem = jnp.concatenate([s[:3] * s[3:4], s[3:4]], axis=0)
    else:
        prem = s
    pc = prem.shape[0]
    padded = jnp.pad(prem, ((0, 0), (0, 1), (0, 1)), mode="edge")
    table = padded.reshape(pc, (sh + 1) * (sw + 1)).T  # (texels, pc)
    stride = sw + 1
    i00 = jnp.broadcast_to(v0.astype(jnp.int32) * stride + u0.astype(jnp.int32), (h, w))

    def tap(i):
        return jnp.take(table, i, axis=0, mode="clip")  # (H, W, pc)

    gx = 1.0 - fx
    top = tap(i00) * gx[..., None] + tap(i00 + 1) * fx[..., None]
    bot = tap(i00 + stride) * gx[..., None] + tap(i00 + stride + 1) * fx[..., None]
    out = jnp.moveaxis(top + (bot - top) * fy[..., None], -1, 0)  # (pc, H, W)

    if cp == 4:
        return img + k[None] * (out[:3] - out[3:4] * img)
    return img + k[None] * (out - img)


_draw_image_jit = jax.jit(_draw_image_core)


def draw_image(
    img: Image,
    src: ArrayLike,
    center: Point,
    size: Point | None = None,
    scale: Scalar = 1.0,
    angle: Scalar | None = None,
    alpha: Scalar = 1.0,
    clip: ShapeLike | None = None,
    softness: Scalar = 1.0,
) -> Image:
    """Composite the image ``src`` onto ``img`` with position, size, rotation and clipping.

    The source is placed by its *center* (scale and rotation pivot there), resampled
    bilinearly with premultiplied alpha, and its edge is anti-aliased. Differentiable
    with respect to ``center``, ``size``/``scale``, ``angle``, ``alpha`` and the pixels
    of ``src``.

    Args:
        img: ``float32[3, H, W]`` canvas. Not modified.
        src: Channels-first ``(C, H, W)`` source: C = 3 (RGB) or 4 (RGBA, straight
            alpha); also ``(H, W)`` grayscale and C = 1 or 2 (gray, gray+alpha). Float in
            [0, 1] or integer 0..255. Use :func:`jaxgraphic.from_hwc` for ``(H, W, C)``
            data. Its shape must be static under ``jit``; values may be traced.
        center: ``(x, y)`` canvas position of the middle of ``src``.
        size: Optional ``(width, height)`` in canvas pixels; overrides ``scale``.
        scale: Size multiplier relative to the source's pixel size.
        angle: Rotation in radians about ``center`` (clockwise on screen). Leave as
            ``None`` for axis-aligned placement: the sampling coordinates then stay
            separable (O(H + W) setup work). ``0.0`` takes the general path.
        alpha: Overall opacity in ``[0, 1]``.
        clip: Optional shape; the image is only visible inside it.
        softness: Edge width in pixels for the rectangle edge and ``clip``.

    Note:
        Sampling is bilinear with no low-pass filter: shrinking by more than ~2x can
        alias, so downscale large images first.
    """
    return _draw_image_jit(
        img,
        src,
        _f32(center),
        None if size is None else _f32(size),
        scale,
        None if angle is None else _f32(angle),
        alpha,
        None if clip is None else as_shape(clip),
        softness,
    )


# --------------------------------------------------------------------------
# Chainable, immutable canvas
# --------------------------------------------------------------------------
@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Canvas:
    """An immutable, chainable canvas that works under ``jit``/``vmap``/``grad``.

    Every method returns a *new* ``Canvas``. It is a pytree (one leaf: the image), so
    it can be an argument or result of transformed functions.

    Example:
        >>> @jax.jit
        ... def render(t):
        ...     return (jg.Canvas.new(200, 200, "#fff")
        ...         .fill(jg.Circle((40 + 120 * t, 100), 30), "#e63946")
        ...         .stroke(jg.Box((100, 100), (160, 160)), "#1d3557", width=3))
        >>> render(0.5).save("out.png")      # save outside jit

    Wrapping the whole chain in one ``jit`` lets XLA fuse consecutive fills into a
    single pass over the image.

    Attributes:
        img: ``float32[3, H, W]`` (a tracer under transformations).
    """

    img: Image

    @classmethod
    def new(cls, height: int, width: int, bg: Color = "#ffffff") -> Canvas:
        """A blank canvas filled with ``bg`` (``height``/``width`` are static ints)."""
        return cls(new_canvas(height, width, bg))

    @classmethod
    def from_image(cls, img: Image) -> Canvas:
        """Wrap an existing ``(3, H, W)`` image."""
        return cls(img)

    def tree_flatten(self):
        return (self.img,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])  # no validation: JAX may pass placeholders

    @property
    def height(self) -> int:
        return self.img.shape[-2]

    @property
    def width(self) -> int:
        return self.img.shape[-1]

    @property
    def hwc(self) -> Array:
        """The image as ``(H, W, 3)`` (e.g. for ``matplotlib.imshow``)."""
        return to_hwc(self.img)

    def fill(
        self,
        shape: ShapeLike,
        color: Color,
        *,
        alpha: Scalar = 1.0,
        softness: Scalar = 1.0,
        at: Point | None = None,
        window: tuple[int, int] | None = None,
    ) -> Canvas:
        """See :func:`fill`."""
        return Canvas(
            fill(self.img, shape, color, alpha, softness, at=at, window=window)
        )

    def stroke(
        self,
        shape: ShapeLike,
        color: Color,
        *,
        width: Scalar = 2.0,
        alpha: Scalar = 1.0,
        softness: Scalar = 1.0,
        at: Point | None = None,
        window: tuple[int, int] | None = None,
    ) -> Canvas:
        """See :func:`stroke`."""
        return Canvas(
            stroke(self.img, shape, color, width, alpha, softness, at=at, window=window)
        )

    def gradient(
        self,
        shape: ShapeLike,
        start: Point,
        end: Point,
        c0: Color,
        c1: Color,
        *,
        alpha: Scalar = 1.0,
        softness: Scalar = 1.0,
    ) -> Canvas:
        """See :func:`linear_gradient`."""
        return Canvas(
            linear_gradient(self.img, shape, start, end, c0, c1, alpha, softness)
        )

    def fill_many(
        self,
        shapes: Shape,
        colors,
        alphas: ArrayLike = 1.0,
        *,
        softness: Scalar = 1.0,
        at: ArrayLike | None = None,
        window: tuple[int, int] | None = None,
    ) -> Canvas:
        """See :func:`fill_many`."""
        return Canvas(
            fill_many(self.img, shapes, colors, alphas, softness, at=at, window=window)
        )

    def image(
        self,
        src: ArrayLike,
        center: Point,
        *,
        size: Point | None = None,
        scale: Scalar = 1.0,
        angle: Scalar | None = None,
        alpha: Scalar = 1.0,
        clip: ShapeLike | None = None,
        softness: Scalar = 1.0,
    ) -> Canvas:
        """See :func:`draw_image`."""
        return Canvas(
            draw_image(self.img, src, center, size, scale, angle, alpha, clip, softness)
        )

    def pipe(self, fn, *args, **kwargs) -> Canvas:
        """Apply ``fn(img, *args, **kwargs) -> img`` and keep chaining."""
        return Canvas(fn(self.img, *args, **kwargs))

    def save(self, path: str, scale: int = 1) -> Canvas:
        """Write a PNG and return ``self``. Call outside ``jit``."""
        save_png(self.img, path, scale=scale)
        return self
