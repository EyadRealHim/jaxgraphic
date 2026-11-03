"""Canvas creation and compositing.

Images are ``float32`` arrays of shape ``(H, W, 3)`` with values in ``[0, 1]``.
All functions are pure (they return a new image rather than mutating), so they
compose with ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

Typical use:

    img = jg.new_canvas(256, 256, "#ffffff")
    img = jg.fill(img, jg.circle((128, 128), 60), "#e63946")
    img = jg.stroke(img, jg.circle((128, 128), 60), "#000000", width=3)
"""

from __future__ import annotations

import jax.numpy as jnp

from ._types import Array, Color, Image, Point, Scalar, Shape


def rgb(color: Color) -> Array:
    """Convert a color specification to a float32 array of shape ``(3,)``.

    Args:
        color: ``"#rrggbb"`` or ``"#rgb"`` hex string, or an array-like
            ``(r, g, b)`` with channels in ``[0, 1]`` (may be traced).

    Returns:
        The color as ``float32[3]``.

    Raises:
        ValueError: If a string is not a valid 3- or 6-digit hex color.

    Example:
        >>> rgb("#f80")          # same as "#ff8800"
        >>> rgb((1.0, 0.5, 0.0))
    """
    if isinstance(color, str):
        h = color.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) != 6:
            raise ValueError(f"expected '#rgb' or '#rrggbb', got {color!r}")
        return jnp.asarray(
            [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=jnp.float32
        )
    return jnp.asarray(color, dtype=jnp.float32)


def new_canvas(height: int, width: int, bg: Color = "#ffffff") -> Image:
    """Create a solid-color image.

    Args:
        height: Image height in pixels (static Python ``int``).
        width: Image width in pixels (static Python ``int``).
        bg: Background color (see :func:`rgb`). Default white.

    Returns:
        ``float32[height, width, 3]``.
    """
    return jnp.broadcast_to(rgb(bg), (height, width, 3))


def pixel_grid(height: int, width: int) -> Array:
    """Pixel-center coordinates for an image.

    Returns:
        ``float32[height, width, 2]`` where ``[y, x]`` holds ``(x + 0.5, y + 0.5)``.
        This is the ``p`` that shapes are evaluated on.
    """
    ys, xs = jnp.meshgrid(
        jnp.arange(height) + 0.5, jnp.arange(width) + 0.5, indexing="ij"
    )
    return jnp.stack([xs, ys], axis=-1)


def coverage(sd: Array, softness: Scalar = 1.0) -> Array:
    """Convert signed distance to anti-aliased coverage in ``[0, 1]``.

    Coverage is ``clip(0.5 - sd / softness, 0, 1)``: a linear ramp across the
    edge, ``softness`` pixels wide.

    Args:
        sd: Signed distances (negative inside).
        softness: Edge width in pixels. ``1.0`` gives crisp anti-aliasing.
            Larger values blur the edge, which lets gradients reach further
            from the shape. That is useful for optimization; anneal it down
            toward ``1.0`` as the fit improves. Must be positive.
    """
    return jnp.clip(0.5 - sd / softness, 0.0, 1.0)


def fill(
    img: Image, shape: Shape, color: Color, alpha: Scalar = 1.0, softness: Scalar = 1.0
) -> Image:
    """Paint the inside of ``shape`` over ``img`` (source-over blending).

    Args:
        img: Image to draw on. Not modified.
        shape: A shape (see :mod:`jaxgraphic.shapes`).
        color: Fill color (see :func:`rgb`).
        alpha: Overall opacity in ``[0, 1]``.
        softness: Edge width in pixels (see :func:`coverage`).

    Returns:
        A new image with the shape composited on top.
    """
    h, w = img.shape[:2]
    a = coverage(shape(pixel_grid(h, w)), softness)[..., None] * alpha
    return img * (1.0 - a) + rgb(color) * a


def stroke(
    img: Image,
    shape: Shape,
    color: Color,
    width: Scalar = 2.0,
    alpha: Scalar = 1.0,
    softness: Scalar = 1.0,
) -> Image:
    """Paint the outline of ``shape``: a band ``width`` pixels wide centered on its edge.

    Args:
        img: Image to draw on. Not modified.
        shape: A shape.
        color: Stroke color.
        width: Stroke thickness in pixels.
        alpha: Overall opacity in ``[0, 1]``.
        softness: Edge width in pixels (see :func:`coverage`).

    Returns:
        A new image with the outline composited on top.
    """
    band: Shape = lambda p: jnp.abs(shape(p)) - width / 2.0
    return fill(img, band, color, alpha, softness)


def linear_gradient(
    img: Image,
    shape: Shape,
    start: Point,
    end: Point,
    c0: Color,
    c1: Color,
    alpha: Scalar = 1.0,
    softness: Scalar = 1.0,
) -> Image:
    """Fill ``shape`` with a linear color gradient.

    The color is ``c0`` at ``start``, ``c1`` at ``end``, interpolated linearly
    along the line between them and held constant beyond the ends.

    Args:
        img: Image to draw on. Not modified.
        shape: A shape.
        start: ``(x, y)`` pixel position where the gradient is ``c0``.
        end: ``(x, y)`` pixel position where the gradient is ``c1``.
        c0: Color at ``start``.
        c1: Color at ``end``.
        alpha: Overall opacity in ``[0, 1]``.
        softness: Edge width in pixels (see :func:`coverage`).

    Example:
        A full-canvas sky gradient::

            h, w = img.shape[:2]
            img = jg.linear_gradient(img, jg.box((w/2, h/2), (w, h)),
                                     (0, 0), (0, h), "#ffb86b", "#5ec3ff")
    """
    h, w = img.shape[:2]
    p = pixel_grid(h, w)
    s, e = jnp.asarray(start, jnp.float32), jnp.asarray(end, jnp.float32)
    d = e - s
    t = jnp.clip(jnp.sum((p - s) * d, -1) / (jnp.sum(d * d) + 1e-9), 0.0, 1.0)
    col = rgb(c0) * (1 - t[..., None]) + rgb(c1) * t[..., None]
    a = coverage(shape(p), softness)[..., None] * alpha
    return img * (1.0 - a) + col * a


class Canvas:
    """A stateful, chainable wrapper around the functional drawing API.

    Convenient for scripts and notebooks. It holds a mutable ``img`` attribute,
    so do not use it inside ``jit``/``grad``-transformed code; use the pure
    functions (:func:`fill`, :func:`stroke`, ...) there instead.

    Example:
        >>> (jg.Canvas(200, 200, "#fff")
        ...     .fill(jg.circle((100, 100), 50), "#e63946")
        ...     .stroke(jg.box((100, 100), (120, 120)), "#1d3557", width=3)
        ...     .save("out.png"))

    Attributes:
        img: The current image, ``float32[H, W, 3]``.
    """

    img: Image

    def __init__(self, height: int, width: int, bg: Color = "#ffffff") -> None:
        """Create a canvas filled with the background color ``bg``."""
        self.img = new_canvas(height, width, bg)

    def fill(
        self, shape: Shape, color: Color, *, alpha: Scalar = 1.0, softness: Scalar = 1.0
    ) -> Canvas:
        """Fill ``shape``; see :func:`fill`. Returns ``self`` for chaining."""
        self.img = fill(self.img, shape, color, alpha, softness)
        return self

    def stroke(
        self,
        shape: Shape,
        color: Color,
        *,
        width: Scalar = 2.0,
        alpha: Scalar = 1.0,
        softness: Scalar = 1.0,
    ) -> Canvas:
        """Stroke ``shape``'s outline; see :func:`stroke`. Returns ``self``."""
        self.img = stroke(self.img, shape, color, width, alpha, softness)
        return self

    def gradient(
        self,
        shape: Shape,
        start: Point,
        end: Point,
        c0: Color,
        c1: Color,
        *,
        alpha: Scalar = 1.0,
        softness: Scalar = 1.0,
    ) -> Canvas:
        """Fill ``shape`` with a linear gradient; see :func:`linear_gradient`. Returns ``self``."""
        self.img = linear_gradient(self.img, shape, start, end, c0, c1, alpha, softness)
        return self

    def save(self, path: str, scale: int = 1) -> Canvas:
        """Write the image to a PNG file; see :func:`jaxgraphic.save_png`. Returns ``self``."""
        from .io import save_png

        save_png(self.img, path, scale=scale)
        return self
