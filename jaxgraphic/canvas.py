"""``Canvas``: an immutable, chainable wrapper around an image."""
from __future__ import annotations

from dataclasses import dataclass

from jax import tree_util

from ._types import Array, ArrayLike, Color, Image, Point, Scalar
from .convert import to_hwc
from .images import draw_image
from .paint import Window, fill, fill_many, linear_gradient, stroke
from .png import save_png
from .raster import new_canvas
from .shapes import Shape, ShapeLike


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Canvas:
    """An immutable, chainable canvas that works under ``jit`` and ``vmap``.

    Every method returns a *new* ``Canvas``. A ``Canvas`` is a pytree with one leaf (the
    image), so it can be an argument or result of transformed functions.

    Example:
        >>> @jax.jit
        ... def render(t):
        ...     return (jg.Canvas.new(200, 200, "#fff")
        ...         .fill(jg.Circle((40 + 120 * t, 100), 30), "#e63946")
        ...         .stroke(jg.Box((100, 100), (160, 160)), "#1d3557", width=3))
        >>> render(0.5).save("out.png")      # save outside jit

    Wrap the whole chain in one ``jit``: XLA then fuses consecutive fills into a single
    pass over the image.

    Attributes:
        img: ``float32[3, H, W]`` (a tracer under transformations).
    """

    img: Image

    # -- construction -------------------------------------------------------
    @classmethod
    def new(cls, height: int, width: int, bg: Color = "#ffffff") -> "Canvas":
        """A blank canvas filled with ``bg`` (``height`` and ``width`` are static ints)."""
        return cls(new_canvas(height, width, bg))

    @classmethod
    def from_image(cls, img: Image) -> "Canvas":
        """Wrap an existing ``(3, H, W)`` image."""
        return cls(img)

    # -- pytree protocol ----------------------------------------------------
    def tree_flatten(self):
        return (self.img,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])      # no validation: JAX may pass placeholders while tracing

    # -- inspection ---------------------------------------------------------
    @property
    def height(self) -> int:
        return self.img.shape[-2]

    @property
    def width(self) -> int:
        return self.img.shape[-1]

    @property
    def hwc(self) -> Array:
        """The image as ``(H, W, 3)`` (for example for ``matplotlib.imshow``)."""
        return to_hwc(self.img)

    # -- drawing (each returns a new Canvas) --------------------------------
    def fill(self, shape: ShapeLike, color: Color, *, alpha: Scalar = 1.0,
             softness: Scalar = 1.0, at: Point | None = None,
             window: Window | None = None) -> "Canvas":
        """See :func:`jaxgraphic.fill`."""
        return Canvas(fill(self.img, shape, color, alpha, softness, at=at, window=window))

    def stroke(self, shape: ShapeLike, color: Color, *, width: Scalar = 2.0,
               alpha: Scalar = 1.0, softness: Scalar = 1.0, at: Point | None = None,
               window: Window | None = None) -> "Canvas":
        """See :func:`jaxgraphic.stroke`."""
        return Canvas(stroke(self.img, shape, color, width, alpha, softness, at=at, window=window))

    def gradient(self, shape: ShapeLike, start: Point, end: Point, color0: Color, color1: Color, *,
                 alpha: Scalar = 1.0, softness: Scalar = 1.0) -> "Canvas":
        """See :func:`jaxgraphic.linear_gradient`."""
        return Canvas(linear_gradient(self.img, shape, start, end, color0, color1, alpha, softness))

    def fill_many(self, shapes: Shape, colors, alphas: ArrayLike = 1.0, *,
                  softness: Scalar = 1.0, at: ArrayLike | None = None,
                  window: Window | None = None) -> "Canvas":
        """See :func:`jaxgraphic.fill_many`."""
        return Canvas(fill_many(self.img, shapes, colors, alphas, softness, at=at, window=window))

    def image(self, src: ArrayLike, center: Point, *, size: Point | None = None,
              scale: Scalar = 1.0, angle: Scalar | None = None, alpha: Scalar = 1.0,
              clip: ShapeLike | None = None, softness: Scalar = 1.0) -> "Canvas":
        """See :func:`jaxgraphic.draw_image`."""
        return Canvas(draw_image(self.img, src, center, size, scale, angle, alpha, clip, softness))

    def pipe(self, fn, *args, **kwargs) -> "Canvas":
        """Apply ``fn(img, *args, **kwargs) -> img`` and keep chaining."""
        return Canvas(fn(self.img, *args, **kwargs))

    # -- output (not jittable: it writes a file) ----------------------------
    def save(self, path: str, scale: int = 1) -> "Canvas":
        """Write a PNG and return ``self``. Call outside ``jit``."""
        save_png(self.img, path, scale=scale)
        return self
