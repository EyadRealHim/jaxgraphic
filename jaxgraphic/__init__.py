"""jaxgraphic: a tiny differentiable 2D drawing library built on JAX.

Shapes are signed distance functions; ``fill`` and ``stroke`` rasterize them
with anti-aliasing onto ``(H, W, 3)`` float images. Everything is pure, so it
works with ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

    import jaxgraphic as jg

    img = jg.new_canvas(256, 256, "#ffffff")
    img = jg.fill(img, jg.circle((128, 128), 60), "#e63946")
    img = jg.stroke(img, jg.box((128, 128), (140, 140), angle=0.4), "#1d3557", width=4)
    jg.save_png(img, "out.png")

Conventions: coordinates are pixels with x to the right and y *down*; colors are
``"#rrggbb"`` strings or ``(r, g, b)`` in ``[0, 1]``; images are ``float32``.
See the README for a full API reference.
"""

from ._types import (
    Array,
    ArrayLike,
    Color,
    Image,
    Point,
    Scalar,
    Shape,
)
from .core import (
    Canvas,
    coverage,
    fill,
    linear_gradient,
    new_canvas,
    pixel_grid,
    rgb,
    stroke,
)
from .io import save_png, to_uint8
from .shapes import (
    box,
    circle,
    ellipse,
    intersect,
    outline,
    polygon,
    regular_polygon,
    rotate,
    scale,
    segment,
    smooth_union,
    star,
    subtract,
    translate,
    triangle,
    union,
)

__all__ = [
    # types
    "Array",
    "ArrayLike",
    # canvas & painting
    "Canvas",
    "Color",
    "Image",
    "Point",
    "Scalar",
    "Shape",
    # shapes
    "box",
    "circle",
    "coverage",
    "ellipse",
    "fill",
    # operations
    "intersect",
    "linear_gradient",
    "new_canvas",
    "outline",
    "pixel_grid",
    "polygon",
    "regular_polygon",
    "rgb",
    "rotate",
    # output
    "save_png",
    "scale",
    "segment",
    "smooth_union",
    "star",
    "stroke",
    "subtract",
    "to_uint8",
    "translate",
    "triangle",
    "union",
]
__version__ = "0.1.0"
