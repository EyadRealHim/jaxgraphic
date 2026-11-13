"""jaxgraphic: fast 2D drawing on JAX.

Draw shapes and images with anti-aliasing. Shapes are pytrees, so a whole batch of frames
renders in one compiled call with plain ``jax.vmap``. Shapes are signed-distance-function pytrees; images are planar
``(C, H, W)`` float arrays.

    import jax, jax.numpy as jnp
    import jaxgraphic as jg

    def scene(t):                                   # one frame
        return (jg.Canvas.new(256, 256, "#ffffff")
                .fill(jg.Circle((40 + 170 * t, 128), 40), "#e63946")
                .stroke(jg.Box((128, 128), (200, 200), angle=t), "#1d3557", width=4))

    frames = jax.jit(jax.vmap(lambda t: jg.to_uint8(scene(t).img)))
    video = frames(jnp.linspace(0.0, 1.0, 240))     # (240, 3, 256, 256) uint8

Conventions: coordinates are pixels, x right and y *down*; colors are ``"#rrggbb"``
strings or ``(r, g, b)`` in ``[0, 1]``. See the README for the API and speed guide.
"""
from ._types import Array, ArrayLike, Color, Image, Point, Scalar  # noqa: F401
from .canvas import Canvas
from .convert import from_hwc, to_hwc, to_uint8
from .images import draw_image
from .paint import fill, fill_many, linear_gradient, stroke
from .png import load_png, save_png
from .raster import coords, coverage, new_canvas, rgb
from .shapes import (Box, Circle, Ellipse, FnShape, Intersect, Outline, Polygon,
                     RegularPolygon, Rotate, Scale, Segment, Shape, ShapeLike,
                     SmoothUnion, Star, Subtract, Translate, Triangle, Union,
                     as_shape)

__all__ = [
    # types
    "Array", "ArrayLike", "Color", "Image", "Point", "Scalar", "ShapeLike",
    # canvas & painting
    "Canvas", "draw_image", "fill", "fill_many", "linear_gradient", "stroke",
    "coords", "coverage", "new_canvas", "rgb",
    # conversion & PNG I/O
    "from_hwc", "to_hwc", "to_uint8", "load_png", "save_png",
    # shapes
    "Shape", "FnShape", "as_shape", "Box", "Circle", "Ellipse", "Polygon",
    "RegularPolygon", "Segment", "Star", "Triangle",
    # combinators
    "Intersect", "Outline", "Rotate", "Scale", "SmoothUnion", "Subtract",
    "Translate", "Union",
]
__version__ = "0.3.0"
