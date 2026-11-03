# jaxgraphic

A tiny differentiable 2D drawing library on [JAX](https://github.com/jax-ml/jax).
Shapes are signed distance functions (SDFs) that are rasterized with
anti-aliasing, so drawing works with `jax.jit`, `jax.vmap` and `jax.grad`.
Fully type-annotated.

**Requires:** `jax` and `numpy`. Python 3.10+.

```python
import jaxgraphic as jg

img = jg.new_canvas(256, 256, "#ffffff")
img = jg.fill(img, jg.circle((128, 128), 60), "#e63946")
img = jg.stroke(img, jg.box((128, 128), (140, 140), angle=0.4), "#1d3557", width=4)
jg.save_png(img, "out.png")
```

## Concepts

**Images** are `float32` arrays of shape `(H, W, 3)` with values in `[0, 1]`.
All drawing functions are pure: they return a new image and never modify the one you pass in.

**Coordinates** are pixels, with x to the right and y *down*, and `(0, 0)` at the
top-left corner. A pixel's center is at `(x + 0.5, y + 0.5)`. Because y points
down, a positive rotation angle appears clockwise on screen.

**Colors** are `"#rrggbb"` or `"#rgb"` strings, or an array-like `(r, g, b)` with
channels in `[0, 1]`.

**Shapes** are functions `sdf(p) -> d`. `p` has shape `(..., 2)` (pixel coordinates)
and `d` has shape `(...)`: negative inside the shape, positive outside, zero on the edge.
`fill` and `stroke` evaluate the shape on every pixel and turn the distance into
anti-aliased coverage.

**Composing and differentiating.** Build shapes *inside* the function you
transform. Their parameters can be traced values:

```python
import jax, jax.numpy as jnp


def render(center, radius):
    img = jg.new_canvas(64, 64)
    return jg.fill(img, jg.circle(center, radius), "#000")


loss = lambda c, r: jnp.mean((render(c, r) - target) ** 2)
grads = jax.grad(loss, argnums=(0, 1))(jnp.array([30.0, 30.0]), 10.0)
```

`height`, `width`, and the `sides`/`points` counts of polygons and stars are
*static* (they set array shapes), so pass them as Python ints.

**Optimizing.** Gradients only exist near edges, so a shape far from its target
gets almost no signal. Pass a large `softness=` (e.g. 12) to `fill`/`stroke` to
widen the edge, then anneal it down to `1.0`. `demo_fit.py` shows this.

## API reference

Every public name is exported from the top-level `jaxgraphic` module. Types:
`Array`, `ArrayLike`, `Scalar`, `Point`, `Color`, `Image`, `Shape`.

### Canvas and painting

| Function | Description |
|---|---|
| `new_canvas(height, width, bg="#ffffff") -> Image` | Solid-color image. |
| `fill(img, shape, color, alpha=1.0, softness=1.0) -> Image` | Paint the inside of a shape. |
| `stroke(img, shape, color, width=2.0, alpha=1.0, softness=1.0) -> Image` | Paint an outline band `width` pixels wide, centered on the edge. |
| `linear_gradient(img, shape, start, end, c0, c1, alpha=1.0, softness=1.0) -> Image` | Fill with a gradient from `c0` at `start` to `c1` at `end`. |
| `coverage(sd, softness=1.0) -> Array` | Signed distance to coverage in `[0, 1]`. |
| `pixel_grid(height, width) -> Array` | `(H, W, 2)` pixel-center coordinates. |
| `rgb(color) -> Array` | Color spec to `float32[3]`. Raises `ValueError` on bad hex. |
| `Canvas(height, width, bg)` | Chainable, stateful wrapper with `.fill()`, `.stroke()`, `.gradient()`, `.save()`. Not for use under `jit`/`grad`. |

### Shapes

| Constructor | Description |
|---|---|
| `circle(center, radius)` | Disc. |
| `ellipse(center, radii, angle=0.0)` | Ellipse (distance exact on the boundary, approximate elsewhere). |
| `box(center, size, angle=0.0, corner=0.0)` | Rectangle, optionally rotated, with rounded corners of radius `corner`. |
| `segment(a, b, width=1.0)` | Line segment with round caps. |
| `triangle(a, b, c)` | Triangle. |
| `polygon(points)` | Simple polygon from an `(N, 2)` array; closed automatically. Memory is `O(H*W*N)`. |
| `regular_polygon(center, radius, sides, angle=0.0)` | Regular n-gon; `radius` is center-to-vertex. |
| `star(center, outer, inner, points=5, angle=-pi/2)` | Star; default has one tip pointing up. |

### Operations

| Function | Description |
|---|---|
| `union(*shapes)` / `intersect(*shapes)` | Covered by any / by all of the shapes. |
| `subtract(a, b)` | `a` with `b` cut out. |
| `smooth_union(a, b, k=10.0)` | Union with a rounded blend of width `k` pixels. |
| `outline(shape, width=2.0)` | Hollow band version of a shape (what `stroke` fills). |
| `translate(shape, offset)` | Move by `(dx, dy)`. |
| `rotate(shape, angle, about=(0, 0))` | Rotate by `angle` radians about a point. |
| `scale(shape, s)` | Scale by `s` about the origin. To scale about the shape's center, translate it to the origin first. |

### Output

| Function | Description |
|---|---|
| `save_png(img, path, scale=1)` | Write a PNG; `scale` is an integer nearest-neighbor upscale. Call outside `jit`. |
| `to_uint8(img)` | Float image to a clipped NumPy `uint8` array. |

## Writing your own shape

A shape is any function from pixel coordinates to signed distance, so you can add
your own and combine it with everything above:

```python
def half_plane(y0: jg.Scalar) -> jg.Shape:
    """Everything above the horizontal line y = y0 (remember: y points down)."""
    return lambda p: p[..., 1] - y0  # negative where y < y0, i.e. above the line


img = jg.fill(img, jg.intersect(jg.circle((128, 128), 80), half_plane(128)), "#2a9d8f")
```

The result is a true distance only if you return a distance; shapes that return
a different scale still draw, but strokes and `softness` are measured in
whatever units the function returns.

## Type checking

The package is fully annotated. Under a type checker, `jg.Shape` is
`Callable[[Array], Array]`, and shape parameters accept any `ArrayLike`
(Python numbers, NumPy arrays, JAX arrays, tracers).

## AI

Developed with significant assistance from AI tools