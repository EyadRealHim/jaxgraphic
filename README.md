# jaxgraphic

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EyadRealHim/jaxgraphic/blob/main/demo.ipynb)

Fast 2D drawing on [JAX](https://github.com/jax-ml/jax): anti-aliased shapes and images
rendered as planar `float32` arrays. Shapes are signed distance functions packaged as pytrees,
so a whole scene compiles into one program and a batch of frames renders in one call.<sup>†</sup>

## Quick start

An image is just an array, and drawing is a function: it takes an image and returns a new one.

```python
import jaxgraphic as jg

img = jg.new_canvas(256, 256, "#0b1020")  # (3, 256, 256) float32 image
img = jg.fill(img, jg.Circle((128, 128), 80), "#ff5cc8")  # paint a shape onto it
jg.save_png(img, "circle.png")
```

Combine shapes, then paint fills and outlines:

```python
img = jg.new_canvas(256, 256, "#ffffff")
img = jg.fill(
    img, jg.Circle((100, 128), 70) | jg.Circle((156, 128), 70), "#e63946"
)  # union of two circles
img = jg.stroke(
    img, jg.Star((128, 128), 110, 45, points=5), "#1d3557", width=4
)  # star outline
jg.save_png(img, "shapes.png")
```

`Canvas` wraps the same operations so calls chain. It holds an image and every method returns a new `Canvas`:

```python
canvas = (
    jg.Canvas.new(256, 256, "#ffffff")
    .fill(jg.Circle((100, 128), 70) | jg.Circle((156, 128), 70), "#e63946")
    .stroke(jg.Star((128, 128), 110, 45, points=5), "#1d3557", width=4)
)
canvas.save("shapes.png")
```

Compile with `jax.jit`. Anything that changes (positions, radii, colors) is an argument:

```python
import jax


@jax.jit
def render(x):
    return jg.fill(jg.new_canvas(256, 256), jg.Circle((x, 128), 40), "#e63946")


img = render(100.0)  # (3, 256, 256) float32
```

Animate: write a function for **one** frame, then render all of them in one call.

```python
import jax.numpy as jnp


def scene(t):  # t goes from 0 to 1
    return jg.fill(
        jg.new_canvas(256, 256), jg.Circle((40 + 180 * t, 128), 30), "#e63946"
    )


frame = lambda t: jg.to_uint8(
    scene(t)
)  # convert on the device: 4x fewer bytes to copy back
render = jax.jit(lambda ts: jax.lax.map(frame, ts, batch_size=32))  # 32 frames per step
video = render(jnp.linspace(0.0, 1.0, 120))  # (120, 3, 256, 256) uint8
```

Because images and shapes are plain arrays and pytrees, there is nothing special to learn: `jax.vmap(frame)`
renders every frame at once, and `lax.map(..., batch_size=n)` renders `n` at a time to bound memory.
Image size and shape *types* are fixed per compile; everything that changes between frames is an argument.
`save_png` and `Canvas.save` write a file, so call them outside `jit`.

## Performance tips

- **One `jit` per scene.** XLA fuses consecutive fills into a single pass over the image.
- **Leave `angle=None`** on `Box`, `Ellipse` and friends to keep the cheap separable path;
  `angle=0.0` takes the general rotated path.
- **Many shapes: `fill_many`, with `window=`.** A Python loop of `fill` calls is unrolled into the
  compiled program and touches the whole canvas every time. `fill_many` is one `lax.scan`, and with
  `window=(w, h)` each step touches only a small patch.
- **Don't change structure between frames.** New shape types, polygon vertex counts or image sizes
  trigger a recompile; changing numbers never does.
- **Prefer a `Shape` subclass to a bare function.** A function `f(x, y)` is compiled once per function object.
- **Downscale big source images** before `draw_image`; sampling is bilinear and does not low-pass filter.

## Concepts

**Images** are `float32` arrays in planar layout `(C, H, W)` (channels first), values in
`[0, 1]`. Use `jg.to_hwc(img)` / `jg.from_hwc(arr)` for matplotlib / PIL-style `(H, W, C)` data.

**Coordinates** are pixels, x to the right and y *down*, `(0, 0)` at the top-left. A pixel's
center is at `(x + 0.5, y + 0.5)`. Positive rotation angles appear clockwise on screen.

**Colors** are `"#rrggbb"` / `"#rgb"` strings or `(r, g, b)` array-likes in `[0, 1]`.

**Shapes** are pytrees whose leaves are their numeric parameters. A shape's `sdf(x, y)`
returns the signed distance (negative inside). Because shapes are pytrees, compiled code is
cached by shape *structure* (changing a radius never recompiles), and `jax.vmap` / `lax.scan`
work over *batches* of shapes (`Circle(centers, radii)` with `centers: (N, 2)`).

**`Canvas`** is an immutable, chainable pytree: every method returns a new `Canvas`, and it
passes through `jit` and `vmap`.

## Shapes

| Shape | Description |
|---|---|
| `Circle(center, radius)` | Disc. |
| `Ellipse(center, radii, angle=None)` | Ellipse (distance exact on the boundary, approximate elsewhere). |
| `Box(center, size, angle=None, corner=None)` | Rectangle, optionally rotated, with optional corner radius. |
| `Segment(a, b, width=1.0)` | Line segment with round caps. |
| `Triangle(a, b, c)` | Triangle. |
| `Polygon(points)` | Simple polygon from an `(N, 2)` array, convex or concave, closed automatically. |
| `RegularPolygon(center, radius, sides, angle=0.0)` | Regular n-gon; `sides` is static. |
| `Star(center, outer, inner, points=5, angle=-pi/2)` | Star; `points` is static; one tip up by default. |

Leave `angle` as `None` (the default) for axis-aligned shapes: that keeps the cheap
separable path (see *Performance*). `angle=0.0` takes the general rotated path.

**Combining** (all return shapes):

| Expression | Meaning |
|---|---|
| `a \| b`, `Union(a, b, c)` | Covered by any. |
| `a & b`, `Intersect(a, b)` | Covered by all. |
| `a - b`, `Subtract(a, b)` | `a` with `b` cut out. |
| `a.smooth_union(b, k=10)` | Union with a rounded blend of width `k` pixels. |
| `a.outline(width=2)` | Band of `width` pixels on the edge (what `stroke` paints). |
| `a.translate((dx, dy))` | Move. |
| `a.rotate(angle, about=(0, 0))` | Rotate about a point. |
| `a.scale(s)` | Scale about the origin. |

## Painting

| Function / method | Description |
|---|---|
| `fill(img, shape, color, alpha=1, softness=1, *, at=None, window=None)` / `.fill(...)` | Paint the inside of a shape. |
| `stroke(img, shape, color, width=2, ...)` / `.stroke(...)` | Paint an outline band centered on the edge. |
| `linear_gradient(img, shape, start, end, c0, c1, ...)` / `.gradient(...)` | Gradient fill. |
| `fill_many(img, shapes, colors, alphas=1, softness=1, *, at=None, window=None)` / `.fill_many(...)` | Paint N shapes with one `lax.scan`. |
| `draw_image(img, src, center, size=None, scale=1, angle=None, alpha=1, clip=None, ...)` / `.image(...)` | Composite an RGB/RGBA image. |
| `new_canvas(h, w, bg="#fff")`, `coords`, `coverage`, `rgb` | Building blocks. |
| `Canvas.new(h, w, bg="#fff")` | Immutable, chainable canvas; every method returns a new `Canvas`. A pytree, so it passes through `jit`/`vmap`. Also `.img`, `.hwc`, `.height`, `.width`, `.pipe(fn)`, `.save(path)`. |

`softness` is the anti-aliased edge width in pixels. Anything that takes a shape also
accepts a plain function `f(x, y) -> distance`.

## Drawing images

`draw_image` / `Canvas.image` composites a picture, placed by its **center** (scale and
rotation pivot there):

```python
photo = jg.load_png("photo.png")  # (C, H, W) float32; RGBA keeps transparency
# for JPEG etc.: photo = jg.from_hwc(np.asarray(PIL.Image.open("photo.jpg")))

(
    jg.Canvas.new(256, 256, "#fff")
    .image(photo, (128, 128), scale=0.5, angle=0.3)  # place, scale, rotate
    .image(
        photo, (60, 60), size=(80, 80), clip=jg.Circle((60, 60), 30)
    )  # stretch + circular clip
    .image(photo, (200, 200), alpha=0.5)
)  # translucent
```

Sources are channels-first `(C, H, W)` with C = 3 (RGB) or 4 (RGBA, straight alpha), or
`(H, W)` grayscale; float in `[0, 1]` or integer `0..255`. Sampling is bilinear with
premultiplied alpha, so there are no dark fringes at transparent edges. It does not
low-pass filter: shrinking by more than ~2x can alias, so downscale large images first.

## Many shapes

```python
shapes = jg.Circle(centers, radii)  # centers: (N, 2), radii: (N,)

canvas.fill_many(shapes, colors)  # one lax.scan: compile time independent of N
canvas.fill_many(
    shapes, colors, at=centers, window=(40, 40)
)  # each step touches only a 40x40 window
```

`window=(w, h)` is static and must contain each shape's visible extent (including the
`softness` fringe); the window is clamped to stay on the canvas, and `at` only chooses where
it goes. With a window the cost per shape is O(w * h) instead of O(H * W).

## Writing your own shape

Subclass `Shape`: numeric parameters go in `_leaves` (traced), compile-time values in
`_statics` (hashable), and `sdf(x, y)` returns the signed distance. `x` is `(1, W)` and `y`
is `(H, 1)` when rendering; use only broadcasting operations.

```python
class HalfPlane(jg.Shape):
    _leaves = ("y0",)

    def __init__(self, y0):
        self.y0 = jnp.asarray(y0, jnp.float32)

    def sdf(self, x, y):
        return y - self.y0  # negative above the line (y points down)


jg.Circle((128, 128), 80) & HalfPlane(128)
```

Do not store values derived from the leaves (like `cos(angle)`) as extra leaves: they would
go stale if a leaf changes. Compute them inside `sdf`.

A plain function `f(x, y)` also works anywhere a shape does (wrapped in `FnShape`), but it
is compiled once per function object, so prefer a subclass when parameters change.

## Image I/O

| Function | Description |
|---|---|
| `save_png(img, path, scale=1, compresslevel=6, filter="auto")` | Write an RGB or RGBA PNG. `filter="auto"` compresses the None/Sub/Up row filters and keeps the smallest. Call outside `jit`. |
| `load_png(path, keep_alpha=True)` | Read a PNG to `float32 (C, H, W)`. All color types and bit depths; not interlaced. Pure Python unfiltering makes big Average/Paeth-filtered files slow (a few seconds at 1000x1000). |
| `to_hwc(img)`, `from_hwc(arr)` | Convert between `(C, H, W)` and `(H, W, C)`. |
| `to_uint8(img)` | Float image to a clipped NumPy `uint8` array. |

## Design notes

Why the code is shaped the way it is:

- **Separable coordinates.** Shapes receive `x: (1, W)` and `y: (H, 1)` instead of a stacked
  `(H, W, 2)` array. Work on one axis (`(x - cx)**2`, `abs(y)`) costs O(W) or O(H); only the
  final combination is O(H * W). There are no stack/unstack ops and no size-2 minor dimension.
- **Planar `(C, H, W)` images.** The width axis stays innermost, so loops vectorize on CPU and
  there is no padded size-3 minor dimension on accelerators.
- **Lerp blending.** `img + a * (color - img)` instead of `img * (1 - a) + color * a`.
- **Reciprocals.** Divisions by per-shape or per-edge constants become multiplications by
  reciprocals computed once, off the per-pixel path.
- **Polygons** fold over the edges keeping a running minimum and a parity bit: O(H * W) memory
  instead of O(H * W * N). Small polygons are unrolled (one fused kernel); large ones use `lax.scan`.
- **`draw_image`** premultiplies and edge-pads the source once (O(source), not O(canvas)),
  gathers rows from one flat table instead of 2-D indexing, shares the rotated coordinates
  between the rectangle coverage and the texture lookup, and keeps coordinates separable when
  `angle=None`.

<sup>†</sup> **Differentiability.** The rasterizer is built from ordinary JAX operations, so
`jax.grad` works through it, but speed is the design goal and nothing here is tuned for
gradient descent.

## AI

Developed with significant assistance from AI tools
