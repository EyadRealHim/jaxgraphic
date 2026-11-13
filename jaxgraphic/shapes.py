"""Shapes: signed distance functions packaged as JAX pytrees.

A shape evaluates ``sdf(x, y) -> d``: negative inside, positive outside, zero on
the edge. ``x`` and ``y`` are *separate* coordinate arrays that only need to be
broadcast-compatible. The renderer passes ``x`` as ``(1, W)`` and ``y`` as
``(H, 1)``, so work that depends on one axis alone (``(x - cx)**2``, ``abs(y)``,
...) costs O(W) or O(H), and only the final combination costs O(H*W). Rotation
mixes the axes, so it falls back to full arrays; leave ``angle`` as ``None`` for
axis-aligned shapes to keep the cheap separable path.

Every shape is a registered pytree whose leaves are its numeric parameters, so:

* jitted rendering is cached by shape *structure*: changing a radius or a position
  never recompiles;
* ``jax.vmap`` and ``lax.scan`` work over *batches* of shapes
  (``Circle(centers, radii)`` with ``centers: (N, 2)``), which is how ``fill_many`` and
  a vmapped scene render many shapes and frames in one program.

Shapes combine with operators (``a | b`` union, ``a & b`` intersection,
``a - b`` difference) and methods (``translate``, ``rotate``, ``scale``,
``outline``, ``smooth_union``).

Custom shapes: subclass :class:`Shape`, list parameter names in ``_leaves``
(numeric, traced) and ``_statics`` (hashable, compile-time), and write ``sdf``::

    class HalfPlane(jg.Shape):
        _leaves = ("y0",)
        def __init__(self, y0): self.y0 = jnp.asarray(y0, jnp.float32)
        def sdf(self, x, y): return y - self.y0      # everything above y = y0

A plain function ``f(x, y) -> d`` is also accepted anywhere a shape is expected
(it is wrapped in :class:`FnShape`); it is compiled per function object, so
prefer a subclass when parameters change.
"""
from __future__ import annotations

import math
from functools import reduce
from typing import Callable, ClassVar

import jax.numpy as jnp
from jax import lax, tree_util

from ._types import Array, ArrayLike, Point, Scalar
from ._util import EPS, F32, as_f32, as_f32_or_none, safe_hypot

_UNROLL_MAX = 16      # polygons up to this many edges are unrolled; bigger ones use lax.scan


def local_coords(x: Array, y: Array, center: Array, angle: Array | None) -> tuple[Array, Array]:
    """Pixel coordinates relative to ``center``, rotated by ``-angle`` (skipped when ``None``).

    With ``angle=None`` the result stays separable ((1, W) and (H, 1)); a rotation mixes
    the axes and produces full (H, W) arrays.
    """
    dx = x - center[0]
    dy = y - center[1]
    if angle is None:
        return dx, dy
    c, s = jnp.cos(angle), jnp.sin(angle)
    return c * dx + s * dy, c * dy - s * dx


def rect_sdf(qx: Array, qy: Array, half_x: Array, half_y: Array) -> Array:
    """Signed distance to an origin-centered axis-aligned rectangle with the given half-extents."""
    ax, ay = jnp.abs(qx) - half_x, jnp.abs(qy) - half_y
    outside = safe_hypot(jnp.maximum(ax, 0.0), jnp.maximum(ay, 0.0))
    inside = jnp.minimum(jnp.maximum(ax, ay), 0.0)
    return outside + inside


class Shape:
    """Base class for shapes. See the module docstring for how to subclass."""

    _leaves: ClassVar[tuple[str, ...]] = ()    # numeric parameters (pytree leaves)
    _statics: ClassVar[tuple[str, ...]] = ()   # hashable compile-time parameters

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        leaves, statics = cls._leaves, cls._statics

        def flatten(obj):
            return (tuple(getattr(obj, n) for n in leaves),
                    tuple(getattr(obj, n) for n in statics))

        def unflatten(aux, children):
            # Bypass __init__: JAX may pass placeholders or tracers as children.
            obj = object.__new__(cls)
            for n, v in zip(leaves, children):
                setattr(obj, n, v)
            for n, v in zip(statics, aux):
                setattr(obj, n, v)
            return obj

        tree_util.register_pytree_node(cls, flatten, unflatten)

    def sdf(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array, y: Array) -> Array:
        return self.sdf(x, y)

    def __repr__(self) -> str:
        names = type(self)._leaves + type(self)._statics
        return f"{type(self).__name__}({', '.join(f'{n}={getattr(self, n)!r}' for n in names)})"

    # -- combinators -------------------------------------------------------
    def __or__(self, other: "ShapeLike") -> "Shape":
        return Union(self, other)

    def __and__(self, other: "ShapeLike") -> "Shape":
        return Intersect(self, other)

    def __sub__(self, other: "ShapeLike") -> "Shape":
        return Subtract(self, other)

    def translate(self, offset: Point) -> "Shape":
        """Move by ``(dx, dy)`` pixels."""
        return Translate(self, offset)

    def rotate(self, angle: Scalar, about: Point = (0.0, 0.0)) -> "Shape":
        """Rotate by ``angle`` radians about ``about`` (clockwise on screen, y is down)."""
        return Rotate(self, angle, about)

    def scale(self, s: Scalar) -> "Shape":
        """Scale by ``s`` about the origin ``(0, 0)``."""
        return Scale(self, s)

    def outline(self, width: Scalar = 2.0) -> "Shape":
        """A band ``width`` pixels wide on the edge (what ``stroke`` paints)."""
        return Outline(self, width)

    def smooth_union(self, other: "ShapeLike", k: Scalar = 10.0) -> "Shape":
        """Union with a rounded blend ``k`` pixels wide."""
        return SmoothUnion(self, other, k)


ShapeLike = Shape | Callable[[Array, Array], Array]


class FnShape(Shape):
    """Wraps a plain function ``f(x, y) -> d`` as a shape (compiled per function object)."""

    _statics = ("fn",)

    def __init__(self, fn: Callable[[Array, Array], Array]):
        self.fn = fn

    def sdf(self, x, y):
        return self.fn(x, y)


def as_shape(s: ShapeLike) -> Shape:
    """Return ``s`` if it is a :class:`Shape`, wrap a callable, else raise ``TypeError``."""
    if isinstance(s, Shape):
        return s
    if callable(s):
        return FnShape(s)
    raise TypeError(f"expected a Shape or a callable (x, y) -> distance, got {type(s).__name__}")


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
class Circle(Shape):
    """A disc. ``Circle((x, y), radius)``."""

    _leaves = ("center", "radius")

    def __init__(self, center: Point, radius: Scalar):
        self.center, self.radius = as_f32(center), as_f32(radius)

    def sdf(self, x, y):
        return safe_hypot(x - self.center[0], y - self.center[1]) - self.radius


class Ellipse(Shape):
    """An ellipse with semi-axes ``radii=(rx, ry)``; ``angle=None`` skips rotation.

    The distance is exact on the boundary and approximate elsewhere.
    """

    _leaves = ("center", "radii", "angle")

    def __init__(self, center: Point, radii: Point, angle: Scalar | None = None):
        self.center, self.radii, self.angle = as_f32(center), as_f32(radii), as_f32_or_none(angle)

    def sdf(self, x, y):
        qx, qy = local_coords(x, y, self.center, self.angle)
        irx, iry = 1.0 / self.radii[0], 1.0 / self.radii[1]
        ux, uy = qx * irx, qy * iry                    # q / r
        k0 = safe_hypot(ux, uy)
        vx, vy = ux * irx, uy * iry                    # q / r^2
        return k0 * (k0 - 1.0) * lax.rsqrt(vx * vx + vy * vy + EPS)


class Box(Shape):
    """A rectangle ``Box(center, (w, h), angle=None, corner=None)``.

    ``corner`` is an optional corner radius in pixels (keep it <= min(w, h) / 2).
    Leave ``angle`` as ``None`` for axis-aligned boxes: that keeps the cheap
    separable path (most of the work is O(W + H) instead of O(W * H)).
    """

    _leaves = ("center", "size", "angle", "corner")

    def __init__(self, center: Point, size: Point, angle: Scalar | None = None,
                 corner: Scalar | None = None):
        self.center, self.size = as_f32(center), as_f32(size)
        self.angle, self.corner = as_f32_or_none(angle), as_f32_or_none(corner)

    def sdf(self, x, y):
        qx, qy = local_coords(x, y, self.center, self.angle)
        hx, hy = self.size[0] * 0.5, self.size[1] * 0.5
        r = self.corner
        if r is not None:
            hx, hy = hx - r, hy - r
        sd = rect_sdf(qx, qy, hx, hy)
        return sd if r is None else sd - r


class Segment(Shape):
    """A line segment from ``a`` to ``b`` with round caps, ``width`` pixels thick."""

    _leaves = ("a", "b", "width")

    def __init__(self, a: Point, b: Point, width: Scalar = 1.0):
        self.a, self.b, self.width = as_f32(a), as_f32(b), as_f32(width)

    def sdf(self, x, y):
        pax, pay = x - self.a[0], y - self.a[1]
        bax, bay = self.b[0] - self.a[0], self.b[1] - self.a[1]
        inv = 1.0 / (bax * bax + bay * bay + EPS)
        h = jnp.clip((pax * bax + pay * bay) * inv, 0.0, 1.0)
        return safe_hypot(pax - bax * h, pay - bay * h) - self.width * 0.5


def _polygon_sdf(x: Array, y: Array, pts: Array) -> Array:
    """Signed distance to a closed simple polygon with vertices ``pts (N, 2)``.

    Folds over the edges keeping only a running min-distance and an inside/outside
    parity, so memory is O(H*W) rather than O(H*W*N). Per-edge constants (including
    the reciprocal edge length) are computed once, outside the per-pixel work.
    Small polygons are unrolled (XLA fuses them into one kernel); big ones use
    ``lax.scan`` so compile time stays flat.
    """
    n = pts.shape[0]
    vx, vy = pts[:, 0], pts[:, 1]
    nvy = jnp.roll(vy, -1)
    ex, ey = jnp.roll(vx, -1) - vx, nvy - vy
    inv = 1.0 / (ex * ex + ey * ey + EPS)
    consts = (vx, vy, nvy, ex, ey, inv)

    def step(state, c):
        d2, inside = state
        px, py, qy, ex_, ey_, inv_ = c
        wx, wy = x - px, y - py
        t = jnp.clip((wx * ex_ + wy * ey_) * inv_, 0.0, 1.0)
        bx, by = wx - ex_ * t, wy - ey_ * t
        d2 = jnp.minimum(d2, bx * bx + by * by)
        c1, c2, c3 = y >= py, y < qy, ex_ * wy > ey_ * wx
        flip = (c1 & c2 & c3) | ~(c1 | c2 | c3)      # crossing parity (winding number)
        return d2, jnp.logical_xor(inside, flip)

    if n <= _UNROLL_MAX:
        state = (jnp.asarray(3e38, F32), jnp.zeros((), dtype=bool))
        for i in range(n):
            state = step(state, tuple(a[i] for a in consts))
    else:
        full = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
        init = (jnp.full(full, 3e38, dtype=F32), jnp.zeros(full, dtype=bool))
        state, _ = lax.scan(lambda s, c: (step(s, c), None), init, consts)
    d2, inside = state
    return jnp.where(inside, -1.0, 1.0) * jnp.sqrt(d2 + EPS)


class Polygon(Shape):
    """A filled simple polygon from an ``(N, 2)`` array of vertices (closed automatically)."""

    _leaves = ("points",)

    def __init__(self, points: ArrayLike):
        self.points = as_f32(points)

    def sdf(self, x, y):
        return _polygon_sdf(x, y, self.points)


class Triangle(Shape):
    """A triangle ``Triangle(a, b, c)``."""

    _leaves = ("points",)

    def __init__(self, a: Point, b: Point, c: Point):
        self.points = jnp.stack([as_f32(a), as_f32(b), as_f32(c)])

    def sdf(self, x, y):
        return _polygon_sdf(x, y, self.points)


class RegularPolygon(Shape):
    """A regular ``sides``-gon inscribed in a circle of ``radius`` (``sides`` is static)."""

    _leaves = ("center", "radius", "angle")
    _statics = ("sides",)

    def __init__(self, center: Point, radius: Scalar, sides: int, angle: Scalar = 0.0):
        self.center, self.radius, self.angle = as_f32(center), as_f32(radius), as_f32(angle)
        self.sides = int(sides)

    def sdf(self, x, y):
        k = jnp.arange(self.sides, dtype=F32)
        th = self.angle + (2.0 * math.pi / self.sides) * k
        pts = jnp.stack([self.center[0] + self.radius * jnp.cos(th),
                         self.center[1] + self.radius * jnp.sin(th)], axis=-1)
        return _polygon_sdf(x, y, pts)


class Star(Shape):
    """A star with ``points`` tips (static). The default angle points one tip straight up."""

    _leaves = ("center", "outer", "inner", "angle")
    _statics = ("points",)

    def __init__(self, center: Point, outer: Scalar, inner: Scalar, points: int = 5,
                 angle: Scalar = -math.pi / 2):
        self.center, self.outer, self.inner = as_f32(center), as_f32(outer), as_f32(inner)
        self.angle, self.points = as_f32(angle), int(points)

    def sdf(self, x, y):
        k = jnp.arange(2 * self.points, dtype=F32)
        r = jnp.where(k % 2 == 0, self.outer, self.inner)
        th = self.angle + (math.pi / self.points) * k
        pts = jnp.stack([self.center[0] + r * jnp.cos(th),
                         self.center[1] + r * jnp.sin(th)], axis=-1)
        return _polygon_sdf(x, y, pts)


# --------------------------------------------------------------------------
# Combinators
# --------------------------------------------------------------------------
class Union(Shape):
    """Region covered by any of the shapes. ``Union(a, b, c)`` or ``a | b | c``."""

    _leaves = ("shapes",)

    def __init__(self, *shapes: ShapeLike):
        self.shapes = tuple(as_shape(s) for s in shapes)

    def sdf(self, x, y):
        return reduce(jnp.minimum, [s.sdf(x, y) for s in self.shapes])


class Intersect(Shape):
    """Region covered by all of the shapes. ``Intersect(a, b)`` or ``a & b``."""

    _leaves = ("shapes",)

    def __init__(self, *shapes: ShapeLike):
        self.shapes = tuple(as_shape(s) for s in shapes)

    def sdf(self, x, y):
        return reduce(jnp.maximum, [s.sdf(x, y) for s in self.shapes])


class Subtract(Shape):
    """``a`` with ``b`` cut out. ``Subtract(a, b)`` or ``a - b``."""

    _leaves = ("a", "b")

    def __init__(self, a: ShapeLike, b: ShapeLike):
        self.a, self.b = as_shape(a), as_shape(b)

    def sdf(self, x, y):
        return jnp.maximum(self.a.sdf(x, y), -self.b.sdf(x, y))


class SmoothUnion(Shape):
    """Union of ``a`` and ``b`` with a rounded blend of width ``k`` (> 0) pixels."""

    _leaves = ("a", "b", "k")

    def __init__(self, a: ShapeLike, b: ShapeLike, k: Scalar = 10.0):
        self.a, self.b, self.k = as_shape(a), as_shape(b), as_f32(k)

    def sdf(self, x, y):
        da, db = self.a.sdf(x, y), self.b.sdf(x, y)
        h = jnp.clip(0.5 + 0.5 * (db - da) * (1.0 / self.k), 0.0, 1.0)
        return db + h * (da - db) - self.k * h * (1.0 - h)


class Outline(Shape):
    """A band ``width`` pixels wide centered on the edge of ``shape``."""

    _leaves = ("shape", "width")

    def __init__(self, shape: ShapeLike, width: Scalar = 2.0):
        self.shape, self.width = as_shape(shape), as_f32(width)

    def sdf(self, x, y):
        return jnp.abs(self.shape.sdf(x, y)) - self.width * 0.5


class Translate(Shape):
    """``shape`` moved by ``offset = (dx, dy)``."""

    _leaves = ("shape", "offset")

    def __init__(self, shape: ShapeLike, offset: Point):
        self.shape, self.offset = as_shape(shape), as_f32(offset)

    def sdf(self, x, y):
        return self.shape.sdf(x - self.offset[0], y - self.offset[1])


class Rotate(Shape):
    """``shape`` rotated by ``angle`` radians about ``about`` (clockwise on screen)."""

    _leaves = ("shape", "angle", "about")

    def __init__(self, shape: ShapeLike, angle: Scalar, about: Point = (0.0, 0.0)):
        self.shape, self.angle, self.about = as_shape(shape), as_f32(angle), as_f32(about)

    def sdf(self, x, y):
        c, s = jnp.cos(self.angle), jnp.sin(self.angle)
        dx, dy = x - self.about[0], y - self.about[1]
        return self.shape.sdf(c * dx + s * dy + self.about[0], c * dy - s * dx + self.about[1])


class Scale(Shape):
    """``shape`` scaled by ``s`` about the origin."""

    _leaves = ("shape", "s")

    def __init__(self, shape: ShapeLike, s: Scalar):
        self.shape, self.s = as_shape(shape), as_f32(s)

    def sdf(self, x, y):
        inv = 1.0 / self.s
        return self.shape.sdf(x * inv, y * inv) * self.s

