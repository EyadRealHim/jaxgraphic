"""Shapes as signed distance functions (SDFs).

A *shape* is a function ``sdf(p) -> d`` where ``p`` has shape ``(..., 2)`` and
holds ``(x, y)`` pixel coordinates (x right, y down), and ``d`` has shape
``(...)``. The distance is negative inside the shape, positive outside, and
zero on the boundary.

Every parameter (centers, radii, angles, ...) may be a traced JAX value, so
shapes work under ``jit``, ``vmap`` and ``grad``. Shape constructors are cheap;
build them inside the function you jit.

You can write your own shape: any callable with the signature above works
anywhere a shape is expected.

    def half_plane(y0):
        return lambda p: p[..., 1] - y0      # everything above the line y = y0
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from ._types import Array, ArrayLike, Point, Scalar, Shape

_EPS = 1e-9


def _f32(x: ArrayLike) -> Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _length(v: Array) -> Array:
    """Euclidean norm over the last axis, with a finite gradient at zero.

    ``jnp.linalg.norm`` has a NaN gradient at exactly ``v == 0``, which would
    poison optimization when a pixel center coincides with a shape center.
    """
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + _EPS)


def _rot(v: Array, angle: Scalar) -> Array:
    """Rotate vectors ``v`` of shape ``(..., 2)`` by ``angle`` radians.

    Standard rotation matrix: counter-clockwise in a y-up frame, which appears
    clockwise on screen where y points down.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    x, y = v[..., 0], v[..., 1]
    return jnp.stack([c * x - s * y, s * x + c * y], axis=-1)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def circle(center: Point, radius: Scalar) -> Shape:
    """A filled disc.

    Args:
        center: ``(x, y)`` center in pixels.
        radius: Radius in pixels.

    Returns:
        A shape. Example: ``jg.circle((128, 128), 40)``.
    """
    c = _f32(center)

    def sdf(p: Array) -> Array:
        return _length(p - c) - radius

    return sdf


def ellipse(center: Point, radii: Point, angle: Scalar = 0.0) -> Shape:
    """A filled ellipse.

    The distance is exact on the boundary and approximate elsewhere, which is
    fine for anti-aliasing and strokes of moderate width.

    Args:
        center: ``(x, y)`` center in pixels.
        radii: ``(rx, ry)`` semi-axes in pixels, before rotation.
        angle: Rotation in radians. Positive angles appear clockwise on
            screen because y points down.
    """
    c, r = _f32(center), _f32(radii)

    def sdf(p: Array) -> Array:
        q = _rot(p - c, -angle)
        k0 = _length(q / r)
        k1 = _length(q / (r * r))
        return k0 * (k0 - 1.0) / k1

    return sdf


def box(center: Point, size: Point, angle: Scalar = 0.0, corner: Scalar = 0.0) -> Shape:
    """A rectangle, optionally rotated and with rounded corners.

    Args:
        center: ``(x, y)`` center in pixels.
        size: ``(width, height)`` in pixels, before rotation.
        angle: Rotation about the center, in radians.
        corner: Corner radius in pixels. The outer size is unchanged; ``0``
            gives sharp corners. Keep it at most ``min(size) / 2``.

    Example:
        ``jg.box((100, 80), (120, 60), angle=0.3, corner=8)``
    """
    c, half = _f32(center), _f32(size) / 2.0

    def sdf(p: Array) -> Array:
        q = _rot(p - c, -angle)
        d = jnp.abs(q) - half + corner
        outside = _length(jnp.maximum(d, 0.0))
        inside = jnp.minimum(jnp.maximum(d[..., 0], d[..., 1]), 0.0)
        return outside + inside - corner

    return sdf


def segment(a: Point, b: Point, width: Scalar = 1.0) -> Shape:
    """A line segment from ``a`` to ``b`` with round caps (a capsule).

    Args:
        a: ``(x, y)`` start point in pixels.
        b: ``(x, y)`` end point in pixels.
        width: Thickness in pixels.
    """
    a, b = _f32(a), _f32(b)

    def sdf(p: Array) -> Array:
        pa, ba = p - a, b - a
        h = jnp.clip(jnp.sum(pa * ba, -1) / (jnp.sum(ba * ba) + _EPS), 0.0, 1.0)
        return _length(pa - ba * h[..., None]) - width / 2.0

    return sdf


def polygon(points: ArrayLike) -> Shape:
    """A filled simple polygon (convex or concave, either winding order).

    Cost grows with the number of vertices ``N``: the distance to every edge is
    computed for every pixel, so memory is ``O(H * W * N)``.

    Args:
        points: Array-like of shape ``(N, 2)`` with vertices in order. The
            polygon is closed automatically (last vertex connects to first).
            Self-intersecting polygons have undefined interiors.
    """
    v = _f32(points)
    v_next = jnp.roll(v, -1, axis=0)
    e = v_next - v

    def sdf(p: Array) -> Array:
        pe = p[..., None, :]  # (..., 1, 2)
        w = pe - v  # (..., N, 2)
        t = jnp.clip(jnp.sum(w * e, -1) / (jnp.sum(e * e, -1) + _EPS), 0.0, 1.0)
        b = w - e * t[..., None]
        d2 = jnp.min(jnp.sum(b * b, -1), axis=-1)  # (...)
        # Winding-number sign (Inigo Quilez's polygon SDF).
        y = pe[..., 1]
        c1 = y >= v[:, 1]
        c2 = y < v_next[:, 1]
        c3 = e[:, 0] * w[..., 1] > e[:, 1] * w[..., 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        sign = jnp.prod(jnp.where(flip, -1.0, 1.0), axis=-1)
        return sign * jnp.sqrt(d2 + _EPS)

    return sdf


def triangle(a: Point, b: Point, c: Point) -> Shape:
    """A filled triangle with vertices ``a``, ``b``, ``c`` (pixels)."""
    return polygon(jnp.stack([_f32(a), _f32(b), _f32(c)]))


def regular_polygon(
    center: Point, radius: Scalar, sides: int, angle: Scalar = 0.0
) -> Shape:
    """A regular ``sides``-gon inscribed in a circle.

    Args:
        center: ``(x, y)`` center in pixels.
        radius: Circumradius (center to vertex) in pixels.
        sides: Number of sides; a static Python ``int`` (it sets array shapes,
            so it cannot be traced under ``jit``).
        angle: Rotation in radians; ``0`` puts the first vertex to the right.
    """
    k = jnp.arange(sides, dtype=jnp.float32)
    th = angle + 2.0 * jnp.pi * k / sides
    pts = _f32(center) + radius * jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)
    return polygon(pts)


def star(
    center: Point,
    outer: Scalar,
    inner: Scalar,
    points: int = 5,
    angle: Scalar = -math.pi / 2,
) -> Shape:
    """A star polygon.

    Args:
        center: ``(x, y)`` center in pixels.
        outer: Radius of the tips in pixels.
        inner: Radius of the inner vertices in pixels (smaller = spikier).
        points: Number of tips; a static Python ``int``.
        angle: Rotation in radians. The default points one tip straight up
            (y grows downward on screen).
    """
    k = jnp.arange(2 * points, dtype=jnp.float32)
    r = jnp.where(k % 2 == 0, outer, inner)
    th = angle + jnp.pi * k / points
    pts = _f32(center) + r[:, None] * jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)
    return polygon(pts)


# --------------------------------------------------------------------------
# Operations (shape -> shape)
# --------------------------------------------------------------------------
def union(*shapes: Shape) -> Shape:
    """The region covered by any of ``shapes``."""
    return lambda p: jnp.min(jnp.stack([s(p) for s in shapes]), axis=0)


def intersect(*shapes: Shape) -> Shape:
    """The region covered by all of ``shapes``."""
    return lambda p: jnp.max(jnp.stack([s(p) for s in shapes]), axis=0)


def subtract(a: Shape, b: Shape) -> Shape:
    """Shape ``a`` with the region of ``b`` cut out of it."""
    return lambda p: jnp.maximum(a(p), -b(p))


def smooth_union(a: Shape, b: Shape, k: Scalar = 10.0) -> Shape:
    """Union of ``a`` and ``b`` with a rounded blend.

    Args:
        a: First shape.
        b: Second shape.
        k: Blend width in pixels; larger values merge the shapes more gently.
            Must be positive.
    """

    def sdf(p: Array) -> Array:
        da, db = a(p), b(p)
        h = jnp.clip(0.5 + 0.5 * (db - da) / k, 0.0, 1.0)
        return db * (1 - h) + da * h - k * h * (1 - h)

    return sdf


def outline(shape: Shape, width: Scalar = 2.0) -> Shape:
    """A hollow version of ``shape``: a band ``width`` pixels wide on its edge.

    ``stroke(img, s, ...)`` is equivalent to ``fill(img, outline(s), ...)``.
    """
    return lambda p: jnp.abs(shape(p)) - width / 2.0


def translate(shape: Shape, offset: Point) -> Shape:
    """Move ``shape`` by ``offset = (dx, dy)`` pixels."""
    off = _f32(offset)
    return lambda p: shape(p - off)


def rotate(shape: Shape, angle: Scalar, about: Point = (0.0, 0.0)) -> Shape:
    """Rotate ``shape`` by ``angle`` radians around the point ``about``.

    Because y points down in image coordinates, a positive angle appears
    clockwise on screen.
    """
    o = _f32(about)
    return lambda p: shape(_rot(p - o, -angle) + o)


def scale(shape: Shape, s: Scalar) -> Shape:
    """Scale ``shape`` by a factor ``s`` about the origin ``(0, 0)``.

    To scale about the shape's own center, wrap it:
    ``translate(scale(translate(shape, -c), s), c)``.
    """
    return lambda p: shape(p / s) * s
