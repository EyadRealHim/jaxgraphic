"""Drawing images onto a canvas: ``draw_image``.

The source is placed by its center, resampled bilinearly (with premultiplied alpha so
transparent edges don't darken), and composited with an anti-aliased rectangular edge.
The steps are small named helpers so each can be read on its own.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ._types import Array, ArrayLike, Image, Point, Scalar
from ._util import F32, as_f32
from .raster import coords, coverage
from .shapes import ShapeLike, as_shape, local_coords, rect_sdf


def _as_planar(src: ArrayLike) -> Array:
    """Normalize a source image to ``float32[C, H, W]`` with C in {3, 4}.

    Accepts float in [0, 1] or integer 0..255; ``(H, W)`` grayscale and channels-first
    ``(1|2|3|4, H, W)`` arrays (1 = gray, 2 = gray+alpha).
    """
    a = jnp.asarray(src)
    a = (a.astype(F32) / 255.0) if jnp.issubdtype(a.dtype, jnp.integer) else a.astype(F32)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] not in (1, 2, 3, 4):
        raise ValueError(f"expected a channels-first (C, H, W) image with C in 1..4 "
                         f"(use jaxgraphic.from_hwc for (H, W, C) data), got shape {a.shape}")
    if a.shape[0] == 1:
        a = jnp.repeat(a, 3, axis=0)
    elif a.shape[0] == 2:
        a = jnp.concatenate([jnp.repeat(a[:1], 3, axis=0), a[1:]], axis=0)
    return a


def _premultiplied_table(src: Array) -> tuple[Array, int]:
    """Flatten the source into a row-per-texel lookup table for gathering.

    Color is premultiplied by alpha once (O(source), not O(canvas)), and the image is
    edge-padded by one texel on the right and bottom so the "+1" neighbours of a bilinear
    tap never need clamping.

    Returns:
        ``(table, row_stride)`` where ``table`` is ``((sh + 1) * (sw + 1), planes)`` and
        texel ``(row, col)`` lives at index ``row * row_stride + col``.
    """
    premult = jnp.concatenate([src[:3] * src[3:4], src[3:4]], axis=0) if src.shape[0] == 4 else src
    planes, sh, sw = premult.shape
    padded = jnp.pad(premult, ((0, 0), (0, 1), (0, 1)), mode="edge")
    return padded.reshape(planes, (sh + 1) * (sw + 1)).T, sw + 1


def _bilinear_gather(table: Array, stride: int, u: Array, v: Array) -> Array:
    """Bilinearly sample ``table`` at texel coordinates ``(u, v)``; returns ``(planes, H, W)``.

    ``u`` and ``v`` must already be clamped to ``[0, sw - 1]`` / ``[0, sh - 1]``; each may be
    separable ((1, W) and (H, 1)) or full (H, W).
    """
    h, w = jnp.broadcast_shapes(jnp.shape(u), jnp.shape(v))
    u0, v0 = jnp.floor(u), jnp.floor(v)
    fx = jnp.broadcast_to(u - u0, (h, w))[..., None]
    fy = jnp.broadcast_to(v - v0, (h, w))[..., None]
    top_left = jnp.broadcast_to(v0.astype(jnp.int32) * stride + u0.astype(jnp.int32), (h, w))

    def texel(index):
        return jnp.take(table, index, axis=0, mode="clip")                   # (H, W, planes)

    top = texel(top_left) * (1.0 - fx) + texel(top_left + 1) * fx
    bottom = texel(top_left + stride) * (1.0 - fx) + texel(top_left + stride + 1) * fx
    return jnp.moveaxis(top + (bottom - top) * fy, -1, 0)


def _draw_image(img, src, center, size, scale, angle, alpha, clip, softness):
    _, h, w = img.shape
    source = _as_planar(src)
    has_alpha = source.shape[0] == 4
    _, sh, sw = source.shape
    x, y = coords(h, w)

    # 1. Pixel coordinates in the image's own frame (separable when angle is None).
    qx, qy = local_coords(x, y, center, angle)
    display = size if size is not None else jnp.asarray([sw, sh], dtype=F32) * scale

    # 2. Where the image is visible: its anti-aliased rectangle, optionally clipped.
    visible = coverage(rect_sdf(qx, qy, display[0] * 0.5, display[1] * 0.5), softness)
    if clip is not None:
        visible = visible * coverage(clip.sdf(x, y), softness)
    k = jnp.broadcast_to(visible * alpha, (h, w))

    # 3. Texel coordinates (pixel-center convention), clamped so edge handling is free.
    u = jnp.clip(qx * (sw / display[0]) + (sw * 0.5 - 0.5), 0.0, sw - 1.0)
    v = jnp.clip(qy * (sh / display[1]) + (sh * 0.5 - 0.5), 0.0, sh - 1.0)

    # 4. Sample, then composite. For RGBA the sample is premultiplied:
    #    out = img * (1 - k * a) + k * (rgb * a).
    table, stride = _premultiplied_table(source)
    sample = _bilinear_gather(table, stride, u, v)
    if has_alpha:
        return img + k[None] * (sample[:3] - sample[3:4] * img)
    return img + k[None] * (sample - img)


_draw_image_jit = jax.jit(_draw_image)


def draw_image(img: Image, src: ArrayLike, center: Point, size: Point | None = None,
               scale: Scalar = 1.0, angle: Scalar | None = None, alpha: Scalar = 1.0,
               clip: ShapeLike | None = None, softness: Scalar = 1.0) -> Image:
    """Composite the image ``src`` onto ``img`` with position, size, rotation and clipping.

    The source is placed by its *center* (scale and rotation pivot there), resampled
    bilinearly with premultiplied alpha, and its edge is anti-aliased.

    Args:
        img: ``float32[3, H, W]`` canvas. Not modified.
        src: Channels-first ``(C, H, W)`` source: C = 3 (RGB) or 4 (RGBA, straight alpha);
            also ``(H, W)`` grayscale and C = 1 or 2 (gray, gray+alpha). Float in [0, 1] or
            integer 0..255. Use :func:`jaxgraphic.from_hwc` for ``(H, W, C)`` data. Its
            shape must be static under ``jit``.
        center: ``(x, y)`` canvas position of the middle of ``src``.
        size: Optional ``(width, height)`` in canvas pixels; overrides ``scale``.
        scale: Size multiplier relative to the source's pixel size.
        angle: Rotation in radians about ``center`` (clockwise on screen). Leave as ``None``
            for axis-aligned placement: the sampling coordinates then stay separable and
            the setup work is O(H + W). ``0.0`` takes the general (slower) path.
        alpha: Overall opacity in ``[0, 1]``.
        clip: Optional shape; the image is only visible inside it.
        softness: Edge width in pixels for the rectangle edge and ``clip``.

    Note:
        Sampling is bilinear with no low-pass filter: shrinking by more than ~2x can alias,
        so downscale large images first.
    """
    return _draw_image_jit(img, src, as_f32(center), None if size is None else as_f32(size),
                           scale, None if angle is None else as_f32(angle), alpha,
                           None if clip is None else as_shape(clip), softness)
