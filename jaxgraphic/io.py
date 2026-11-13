"""Image input/output. Dependency-free PNG reader and writer (NumPy + standard library)."""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
from numpy.typing import NDArray

from ._types import ArrayLike


def to_uint8(img: ArrayLike) -> NDArray[np.uint8]:
    """Convert a float image in ``[0, 1]`` to 8-bit (clip, scale, round to nearest).

    Returns a NumPy ``uint8`` array of the same shape. This forces device
    computation, so call it outside ``jit``.
    """
    return np.asarray(np.clip(np.asarray(img), 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def to_hwc(img: ArrayLike):
    """``(C, H, W)`` -> ``(H, W, C)`` (a view for NumPy arrays; works on JAX arrays too)."""
    return (
        img.transpose(1, 2, 0)
        if hasattr(img, "transpose")
        else np.transpose(img, (1, 2, 0))
    )


def from_hwc(img: ArrayLike) -> NDArray[np.float32]:
    """Convert PIL/matplotlib-style data to the channels-first float32 layout.

    Accepts ``(H, W)`` grayscale or ``(H, W, C)``, floats in [0, 1] or integers 0..255
    (e.g. ``from_hwc(np.asarray(PIL.Image.open(path)))``). Returns ``float32[C, H, W]``
    (``[1, H, W]`` for grayscale).
    """
    a = np.asarray(img)
    a = (
        a.astype(np.float32) / 255.0
        if np.issubdtype(a.dtype, np.integer)
        else a.astype(np.float32)
    )
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3:
        raise ValueError(f"expected (H, W) or (H, W, C), got shape {a.shape}")
    return np.ascontiguousarray(a.transpose(2, 0, 1))


_FILTERS = {"none": 0, "sub": 1, "up": 2}


def _filtered(rows: NDArray[np.uint8], c: int, kind: str) -> bytes:
    """Apply one PNG filter (None / Sub / Up) to every row, vectorized."""
    h, n = rows.shape
    raw = np.empty((h, n + 1), dtype=np.uint8)
    raw[:, 0] = _FILTERS[kind]
    raw[:, 1:] = rows
    if kind == "sub":  # byte minus the same channel of the pixel to the left
        raw[:, 1 + c :] -= rows[:, :-c]  # uint8 wraparound is the PNG definition
    elif kind == "up":  # byte minus the byte in the row above
        raw[1:, 1:] -= rows[:-1]
    return raw.tobytes()


def save_png(
    img: ArrayLike,
    path: str | os.PathLike[str],
    scale: int = 1,
    compresslevel: int = 6,
    filter: str = "auto",
) -> str | os.PathLike[str]:
    """Save a channels-first RGB or RGBA image as a PNG file.

    No single PNG row filter wins everywhere: flat-color renders compress best
    unfiltered, smooth photo-like images are ~40% smaller with ``sub``/``up``. With
    ``filter="auto"`` (default) the three vectorized candidates are compressed at the
    requested level and the smallest is kept.

    Args:
        img: Float image ``(3, H, W)`` (RGB) or ``(4, H, W)`` (RGBA, straight alpha)
            with values in ``[0, 1]`` (out-of-range values are clipped).
        path: Output file path.
        scale: Integer nearest-neighbor upscale factor (handy for viewing small renders).
        compresslevel: zlib level 0-9 (6 is the zlib default; 9 is several times
            slower for ~10-15% smaller files).
        filter: ``"auto"``, ``"none"``, ``"sub"`` or ``"up"``.

    Returns:
        ``path``, for convenience.

    Raises:
        ValueError: If ``img`` is not ``(3|4, H, W)``, ``scale < 1`` or ``filter`` is unknown.
    """
    a = to_uint8(img)
    if a.ndim != 3 or a.shape[0] not in (3, 4):
        raise ValueError(
            f"expected a channels-first (3|4, H, W) image, got shape {a.shape}"
            " (use jaxgraphic.to_hwc / from_hwc to convert)"
        )
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if filter != "auto" and filter not in _FILTERS:
        raise ValueError(
            f"filter must be 'auto', 'none', 'sub' or 'up', got {filter!r}"
        )
    if scale > 1:
        a = a.repeat(scale, axis=1).repeat(scale, axis=2)
    c, h, w = a.shape
    rows = np.ascontiguousarray(a.transpose(1, 2, 0)).reshape(h, w * c)
    kinds = tuple(_FILTERS) if filter == "auto" else (filter,)
    idat = min(
        (zlib.compress(_filtered(rows, c, k), compresslevel) for k in kinds), key=len
    )
    color_type = 2 if c == 3 else 6  # truecolor / truecolor+alpha

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)
    return path


_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}  # PNG color type -> samples per pixel


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> bytearray:
    """Undo PNG scanline filtering (spec section 9). Returns ``h * stride`` bytes."""
    out = bytearray(h * stride)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(h):
        ftype = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1).copy()
        pos += 1 + stride
        if ftype == 0:  # None
            pass
        elif ftype == 1:  # Sub
            cols = line.reshape(-1, bpp).astype(np.uint32)
            line = (np.cumsum(cols, axis=0) & 0xFF).astype(np.uint8).reshape(-1)
        elif ftype == 2:  # Up
            line = ((line.astype(np.uint16) + prev) & 0xFF).astype(np.uint8)
        elif ftype == 3:  # Average
            l, pv = line.tolist(), prev.tolist()
            for i in range(stride):
                left = l[i - bpp] if i >= bpp else 0
                l[i] = (l[i] + ((left + pv[i]) >> 1)) & 0xFF
            line = np.asarray(l, dtype=np.uint8)
        elif ftype == 4:  # Paeth
            l, pv = line.tolist(), prev.tolist()
            for i in range(stride):
                a = l[i - bpp] if i >= bpp else 0
                b = pv[i]
                c = pv[i - bpp] if i >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                l[i] = (l[i] + pred) & 0xFF
            line = np.asarray(l, dtype=np.uint8)
        else:
            raise ValueError(f"invalid PNG filter type {ftype} on row {y}")
        out[y * stride : (y + 1) * stride] = line.tobytes()
        prev = line
    return out


def load_png(
    path: str | os.PathLike[str], keep_alpha: bool = True
) -> NDArray[np.float32]:
    """Read a PNG file into a channels-first float image, with no dependencies beyond NumPy.

    Supports every standard color type (grayscale, RGB, palette, gray+alpha,
    RGBA) at every bit depth (1, 2, 4, 8, 16), and ``tRNS`` transparency. Gamma
    and color profiles are ignored. Not supported: interlaced (Adam7) files.
    For JPEG or other formats, load with PIL or imageio and pass the array to
    :func:`jaxgraphic.draw_image`.

    Args:
        path: PNG file to read.
        keep_alpha: If ``True`` (default), images that have transparency are
            returned as RGBA (straight alpha). If ``False``, or if the file has
            no transparency, the result is RGB and any alpha is discarded.

    Returns:
        ``float32[3, H, W]`` or ``float32[4, H, W]`` with values in ``[0, 1]``: a
        NumPy array in the layout :func:`jaxgraphic.draw_image` expects.

    Raises:
        ValueError: If the file is not a valid PNG (bad signature, CRC or sizes).
        NotImplementedError: For interlaced PNGs.

    Note:
        Pure-Python unfiltering makes very large images with Average or Paeth
        filtering slow (a 1000x1000 image takes a few seconds).
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file (bad signature)")

    pos, ihdr, plte, trns, idat = 8, None, None, None, []
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        if crc != (zlib.crc32(tag + body) & 0xFFFFFFFF):
            raise ValueError(f"corrupt PNG: bad CRC in {tag!r} chunk")
        pos += 12 + length
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            plte = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            trns = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if ihdr is None or not idat:
        raise ValueError("corrupt PNG: missing IHDR or IDAT")
    w, h, depth, ctype, _, _, interlace = ihdr
    if interlace:
        raise NotImplementedError("interlaced (Adam7) PNGs are not supported")
    if ctype not in _CHANNELS:
        raise ValueError(f"invalid PNG color type {ctype}")

    nch = _CHANNELS[ctype]
    bits_pp = nch * depth
    bpp = max(1, bits_pp // 8)
    stride = (w * bits_pp + 7) // 8
    raw = zlib.decompress(b"".join(idat))
    if len(raw) != h * (stride + 1):
        raise ValueError("corrupt PNG: decompressed size does not match the header")
    rows = np.frombuffer(bytes(_unfilter(raw, h, stride, bpp)), dtype=np.uint8).reshape(
        h, stride
    )

    # Unpack to integer samples of shape (h, w, nch).
    if depth == 8:
        samples = rows.reshape(h, w, nch).astype(np.uint32)
    elif depth == 16:
        samples = rows.reshape(h, w * nch, 2)
        samples = ((samples[..., 0].astype(np.uint32) << 8) | samples[..., 1]).reshape(
            h, w, nch
        )
    else:  # 1, 2, 4 bits: gray or palette, one channel
        bits = np.unpackbits(rows, axis=1)[:, : w * depth].reshape(h, w, depth)
        weights = 1 << np.arange(depth - 1, -1, -1)
        samples = (bits * weights).sum(-1).astype(np.uint32)[..., None]
    maxval = float((1 << depth) - 1)

    alpha = None
    if ctype == 3:  # palette
        if plte is None:
            raise ValueError("corrupt PNG: palette image without PLTE")
        idx = samples[..., 0]
        rgb = plte[np.minimum(idx, len(plte) - 1)].astype(np.float32) / 255.0
        if trns is not None:
            table = np.full(len(plte), 255, dtype=np.uint8)
            table[: len(trns)] = np.frombuffer(trns, dtype=np.uint8)[: len(plte)]
            alpha = table[np.minimum(idx, len(plte) - 1)].astype(np.float32) / 255.0
    elif ctype in (0, 4):  # gray (+ alpha)
        g = samples[..., :1].astype(np.float32) / maxval
        rgb = np.repeat(g, 3, axis=-1)
        if ctype == 4:
            alpha = samples[..., 1].astype(np.float32) / maxval
        elif trns is not None:
            key = struct.unpack(">H", trns[:2])[0]
            alpha = (samples[..., 0] != key).astype(np.float32)
    else:  # RGB (+ alpha)
        rgb = samples[..., :3].astype(np.float32) / maxval
        if ctype == 6:
            alpha = samples[..., 3].astype(np.float32) / maxval
        elif trns is not None:
            key = np.asarray(struct.unpack(">HHH", trns[:6]), dtype=np.uint32)
            alpha = (~np.all(samples == key, axis=-1)).astype(np.float32)

    if alpha is not None and keep_alpha:
        planes = np.concatenate([rgb, alpha[..., None]], axis=-1)
    else:
        planes = rgb
    return np.ascontiguousarray(planes.transpose(2, 0, 1), dtype=np.float32)
