"""A dependency-free PNG reader and writer (NumPy + the standard library).

Images are channels-first ``(C, H, W)`` float arrays, matching the rest of jaxgraphic.
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np
from numpy.typing import NDArray

from ._types import ArrayLike
from .convert import to_uint8

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}      # PNG color type -> samples per pixel
_FILTER_IDS = {"none": 0, "sub": 1, "up": 2}


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def _apply_filter(rows: NDArray[np.uint8], channels: int, kind: str) -> bytes:
    """Prefix every row with its filter id and apply the None / Sub / Up filter, vectorized."""
    height, row_bytes = rows.shape
    raw = np.empty((height, row_bytes + 1), dtype=np.uint8)
    raw[:, 0] = _FILTER_IDS[kind]
    raw[:, 1:] = rows
    if kind == "sub":                              # byte minus the same channel one pixel to the left
        raw[:, 1 + channels:] -= rows[:, :-channels]   # uint8 wraparound is the PNG definition
    elif kind == "up":                             # byte minus the byte in the row above
        raw[1:, 1:] -= rows[:-1]
    return raw.tobytes()


def save_png(img: ArrayLike, path: str | os.PathLike[str], scale: int = 1,
             compresslevel: int = 6, filter: str = "auto") -> str | os.PathLike[str]:
    """Save a channels-first RGB or RGBA image as a PNG file.

    No single PNG row filter wins everywhere: flat-color renders compress best unfiltered,
    smooth photo-like images are ~40% smaller with ``sub``/``up``. With ``filter="auto"``
    (default) the three vectorized candidates are compressed at the requested level and
    the smallest is kept.

    Args:
        img: Float image ``(3, H, W)`` (RGB) or ``(4, H, W)`` (RGBA, straight alpha) with
            values in ``[0, 1]`` (out-of-range values are clipped). Call outside ``jit``.
        path: Output file path.
        scale: Integer nearest-neighbor upscale factor (handy for viewing small renders).
        compresslevel: zlib level 0-9 (6 is zlib's default; 9 is several times slower for
            ~10-15% smaller files).
        filter: ``"auto"``, ``"none"``, ``"sub"`` or ``"up"``.

    Returns:
        ``path``, for convenience.

    Raises:
        ValueError: If ``img`` is not ``(3|4, H, W)``, ``scale < 1`` or ``filter`` is unknown.
    """
    pixels = np.asarray(to_uint8(img))
    if pixels.ndim != 3 or pixels.shape[0] not in (3, 4):
        raise ValueError(f"expected a channels-first (3|4, H, W) image, got shape {pixels.shape}"
                         " (use jaxgraphic.to_hwc / from_hwc to convert)")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if filter != "auto" and filter not in _FILTER_IDS:
        raise ValueError(f"filter must be 'auto', 'none', 'sub' or 'up', got {filter!r}")
    if scale > 1:
        pixels = pixels.repeat(scale, axis=1).repeat(scale, axis=2)

    channels, height, width = pixels.shape
    rows = np.ascontiguousarray(pixels.transpose(1, 2, 0)).reshape(height, width * channels)
    kinds = tuple(_FILTER_IDS) if filter == "auto" else (filter,)
    idat = min((zlib.compress(_apply_filter(rows, channels, k), compresslevel) for k in kinds), key=len)

    color_type = 2 if channels == 3 else 6          # truecolor / truecolor + alpha
    png = (_SIGNATURE
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
           + _chunk(b"IDAT", idat)
           + _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    return path


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _read_chunks(data: bytes) -> dict:
    """Parse and CRC-check the chunks we need; returns ``ihdr``, ``plte``, ``trns`` and ``idat``."""
    if data[:8] != _SIGNATURE:
        raise ValueError("not a PNG file (bad signature)")
    found: dict = {"ihdr": None, "plte": None, "trns": None, "idat": []}
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if crc != (zlib.crc32(tag + body) & 0xFFFFFFFF):
            raise ValueError(f"corrupt PNG: bad CRC in {tag!r} chunk")
        pos += 12 + length
        if tag == b"IHDR":
            found["ihdr"] = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            found["plte"] = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            found["trns"] = body
        elif tag == b"IDAT":
            found["idat"].append(body)
        elif tag == b"IEND":
            break
    if found["ihdr"] is None or not found["idat"]:
        raise ValueError("corrupt PNG: missing IHDR or IDAT")
    return found


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> bytearray:
    """Undo PNG scanline filtering (spec section 9). Returns ``height * stride`` bytes.

    None, Sub and Up are vectorized per row; Average and Paeth are sequential along the
    row and run in pure Python.
    """
    out = bytearray(height * stride)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1).copy()
        pos += 1 + stride
        if filter_type == 0:                                     # None
            pass
        elif filter_type == 1:                                   # Sub
            cols = line.reshape(-1, bpp).astype(np.uint32)
            line = (np.cumsum(cols, axis=0) & 0xFF).astype(np.uint8).reshape(-1)
        elif filter_type == 2:                                   # Up
            line = ((line.astype(np.uint16) + prev) & 0xFF).astype(np.uint8)
        elif filter_type == 3:                                   # Average
            cur, above = line.tolist(), prev.tolist()
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + above[i]) >> 1)) & 0xFF
            line = np.asarray(cur, dtype=np.uint8)
        elif filter_type == 4:                                   # Paeth
            cur, above = line.tolist(), prev.tolist()
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0              # left
                b = above[i]                                     # up
                c = above[i - bpp] if i >= bpp else 0            # up-left
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                predictor = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + predictor) & 0xFF
            line = np.asarray(cur, dtype=np.uint8)
        else:
            raise ValueError(f"invalid PNG filter type {filter_type} on row {y}")
        out[y * stride:(y + 1) * stride] = line.tobytes()
        prev = line
    return out


def _unpack_samples(rows: NDArray[np.uint8], width: int, height: int, depth: int,
                    channels: int) -> NDArray[np.uint32]:
    """Turn unfiltered scanline bytes into integer samples of shape ``(H, W, channels)``."""
    if depth == 8:
        return rows.reshape(height, width, channels).astype(np.uint32)
    if depth == 16:
        pairs = rows.reshape(height, width * channels, 2)
        return ((pairs[..., 0].astype(np.uint32) << 8) | pairs[..., 1]).reshape(height, width, channels)
    # 1, 2 or 4 bits per sample (gray or palette, one channel): unpack the bit stream.
    bits = np.unpackbits(rows, axis=1)[:, :width * depth].reshape(height, width, depth)
    weights = 1 << np.arange(depth - 1, -1, -1)
    return (bits * weights).sum(-1).astype(np.uint32)[..., None]


def _samples_to_planes(samples, depth: int, color_type: int, plte, trns, keep_alpha: bool):
    """Convert integer samples to float ``(C, H, W)`` planes, applying palette and tRNS transparency."""
    max_value = float((1 << depth) - 1)
    alpha = None
    if color_type == 3:                                          # palette
        if plte is None:
            raise ValueError("corrupt PNG: palette image without PLTE")
        index = np.minimum(samples[..., 0], len(plte) - 1)
        color = plte[index].astype(np.float32) / 255.0
        if trns is not None:
            table = np.full(len(plte), 255, dtype=np.uint8)
            table[:len(trns)] = np.frombuffer(trns, dtype=np.uint8)[:len(plte)]
            alpha = table[index].astype(np.float32) / 255.0
    elif color_type in (0, 4):                                   # gray, gray + alpha
        color = np.repeat(samples[..., :1].astype(np.float32) / max_value, 3, axis=-1)
        if color_type == 4:
            alpha = samples[..., 1].astype(np.float32) / max_value
        elif trns is not None:                                   # one fully transparent gray value
            alpha = (samples[..., 0] != struct.unpack(">H", trns[:2])[0]).astype(np.float32)
    else:                                                        # RGB, RGBA
        color = samples[..., :3].astype(np.float32) / max_value
        if color_type == 6:
            alpha = samples[..., 3].astype(np.float32) / max_value
        elif trns is not None:                                   # one fully transparent RGB value
            key = np.asarray(struct.unpack(">HHH", trns[:6]), dtype=np.uint32)
            alpha = (~np.all(samples == key, axis=-1)).astype(np.float32)

    planes = np.concatenate([color, alpha[..., None]], axis=-1) if alpha is not None and keep_alpha else color
    return np.ascontiguousarray(planes.transpose(2, 0, 1), dtype=np.float32)


def load_png(path: str | os.PathLike[str], keep_alpha: bool = True) -> NDArray[np.float32]:
    """Read a PNG file into a channels-first float image, with no dependencies beyond NumPy.

    Supports every standard color type (grayscale, RGB, palette, gray+alpha, RGBA) at every
    bit depth (1, 2, 4, 8, 16), and ``tRNS`` transparency. Gamma and color profiles are
    ignored. Not supported: interlaced (Adam7) files. For JPEG or other formats, load with
    PIL or imageio and convert with :func:`jaxgraphic.from_hwc`.

    Args:
        path: PNG file to read.
        keep_alpha: If ``True`` (default), images with transparency are returned as RGBA
            (straight alpha). If ``False``, or if the file has no transparency, the result is
            RGB and any alpha is discarded.

    Returns:
        ``float32[3, H, W]`` or ``float32[4, H, W]`` with values in ``[0, 1]``: a NumPy array
        in the layout :func:`jaxgraphic.draw_image` expects.

    Raises:
        ValueError: If the file is not a valid PNG (bad signature, CRC or sizes).
        NotImplementedError: For interlaced PNGs.

    Note:
        Pure-Python unfiltering makes very large images with Average or Paeth filtering slow
        (a 1000x1000 image takes a few seconds).
    """
    with open(path, "rb") as f:
        chunks = _read_chunks(f.read())
    width, height, depth, color_type, _, _, interlace = chunks["ihdr"]
    if interlace:
        raise NotImplementedError("interlaced (Adam7) PNGs are not supported")
    if color_type not in _CHANNELS:
        raise ValueError(f"invalid PNG color type {color_type}")

    channels = _CHANNELS[color_type]
    bits_per_pixel = channels * depth
    bpp = max(1, bits_per_pixel // 8)                            # filter "previous pixel" distance in bytes
    stride = (width * bits_per_pixel + 7) // 8
    raw = zlib.decompress(b"".join(chunks["idat"]))
    if len(raw) != height * (stride + 1):
        raise ValueError("corrupt PNG: decompressed size does not match the header")

    rows = np.frombuffer(bytes(_unfilter(raw, height, stride, bpp)), dtype=np.uint8).reshape(height, stride)
    samples = _unpack_samples(rows, width, height, depth, channels)
    return _samples_to_planes(samples, depth, color_type, chunks["plte"], chunks["trns"], keep_alpha)
