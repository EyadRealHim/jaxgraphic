"""Small numeric helpers shared by the shape, painting and image modules."""
from __future__ import annotations

import jax.numpy as jnp

from ._types import Array, ArrayLike

F32 = jnp.float32

#: Added under square roots so their gradient stays finite at 0 (costs ~1e-6 px of bias).
EPS = 1e-12


def as_f32(x: ArrayLike) -> Array:
    """Convert to a float32 array (keeps parameters strongly typed, so jit caches stay small)."""
    return jnp.asarray(x, dtype=F32)


def as_f32_or_none(x: ArrayLike | None) -> Array | None:
    """Like :func:`as_f32`, but passes ``None`` through (used for optional parameters)."""
    return None if x is None else as_f32(x)


def safe_hypot(a: Array, b: Array) -> Array:
    """``sqrt(a*a + b*b)`` with a finite gradient at the origin (``jnp.hypot`` has a NaN one)."""
    return jnp.sqrt(a * a + b * b + EPS)
