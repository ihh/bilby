"""Rotary Position Embedding for Flax."""
# From https://github.com/crowsonkb/rope-flax

from functools import wraps
from typing import Optional, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp


def rotate_half(x: jax.Array) -> jax.Array:
    x = einops.rearrange(x, "b l h (d r)->b l h d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return einops.rearrange(x, "b l h d r->b l h (d r)")


def apply_rotary_emb(
    freqs: jax.Array, t: jax.Array
) -> jax.Array:
    t = (t * jnp.cos(freqs)) + (rotate_half(t) * jnp.sin(freqs))
    return t

# theta should be (roughly) the context window length
# this will correspond to the longest period
def freqs_lang(theta: float = 10000.0) -> callable:
    @wraps(freqs_lang)
    def init(key, shape, dtype=jnp.float32):
        dim = shape[-1] * 2
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
        return jnp.broadcast_to(jnp.log(freqs), shape)

    return init


class RoPE(nn.Module):
    dim: int  # D
    num_heads: int = 1  # H
    dtype: jnp.dtype = jnp.float32
    freqs_init: callable = freqs_lang()

    def setup(self):
        shape = self.num_heads, self.dim // 2
        self.freqs = self.param("freqs", self.freqs_init, shape)  # (H, D/2)

    def get_freqs(self, pos: jax.Array) -> jax.Array:  # pos: (1, L)
        freqs = jnp.repeat(jnp.exp(self.freqs), 2, axis=-1)  # (H, D)
        return pos[..., None, None] * freqs.astype(self.dtype)  # (1, L, H, D)

    def __call__(self, x: jax.Array, pos: jax.Array) -> jax.Array:  # x: (B, L, H, D)  pos: (1, L)
        freqs = self.get_freqs(pos)  # (B, L, H, D)
        return apply_rotary_emb(freqs, x)

