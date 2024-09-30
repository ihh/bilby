import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import flax.linen as nn
from flax.linen.pooling import pool, avg_pool
import einops

def lse_reduce (x, y):
    return jnp.maximum(x,y) + jax.nn.softplus(-jnp.abs(x-y))

def sum_reduce (x, y):
    return x + y

def lse_pool(inputs, window_shape, strides=None, padding='VALID'):
  """Pools the input by taking the logsumexp of a window slice.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of ``n`` integers, representing the inter-window
      strides (default: ``(1, ..., 1)``).
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
      and after each spatial dimension (default: ``'VALID'``).
  Returns:
    The maximum for each window slice.
  """
  y = pool(inputs, -jnp.inf, lse_reduce, window_shape, strides, padding)
  return y


class WeightedAveragePooling(nn.Module):
    pool_size: int
    num_heads: int = 1

    @nn.compact
    def __call__(self, x):
        D = x.shape[-1]
        if D % self.num_heads != 0:
            raise ValueError('Number of input features must be divisible by the number of heads.')

        weights = nn.Dense (self.num_heads, kernel_init=nn.initializers.lecun_normal, bias_init=nn.initializers.zeros) (x)  # (B, L, H)
        weights = jax.nn.softplus (weights)

        x = einops.rearrange (x, '... (h d) -> ... h d', h=self.num_heads)  # (B, L, H, D/H)
        x = x * weights[...,None]
        x = einops.rearrange (x, '... h d -> ... (h d)')  # (B, L, D)

        y = avg_pool(x, window_shape=(self.pool_size,), strides=self.pool_size, padding='SAME')  # (B, L/pool_size, D)
        y_norm = pool (weights, 0, sum_reduce, window_shape=(self.pool_size,), strides=self.pool_size, padding='SAME')  # (B, L/pool_size, H)

        y = einops.rearrange (y, 'b l (h d) -> b l h d')  # (B, L/pool_size, H, D/H)
        y = y / y_norm[...,None]
        y = einops.rearrange (y, 'b l h d -> b l (h d)')  # (B, L/pool_size, D)

        return y


class WeightedSummary(nn.Module):
    num_heads: int = 1
    keep_weights: bool = False

    @nn.compact
    def __call__(self, x):
        D = x.shape[-1]
        if D % self.num_heads != 0:
            raise ValueError('Number of input features must be divisible by the number of heads.')

        weights = nn.Dense (self.num_heads, kernel_init=nn.initializers.lecun_normal, bias_init=nn.initializers.zeros) (x)  # (B, L, H)
        weights = jax.nn.softplus (weights)

        x = einops.rearrange (x, '... (h d) -> ... h d', h=self.num_heads)  # (B, L, H, D/H)
        x = x * weights[...,None]

        y = jnp.sum (x, axis=-3)  # (B, H, D/H)

        if self.keep_weights:
            weights = jnp.sum (weights, axis=-2)[...,None]  # (B, H, 1)
            y = jnp.concatenate ([weights, y], axis=-1)  # (B, H, D/H+1)

        return y
