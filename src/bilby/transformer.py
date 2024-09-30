import functools
from typing import Any, Callable, Optional, Tuple
from dataclasses import field

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen import initializers
from flax.linen.linear import (
  DenseGeneral,
  PrecisionLike,
  default_kernel_init,
)
from flax.linen.module import compact, merge_param
from flax.linen.normalization import LayerNorm

from flash_enformer import flash_attention as enformer_flash_attention
from flash import flash_attention
from rope_flax import RoPE, freqs_lang

PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

# Positionally-encoded self-attention, per Methods section of...
# Effective gene expression prediction from sequence by integrating long-range interactions
# Avsec et al, Nature Methods, 2021
class EnformerDotProductSelfAttention(nn.Module):
    use_flash_attention: bool = False
    use_rope: bool = False
    pos_emb_dim: int = None

    @nn.compact
    def __call__(self, query, key, value, **kwargs):
        # query: (B, L, H, Dqk)
        #   key: (B, L, H, Dqk)
        # value: (B, L, H, Dv)
        #     => (B, L, H, Dv)
        assert query.ndim == key.ndim, 'q, k must have same rank.'
        assert query.shape[-4] == key.shape[-4], 'q, k batch dims must match.'
        assert query.shape[-3] == key.shape[-3], 'q, k lengths must match.'
        assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        num_batches = query.shape[-4]
        seq_length = query.shape[-3]
        num_heads = query.shape[-2]
        qk_depth = query.shape[-1]

        if self.use_rope:
          rope = RoPE (dim=qk_depth, num_heads=num_heads, freqs_init=freqs_lang(seq_length))
          pos = jnp.arange(seq_length)[None,:]  # (1,L)
          #print(f"Shapes before applying RoPE: query={query.shape}, key={key.shape}, pos={pos.shape}")
          query = rope (query, pos)
          key = rope (key, pos)
          #print(f"Shapes after applying RoPE: query={query.shape}, key={key.shape}, value={value.shape}")
          if self.use_flash_attention:
            return flash_attention (query, key, value)
          else:
            return vanilla_attention (query, key, value, **kwargs)

        else:
          if self.pos_emb_dim is not None:
            pos_emb_dim = self.pos_emb_dim
          else:
            pos_emb_dim = qk_depth

          u = self.param("u", nn.initializers.he_normal(), (num_heads, qk_depth))
          v = self.param("v", nn.initializers.he_normal(), (num_heads, qk_depth))
          w = self.param("w", nn.initializers.he_normal(), (num_heads, qk_depth, pos_emb_dim))

          if self.use_flash_attention:
            return enformer_flash_attention (query, key, value, u, v, w)
          else:
            return enformer_attention (query, key, value, u, v, w, **kwargs)

def enformer_attention (query, key, value, u, v, w, **kwargs):
  num_batches = query.shape[-4]
  seq_length = query.shape[-3]
  num_heads = query.shape[-2]
  qk_depth = query.shape[-1]
  pos_emb_dim = w.shape[-1]

  pos = jnp.arange(seq_length)  # (L,)
  pos = jnp.repeat(pos[None,:],num_batches,axis=0)  # (B,L)

  distance = pos[:, :, None] - pos[:, None, :]  # (B,L,L)
  distance = jnp.repeat(distance[:,:,:,None],pos_emb_dim // 2,axis=-1)  # (B,L,L,N/2)

  pow_rate = jnp.exp(jnp.log((seq_length + 1) / 2) / (pos_emb_dim // 2)).astype("float32")
  center_widths = jnp.power(pow_rate, jnp.arange(1, (pos_emb_dim // 2) + 1, dtype=jnp.float32))  # (N/2,)
  unsigned_basis = jnp.where (jnp.abs(distance) <= center_widths, 1, 0)  # (B,L,L,N/2)
  signed_basis = jnp.sign(distance) * unsigned_basis  # (B,L,L,N/2)
  basis = jnp.concatenate ((unsigned_basis, signed_basis), axis=-1)  # (B,L,L,N)

#  jax.debug.print("basis={}", basis)

  # bias(b,h,i,j) = sum_d (q(b,i,h,d) * r(b,h,i,j,d) + u(h,d) * k(b,j,h,d) + v(h,d) * r(b,h,i,j,d))
  # r(b,h,i,j,d) = sum_n w(h,d,n) * basis(b,i,j,n)
  r = jnp.einsum('hdn,bijn->bhijd', w, basis)  # (B,H,L,L,Dqk)
  qr_term = jnp.einsum('bihd,bhijd->bhij', query, r)  # (B, H, L, L)
  uk_term = jnp.einsum('hd,bjhd->bhj', u, key)[:,:,None,:]  # (B, H, 1, L)
  vr_term = jnp.einsum('hd,bhijd->bhij', v, r)  # (B, H, L, L)
  bias = qr_term + uk_term + vr_term  # (B, H, L, L)
  bias = bias / jnp.sqrt(qk_depth)

  # compute attention weights
  attn_weights = nn.dot_product_attention_weights (query, key, **kwargs, bias=bias)  # (B, H, L, L)

#  jax.debug.print("p={}", attn_weights)
#  jax.debug.print("bias={}", bias)

  # return weighted sum over values for each query position
  return jnp.einsum('bhij,bjhd->bihd', attn_weights, value)


def vanilla_attention (query, key, value, **kwargs):
  # compute attention weights
  attn_weights = nn.dot_product_attention_weights (query, key, **kwargs)  # (B, H, L, L)

  # return weighted sum over values for each query position
  return jnp.einsum('bhij,bjhd->bihd', attn_weights, value)


class EnformerMultiHeadAttention(nn.Module):
  """Enformer/Baskerville-style multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: infer from inputs and params)
    param_dtype: the dtype passed to parameter initializers (default: float32)
    qk_features: dimension of the key and query.
    v_features: dimension of the value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly using
      dropout, whereas if true, the attention weights are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias_qkv: bool: whether pointwise QKV dense transforms use bias.
    use_bias_out: bool: whether pointwise output dense transform uses bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
  """

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qk_features: Optional[int] = None
  v_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[
    [PRNGKey, Shape, Dtype], Array
  ] = initializers.zeros_init()
  use_bias_qkv: bool = False
  use_bias_out: bool = True
  normalize_qk: bool = False
  attention_fn: Callable[..., Array] = field(default_factory=EnformerDotProductSelfAttention)

  @compact
  def __call__(
    self,
    inputs: Array,
    deterministic: Optional[bool] = None,
    *,
    dropout_rng: Optional[PRNGKey] = None,
    return_weights: bool = False,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs: input queries of shape `[batch_sizes..., length, features]`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      dropout_rng: optional rng key to pass to the attention layer's dropout
        mask. Otherwise, self.make_rng('dropout') is used instead.
      return_weights: if `True`, the attention weights are sowed into the
        'intermediates' collection. Remember to mark 'intermediates' as
        mutable via `mutable=['intermediates'] in order to have that
        collection returned.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    
    features = self.out_features or inputs.shape[-1]
    qk_features = self.qk_features or inputs.shape[-1]
    v_features = self.v_features or inputs.shape[-1]
    assert qk_features % self.num_heads == 0, (
      f'Query & key dimension ({qk_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    assert v_features % self.num_heads == 0, (
      f'Value dimension ({v_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )

    dense = functools.partial(
      DenseGeneral,
      axis=-1,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias_qkv,
      precision=self.precision
    )
    # project inputs to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
      dense(name='query',features=(self.num_heads,qk_features))(inputs),
      dense(name='key',features=(self.num_heads,qk_features))(inputs),
      dense(name='value',features=(self.num_heads,v_features))(inputs),
    )

    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = LayerNorm(name='query_ln', use_bias=False)(query)  # type: ignore[call-arg]
      key = LayerNorm(name='key_ln', use_bias=False)(key)  # type: ignore[call-arg]

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    if return_weights:
      x = self.attention_fn(
        query,
        key,
        value,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        module=self if return_weights else None,
      )  # pytype: disable=wrong-keyword-args
    else:
      x = self.attention_fn(
        query,
        key,
        value,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
      )

    # LayerNorm (this is not in flax's transformer; added for parity with Enformer/Baskerville)
    x = LayerNorm(name='out_ln', feature_axes=(-1,-2), reduction_axes=(-1,-2), use_bias=True)(x)

    # back to the original inputs dimensions
    out = DenseGeneral(
      features=features,
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias_out,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      name='out',  # type: ignore[call-arg]
    )(x)

    return out


class TransNao(nn.Module):
    dropout_rate: float = 0.2
    key_size: int = 64
    heads: int = 4
    pos_emb_dim: int = 32
    transformer_features: int = 768
    transformer_args: dict = field(default_factory=dict)

    dense_expansion: int = 2

    norm_type: str = "none"
    bn_momentum: float = 0.9
    activation: str = "none"

    checkpoint_trans: bool = False

    diagnostics: dict = field(default_factory=dict)

    use_flash_attention: bool = False
    use_rope: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        if self.norm_type == "layer":
            norm = nn.LayerNorm()
        elif self.norm_type == "group":
            norm = nn.GroupNorm()
        elif self.norm_type == "rms":
            norm = nn.RMSNorm()
        elif self.norm_type == "none":
            norm = lambda x: x
        else:
            raise Exception(f"Unknown norm_type: {self.norm_type}")
        
        if self.activation == "relu":
            activate = nn.relu
        elif self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "none":
            activate = lambda x: x
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        trans = EnformerMultiHeadAttention
        if self.checkpoint_trans:
           trans = nn.checkpoint(trans, static_argnums=(2,))

        # normalize, attend, dropout, residual
        skip = x

        x = norm(x)

        x = trans (
            num_heads=self.heads,
            qk_features=self.key_size,
            v_features=self.transformer_features // self.heads,
            out_features=self.transformer_features,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.he_normal(),
            attention_fn=EnformerDotProductSelfAttention (pos_emb_dim=self.pos_emb_dim, use_flash_attention=self.use_flash_attention, use_rope=self.use_rope),
            **self.transformer_args,
        ) (x, not train)

        x = nn.Dropout(rate=self.dropout_rate) (x, deterministic=not train)

        x = x + skip

        # dense expansion
        x = norm(x)
        skip = x
        x = nn.Dense (self.dense_expansion*self.transformer_features, kernel_init=nn.initializers.he_normal()) (x)
        x = nn.Dropout(rate=self.dropout_rate) (x, deterministic=not train)
        x = activate(x)
        x = nn.Dense (self.transformer_features, kernel_init=nn.initializers.he_normal()) (x)
        x = nn.Dropout(rate=self.dropout_rate) (x, deterministic=not train)
        x = x + skip

        return x
