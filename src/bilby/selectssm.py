import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

import math
from functools import reduce

import einops
import flax.linen as nn

import jax
import jax.numpy as jnp

from ssmrecscan import ssm_recursive_scan, ssm_scan

def l2_norm(params, alpha = 1.):
    return alpha * jnp.sum (jnp.array ([jnp.sum(x*x) for x in jax.tree_util.tree_leaves(params)]))

def inverse_softplus(x):
    return x + jnp.log(1 - jnp.exp(-x))

def debug_log(fmt: str, *args, **kwargs):
  jax.debug.callback(
      lambda *args, **kwargs: logging.warning(fmt.format(*args, **kwargs)),
      *args, **kwargs)

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k

# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size: int = None, n_channel_groups: int = 1):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    if n_channel_groups is not None:
        K = n_channel_groups
    else:
        K = 1
    if D % K != 0:
        raise ValueError(f"n_channel_groups={n_channel_groups} must divide D={D}")

    if chunk_size is None:
        chunk_size = largest_factor_up_to(int(math.sqrt(K*L)),L)

    if L % chunk_size != 0:
        raise ValueError(f"chunk_size={chunk_size} must divide L={L}")
    n_chunks = L // chunk_size

    # Transpose length & batch dimensions to make the scan over length, and split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse
    x_chunks = einops.rearrange (x, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)
    A_blocks = einops.rearrange (Acoeff, '(k d) n -> k d n', k=K)
    B_chunks = einops.rearrange (Bcoeff, 'b (c l) n -> c l b n', c=n_chunks)
    C_chunks = einops.rearrange (Ccoeff, 'b (c l) n -> c l b n', c=n_chunks)
    dt_chunks = einops.rearrange (dt, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    @jax.remat
    def scan_chunk (carry, chunk):
        # For the purposes of shape annotation within this code we write D instead of D/K
        g_init, h_init = carry  # (1, B, D, N)  (1, B, D, N)

        x_chunk, A_block, B_chunk, C_chunk, dt_chunk = chunk
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        dA = jnp.exp (jnp.einsum ('dn,lbd->lbdn', A_block, dt_chunk))  # (chunk_size, B, D, N)
        dB = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # The associative scan is a product of matrices of the form ((g,h),(0,1)) where g_i=exp(A*dt)x_i and h_i=B*dt*x_i
        # Since matrices of this form are are closed under multiplication, we can represent all intermediate products in the same way
        @jax.remat
        def associative_scan_fn (l, r):  # l, r, and return value are tuples of the form ((B,D,N), (B,D,N))
            g_l, h_l = l
            g_r, h_r = r
            return tuple((g_l*g_r, g_r*h_l + h_r))
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * gs
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1:,...] * g_init, hs[-1:,...]), y_chunk  # note g_init incorporated here

    # A wrapper that splits the dimensions into K blocks and does the inner associative scan for each block, re-using B and C (which don't change across dimensions)
    @jax.remat
    def scan_chunk_mapped (carry, chunk):
        g_init, h_init = carry  # (K,1,B,D/K,N) (K,1,B,D/K,N)

        x_chunk, B_chunk, C_chunk, dt_chunk = chunk   # (K,B,L,D/K), (B,L,N), (B,L,N), (K,B,L,D/K)
        @jax.remat
        def scan_chunk_wrapper (block):
            dA_init_block, dB_init_block, x_chunk_block, A_block, dt_chunk_block = block
            return scan_chunk ((dA_init_block, dB_init_block), (x_chunk_block, A_block, B_chunk, C_chunk, dt_chunk_block))
        return jax.lax.map (scan_chunk_wrapper, (g_init, h_init, x_chunk, A_blocks, dt_chunk))

    
    # Perform the scan over chunks recurrently (with rematerialization as noted above), with each chunk being an associative scan
    (_A_final, _h_final), y_chunks = jax.lax.scan (scan_chunk_mapped, (jnp.ones((K,1,B,D//K,N)), jnp.zeros((K,1,B,D//K,N))), (x_chunks, B_chunks, C_chunks, dt_chunks))  # (K, n_chunks, B, D//K)

    return einops.rearrange (y_chunks, 'c k l b d -> b (c l) (k d)')  # (B, L, D)


class SelectiveSSM(nn.Module):
    """ A variation on MAMBA: https://arxiv.org/pdf/2312.00752.pdf """

    reverse: bool = False
    complement: bool = False  # only checked if reverse is true
    
    hidden_features: int = 16  # N
    chunk_size: int = None
    n_channel_groups: int = None

    dt_rank: Union[int, str] = 'auto'  # R
    dt_proj: bool = True   # whether to use a linear projection (vs broadcast) to map dt_rank to D

    dt_min: float = 0.001  # 1/(long-range context length)
    dt_max: float = 0.1    # 1/(short-range context length)

    a_init_scale: float = 1.0

    l2_scale: float = 0.0

    shift_conv_size: int = 3

    activation: str = "silu"

    diagnostics: dict = field(default_factory=dict)

    recursive_scan: bool = False
    min_recursion_length: int = 2
    recursive_split: int = 2

    custom_vjp_scan: bool = False

    @nn.compact
    def __call__(
        self,
        x,  # (B, L, D)
        train: bool = False,
    ):
        B = x.shape[-3]
        L = x.shape[-2]
        D = x.shape[-1]  # if called by BidirectionalMamba, this is actually E*D

        N = self.hidden_features
 
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(D / 16)
        else:
            dt_rank = self.dt_rank

        if train and 'ssm_input_norm' in self.diagnostics:
            self.sow("diagnostics", "ssm_input_mean", jnp.mean(x))
            self.sow("diagnostics", "ssm_input_sd", jnp.std(x))

        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)

        u = nn.Conv (features=D, feature_group_count=D, kernel_size=(self.shift_conv_size,), strides=(1,), padding="SAME", use_bias=False, name="shift_conv", kernel_init=nn.initializers.lecun_normal()) (x)  # (B, L, D)

        if train and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "conv_mean", jnp.mean(u))
            self.sow("diagnostics", "conv_sd", jnp.std(u))

        if self.activation == "gelu":
            u = nn.gelu(u)
        elif self.activation == "relu":
            u = nn.relu(u)
        elif self.activation == "silu":
            u = nn.silu(u)
        elif self.activation is not None:
            raise Exception(f"Unknown activation: {self.activation}")

        # Initialize A nonrandomly with evenly spaced eigenvalues; keep parameterization in log space to guarantee A<0
        Acoeff = -jnp.exp (self.param ('A_log', lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N+1,dtype=jnp.float32)[None,:], D, axis=0))))  # (D, N)
        Bcoeff, Ccoeff = jnp.split (nn.Dense (features=2*N, name='BC', use_bias=True, kernel_init=nn.initializers.lecun_normal()) (u), 2, axis=-1)  # (B, L, N) *2
        Dcoeff = self.param ('D', lambda rng: jnp.ones((D,)))  # (D,)

        dt_bias_init = lambda rng, shape, dtype: inverse_softplus (jax.random.uniform (rng, shape=shape, dtype=dtype, minval=self.dt_min, maxval=self.dt_max))
        dt = nn.Dense (features=dt_rank, use_bias=True, name='dt',
                       kernel_init=nn.initializers.lecun_normal(),
                       bias_init=nn.initializers.zeros if self.dt_proj else dt_bias_init) (u)  # (B, L, dt_rank)

        if train and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "dt_lowrank_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_lowrank_sd", jnp.std(dt))

        if self.dt_proj:
            dt = nn.Dense (features=D, use_bias=True, kernel_init=nn.initializers.lecun_normal(), bias_init=dt_bias_init, name='dt_proj') (dt)  # (B, L, D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, D)
        dt = nn.activation.softplus (dt)  # (B, L, D) or (B, L, 1)

        if train and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "activated_conv_mean", jnp.mean(u))
            self.sow("diagnostics", "activated_conv_sd", jnp.std(u))
            self.sow("diagnostics", "dt_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_sd", jnp.std(dt))
            self.sow("diagnostics", "A_mean", jnp.mean(Acoeff))
            self.sow("diagnostics", "A_sd", jnp.std(Acoeff))
            self.sow("diagnostics", "B_sd", jnp.std(Bcoeff))
            self.sow("diagnostics", "C_sd", jnp.std(Ccoeff))

        # Perform SSM scan
        if self.custom_vjp_scan:
            y = ssm_scan (x, Acoeff, Bcoeff, Ccoeff, dt, min_recursion_length=self.min_recursion_length, recursive_split=self.recursive_split)  # (B, L, D)
        elif self.recursive_scan:
            y = ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, dt, min_recursion_length=self.min_recursion_length, recursive_split=self.recursive_split)  # (B, L, D)
        else:
            y = ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size=self.chunk_size, n_channel_groups=self.n_channel_groups)  # (B, L, D)

        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        if train and 'ssm_residual' in self.diagnostics:
            self.sow("diagnostics", "ssm_residual_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_residual_sd", jnp.std(y))

        # Add in the skip connection term
        y = y + jnp.einsum ('bld,d->bld', x, Dcoeff)

        # Regularizers
        if train:
            # add l2 norm for params
            self.sow("losses", "ssm_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        if train and 'ssm_output_norm' in self.diagnostics:
            self.sow("diagnostics", "ssm_output_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_output_sd", jnp.std(y))

        return y

class BidirectionalMamba(nn.Module):

    hidden_features: int   # N
    expansion_factor: float  # E

    dt_rank: Union[int, str] = 'auto'

    # For an RC-equivariant model, set all of {complement,tie_in_proj,tie_gate,concatenate_fwd_rev} to True
    complement: bool = False
    tie_in_proj: bool = False
    tie_gate: bool = False
    concatenate_fwd_rev: bool = True

    activation: str = "silu"
    norm_type: str = "rms"

    bn_momentum: float = 0.9

    mlp_layer: bool = False
    dense_expansion: int = 2
    mlp_dropout_rate: float = 0.1

    ssm_args: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    l2_scale: float = 1e-6

    @nn.compact
    def __call__(self, x, train: bool = False):

        input_features = x.shape[-1]  # D
        
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(input_features / 16)
        else:
            dt_rank = self.dt_rank

        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = nn.relu
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        skip = x
        if 'skip' in self.diagnostics and train:
            self.sow ("diagnostics", "skip_mean", jnp.mean(skip))
            self.sow ("diagnostics", "skip_sd", jnp.std(skip))

        # normalize
        if self.norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.norm_type == "rms":
            x = nn.RMSNorm()(x)

        ED = math.ceil (self.expansion_factor * input_features)
        # project to expanded dimension
        n_in_proj = 1 if self.tie_in_proj else 2
        n_gate = 1 if self.tie_gate else 2
        [xf, _xr, zf, _zr] = jnp.split (nn.Dense (features=((n_in_proj+n_gate)*ED), name='in_proj', kernel_init=nn.initializers.lecun_normal()) (x), [k*ED for k in [1,n_in_proj,n_in_proj+1]], axis=-1)
        xr = xf if self.tie_in_proj else _xr
        zr = zf if self.tie_gate else _zr

        # forward and backward SSM
        ssm = SelectiveSSM
        xf = ssm(hidden_features=self.hidden_features, reverse=False, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args) (xf, train)
        xr = ssm(hidden_features=self.hidden_features, reverse=True, complement=self.complement, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args) (xr, train)

        if 'gate' in self.diagnostics and train:
            self.sow ("diagnostics", "gate_fwd_mean", jnp.mean(zf))
            self.sow ("diagnostics", "gate_fwd_sd", jnp.std(zf))
            self.sow ("diagnostics", "gate_rev_mean", jnp.mean(zr))
            self.sow ("diagnostics", "gate_rev_sd", jnp.std(zr))

        # concatenate (or add) forward and backward channels, multiplied by respective activated gates
        if self.concatenate_fwd_rev:
            x = jnp.concatenate ([xf * activate(zf), xr * activate(zr)], axis=-1)
        else:
            x = xf * activate(zf) + xr * activate(zr)

        if 'gated' in self.diagnostics and train:
            self.sow ("diagnostics", "gated_mean", jnp.mean(x))
            self.sow ("diagnostics", "gated_sd", jnp.std(x))

        # project back down
        x = nn.Dense (features=input_features, name='out_proj', kernel_init=nn.initializers.lecun_normal()) (x)

        # residual add
        if 'residual' in self.diagnostics and train:
            self.sow ("diagnostics", "residual_mean", jnp.mean(x))
            self.sow ("diagnostics", "residual_sd", jnp.std(x))

        x = skip + x

        # MLP layer (optional)
        if self.mlp_layer:
            skip = x
            x = nn.Dense(self.dense_expansion*input_features, name="mlp", kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dropout(rate=self.mlp_dropout_rate, deterministic=not train)(x)
            x = activate(x)
            x = nn.Dense(input_features, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dropout(rate=self.mlp_dropout_rate, deterministic=not train)(x)
            x = skip + x

        # Regularizers
        if train:
            self.sow("losses", "mamba_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        return x
