import logging

from functools import partial

import einops
import flax.linen as nn

import jax
from jax import custom_vjp, jit, numpy as jnp

# The associative scan is a product of matrices of the form ((g,h),(0,1)) where g_i=exp(A*Delta)x_i and h_i=B*Delta*x_i
# Since matrices of this form are are closed under multiplication, we can represent all intermediate products in the same way
@jax.remat
def associative_scan_fn (l, r):
    g_l, h_l = l
    g_r, h_r = r
    return tuple((g_l*g_r, g_r*h_l + h_r))

# alpha = exp(A*Delta) [zero-order hold], beta = B*Delta*x [Euler step]
@jit
def compute_alpha (Acoeff, Delta_chunk):
    return jnp.exp (jnp.einsum ('dn,lbd->lbdn', Acoeff, Delta_chunk))  # (chunk_size, B, D, N)

# The zero-order hold is empirically only really necessary for alpha, since it has a gating effect
@jit
def compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk):
    alpha = compute_alpha (Acoeff, Delta_chunk)  # (chunk_size, B, D, N)
    beta = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, Delta_chunk)  # (chunk_size, B, D, N)
    return alpha, beta

# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# Delta: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length: int = 2, recursive_split: int = 2):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    # Transpose length & batch dimensions to make the scan over length, and split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse in terms of GPU memory usage
    x = einops.rearrange (x, 'b l d -> l b d')
    Bcoeff = einops.rearrange (Bcoeff, 'b l n -> l b n')
    Ccoeff = einops.rearrange (Ccoeff, 'b l n -> l b n')
    Delta = einops.rearrange (Delta, 'b l d -> l b d')

    # Recursive function to do associative scan
    @jax.remat
    def scan_chunk (carry, chunk):
        g_init, h_init = carry  # (B, D, N)  (B, D, N)
        x_chunk, B_chunk, C_chunk, Delta_chunk = chunk
        chunk_size = x_chunk.shape[0]

        if chunk_size > min_recursion_length and chunk_size % recursive_split == 0:
            # Split inputs into chunks, scan each chunk, and concatenate results
            # Again, this seems inefficient, but empirically uses less GPU memory than passing an index range and doing dynamic slicing
            x_chunk = einops.rearrange (x_chunk, '(c l) b d -> c l b d', c=recursive_split)
            B_chunk = einops.rearrange (B_chunk, '(c l) b n -> c l b n', c=recursive_split)
            C_chunk = einops.rearrange (C_chunk, '(c l) b n -> c l b n', c=recursive_split)
            Delta_chunk = einops.rearrange (Delta_chunk, '(c l) b d -> c l b d', c=recursive_split)
            (g_init, h_init), y_chunk = jax.lax.scan (scan_chunk, (g_init, h_init), (x_chunk, B_chunk, C_chunk, Delta_chunk))
            y_chunk = einops.rearrange (y_chunk, 'c l b d -> (c l) b d')
            return (g_init, h_init), y_chunk

        alpha, beta = compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk)  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (alpha, beta))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * As
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1,...] * g_init, hs[-1,...]), y_chunk  # note g_init incorporated here

    (_A_final, _h_final), y = scan_chunk ((jnp.ones((B,D,N)), jnp.zeros((B,D,N))), (x, Bcoeff, Ccoeff, Delta))

    return einops.rearrange (y, 'l b d -> b l d')  # (B, L, D)


@partial(custom_vjp, nondiff_argnums=(5,6))
@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length: int = 2, recursive_split: int = 2):
    return ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length, recursive_split)

@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan_forward (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length, recursive_split):
    y = ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length, recursive_split)
    return y, (x, Acoeff, Bcoeff, Ccoeff, Delta)

@jit
def forward_scan_fn (h_tMinus1, chunk):
    alpha_t, beta_t = chunk
    h_t = h_tMinus1 * alpha_t + beta_t
    return h_t, h_tMinus1

@jit
def backward_scan_fn (f_alpha_tPlus1, chunk):
    alpha_t, C_dy_t = chunk
    f_t = f_alpha_tPlus1 + C_dy_t
    return f_t * alpha_t, f_t


# x_chunk: (L, B, D)
# Acoeff: (D, N)
# B_chunk: (L, B, N)
# C_chunk: (L, B, N)
# Delta_chunk: (L, B, D)
# dy_chunk: (L, B, D)
# h_left: (B, D, N)
# f_alpha_right: (B, D, N)
@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan_backward_recursive (x_chunk, Acoeff, B_chunk, C_chunk, Delta_chunk, dy_chunk, h_left, f_alpha_right, min_recursion_length: int = 2, recursive_split: int = 2):
    L = x_chunk.shape[0]
    if L > min_recursion_length and L % recursive_split == 0:
        mid = jnp.ceil (L // 2)
        x_chunk = einops.rearrange (x_chunk, '(c l) b d -> c l b d', c=recursive_split)
        B_chunk = einops.rearrange (B_chunk, '(c l) b n -> c l b n', c=recursive_split)
        C_chunk = einops.rearrange (C_chunk, '(c l) b n -> c l b n', c=recursive_split)
        Delta_chunk = einops.rearrange (Delta_chunk, '(c l) b d -> c l b d', c=recursive_split)
        dy_chunk = einops.rearrange (dy_chunk, '(c l) b d -> c l b d', c=recursive_split)
        @jit
        def slim_backward_scan_fn (f_alpha_tPlus1, chunk):
            C_t, Delta_t, dy_t = chunk
            alpha_t = jnp.exp (jnp.einsum ('dn,bd->bdn', Acoeff, Delta_t))
            C_dy_t = jnp.einsum ('bn,bd->bdn', C_t, dy_t)
            f_t = f_alpha_tPlus1 + C_dy_t
            return f_t * alpha_t, None
        @jit
        def backward_scan_chunks (f_alpha, chunk):
            C, Delta, dy = chunk
            next_f_alpha, _ = jax.lax.scan (slim_backward_scan_fn, f_alpha, (C, Delta, dy), reverse=True)
            return next_f_alpha, f_alpha
        _f_alpha_left, f_alphas = jax.lax.scan (backward_scan_chunks, f_alpha_right, (C_chunk, Delta_chunk, dy_chunk), reverse=True)
        @jit
        def forward_scan_chunks (carry, chunk):
            dA, h_left = carry
            x, B, C, Delta, dy, f_alpha_right = chunk
            dx_chunk, dA_chunk, dB_chunk, dC_chunk, dDelta_chunk, h_right = ssm_scan_backward_recursive (x, Acoeff, B, C, Delta, dy, h_left, f_alpha_right, min_recursion_length=min_recursion_length, recursive_split=recursive_split)
            dA = dA + dA_chunk
            return (dA, h_right), (dx_chunk, dB_chunk, dC_chunk, dDelta_chunk)
        (dA, h_right), (dxs, dBs, dCs, dDeltas) = jax.lax.scan (forward_scan_chunks,
                                                                (jnp.zeros_like(Acoeff), h_left), 
                                                                (x_chunk, B_chunk, C_chunk, Delta_chunk, dy_chunk, f_alphas))
        dxs = einops.rearrange (dxs, 'c l b d -> (c l) b d')
        dBs = einops.rearrange (dBs, 'c l b n -> (c l) b n')
        dCs = einops.rearrange (dCs, 'c l b n -> (c l) b n')
        dDeltas = einops.rearrange (dDeltas, 'c l b d -> (c l) b d')
        return dxs, dA, dBs, dCs, dDeltas, h_right
    else:
        alpha, beta = compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk)   # (L,B,D,N) (L,B,D,N)
        C_dy = jnp.einsum ('lbn,lbd->lbdn', C_chunk, dy_chunk)  # (L,B,D,N)
        h_right, hs = jax.lax.scan (forward_scan_fn, h_left, (alpha, beta))  # (B,D,N) (L,B,D,N)
        _f_alpha_left, fs = jax.lax.scan (backward_scan_fn, f_alpha_right, (alpha, C_dy), reverse=True)  # (B,D,N) (L,B,D,N)
        Delta_fs = jnp.einsum ('lbd,lbdn->lbdn', Delta_chunk, fs)
        alpha_hs = jnp.einsum ('lbdn,lbdn->lbdn', alpha, hs)
        dx = jnp.einsum ('lbdn,lbn->lbd', Delta_fs, B_chunk)
        dA = jnp.einsum ('lbdn,lbdn->dn', Delta_fs, alpha_hs)
        dB = jnp.einsum ('lbdn,lbd->lbn', Delta_fs, x_chunk)
        dC = jnp.einsum ('lbd,lbdn->lbn', dy_chunk, jnp.concatenate ([hs[1:,...], h_right[None,...]], axis=0))
        dDelta = jnp.einsum ('lbdn,lbdn->lbd', fs, jnp.einsum('dn,lbdn->lbdn', Acoeff, alpha_hs) + jnp.einsum('lbn,lbd->lbdn', B_chunk, x_chunk))
        return dx, dA, dB, dC, dDelta, h_right

@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan_backward (min_recursion_length, recursive_split, res, dy):
    x, Acoeff, Bcoeff, Ccoeff, Delta = res
    B = x.shape[-3]
    D = x.shape[-1]
    N = Acoeff.shape[-1]
    x = einops.rearrange (x, 'b l d -> l b d')
    Bcoeff = einops.rearrange (Bcoeff, 'b l n -> l b n')
    Ccoeff = einops.rearrange (Ccoeff, 'b l n -> l b n')
    Delta = einops.rearrange (Delta, 'b l d -> l b d')
    dy = einops.rearrange (dy, 'b l d -> l b d')
    h_left = jnp.zeros ((B, D, N))
    f_alpha_right = jnp.zeros ((B, D, N))
    dx, dA, dB, dC, dDelta, _h_right = ssm_scan_backward_recursive (x, Acoeff, Bcoeff, Ccoeff, Delta, dy, h_left, f_alpha_right, min_recursion_length=min_recursion_length, recursive_split=recursive_split)
    dx = einops.rearrange (dx, 'l b d -> b l d')
    dB = einops.rearrange (dB, 'l b n -> b l n')
    dC = einops.rearrange (dC, 'l b n -> b l n')
    dDelta = einops.rearrange (dDelta, 'l b d -> b l d')
    return dx, dA, dB, dC, dDelta

ssm_scan.defvjp (ssm_scan_forward, ssm_scan_backward)