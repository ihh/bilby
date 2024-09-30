# Based on...
# https://github.com/lucidrains/flash-attention-jax/blob/d73dcdd6cb4b4f8d786922e2ec6826036dae896a/flash_attention_jax/flash_attention.py

# Using the relative positional encoding described in...
# https://www.nature.com/articles/s41592-021-01252-x

import math
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
from jax.numpy import einsum

from einops import rearrange

# constants

EPSILON = 1e-10
MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024

# flash attention

def prepare_basis (q_chunk_idx, q_chunk_sizes, qkv_len, batch, dim):
    pow_rate = jnp.exp(jnp.log((qkv_len + 1) / 2) / (dim // 2)).astype("float32")
    center_widths = jnp.power(pow_rate, jnp.arange(1, (dim // 2) + 1, dtype=jnp.float32))  # (D/2,)

    q_pos = jnp.arange(q_chunk_sizes) + q_chunk_idx  # (L,)
    q_pos = jnp.repeat(q_pos[None,:],batch,axis=0)  # (B,L)

    return center_widths, q_pos

def make_chunked_basis (center_widths, q_pos, k_chunk_idx, k_chunk_sizes):
    batch, dim = q_pos.shape[0], center_widths.shape[0]*2

    k_pos = jnp.arange(k_chunk_sizes) + k_chunk_idx  # (K,)
    k_pos = jnp.repeat(k_pos[None,:],batch,axis=0)  # (B,K)
    distance = q_pos[:, :, None] - k_pos[:, None, :]  # (B,L,K)
    distance = jnp.repeat(distance[:,:,:,None],dim // 2,axis=-1)  # (B,L,K,D/2)

    unsigned_basis = jnp.where (jnp.abs(distance) <= center_widths, 1, 0)  # (B,L,K,D/2)
    signed_basis = jnp.sign(distance) * unsigned_basis  # (B,L,K,D/2)
    basis = jnp.concatenate ((unsigned_basis, signed_basis), axis=-1)  # (B,L,K,D)

    return basis

def _query_chunk_flash_attention(q_chunk_idx, q, k, v, upos, vpos, wpos):
    q_chunk_len, batch, heads, dim, k_len, v_dim, pos_emb_dim = *q.shape, k.shape[0], v.shape[-1], wpos.shape[-1]

    scale = 1 / jnp.sqrt(dim)
    q_scaled  = q * scale

    center_widths, q_pos = prepare_basis(q_chunk_idx, q_chunk_len, k_len, batch, pos_emb_dim)

    def chunk_scanner(carries, _):
        k_chunk_idx, out, row_sum, row_max = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (k_chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, heads, dim))
        v_chunk = lax.dynamic_slice(v, (k_chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, heads, v_dim))

        basis_chunk = make_chunked_basis(center_widths, q_pos, k_chunk_idx, k_chunk_sizes)  # (B,L,K,D)

#        jax.debug.print("q_chunk_idx={} k_chunk_idx={} basis_chunk={}", q_chunk_idx, k_chunk_idx, basis_chunk)

        r = einsum('hdn,bijn->bhijd', wpos, basis_chunk)  # (B,H,L,K,D)
        qr_term = einsum('ibhd,bhijd->ibhj', q, r)  # (L,B,H,K)
        uk_term = einsum('hd,jbhd->bhj', upos, k_chunk)[None,:,:,:]  # (1,B,H,K)
        vr_term = einsum('hd,bhijd->ibhj', vpos, r)  # (L,B,H,K)

        bias = qr_term + uk_term + vr_term  # (L,B,H,K)
        bias = bias * scale

        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk) + bias  # (L,B,H,K)

        block_row_max = jnp.max(attn_weights, axis = -1, keepdims = True)

        new_row_max = jnp.maximum(block_row_max, row_max)
        exp_weights = jnp.exp(attn_weights - new_row_max)

        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True) + EPSILON

        exp_values = einsum('i ... j, j ... d -> i ... d', exp_weights, v_chunk)

        exp_row_max_diff = jnp.exp(row_max - new_row_max)

        new_row_sum = exp_row_max_diff * row_sum + block_row_sum

#        jax.debug.print("fwd: p={}", rearrange(exp_weights / new_row_sum,'l b h k -> b h l k'))
#        jax.debug.print("fwd: bias={}", rearrange(bias,'l b h k -> b h l k'))

        out = (row_sum / new_row_sum) * exp_row_max_diff * out + \
              (1. / new_row_sum) * exp_values

        return (k_chunk_idx + k_chunk_sizes, out, new_row_sum, new_row_max), None

    out = jnp.zeros((q_chunk_len, batch, heads, v_dim))
    row_sum = jnp.zeros((q_chunk_len, batch, heads, 1))
    row_max = jnp.ones((q_chunk_len, batch, heads, 1)) * -1e6

    (_, out, row_sum, row_max), _ = lax.scan(chunk_scanner, init = (0, out, row_sum, row_max), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    row_sum = rearrange(row_sum, 'n ... 1 -> n ...')
    row_max = rearrange(row_max, 'n ... 1 -> n ...')

    lse = jnp.log(row_sum) + row_max

    return out, lse

def _flash_attention(q, k, v, upos, vpos, wpos):
    batch, q_len, heads, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(q_chunk_idx, _):
        q_chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (q_chunk_idx, 0, 0, 0), slice_sizes = (q_chunk_sizes, batch, heads, dim))

        return (q_chunk_idx + q_chunk_sizes, _query_chunk_flash_attention(q_chunk_idx, q_chunk, k, v, upos, vpos, wpos))

    q, k, v = map(lambda t: rearrange(t, 'b n h d -> n b h d'), (q, k, v))
    
    _, (out, lse) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = rearrange(out, 'c n b h d -> b (c n) h d')
    lse = rearrange(lse, 'c n b h -> b h (c n)')

    return out, lse

@custom_vjp
@jit
def flash_attention(q, k, v, upos, vpos, wpos):
  assert q.shape[0] == k.shape[0] == v.shape[0], 'batch dimension of q, k, v must be the same'
  assert q.shape[1] == k.shape[1] == v.shape[1], 'sequence length dimension of q, k, v must be the same'
  assert q.shape[2] == k.shape[2] == v.shape[2], 'heads dimension of q, k, v must be the same'
  assert q.shape[3] == k.shape[3], 'feature dimension of q, k must be the same'
  out, _ = _flash_attention(q, k, v, upos, vpos, wpos)
  return out

@jit
def flash_attention_forward(q, k, v, upos, vpos, wpos):
    out, lse = _flash_attention(q, k, v, upos, vpos, wpos)
    return out, (q, k, v, upos, vpos, wpos, out, lse)

def _query_chunk_flash_attention_backward(q_chunk_idx, q, k, v, upos, vpos, wpos, o, do, lse):
    q_chunk_len, batch, heads, dim, k_len, v_dim, pos_emb_dim = *q.shape, v.shape[0], v.shape[-1], wpos.shape[-1]

    scale = 1 / jnp.sqrt(dim)
    q_scaled = q * scale

    center_widths, q_pos = prepare_basis(q_chunk_idx, q_chunk_len, k_len, batch, pos_emb_dim)

    def chunk_scanner(carries, _):
        k_chunk_idx, dq, dupos, dvpos, dwpos = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (k_chunk_idx, batch, heads, 0), slice_sizes=(k_chunk_sizes, batch, heads, dim))  # (K,B,H,D)
        v_chunk = lax.dynamic_slice(v, (k_chunk_idx, batch, heads, 0), slice_sizes=(k_chunk_sizes, batch, heads, v_dim))  # (K,B,H,D)
        
        basis_chunk = make_chunked_basis(center_widths, q_pos, k_chunk_idx, k_chunk_sizes)  # (B,L,K,D)

        r = einsum('hdn,bijn->bhijd', wpos, basis_chunk)  # (B,H,L,K,D)
        qr_term = einsum('ibhd,bhijd->ibhj', q, r)  # (L,B,H,K)
        uk_term = einsum('hd,jbhd->bhj', upos, k_chunk)[None,:,:,:]  # (1,B,H,K)
        vr_term = einsum('hd,bhijd->ibhj', vpos, r)  # (L,B,H,K)

        bias = qr_term + uk_term + vr_term  # (L,B,H,K)
        bias = bias * scale

        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk) + bias  # (L,B,H,K)

        p = jnp.exp(attn_weights - lse)  # (L,B,H,K)

#        jax.debug.print("back: p={}", rearrange(p,'l b h k -> b h l k'))
#        jax.debug.print("back: bias={}", rearrange(bias,'l b h k -> b h l k'))

        dv_chunk = einsum('i ... j, i ... d -> j ... d', p, do)  # (K,B,H,D)
        dp = einsum('i ... d, j ... d -> i ... j', do, v_chunk)  # (L,B,H,K)

        D = jnp.sum(do * o, axis = -1, keepdims = True)  # (L,B,H,1)
        ds = p * scale * (dp - D)  # (L,B,H,K)

#        jax.debug.print("back: p={}", p)
#        jax.debug.print("back: dp={}", dp)
#        jax.debug.print("back: ds={}", ds)

        dq_chunk = einsum('i ... j, j ... d -> i ... d', ds, k_chunk)  # (L,B,H,D)
        dq_chunk = dq_chunk + einsum('ibhj,bhijd->ibhd', ds, r)

        dk_chunk = einsum('i ... j, i ... d -> j ... d', ds, q)  # (K,B,H,D)
        dk_chunk = dk_chunk + einsum('ibhj,hd->jbhd', ds, upos)

        dupos_chunk = einsum('ibhj,jbhd->hd', ds, k_chunk)  # (H,D)
        dvpos_chunk = einsum('ibhj,bhijd->hd', ds, r)  # (H,D)
        dwpos_chunk = einsum('ibhj,ibhd,bijn->hdn', ds, q + vpos[None,None,:,:], basis_chunk)  # (H,D,N)

        return (k_chunk_idx + k_chunk_sizes, dq + dq_chunk, dupos + dupos_chunk, dvpos + dvpos_chunk, dwpos + dwpos_chunk), (dk_chunk, dv_chunk)

    dq = jnp.zeros_like(q)
    dupos = jnp.zeros_like(upos)
    dvpos = jnp.zeros_like(vpos)
    dwpos = jnp.zeros_like(wpos)

    (_, dq, dupos, dvpos, dwpos), (dk, dv) = lax.scan(chunk_scanner, init = (0, dq, dupos, dvpos, dwpos), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    dk = rearrange(dk, 'c n ... -> (c n) ...')
    dv = rearrange(dv, 'c n ... -> (c n) ...')
    return dq, dk, dv, dupos, dvpos, dwpos

@jit
def flash_attention_backward(res, do):
    q, k, v, upos, vpos, wpos, o, lse = res

    batch, q_len, heads, dim = q.shape

    lse = rearrange(lse, 'b h n -> n b h 1')

    q, k, v, o, do = map(lambda t: rearrange(t, 'b n h d -> n b h d'), (q, k, v, o, do))
    
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    dupos = jnp.zeros_like(upos)
    dvpos = jnp.zeros_like(vpos)
    dwpos = jnp.zeros_like(wpos)

    def chunk_scanner(carries, _):
        chunk_idx, dk, dv, dupos, dvpos, dwpos = carries

        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, q.shape[-1]))
        lse_chunk = lax.dynamic_slice(lse, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, do.shape[-1]))

        dq_chunk, dk_chunk, dv_chunk, dupos_chunk, dvpos_chunk, dwpos_chunk = _query_chunk_flash_attention_backward(chunk_idx, q_chunk, k, v, upos, vpos, wpos, o_chunk, do_chunk, lse_chunk)
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk, dupos + dupos_chunk, dvpos + dvpos_chunk, dwpos + dwpos_chunk), dq_chunk

    (_, dk, dv, dupos, dvpos, dwpos), dq = lax.scan(chunk_scanner, init = (0, dk, dv, dupos, dvpos, dwpos), xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    dq = rearrange(dq, 'c n b h d -> b (c n) h d')
    dk, dv = map(lambda t: rearrange(t, 'n b h d -> b n h d'), (dk, dv))

    return dq, dk, dv, dupos, dvpos, dwpos

flash_attention.defvjp(flash_attention_forward, flash_attention_backward)