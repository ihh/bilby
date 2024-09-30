import jax
from jax import nn
from jax import jit, numpy as jnp
from jax.numpy import einsum

from einops import rearrange

EPSILON = 1e-10
MASK_VALUE = -1e10
COSINE_SIM_SCALE = 10

def ref_attention(q, k, v, upos, vpos, wpos):
    return calc_o_from_s (calc_s(q, k, upos, vpos, wpos), v)

def calc_s(q, k, upos, vpos, wpos):
    q = rearrange(q, 'b l h d -> l b h d')
    k = rearrange(k, 'b l h d -> l b h d')

    num_batches, seq_length, dim = q.shape[1], q.shape[0], q.shape[3]

    pos = jnp.arange(seq_length)  # (L,)
    pos = jnp.repeat(pos[None,:],num_batches,axis=0)  # (B,L)

    distance = pos[:, :, None] - pos[:, None, :]  # (B,L,L)
    distance = jnp.repeat(distance[:,:,:,None],dim // 2,axis=-1)  # (B,L,L,Dqk/2)

    pow_rate = jnp.exp(jnp.log((seq_length + 1) / 2) / (dim // 2)).astype("float32")
    center_widths = jnp.power(pow_rate, jnp.arange(1, (dim // 2) + 1, dtype=jnp.float32))  # (Dqk/2,)
    unsigned_basis = jnp.where (jnp.abs(distance) <= center_widths, 1, 0)  # (B,L,L,Dqk/2)
    signed_basis = jnp.sign(distance) * unsigned_basis  # (B,L,L,Dqk/2)
    basis = jnp.concatenate ((unsigned_basis, signed_basis), axis=-1)  # (B,L,L,Dqk)

    r = einsum('de,bije->bijd', wpos, basis)  # (B,L,K,D)
    qr_term = einsum('ibhd,bijd->ibhj', q, r)  # (L,B,H,K)
    uk_term = einsum('hd,jbhd->bhj', upos, k)[None,:,:,:]  # (1,B,H,K)
    vr_term = einsum('hd,bijd->ibhj', vpos, r)  # (L,B,H,K)

    bias = qr_term + uk_term + vr_term  # (L,B,H,K)

    s = einsum('i b h d, j b h d -> i b h j', q, k)  # (L,B,H,L)
    s = s + bias

    return s

def calc_p (s, d):
    p = nn.softmax(s / jnp.sqrt(d), axis = -1)
    return p

def calc_o_from_s (s, v):
    p = calc_p(s,v.shape[-1])
    return calc_o_from_p(p, v)

def calc_o_from_p (p, v):
    o = einsum('i b h j, b j h d -> b h i d', p, v)
    return o
