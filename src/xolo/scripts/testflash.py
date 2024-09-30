import sys
import os
from functools import partial

import jax
from jax import random
from jax import value_and_grad
import jax.numpy as jnp

sys.path.append (os.path.dirname(__file__) + '/..')
from transformer import enformer_attention
from flash_enformer import flash_attention
from attention import ref_attention, calc_o_from_s, calc_o_from_p, calc_s, calc_p

from jsonargparse import CLI

def value_and_grad_wrapper(fn, **kwargs):
    @partial(value_and_grad, **kwargs)
    def inner(*args, **kwargs):
        return jnp.sum(fn(*args, **kwargs))
    return inner

def diff(t1, t2):
    return jnp.max(jnp.abs(t1 - t2))

def PRNGKeyGenerator(seed = 42):
    key = random.PRNGKey(seed)
    while True:
        sub_key, key = random.split(key)
        yield sub_key

def value_and_grad_difference(
    fn1,
    fn2,
    seed = 42,
    batch = 2,
    heads = 4,
    seq_len = 128,
    dim = 512,
    pos_emb_dim = 32,
    uscale = 1.0,
    vscale = 1.0,
    wscale = 1.0,
    show_s = False,
):
    key_gen = PRNGKeyGenerator(seed)

    q = random.normal(next(key_gen), (batch, seq_len, heads, dim))
    k = random.normal(next(key_gen), (batch, seq_len, heads, dim))
    v = random.normal(next(key_gen), (batch, seq_len, heads, dim))

    upos = random.normal(next(key_gen), (heads, dim)) * uscale
    vpos = random.normal(next(key_gen), (heads, dim)) * vscale
    wpos = random.normal(next(key_gen), (heads, dim, pos_emb_dim)) * wscale

    fn1_value_and_grad, fn2_value_and_grad = map(partial(value_and_grad_wrapper, argnums = (0, 1, 2, 3, 4, 5)), (fn1, fn2))

    args = (q, k, v, upos, vpos, wpos)

    o1, grads1 = fn1_value_and_grad(*args)
    o2, grads2 = fn2_value_and_grad(*args)

    if show_s:
        s = calc_s(q, k, upos, vpos, wpos)
        p = calc_p(s, dim)
        ref_grad_s = value_and_grad_wrapper(calc_o_from_s, argnums = 0)(s, v)
        ref_grad_p = value_and_grad_wrapper(calc_o_from_p, argnums = 0)(p, v)
        print('s=', s)
        print('p=', p)
        print('sum(calc_o_from_s(s))', ref_grad_s[0])
        print('sum(calc_o_from_p(p))', ref_grad_p[0])
        print('from fn1', o1)
        print('from fn2', o2)
        print('d(sum(calc_o_from_s(s)))/ds:', ref_grad_s[1])
        print('d(sum(calc_o_from_p(p)))/dp:', ref_grad_p[1])

    return diff(o1, o2), [diff(*args) for args in zip(grads1, grads2)]

def main(seed: int = 42,
         batch: int = 2,
         heads: int = 4,
         len: int = 128,
         dim: int = 512,
         pos_emb_dim: int = 32,
         uscale: float = 1.0,
         vscale: float = 1.0,
         wscale: float = 1.0,
         ref: bool = False,
         show_s: bool = False,
        ):
    """
    Compare attention and flash_attention

    Args:
        seed: random number generator seed
        batch: batch size
        heads: number of heads
        len: sequence length
        dim: feature dimension
        pos_emb_dim: positional embedding dimension
        uscale: scale for positional bias parameter u
        vscale: scale for positional bias parameter v
        wscale: scale for positional bias parameter w
        show_s: show the attention scores and derivatives w.r.t. these scores
        ref: use reference implementation instead of flash attention
    """
    diff, (dq_diff, dk_diff, dv_diff, dupos_diff, dvpos_diff, dwpos_diff) = value_and_grad_difference(
        enformer_attention,
        ref_attention if ref else flash_attention,
        seed = seed,
        batch = batch,
        heads = heads,
        seq_len = len,
        dim = dim,
        pos_emb_dim = pos_emb_dim,
        uscale = uscale,
        vscale = vscale,
        wscale = wscale,
        show_s = show_s,
    )

    print('shows differences between normal and flash attention for output, dq, dk, dv, dupos, dvpos, dwpos')
    print(f'o: {diff}')       # < 1e-4
    print(f'dq: {dq_diff}')   # < 1e-4
    print(f'dk: {dk_diff}')   # < 1e-4
    print(f'dv: {dv_diff}')   # < 1e-5
    print(f'dupos: {dupos_diff}')  # < 1e-4
    print(f'dvpos: {dvpos_diff}')  # < 1e-3
    print(f'dwpos: {dwpos_diff}')  # < 1e-4


if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')
