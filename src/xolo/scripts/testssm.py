import sys
import os
from functools import partial

import jax
from jax import random
from jax import value_and_grad
import jax.numpy as jnp
import einops

sys.path.append (os.path.dirname(__file__) + '/..')
from selectssm import ssm_chunked_scan
from ssmrecscan import ssm_recursive_scan, ssm_scan, compute_alpha_beta

from jsonargparse import CLI

def naive_ssm_scan (x, Acoeff, Bcoeff, Ccoeff, Delta):
    B, L, D = x.shape
    N = Acoeff.shape[1]
    x = einops.rearrange (x, 'b l d -> l b d')
    Bcoeff = einops.rearrange (Bcoeff, 'b l n -> l b n')
    Ccoeff = einops.rearrange (Ccoeff, 'b l n -> l b n')
    Delta = einops.rearrange (Delta, 'b l d -> l b d')
    alpha, beta = compute_alpha_beta (x, Acoeff, Bcoeff, Delta)  # (L,B,D,N), (L,B,D,N)
    def scan (h_tMinus1, chunk):
        alpha_t, beta_t, C_t = chunk
        h_t = h_tMinus1 * alpha_t + beta_t
        return h_t, jnp.einsum ('bn,bdn->bd', C_t, h_t)
    _h, ys = jax.lax.scan (scan, jnp.zeros((B,D,N)), (alpha, beta, Ccoeff))
    return einops.rearrange (ys, 'l b d -> b l d')

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
    B = 2,
    D = 8,
    N = 8,
    L = 128,
    Delta_min = 0.001,
    Delta_max = 0.1,
    show_inputs = False,
    show_outputs = False,
    show_grads = False,
):
    key_gen = PRNGKeyGenerator(seed)

    A_noise_scale = 0.1

    x = random.normal(next(key_gen), (B, L, D))
    Acoeff = -(jnp.repeat (jnp.arange(1,N+1)[None,:], D, axis=0) + random.normal(next(key_gen), (D, N)) * A_noise_scale)
    Bcoeff = random.normal(next(key_gen), (B, L, N))
    Ccoeff = random.normal(next(key_gen), (B, L, N))
    Delta = random.uniform (next(key_gen), shape=(B,L,D), minval=Delta_min, maxval=Delta_max)

    if show_inputs:
        print(f'x: {x}')
        print(f'Acoeff: {Acoeff}')
        print(f'Bcoeff: {Bcoeff}')
        print(f'Ccoeff: {Ccoeff}')
        print(f'Delta: {Delta}')

    fn1_value_and_grad, fn2_value_and_grad = map(partial(value_and_grad_wrapper, argnums = (0, 1, 2, 3, 4)), (fn1, fn2))

    args = (x, Acoeff, Bcoeff, Ccoeff, Delta)

    y1, grads1 = fn1_value_and_grad(*args)
    y2, grads2 = fn2_value_and_grad(*args)

    if show_outputs:
        print(f'y1: {fn1(*args)}')
    
    if show_grads:
        dx1, dA1, dB1, dC1, dDelta1 = grads1
        print(f'dx1: {dx1}')
        print(f'dA1: {dA1}')
        print(f'dB1: {dB1}')
        print(f'dC1: {dC1}')
        print(f'dDelta1: {dDelta1}')

    return diff(y1, y2), [diff(*args) for args in zip(grads1, grads2)]

def eval_fn (fname, min_recursion_length, recursive_split):
    f = eval(fname)
    if fname == 'ssm_scan' or fname == 'ssm_recursive_scan':
        if min_recursion_length is not None:
            f = partial(f, min_recursion_length = min_recursion_length)
        if recursive_split is not None:
            f = partial(f, recursive_split = recursive_split)
    return f

def main(seed: int = 42,
         batch: int = 2,
         len: int = 128,
         dim: int = 512,
         hidden: int = 8,
         Delta_min: float = 0.001,
         Delta_max: float = 0.1,
         first: str = 'ssm_scan',
         second: str = 'ssm_recursive_scan',
         min_recursion_length: int = None,
         recursive_split: int = None,
         show_inputs: bool = False,
         show_outputs: bool = False,
         show_grads: bool = False,
        ):
    """
    Compare ssm_chunked_scan, ssm_recursive_scan and ssm_scan

    Args:
        seed: random number generator seed
        batch: batch size
        len: sequence length
        dim: feature dimension
        hidden: hidden dimension
        Delta_min: minimum value for Delta
        Delta_max: maximum value for Delta
        first: first function to compare: ssm_scan, ssm_recursive_scan, ssm_chunked_scan, or naive_ssm_scan
        second: second function to compare: ssm_scan, ssm_recursive_scan, ssm_chunked_scan, or naive_ssm_scan
        min_recursion_length: minimum length for recursion (ssm_scan and ssm_recursive_scan only)
        recursive_split: split length for recursion (ssm_scan and ssm_recursive_scan only)
        show_inputs: show values of A, B, C, Delta, and x
        show_outputs: show values of y1
        show_grads: show values of dx1, dA1, dB1, dC1, and dDelta1
    """
    diff, (dx_diff, dA_diff, dB_diff, dC_diff, dDelta_diff) = value_and_grad_difference(
        eval_fn(first, min_recursion_length, recursive_split),
        eval_fn(second, min_recursion_length, recursive_split),
        seed = seed,
        B = batch,
        D = dim,
        N = hidden,
        L = len,
        Delta_min = Delta_min,
        Delta_max = Delta_max,
        show_inputs = show_inputs,
        show_outputs = show_outputs,
        show_grads = show_grads,
    )

    print('shows differences between SSM scan implementations for y, dx, dA, dB, dC, dDelta: ')
    print(f'y: {diff}')       # 0
    print(f'dx: {dx_diff}')   # < 1e-6
    print(f'dA: {dA_diff}')   # < 1e-6
    print(f'dB: {dB_diff}')   # < 1e-5
    print(f'dC: {dC_diff}')   # < 1e-5
    print(f'dDelta: {dDelta_diff}')  # < 1e-4

if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')
