import os
import sys
import json
import logging
from itertools import chain
import numpy as np

import jax
import jax.numpy as jnp

sys.path.append (os.path.dirname(__file__) + '/..')
from xxj import JunctionCountsModel
from poisson import weighted_poisson_loss

seqlen = 8192
crop = 2048
n_batches = 2
n_in_dims = 768
n_out_tracks = 44
n_counts = 8658

def n_pairs (max_len):
    return max_len * (max_len + 1) // 2

def index_to_pair (idx):
    y = int((2 * idx + 0.25) ** 0.5 - 0.5)
    x = idx - y * (y + 1) // 2
    return x, y

def uniform_sampler():
    while True:
        idx = np.random.randint (n_pairs (seqlen))
        x, y = index_to_pair (idx)
        yield np.random.randint(n_batches), np.random.randint(n_out_tracks), x, y

uniform_sample_generator = uniform_sampler()

rng = np.random.default_rng()
x = jnp.array (rng.standard_normal((n_batches, seqlen, n_in_dims)).astype(np.float32))
xxj_coords = jnp.array ([next(uniform_sample_generator) for _ in range(n_counts)], dtype=np.uint16)
xxj_counts = jnp.concatenate ([rng.standard_gamma(1., (n_counts,1), dtype=np.float32), jnp.ones((n_counts,1), dtype=np.float32)], axis=-1)

model = JunctionCountsModel(features=n_out_tracks)
prng = jax.random.PRNGKey (42)
vars = model.init (prng, x, xxj_coords)

def loss (params):
    return weighted_poisson_loss (model.apply ({ 'params': params }, x, xxj_coords), xxj_counts[:,0], xxj_counts[:,1])

print('loss:', loss(vars['params']))

loss_grad = jax.grad (loss)
loss_grad_jit = jax.jit (loss_grad)

print('loss_grad:', loss_grad_jit(vars['params']))