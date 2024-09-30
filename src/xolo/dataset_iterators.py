import math

import numpy as np

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

import logging

def round_robin_iter (seqDatasets, batch_size):
    add_splice_sites = len(seqDatasets) > 0 and seqDatasets[0].add_splice_sites
    n_xxj_targets = len(seqDatasets) > 0 and seqDatasets[0].num_xxj_targets
    target_length = len(seqDatasets) > 0 and seqDatasets[0].target_length
    def makeIter(sd):
        for batch in tfds.as_numpy (sd.dataset):
            yield batch
        sd.make_dataset()
    def roundRobinIter():
        iterators = [makeIter(sd) for sd in seqDatasets]
        while len(iterators) > 0:
            iter = iterators.pop(0)
            try:
                yield next (iter)
                iterators.append (iter)
            except StopIteration:
                pass
    def combineBatchIter(iter):
        while True:
            try:
                batch = [next(iter) for _ in range (batch_size)]
                # Each element in batch is a tuple b = (x,y,xxj_coords,xxj_counts) with shapes as follows:
                #  b[0] = x: (1, seq_length, seq_depth)
                #  b[1] = y: (1, target_length, num_targets)
                #  b[2] = xxj_coords: (1, N, 3) where each row has the form (t,d,a) where t=target, d=donor bin, a=acceptor bin
                #  b[3] = xxj_counts: (1, N) where each element is the count of the junction at the corresponding row in xxj_coords
                # We want to concatenate all these on the first (batch) axis, and add an initial batch index to each row of the the xxj_coords
                # Optionally we also want to append the donor and acceptor counts to y
                xs = np.concatenate ([b[0] for b in batch], axis=0)
                ys = np.concatenate ([b[1] for b in batch], axis=0)
                xxj_coords = np.concatenate ([np.concatenate([n*np.ones ((b[2].shape[1], 1), dtype=np.uint16),
                                                              np.squeeze (b[2], axis=0)], axis=-1) for n, b in enumerate(batch)], axis=0)
                xxj_counts = np.concatenate ([np.squeeze(b[3],axis=0) for b in batch], axis=0)

                # append donor and acceptor count vectors onto targets
                if add_splice_sites:
                    donor_count = np.zeros ((batch_size, target_length, n_xxj_targets), dtype=np.float32)
                    acceptor_count = np.zeros ((batch_size, target_length, n_xxj_targets), dtype=np.float32)
                    for (batch, track, donor, acceptor), count in zip(xxj_coords, xxj_counts):
                        donor_count[batch, donor, track] += count
                        acceptor_count[batch, acceptor, track] += count
                    ys = np.concatenate([ys, donor_count, acceptor_count], axis=-1)

                yield xs, ys, xxj_coords, xxj_counts
            except StopIteration:
                break
    while True:
        yield combineBatchIter(roundRobinIter())

# wrap the validation data iterator, so we can switch it out with one that makes random data for debugging purposes
def real_data_iter(seqDatasets):
    def makeIter():
        for sd in seqDatasets:
            for batch in tfds.as_numpy (sd.dataset):
                yield batch
            sd.make_dataset()
    while True:
        yield makeIter()

# TODO: remove jax dependence from this function, in case running it in a thread causes OOM
def fake_data_iter(seqDatasets,prng,seq_length,seq_depth,target_length,num_targets,lam=1.):
    batch_size = seqDatasets[0].batch_size
    x_rng, y_rng = jax.random.split (prng)
    def makeIter(k):
        x_rng_k = jax.random.fold_in (x_rng, k)
        y_rng_k = jax.random.fold_in (y_rng, k)
        j = 0
        for sd in seqDatasets:
            for i in range (0, sd.num_seqs, sd.batch_size):
                batch_size = jnp.minimum (sd.batch_size, sd.num_seqs - i)
                batch_x_rng = jax.random.fold_in (x_rng_k, j)
                batch_y_rng = jax.random.fold_in (y_rng_k, j)
                j = j + 1
                x = jax.nn.one_hot (jax.random.randint (batch_x_rng, (batch_size, seq_length), 0, seq_depth), seq_depth)
                y = jax.random.poisson (batch_y_rng, lam, (batch_size, target_length, num_targets))
                xxj_coords = []
                xxj_counts = []
                yield x, y, xxj_coords, xxj_counts
    k = 0
    while True:
        yield makeIter(k)
        k = k + 1

def batch_limiter(iter,limit,first):
    def makeIter(batchIter):
        for i,batch in enumerate(batchIter):
            if i+1 >= (first or 0):
                yield batch
            if limit and i+1 >= limit + (first or 0):
                logging.warning(f"STOPPING at batch {i+1}")
# Commented out because it was slowing down --batch_limit. This may result in some datasets being left in a weird state when this option is used.
#                while next(batchIter,False):  # fast wind to end, resetting all datasets
#                    pass
                break
    while True:
        yield makeIter(next(iter))

def count_batches(seqDatasets,batch_size,limit=None,first=None):
    n = sum (math.ceil (sd.num_seqs / batch_size) for sd in seqDatasets)
    if first is not None:
        n = max (n - first, 0)
    if limit is not None:
        n = min (n, limit)
    return n