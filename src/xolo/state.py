import os
import time
import psutil
import logging
import pickle
from datetime import timedelta
from dataclasses import field
import random
import math
from copy import deepcopy
from itertools import chain, accumulate
from bisect import bisect_right
import heapq
import pdb
import numpy as np
import threading

from memprof import display_top
import tracemalloc
import cProfile

import jax
import jax.numpy as jnp
from flax.training import train_state

import einops

import tensorflow as tf

from dna import stochastic_revcomp_batch
from poisson import compute_xy_moments, zero_xy_moments, pearson_r, r_squared

class TrainState(train_state.TrainState):
    revcomp_prng: jax.Array
    dropout_prng: jax.Array
    strand_pair: jax.Array
    n_xxj_targets: int = 0
    max_shift: int = 0
    max_epochs: int = None
    max_seconds: int = None
    prevalidate: bool = False
    patience: int = None
    batch_stats: dict = field(default_factory=dict)
    last_y_pred: any = None
    last_grads: any = None
    last_diagnostics: any = None
    last_pearsonR_moments: any = None
    last_xxj_pearsonR_moments: any = None
    last_losses: any = None

    def vars (self):
        return { 'params': self.params,
                 'batch_stats': self.batch_stats }


class TrainLogger():
    def __init__(self, save_filename: str = None, device_prof_dir: str = None,
                max_device_prof_epoch = 1, max_device_prof_batch = 1,
                trace_prof_dir: str = None, summary_dir: str = None, log_dir: str = None,
                summary_period: int = 100, summaries: dict = {}, diagnostics: dict = None, verbose: bool = False,
                global_clip: float = None, block_clip: float = None, memory_stats: bool = True):
        self.save_filename = save_filename
        self.device_prof_dir = device_prof_dir
        self.max_device_prof_epoch = max_device_prof_epoch
        self.max_device_prof_batch = max_device_prof_batch
        self.trace_prof_dir = trace_prof_dir
        self.summary_dir = summary_dir
        self.summary_period = summary_period
        self.summaries = summaries
        self.diagnostics = diagnostics
        self.log_dir = log_dir
        self.verbose = verbose
        self.global_clip = global_clip
        self.block_clip = block_clip
        self.memory_stats = memory_stats

        self.summaryWriter = None
        if device_prof_dir is not None:
            os.makedirs(device_prof_dir,exist_ok=True)
        if trace_prof_dir is not None:
            os.makedirs(trace_prof_dir,exist_ok=True)
        if summary_dir is not None:
            os.makedirs(summary_dir,exist_ok=True)
            self.summaryWriter = tf.summary.create_file_writer(summary_dir)
            self.summaryWriter.set_as_default()

    def __del__ (self):
        if self.trace_prof_dir is not None:
            jax.profiler.stop_trace()

    def make_save_filename (self, epoch = None):
        filename = self.save_filename
        if epoch is not None:
            filename = f"{filename}.ep{epoch}"
        return filename

    def save_vars (self, vars, epoch = None):
        if self.save_filename is not None:
            filename = self.make_save_filename(epoch)
            with open (filename, mode="wb") as f:
                pickle.dump (vars, f)
                f.close()

    def infer_starting_epoch (self):
        epoch = 0
        if self.save_filename is not None:
            while os.path.isfile(self.make_save_filename(epoch+1)):
                epoch = epoch + 1
        return epoch

    def writeSummary(self,d,path=[],**kwargs):
        if self.summaryWriter is not None and self.includeSummaries(path):
            if type(d) == dict:
                for k in d.keys():
                    if k == "filter":  # hack: handle filters specially as images
                        filter = d[k]  # (1,L,O,D)
                        fmin = jnp.min(filter,axis=1)[:,None,:,:]  # (1,1,O,D)
                        fmax = jnp.max(filter,axis=1)[:,None,:,:]  # (1,1,O,D)
                        filter = (filter - fmin) / (fmax - fmin)   # rescale each filter to [0,1]
                        tf.summary.image('/'.join(path+[k]),einops.rearrange (filter, "1 l o d -> 1 (o d) l 1"),**kwargs)
                    else:
                        self.writeSummary(d[k],path+[k],**kwargs)
            elif jnp.isscalar(d) or (isinstance(d,jnp.ndarray) and d.size == 1):
                tf.summary.scalar('/'.join(path),d,**kwargs)
            elif isinstance(d,jnp.ndarray):
                tf.summary.histogram('/'.join(path),d,**kwargs)
            elif isinstance(d,tuple):
                for n, x in enumerate(d):
                    self.writeSummary(x,path+[str(n)],**kwargs)
            else:
                logging.warning(f"ignoring {type(d)} at {'/'.join(path)}")

    def includeSummaries(self,path):
        d = self.summaries
        for p in path:
            if p not in d and "*" not in d:
                return False
            d = d[p] if p in d else d["*"]
            if d == True:
                return True
        return True

    def writeBatchSummaries(self, state: TrainState, batch_num: int, loss: float, R: float, R2: float, R_xxj: float = None):
        if batch_num % self.summary_period == 0:
            self.writeSummary (loss, path=['batch','loss'], step=batch_num)
            self.writeSummary (state.last_y_pred, path=['batch','y_pred'], step=batch_num)
            self.writeSummary (state.last_grads, path=['batch','grad'], step=batch_num)
            self.writeSummary (state.last_diagnostics, path=['batch','diagnostics'], step=batch_num)
            self.writeSummary (R, path=['batch','R','all'], step=batch_num)
            self.writeSummary (R2, path=['batch','R2','all'], step=batch_num)
            if R_xxj is not None:
                self.writeSummary (R_xxj, path=['batch','R_xxj'], step=batch_num)
            self.writeSummary (pearson_r(state.last_pearsonR_moments,keep_features=True), path=['batch','R','by_feature'], step=batch_num)
            self.writeSummary (r_squared(state.last_pearsonR_moments,keep_features=True), path=['batch','R2','by_feature'], step=batch_num)
            if state.batch_stats:
                self.writeSummary (state.batch_stats, path=['batch','batchnorm_stats'], step=batch_num)
            self.writeSummary (state.params, path=['batch','params'], step=batch_num)
            self.writeSummary (state.last_pearsonR_moments[:,3], path=['batch','l2_outputs'], step=batch_num)
        
    def writeEpochSummaries(self, state: TrainState, epoch_num: int, vmetrics: dict, tmetrics: dict):
        self.writeSummary (vmetrics, path=['epoch','vmetrics'], step=epoch_num)
        self.writeSummary (tmetrics, path=['epoch','tmetrics'], step=epoch_num)
        self.writeSummary (state.params, path=['epoch','params'], step=epoch_num)

    def writeMemoryStats(self):
        if self.memory_stats:
            logging.warning('Device memory stats:')
            for i,d in enumerate(jax.devices()):
                m = d.memory_stats()
                for x in m.keys():
                    logging.warning(f"device {i}: {x} {m[x]}")


def train_step (state, loss_fn, x, y, xxj_loss_fn, xxj_loss_weight, xxj_coords, xxj_counts, xxj_weights):
    revcomp_prng = jax.random.fold_in (state.revcomp_prng, state.step)
    dropout_prng = jax.random.fold_in (state.dropout_prng, state.step)

    def train_loss (params, x, y_true, xxj_coords, xxj_counts, xxj_weights):
        batch_size = x.shape[0]
        v = { 'params': params, 'losses': {}, 'diagnostics': {}, 'batch_stats': state.batch_stats }
        apply_args = { 'train': True, 'mutable': ['losses','batch_stats','diagnostics'], 'rngs': { 'dropout': dropout_prng } }
        if xxj_loss_fn is not None:
            pred, out_vars = state.apply_fn (v, x, xxj_coords, **apply_args)
            y_pred, xxj_counts_pred = pred
        else:
            y_pred, out_vars = state.apply_fn (v, x, **apply_args)
        loss = loss_fn (y_pred, y_true) + sum(jax.tree_util.tree_leaves(out_vars['losses']))
        if xxj_loss_fn is not None:
            xxj_loss = xxj_loss_fn (xxj_counts_pred, xxj_counts, xxj_weights[:,None]) / batch_size
            jax.debug.print ("loss {} xxj_loss {}", loss, xxj_loss)
            loss = (1-xxj_loss_weight) * loss + xxj_loss_weight * xxj_loss
            out_vars['xxj_pearsonR_moments'] = compute_xy_moments (xxj_counts_pred, xxj_counts, weights=xxj_weights[:,None])
        out_vars['pearsonR_moments'] = compute_xy_moments (y_pred, y_true)
        out_vars['last_y_pred'] = y_pred
        return loss, out_vars
    
    loss_value_and_grad = jax.value_and_grad (train_loss, has_aux=True)

    x, y, _revcomp_flag, _shift, xxj_coords, xxj_counts = stochastic_revcomp_batch (revcomp_prng, x, y, state.strand_pair, xxj_coords, xxj_counts, max_shift=state.max_shift)
    (loss, out_vars), grads = loss_value_and_grad (state.params, x, y, xxj_coords, xxj_counts, xxj_weights)
    # create and return new state
    state = state.apply_gradients (grads = grads)
    state = state.replace (last_grads = grads,
                            last_diagnostics = out_vars['diagnostics'],
                            last_losses = out_vars['losses'],
                            last_pearsonR_moments = out_vars['pearsonR_moments'],
                            last_xxj_pearsonR_moments = out_vars.get('xxj_pearsonR_moments',None),
                            last_y_pred = out_vars['last_y_pred'],
                            batch_stats = out_vars['batch_stats'])
    return loss, state

# Augment the xxj_coords and xxj_counts arrays, which represent the nonzero elements of a sparse matrix,
# with zero-count decoys at random locations, c.f. contrastive learning.
# Om input, the xxj_coords array has shape (n_positives,4) with each row consisting of (batch index, track index, donor bin, acceptor bin)
# Tracks alternate between forward- and reverse-strand, with donor<=acceptor on forward-strand tracks and donor>=acceptor on reverse-strand tracks
# The orientation of a track is determined by the track index modulo 2, with 0 indicating forward-strand and 1 indicating reverse-strand
# On output, we want xxj_coords to have shape (n_positives+n_decoys,3) with each row consisting of (batch,donor,acceptor)
# and xxj_counts to have shape (n_positives+n_decoys,n_tracks) with each row consisting of the counts for the corresponding track.
# Decoys are drawn by combining a number of different possible interval sets, using combinations of donor/acceptor sites drawn uniformly
# or from intervals in the batch, enriched for intervals enclosed by positive intervals.
def augment_xxj_counts (xxj_coords, xxj_counts, n_batches: int = None, n_out_bins: int = None, n_out_tracks: int = None,
                        top_n_per_track: int = 3, top_n_per_batch_ratio: int = 2, sampled_per_batch_ratio: int = 2,
                        decoys_per_positive: float = 1, no_randoms: bool = False):
    top_n_per_batch = top_n_per_track * n_out_tracks * top_n_per_batch_ratio
    max_positives_per_batch = top_n_per_batch * sampled_per_batch_ratio
    # Partition positives by batch and track, also aggregating columns, then sort each track by count
    logging.warning ("Sorting positives")
    positives_by_bt = [[[] for _t in range(n_out_tracks)] for _b in range(n_batches)]
    columns_by_batch = [{} for _b in range(n_batches)]
    for n, c in enumerate(xxj_coords):
        b, t, d, a = c
        count = xxj_counts[n]
        interval = (min(d,a), max(d,a))
        positives_by_bt[b][t].append((count,interval))
        count, column = columns_by_batch[b].get (interval, (0, np.zeros(n_out_tracks, dtype=np.float32)))
        column[t] += count
        columns_by_batch[b][interval] = (count+1, column)
    augmented_xxj_coords = []
    augmented_xxj_counts = []
    for b in range(n_batches):
        interval_count_and_column = columns_by_batch[b]
        batch_intervals = list(interval_count_and_column.keys())
        sampled = {}
        if len(batch_intervals) <= max_positives_per_batch:
            for interval in batch_intervals:
                sampled[interval] = interval_count_and_column[interval][1]
        else:
            logging.warning (f"Finding top {top_n_per_track} positives for each track in batch {b}")
            top_intervals_by_track = [heapq.nlargest(top_n_per_track,track) for track in positives_by_bt[b]]
            for track in top_intervals_by_track:
                for (_count, interval) in track:
                    if interval not in sampled:
                        sampled[interval] = interval_count_and_column[interval][1]
                        del interval_count_and_column[interval]
            logging.warning (f"Picked {len(sampled)} intervals; finding top {top_n_per_batch} positives across all tracks for batch {b}")
            top_n_for_batch = min (top_n_per_batch, len(batch_intervals))
            top_batch_intervals = heapq.nlargest(top_n_for_batch,interval_count_and_column.keys(),key=lambda interval: interval_count_and_column[interval][0])
            for interval in top_batch_intervals:
                if interval not in sampled:
                    sampled[interval] = interval_count_and_column[interval][1]
                    del interval_count_and_column[interval]
                    if len(sampled) >= top_n_per_batch:
                        break
            if not no_randoms:
                remaining_batch_intervals = list(interval_count_and_column.keys())
                n_positives_to_sample = min (max_positives_per_batch - len(sampled), len(remaining_batch_intervals))
                logging.warning (f"Picked {len(sampled)} intervals; sampling {n_positives_to_sample} more positives for batch {b}")
                indices_to_keep = np.random.permutation(len(remaining_batch_intervals))[:n_positives_to_sample]
                for i in indices_to_keep:
                    interval = remaining_batch_intervals[i]
                    sampled[interval] = interval_count_and_column[interval][1]
                    
        # Generate decoys
        if not no_randoms:
            n_decoys = math.ceil (decoys_per_positive * len(sampled))
            logging.warning (f"Sampled {len(sampled)} intervals; generating {n_decoys} decoys for batch {b}")
            uniform_interval_generator = uniform_sampler (n_out_bins)
            batch_starts, batch_ends = tuple (list(set(interval[i] for interval in batch_intervals)) for i in (0,1))
            batch_combo_generator = legal_combo_sampler (batch_starts, batch_ends)
            batch_starts_generator = from_start_sampler (batch_starts, n_out_bins)
            batch_ends_generator = to_end_sampler (batch_ends)
            contained_batch_starts_generator = from_start_contained_sampler (batch_intervals)
            contained_batch_ends_generator = to_end_contained_sampler (batch_intervals)
            contained_generator = contained_sampler (list(sampled.keys()))
            generator_type = {'uniform': uniform_interval_generator,
                            'combo': batch_combo_generator,
                            'starts': batch_starts_generator,
                            'ends': batch_ends_generator,
                            'contained-starts': contained_batch_starts_generator,
                            'contained-ends': contained_batch_ends_generator,
                            'contained': contained_generator}
            generator = list(generator_type.values())
            decoy_count_by_type = [0] * len(generator)
            while n_decoys > 0:
                decoy_type = np.random.randint(len(generator))
                interval = next(generator[decoy_type])
                if interval not in sampled:
                    sampled[interval] = interval_count_and_column.get(interval, (0, np.zeros(n_out_tracks, dtype=np.float32)))[1]
                    n_decoys -= 1
                    decoy_count_by_type[decoy_type] += 1
            logging.warning ("Generated decoys: " + ', '.join([f"{v} {k}" for k,v in zip(generator_type.keys(),decoy_count_by_type)]))
        augmented_xxj_coords.extend ([(b,d,a) for d,a in sampled.keys()])
        augmented_xxj_counts.extend (sampled.values())
    if len(augmented_xxj_coords) == 0:
        xxj_coords = np.zeros ((0,3), dtype=np.uint16)
        xxj_counts = np.zeros ((0,n_out_tracks), dtype=np.float32)
    else:
        xxj_coords = np.array(augmented_xxj_coords,dtype=np.uint16)
        xxj_counts = np.array(augmented_xxj_counts,dtype=np.float32)
    logging.warning("Augmented xxj_coords to shape %s, xxj_counts to shape %s", xxj_coords.shape, xxj_counts.shape)
    return xxj_coords, xxj_counts

# Interval generator functions for augment_xxj_counts
def n_pairs (max_len):
    return max_len * (max_len + 1) // 2
def index_to_pair (idx):
    y = int((2 * idx + 0.25) ** 0.5 - 0.5)
    x = idx - y * (y + 1) // 2
    return x, y
def uniform_sampler (n_out_bins):
    while True:
        idx = np.random.randint (n_pairs (n_out_bins))
        yield index_to_pair (idx)
def by_group_sampler (n_members_by_group, make_interval):
    n_members_at_or_before_group = list(accumulate(n_members_by_group))
    n_members = n_members_at_or_before_group[-1]
    while True:
        idx = np.random.randint (n_members)
        n_group = bisect_right(n_members_at_or_before_group, idx)
        n_member = idx - n_members_at_or_before_group[n_group-1] if n_group > 0 else idx
        yield make_interval (n_group, n_member)
def find_first_valid_end (starts, ends):
    j0 = 0
    for i in range(len(starts)):
        while j0 < len(ends) and ends[j0] < starts[i]:
            j0 += 1
        yield j0
def legal_combo_sampler (starts, ends):
    first_valid_end_idx = [j for j in find_first_valid_end(starts,ends)]
    n_valid_by_start = [len(ends) - j for j in first_valid_end_idx]
    def make_interval (n_start, n_end):
        return starts[n_start], ends[n_end]
    return by_group_sampler (n_valid_by_start, make_interval)
def from_start_sampler (starts, n_out_bins):
    n_valid_by_start = [n_out_bins - i for i in starts]
    def make_interval (n_start, end_offset):
        return starts[n_start], starts[n_start] + end_offset
    return by_group_sampler (n_valid_by_start, make_interval)
def to_end_sampler (ends):
    n_valid_by_end = [i+1 for i in ends]
    def make_interval (n_end, start):
        return start, ends[n_end]
    return by_group_sampler (n_valid_by_end, make_interval)
def contained_sampler (intervals):
    interval_len = [e+1-s for s,e in intervals]
    n_contained_by_interval = [n_pairs(l) for l in interval_len]
    def make_interval (n_interval, n_subinterval):
        interval_start, _interval_end = intervals[n_interval]
        start_offset, end_offset = index_to_pair (n_subinterval)
        return interval_start + start_offset, interval_start + end_offset
    return by_group_sampler (n_contained_by_interval, make_interval)
def from_start_contained_sampler (intervals):
    n_contained_by_interval = [e-s for s,e in intervals]
    def make_interval (n_interval, end_offset):
        interval_start, _interval_end = intervals[n_interval]
        return interval_start, interval_start + end_offset
    return by_group_sampler (n_contained_by_interval, make_interval)
def to_end_contained_sampler (intervals):
    n_contained_by_interval = [e-s for s,e in intervals]
    def make_interval (n_interval, start_offset):
        interval_start, interval_end = intervals[n_interval]
        return interval_start + start_offset + 1, interval_end
    return by_group_sampler (n_contained_by_interval, make_interval)


# Pad the xxj_coords and xxj_counts arrays to a length that is an integer power of length_multiplier
# This avoids excessive jax recompilation of the loss function
# To avoid including the padded values in the loss, we return an extra weights vector which contains 1's for actual counts and 0's for padded counts
def pad_xxj_counts (xxj_coords, xxj_counts, length_multiplier = 2):
    len = xxj_counts.shape[0]
    n_out_tracks = xxj_counts.shape[1]
    assert xxj_coords.shape == (len, 3)
    assert xxj_counts.shape == (len, n_out_tracks)
    if length_multiplier == 2:  # avoid doubling length due to precision errors
        pad_length = 1 << (len-1).bit_length()
    else:
        pad_length = int (np.ceil (length_multiplier ** np.ceil (np.log(len) / np.log(length_multiplier))))
    n_pad_zeros = pad_length - len
    xxj_coords = np.concatenate ([xxj_coords, np.zeros ((n_pad_zeros, 3), dtype=np.uint16)], axis=0)
    xxj_counts = np.concatenate ([xxj_counts, np.zeros ((n_pad_zeros, n_out_tracks), dtype=np.float32)], axis=0)
    xxj_weights = np.concatenate ([np.ones ((len,), dtype=np.float32), np.zeros ((n_pad_zeros,), dtype=np.float32)], axis=0)
    logging.warning("Padded xxj_coords to shape %s, xxj_counts to shape %s, xxj_weights to shape %s", xxj_coords.shape, xxj_counts.shape, xxj_weights.shape)
    return xxj_coords, xxj_counts, xxj_weights

def run_training_loop(state: TrainState, tlog: TrainLogger, loss_fn, xxj_loss_fn, xxj_loss_weight, valid_iter, train_iter, n_valid_batches, n_train_batches, top_n_per_track=3, recompute_train_metrics=False, use_jit=True, use_threads=False, use_tracemalloc=False):
    if use_tracemalloc:
        tracemalloc.start()

    # Validation metrics
    compute_metrics = Metrics (state.apply_fn, loss_fn, xxj_loss_fn, xxj_loss_weight, use_jit=use_jit, n_xxj_targets=state.n_xxj_targets, top_n_per_track=top_n_per_track)
    init_vars = state.vars()
    tlog.save_vars(init_vars)

    vmetrics_history = []
    if state.prevalidate:
        vmetrics_history.append (compute_metrics(init_vars,valid_iter,n_batches=n_valid_batches,fold_name="validation"))
        logging.warning (f"Validation metrics before training: {metrics_str(vmetrics_history[-1])}")

    train_step_jit = jax.jit (train_step, static_argnames=['loss_fn','xxj_loss_fn']) if use_jit else train_step

    logging.warning (f'Initial stats:\n{stats_str(state.vars(),format="{0} mean={2} sd={4} shape={6}")}')
    logging.warning('starting training loop')
    batches = 0
    epochs = init_epochs = tlog.infer_starting_epoch()
    best_vars = state.vars()
    train_eta = ETA(n=state.max_epochs,limit=state.max_seconds)
    while True:
        epoch_eta = ETA(n=n_train_batches)
        train_loss = 0
        n_train_seqs = 0
        batch_iter = enumerate(next(train_iter))
        next_batch = None
        
        def prepare_next_batch():
            nonlocal next_batch, batch_iter
            try:
                i, (x, y, xxj_coords, xxj_counts) = next(batch_iter)
            except StopIteration:
                next_batch = None
                return
            if xxj_loss_fn is not None:
                xxj_counts = np.minimum (xxj_counts, np.finfo(xxj_counts.dtype).max)  # guard against carelessly encoded inf's (ugh)
                xxj_coords, xxj_counts = augment_xxj_counts (xxj_coords, xxj_counts, n_batches=y.shape[0], n_out_bins=y.shape[1], n_out_tracks=state.n_xxj_targets, top_n_per_track=top_n_per_track)
                xxj_coords, xxj_counts, xxj_weights = pad_xxj_counts (xxj_coords, xxj_counts)
                assert all (0<=b<y.shape[0] and 0<=d<y.shape[1] and 0<=a<y.shape[1] for (b,d,a) in xxj_coords)
                assert np.all (np.isfinite (xxj_counts))
                assert np.all (np.isfinite (xxj_weights))
            else:
                xxj_coords = np.zeros ((0,3), dtype=np.uint16)
                xxj_counts = np.zeros ((0,state.n_xxj_targets), dtype=np.float32)
                xxj_weights = np.zeros ((0,), dtype=np.float32)
            next_batch = i, (x, y, xxj_coords, xxj_counts, xxj_weights)
        
        def process_current_batch():
            nonlocal current_batch, state, train_loss, n_train_seqs, batches, epochs
            i, (x, y, xxj_coords, xxj_counts, xxj_weights) = current_batch
            loss, state = train_step_jit (state, loss_fn, x, y, xxj_loss_fn, xxj_loss_weight, xxj_coords, xxj_counts, xxj_weights)
            batch_size = x.shape[0]
            n_train_seqs = n_train_seqs + batch_size
            train_loss = train_loss + loss * batch_size
            used_gb = psutil.virtual_memory().used / 1024 / 1024 / 1024
            if tlog.device_prof_dir is not None and epochs < tlog.max_device_prof_epoch and i < tlog.max_device_prof_batch:
                filename = f"{tlog.device_prof_dir}/train{epochs}-{i}.prof"
                logging.warning(f"Saving device memory profile to {filename}")
                jax.profiler.save_device_memory_profile(filename)
            batches = batches + 1
            R = pearson_r(state.last_pearsonR_moments)
            R2 = r_squared(state.last_pearsonR_moments)
            bsumms = { 'loss': loss, 'R': R, 'R2': R2 }
            if state.last_xxj_pearsonR_moments is not None:
                R_xxj = pearson_r(state.last_xxj_pearsonR_moments)
                xxj_r = f", r_xxj: {R_xxj:.4f}"
                bsumms['R_xxj'] = R_xxj
            else:
                xxj_r = ''
            tlog.writeBatchSummaries (state=state, batch_num=batches, **bsumms)
            path_norms = jax.tree_util.tree_leaves_with_path(jax.tree_map(jnp.linalg.norm, state.last_grads))
            global_norm = jnp.linalg.norm(jnp.array([pn[1] for pn in path_norms]))
            aberrant_norms = "".join([f"\nLarge gradient norm for {'.'.join([k.key if type(k)==jax.tree_util.DictKey else str(k) for k in pn[0]])}: {pn[1]}" for pn in path_norms if pn[1] > tlog.block_clip])
            logging.warning (f"Epoch {epochs+1} batch {i+1}/{n_train_batches} (size {x.shape[0]}) loss: {loss:.6f}, r: {R:.4f}, r2: {R2:.4f}{xxj_r}, norm(grad): {global_norm:.4f}, used {used_gb:.2f} Gb, ETA {epoch_eta(i)}{aberrant_norms}")
            if tlog.summaries or tlog.verbose:
                logging.warning (f'Params:\n{stats_str(state.vars(),format="{0} mean={2} sd={4}")}')
                logging.warning (f'Gradients:\n{stats_str(state.last_grads,format="grad({0}) mean={2} sd={4} l2={1}")}')
                if state.batch_stats:
                    logging.warning (f'Batch stats:\n{stats_str(state.batch_stats,format="batch_stats({0}) l2={1}")}')
                logging.warning (f'Outputs:\n{stats_str(state.last_y_pred,format="output({0}) mean={2} sd={4} l2={1}")}')
                logging.warning (f'Losses (regularizers):\n{stats_str(state.last_losses,format="losses({0}) {2}")}')
            if tlog.diagnostics and state.last_diagnostics:
                logging.warning (f'Diagnostics:\n{leaves_str(state.last_diagnostics)}')
            if batches == 1:
                tlog.writeMemoryStats()  # log memory stats at end of first batch

        prepare_next_batch()
        while next_batch is not None:
#            with cProfile.Profile() as pr:
                current_batch = next_batch
                if use_threads:
                    augment_thread = threading.Thread(target=prepare_next_batch, name='augment')
                    augment_thread.start()
                    process_current_batch()
                    augment_thread.join()
                else:
                    process_current_batch()
                    prepare_next_batch()
#                pr.dump_stats(tlog.log_dir + f"/train{epochs}-{batches}.prof")
                if use_tracemalloc:
                    snapshot = tracemalloc.take_snapshot()
                    display_top(snapshot,limit=5)

        epochs = epochs + 1
        # compute validation loss and metrics
        vmetrics = compute_metrics(state.vars(),valid_iter,n_batches=n_valid_batches,fold_name="validation")
        vmetrics_history.append (vmetrics)

        best_vr_idx = jnp.argmax(jnp.array([vm["pearson_r"] + vm["r_squared"]/4 for vm in vmetrics_history]))
        this_is_best_epoch = (best_vr_idx == len(vmetrics_history) - 1)

        # if requested, recompute training loss with latest params for direct comparison with validation loss, along with training metrics
        train_loss = train_loss / n_train_seqs
        tmetrics_str = ""
        if recompute_train_metrics:
            tmetrics = compute_metrics(state.vars(),train_iter,n_batches=n_train_batches,fold_name="training")
            tlog.writeEpochSummaries (state=state, epoch_num=epochs, tmetrics=tmetrics, vmetrics=vmetrics)
            tmetrics_str = f"recomputed training {metrics_str(tmetrics)}, "

        logging.warning (f"Epoch {epochs}: time {epoch_eta.lapsed()} (total {train_eta.lapsed()}), running-total training loss={train_loss:.6f}, {tmetrics_str}validation {metrics_str(vmetrics_history[-1])}" +
            (" ...nice" if this_is_best_epoch else f" ...best was after {best_vr_idx+init_epochs+1} epochs"))

        tlog.writeMemoryStats()  # log memory stats at end of every epoch

        if state.max_epochs and epochs >= state.max_epochs:
            logging.warning ("Max epochs reached")
            break

        vars = state.vars()
        tlog.save_vars (vars, epoch=epochs)
        if this_is_best_epoch:
            best_vars = vars
            tlog.save_vars (best_vars)
        elif len(vmetrics_history) - best_vr_idx > state.patience + 1:
            logging.warning ("Patience exceeded")
            break

        if train_eta.past_limit():
            logging.warning (f"Max wall clock time exceeded")
            break
    
    logging.warning (f"Training finished after {epochs} epochs, {batches} batches, {train_eta.lapsed()} elapsed time")

    if epochs > 1:
        tlog.writeMemoryStats()  # log memory stats at end of entire run

    return state

class ETA:
    def __init__(self, n=None, limit=None):
        self.start_time = time.time()
        self.limit = limit
        self.n = n
    def __call__(self, i):
        lapsed_secs = self.lapsed_secs()
        eta = lapsed_secs * (self.n-1-i) / (i+1)
        if self.limit is not None:
            eta = min (eta, self.limit - lapsed_secs)
        return timedelta(seconds=eta)
    def lapsed_secs(self):
        return time.time()-self.start_time
    def lapsed(self):
        return timedelta(seconds=self.lapsed_secs())
    def past_limit(self):
        return self.limit is not None and self.lapsed_secs() > self.limit


class Metrics:
    def __init__(self, apply_fn, loss_fn, xxj_loss_fn, xxj_loss_weight, n_xxj_targets, top_n_per_track=None, device_prof_dir=None, use_jit=True):
        self.apply_fn = jax.jit (apply_fn) if use_jit else apply_fn
        self.loss_fn = jax.jit (loss_fn) if use_jit else loss_fn
        self.xxj_loss_fn = jax.jit (xxj_loss_fn) if use_jit and xxj_loss_fn is not None else xxj_loss_fn
        self.xxj_loss_weight = xxj_loss_weight
        self.device_prof_dir = device_prof_dir
        self.top_n_per_track = top_n_per_track
        self.n_xxj_targets = n_xxj_targets

    def __call__ (self, vars, iter, n_batches, fold_name, device_memory_profile_prefix=None, return_per_feature_metrics=False, return_per_seq_metrics=False, warn_if_zero=True):
        eta = ETA(n=n_batches)
        loss = 0
        n_train_seqs = 0
        moments = zero_xy_moments()  # (feature,moment)
        xxj_moments = zero_xy_moments()
        if return_per_seq_metrics:
            by_seq = { "by_seq": { "pearson_r": [], "r_squared": [], "loss": [] }}
        else:
            by_seq = {}
        for i, (x, y_true, xxj_coords, xxj_counts) in enumerate(next(iter)):
            batch_size = x.shape[0]
            if self.xxj_loss_fn is not None:
                # During validation, we double up the top_n_per_track instead of sampling random decoys, in order to get more reproducible results
                xxj_coords, xxj_counts = augment_xxj_counts (xxj_coords, xxj_counts, n_batches=batch_size, n_out_bins=x.shape[1], n_out_tracks=self.n_xxj_targets, no_randoms=True, top_n_per_track=2*self.top_n_per_track)
                xxj_coords, xxj_counts, xxj_weights = pad_xxj_counts (xxj_coords, xxj_counts)
                y_pred, xxj_counts_pred = self.apply_fn (vars, x, xxj_coords)
                assert xxj_counts_pred.shape[0] == xxj_counts.shape[0], f"predicted xxj_counts length {xxj_counts_pred.shape[0]} does not match desired length {xxj_counts.shape[0]}"
            else:
                y_pred = self.apply_fn (vars, x)
            assert y_pred.shape == y_true.shape, f"predicted shape {y_pred.shape} != required shape {y_true.shape}"
            l = self.loss_fn (y_pred, y_true)
            m = compute_xy_moments (y_pred, y_true, warn_if_zero=warn_if_zero)
            moments = moments + m
            if self.xxj_loss_fn is not None:
                l = (1-self.xxj_loss_weight) * l + self.xxj_loss_weight * self.xxj_loss_fn (xxj_counts_pred, xxj_counts, xxj_weights[:,None]) / batch_size
                m_xxj = compute_xy_moments (xxj_counts_pred, xxj_counts, weights=xxj_weights[:,None])
                xxj_moments = xxj_moments + m_xxj
            if return_per_seq_metrics:
                by_seq["by_seq"]["pearson_r"].append (float (pearson_r(m)))
                by_seq["by_seq"]["r_squared"].append (float (r_squared(m)))
                by_seq["by_seq"]["loss"].append (float (l))
            loss = loss + l * batch_size
            n_train_seqs = n_train_seqs + batch_size
            used_gb = psutil.virtual_memory().used / 1024 / 1024 / 1024
            if device_memory_profile_prefix is not None and self.device_prof_dir is not None:
                jax.profiler.save_device_memory_profile(f"{self.device_prof_dir}/{device_memory_profile_prefix}{i}.prof")
            logging.warning (f"computed predictions for {fold_name} batch {i+1}/{n_batches}, r: {pearson_r(m):.4f}, input {x.shape}, output {y_pred.shape}, used {used_gb:.2f} Gb, ETA {eta(i)}")
        R = pearson_r (moments)
        R2 = r_squared (moments)
        loss = loss / n_train_seqs
        logging.warning (f"elapsed time computing {fold_name} loss: {eta.lapsed()}")
        result = { "loss": float(loss), "pearson_r": float(R), "r_squared": float(R2), **by_seq }
        if self.xxj_loss_fn is not None:
            result["pearson_r_xxj"] = float(pearson_r (xxj_moments))
            result["r_squared_xxj"] = float(r_squared (xxj_moments))
        if return_per_feature_metrics:
            n_features = moments.shape[0]
            result["by_feature"] = { "pearson_r": [float(pearson_r(moments[n,:])) for n in range(n_features)],
                                     "r_squared": [float(r_squared(moments[n,:])) for n in range(n_features)] }
        return result

def init_params (prng, model, filename, seq_length, seq_depth, model_predicts_xxj):
    if filename is not None and os.path.isfile(filename):
        logging.warning('loading parameters from {}'.format(filename))
        with open (filename, mode="rb") as f:
            init_vars = pickle.load (f)
            f.close()
    else:
        logging.warning('initializing parameters')
        dummy_x = jnp.zeros ((1, seq_length, seq_depth))
        dummy_xxj_sparse = [jnp.zeros((0,4),dtype=jnp.uint16)] if model_predicts_xxj else []
        init_vars = model.init (prng, dummy_x, *dummy_xxj_sparse, train=False)
        logging.warning('initializing parameters... done')
    return init_vars

def metrics_str (metrics):
    return f"loss={metrics['loss']:.6f} r={metrics['pearson_r']:.4f} r2={metrics['r_squared']:.4f}"

def leaf_mv (kv):
    x = kv[1]
    if type(x) == tuple:
        mv = [leaf_mv(y) for y in x]
        return jnp.mean(mv[0]), jnp.mean(mv[1])
    return jnp.mean(x), jnp.var(x)

def leaf_func (x, f):
    if type(x) == tuple:
        return jnp.mean (jnp.array ([leaf_func(y,f) for y in x]))
    return f(x)

def mean_l2(x):
    return jnp.mean(x*x)

# format args: 0=param_name, 1=mean_of_l2, 2=mean, 3=var, 4=std, 5=std_excluding_nans
def stats_str (vars, format="{} {} {}"):
    return ''.join([format.format('.'.join([k.key for k in kv[0]]),*[leaf_func(kv[1],f) for f in [mean_l2,jnp.mean,jnp.var,jnp.std,jnp.nanstd,jnp.shape]]) + "\n" for kv in jax.tree_util.tree_leaves_with_path(vars,is_leaf=lambda l:type(l) != dict)])

def leaves_str (vars):
    return ''.join(["diagnostic({}) {}\n".format('.'.join([k.key for k in kv[0]]),kv[1]) for kv in jax.tree_util.tree_leaves_with_path(vars,is_leaf=lambda l:type(l) != dict)])
