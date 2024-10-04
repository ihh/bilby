import sys
import os
import socket
import re

import json
from jsonargparse import CLI
from typing import List

import logging
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.profiler

import optax

sys.path.append (os.path.dirname(__file__) + '/..')
from dataset import SeqDataset
from dataset_iterators import fake_data_iter, real_data_iter, round_robin_iter, batch_limiter, count_batches
from models import models
from state import TrainState, TrainLogger, Metrics, init_params, run_training_loop
from dna import ensemble_fwd_rev, ensemble_shift
from poisson import poisson_loss, poisson_multinomial_loss, weighted_poisson_loss

import traceback
import warnings

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def main(data_dir: str = os.path.dirname(__file__)+'/../../../data',
         train: List[str] = ['train'],
         valid: List[str] = ['valid'],
         test: List[str] = ['test'],
         model_name: str = 'testcnn',
         model_args: dict = {},
         poisson: bool = False,
         poisson_weight: float = 0.2,
         load: str = None,
         save: str = None,
         tabulate: bool = False,
         eval: str = None,
         rc_ensemble_eval: bool = True,
         shift_ensemble_eval: bool = True,
         dummy: bool = False,
         logdir: str = None,
         logfile: str = None,
         tlog_args: dict = {},
         device_prof_dir: str = None,
         trace_prof_dir: str = None,
         memory_stats: bool = True,
         use_tracemalloc: bool = False,
         verbose: bool = False,
         summaries: dict = {},
         summary_dir: str = None,
         summary_period: int = 100,
         jax_debug_nans: bool = False,
         jax_log_compiles: bool = False,
         disable_jit: bool = False,
         use_threads: bool = False,
         warn_trace: bool = False,
         learn_rate: float = 0.0001,
         warmup_steps: float = 10000,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         block_clip: float = 5.0,
         global_clip: float = 10.0,
         batch_size: int = 2,
         first_batch: int = None,
         batch_limit: int = None,
         max_epochs: int = None,
         max_seconds: float = None,
         prevalidate: bool = False,
         patience: int = 25,
         recompute_train_loss: bool = False,
         max_shift: int = 3,
         rng_key: int = 42,
         rnd_valid: bool = False):
    """
    Train neural network

    Args:
        data_dir: data directory
        train: training dataset split label(s)
        valid: validation dataset split label(s)
        test: test dataset split label(s)
        model_name: specify model
        model_args: specify model arguments (JSON-formatted object)
        poisson: use Poisson loss rather than Poisson-multinomial loss
        poisson_weight: relative weight of Poisson term in Poisson-multinomial loss
        load: load initial parameters
        save: save final parameters (and load from this file if available)
        tabulate: show model layers as table, then stop
        eval: save fine-grained test data metrics to file, then stop
        rc_ensemble_eval: average forward and reverse complement predictions for evaluation
        shift_ensemble_eval: average predictions over shifted sequences for evaluation
        dummy: dummy run; do not save checkpoints
        logdir: logfile directory for progress messages and metrics (not params)
        logfile: logfile name within logdir
        device_prof_dir: save jax.profiler device memory profiles during validation and training
        trace_prof_dir: save jax.profiler trace profiles during validation and training
        memory_stats: report final device memory statistics
        use_tracemalloc: enable tracemalloc for memory profiling
        verbose: log extra diagnostics during training
        summary_dir: save TensorBoard summaries during training
        summaries: specify which TensorBoard summaries to save (JSON-formatted object)
        tlog_args: other miscellaneous TrainLogger arguments (JSON-formatted object)
        summary_period: how frequently to save TensorBoard and other summaries
        jax_debug_nans: enable NaN debugging in JAX
        disable_jit: disable JAX JIT compilation
        use_threads: use separate thread for data loading and augmentation
        warn_trace: provide full traceback for warnings
        learn_rate: learning rate
        warmup_steps: number of warmup steps for learning rate schedule
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        block_clip: block gradient norm clipping parameter
        global_clip: global gradient norm clipping parameter
        batch_size: batch size
        first_batch: start training at specified batch
        batch_limit: truncate training & validation sets to specified number of batches
        max_epochs: max number of epochs
        max_seconds: max number of seconds on wall clock
        patience: patience (number of epochs that a validation loss increase will be forgiven)
        recompute_train_loss: recompute training loss at end of epoch
        max_shift: maximum stochastic shift for training DNA sequences
        rng_key: pseudorandom number key
        rnd_valid: use random data instead of supplied validation data
    """

    if warn_trace:
        warnings.showwarning = warn_with_traceback

    if not (save or dummy or tabulate or eval):
        sys.exit ("You must specify --save (to save parameters), --eval (to save evaluation metrics), --tabulate (to print model info), or --dummy (for a dummy run)")

    if not model_name in models:
        sys.exit ("Model '" + model_name + "' not known. Available models: " + ' '.join(list(models.keys())))

    if logdir or logfile:
        logdir = logdir or '.'
        logfile = logfile or 'log'
        os.makedirs(logdir,exist_ok=True)
        logging.basicConfig(filename=f"{logdir}/{logfile}", level=logging.WARNING, format='%(asctime)s %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.warning("Args: " + ' '.join(sys.argv))
    logging.warning("Date: " + datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    logging.warning("Local devices: " + ", ".join ([d.device_kind for d in jax.local_devices()]))
    logging.warning("Host: " + socket.gethostname())

    gitDir = os.path.dirname(__file__) + '/../../../.git'
    with open(f"{gitDir}/HEAD") as headfile:
        head = headfile.readline().strip()
        p = re.compile('ref: (\\S+)')
        m = p.match(head)
        if m:
            with open(f"{gitDir}/{m.group(1)}") as ref:
                head = ref.readline().strip()
        logging.warning(f"git HEAD: {head}")

    jax.config.update("jax_debug_nans", jax_debug_nans)
    jax.config.update("jax_log_compiles", jax_log_compiles)

    # load datasets
    # for the training set we always use batch size 1, since we use the round-robin iterator to mix batches across datasets
    model_info = models[model_name]
    
    logging.warning('loading datasets from {}'.format(data_dir))
    train = [SeqDataset (data_dir=data_dir, split_label=label, batch_size=1) for label in train]
    valid = [SeqDataset (data_dir=data_dir, split_label=label, batch_size=1) for label in valid]
    test = [SeqDataset (data_dir=data_dir, split_label=label, batch_size=1) for label in test]

    # figure out data shape
    representative = (train + valid + test)[0]
    seq_length, seq_depth, target_length, n_targets = representative.seq_length, representative.seq_depth, representative.target_length, representative.num_targets
    logging.warning(f"seq_length={seq_length}, seq_depth={seq_depth}, target_length={target_length}, n_targets={n_targets}")

    # create RNGs
    prng, revcomp_prng, dropout_prng, fake_data_rng, init_rng = jax.random.split (jax.random.PRNGKey(rng_key), 5)
    valid_iter = batch_limiter (fake_data_iter(valid,fake_data_rng,seq_length,seq_depth,target_length,n_targets) if rnd_valid else round_robin_iter(valid,batch_size), batch_limit, first_batch)
    train_iter = batch_limiter (round_robin_iter(train,batch_size), batch_limit, first_batch)
    test_iter = batch_limiter (round_robin_iter(test,batch_size), batch_limit, first_batch)
    n_valid_batches = count_batches (valid, batch_size, batch_limit, first_batch)
    n_train_batches = count_batches (train, batch_size, batch_limit, first_batch)
    n_test_batches = count_batches (test, batch_size, batch_limit, first_batch)
    logging.warning (f"counted {n_valid_batches} validation and {n_train_batches} training batches")

    # initialize the model
    conv_net = model_info["new_model"](features=n_targets, **{**model_info.get("init_args",{}), **model_args})

    if tabulate:
        dummy_x = jnp.ones((1, seq_length, seq_depth))
        logging.warning(conv_net.tabulate(prng, dummy_x, train=False))
        sys.exit()

    # Loss function(s)
    poisson_mn = lambda y_pred, y_true: poisson_multinomial_loss(y_pred,y_true,total_weight=poisson_weight)
    loss_fn = poisson_loss if poisson else poisson_mn
    
    # Parameters
    init_vars = init_params (prng=init_rng, model=conv_net, filename=load or save, seq_length=seq_length, seq_depth=seq_depth)
    logging.warning(f"model has {sum([a.size for a in jax.tree_util.tree_leaves(init_vars)])} parameters")

    if eval:
        logging.warning('computing test set metrics')
        apply_fn = conv_net.apply
        if not disable_jit:
            apply_fn = jax.jit (apply_fn)
        if rc_ensemble_eval:
            apply_fn = ensemble_fwd_rev(apply_fn, test[0].strand_pair)
        if shift_ensemble_eval:
            apply_fn = ensemble_shift(apply_fn, max_shift)
        compute_metrics = Metrics (apply_fn, loss_fn, use_jit=not disable_jit)
        test_metrics = compute_metrics(init_vars,test_iter,n_batches=n_test_batches,fold_name="test",return_per_feature_metrics=True,warn_if_zero=False)
        with open(eval, 'w') as f:
            f.write (json.dumps(test_metrics))
        sys.exit()

    # initialize the trainer
    logging.warning('creating TrainState')
    learn_schedule = optax.linear_schedule (init_value=0.0, end_value=learn_rate, transition_steps=warmup_steps)
    state = TrainState.create(
    apply_fn = conv_net.apply,
    params = init_vars['params'],
    batch_stats = init_vars.get('batch_stats',{}),
    max_shift = max_shift,
    max_epochs = max_epochs,
    max_seconds = max_seconds,
    prevalidate = prevalidate,
    patience = patience,
    tx = optax.chain(
        optax.clip_by_block_rms(block_clip),  # Clip each block by RMS gradient norm
        optax.clip_by_global_norm(global_clip),  # Clip overall gradient by the global norm
        optax.scale_by_adam(b1=adam_beta1, b2=adam_beta2),  # Use the updates from adam.
        optax.scale_by_schedule(learn_schedule),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0)
    ),
    revcomp_prng = revcomp_prng,
    dropout_prng = dropout_prng,
    strand_pair = train[0].strand_pair,
    )

    logging.warning('creating TrainLogger')
    # automatically enable summary logging of internal model diagnostics, if they are being collected
    if 'diagnostics' in model_args:
        if 'batch' not in summaries:
            summaries['batch'] = {}
        if type(summaries['batch']) is dict:
            summaries['batch']['diagnostics'] = model_args['diagnostics']
    # create the TrainLogger
    tlog = TrainLogger(
    trace_prof_dir = trace_prof_dir,
    device_prof_dir = device_prof_dir,
    global_clip = global_clip,
    block_clip = block_clip,
    verbose = verbose,
    summary_dir = summary_dir,
    summaries = summaries,
    diagnostics = model_args.get('diagnostics',None),
    log_dir = logdir,
    summary_period = summary_period,
    save_filename = save,
    memory_stats = memory_stats,
    **tlog_args,
    )

    # Run it
    run_training_loop(state=state,tlog=tlog,
                      loss_fn=loss_fn,
                      valid_iter=valid_iter,train_iter=train_iter,
                      n_valid_batches=n_valid_batches,n_train_batches=n_train_batches,
                      recompute_train_metrics=recompute_train_loss,
                      use_jit=not disable_jit,
                      use_threads=use_threads,
                      use_tracemalloc=use_tracemalloc,)
    logging.warning('finished')

if __name__ == "__main__":
    CLI(main, parser_mode='jsonnet')