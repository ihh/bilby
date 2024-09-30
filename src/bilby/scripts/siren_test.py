import sys
import os

from jsonargparse import CLI
from typing import List

from functools import partial

import logging
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.profiler

import flax.linen as nn

sys.path.append (os.path.dirname(__file__) + '/..')
from dataset import SeqDataset
from dataset_iterators import fake_data_iter, real_data_iter, batch_limiter, count_batches
from hyena import Siren, SinePosEmbedding, ExponentialModulation, fftconv


def main(len: int = 8192,
         pos_emb_dim: int = 8,
            scale_in: float = 6,
            scale_internal: float = 6,
            scale_out: float = 6,
            scale_bias: float = 0,
            freq: float = 1,
            layers: int = 4,
            hidden: int = 16,
            trials: int = 1,
            seed: int = 42,
            hyena: bool = False,
#            modulate: bool = False,
#            fast_decay_pct: float = 0.3,
#            slow_decay_pct: float = 1.5,
#            target: float = 1e-2,
#            shift: float = 0.0,
#            convolve: bool = False,
         ):
    """
    Explore SIREN initialization

    Args:
        len: sequence length
        scale_in: scale for first layer
        scale_internal: scale for internal layers
        scale_out: scale for output layer
        scale_bias: scale for bias
        freq: frequency multiplier for first layer (redundant?)
        layers: number of layers
        hidden: number of hidden features per layer
        trials: number of trials
        seed: random number generator seed
        hyena: whether to use Hyena implementation
        pos_emb_dim: positional embedding dimension (excluding linear component) for Hyena implementation
    """
#        modulate: whether to modulate output
#        fast_decay_pct: max decay multiplier
#        slow_decay_pct: min decay multiplier
#        target: target decay at end of sequence
#        shift: modulation offset
#        convolve: whether to convolve output with random signal
#    """
    if hyena:
        if scale_bias != 0:
            logging.warning (f"scale_bias {scale_bias} ignored for Hyena implementation")
        siren = Siren (hidden_features=hidden, out_features=1, num_layers=layers, freq=freq, layer0_freq_init=scale_in/3, internal_freq_init=scale_internal/3, out_proj_init=scale_out/3)
#        t = SinePosEmbedding (len, dim=pos_emb_dim, max_seq_len=len,
#                              wrap_to_negative_offsets=True, include_linear_coord=True, geom_spaced_freqs=False)[None, :, :]  # (1, 2*len, pos_emb_dim)
        t = jnp.linspace(-1, 1, 2*len)[:,None]  # (1, 2*len, 1)
        
        vars_key = jax.random.PRNGKey(seed)
        vars = [siren.init (k, t) for k in jax.random.split (vars_key, trials)]

        vt = {}
        for i in range(trials):
            tl = jax.tree_util.tree_leaves_with_path(vars[i])
            for path, leaf in tl:
                vt[str(path)] = vt.get(str(path), []) + [leaf]
        for path, leaf in vt.items():
            leaf = jnp.array(leaf)
            print (f"{path}: shape {leaf.shape} variance {jnp.var(leaf)}")

        x = jnp.array ([siren.apply(v,t) for v in vars])  # (trials, 1, len, 1)
        print (f"mean: {jnp.mean(x)} variance: {jnp.var(x)}")

    else:
        trial_keys = jax.random.split (jax.random.PRNGKey(seed), trials)
        trial_results = []
        trial_w = []
        trial_b = []
        for trial in jnp.arange(trials):
            xk, wk = jax.random.split (trial_keys[trial])
            weights_key = jax.random.split (wk, layers + 1)
            x = jax.random.uniform(xk,shape=(2*len,1),minval=-1,maxval=1)
            result = [x]
            layer_w = []
            layer_b = []
            for layer in jnp.arange(layers):
                wkey, bkey = jax.random.split (weights_key[layer])
                dim_in = x.shape[-1]
                c = jnp.sqrt((scale_in if layer==0 else scale_internal) / dim_in)
                w = jax.random.uniform (wkey, (dim_in, hidden), minval=-c, maxval=c) * (freq if layer==0 else 1)
                b = jax.random.uniform (bkey, (hidden,), minval=-jnp.pi*scale_bias, maxval=jnp.pi*scale_bias)
                layer_w.append(w)
                layer_b.append(b)
                a = jnp.dot (x, w)
                f = freq if layer==0 else 1
                x = jnp.sin (f * a + b)
                result.append(x)

            c = jnp.sqrt(scale_out / x.shape[-1])
            w = jax.random.uniform (weights_key[layers], (x.shape[-1], 1), minval=-c, maxval=c)
            out = jnp.dot (x, w)
            result.append (out)
            trial_results.append (result)
            trial_w.append (layer_w)
            trial_b.append (layer_b)
            if trial == 0:
                print (f"trial {trial} shapes={[r.shape for r in result]}")

        for layer in jnp.arange(layers):
                layer_w = jnp.array([lw[layer] for lw in trial_w])
                layer_b = jnp.array([lb[layer] for lb in trial_b])
                print(f"layer {layer+1} kernel: shape {layer_w.shape} variance {jnp.var(layer_w)}")
                print(f"layer {layer+1} bias: shape {layer_b.shape} variance {jnp.var(layer_b)}")

        for layer in jnp.arange(layers+2):
            layer_results = jnp.array ([r[layer] for r in trial_results])
            print (f"layer {layer} mean: {jnp.mean(layer_results)} variance: {jnp.var(layer_results)}")


#    if modulate:
#        mod_fn = ExponentialModulation (fast_decay_pct=fast_decay_pct, slow_decay_pct=slow_decay_pct, target=target, shift=shift)
#        mod_t = jnp.linspace(-1, 1, 2*len)[None,:,None]  # (1, 2*len, 1)
#        mod_vars = mod_fn.init(jax.random.PRNGKey(0), mod_t, x)
#        x = mod_fn.apply(mod_vars, mod_t, x)  # (trials, 1, 2*len, 1)
#
#    x = x[:,0,:,:]  # (trials, 2*len, 1)
#    if convolve:
#        signal = jax.random.uniform (rng_key, (trials, len, 1), minval=-1, maxval=1)
#        print (f"signal: {signal.shape} x: {x.shape}")
#        x = fftconv (signal, x, 0)  # (trials, len, 1)


if __name__ == "__main__":
    CLI(main, parser_mode='jsonnet')