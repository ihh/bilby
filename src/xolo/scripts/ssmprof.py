import sys
import os
import time

from jsonargparse import CLI
from functools import partial

import logging

import jax
import jax.numpy as jnp
import jax.profiler

import flax.linen as nn

import einops

sys.path.append (os.path.dirname(__file__) + '/..')


# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size = 128):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    if L % chunk_size != 0:
        raise ValueError(f"chunk_size={chunk_size} must divide L={L}")
    n_chunks = L // chunk_size

    # Transpose length & batch dimensions to make the scan over length, and split into chunks
    x_chunks = einops.rearrange (x, 'b (c l) d -> c l b d', c=n_chunks)
    B_chunks = einops.rearrange (Bcoeff, 'b (c l) n -> c l b n', c=n_chunks)
    C_chunks = einops.rearrange (Ccoeff, 'b (c l) n -> c l b n', c=n_chunks)
    dt_chunks = einops.rearrange (dt, 'b (c l) d -> c l b d', c=n_chunks)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    @jax.remat
    def scan_chunk (carry, chunk):
        dA_init, dB_init = carry  # (1, B, D, N)  (1, B, D, N)
        x_chunk, B_chunk, C_chunk, dt_chunk = chunk
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        dA = jnp.exp (jnp.einsum ('dn,lbd->lbdn', Acoeff, dt_chunk))  # (chunk_size, B, D, N)
        dB = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # We can think of the pair (dA,dB) as the top row of a matrix ((dA,dB),(0,1)) and multiply it by the previous state to get the next state
        # Since matrix multiplication is associative, we can do an associative scan to materialize the hidden state
        def associative_scan_fn (l, r):  # l, r, and return value are both tuples of the form ((B,D,N), (B,D,N))
            A_l, B_l = l
            A_r, B_r = r
            return tuple((A_l*A_r, A_r*B_l + B_r))
        As, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N
        hs = As * dB_init + hs  # Incorporate dB_init here so that it is reflected in y_chunk
        # We only need to keep the last element of As, so we can discard the rest. Otherwise we would incorporate dA_init here, like so:
        # As = dA_init * As
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (As[-1:,...] * dA_init, hs[-1:,...]), y_chunk  # note dA_init incorporated into last state here

    # Perform the scan over chunks recurrently (with rematerialization as noted above), with each chunk being an associative scan
    (A_final, h_final), y_chunks = jax.lax.scan (scan_chunk, (jnp.ones((1,B,D,N)), jnp.zeros((1,B,D,N))), (x_chunks, B_chunks, C_chunks, dt_chunks))

    return jnp.concatenate (y_chunks, axis=0).transpose((1,0,2))  # (L, B, D)


def ssm_dynamic_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size = 128):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    if L % chunk_size != 0:
        raise ValueError(f"chunk_size={chunk_size} must divide L={L}")
    chunk_start_positions = jnp.arange(0, L, chunk_size)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    @jax.remat
    def scan_chunk (carry, chunk_start_pos):
        dA_init, dB_init = carry  # (1, B, D, N)  (1, B, D, N)
        # Take dynamic slices to save memory
        x_chunk = jax.lax.dynamic_slice_in_dim (x, chunk_start_pos, chunk_size, axis=-2)  # (B, chunk_size, D)
        B_chunk = jax.lax.dynamic_slice_in_dim (Bcoeff, chunk_start_pos, chunk_size, axis=-2)  # (B, chunk_size, N)
        C_chunk = jax.lax.dynamic_slice_in_dim (Ccoeff, chunk_start_pos, chunk_size, axis=-2)  # (B, chunk_size, N)
        dt_chunk = jax.lax.dynamic_slice_in_dim (dt, chunk_start_pos, chunk_size, axis=-2)  # (B, chunk_size, D)
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        # We can think of the pair (dA,dB) as the top row of a matrix ((dA,dB),(0,1)) and multiply it by the previous state to get the next state
        dA = jnp.exp (jnp.einsum ('dn,bld->lbdn', Acoeff, dt_chunk))  # (chunk_size, B, D, N)
        dB = jnp.einsum ('bln,bld,bld->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # Since matrix multiplication is associative, we can do an associative scan to materialize the hidden state
        def associative_scan_fn (l, r):  # l, r, and return value are both tuples of the form ((B,D,N), (B,D,N))
            A_l, B_l = l
            A_r, B_r = r
            return tuple((A_l*A_r, A_r*B_l + B_r))
        As, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = As * dB_init + hs
        As = dA_init * As
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (As[-1:,...], hs[-1:,...]), y_chunk

    # Perform the scan over chunks recurrently (with rematerialization as noted above), with each chunk being an associative scan
    (A_final, h_final), y_chunks = jax.lax.scan (scan_chunk, (jnp.ones((1,B,D,N)), jnp.zeros((1,B,D,N))), chunk_start_positions)

    return jnp.concatenate (y_chunks, axis=0).transpose((1,0,2))  # (L, B, D)


# Older implementations
# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_scan (x, Acoeff, Bcoeff, Ccoeff, dt, reverse=False, channel_split=1, associative_scan=True):
    D = x.shape[-1]
    L = x.shape[-2]
    B = x.shape[-3]
    N = Acoeff.shape[-1]

    # The associative scan gives up the smaller memory footprint of the recurrent scan in favor of speed
    # It has to materialize the full hidden state h, as well as time-resolved SSM coefficients dA and dB, each of which has shape (L,D,N)
    def perform_associative_scan (args):  # xb: (L,D) A: (D,N) Bb: (L,N) Cb: (L,N) dtb: (L,D)
        xb, A, Bb, Cb, dtb = args
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        dA = jnp.exp (jnp.einsum ('dn,ld->ldn', A, dtb))  # (L, D, N)
        dB = jnp.einsum ('ln,ld,ld->ldn', Bb, xb, dtb)  # (L, D, N)
        # We can think of the pair (dA,dB) as the top row of a matrix ((dA,dB),(0,1)) and multiply it by the previous state to get the next state
        # Since matrix multiplication is associative, we can do an associative scan to materialize the hidden state
        # @jax.remat
        def associative_scan_fn (l, r):  # l, r, and return value are both tuples of the form ((D,N), (D,N))
            A_l, B_l = l
            A_r, B_r = r
            return tuple((A_l*A_r, A_r*B_l + B_r))
        _dummy, h = jax.lax.associative_scan (associative_scan_fn, (dA, dB), reverse=reverse)
        return jnp.einsum ('ln,ldn->ld', Cb, h)  # (L,D)

    def perform_associative_scan_for_batch (args):  # xb: (L,D) Bb: (L,N) Cb: (L,N) dtb: (L,D)
        xb, Bb, Cb, dtb = args
        # @jax.remat
        def process_channel_split (channel_start):
            return perform_associative_scan ((jax.lax.dynamic_slice (xb, (0,channel_start), (L,channels_per_split)),
                                                jax.lax.dynamic_slice (Acoeff, (channel_start,0), (channels_per_split,N)),
                                                Bb, Cb,
                                                jax.lax.dynamic_slice (dtb, (0,channel_start), (L,dt_channels_per_split))))
        return jnp.concatenate (jax.lax.map (process_channel_split, channel_offsets), axis=-1)

    # The recurrent scan keeps everything inside the scan function; memory usage is low, so it can parallelize over dimensions and batches with no need to split
    @jax.remat
    def perform_recurrent_scan():
        # Transpose length & batch dimensions to make the scan over length
        x_lbd = einops.rearrange (x, 'b l d -> l b d')
        Bcoeff_lbn = einops.rearrange (Bcoeff, 'b l n -> l b n')
        Ccoeff_lbn = einops.rearrange (Ccoeff, 'b l n -> l b n')
        dt_lbd = einops.rearrange (dt, 'b l d -> l b d')
        @jax.remat
        def scan_fn (h, args):
            xt, Bt, Ct, dt = args  # (B,D) (B,N) (B,N) (B,D)
            dA = jnp.exp (jnp.einsum ('dn,bd->bdn', Acoeff, dt))  # (B,D,N)
            dB = jnp.einsum ('bn,bd,bd->bdn', Bt, xt, dt)  # (B,D,N)
            h = h*dA + dB
            return h, jnp.einsum('bn,bdn->bd', Ct, h)  # (B,D,N) (B,D)
        _h, y = jax.lax.scan (scan_fn, jnp.zeros((B,D,N)), (x_lbd, Bcoeff_lbn, Ccoeff_lbn, dt_lbd), reverse=reverse, unroll=False)  # (B,D,N) (L,B,D)
        return einops.rearrange (y, 'l b d -> b l d')  # (B,L,D)

    if associative_scan:
        if D % channel_split != 0:
            raise ValueError(f"channel_split={channel_split} must divide D={D}")
        channels_per_split = D // channel_split
        dt_channels_per_split = channels_per_split if dt.shape[-1] == D else 1
        channel_offsets = jnp.arange(0, D, channels_per_split)
        y = jax.lax.map (perform_associative_scan_for_batch, (x, Bcoeff, Ccoeff, dt))  # (B, L, D)
    else:
        y = perform_recurrent_scan()  # (B, L, D)

    if reverse:
        y = jnp.flip (y, axis=-2)

    return y






# Main
def main(B: int = 1,
         L: int = 8192,
         D: int = 768,
         N: int = 8,
         chunked: bool = False,
         dynamic: bool = False,
         chunk_size: int = 128,
         recurrent: bool = False,
         grad: bool = False,
         jit: bool = False,
         cost: bool = False,
         trials: int = 1,
         ):
    """
    Profile SSM scan

    Args:
        B: number of batches
        L: sequence length
        D: embedding dimension
        N: hidden states per dimension
        chunked: use chunked scan
        dynamic: use dynamic chunked scan
        chunk_size: chunk size -- valid only if chunked
        recurrent: use recurrent (vs associative) scan -- valid only if not chunked
        grad: whether to take gradient
        jit: whether to JIT-compile
        cost: show cost analysis for JIT-compiled function
        trials: number of trials
    """
    x = jnp.ones((B, L, D))
    Acoeff = jnp.ones ((D, N)) / 100
    Bcoeff = jnp.ones ((B, L, N))
    Ccoeff = jnp.ones ((B, L, N))
    dt = jnp.ones ((B, L, 1)) / 100
    args = (x, Acoeff, Bcoeff, Ccoeff, dt)
    if dynamic:
        ssm = lambda args: jnp.mean (ssm_dynamic_chunked_scan (*args, chunk_size=chunk_size))
    elif chunked:
        ssm = lambda args: jnp.mean (ssm_chunked_scan (*args, chunk_size=chunk_size))
    else:
        ssm = lambda args: jnp.mean (ssm_scan (*args, associative_scan=not recurrent))
    if grad:
        ssm = jax.value_and_grad(ssm)
    if jit:
        ssm = jax.jit(ssm)
        if cost:
            ssm_cost = ssm.lower(args).compile().cost_analysis()
            print(f"JIT cost analysis: {ssm_cost}")
    print ("Starting dummy run")
    dummy = ssm (args)  # call once for JIT
    print(f"Result: {dummy}")
    start = time.time()
    last_reported_time = start
    for i in range(trials):
        if i == 0 or time.time() - last_reported_time > 5:
            print(f"Starting trial {i+1}/{trials}")
            last_reported_time = time.time()
        dummy = ssm (args)
    end = time.time()
    time_per_trial = (end - start) / trials
    bytes = jax.devices()[0].memory_stats()['peak_bytes_in_use']
    print(f"Memory usage: {bytes} bytes = {bytes/1024**3:.2f} GB")
    print(f"Time/trial: {time_per_trial:.3f} s")

if __name__ == "__main__":
    CLI(main, parser_mode='jsonnet')