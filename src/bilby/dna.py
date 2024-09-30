from typing import Callable, Any

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from typing import Callable, Tuple, Sequence, Optional

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

dna_alph = 'acgt'
def one_hot_dna (str, dtype=jnp.float32):
    # The following tortured procedure is motivated because jax.nn.one_hot is agonizingly slow to compile, but creating a jax.numpy array from a numpy array is not
    idx = [dna_alph.find(x) for x in str.lower()]
    a = np.zeros ((len(str),len(dna_alph)))
    for pos, i in enumerate(idx):
        if i >= 0:
            a[pos,i] = 1
    return jnp.array(a, dtype=dtype)

def one_hot_dna_with_nonzero_ns (str, dtype=jnp.float32):
    # Again, the reason we do things the following way is that jnp.stack takes much longer than jnp.array, for some reason
    def create_one_hot (dims, pos):
        a = np.zeros (dims)
        a[pos] = 1
        return a
    lookup = dict((x,create_one_hot(len(dna_alph),dna_alph.index(x))) for x in dna_alph)
    unknown = np.ones(len(dna_alph)) / len(dna_alph)
    a = np.stack ([lookup.get(x,unknown) for x in str.lower()])
    return jnp.array(a, dtype=dtype)

def one_hot_dna_set (one_hot, pos, char):
    assert one_hot.ndim == 2 and one_hot.shape[-1] == 4
    return one_hot.at[pos].set (one_hot_dna(char)[0])

def one_hot_dna_insert (one_hot, pos, str):
    assert one_hot.ndim == 2 and one_hot.shape[-1] == 4
    return jnp.insert (one_hot, pos, one_hot_dna(str), axis=0)

def one_hot_dna_delete (one_hot, pos, len=1):
    assert one_hot.ndim == 2 and one_hot.shape[-1] == 4
    return jnp.delete (one_hot, jnp.arange(pos,pos+len), axis=0)

# Assumes seq is one-hot encoded with shape (sequence_length,4) with lexically sorted nucleotide ordering i.e. A,C,G,T
# The function one_hot_dna returns encodings in this format
# Further assumes that out has shape (binned_sequence_length,num_tracks)
def stochastic_revcomp (prng, seq, out, strand_pair, max_shift=0):
    revcomp_prng, shift_prng = jax.random.split (prng)
    revcomp = jax.random.bernoulli (revcomp_prng)
    shift = jax.random.randint (key=shift_prng, shape=(1,), minval=0, maxval=max_shift + 1)
    seq = jnp.roll (seq, shift, axis=-2)
    seq = jnp.where (revcomp, seq[::-1,::-1], seq)
    out = jnp.where (revcomp, out[::-1,strand_pair], out)
    return seq, out, revcomp, shift

# bda has the form (batch_id,donor_bin,acceptor_bin)
def revcomp_xxj_coords (bda, out_len):
    return jnp.array ([bda[0], out_len-1-bda[1], out_len-1-bda[2]])

# xxj_counts has the form (forward#1,reverse#1,forward#2,reverse#2,...)
def revcomp_xxj_counts (xxj_counts):
    return jnp.stack ([xxj_counts[1::2], xxj_counts[0::2]], axis=-1).reshape((-1,))

# As stochastic_revcomp, but assumes additional leading axis is index within batch
# xxj_coords has shape (num_xxj_sites,3) with each row having the form (batch_id,donor_bin,acceptor_bin)
# xxj_counts has shape (num_xxj_sites,num_tracks) with strand-paired tracks in adjacent positions (0=forward#1, 1=reverse#1, 2=forward#2, 3=reverse#2, ...)
def stochastic_revcomp_batch (prng, seq_batch, out_batch, strand_pair, xxj_coords, xxj_counts, max_shift=0):
    seq_out_revcomp_shift = [stochastic_revcomp(*a,strand_pair,max_shift=max_shift) for a in zip (jax.random.split (prng, seq_batch.shape[0]), seq_batch, out_batch)]
    out_len = out_batch.shape[1]
    revcomp_batch = jnp.array ([sors[2] for sors in seq_out_revcomp_shift], dtype=jnp.bool_)
    if xxj_coords.shape[0] > 0:
        xxj_coords_batch = xxj_coords[:,0]
        xxj_coords_revcomp = revcomp_batch[xxj_coords_batch]
        xxj_coords = jnp.stack ([jnp.where (xxj_coords_revcomp[i], revcomp_xxj_coords(xxj_coords[i,:],out_len), xxj_coords[i,:])
                                 for i in range(xxj_coords.shape[0])], axis=0)
        xxj_counts = jnp.stack ([jnp.where (xxj_coords_revcomp[i], revcomp_xxj_counts(xxj_counts[i,:]), xxj_counts[i,:])
                                 for i in range(xxj_counts.shape[0])], axis=0)
    return tuple (jnp.array([sors[i] for sors in seq_out_revcomp_shift]) for i in range(4)) + (xxj_coords, xxj_counts)

# Shift DNA, padding with 1/4 at the end
def shift_dna (seq, shift):
    if shift == 0:
        return seq
    elif shift < 0:
        return jnp.concatenate ([seq[...,:shift,:], jnp.ones_like(seq[...,shift:,:]) / 4], axis=-2)
    return jnp.concatenate ([jnp.ones_like(seq[...,:shift,:]) / 4, seq[...,shift:,:]], axis=-2)

# Ensemble forward and reverse sequences for evaluation
def ensemble_fwd_rev (predict_fn, strand_pair):
    def predict_wrapper (vars, seq, *args, **kwargs):
        y = predict_fn (vars, seq, *args, **kwargs)
        y = y + predict_fn (vars, seq[...,::-1,::-1], *args, **kwargs) [...,::-1,strand_pair]
        return y / 2
    return predict_wrapper

# Ensemble shift for evaluation
def ensemble_shift (predict_fn, max_shift):
    def predict_wrapper (vars, seq, *args, **kwargs):
        y = 0
        for shift in range(max_shift + 1):
            y = y + predict_fn (vars, shift_dna(seq,shift), *args, **kwargs)
        return y / (max_shift + 1)
    return predict_wrapper

# Reverse-complement equivariant wrapper for flax modules
class RevCompEquivariantBlock(nn.Module):
    func: Callable[[Array],Array]

    @nn.compact
    def __call__ (self, inputs, *args, **kwargs):
        fwd_outputs = self.func (inputs, *args, **kwargs)
        rev_outputs = self.func (inputs[...,::-1,::-1], *args, **kwargs)
        return jnp.concatenate ([fwd_outputs, rev_outputs[...,::-1,::-1]], axis=-1)

# Reverse-complement invariant wrapper for flax modules
class RevCompInvariantBlock(nn.Module):
    func: Callable[[Array],Array]

    @nn.compact
    def __call__ (self, inputs, *args, **kwargs):
        fwd_outputs = self.func (inputs, *args, **kwargs)
        rev_outputs = self.func (inputs[...,::-1,::-1], *args, **kwargs)
        return fwd_outputs + rev_outputs


# Flax module for reverse-complement equivariant convolution
class RevCompEquivariantConv1D(nn.Module):
    features: int
    kernel_size: int
    strides: int = 1
    kernel_dilation: int = 1
    padding: str = 'SAME'
    use_bias: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__ (self, inputs: Array) -> Array:
        if inputs.ndim != 3:
            raise ValueError ("Input must have shape (batch, length, features)")
        if inputs.shape[-1] % 2 != 0:
            raise ValueError ("Input dimension must be even for RC-equivariant convolution")
        if self.features % 2 != 0:
            raise ValueError ("Output dimension must be even for RC-equivariant convolution")

        kernel_size: Sequence[int]
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size,)
        else:
            kernel_size = tuple(self.kernel_size)
        if len(kernel_size) != 1:
            raise ValueError ("Only 1D kernel size supported for RC-equivariant convolution")
        
        if isinstance(self.strides,int):
            strides = (self.strides,)
        else:
            strides = tuple(self.strides)

        if isinstance(self.kernel_dilation,int):
            kernel_dilation = (self.kernel_dilation,)
        else:
            kernel_dilation = tuple(self.kernel_dilation)

        D = inputs.shape[-1]
        F = self.features // 2

        kernel = self.param ('kernel', self.kernel_init, kernel_size + (D, F), self.param_dtype)
        kernel = jnp.concatenate ([kernel, kernel[::-1,::-1,::-1]], axis=-1)

        if self.use_bias:
            bias = self.param ('bias', self.bias_init, (F,), self.param_dtype)
            bias = jnp.concatenate ([bias, bias[::-1]], axis=-1)
        else:
            bias = 0

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        return jax.lax.conv_general_dilated (inputs,
                                             kernel,
                                             window_strides=strides,
                                             rhs_dilation=kernel_dilation,
                                             padding=self.padding,
                                             dimension_numbers=('NLC', 'LIO', 'NLC'),
                                            ) + bias

# Flax module for reverse-complement equivariant dense network
class RevCompEquivariantDense(nn.Module):
    features: int
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__ (self, inputs: Array) -> Array:
        if inputs.shape[-1] % 2 != 0:
            raise ValueError ("Input dimension must be even for RC-equivariant dense network")
        if self.features % 2 != 0:
            raise ValueError ("Output dimension must be even for RC-equivariant dense network")
        
        D = inputs.shape[-1]
        F = self.features // 2

        kernel = self.param ('kernel', self.kernel_init, (D, F), self.param_dtype)
        kernel = jnp.concatenate ([kernel, kernel[::-1,::-1]], axis=-1)

        if self.use_bias:
            bias = self.param ('bias', self.bias_init, (F,), self.param_dtype)
            bias = jnp.concatenate ([bias, bias[::-1]], axis=-1)
        else:
            bias = 0

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        return jnp.matmul (inputs, kernel) + bias



# Flax module for "weakly" reverse-complement equivariant convolution, i.e. a convolution with regularization encouraging equivariance
class WeakRevCompEquivariantConv1D(nn.Module):
    features: int
    kernel_size: int
    strides: int = 1
    kernel_dilation: int = 1
    padding: str = 'SAME'
    use_bias: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    l2_equiv_reg: float = 1e-6

    @nn.compact
    def __call__ (self, inputs: Array) -> Array:
        if inputs.ndim != 3:
            raise ValueError ("Input must have shape (batch, length, features)")
        if inputs.shape[-1] % 2 != 0:
            raise ValueError ("Input dimension must be even for RC-equivariant convolution")
        if self.features % 2 != 0:
            raise ValueError ("Output dimension must be even for RC-equivariant convolution")

        kernel_size: Sequence[int]
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size,)
        else:
            kernel_size = tuple(self.kernel_size)
        if len(kernel_size) != 1:
            raise ValueError ("Only 1D kernel size supported for RC-equivariant convolution")
        
        if isinstance(self.strides,int):
            strides = (self.strides,)
        else:
            strides = tuple(self.strides)

        if isinstance(self.kernel_dilation,int):
            kernel_dilation = (self.kernel_dilation,)
        else:
            kernel_dilation = tuple(self.kernel_dilation)

        D = inputs.shape[-1]
        F = self.features

        kernel = self.param ('kernel', self.kernel_init, kernel_size + (D, F), self.param_dtype)

        rc_kernel = kernel[::-1,::-1,::-1]
        self.sow ('losses', 'rc_equiv_reg', self.l2_equiv_reg * jnp.mean ((kernel - rc_kernel) ** 2))

        if self.use_bias:
            bias = self.param ('bias', self.bias_init, (F,), self.param_dtype)
            rc_bias = bias[::-1]
            self.sow ('losses', 'rc_equiv_reg', self.l2_equiv_reg * jnp.mean ((bias - rc_bias) ** 2))
        else:
            bias = 0

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        return jax.lax.conv_general_dilated (inputs,
                                             kernel,
                                             window_strides=strides,
                                             rhs_dilation=kernel_dilation,
                                             padding=self.padding,
                                             dimension_numbers=('NLC', 'LIO', 'NLC'),
                                            ) + bias


# Flax module for "weakly" reverse-complement equivariant dense network, i.e. a dense network with regularization encouraging equivariance
class WeakRevCompEquivariantDense(nn.Module):
    features: int
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    param_dtype: Dtype = jnp.float32

    l2_equiv_reg: float = 1e-6

    @nn.compact
    def __call__ (self, inputs: Array) -> Array:
        if inputs.shape[-1] % 2 != 0:
            raise ValueError ("Input dimension must be even for RC-equivariant dense network")
        if self.features % 2 != 0:
            raise ValueError ("Output dimension must be even for RC-equivariant dense network")
        
        D = inputs.shape[-1]
        F = self.features

        kernel = self.param ('kernel', self.kernel_init, (D, F), self.param_dtype)
        rc_kernel = kernel[::-1,::-1]
        self.sow ('losses', 'rc_equiv_reg', self.l2_equiv_reg * jnp.mean ((kernel - rc_kernel) ** 2))

        if self.use_bias:
            bias = self.param ('bias', self.bias_init, (F,), self.param_dtype)
            rc_bias = bias[::-1]
            self.sow ('losses', 'rc_equiv_reg', self.l2_equiv_reg * jnp.mean ((bias - rc_bias) ** 2))
        else:
            bias = 0

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        return jnp.matmul (inputs, kernel) + bias

