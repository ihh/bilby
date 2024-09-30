# based on https://github.com/irhum/hyena/blob/main/hyena/hyena.py
# with modifications by IH to generate acausal filters and include positional embeddings

from functools import partial
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

import einops
import flax.linen as nn

import jax
import jax.numpy as jnp

from poisson import l2_norm

from mixconv import MixConv

class ExponentialModulation(nn.Module):
    """Applies exponential decay window to a batch of filtering signals.

    Ths ensures that, at initialization tokens primarily receive input from
    nearby, more recent tokens."""

    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    shift: float = 0.0

    learnable_decay: bool = False
    decay_init: Any = None

    @nn.compact
    def __call__(self, t, h):  # t: (1, T, 1, 1), h: (1, T, O, F)
        # Get number of feature dimensions.
        features = h.shape[-1]  # F
        # Compute the decay coefficients per feature dimension.
        if self.learnable_decay:
            decay_init = self.decay_init or (lambda rng: decay_rates (features, self.fast_decay_pct, self.slow_decay_pct, self.target))
            deltas = self.param("deltas", decay_init)  # (F,)
            shift = self.param("shift", lambda rng: jnp.zeros((1,1,1,features)))  # (F,)
        else:
            deltas = decay_rates (features, self.fast_decay_pct, self.slow_decay_pct, self.target)  # (F,)
            shift = self.shift
        # Apply the decay window to each filter.
        decay = jnp.exp(-jnp.abs(t * deltas))  # (1, T, 1, F)
        h = h * (decay + shift)  # (1, T, O, F)
        return h

def decay_rates (features: int, fast_decay_pct: float = 0.3, slow_decay_pct: float = 1.5, target: float = 1e-2):
    """Compute the decay rates for the exponential decay window."""
    max_decay = jnp.log(target) / fast_decay_pct
    min_decay = jnp.log(target) / slow_decay_pct
    return jnp.linspace(min_decay, max_decay, features)

def SinePosEmbedding (seq_len: int, dim: int = 8, max_seq_len: int = 1e8, wrap_to_negative_offsets = True, include_linear_coord = True, geom_spaced_freqs = True):
    """Cosine/sine embedding (Transformer-style) plus a linear coordinate"""
    offset = jnp.arange (seq_len)  # L = seq_len
    if wrap_to_negative_offsets:
        offset = jnp.concatenate ([offset, offset - seq_len])  # L = 2*seq_len
    if geom_spaced_freqs:
        periods = jnp.geomspace (4, max_seq_len, int(dim / 2))
    else:
        periods = jnp.linspace (4, max_seq_len, int(dim / 2))
    freqs = 2 * jnp.pi / periods
    phases = jnp.outer (offset, freqs)
    pos_emb = jnp.concatenate ((jnp.cos(phases), jnp.sin(phases)), axis=-1)  # (L, dim)
    if include_linear_coord:
        pos_emb = jnp.concatenate ((offset[:,None] / max_seq_len, pos_emb), axis=-1)  # (L, dim+1)
    return pos_emb

def ThresholdEmbedding (seq_len: int, dim: int = 8, max_seq_len: int = 1e8, wrap_to_negative_offsets = True, include_linear_coord = True):
    """Geometrically spaced distance indicators (Enformer-style) plus a linear coordinate"""
    offset = jnp.arange (seq_len)  # L = seq_len
    if wrap_to_negative_offsets:
        offset = jnp.concatenate ([offset, offset - seq_len])  # L = 2*seq_len
    offset = offset[:,None]

    pow_rate = jnp.exp(jnp.log((seq_len + 1) / 2) / (dim // 2)).astype("float32")
    center_widths = jnp.power(pow_rate, jnp.arange(1, (dim // 2) + 1, dtype=jnp.float32))  # (dim/2,)
    unsigned_basis = jnp.where (jnp.abs(offset) <= center_widths[None,:], 1, 0)  # (L,dim/2)
    signed_basis = jnp.sign(offset) * unsigned_basis  # (L,dim/2)
    pos_emb = jnp.concatenate ((unsigned_basis, signed_basis), axis=-1)  # (L,dim)

    if include_linear_coord:
        pos_emb = jnp.concatenate ((offset / max_seq_len, pos_emb), axis=-1)  # (L, dim+1)
    return pos_emb

def LinearPosEmbedding (seq_len: int, wrap_to_negative_offsets = True):
    """No positional embedding except the linear coordinate"""
    offset = jnp.arange (seq_len)  # L = seq_len
    if wrap_to_negative_offsets:
        offset = jnp.concatenate ([offset, offset - seq_len])  # L = 2*seq_len
    offset = offset[:,None]

    return offset


class Siren(nn.Module):
    """This is a SIREN network used to generate the convolution filters.

    At initialization, the network is initialized to produce outputs of
    high frequency, which has been proven to lead to faster convergence
    when the target signals have a rich structure."""

    hidden_features: int
    out_features: Union[int, Sequence[int]]
    num_layers: int
    freq: float = 10.0
    l2_scale: float = 1e-6

    layer0_freq_init: float = 1/3
    internal_freq_init: float = 2
    out_proj_init: float = 2

    @nn.compact
    def __call__(self, x, diagnostics: dict = {}, deterministic: bool = True):
        init_fn = partial(
            nn.initializers.variance_scaling, mode="fan_in", distribution="uniform"
        )
        # x initially has shape (1, T, D)
        x = nn.Dense(self.hidden_features, name='layer0', kernel_init=init_fn(self.layer0_freq_init))(x)  # (1, T, H)
        x = jnp.sin(self.freq * x)  # (1, T, H)

        for layer in range(self.num_layers):
            layer_name = f"layer{layer+1}"
            x = nn.Dense(self.hidden_features, name=layer_name, kernel_init=init_fn(self.internal_freq_init))(x)  # (1, T, H)
            x = jnp.sin(x)  # (1, T, H)

        # Project to output dimension.
        x = nn.DenseGeneral(
            self.out_features, use_bias=False, name="out_proj", kernel_init=init_fn(self.out_proj_init)
        )(x)  # (1, T) + out_features

        if not deterministic:
            self.sow("losses", "regularizer", l2_norm (self.variables['params'], self.l2_scale))
            if 'filter' in diagnostics:
                self.sow("diagnostics", "filter_mean", jnp.mean(x))
                self.sow("diagnostics", "filter_sd", jnp.std(x))
        
        return x


class HyenaOperator(nn.Module):
    """This implements the Hyena operator, with an input and output projection."""

    features: int  # D
    max_len: int
    filter_fn: Callable[[Tuple[int]], nn.Module]
    modulation_fn: Callable[[], nn.Module]
    modulation_args: Any = None
    order: int = 2  # O
    dropout: float = 0.0
    out_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    pos_emb_dim: int = 8  # P
    pos_emb_type: str = "sine_lin" # "sine_geom", "sine_lin", "threshold", "linear"

    fft_bias: bool = True
    checkpoint_fftconv: bool = False

    bias_init: float = 1
    proj_init: float = 0.02
    l2_scale: float = 1e-6

    short_conv_size: int = 3
    short_conv_init: float = 0.02
    short_conv_grouped: bool = True

    mix_in_proj: bool = False
    in_proj_mix_components: int = 16
    in_proj_conv_size: int = 1
    in_proj_mix_softmax_weights: bool = False

    @nn.compact
    def __call__(
        self,
        u,  # (B, T, D)
        deterministic: bool = True,
        diagnostics: dict = {},
    ):
        seq_len = u.shape[-2]  # T
        D = u.shape[-1]  # D
        F = self.features
        
        assert seq_len <= self.max_len

        if not deterministic and 'hyena_input' in diagnostics:
            self.sow("diagnostics", "hyena_input_mean", jnp.mean(u))
            self.sow("diagnostics", "hyena_input_sd", jnp.std(u))

        # Generate the filters.
        if self.pos_emb_type == "sine_geom":
            t = SinePosEmbedding (seq_len, dim=self.pos_emb_dim, max_seq_len=self.max_len,
                                  wrap_to_negative_offsets=True, include_linear_coord=True, geom_spaced_freqs=True)[None, :, :]  # (1, 2T, P+1)
        elif self.pos_emb_type == "sine_lin":
            t = SinePosEmbedding (seq_len, dim=self.pos_emb_dim, max_seq_len=self.max_len,
                                  wrap_to_negative_offsets=True, include_linear_coord=True, geom_spaced_freqs=False)[None, :, :]  # (1, 2T, P+1)
        elif self.pos_emb_type == "threshold":
            t = ThresholdEmbedding (seq_len, dim=self.pos_emb_dim, max_seq_len=self.max_len,
                                    wrap_to_negative_offsets=True, include_linear_coord=True)[None, :, :]  # (1, 2T, P+1)
        else:
            t = LinearPosEmbedding (seq_len, wrap_to_negative_offsets=True)[None, :, :]  # (1, 2T, 1)

        h = self.filter_fn(out_features=(self.order, self.features))(
            t, deterministic=deterministic, diagnostics=diagnostics
        )  # (1, 2T, O, F)
        if not deterministic and 'unmodulated_filter' in diagnostics:
            self.sow("diagnostics", "unmodulated_filter_mean", jnp.mean(h))
            self.sow("diagnostics", "unmodulated_filter_sd", jnp.std(h))
        
        # Apply exponential decay window to filters.
        mod_t = jnp.linspace(0, 1, self.max_len)[:seq_len]
        mod_t = jnp.concatenate ([mod_t, jnp.flip(mod_t)])[None,:,None,None]  # (1, 2T, 1, 1)
        h = self.modulation_fn(**(self.modulation_args or {}))(mod_t, h)  # (1, 2T, O, F)

        if not deterministic and 'modulated_filter' in diagnostics:
            self.sow("diagnostics", "modulated_filter_mean", jnp.mean(h))
            self.sow("diagnostics", "modulated_filter_sd", jnp.std(h))

        # Reorder the filter axes for compatibility with input signal.
        h = einops.rearrange(h, "1 l o d -> o 1 l d", o=self.order)  # (O, 1, 2T, D)

        if self.fft_bias:
            bias = self.param("bias",
                              nn.initializers.normal(stddev=self.bias_init),
                              (self.order, 1, 1, self.features))
        else:
            bias = jnp.zeros ((self.order, 1, 1, self.features))

        # IH note: the irhum hyena code uses (self.order + 2) in the following expression
        # As far as I can tell, this is due to a misunderstanding(?) of the return value of Algorithm 1, p8 of Hyena paper
        # The paper appears to say that the projection should return x^1...x^O,v,x^n which would be a list of O+2 elements
        # However I think it means x^1...x^O,v which is O+1 elements
        # I have patched the code to what I believe is the correct interpretation
        inner_width = self.features * (self.order + 1)

        short_conv_init = nn.initializers.normal(stddev=self.short_conv_init) if self.short_conv_init else nn.initializers.lecun_normal()

        # Pass through dynamically-mixed dense layer and/or short convolution *before* projecting into layer
        # This is an attempt to reproduce the apparent benefits of cross-channel convolution in the short_filter convolution later on
        # (accidentally introduced in our fork of this code), but using fewer parameters
        if self.mix_in_proj:
            u = MixConv (out_features=D,
                         mix_components=self.in_proj_mix_components,
                         mix_weight_conv_size=self.in_proj_conv_size,
                         conv_size=self.in_proj_conv_size,
                         softmax_weights=self.in_proj_mix_softmax_weights) (u)
        elif self.in_proj_conv_size > 1:
            u = nn.Conv(
                features=D,
                kernel_size=(self.in_proj_conv_size,),
                strides=(1,),
                padding="SAME",
                use_bias=True,
                kernel_init=short_conv_init,
                name="in_proj_filter",
            )(u)

        # Affine projection "into" the layer
        proj_init = nn.initializers.normal(stddev=self.proj_init) if self.proj_init else nn.initializers.lecun_normal()
        u = nn.Dense(
            inner_width, name="in_proj", kernel_init=proj_init
        )(
            u
        )  # (B, T, (O+1)*F)    where B = batch size

        # Short convolution
        u = nn.Conv(
            features=inner_width,
            feature_group_count=inner_width if self.short_conv_grouped else 1,
            kernel_size=(self.short_conv_size,),
            strides=(1,),
            padding="SAME",
            use_bias=True,
            kernel_init=short_conv_init,
            name="short_filter",
        )(u)

        # Get the generated "diagonals" and the input signal.
        v, *x = jnp.split(u, self.order + 1, axis=-1)  # (B, T, F), (B, T, F) * O

        # We then apply the sequence of filters and diagonals
        fftconv_fn = fftconv
        if self.checkpoint_fftconv:
            fftconv_fn = jax.checkpoint(fftconv_fn)
        for o, x_i in enumerate(x):
            v = fftconv_fn(v, h[o], bias[o])  # (B, T, F)
            if 'postFFT' in diagnostics and not deterministic:
                self.sow ("diagnostics", f"postFFT{o}_mean", jnp.mean(v))
                self.sow ("diagnostics", f"postFFT{o}_sd", jnp.std(v))
            v = v * x_i  # (B, T, F)
            if 'postGate' in diagnostics and not deterministic:
                self.sow ("diagnostics", f"postGate{o}_mean", jnp.mean(v))
                self.sow ("diagnostics", f"postGate{o}_sd", jnp.std(v))

        v = nn.Dropout(rate=self.dropout, deterministic=deterministic)(v)
        # We then project back to the input space, to add as a residual.
        y = nn.Dense(D, name="out_proj", kernel_init=self.out_init)(
            v
        )  # (B, T, F)

        # Regularizers
        if not deterministic:
            self.sow("losses", "hyena_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        if not deterministic and 'hyena_output' in diagnostics:
            self.sow("diagnostics", "hyena_output_mean", jnp.mean(y))
            self.sow("diagnostics", "hyena_output_sd", jnp.std(y))

        return y

def fftconv(v, h, bias):
    # Zero pad to 2x the length to create causal convolution.
    # IH note: for non-autoregressive applications, we don't care about causality,
    # so we want to pad the sequence (so that we can still use a circular convolution)
    # but NOT the filter (we want to be able to look forward as well as back)
    # See paragraph at top of p8 of Hyena paper, immediately after "Proposition 3.1 (Causal Hyenas)"
    # In practice this is accomplished here by passing in a filter (h) that is already twice as long as the sequence (v)
    seqlen = v.shape[-2]
    filter_size = 2 * seqlen
    assert h.shape[-2] == filter_size
    
    # Real valued input signals, complex valued output frequencies.
    h_f = jnp.fft.rfft(h, n=filter_size, axis=-2)
    v_f = jnp.fft.rfft(v, n=filter_size, axis=-2)  # auto-padded

    # Multiply in the frequency domain.
    y_f = v_f * h_f / filter_size
    # Invert FFT to get the output signal.
    y = jnp.fft.irfft(y_f, axis=-2, n=filter_size, norm="forward")[:, :seqlen, :]
    return y + v * bias


class HyenaNac(nn.Module):
    features: int = None

    dropout_rate: float = 0.5
    siren_features: int = 16
    siren_layers: int = 4
    siren_freq: float = 300.0
    siren_args: Any = None
    modulation_args: Any = None
    hyena_args: Any = None
    positional_embedding_dimension: int = 8
    hyena_features: int = 768
    hyena_order: int = 2
    hyena_layers: int = 1
    norm_type: str = None
    bn_momentum: float = 0.9
    activation: str = "gelu"

    mlp_layer: bool = False
    dense_expansion: int = 2

    diagnostics: dict = field(default_factory=dict)
    checkpoint_hyena: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        conv_len = x.shape[-2]

        if self.activation == "relu":
            activate = nn.relu
        elif self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        else:
            activate = lambda x: x


        siren = partial(Siren, hidden_features=self.siren_features, num_layers=self.siren_layers, freq=self.siren_freq, out_proj_init=2/conv_len, **(self.siren_args or {}))
        hyena = partial(HyenaOperator, filter_fn=siren, modulation_fn=ExponentialModulation, modulation_args=self.modulation_args, dropout=self.dropout_rate, pos_emb_dim=self.positional_embedding_dimension, features=self.hyena_features, order=self.hyena_order, **(self.hyena_args or {}))
        if self.checkpoint_hyena:
            hyena = nn.checkpoint(hyena,static_argnums=(2,))

        skip = x

        # normalize
        if self.norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.norm_type == "rms":
            x = nn.RMSNorm()(x)

        # activate
        if 'activation' in self.diagnostics and train:
            self.sow("diagnostics", "activation_mean", jnp.mean(x))
            self.sow("diagnostics", "activation_sd", jnp.std(x))

        x = activate(x)

        # convolve (hyena)
        x = hyena(features=self.hyena_features, max_len=x.shape[-2])(x, not train, diagnostics=self.diagnostics)

        if 'residual' in self.diagnostics and train:
            self.sow ("diagnostics", "residual_mean", jnp.mean(x))
            self.sow ("diagnostics", "residual_sd", jnp.std(x))

        if 'skip' in self.diagnostics and train:
            self.sow ("diagnostics", "skip_mean", jnp.mean(skip))
            self.sow ("diagnostics", "skip_sd", jnp.std(skip))

        x = skip + x

        # MLP layer (optional)
        if self.mlp_layer:
            skip = x
            x = nn.Dense(self.dense_expansion*self.hyena_features, name="mlp", kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = activate(x)
            x = nn.Dense(self.hyena_features, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = skip + x

        return x
