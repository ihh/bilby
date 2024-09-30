from dataclasses import field
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from poisson import l2_norm
from dna import RevCompEquivariantConv1D, RevCompEquivariantDense, WeakRevCompEquivariantConv1D, WeakRevCompEquivariantDense

class ConvNac(nn.Module):
    filters: int = None
    kernel_size: int = 1
    activation: str = "relu"
    stride: int = 1
    dilation_rate: int = 1
    l2_scale: float = 0
    dropout: float = 0
    residual: bool = False
    pool_size: int = 1
    norm_type: str = None
    bn_momentum: float = 0.99
    kernel_initializer: str = "he_normal"
    padding: str = "SAME"
    equivariant: str = "none"  # "none", "weak", "strict"
    checkpoint: bool = False
    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs, train=False):
        """Construct a single convolution block.

            Args:
            inputs:        [batch_size, seq_length, features] input sequence
            filters:       Conv1D filters
            kernel_size:   Conv1D kernel_size
            activation:    relu/gelu/etc
            stride:        Conv1D stride
            dilation_rate: Conv1D dilation rate
            l2_scale:      L2 regularization weight.
            dropout:       Dropout rate probability
            residual:      Residual connection boolean
            pool_size:     Max pool width
            norm_type:     Apply batch or layer normalization
            bn_momentum:   BatchNorm momentum

            Returns:
            [batch_size, seq_length, features] output sequence
        """

        # flow through variable current
        current = inputs

        filters = self.filters if self.filters is not None else inputs.shape[-1]
        
        # normalize
        if self.norm_type == "batch":
            current = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(current)
        elif self.norm_type == "layer":
            current = nn.LayerNorm()(current)
        elif self.norm_type == "group":
            current = nn.GroupNorm()(current)
        elif self.norm_type == "rms":
            current = nn.RMSNorm()(current)
        else:
            raise Exception(f"Unknown norm_type: {self.norm_type}")

        if 'activation' in self.diagnostics and train:
            self.sow("diagnostics", "activation_mean", jnp.mean(current))
            self.sow("diagnostics", "activation_sd", jnp.std(current))

        # activation
        if self.activation == "relu":
            current = nn.relu (current)
        elif self.activation == "gelu":
            current = nn.gelu (current)
        elif self.activation != "linear":
            raise Exception(f"Unknown activation: {self.activation}")

        kernel_init = (nn.initializers.he_normal if self.kernel_initializer=="he_normal" else nn.initializers.lecun_normal)

        if self.equivariant == "none":
            conv_cls = nn.Conv
        elif self.equivariant == "weak":
            conv_cls = WeakRevCompEquivariantConv1D
        elif self.equivariant == "strict":
            conv_cls = RevCompEquivariantConv1D
        else:
            raise Exception(f"Unknown equivariance type: {self.equivariant}")

        if self.checkpoint:
            conv_cls = nn.checkpoint(conv_cls)

        # convolution
        conv = conv_cls(
            features=filters,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=self.padding,
            use_bias=True,
            kernel_dilation=self.dilation_rate,
            kernel_init=kernel_init(),
        )
        current = conv(current)

        if train:
            self.sow("losses", "kernel_regularizer", l2_norm (conv.variables['params'], self.l2_scale))

        # dropout
        if self.dropout > 0:
            current = nn.Dropout(rate=self.dropout, deterministic=not train)(current)

        # residual add
        if 'residual' in self.diagnostics and train:
            self.sow("diagnostics", "residual_mean", jnp.mean(current))
            self.sow("diagnostics", "residual_sd", jnp.std(current))

        if 'skip' in self.diagnostics and train:
            self.sow("diagnostics", "skip_mean", jnp.mean(inputs))
            self.sow("diagnostics", "skip_sd", jnp.std(inputs))

        if self.residual:
            current = current + inputs

        # Pool
        if self.pool_size > 1:
            current = nn.max_pool(current, window_shape=(self.pool_size,), strides=(self.pool_size,), padding=self.padding)

        return current


class ConvDNA(nn.Module):

    filters: int = None
    kernel_size: int = 15
    activation: str = "relu"
    stride: int = 1
    l2_scale: float = 0
    dropout: float = 0
    pool_size: int = 1
    norm_type: str = None
    bn_momentum: float = 0.99
    use_bias: bool = None
    kernel_initializer: str = "he_normal"
    padding: str = "SAME"
    equivariant: str = "none"  # "none", "weak", "strict"
    checkpoint: bool = False
    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs, train=False):
        """Construct a single convolution block, assumed to be operating on DNA.

            Args:
            inputs:        [batch_size, seq_length, features] input sequence
            filters:       Conv1D filters
            kernel_size:   Conv1D kernel_size
            activation:    relu/gelu/etc
            stride:        Conv1D stride
            l2_scale:      L2 regularization weight.
            dropout:       Dropout rate probability
            conv_type:     Conv1D layer type
            pool_size:     Max pool width
            norm_type:     Apply batch or layer normalization
            bn_momentum:   BatchNorm momentum

            Returns:
            [batch_size, seq_length, features] output sequence
        """

        # flow through variable current
        current = inputs

        filters = self.filters if self.filters is not None else inputs.shape[-1]

        kernel_init = (nn.initializers.he_normal if self.kernel_initializer=="he_normal" else nn.initializers.lecun_normal)

        if self.equivariant == "none":
            conv_cls = nn.Conv
        elif self.equivariant == "weak":
            conv_cls = WeakRevCompEquivariantConv1D
        elif self.equivariant == "strict":
            conv_cls = RevCompEquivariantConv1D
        else:
            raise Exception(f"Unknown equivariance type: {self.equivariant}")

        if self.checkpoint:
            conv_cls = nn.checkpoint(conv_cls)

        # convolution
        conv = conv_cls(
            features=filters,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=kernel_init(),
        )
        current = conv(current)

        if train:
            self.sow("losses", "kernel_regularizer", l2_norm (conv.variables['params'], self.l2_scale))

        # normalize
        if self.norm_type == "batch":
            current = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(current)
        elif self.norm_type == "layer":
            current = nn.LayerNorm()(current)
        elif self.norm_type == "group":
            current = nn.GroupNorm()(current)
        elif self.norm_type == "rms":
            current = nn.RMSNorm()(current)
        
        if 'activation' in self.diagnostics and train:
            self.sow("diagnostics", "activation_mean", jnp.mean(current))
            self.sow("diagnostics", "activation_sd", jnp.std(current))

        # activation
        if self.activation == "relu":
            current = nn.relu (current)
        elif self.activation == "gelu":
            current = nn.gelu (current)
        elif self.activation != "linear":
            raise Exception(f"Unknown activation: {self.activation}")

        # dropout
        if self.dropout > 0:
            current = nn.Dropout(rate=self.dropout, deterministic=not train)(current)

        # Pool
        if self.pool_size > 1:
            current = nn.max_pool(current, window_shape=(self.pool_size,), strides=(self.pool_size,), padding=self.padding)

        return current

def make_filter_schedule (features_init, features_end, repeat, divisible_by):

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    mul = np.exp(np.log(features_end / features_init) / (repeat - 1))
    
    filter_schedule = []
    features = features_init
    for ri in range(repeat):
        filter_schedule.append (_round(features))
        features *= mul

    return filter_schedule

class ResLayer(nn.Module):
    filters: int = None
    kernel_size: int = 1
    dropout: float = 0

    activation: str = "relu"
    norm_type: str = "none"
    bn_momentum: float = 0.99
    kernel_initializer: str = "he_normal"
    l2_scale: float = 0

    equivariant: str = "none"  # "none", "weak", "strict"
    checkpoint: bool = False

    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__ (self, x, train=False, **kwargs):

        nac = ConvNac(filters=self.filters, equivariant=self.equivariant,
                    kernel_size=self.kernel_size, activation=self.activation,
                    norm_type=self.norm_type, bn_momentum=self.bn_momentum, kernel_initializer=self.kernel_initializer,
                    l2_scale=self.l2_scale,
                    checkpoint=self.checkpoint,
                    )

        x = nac (x, train=train)
        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout, deterministic=not train) (x)
        return x

class ResTower(nn.Module):
    features_init: int = 384
    features_end: int = 768
    repeat: int = 4
    layers_to_return: int = 1  # set to >1 for U-Net style incorporation of intermediate layers
    divisible_by: int = 16

    activation: str = "relu"
    kernel_size: int = 1
    pool_size: int = 2

    dropout: float = 0

    norm_type: str = "none"
    bn_momentum: float = 0.99
    kernel_initializer: str = "he_normal"
    l2_scale: float = 0

    equivariant: str = "none"  # "none", "weak", "strict"
    checkpoint: bool = False
    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__ (self, x, train=False):
        filter_schedule = make_filter_schedule (self.features_init, self.features_end, self.repeat, self.divisible_by)

        result = []
        for layer, filters in enumerate(filter_schedule):
            x = ResLayer(filters=filters,
                         equivariant=self.equivariant,
                         kernel_size=self.kernel_size,
                         dropout=self.dropout,
                         activation=self.activation,
                         norm_type=self.norm_type,
                         bn_momentum=self.bn_momentum,
                         kernel_initializer=self.kernel_initializer,
                         l2_scale=self.l2_scale,
                         diagnostics=self.diagnostics,
                         checkpoint=self.checkpoint,
                        ) (x, train=train)
            if self.layers_to_return > 1 and layer > self.repeat - self.layers_to_return:
                result.append(x)
            if self.pool_size > 1:
                x = nn.max_pool(x, window_shape=(self.pool_size,), strides=(self.pool_size,), padding='SAME')

        return result + [x] if self.layers_to_return > 1 else x


class Final(nn.Module):
    units: int = 1
    activation: str = "linear"
    kernel_initializer: str = "he_normal"
    checkpoint: bool = False
    l2_scale: float = 0

    @nn.compact
    def __call__ (self, inputs, train=False, **kwargs):

        current = inputs

        kernel_init = (nn.initializers.he_normal if self.kernel_initializer=="he_normal" else nn.initializers.lecun_normal)

        dense_cls = nn.Dense
        if self.checkpoint:
            dense_cls = nn.checkpoint(dense_cls)

        dense = dense_cls (self.units, kernel_init=kernel_init())
        current = dense(current)

        if train:
            self.sow("losses", "kernel_regularizer", l2_norm (dense.variables['params'], self.l2_scale))
        
        # activation
        if self.activation == "relu":
            current = nn.relu (current)
        elif self.activation == "gelu":
            current = nn.gelu (current)
        elif self.activation == "softplus":
            current = nn.activation.softplus (current)
        elif self.activation != "linear":
            raise Exception(f"Unknown activation: {self.activation}")

        return current


class UNet(nn.Module):
    kernel_size: int = 3
    norm_type: str = "none"
    activation: str = "relu"
    bn_momentum: float = 0.99
    upsample_conv: bool = False
    equivariant: str = "none"  # "none", "weak", "strict"

    @nn.compact
    def __call__(self, x, u, train: bool = False):

        filters = x.shape[-1]
        
        stride = u.shape[-2] // x.shape[-2]
        assert x.shape[-2] * stride == u.shape[-2]

        if self.norm_type == "layer":
            norm = nn.LayerNorm
        elif self.norm_type == "group":
            norm = nn.GroupNorm
        elif self.norm_type == "rms":
            norm = nn.RMSNorm
        elif self.norm_type == "batch":
            norm = partial (nn.BatchNorm, momentum=self.bn_momentum, use_running_average=not train)
        elif self.norm_type == "none":
            norm = lambda: lambda x: x
        else:
            raise Exception(f"Unknown norm_type: {self.norm_type}")

        if self.activation == "relu":
            activate = nn.relu
        elif self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "none":
            activate = lambda x: x
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        if self.equivariant == "none":
            conv_cls = nn.Conv
            dense_cls = nn.Dense
        elif self.equivariant == "weak":
            conv_cls = WeakRevCompEquivariantConv1D
            dense_cls = WeakRevCompEquivariantDense
        elif self.equivariant == "strict":
            conv_cls = RevCompEquivariantConv1D
            dense_cls = RevCompEquivariantDense
        else:
            raise Exception(f"Unknown equivariance type: {self.equivariant}")
        
        x = norm()(x)
        u = norm()(u)

        x = activate(x)
        u = activate(u)

        if self.upsample_conv:
            x = dense_cls (filters, kernel_init=nn.initializers.he_normal()) (x)

        u = dense_cls (filters, kernel_init=nn.initializers.he_normal()) (u)

        x = jnp.repeat (x, stride, axis=-2)
        x = x + u

        # Perform depthwise separable convolution
        x = conv_cls (filters, feature_group_count=filters, kernel_size=(self.kernel_size,), kernel_init=nn.initializers.he_normal()) (x)
        x = dense_cls (filters, kernel_init=nn.initializers.he_normal(), use_bias=False) (x)

        return x
