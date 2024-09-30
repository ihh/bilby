import logging
from typing import Any, Callable, Sequence, Union, Tuple, Optional
from dataclasses import field

import math

import einops
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from poisson import l2_norm

# See Figure 3, Appendix B in the following paper:
#  Sequence Modeling with Multiresolution Convolutional Memory
#  https://arxiv.org/abs/2305.01638
#  Jiaxin Shi, Ke Alexander Wang, Emily B. Fox
class MultiResLayer(nn.Module):
    filter_width: int = 2       # K
    depth: Optional[int] = None # J

    pad_right: bool = False

    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(
        self,
        x,  # (B, L, D)
        train: bool = False,
    ):
        B = x.shape[-3]
        L = x.shape[-2]
        D = x.shape[-1]
        
        K = self.filter_width
        J = math.ceil(math.log((L - 1) / (K - 1) + 1) / math.log(K))  # Shi et al have the log base as 2, but we can dilate faster if we have bigger kernels
        if self.depth is not None:
            J = jnp.minimum (J, self.depth)

        # Note on initialization:
        # After J dilated convolutions, the final sum (a_J) will be like the expansion of (sum_{k=1}^K h_k)^J with each term multiplied by an independent x_i
        # ... for example, a_2 = x_1 * (h_1^2) + (x_2 + x_3) * (h_1 * h_2) + x_4 * (h_2^2)
        # Thus E[a_J] = 0 and V[a_J] = E[a_J^2] = E[H^J] * V[x_i]  where H = sum_{k=1}^K h_k^2  because cross-terms involving x_i*x_j will disappear unless i=j (since we assume E[x_i]=0)
        # Now H/V[h] ~ ChiSquared(K) so E[H^J] = (2*V[h])^J * gamma(J + K/2) / gamma(K/2)
        # Thus for V[a_J]=1, assuming V[x_i]=1, we want V[h] = (gamma(J + K/2) / gamma(K/2))^(1/J) / 2
        h_scale = jnp.sqrt ((gamma(K/2) / gamma(J + K/2))**(1/J) / 2)
        h0 = self.param('h0', nn.initializers.normal(stddev=h_scale), (D, K))[None,:,:]  # (1, D, K)
        h1 = self.param('h1', nn.initializers.normal(stddev=h_scale), (D, K))[None,:,:]  # (1, D, K)
        w = self.param('w', nn.initializers.normal(stddev=1/jnp.sqrt(J+2)), (J+2, D))  # (J+2, D)

        y = jnp.zeros_like(x)
        a = x
        for j, wj in enumerate(w[1:-1,:]):
            dilation = K ** j   # Shi et al have this as 2**j, but we can dilate faster if we have bigger kernels
            padding = dilation * (K - 1)
            a = jnp.pad (a, ((0,0),(0,padding) if self.pad_right else (padding,0),(0,0)))
            b = jax.lax.conv_general_dilated (lhs=a, rhs=h1, window_strides=(1,),
                                                padding='VALID',
                                                dimension_numbers=('NLC', 'IOL', 'NLC'),
                                                rhs_dilation=(dilation,),
                                                feature_group_count=D)
            a = jax.lax.conv_general_dilated (lhs=a, rhs=h0, window_strides=(1,),
                                                padding='VALID',
                                                dimension_numbers=('NLC', 'IOL', 'NLC'),
                                                rhs_dilation=(dilation,),
                                                feature_group_count=D)

            y = y + wj[None,None,:] * b

            if 'multires' in self.diagnostics and train:
                self.sow ("diagnostics", f"a{j}_mean", jnp.mean(a))
                self.sow ("diagnostics", f"a{j}_sd", jnp.std(a))
                self.sow ("diagnostics", f"b{j}_mean", jnp.mean(b))
                self.sow ("diagnostics", f"b{j}_sd", jnp.std(b))
                self.sow ("diagnostics", f"y{j}_mean", jnp.mean(y))
                self.sow ("diagnostics", f"y{j}_sd", jnp.std(y))
        
        y = y + w[None,:1,:] * x + w[None,-1:,:] * a

        return y


class MultiResBlock(nn.Module):
    filter_width: int = 2       # K
    depth: Optional[int] = None # J

    activation: str = "gelu"
    norm_type: str = "rms"

    bn_momentum: float = 0.9

    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(
        self,
        x,  # (B, L, D)
        train: bool = False,
    ):
        # activation function
        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = nn.relu
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        # skip connection
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

        # MultiResConv
        x = MultiResLayer(filter_width=self.filter_width, depth=self.depth, diagnostics=self.diagnostics)(x, train=train)

        if 'activation' in self.diagnostics and train:
            self.sow ("diagnostics", "activation_mean", jnp.mean(x))
            self.sow ("diagnostics", "activation_sd", jnp.std(x))

        # activate
        x = activate(x)

        if 'activated' in self.diagnostics and train:
            self.sow ("diagnostics", "activated_mean", jnp.mean(x))
            self.sow ("diagnostics", "activated_sd", jnp.std(x))

        # 1x1 convolution, doubling up features for the GLU
        x = nn.Conv(features=2*x.shape[-1], kernel_size=(1,))(x)

        # GLU
        x = nn.glu(x)

        if 'residual' in self.diagnostics and train:
            self.sow ("diagnostics", "skip_mean", jnp.mean(skip))
            self.sow ("diagnostics", "skip_sd", jnp.std(skip))
            self.sow ("diagnostics", "residual_mean", jnp.mean(x))
            self.sow ("diagnostics", "residual_sd", jnp.std(x))

        # residual
        x = x + skip

        return x