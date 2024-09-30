import jax
import jax.numpy as jnp

import flax.linen as nn

import einops

# A 1-D layer where the output is a site-specific mixture of several input convolutions.
# Reference:
#  Dynamic Convolution: Attention over Convolution Kernels
#  Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Lu Yuan, Zicheng Liu
#  https://arxiv.org/abs/1912.03458
class MixConv(nn.Module):

    out_features: int = None  # E
    mix_weight_conv_size: int = None  # K1
    mix_components: int = 8  # M
    conv_size: int = 1  # K2

    softmax_weights: bool = False

    @nn.compact
    def __call__ (
        self,
        u, # (B, L, D)
    ):
        D = u.shape[-1]
        E = self.out_features or D
        M = self.mix_components

        K2 = self.conv_size
        K1 = self.mix_weight_conv_size or K1

        components = nn.Conv (features=M*E, kernel_size=(K2,), kernel_init=nn.initializers.normal(stddev=1/jnp.sqrt(M)), padding='SAME', name='components') (u)  # (B, L, M*E)
        weights = nn.Conv (features=M, kernel_size=(K1,), kernel_init=nn.initializers.normal(stddev=1/jnp.sqrt(D+2)), padding='SAME', name='weights') (u)  # (B, L, M)

        components = einops.rearrange (components, 'b l (m e) -> b l m e', m=M)

        if self.softmax_weights:
            weights = jax.nn.softmax (weights, axis=-1)

        return jnp.einsum('...m,...me->...e', weights, components)  # (B, L, E)
