import jax
import jax.numpy as jnp

import flax.linen as nn

import einops

# A 1-D locally connected layer where the (local) convolution weights are computed dynamically from a lower-dimensional projection of the input features
# See e.g. X. Jia, B. De Brabandere, T. Tuytelaars, and L. V. Gool. Dynamic filter networks. In Proc. NIPS, pages 667–675, 2016.
# https://arxiv.org/abs/1605.09673
class DynamicConv(nn.Module):

    out_features: int  # E
    inner_features: int = 8  # F
    kernel_params_conv_size: int = 1  # K1
    conv_size: int = 1  # K2

    @nn.compact
    def __call__ (
        self,
        u, # (B, L, D)
    ):
        D = u.shape[-1]
        E = self.out_features or D
        F = self.inner_features

        K1 = self.kernel_params_conv_size
        K2 = self.conv_size

        kernel_params_feat_proj = self.param("kernel_params_feat_proj",
                                                nn.initializers.normal(stddev=1/jnp.sqrt(D)),
                                                (D, F))  # (D, F)
        kernel_params_conv_weights = self.param("kernel_params_conv_weights",
                                                nn.initializers.zeros_init(),
                                                (E, K1, F, K2*D))  # (E, K1, F, K2*D)
        kernel_params_conv_bias = self.param("kernel_params_conv_bias",
                                                nn.initializers.normal(stddev=1/jnp.sqrt(K2*D)),
                                                (E, 1, K2*D, 1))  # (E, 1, K2*D, 1)
        
        # We have to map over batches because jax.lax.conv_general_dilated_local doesn't allow the filter to have a batch dimension
        def process_batch (u_batch):  # u_batch: (L, D)
            f_batch = jnp.einsum ('id,df->if', u_batch, kernel_params_feat_proj)  # (L, F)

            # We have to map over channels because the (T,D,E) tensor for the locally connected network kernel is too large to fit in GPU memory
            @jax.remat
            def process_channel (params_for_channel):
                kernel_params_conv_weights_for_channel, kernel_params_conv_bias_for_channel = params_for_channel  # (K1, F, K2*D)  (1, K2*D, 1)
                print(f"kernel_params_conv_weights_for_channel: {kernel_params_conv_weights_for_channel.shape}")
                print(f"kernel_params_conv_bias_for_channel: {kernel_params_conv_bias_for_channel.shape}")
                print(f"f_batch: {f_batch.shape}")
                kernel_params = jax.lax.conv_general_dilated (
                    lhs=f_batch[None,:,:],  # (1, L, F)
                    rhs=kernel_params_conv_weights_for_channel,
                    window_strides=(1,),
                    padding='SAME',
                    dimension_numbers=('NLC', 'LIO', 'NLC'),
                )  # (1, T, K2*D)
                kernel_params = einops.rearrange (kernel_params, '1 l i -> l i 1')  # (T, K2*D, 1)
                kernel_params = kernel_params + kernel_params_conv_bias_for_channel  # (T, K2*D, 1)
                u_channel = jax.lax.conv_general_dilated_local (
                    lhs=u_batch[None,...],  # (1, T, D)
                    rhs=kernel_params,  # (T, K2*D, 1)
                    window_strides=(1,),
                    padding='SAME',
                    filter_shape=(K2,),  # (K2,) so rhs unfolds to (T, K2, D, 1)
                    dimension_numbers=('NLC', 'LIO', 'NLC'))  # (1, T, 1)
                return u_channel[0,:,0]  # returns (T,)
            
            u_batch = jax.lax.map (process_channel, (kernel_params_conv_weights, kernel_params_conv_bias))  # (E, T)
            print(f"u_batch: {u_batch.shape}")
            return u_batch.transpose()  # (T, E)
            
        u = jax.vmap(process_batch) (u)  # (B, T, E)
        return u
