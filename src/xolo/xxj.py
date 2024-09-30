from dataclasses import field
import itertools
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn

from poisson import l2_norm

class JunctionCountsModel(nn.Module):
    features: int = None
    hidden_features: int = 768
    distance_features: int = None  # size of distance embedding, which is just a binary threshold function

    use_gather_instead_of_slice: bool = False
    debug_print: bool = True

    l2_scale: float = 1e-6

    diagnostics: dict = field(default_factory=dict)

    # Inputs:
    #  x has shape (B,L,K) where B=batch, L=length, K=features
    #  xxj_sparse has shape (N,3) where each row has the form (b,d,a) where b=batch, d=donor bin, a=acceptor bin
    # Output:
    #  y has shape (N,T) where T=number of output features
    @nn.compact
    def __call__(self, x, xxj_sparse, train: bool = False):

        B = x.shape[0]
        L = x.shape[1]
        K = x.shape[2]
        N = xxj_sparse.shape[0]

        T = self.features
        H = self.hidden_features
        D = self.distance_features or L.bit_length() - 1

        if 'xxj' in self.diagnostics:
            self.sow("diagnostics", "xxj_input_mean", jnp.mean(x))
            self.sow("diagnostics", "xxj_input_sd", jnp.std(x))

        # Extract batch, target, donor and acceptor indices
        batch_indices = xxj_sparse[:,0]  # (N,)
        donor_indices = xxj_sparse[:,1]  # (N,)
        acceptor_indices = xxj_sparse[:,2]  # (N,)

        # Retrieve/initialize parameters
        stddev_trunc = .87962566103423978  # stddev of standard normal truncated to (-2, 2)
        input_init = nn.initializers.truncated_normal (stddev=jnp.sqrt(2/(2*K+D+1+H))/stddev_trunc, lower=-2.0, upper=2.0)
        zeros = nn.initializers.zeros
        mlp_donor_input_params = self.param ('mlp_donor_input', input_init, (K, H))
        mlp_acceptor_input_params = self.param ('mlp_acceptor_input', input_init, (K, H))
        mlp_distance_input_params = self.param ('mlp_distance_input', input_init, (H,))
        mlp_dist_embed_input_params = self.param ('mlp_dist_embed_input', input_init, (D, H))
        mlp_input_bias_params = self.param ('mlp_input_bias', zeros, (H,))

        # Take donor and acceptor slices of input
        if self.use_gather_instead_of_slice:   # for performance comparison; closer to XLA backend?
            batch_donor_indices_for_gather = xxj_sparse[:,(0,1)]  # (N,2)
            x_donor = jax.lax.gather (x, batch_donor_indices_for_gather, jax.lax.GatherDimensionNumbers((1,),(0,1),(0,1)),[1,1,K])  # (N, K)

            batch_acceptor_indices_for_gather = xxj_sparse[:,(0,2)]  # (N,2)
            x_acceptor = jax.lax.gather (x, batch_acceptor_indices_for_gather, jax.lax.GatherDimensionNumbers((1,),(0,1),(0,1)),[1,1,K])  # (N, K)

        else:  # default to slices which are much simpler to understand
            x_donor = x[batch_indices,donor_indices,:]  # (N, K)
            x_acceptor = x[batch_indices,acceptor_indices,:]  # (N, K)

        # Calculate distances between donor and acceptor indices (as fraction of total length), and a binary-threshold embedding
        distances = jnp.abs (acceptor_indices - donor_indices)  # (N,)
        dist_embed = distances[:,None] >= 1<<jnp.arange(D)[None,:]  # (N, D)
        distances = distances / L

        # MLP input -> hidden layer
        y = jnp.einsum ('nk,kh->nh', x_donor, mlp_donor_input_params)
        y = y + jnp.einsum ('nk,kh->nh', x_acceptor, mlp_acceptor_input_params)
        y = y + jnp.einsum ('n,h->nh', distances, mlp_distance_input_params)
        y = y + jnp.einsum ('nd,dh->nh', dist_embed, mlp_dist_embed_input_params)
        y = y + mlp_input_bias_params[None,:]  # (N, H)
        y = nn.gelu(y)  # (N, H)

        if self.debug_print:
            jax.debug.print ("MLP hidden layer: shape {} mean {} sd {}", y.shape, jnp.mean(y), jnp.std(y))

        if 'xxj' in self.diagnostics:
            self.sow("diagnostics", "xxj_hidden_mean", jnp.mean(y))
            self.sow("diagnostics", "xxj_hidden_sd", jnp.std(y))

        # Hidden layer -> output
        out_layer = nn.Dense (T, kernel_init=nn.initializers.glorot_normal())
        y = out_layer (y)  # (N, T)
        y = nn.softplus(y)  # (N, T)

        if self.debug_print:
            jax.debug.print ("MLP output layer: shape {} mean {} sd {}", y.shape, jnp.mean(y), jnp.std(y))

        if 'xxj' in self.diagnostics:
            self.sow("diagnostics", "xxj_output_mean", jnp.mean(y))
            self.sow("diagnostics", "xxj_output_sd", jnp.std(y))

        # Regularization
        if train:
            self.sow("losses", "xxj_hidden_regularizer", l2_norm (self.variables['params'], self.l2_scale))
            self.sow("losses", "xxj_output_regularizer", l2_norm (out_layer.variables['params'], self.l2_scale))

        return y