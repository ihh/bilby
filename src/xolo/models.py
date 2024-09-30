from functools import partial
from dataclasses import field

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Any

from blocks import ConvDNA, ResTower, Final, UNet
from hyena import HyenaNac
from selectssm import BidirectionalMamba
from transformer import TransNao, EnformerMultiHeadAttention
from multires import MultiResBlock
from xxj import JunctionCountsModel

# DRK's params_conv.json example (some params hard-coded):
class DRKTrunk(nn.Module):
    repeat: int = 4
    layers_to_return: int = 1
    preconv_features: int = 320
    features_init: int = 384
    features: int = 768
    norm_type: str = "batch"
    bn_momentum: float = 0.9
    equivariant: str = "none"  # "none", "weak", "strict"
    checkpoint: bool = False
    @nn.compact
    def __call__(self, x, train: bool = False, diagnostics: dict = {}):
        kwargs = { "norm_type": self.norm_type,
                   "bn_momentum": self.bn_momentum,
                   "kernel_initializer": "lecun_normal",
                   "l2_scale": 1e-6,
                   "equivariant": self.equivariant,
                   "diagnostics": diagnostics }
        x = ConvDNA(filters=self.preconv_features, kernel_size=9, activation="linear", use_bias=True, pool_size=2, checkpoint=self.checkpoint, **{**kwargs, 'norm_type': None}) (x, train=train)
        x = ResTower(repeat=self.repeat, layers_to_return=self.layers_to_return, features_init=self.features_init, features_end=self.features, activation="gelu", kernel_size=5, pool_size=2, checkpoint=self.checkpoint, **kwargs) (x, train=train)
        return x


# seqlen 393216 => 32-fold dilation (5 rounds of pool_size=2) (in DRKTrunk) => crop 2048 off each end (in DRKHead) => 8192 bins
def crop_ends (x, crop: int = 2048):
    return jax.lax.dynamic_slice_in_dim (x, crop, x.shape[-2]-2*crop, -2)

class DRKHead(nn.Module):
    features: int = None
    crop: int = 2048
    checkpoint: bool = False
    @nn.compact
    def __call__(self, x, train: bool = False):
        x = crop_ends (x, self.crop)
        x = nn.gelu(x)
        final = Final
        if self.checkpoint:
            final = nn.checkpoint(final)
        x = final(units=self.features, activation="softplus", checkpoint=self.checkpoint) (x)
        return x

class DRKCNN(nn.Module):
    trunk_feature_scale: int = 1
    features: int = None
    crop: int = 2048
    equivariant: str = "none"  # "none", "weak", "strict"
    @nn.compact
    def __call__(self, x, train: bool = False):
        x = DRKTrunk(equivariant=self.equivariant,preconv_features=int(DRKTrunk.preconv_features*self.trunk_feature_scale),features_init=int(DRKTrunk.features_init*self.trunk_feature_scale),features=int(DRKTrunk.features*self.trunk_feature_scale)) (x, train)
        x = DRKHead(features=self.features, crop=self.crop) (x, train)
        return x

# Test CNN
class TestCNN(nn.Module):
    features: int = None
    crop: int = 2048
    @nn.compact
    def __call__(self, x, train: bool = False):
        kwargs = { "norm_type": "batch",
                   "bn_momentum": 0.9,
                   "kernel_initializer": "lecun_normal",
                   "l2_scale": 1e-6,
                   "train": train }
        # seqlen  => 32-fold dilation (5 rounds of pool_size=2) => crop 2048 off each end => 8192 bins
        x = DRKTrunk() (x, train)
        x = nn.max_pool (x, window_shape=(2,), strides=(2,), padding="SAME")
        x = nn.gelu(x)
        x = Final(units=self.features, activation="softplus") (x)
        return x


class DRKHyena(nn.Module):
    features: int = None
    crop: int = 2048

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

    trunk_norm_type: str = "batch"
    final_norm_type: str = None

    checkpoint_trunk: bool = False
    checkpoint_head: bool = False
    checkpoint_hyena: bool = False
    checkpoint_nac: bool = False

    nac_args: dict = field(default_factory=dict)

    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, train: bool = False):

        trunk = DRKTrunk
        if self.checkpoint_trunk:
            trunk = nn.checkpoint(trunk, static_argnums=(2,))
        x = trunk(features=self.hyena_features, norm_type=self.trunk_norm_type) (x, train, self.diagnostics)

        hyena_nac = HyenaNac
        if self.checkpoint_nac:
            hyena_nac = nn.checkpoint(hyena_nac, static_argnums=(2,))
        for _n in jnp.arange(self.hyena_layers):
            x = hyena_nac( dropout_rate=self.dropout_rate,
                            siren_features=self.siren_features,
                            siren_layers=self.siren_layers,
                            siren_freq=self.siren_freq,
                            siren_args=self.siren_args,
                            modulation_args=self.modulation_args,
                            hyena_args=self.hyena_args,
                            positional_embedding_dimension=self.positional_embedding_dimension, 
                            hyena_features=self.hyena_features,
                            hyena_order=self.hyena_order, 
                            hyena_layers=self.hyena_layers,
                            norm_type=self.norm_type,
                            bn_momentum=self.bn_momentum,
                            diagnostics=self.diagnostics, 
                            checkpoint_hyena=self.checkpoint_hyena,
                            activation=self.activation,
                            **self.nac_args,
                            ) (x, train)

        # normalize
        if self.final_norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.final_norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.final_norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.final_norm_type == "rms":
            x = nn.RMSNorm()(x)

        head = partial (DRKHead, features=self.features, crop=self.crop)
        if self.checkpoint_head:
            head = nn.checkpoint(head)
        x = head() (x, train)

        return x


class DRKTransformer(nn.Module):
    features: int = None
    crop: int = 2048

    dropout_rate: float = 0.2
    key_size: int = 64
    heads: int = 4
    pos_emb_dim: int = 32
    transformer_layers: int = 1
    transformer_features: int = 768
    num_basis_functions: int = 32
    transformer_args: dict = field(default_factory=dict)

    trunk_norm_type: str = "batch"
    norm_type: str = "none"
    unet_norm_type: str = "layer"
    bn_momentum: float = 0.9
    activation: str = "none"

    checkpoint_trunk: bool = False
    checkpoint_head: bool = False
    checkpoint_trans: bool = False
    checkpoint_nao: bool = False

    use_flash_attention: bool = False

    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, train: bool = False):

        trunk = DRKTrunk
        if self.checkpoint_trunk:
            trunk = nn.checkpoint(trunk, static_argnums=(2,))

        trans_nao = TransNao
        if self.checkpoint_nao:
            trans_nao = nn.checkpoint(trans_nao, static_argnums=(2,))

        if self.norm_type == "layer":
            norm = nn.LayerNorm()
        elif self.norm_type == "group":
            norm = nn.GroupNorm()
        elif self.norm_type == "rms":
            norm = nn.RMSNorm()
        elif self.norm_type == "none":
            norm = lambda x: x
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

        # Trunk
        [x32,x64,x] = trunk(features=self.transformer_features, repeat=6, layers_to_return=3, norm_type=self.trunk_norm_type) (x, train)

        for _n in jnp.arange(self.transformer_layers):
            x = trans_nao( dropout_rate=self.dropout_rate,
                            key_size=self.key_size,
                            heads=self.heads,
                            pos_emb_dim=self.pos_emb_dim,
                            transformer_features=self.transformer_features,
                            transformer_args=self.transformer_args,
                            norm_type=self.norm_type,
                            bn_momentum=self.bn_momentum,
                            activation=self.activation,
                            checkpoint_trans=self.checkpoint_trans,
                            diagnostics=self.diagnostics,
                            use_flash_attention=self.use_flash_attention,
                            **self.transformer_args,
                            ) (x, train)

        # Combine transformer and trunk outputs
        x = UNet (kernel_size=3, norm_type=self.unet_norm_type) (x, x64, train)
        x = UNet (kernel_size=3, norm_type=self.unet_norm_type) (x, x32, train)

        x = nn.Dropout(rate=self.dropout_rate) (x, deterministic=not train)

        # Head
        head = partial (DRKHead, features=self.features, crop=self.crop)
        if self.checkpoint_head:
            head = nn.checkpoint(head, static_argnums=(2,))
        x = head() (x, train)

        return x



class StripedHyena(nn.Module):
    features: int = None
    crop: int = 2048

    dropout_rate: float = 0.2
    siren_features: int = 16
    siren_layers: int = 4
    siren_freq: float = 300.0
    siren_args: Any = None
    modulation_args: Any = None
    hyena_args: Any = None
    positional_embedding_dimension: int = 8
    hyena_features: int = 768
    hyena_order: int = 2
    hyena_layers: int = 1  # per transformer layer
    norm_type: str = None
    bn_momentum: float = 0.9
    activation: str = "gelu"

    trans_pool_size: int = 4
    key_size: int = 64
    heads: int = 4
    pos_emb_dim: int = 32
    transformer_layers: int = 1
    transformer_args: dict = field(default_factory=dict)

    trunk_norm_type: str = "batch"
    final_norm_type: str = None

    checkpoint_trunk: bool = False
    checkpoint_head: bool = False
    checkpoint_hyena: bool = False
    checkpoint_nac: bool = False
    checkpoint_trans: bool = False
    checkpoint_nao: bool = False

    nac_args: dict = field(default_factory=dict)

    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, train: bool = False):

        trunk = DRKTrunk
        if self.checkpoint_trunk:
            trunk = nn.checkpoint(trunk, static_argnums=(2,))
        x = trunk(features=self.hyena_features, norm_type=self.trunk_norm_type) (x, train, diagnostics=self.diagnostics)

        hyena_nac = HyenaNac
        if self.checkpoint_nac:
            hyena_nac = nn.checkpoint(hyena_nac, static_argnums=(2,))

        trans_nao = TransNao
        if self.checkpoint_nao:
            trans_nao = nn.checkpoint(trans_nao, static_argnums=(2,))

        # Transformer tower
        for _n in jnp.arange(self.transformer_layers):
            for _n in jnp.arange(self.hyena_layers):
                x = hyena_nac( dropout_rate=self.dropout_rate,
                                siren_features=self.siren_features,
                                siren_layers=self.siren_layers,
                                siren_freq=self.siren_freq,
                                siren_args=self.siren_args,
                                modulation_args=self.modulation_args,
                                hyena_args=self.hyena_args,
                                positional_embedding_dimension=self.positional_embedding_dimension, 
                                hyena_features=self.hyena_features,
                                hyena_order=self.hyena_order, 
                                hyena_layers=self.hyena_layers,
                                norm_type=self.norm_type,
                                bn_momentum=self.bn_momentum,
                                diagnostics=self.diagnostics, 
                                checkpoint_hyena=self.checkpoint_hyena,
                                activation=self.activation,
                                **self.nac_args,
                                ) (x, train)

            x_pooled = nn.max_pool(x, window_shape=(self.trans_pool_size,), strides=(self.trans_pool_size,), padding='SAME')
            
            x_pooled = trans_nao( dropout_rate=self.dropout_rate,
                            key_size=self.key_size,
                            heads=self.heads,
                            pos_emb_dim=self.pos_emb_dim,
                            transformer_features=self.hyena_features,
                            transformer_args=self.transformer_args,
                            norm_type=self.norm_type,
                            bn_momentum=self.bn_momentum,
                            activation=self.activation,
                            checkpoint_trans=self.checkpoint_trans,
                            diagnostics=self.diagnostics,
                            use_flash_attention=True,
                            **self.transformer_args,
                            ) (x_pooled, train)

            x = UNet (kernel_size=3, norm_type='batch') (x_pooled, x, train)

        # normalize
        if self.final_norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.final_norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.final_norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.final_norm_type == "rms":
            x = nn.RMSNorm()(x)

        head = partial (DRKHead, features=self.features, crop=self.crop)
        if self.checkpoint_head:
            head = nn.checkpoint(head, static_argnums=(2,))
        x = head() (x, train)

        return x



class StripedMamba(nn.Module):
    features: int = None
    crop: int = 2048

    dropout_rate: float = 0.2
    norm_type: str = "layer"
    bn_momentum: float = 0.9
    activation: str = "gelu"

    mamba_features: int = 768
    mamba_expansion_factor: int = 1
    ssm_hidden_features: int = 8
    bn_momentum: float = 0.9
    mamba_layers: int = 3  # per transformer layer
    use_rope: bool = False

    trans_pool_size: int = 4
    key_size: int = 64
    heads: int = 4
    pos_emb_dim: int = 32
    transformer_layers: int = 2

    trunk_norm_type: str = "batch"
    unet_norm_type: str = "batch"
    final_norm_type: str = None

    checkpoint_trunk: bool = False
    checkpoint_head: bool = False
    checkpoint_trans: bool = False
    checkpoint_nao: bool = False

    transformer_args: dict = field(default_factory=dict)
    mamba_args: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, train: bool = False):

        trunk = DRKTrunk
        if self.checkpoint_trunk:
            trunk = nn.checkpoint(trunk, static_argnums=(2,))
        
        x = trunk(features=self.mamba_features, norm_type=self.trunk_norm_type) (x, train, diagnostics=self.diagnostics)

        trans_nao = TransNao
        if self.checkpoint_nao:
            trans_nao = nn.checkpoint(trans_nao, static_argnums=(2,))

        # Transformer tower
        for _n in jnp.arange(self.transformer_layers):
            for _n in jnp.arange(self.mamba_layers):
                x = BidirectionalMamba( hidden_features=self.ssm_hidden_features,
                                    expansion_factor=self.mamba_expansion_factor,
                                    norm_type=self.norm_type,
                                    bn_momentum=self.bn_momentum,
                                    diagnostics=self.diagnostics, 
                                    concatenate_fwd_rev=False,
                                    tie_in_proj=True,
                                    **self.mamba_args,
                                    ) (x, train=train)

            x_pooled = nn.max_pool(x, window_shape=(self.trans_pool_size,), strides=(self.trans_pool_size,), padding='SAME')
            
            x_pooled = trans_nao( dropout_rate=self.dropout_rate,
                            key_size=self.key_size,
                            heads=self.heads,
                            pos_emb_dim=self.pos_emb_dim,
                            transformer_features=self.mamba_features,
                            transformer_args=self.transformer_args,
                            norm_type=self.norm_type,
                            bn_momentum=self.bn_momentum,
                            activation=self.activation,
                            checkpoint_trans=self.checkpoint_trans,
                            diagnostics=self.diagnostics,
                            use_flash_attention=True,
                            use_rope=self.use_rope,
                            **self.transformer_args,
                            ) (x_pooled, train)

            x = UNet (kernel_size=3, norm_type=self.unet_norm_type) (x_pooled, x, train)

        # normalize
        if self.final_norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.final_norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.final_norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.final_norm_type == "rms":
            x = nn.RMSNorm()(x)

        head = partial (DRKHead, features=self.features, crop=self.crop)
        if self.checkpoint_head:
            head = nn.checkpoint(head, static_argnums=(2,))
        x = head() (x, train)

        return x



class DRKMamba(nn.Module):
    features: int = None
    crop: int = 2048

    mamba_features: int = 768
    mamba_expansion_factor: int = 2
    ssm_hidden_features: int = 8
    bn_momentum: float = 0.9
    mamba_layers: int = 1

    mamba_args: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    norm_type: str = "layer"
    trunk_norm_type: str = "layer"

    @nn.compact
    def __call__(self, x, train: bool = False):

        mamba = nn.checkpoint(BidirectionalMamba, static_argnums=(2,))
        head = nn.checkpoint(DRKHead, static_argnums=(2,))
        trunk = nn.checkpoint(DRKTrunk, static_argnums=(2,))

        x = trunk(features=self.mamba_features, norm_type=self.trunk_norm_type, checkpoint=True) (x, train)

        for _n in jnp.arange(self.mamba_layers):
            x = mamba( hidden_features=self.ssm_hidden_features,
                        expansion_factor=self.mamba_expansion_factor,
                        norm_type=self.norm_type,
                        bn_momentum=self.bn_momentum,
                        diagnostics=self.diagnostics, 
                        **self.mamba_args,
                        ) (x, train)

        x = head(features=self.features, crop=self.crop, checkpoint=True) (x, train)

        return x


class DRKMulti(nn.Module):
    features: int = None
    crop: int = 2048

    multi_features: int = 768
    multi_layers: int = 1

    trunk_norm_type: str = "layer"

    multi_args: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, train: bool = False):

        x = DRKTrunk(features=self.multi_features, norm_type=self.trunk_norm_type) (x, train, diagnostics=self.diagnostics)

        for _n in jnp.arange(self.multi_layers):
            x = MultiResBlock ( diagnostics=self.diagnostics, 
                                **self.multi_args,
                                ) (x, train=train)

        x = DRKHead(features=self.features, crop=self.crop) (x, train)

        return x




class DRKXXJ(nn.Module):
    features: int = None
    xxj_features: int = None
    crop: int = 2048

    mamba_features: int = 768
    mamba_expansion_factor: int = 0.75
    ssm_hidden_features: int = 8
    bn_momentum: float = 0.9
    mamba_layers: int = 6

    mamba_args: dict = field(default_factory=dict)
    xxj_args: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    norm_type: str = "layer"
    trunk_norm_type: str = "layer"

    # x has shape (B,L,I)
    # xxj_sparse has shape (N,4) where each row has the form (b,t,d,a) where b=batch, t=target, d=donor bin, a=acceptor bin
    @nn.compact
    def __call__(self, x, xxj_sparse, train: bool = False):

        trunk = nn.checkpoint(DRKTrunk, static_argnums=(2,))
        x = trunk(features=self.mamba_features, norm_type=self.trunk_norm_type) (x, train, diagnostics=self.diagnostics)

        for _n in jnp.arange(self.mamba_layers):
            mamba = nn.checkpoint(BidirectionalMamba, static_argnums=(2,))
            x = mamba( hidden_features=self.ssm_hidden_features,
                        expansion_factor=self.mamba_expansion_factor,
                        norm_type=self.norm_type,
                        bn_momentum=self.bn_momentum,
                        diagnostics=self.diagnostics, 
                        concatenate_fwd_rev=False,
                        tie_in_proj=True,
                        **self.mamba_args,
                        ) (x, train)

        x = crop_ends (x, self.crop)

        y = nn.gelu(x)
        final = nn.checkpoint(Final)
        y = final(units=self.features, activation="softplus") (x)

        jcmodel = nn.checkpoint(JunctionCountsModel, static_argnums=(3,))
        xxj = jcmodel(features=self.xxj_features, diagnostics=self.diagnostics, **self.xxj_args) (x, xxj_sparse, train)

        return y, xxj

models = { 'testcnn': { "new_model": TestCNN, "seq_len": 393216, "targets_length": 8192 },
           'drkcnn': { "new_model": DRKCNN, "seq_len": 393216, "targets_length": 8192 },
           'drkhyena': { "new_model": DRKHyena, "seq_len": 393216, "targets_length": 8192 },
           'drktrans': { "new_model": DRKTransformer, "seq_len": 393216, "targets_length": 8192 },
           'drkmamba': { "new_model": DRKMamba, "seq_len": 393216, "targets_length": 8192 },
           'drkmulti': { "new_model": DRKMulti, "seq_len": 393216, "targets_length": 8192 },
           'stripedhyena': { "new_model": StripedHyena, "seq_len": 393216, "targets_length": 8192 },
           'stripedmamba': { "new_model": StripedMamba, "seq_len": 393216, "targets_length": 8192 },
           'drkxxj': { "new_model": DRKXXJ, "seq_len": 393216, "targets_length": 8192, "predicts_xxj": True },
         }
