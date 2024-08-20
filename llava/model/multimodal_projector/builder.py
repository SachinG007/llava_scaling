import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, tower_config, delay_load=False, **kwargs):
    modules = []

    token_compression_type = getattr(config, 'mm_vision_token_compression_type', None)
    if token_compression_type == 'token-packer':
        from .token_packer import TokenPacker
        token_compressor = TokenPacker(hidden_size=config.hidden_size, down_rate=config.token_packer_down_rate)
        token_compressor.requires_grad_(True)
        modules.append(token_compressor)    
        return token_compressor
    elif token_compression_type == 'token-packer-reduced':
        from .token_packer_reduced import TokenPacker
        token_compressor = TokenPacker(hidden_size=config.hidden_size, down_rate=config.token_packer_down_rate)
        token_compressor.requires_grad_(True)
        modules.append(token_compressor)    
        return token_compressor

    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        modules = build_token_compressor(modules, config, tower_config)
        return nn.Sequential(*modules)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        modules = build_token_compressor(modules, config, tower_config)
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_token_compressor(modules, config, tower_config):
    token_compression_type = getattr(config, 'mm_vision_token_compression_type', None)
    if token_compression_type is not None:
        if token_compression_type == 'self-attn':
            from .token_compressor import TokenAttention as Compressor
        elif token_compression_type == "self-attn-deep":
            from .token_compressor import TokenAttentionDeep as Compressor
        elif token_compression_type == 'conv-self-attn':
            from .token_compressor import TokenConvAttention as Compressor
        elif token_compression_type == "conv-self-attn-deep":
            from .token_compressor import TokenConvAttentionDeep as Compressor
        elif token_compression_type == "local-conv-self-attn-deep":
            from .token_compressor import TokenLocalConvAttentionDeep as Compressor
        elif token_compression_type == 'conv':
            from .token_compressor import TokenConv as Compressor
        elif token_compression_type == 'attn-prune':
            from .token_compressor import TokenAttnPrune as Compressor
        elif token_compression_type == "query-attn":
            from .query_based_token_compressor import QueryTokenAttention as Compressor
        elif token_compression_type == "conv-embedding":
            from .conv_embedding_compressor import ConvEmbeddingCompressor as Compressor
        elif token_compression_type == "query-attn-deep":
            from .query_based_token_compressor import QueryTokenAttentionDeep as Compressor
        elif token_compression_type == "query-attn-deep-lessparams":
            from .query_based_token_compressor import QueryTokenAttentionDeepLessParams as Compressor
        elif token_compression_type == "half-query-attn-deep":
            from .query_based_token_compressor import HalfQueryTokenAttentionDeep as Compressor
        elif token_compression_type == "entity-attn-deep":
            from .query_based_token_compressor import EntityTokenAttentionDeep as Compressor
        elif token_compression_type == "query-local-conv-self-attn-deep":
            from .query_based_token_compressor import QueryLocalConvAttentionDeep as Compressor
        
        
        token_compressor = Compressor(tower_config=tower_config, training_config=config)
        token_compressor.requires_grad_(True)

        placement = getattr(config, 'mm_vision_token_compressor_placement', 'start')
        if placement == 'start':
            modules.insert(0, token_compressor)
        elif placement == 'end':
            modules.append(token_compressor)
        else:
            raise Exception(f"Unknown mm_vision_token_compressor_placement {placement}")
    
    return modules
