from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn


class TokenAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        
        self.embed_dim = tower_config.hidden_size
        self.num_heads = tower_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )  
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.k_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)
        
        self.q_downsample = [nn.Linear(self.visual_token_count, self.combined_token_count)]
        self.q_downsample.append(nn.GELU())
        self.q_downsample.append(nn.Linear(self.combined_token_count, self.combined_token_count))
        self.q_downsample.append(nn.LayerNorm(self.combined_token_count, eps=1e-6))
        self.q_downsample = nn.Sequential(*self.q_downsample)

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        query_states = self.q_downsample(hidden_states.transpose(1, 2)).transpose(1, 2) # B x N x D
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output

class TokenAttentionDeep(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        
        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # Token packer is 1024//128
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )  
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.k_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)
        
        self.q_downsample = [nn.Linear(self.visual_token_count, self.combined_token_count)]
        self.q_downsample.append(nn.GELU())
        self.q_downsample.append(nn.Linear(self.combined_token_count, self.combined_token_count))
        self.q_downsample.append(nn.LayerNorm(self.combined_token_count, eps=1e-6))
        self.q_downsample = nn.Sequential(*self.q_downsample)

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x, attn_mask=None):
        
        x_multi = x[1] # mulit-level
        x = x[0] # original single-level

        query_states = self.q_downsample(x.transpose(1, 2)).transpose(1, 2) # B x N x D
        key_states = self.k_proj(x_multi)
        value_states = self.v_proj(x_multi)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output

class TokenConv(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            ) # might need to figure out how to do this more efficiently since the vision encoder might change...
        
        self.embed_dim = tower_config.hidden_size

        self.kernel_size = getattr(training_config, 'mm_vision_token_compression_kernel_size', 4)
        self.stride = getattr(training_config, 'mm_vision_token_compression_stride', 4)

        # confirm we can convert the visual tokens to square grid
        assert (self.visual_token_count ** 0.5) % 1 == 0
        self.token_grid_size = int(self.visual_token_count ** 0.5)

        # ADJUST THIS LATER
        self.q_downsample = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        hidden_states_2d = einops.rearrange(hidden_states, 'b (h w) d -> b d h w',
                                            h = self.token_grid_size,
                                            w = self.token_grid_size)
        output = einops.rearrange(self.q_downsample(hidden_states_2d),
                                        'b d h w -> b (h w) d')
        return output

class TokenAttnPrune(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            ) # might need to figure out how to do this more efficiently since the vision encoder might change...
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)

        self.embed_dim = tower_config.hidden_size
        self.scale = self.embed_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        query_states = self.q_proj(hidden_states) # B x N x D
        key_states = self.k_proj(hidden_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale # B x N x N
        attn_weights = nn.functional.softmax(attn_weights, dim=-1) # B x N x N

        summed_attn_weights = torch.sum(attn_weights, dim=-2) # B x N

        _, idx = torch.topk(summed_attn_weights, self.combined_token_count, dim=-1, sorted=False)
        final_idx, _ = torch.sort(idx)

        select_hidden_states = torch.take_along_dim(hidden_states, final_idx[..., None], dim=1)

        return select_hidden_states
    
class TokenConvAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            ) # might need to figure out how to do this more efficiently since the vision encoder might change...
        
        self.kernel_size = getattr(training_config, 'mm_vision_token_compression_kernel_size', 4)
        self.stride = getattr(training_config, 'mm_vision_token_compression_stride', 4)

        # confirm we can convert the visual tokens to square grid
        assert (self.visual_token_count ** 0.5) % 1 == 0
        self.token_grid_size = int(self.visual_token_count ** 0.5)

        self.embed_dim = tower_config.hidden_size
        self.num_heads = tower_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.q_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.q_proj.append(nn.GELU())
        self.q_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.q_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.q_proj = nn.Sequential(*self.q_proj)

        self.k_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)

        # ADJUST THIS LATER
        self.q_downsample = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                            kernel_size=self.kernel_size, stride=self.stride,
                                            groups=self.embed_dim)

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        query_states_2d = einops.rearrange(self.q_proj(hidden_states), 'b (h w) d -> b d h w',
                                            h = self.token_grid_size,
                                            w = self.token_grid_size)
        query_states = einops.rearrange(self.q_downsample(query_states_2d),
                                        'b d h w -> b (h w) d')

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)        
        return attn_output

class TokenConvAttentionDeep(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            ) # might need to figure out how to do this more efficiently since the vision encoder might change...
        
        self.kernel_size = getattr(training_config, 'mm_vision_token_compression_kernel_size', 4)
        self.stride = getattr(training_config, 'mm_vision_token_compression_stride', 4)

        # confirm we can convert the visual tokens to square grid
        assert (self.visual_token_count ** 0.5) % 1 == 0
        self.token_grid_size = int(self.visual_token_count ** 0.5)

        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # was tower_config.num_attention_heads but Token Packer uses 1024//128
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.q_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.q_proj.append(nn.GELU())
        self.q_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.q_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.q_proj = nn.Sequential(*self.q_proj)

        self.k_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)

        # ADJUST THIS LATER
        self.q_downsample = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                            kernel_size=self.kernel_size, stride=self.stride,
                                            groups=self.embed_dim)

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, attn_mask=None):

        x_multi = x[1] # mulit-level
        x = x[0] # original single-level

        query_states_2d = einops.rearrange(self.q_proj(x), 'b (h w) d -> b d h w',
                                            h = self.token_grid_size,
                                            w = self.token_grid_size)
        query_states = einops.rearrange(self.q_downsample(query_states_2d),
                                        'b d h w -> b (h w) d')

        key_states = self.k_proj(x_multi)
        value_states = self.v_proj(x_multi)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)        
        return attn_output

class TokenLocalConvAttentionDeep(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            ) # might need to figure out how to do this more efficiently since the vision encoder might change...
        
        self.kernel_size = getattr(training_config, 'mm_vision_token_compression_kernel_size', 4)
        self.stride = getattr(training_config, 'mm_vision_token_compression_stride', 4)

        # confirm we can convert the visual tokens to square grid
        assert (self.visual_token_count ** 0.5) % 1 == 0
        self.token_grid_size = int(self.visual_token_count ** 0.5)

        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # was tower_config.num_attention_heads but Token Packer uses 1024//128
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.q_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.q_proj.append(nn.GELU())
        self.q_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.q_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.q_proj = nn.Sequential(*self.q_proj)

        self.k_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)

        # ADJUST THIS LATER
        self.q_downsample = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                            kernel_size=self.kernel_size, stride=self.stride,
                                            groups=self.embed_dim)

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, attn_mask=None):
        x_multi = x[1] # mulit-level
        x = x[0] # original single-level

        query_states_2d = einops.rearrange(self.q_proj(x), 'b (h w) d -> b d h w',
                                            h = self.token_grid_size,
                                            w = self.token_grid_size)
        downsampled_q = self.q_downsample(query_states_2d) 
        b, _, h, w = downsampled_q.size()

        # makes it so each grid counts as a separate batch
        query_states = einops.rearrange(downsampled_q, 'b d h w -> (b h w) 1 d')

        key_states = self.k_proj(x_multi) # b x 576 x d
        value_states = self.v_proj(x_multi)

        # for "chunking" a 2d tensor into a 2d grid (a c) (b d) -> (a b) c d gives use a*b cxd tensors
        # e.g., setting a,b=2, allows use to segment the tensor into 4 quadrants
        k = self.token_grid_size // h
        l = self.token_grid_size // w
        # b i j = b h w
        key_states = einops.rearrange(key_states, "b (i k j l) d -> (b i j) (k l) d",
                         i=h, j=w, k=k, l=l)
        value_states = einops.rearrange(value_states, "b (i k j l) d -> (b i j) (k l) d",
                         i=h, j=w, k=k, l=l)
        
        # attention is now from each convolution "grid" to the respective tokens
        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        output = einops.rearrange(attn_output, "(b t) 1 d -> b t d", b=b)

        # attn_output = self.out_proj(attn_output)        
        return output