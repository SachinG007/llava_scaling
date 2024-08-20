from typing import List, Optional, Tuple, Union

import torch
import einops
import torch.nn as nn

class QueryTokenAttentionV2(nn.Module):
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
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
        self.embed_dim = tower_config.hidden_size
        self.num_heads = tower_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
               
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
        
        #technically it should be q_upsample, but keeping the name same for maintaing similarity with standard token compressor
        self.q_downsample = [nn.Linear(1, self.combined_token_count)] 
        self.q_downsample.append(nn.GELU())
        self.q_downsample.append(nn.Linear(self.combined_token_count, self.combined_token_count))
        self.q_downsample.append(nn.LayerNorm(self.combined_token_count, eps=1e-6))
        self.q_downsample = nn.Sequential(*self.q_downsample)

        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim
        self.text_upsample = nn.Linear(1, self.combined_token_count)

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        text_features = hidden_states[1]
        hidden_states = hidden_states[0]

        text_features = self.text_projection(text_features)
        query_states = self.q_downsample(text_features.transpose(1, 2)).transpose(1, 2) # B x N x D
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output

class QueryTokenAttention(nn.Module):
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
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
        
        self.embed_dim = tower_config.hidden_size
        self.num_heads = tower_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = tower_config.attention_dropout


        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim
        self.text_upsample = nn.Linear(1, self.combined_token_count)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        text_feature = hidden_states[1] #B*1*D
        hidden_states = hidden_states[0]
        bsz, token_len, embed_dim = hidden_states.size()
        tgt_len = self.combined_token_count

        text_feature = self.text_projection(text_feature)
        query_states = self.text_upsample(text_feature.transpose(1, 2)).transpose(1, 2) # B x N x D
        query_states = self.transpose_for_scores(query_states)#B,n_head,tgt_len,head_dim
        key_states = self.transpose_for_scores(self.k_proj(hidden_states))#B,n_head,token_len,head_dim
        value_states = self.transpose_for_scores(self.v_proj(hidden_states))

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        if attn_weights.size() != (bsz, self.num_heads, tgt_len, token_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, tgt_len, token_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2) #.contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        # Will most likely always be this because of the sequential nature of the mm_projector
        if not output_attentions:
            return attn_output
        
        return attn_output, attn_weights

class QueryTokenAttentionDeep(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        
        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # Token packre is 1024//128
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
        
        self.q_downsample = [nn.Linear(1, self.combined_token_count)]
        self.q_downsample.append(nn.GELU())
        self.q_downsample.append(nn.Linear(self.combined_token_count, self.combined_token_count))
        self.q_downsample.append(nn.LayerNorm(self.combined_token_count, eps=1e-6))
        self.q_downsample = nn.Sequential(*self.q_downsample)

        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        text_features = hidden_states[1]
        hidden_states = hidden_states[0]

        text_features = self.text_projection(text_features)
        query_states = self.q_downsample(text_features.transpose(1, 2)).transpose(1, 2) # B x N x D
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output

class QueryTokenAttentionDeepLessParams(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        
        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # Token packre is 1024//128
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )  
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.k_proj = nn.Linear(self.hidden_size, self.embed_dim)
        # self.k_proj.append(nn.GELU())
        # self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        # self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        # self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = nn.Linear(self.hidden_size, self.embed_dim)
        # self.v_proj.append(nn.GELU())
        # self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        # self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        # self.v_proj = nn.Sequential(*self.v_proj)
        
        self.q_downsample = nn.Linear(1, self.combined_token_count)
        # self.q_downsample.append(nn.GELU())
        # self.q_downsample.append(nn.Linear(self.combined_token_count, self.combined_token_count))
        # self.q_downsample.append(nn.LayerNorm(self.combined_token_count, eps=1e-6))
        # self.q_downsample = nn.Sequential(*self.q_downsample)

        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        text_features = hidden_states[1]
        hidden_states = hidden_states[0]

        text_features = self.text_projection(text_features)
        query_states = self.q_downsample(text_features.transpose(1, 2)).transpose(1, 2) # B x N x D
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output
    

class HalfQueryTokenAttentionDeep(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        
        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # Token packre is 1024//128
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
        
        self.combined_token_count_half = self.combined_token_count//2

        self.q_downsample_text = [nn.Linear(1, self.combined_token_count_half)]
        self.q_downsample_text.append(nn.GELU())
        self.q_downsample_text.append(nn.Linear(self.combined_token_count_half, self.combined_token_count_half))
        self.q_downsample_text.append(nn.LayerNorm(self.combined_token_count_half, eps=1e-6))
        self.q_downsample_text = nn.Sequential(*self.q_downsample_text)

        self.q_downsample_image = [nn.Linear(self.visual_token_count, self.combined_token_count_half)]
        self.q_downsample_image.append(nn.GELU())
        self.q_downsample_image.append(nn.Linear(self.combined_token_count_half, self.combined_token_count_half))
        self.q_downsample_image.append(nn.LayerNorm(self.combined_token_count_half, eps=1e-6))
        self.q_downsample_image = nn.Sequential(*self.q_downsample_image)

        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        x = hidden_states[2] #orignal single level image features
        text_features = hidden_states[1]
        hidden_states = hidden_states[0] #multi level image features
        

        text_features = self.text_projection(text_features)
        query_states_text = self.q_downsample_text(text_features.transpose(1, 2)).transpose(1, 2) # B x N x D

        query_states_image = self.q_downsample_image(x.transpose(1, 2)).transpose(1, 2) # B x N x D
        query_states = torch.cat((query_states_text, query_states_image), dim=1)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output
    
class EntityTokenAttentionDeep(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        
        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8 # Token packre is 1024//128
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
        
        self.q_downsample = [nn.Linear(self.combined_token_count, self.combined_token_count)]
        self.q_downsample.append(nn.GELU())
        self.q_downsample.append(nn.Linear(self.combined_token_count, self.combined_token_count))
        self.q_downsample.append(nn.LayerNorm(self.combined_token_count, eps=1e-6))
        self.q_downsample = nn.Sequential(*self.q_downsample)

        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        text_features = hidden_states[1]
        hidden_states = hidden_states[0]

        text_features = self.text_projection(text_features)
        query_states = self.q_downsample(text_features.transpose(1, 2)).transpose(1, 2) # B x N x D
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # attn_output = self.out_proj(attn_output)
        return attn_output

class QueryLocalConvAttentionDeep(nn.Module):
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
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 768)
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
        
        self.text_projection = nn.Linear(self.text_embedding_dim, self.embed_dim) #ADD ARGUMENT FOR 768 i.e. text embedding dim

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        x = hidden_states[2] #orignal single level image features
        text_features = hidden_states[1]
        x_multi = hidden_states[0] #multi level image features

        text_features = self.text_projection(text_features)
        text_features = text_features.repeat(1, self.visual_token_count, 1)

        #add the query feature to the self
        x = x + text_features

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