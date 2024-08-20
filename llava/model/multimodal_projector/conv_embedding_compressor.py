from typing import List, Optional, Tuple, Union

import einops
import torch
import numpy as np
import torch.nn as nn


class ConvEmbeddingCompressor(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()

        self.embed_dim = tower_config.hidden_size
        self.num_heads = 8
        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            )
        self.token_grid_size = np.sqrt(self.visual_token_count).astype(int)
        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

        conv_kernel_size=5
        stride=3
        self.conv_layer = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=conv_kernel_size, stride=stride, padding=0)
        # Calculate the new dimensions after convolution
        self.pooled_height = (self.token_grid_size - conv_kernel_size) // stride + 1
        self.pooled_width = (self.token_grid_size - conv_kernel_size) // stride + 1
        
        intermediate_dim = getattr(training_config, 'mm_conv_token_reduction_intermediate_dim', 256)
        self.combined_token_count = getattr(training_config, 'mm_vision_output_combined_token_count', 1)
        # Projection layer: maps pooled image features to N * embed_dim through an intermediate dimension
        self.query_projection = nn.Sequential(
            nn.Linear(self.pooled_height * self.pooled_width * self.embed_dim, intermediate_dim),  # Reduce to intermediate dimension
            nn.ReLU(),  # Non-linearity
            nn.Linear(intermediate_dim, self.combined_token_count * self.embed_dim)  # Expand to N queries of dimension embed_dim
        )

        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        image_features = hidden_states.transpose(1, 2)
        patch_1d_size  = np.sqrt(image_features.shape[2]).astype(int)
        batch_size = image_features.shape[0]
        hidden_size = image_features.shape[1]
        image_features = image_features.reshape(batch_size, hidden_size, patch_1d_size, patch_1d_size)  # Embed_dim as channels

        # Apply 2D average pooling
        pooled_features = self.conv_layer(image_features)
        #reshape to have (batch_size, embed_dim, -1)
        pooled_features = pooled_features.reshape(batch_size, hidden_size, -1)
        # Flatten pooled features to (batch, pooled_height * pooled_width * embed_dim)
        pooled_features = pooled_features.reshape(batch_size, -1)
        
        # Project pooled features to create queries
        projected_queries = self.query_projection(pooled_features)  # (batch_size, N * embed_dim)
        query = projected_queries.view(batch_size, hidden_size, -1)  # (batch_size, embed_dim, N)
        #tranpose 
        query = query.transpose(1, 2) # (batch_size, N, embed_dim)
        
        # Apply self-attention
        image_features = image_features.reshape(batch_size, hidden_size, -1)  # (batch_size, embed_dim, num_patches)
        #transpose for self attention batch first false
        image_features = image_features.permute(2, 0, 1)  # (num_patches, batch_size, embed_dim)
        
        attn_output, _ = self.self_attention(query.transpose(0, 1), image_features, image_features)

        return attn_output.transpose(0, 1)