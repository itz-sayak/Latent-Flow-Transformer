"""Time embedding module for conditioning on timesteps."""

import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """
    Time embedding module that converts scalar timesteps to embeddings.
    Uses sinusoidal encoding followed by MLP.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Args:
            embedding_dim: Dimension of sinusoidal embedding
            hidden_dim: Output dimension (typically same as model hidden_dim)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # MLP to project sinusoidal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps in [0, 1]
            
        Returns:
            (B, hidden_dim) time embeddings
        """
        # Get sinusoidal embeddings
        t_emb = self._sinusoidal_embedding(t)
        
        # Pass through MLP
        t_emb = self.mlp(t_emb)
        
        return t_emb
    
    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal time embeddings.
        
        Args:
            timesteps: (B,) tensor of time values in [0, 1]
            
        Returns:
            (B, embedding_dim) tensor of sinusoidal embeddings
        """
        half_dim = self.embedding_dim // 2
        
        # Create frequency scale (same as in standard diffusion models)
        emb_scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
        
        # Ensure timesteps is 2D
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
        
        # Compute sinusoidal embeddings
        args = timesteps * frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Handle odd embedding_dim
        if self.embedding_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
        return embeddings
