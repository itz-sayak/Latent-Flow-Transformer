"""Utility functions for Latent Flow Transformer."""

import torch
import torch.nn as nn
import math
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def get_sinusoidal_embeddings(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Generate sinusoidal time embeddings.
    
    Args:
        timesteps: (B,) or (B, 1) tensor of time values in [0, 1]
        embedding_dim: Dimension of the embedding
        
    Returns:
        (B, embedding_dim) tensor of sinusoidal embeddings
    """
    half_dim = embedding_dim // 2
    
    # Create frequency scale
    emb_scale = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
    
    # Ensure timesteps is 2D
    if timesteps.dim() == 1:
        timesteps = timesteps.unsqueeze(1)
    elif timesteps.dim() == 2 and timesteps.size(1) == 1:
        pass
    else:
        raise ValueError(f"Unexpected timesteps shape: {timesteps.shape}")
    
    # Compute sinusoidal embeddings
    args = timesteps * frequencies.unsqueeze(0)
    embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Handle odd embedding_dim
    if embedding_dim % 2 == 1:
        embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
    
    return embeddings


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding module: sinusoidal encoding -> MLP.
    Used for time conditioning in velocity estimator.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
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
        t_emb = get_sinusoidal_embeddings(t, self.embedding_dim)
        
        # Pass through MLP
        t_emb = self.mlp(t_emb)
        
        return t_emb


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    prefix: str = "checkpoint"
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, f"{prefix}_step_{step}.pt")
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if config is not None:
        checkpoint["config"] = config
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save config as JSON
    if config is not None:
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    step = checkpoint.get("step", 0)
    
    return step


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_causal_mask(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device for tensor
        
    Returns:
        (seq_len, seq_len) boolean mask where True means masked (not allowed)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between x0 and x1.
    
    Args:
        x0: (B, S, D) starting latent
        x1: (B, S, D) target latent
        t: (B,) or (B, 1, 1) time values in [0, 1]
        
    Returns:
        (B, S, D) interpolated latent
    """
    if t.dim() == 1:
        t = t.view(-1, 1, 1)
    
    return (1 - t) * x0 + t * x1


def get_velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Compute target velocity for flow matching.
    
    Args:
        x0: (B, S, D) starting latent
        x1: (B, S, D) target latent
        
    Returns:
        (B, S, D) velocity target
    """
    return x1 - x0
