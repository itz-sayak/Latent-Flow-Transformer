"""Velocity estimator: DiT-style transformer block with time conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .time_embedding import TimeEmbedding


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep.
    Applies scale (gamma) and shift (beta) based on time embedding.
    
    As shown in Figure 1 of the paper, each AdaLN produces:
    - gamma (scale for normalized input)
    - beta (shift for normalized input)
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        
        # Linear layer to predict scale (gamma) and shift (beta) from time embedding
        self.ada_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        
        # Initialize to identity (scale=1, shift=0)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input hidden states
            t_emb: (B, D) time embeddings
            
        Returns:
            (B, S, D) normalized and modulated hidden states
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Get scale (gamma) and shift (beta) from time embedding
        ada_params = self.ada_proj(t_emb)  # (B, 2*D)
        gamma, beta = ada_params.chunk(2, dim=-1)  # Each (B, D)
        
        # Apply adaptive modulation: gamma * norm(x) + beta
        # Expand to match x_norm: (B, D) -> (B, 1, D)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return x_norm * (1 + gamma) + beta


class TransformerBlock(nn.Module):
    """
    Single transformer block with DiT-style AdaLN conditioning.
    
    Supports two architecture variants from Figure 1 of the paper:
    
    Sequential (Figure 1a/b - Standard DiT):
        x -> AdaLN -> Attention -> Scale(α₁) -> + -> AdaLN -> FFN -> Scale(α₂) -> +
             (γ₁,β₁)                            |    (γ₂,β₂)                       |
                                                x                                  x
    
    Parallel (Figure 1c - Pythia-style):
        x -> AdaLN -> Attention -> Scale(α₁) ----+
             (γ₁,β₁)                              |
        x -> AdaLN -> FFN -> Scale(α₂) ----------+---> x + attn + ffn
             (γ₂,β₂)                              |
    
    The time conditioning MLP produces 6 parameters per block:
    (γ₁, β₁, α₁) for attention and (γ₂, β₂, α₂) for FFN
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_causal_mask: bool = True,
        architecture_variant: str = "parallel",  # "sequential" or "parallel"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        self.architecture_variant = architecture_variant
        
        # Layer Norms (without elementwise affine, since AdaLN handles it)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, elementwise_affine=False)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        # MLP (Feed Forward)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Time conditioning MLP produces 6 parameters:
        # (γ₁, β₁, α₁) for attention and (γ₂, β₂, α₂) for FFN
        self.ada_proj = nn.Linear(hidden_dim, 6 * hidden_dim)
        
        # Initialize to identity
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input hidden states
            t_emb: (B, D) time embeddings
            attention_mask: Optional (S, S) causal mask
            
        Returns:
            (B, S, D) output hidden states
        """
        # Get all 6 conditioning parameters from time embedding
        ada_params = self.ada_proj(t_emb)  # (B, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = ada_params.chunk(6, dim=-1)  # Each (B, D)
        
        # Expand for broadcasting: (B, D) -> (B, 1, D)
        gamma1 = gamma1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        
        # Create causal mask if needed
        if self.use_causal_mask and attention_mask is None:
            seq_len = x.size(1)
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()
        
        if self.architecture_variant == "parallel":
            # === Parallel Architecture (Figure 1c - Pythia-style) ===
            # Attention and FFN are computed in parallel from the same input x
            
            # Attention branch
            x_norm1 = self.norm1(x)
            x_mod1 = x_norm1 * (1 + gamma1) + beta1
            attn_output, _ = self.attention(
                x_mod1, x_mod1, x_mod1,
                attn_mask=attention_mask,
                need_weights=False,
            )
            
            # FFN branch (in parallel)
            x_norm2 = self.norm2(x)
            x_mod2 = x_norm2 * (1 + gamma2) + beta2
            mlp_output = self.mlp(x_mod2)
            
            # Combine: x = x + α₁*attn + α₂*ffn
            x = x + alpha1 * attn_output + alpha2 * mlp_output
            
        else:
            # === Sequential Architecture (Figure 1a/b - Standard DiT) ===
            # Attention first, then FFN
            
            # Attention branch
            x_norm = self.norm1(x)
            x_modulated = x_norm * (1 + gamma1) + beta1
            attn_output, _ = self.attention(
                x_modulated, x_modulated, x_modulated,
                attn_mask=attention_mask,
                need_weights=False,
            )
            x = x + alpha1 * attn_output
            
            # FFN branch (sequential, uses updated x)
            x_norm = self.norm2(x)
            x_modulated = x_norm * (1 + gamma2) + beta2
            mlp_output = self.mlp(x_modulated)
            x = x + alpha2 * mlp_output
        return x


class VelocityEstimator(nn.Module):
    """
    Velocity field estimator u_θ(x_t, t).
    
    Architecture:
        1. Time embedding: t -> t_emb
        2. Transformer block: (x_t, t_emb) -> y
        3. Velocity: v = y - x_t
    
    This is the core learnable component of the LFT.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_causal_mask: bool = True,
        time_embed_dim: Optional[int] = None,
        architecture_variant: str = "parallel",  # "sequential" or "parallel"
    ):
        """
        Args:
            hidden_dim: Model hidden dimension (D)
            num_heads: Number of attention heads
            intermediate_dim: MLP intermediate dimension
            num_layers: Number of transformer blocks (default=1)
            dropout: Dropout probability
            attention_dropout: Attention dropout
            layer_norm_eps: Layer norm epsilon
            use_causal_mask: Whether to use causal masking
            time_embed_dim: Time embedding dimension (default=hidden_dim)
            architecture_variant: "sequential" (DiT) or "parallel" (Pythia-style)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.architecture_variant = architecture_variant
        
        # Time embedding
        if time_embed_dim is None:
            time_embed_dim = hidden_dim
        
        self.time_embed = TimeEmbedding(
            embedding_dim=time_embed_dim,
            hidden_dim=hidden_dim,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                use_causal_mask=use_causal_mask,
                architecture_variant=architecture_variant,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Estimate velocity field at (x_t, t).
        
        Args:
            x_t: (B, S, D) latent state at time t
            t: (B,) timesteps in [0, 1]
            attention_mask: Optional (S, S) attention mask
            
        Returns:
            (B, S, D) predicted velocity v_t
        """
        # Get time embeddings
        t_emb = self.time_embed(t)  # (B, D)
        
        # Apply transformer blocks
        y = x_t
        for block in self.blocks:
            y = block(y, t_emb, attention_mask)
        
        # Compute velocity as difference: v = y - x
        v_t = y - x_t
        
        return v_t
    
    def get_output_before_residual(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get output y before computing residual (for debugging/analysis).
        
        Returns:
            (B, S, D) output y (before v = y - x)
        """
        t_emb = self.time_embed(t)
        
        y = x_t
        for block in self.blocks:
            y = block(y, t_emb, attention_mask)
        
        return y
