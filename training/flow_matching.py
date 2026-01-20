"""Standard Flow Matching (SFM) training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from models.flow_layer import FlowLayer

logger = logging.getLogger(__name__)


class FlowMatchingTrainer:
    """
    Trainer for Standard Flow Matching (Algorithm 1 in paper).
    
    Loss: E_t || u_θ(x_t, t) - (x1 - x0) ||^2
    where x_t = (1-t)*x0 + t*x1 (linear interpolation)
    """
    
    def __init__(
        self,
        flow_layer: FlowLayer,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            flow_layer: Flow layer to train
            optimizer: Optimizer for flow layer parameters
            device: Device for computation
            max_grad_norm: Gradient clipping threshold
        """
        self.flow_layer = flow_layer
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute SFM loss for a batch.
        
        Args:
            x0: (B, S, D) source latents
            x1: (B, S, D) target latents
            attention_mask: Optional (S, S) causal mask
            
        Returns:
            dict with 'loss' and additional metrics
        """
        batch_size = x0.size(0)
        
        # Sample random timesteps t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=self.device)
        
        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Target velocity: v_target = x1 - x0
        v_target = x1 - x0
        
        # Predict velocity
        v_pred = self.flow_layer.velocity_estimator(x_t, t, attention_mask)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return {
            "loss": loss,
            "mse": loss.item(),
            "mean_t": t.mean().item(),
        }
    
    def training_step(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Single training step.
        
        Args:
            x0: (B, S, D) source latents
            x1: (B, S, D) target latents
            attention_mask: Optional attention mask
            
        Returns:
            dict with loss and metrics
        """
        self.optimizer.zero_grad()
        
        # Compute loss
        loss_dict = self.compute_loss(x0, x1, attention_mask)
        loss = loss_dict["loss"]
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.flow_layer.parameters(),
            self.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Convert loss to item for logging
        loss_dict["loss"] = loss.item()
        loss_dict["grad_norm"] = grad_norm.item()
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Evaluate flow layer on validation data.
        
        Args:
            x0: (B, S, D) source latents
            x1: (B, S, D) target latents
            num_steps: Number of inference steps
            attention_mask: Optional attention mask
            
        Returns:
            dict with evaluation metrics
        """
        # Compute training loss
        loss_dict = self.compute_loss(x0, x1, attention_mask)
        
        # Inference
        x1_hat = self.flow_layer(x0, num_steps, attention_mask)
        
        # NMSE
        mse = F.mse_loss(x1_hat, x1)
        x1_norm_sq = (x1 ** 2).mean()
        nmse = mse / (x1_norm_sq + 1e-8)
        
        return {
            "loss": loss_dict["loss"].item(),
            "mse": mse.item(),
            "nmse": nmse.item(),
        }
