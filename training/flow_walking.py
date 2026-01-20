"""Flow Walking (FW) training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from models.flow_layer import FlowLayer

logger = logging.getLogger(__name__)


class FlowWalkingTrainer:
    """
    Trainer for Flow Walking (Algorithm 2 in paper).
    
    Uses k=3 step numerical integration:
    1. x_t1 = step(x0, 0 -> t1)
    2. x_t2 = step(x_t1, t1 -> t2)
    3. x_hat = step(x_t2, t2 -> 1)
    
    Loss: || x_hat - x1 ||^2
    
    Optionally supports hybrid loss (Equation 9):
    L = L_FW + alpha * L_SFM
    """
    
    def __init__(
        self,
        flow_layer: FlowLayer,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        num_steps: int = 3,
        max_grad_norm: float = 1.0,
        use_gradient_checkpointing: bool = False,
        sfm_regularization: float = 0.0,  # alpha for hybrid loss
    ):
        """
        Args:
            flow_layer: Flow layer to train
            optimizer: Optimizer for flow layer parameters
            device: Device for computation
            num_steps: Number of integration steps (k=3 in paper)
            max_grad_norm: Gradient clipping threshold
            use_gradient_checkpointing: Whether to use gradient checkpointing
            sfm_regularization: Alpha for hybrid FW + SFM loss (Equation 9)
        """
        self.flow_layer = flow_layer
        self.optimizer = optimizer
        self.device = device
        self.num_steps = num_steps
        self.max_grad_norm = max_grad_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.sfm_regularization = sfm_regularization
        
        assert num_steps >= 2, "Flow Walking requires at least 2 steps"
    
    def compute_fw_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Flow Walking loss."""
        batch_size = x0.size(0)
        
        # Sample k-1 random intermediate timesteps
        t_intermediate = torch.rand(batch_size, self.num_steps - 1, device=self.device)
        t_intermediate = torch.sort(t_intermediate, dim=1)[0]
        
        # Build full timestep sequence: [0, t1, t2, ..., t_{k-1}, 1]
        t_start = torch.zeros(batch_size, 1, device=self.device)
        t_end = torch.ones(batch_size, 1, device=self.device)
        t_all = torch.cat([t_start, t_intermediate, t_end], dim=1)
        
        # Perform k-step integration
        x = x0
        for i in range(self.num_steps):
            t_curr = t_all[:, i]
            t_next = t_all[:, i + 1]
            
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    self.flow_layer.step,
                    x, t_curr, t_next, attention_mask,
                    use_reentrant=False
                )
            else:
                x = self.flow_layer.step(x, t_curr, t_next, attention_mask)
        
        return F.mse_loss(x, x1)
    
    def compute_sfm_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Standard Flow Matching loss (for regularization)."""
        batch_size = x0.size(0)
        
        t = torch.rand(batch_size, device=self.device)
        t_expanded = t.view(-1, 1, 1)
        
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        v_target = x1 - x0
        v_pred = self.flow_layer.velocity_estimator(x_t, t, attention_mask)
        
        return F.mse_loss(v_pred, v_target)
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute loss for a batch.
        
        If sfm_regularization > 0, uses hybrid loss (Equation 9):
        L = L_FW + alpha * L_SFM
        """
        fw_loss = self.compute_fw_loss(x0, x1, attention_mask)
        
        metrics = {
            "fw_loss": fw_loss.item(),
        }
        
        if self.sfm_regularization > 0:
            sfm_loss = self.compute_sfm_loss(x0, x1, attention_mask)
            loss = fw_loss + self.sfm_regularization * sfm_loss
            metrics["sfm_loss"] = sfm_loss.item()
        else:
            loss = fw_loss
        
        metrics["loss"] = loss
        metrics["mse"] = fw_loss.item()
        
        return metrics
    
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
        num_steps: int = 3,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Evaluate flow layer on validation data.
        
        Args:
            x0: (B, S, D) source latents
            x1: (B, S, D) target latents
            num_steps: Number of inference steps (can differ from training)
            attention_mask: Optional attention mask
            
        Returns:
            dict with evaluation metrics
        """
        # Compute training loss (with training num_steps)
        loss_dict = self.compute_loss(x0, x1, attention_mask)
        
        # Inference with specified num_steps
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
