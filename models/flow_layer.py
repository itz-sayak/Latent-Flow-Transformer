"""Flow layer: wraps velocity estimator with stepping logic."""

import torch
import torch.nn as nn
from typing import Optional

from .velocity_estimator import VelocityEstimator


class FlowLayer(nn.Module):
    """
    Flow layer that performs latent transport using learned velocity field.
    
    Wraps VelocityEstimator and provides:
    - Single-step evolution (Euler/midpoint)
    - Multi-step unrolled inference
    """
    
    def __init__(
        self,
        velocity_estimator: VelocityEstimator,
        use_midpoint: bool = True,
    ):
        """
        Args:
            velocity_estimator: The learned velocity field u_θ
            use_midpoint: Whether to use midpoint integration (more accurate)
        """
        super().__init__()
        self.velocity_estimator = velocity_estimator
        self.use_midpoint = use_midpoint
    
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single integration step from time t to t_next.
        
        Uses either Euler or midpoint method:
        - Euler: x_{t+d} = x_t + d * u_θ(x_t, t)
        - Midpoint: x_{t+d} = x_t + d * u_θ(x_t + d/2 * u_θ(x_t, t), t + d/2)
        
        Args:
            x: (B, S, D) latent state at time t
            t: (B,) current timesteps
            t_next: (B,) next timesteps
            attention_mask: Optional (S, S) attention mask
            
        Returns:
            (B, S, D) latent state at time t_next
        """
        d = t_next - t  # (B,)
        
        if not self.use_midpoint:
            # Euler method
            v_t = self.velocity_estimator(x, t, attention_mask)
            x_next = x + d.view(-1, 1, 1) * v_t
        else:
            # Midpoint method (Equation 5 in paper)
            # First estimate velocity at current position
            v_t = self.velocity_estimator(x, t, attention_mask)
            
            # Compute midpoint
            t_mid = t + d / 2
            x_mid = x + (d / 2).view(-1, 1, 1) * v_t
            
            # Estimate velocity at midpoint
            v_mid = self.velocity_estimator(x_mid, t_mid, attention_mask)
            
            # Final step using midpoint velocity
            x_next = x + d.view(-1, 1, 1) * v_mid
        
        return x_next
    
    def forward(
        self,
        x0: torch.Tensor,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-step unrolled inference from t=0 to t=1.
        
        Args:
            x0: (B, S, D) initial latent state
            num_steps: Number of discrete integration steps (k)
            attention_mask: Optional (S, S) attention mask
            
        Returns:
            (B, S, D) final latent state at t=1
        """
        batch_size = x0.size(0)
        device = x0.device
        
        x = x0
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            t_next = torch.full((batch_size,), (i + 1) * dt, device=device)
            
            x = self.step(x, t, t_next, attention_mask)
        
        return x
    
    def forward_trajectory(
        self,
        x0: torch.Tensor,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Return full trajectory [x(0), x(dt), x(2dt), ..., x(1)].
        
        Useful for visualization and debugging.
        
        Args:
            x0: (B, S, D) initial latent state
            num_steps: Number of discrete integration steps
            attention_mask: Optional attention mask
            
        Returns:
            List of (num_steps + 1) tensors, each (B, S, D)
        """
        batch_size = x0.size(0)
        device = x0.device
        
        trajectory = [x0]
        x = x0
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            t_next = torch.full((batch_size,), (i + 1) * dt, device=device)
            
            x = self.step(x, t, t_next, attention_mask)
            trajectory.append(x)
        
        return trajectory
    
    def get_velocity_at_time(
        self,
        x: torch.Tensor,
        t: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get velocity field at specific time.
        
        Args:
            x: (B, S, D) latent state
            t: Scalar time value in [0, 1]
            attention_mask: Optional attention mask
            
        Returns:
            (B, S, D) velocity
        """
        batch_size = x.size(0)
        device = x.device
        
        t_tensor = torch.full((batch_size,), t, device=device)
        v = self.velocity_estimator(x, t_tensor, attention_mask)
        
        return v
