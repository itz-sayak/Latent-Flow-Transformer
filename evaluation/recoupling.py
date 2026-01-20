"""Recoupling ratio computation using Optimal Transport."""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cost matrix (Euclidean distance).
    
    Args:
        x: (N, D) source points
        y: (M, D) target points
        
    Returns:
        (N, M) cost matrix
    """
    # Compute pairwise squared distances
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (M, 1)
    
    cost = x_norm + y_norm.T - 2 * torch.mm(x, y.T)
    cost = torch.sqrt(torch.clamp(cost, min=0))  # Numerical stability
    
    return cost


def sinkhorn_ot(
    cost: torch.Tensor,
    reg: float = 0.1,
    num_iters: int = 100,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Sinkhorn algorithm for entropic regularized optimal transport.
    
    Args:
        cost: (N, M) cost matrix
        reg: Entropic regularization parameter
        num_iters: Number of Sinkhorn iterations
        eps: Numerical stability constant
        
    Returns:
        (N, M) transport plan matrix
    """
    N, M = cost.shape
    device = cost.device
    
    # Uniform marginals
    a = torch.ones(N, device=device) / N
    b = torch.ones(M, device=device) / M
    
    # Kernel matrix
    K = torch.exp(-cost / reg)
    
    # Initialize
    u = torch.ones(N, device=device)
    v = torch.ones(M, device=device)
    
    # Sinkhorn iterations
    for _ in range(num_iters):
        u = a / (K @ v + eps)
        v = b / (K.T @ u + eps)
    
    # Transport plan
    transport = u.unsqueeze(1) * K * v.unsqueeze(0)
    
    return transport


def hard_assignment_from_ot(transport: torch.Tensor) -> torch.Tensor:
    """
    Convert soft transport plan to hard assignment (matching matrix).
    
    Args:
        transport: (N, M) soft transport plan
        
    Returns:
        (N, M) binary matching matrix
    """
    N, M = transport.shape
    
    # Greedy matching: iteratively assign max transport
    matching = torch.zeros_like(transport)
    transport_copy = transport.clone()
    
    for _ in range(min(N, M)):
        # Find maximum
        max_idx = torch.argmax(transport_copy)
        i = max_idx // M
        j = max_idx % M
        
        # Assign
        matching[i, j] = 1.0
        
        # Zero out row and column
        transport_copy[i, :] = 0
        transport_copy[:, j] = 0
    
    return matching


def compute_recoupling_ratio(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: str = "sinkhorn",
    reg: float = 0.1,
    num_iters: int = 100,
) -> Tuple[float, torch.Tensor]:
    """
    Compute recoupling ratio between paired latents.
    
    R = 1 - Tr(M) / |M|
    
    where M is the OT matching matrix.
    Lower R means better alignment (fewer flow crossings).
    
    Args:
        x0: (N, D) source latents
        x1: (N, D) target latents (paired with x0)
        method: "sinkhorn" or "emd"
        reg: Sinkhorn regularization
        num_iters: Sinkhorn iterations
        
    Returns:
        recoupling_ratio: R value
        matching: (N, N) matching matrix
    """
    N, D = x0.shape
    assert x1.shape[0] == N, "x0 and x1 must have same number of samples"
    
    # Move to CPU for OT computation (more stable)
    device = x0.device
    x0_cpu = x0.cpu()
    x1_cpu = x1.cpu()
    
    # Compute cost matrix
    cost = compute_cost_matrix(x0_cpu, x1_cpu)
    
    if method == "sinkhorn":
        # Soft OT with Sinkhorn
        transport = sinkhorn_ot(cost, reg=reg, num_iters=num_iters)
        
        # Convert to hard matching
        matching = hard_assignment_from_ot(transport)
    else:
        # Could implement EMD here using scipy or POT
        # For now, fall back to Sinkhorn
        logger.warning("EMD not implemented, using Sinkhorn")
        transport = sinkhorn_ot(cost, reg=reg, num_iters=num_iters)
        matching = hard_assignment_from_ot(transport)
    
    # Compute recoupling ratio
    # Tr(M) counts how many original pairings are preserved
    trace = torch.trace(matching)
    order = matching.size(0)
    
    recoupling_ratio = 1.0 - (trace / order).item()
    
    # Move matching back to original device
    matching = matching.to(device)
    
    return recoupling_ratio, matching


def analyze_layer_recoupling(
    teacher_interface,
    dataset,
    start_layer_range: Tuple[int, int],
    end_layer_range: Tuple[int, int],
    num_samples: int = 256,
    batch_size: int = 32,
) -> Dict[Tuple[int, int], float]:
    """
    Analyze recoupling ratios for different layer pairs.
    
    Args:
        teacher_interface: TeacherInterface instance
        dataset: Dataset to sample from
        start_layer_range: (min, max) for start layer
        end_layer_range: (min, max) for end layer
        num_samples: Number of samples to use
        batch_size: Batch size for extraction
        
    Returns:
        Dict mapping (start_layer, end_layer) -> recoupling_ratio
    """
    logger.info(f"Analyzing recoupling ratios for layer pairs...")
    logger.info(f"Start layers: {start_layer_range}")
    logger.info(f"End layers: {end_layer_range}")
    
    # Collect all latents for all layers
    all_latents = {}
    
    # Sample data
    indices = torch.randperm(len(dataset))[:num_samples]
    
    logger.info(f"Extracting latents for {num_samples} samples...")
    
    # We need to extract latents for all layers in range
    for layer_idx in range(start_layer_range[0], end_layer_range[1] + 2):
        latents_list = []
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [dataset[int(idx)] for idx in batch_indices]
            
            input_ids = torch.stack([item["input_ids"] for item in batch])
            
            # Extract latent at this layer
            with torch.no_grad():
                outputs = teacher_interface.model(
                    input_ids=input_ids.to(teacher_interface.device),
                    output_hidden_states=True,
                )
                latent = outputs.hidden_states[layer_idx]  # (B, S, D)
                
                # Flatten batch and sequence dims
                latent = latent.view(-1, latent.size(-1))  # (B*S, D)
                latents_list.append(latent.cpu())
        
        all_latents[layer_idx] = torch.cat(latents_list, dim=0)
    
    # Compute recoupling ratios for all pairs
    results = {}
    
    for start_layer in range(start_layer_range[0], start_layer_range[1] + 1):
        for end_layer in range(end_layer_range[0], end_layer_range[1] + 1):
            if end_layer <= start_layer:
                continue
            
            x0 = all_latents[start_layer]
            x1 = all_latents[end_layer + 1]  # +1 because hidden_states[i] is output of layer i-1
            
            # Sample subset for OT (full OT is expensive)
            ot_samples = min(256, x0.size(0))
            sample_indices = torch.randperm(x0.size(0))[:ot_samples]
            
            x0_sample = x0[sample_indices]
            x1_sample = x1[sample_indices]
            
            # Compute recoupling ratio
            ratio, _ = compute_recoupling_ratio(x0_sample, x1_sample)
            
            results[(start_layer, end_layer)] = ratio
            
            logger.info(f"Layers {start_layer}->{end_layer}: R = {ratio:.4f}")
    
    return results
