"""Evaluation metrics for Latent Flow Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_nmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute Normalized Mean Squared Error.
    
    NMSE = E[||pred - target||^2] / E[||target||^2]
    
    Args:
        pred: (B, S, D) predicted latents
        target: (B, S, D) target latents
        eps: Small constant for numerical stability
        
    Returns:
        NMSE value
    """
    mse = F.mse_loss(pred, target)
    target_norm_sq = (target ** 2).mean()
    nmse = mse / (target_norm_sq + eps)
    
    return nmse.item()


def compute_kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Compute KL divergence between two distributions.
    
    KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
    
    Args:
        logits_p: (B, S, V) logits for distribution P (teacher)
        logits_q: (B, S, V) logits for distribution Q (LFT)
        temperature: Temperature for softmax
        
    Returns:
        KL divergence value
    """
    # Apply temperature
    logits_p = logits_p / temperature
    logits_q = logits_q / temperature
    
    # Compute log probabilities
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    
    # Compute probabilities for P
    p = F.softmax(logits_p, dim=-1)
    
    # KL divergence: sum_i p(i) * (log_p(i) - log_q(i))
    kl = (p * (log_p - log_q)).sum(dim=-1).mean()
    
    return kl.item()


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute perplexity given logits and labels.
    
    PPL = exp(cross_entropy_loss)
    
    Args:
        logits: (B, S, V) logits
        labels: (B, S) token labels
        attention_mask: (B, S) mask for padding
        
    Returns:
        Perplexity value
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
    else:
        shift_mask = None
    
    # Compute cross-entropy
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Apply mask if provided
    if shift_mask is not None:
        losses = losses.view(shift_labels.shape)
        losses = (losses * shift_mask).sum() / shift_mask.sum()
    else:
        losses = losses.mean()
    
    # Compute perplexity
    ppl = torch.exp(losses).item()
    
    return ppl


@torch.no_grad()
def evaluate_lft(
    lft_model,
    teacher_model,
    eval_dataloader,
    num_steps: int = 1,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Comprehensive evaluation of LFT model.
    
    Args:
        lft_model: LFT model in full mode
        teacher_model: Teacher model for comparison
        eval_dataloader: DataLoader with evaluation data
        num_steps: Number of flow steps for inference
        device: Device for computation
        
    Returns:
        Dict with evaluation metrics
    """
    lft_model.eval()
    teacher_model.eval()
    
    lft_model.set_mode("full")
    
    total_nmse = 0.0
    total_kl_latent = 0.0
    total_kl_lm = 0.0
    total_ppl_teacher = 0.0
    total_ppl_lft = 0.0
    num_batches = 0
    
    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Teacher forward
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        teacher_logits = teacher_outputs.logits
        teacher_hidden_states = teacher_outputs.hidden_states
        
        # Get target latent (output of layer n)
        x1_target = teacher_hidden_states[lft_model.end_layer + 1]
        
        # LFT forward
        lft_outputs = lft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_steps=num_steps,
            return_hidden_states=True,
        )
        lft_logits = lft_outputs["logits"]
        x1_pred = lft_outputs["x1_hat"]
        
        # Compute NMSE on latents
        nmse = compute_nmse(x1_pred, x1_target)
        total_nmse += nmse
        
        # Compute KL divergence on latents (if we convert to logits)
        # For latents, we'll skip this or use a simplified version
        
        # Compute KL divergence on LM logits
        kl_lm = compute_kl_divergence(teacher_logits, lft_logits)
        total_kl_lm += kl_lm
        
        # Compute perplexity
        ppl_teacher = compute_perplexity(teacher_logits, input_ids, attention_mask)
        ppl_lft = compute_perplexity(lft_logits, input_ids, attention_mask)
        total_ppl_teacher += ppl_teacher
        total_ppl_lft += ppl_lft
        
        num_batches += 1
    
    # Average metrics
    metrics = {
        "nmse": total_nmse / num_batches,
        "kl_lm": total_kl_lm / num_batches,
        "ppl_teacher": total_ppl_teacher / num_batches,
        "ppl_lft": total_ppl_lft / num_batches,
    }
    
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return metrics
