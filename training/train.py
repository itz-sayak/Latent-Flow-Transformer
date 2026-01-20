"""Main training loop for Latent Flow Transformer."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Literal

from config import ModelConfig, TrainingConfig
from models import VelocityEstimator, FlowLayer, LatentFlowTransformer
from utils import save_checkpoint, load_checkpoint, count_parameters, ensure_dir
from training.flow_matching import FlowMatchingTrainer
from training.flow_walking import FlowWalkingTrainer
from training.teacher_interface import LatentPairDataset

logger = logging.getLogger(__name__)


def create_flow_layer(config: ModelConfig) -> FlowLayer:
    """Create flow layer from config."""
    velocity_estimator = VelocityEstimator(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_attention_heads,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_velocity_layers,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        layer_norm_eps=config.layer_norm_eps,
        use_causal_mask=config.use_causal_mask,
        time_embed_dim=config.time_embed_dim,
    )
    
    flow_layer = FlowLayer(
        velocity_estimator=velocity_estimator,
        use_midpoint=True,  # Always use midpoint for better accuracy
    )
    
    return flow_layer


def train_lft(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    train_dataset: LatentPairDataset,
    val_dataset: Optional[LatentPairDataset] = None,
    resume_from: Optional[str] = None,
) -> FlowLayer:
    """
    Main training function for LFT.
    
    Args:
        model_config: Model configuration
        train_config: Training configuration
        train_dataset: Training dataset of (x0, x1) pairs
        val_dataset: Optional validation dataset
        resume_from: Optional checkpoint path to resume from
        
    Returns:
        Trained flow layer
    """
    # Setup output directory
    ensure_dir(train_config.output_dir)
    
    # Create flow layer
    logger.info("Creating flow layer...")
    flow_layer = create_flow_layer(model_config)
    flow_layer = flow_layer.to(model_config.device)
    
    num_params = count_parameters(flow_layer)
    logger.info(f"Flow layer parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        flow_layer.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    
    # Create learning rate scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < train_config.warmup_steps:
            return step / max(1, train_config.warmup_steps)
        progress = (step - train_config.warmup_steps) / max(1, train_config.max_steps - train_config.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Cosine decay to 10%
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create trainer based on method
    if train_config.method == "sfm":
        logger.info("Using Standard Flow Matching (SFM)")
        trainer = FlowMatchingTrainer(
            flow_layer=flow_layer,
            optimizer=optimizer,
            device=model_config.device,
            max_grad_norm=train_config.max_grad_norm,
        )
    elif train_config.method in ["fw", "hybrid"]:
        sfm_reg = train_config.sfm_regularization if train_config.method == "hybrid" else 0.0
        logger.info(f"Using Flow Walking (FW) with SFM regularization alpha={sfm_reg}")
        trainer = FlowWalkingTrainer(
            flow_layer=flow_layer,
            optimizer=optimizer,
            device=model_config.device,
            num_steps=train_config.fw_num_steps,
            max_grad_norm=train_config.max_grad_norm,
            use_gradient_checkpointing=train_config.use_gradient_checkpointing,
            sfm_regularization=sfm_reg,
        )
    else:
        raise ValueError(f"Unknown training method: {train_config.method}")
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_step = load_checkpoint(resume_from, flow_layer, optimizer, model_config.device)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for latent pairs
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=0,
        )
    
    # Training loop
    logger.info("Starting training...")
    flow_layer.train()
    
    step = start_step
    epoch = 0
    
    # Create causal mask once (if needed)
    # Assuming all samples have same sequence length
    sample_x0 = train_dataset[0]["x0"]
    seq_len = sample_x0.size(0)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=model_config.device),
        diagonal=1
    ).bool()
    
    while step < train_config.max_steps:
        epoch += 1
        logger.info(f"Epoch {epoch}")
        
        pbar = tqdm(train_loader, desc=f"Training")
        
        for batch in pbar:
            x0 = batch["x0"]  # (B, S, D)
            x1 = batch["x1"]  # (B, S, D)
            
            # Training step
            metrics = trainer.training_step(x0, x1, causal_mask)
            
            # Update learning rate
            scheduler.step()
            
            step += 1
            current_lr = scheduler.get_last_lr()[0]
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "grad": f"{metrics['grad_norm']:.2f}",
                "lr": f"{current_lr:.2e}",
            })
            
            # Logging
            if step % train_config.log_interval == 0:
                logger.info(
                    f"Step {step}/{train_config.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Grad: {metrics['grad_norm']:.2f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Evaluation
            if val_dataset is not None and step % train_config.eval_interval == 0:
                logger.info("Running validation...")
                flow_layer.eval()
                
                val_metrics_accum = {"loss": 0.0, "mse": 0.0, "nmse": 0.0}
                num_val_batches = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_x0 = val_batch["x0"]
                        val_x1 = val_batch["x1"]
                        
                        val_metrics = trainer.evaluate(val_x0, val_x1, num_steps=3, attention_mask=causal_mask)
                        
                        for k, v in val_metrics.items():
                            val_metrics_accum[k] += v
                        num_val_batches += 1
                
                # Average metrics
                for k in val_metrics_accum:
                    val_metrics_accum[k] /= num_val_batches
                
                logger.info(
                    f"Validation | "
                    f"Loss: {val_metrics_accum['loss']:.4f} | "
                    f"MSE: {val_metrics_accum['mse']:.4f} | "
                    f"NMSE: {val_metrics_accum['nmse']:.4f}"
                )
                
                flow_layer.train()
            
            # Save checkpoint
            if step % train_config.save_interval == 0:
                checkpoint_path = save_checkpoint(
                    model=flow_layer,
                    optimizer=optimizer,
                    step=step,
                    output_dir=train_config.output_dir,
                    config={
                        "model_config": model_config.__dict__,
                        "train_config": train_config.__dict__,
                    },
                    prefix="lft",
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Check if max steps reached
            if step >= train_config.max_steps:
                break
        
        if step >= train_config.max_steps:
            break
    
    # Final save
    final_path = save_checkpoint(
        model=flow_layer,
        optimizer=optimizer,
        step=step,
        output_dir=train_config.output_dir,
        config={
            "model_config": model_config.__dict__,
            "train_config": train_config.__dict__,
        },
        prefix="lft_final",
    )
    logger.info(f"Training complete! Final checkpoint: {final_path}")
    
    return flow_layer
