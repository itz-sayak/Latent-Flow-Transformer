"""Evaluation script for Latent Flow Transformer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, InferenceConfig, EvaluationConfig
from models import VelocityEstimator, FlowLayer, LatentFlowTransformer
from inference import LFTInference
from evaluation import evaluate_lft, compute_nmse, compute_kl_divergence, compute_perplexity
from training import LatentPairDataset, TokenizedDataset
from utils import setup_logging, load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Latent Flow Transformer")
    
    # Model config
    parser.add_argument("--teacher-model", type=str, default="EleutherAI/pythia-410m",
                        help="Teacher model name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LFT checkpoint")
    parser.add_argument("--start-layer", type=int, default=6,
                        help="Start layer")
    parser.add_argument("--end-layer", type=int, default=18,
                        help="End layer")
    
    # Inference config
    parser.add_argument("--num-steps", type=int, default=3,
                        help="Number of discrete flow steps (k)")
    parser.add_argument("--mode", type=str, default="full", choices=["standalone", "full"],
                        help="Evaluation mode")
    
    # Data
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Cached latent pairs file")
    parser.add_argument("--num-eval-samples", type=int, default=100,
                        help="Number of evaluation samples")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    
    # Output
    parser.add_argument("--output-file", type=str, default="evaluation_results.txt",
                        help="Output file for results")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("Evaluating Latent Flow Transformer")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")
    
    # Load teacher model
    logger.info(f"Loading teacher model: {args.teacher_model}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        output_hidden_states=True,
    ).to(args.device)
    teacher_model.eval()
    
    # Get model config from teacher
    teacher_config = teacher_model.config
    
    model_config = ModelConfig(
        teacher_name=args.teacher_model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        hidden_dim=teacher_config.hidden_size,
        num_attention_heads=teacher_config.num_attention_heads,
        intermediate_dim=teacher_config.intermediate_size,
        device=args.device,
    )
    
    # Create flow layer
    logger.info("Creating flow layer...")
    velocity_estimator = VelocityEstimator(
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_attention_heads,
        intermediate_dim=model_config.intermediate_dim,
        num_layers=model_config.num_velocity_layers,
        dropout=model_config.dropout,
        use_causal_mask=model_config.use_causal_mask,
    )
    
    flow_layer = FlowLayer(velocity_estimator, use_midpoint=True)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, flow_layer, device=args.device)
    
    # Create LFT model
    logger.info("Creating LFT model...")
    lft_model = LatentFlowTransformer(
        teacher_model_name=args.teacher_model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        flow_layer=flow_layer,
        mode=args.mode,
        freeze_teacher=True,
    )
    lft_model.to(args.device)
    lft_model.eval()
    
    # Create inference wrapper
    inference_config = InferenceConfig(
        mode=args.mode,
        num_steps=args.num_steps,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    lft_inference = LFTInference(lft_model, inference_config)
    
    # Load evaluation data
    if args.mode == "standalone":
        # Use cached latent pairs
        assert args.cache_file is not None, "Need cache file for standalone mode"
        logger.info(f"Loading cached latents: {args.cache_file}")
        eval_dataset = LatentPairDataset(args.cache_file, device=args.device)
    else:
        # Use tokenized text
        logger.info("Creating dummy eval dataset...")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dummy_texts = [
            "The Latent Flow Transformer is a novel architecture for efficient language modeling." * 5
            for _ in range(args.num_eval_samples)
        ]
        eval_dataset = TokenizedDataset(dummy_texts, tokenizer, max_length=512)
    
    # Create dataloader
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    
    if args.mode == "standalone":
        # Standalone evaluation: x0 -> x1_hat
        total_nmse = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                x0 = batch["x0"].to(args.device)
                x1 = batch["x1"].to(args.device)
                
                x1_hat = lft_inference.predict_latent(x0, num_steps=args.num_steps)
                
                # Compute NMSE
                nmse = compute_nmse(x1_hat, x1)
                mse = torch.nn.functional.mse_loss(x1_hat, x1).item()
                
                total_nmse += nmse
                total_mse += mse
                num_batches += 1
        
        avg_nmse = total_nmse / num_batches
        avg_mse = total_mse / num_batches
        
        logger.info(f"Standalone Evaluation Results:")
        logger.info(f"  NMSE: {avg_nmse:.6f}")
        logger.info(f"  MSE: {avg_mse:.6f}")
        
        results_text = f"Standalone Evaluation (k={args.num_steps})\n"
        results_text += f"NMSE: {avg_nmse:.6f}\n"
        results_text += f"MSE: {avg_mse:.6f}\n"
        
    else:
        # Full end-to-end evaluation
        metrics = evaluate_lft(
            lft_model=lft_model,
            teacher_model=teacher_model,
            eval_dataloader=eval_loader,
            num_steps=args.num_steps,
            device=args.device,
        )
        
        logger.info(f"Full Evaluation Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.6f}")
        
        results_text = f"Full End-to-End Evaluation (k={args.num_steps})\n"
        for k, v in metrics.items():
            results_text += f"{k}: {v:.6f}\n"
    
    # Save results
    with open(args.output_file, 'w') as f:
        f.write(results_text)
    
    logger.info(f"Results saved to {args.output_file}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
