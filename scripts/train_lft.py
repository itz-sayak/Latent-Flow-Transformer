"""Training script for Latent Flow Transformer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
import logging

from config import ModelConfig, TrainingConfig
from training import train_lft, TeacherInterface, LatentPairDataset, TokenizedDataset
from models import VelocityEstimator, FlowLayer
from utils import setup_logging, ensure_dir

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Flow Transformer")
    
    # Model config
    parser.add_argument("--teacher-model", type=str, default="EleutherAI/pythia-410m",
                        help="Teacher model name")
    parser.add_argument("--start-layer", type=int, default=6,
                        help="Start layer for compression (m)")
    parser.add_argument("--end-layer", type=int, default=18,
                        help="End layer for compression (n)")
    
    # Training config
    parser.add_argument("--method", type=str, default="sfm", choices=["sfm", "fw", "hybrid"],
                        help="Training method: sfm, fw, or hybrid (fw + sfm regularization)")
    parser.add_argument("--sfm-regularization", type=float, default=0.1,
                        help="SFM regularization weight for hybrid mode (alpha in Eq. 9)")
    parser.add_argument("--cache-dir", type=str, default="./cache/latent_pairs",
                        help="Directory for cached latent pairs")
    parser.add_argument("--output-dir", type=str, default="./outputs/lft",
                        help="Output directory for checkpoints")
    
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps")
    
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Logging interval")
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=5000,
                        help="Checkpoint save interval")
    
    # Data
    parser.add_argument("--max-train-samples", type=int, default=10000,
                        help="Maximum training samples")
    parser.add_argument("--max-eval-samples", type=int, default=1000,
                        help="Maximum evaluation samples")
    parser.add_argument("--use-cached", action="store_true",
                        help="Use cached latent pairs if available")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Resume
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    ensure_dir(args.output_dir)
    log_file = os.path.join(args.output_dir, "train.log")
    setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("Training Latent Flow Transformer")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configs
    model_config = ModelConfig(
        teacher_name=args.teacher_model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        device=args.device,
    )
    
    train_config = TrainingConfig(
        method=args.method,
        sfm_regularization=args.sfm_regularization if args.method == "hybrid" else 0.0,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Check if cached latents exist
    cache_file = os.path.join(
        args.cache_dir,
        f"latents_l{args.start_layer}_to_l{args.end_layer}.pt"
    )
    
    if not os.path.exists(cache_file) or not args.use_cached:
        logger.info("Cached latents not found. Creating from scratch...")
        logger.info("NOTE: This is a demo. In practice, you should cache latents from a real dataset.")
        
        # For demo purposes, create dummy dataset
        # In practice, use real data from The Pile or similar
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dummy text data
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog." * 10
            for _ in range(args.max_train_samples + args.max_eval_samples)
        ]
        
        dummy_dataset = TokenizedDataset(dummy_texts, tokenizer, max_length=512)
        
        # Create teacher interface
        teacher = TeacherInterface(
            model_name=args.teacher_model,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            device=args.device,
        )
        
        # Cache latents
        logger.info("Caching latent pairs...")
        teacher.cache_dataset_latents(
            dataset=dummy_dataset,
            cache_dir=args.cache_dir,
            max_samples=args.max_train_samples + args.max_eval_samples,
            batch_size=args.batch_size,
        )
    
    # Load cached dataset
    logger.info(f"Loading cached latents from {cache_file}")
    full_dataset = LatentPairDataset(cache_file, device=args.device)
    
    # Split into train/val
    train_size = min(args.max_train_samples, len(full_dataset) - args.max_eval_samples)
    val_size = args.max_eval_samples
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Train
    logger.info("Starting training...")
    trained_flow_layer = train_lft(
        model_config=model_config,
        train_config=train_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume_from=args.resume_from,
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
