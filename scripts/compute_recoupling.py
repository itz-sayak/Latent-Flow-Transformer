"""Script to compute recoupling ratio for layer selection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer

from config import ModelConfig, RecouplingConfig
from training import TeacherInterface, TokenizedDataset
from evaluation import compute_recoupling_ratio, analyze_layer_recoupling
from utils import setup_logging, ensure_dir

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute recoupling ratio for layer selection")
    
    # Model config
    parser.add_argument("--teacher-model", type=str, default="EleutherAI/pythia-410m",
                        help="Teacher model name")
    
    # Layer range
    parser.add_argument("--start-layer-min", type=int, default=0,
                        help="Minimum start layer")
    parser.add_argument("--start-layer-max", type=int, default=23,
                        help="Maximum start layer")
    parser.add_argument("--end-layer-min", type=int, default=0,
                        help="Minimum end layer")
    parser.add_argument("--end-layer-max", type=int, default=23,
                        help="Maximum end layer")
    
    # Data
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of samples for OT computation")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    
    # OT config
    parser.add_argument("--ot-method", type=str, default="sinkhorn",
                        choices=["sinkhorn", "emd"],
                        help="Optimal transport method")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1,
                        help="Sinkhorn regularization")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs/recoupling",
                        help="Output directory")
    parser.add_argument("--plot", action="store_true",
                        help="Generate heatmap plot")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for teacher (use cpu for OT)")
    
    return parser.parse_args()


def plot_recoupling_heatmap(results, output_file):
    """Plot recoupling ratio heatmap."""
    # Extract unique layer indices
    all_start = sorted(set(k[0] for k in results.keys()))
    all_end = sorted(set(k[1] for k in results.keys()))
    
    # Create matrix
    matrix = np.full((len(all_start), len(all_end)), np.nan)
    
    for (start, end), ratio in results.items():
        i = all_start.index(start)
        j = all_end.index(end)
        matrix[i, j] = ratio
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=all_end,
        yticklabels=all_start,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Recoupling Ratio"},
    )
    plt.xlabel("End Layer")
    plt.ylabel("Start Layer")
    plt.title("Recoupling Ratio Heatmap (Lower is Better)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Heatmap saved to {output_file}")


def main():
    args = parse_args()
    
    # Setup logging
    ensure_dir(args.output_dir)
    log_file = os.path.join(args.output_dir, "recoupling.log")
    setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("Computing Recoupling Ratios for Layer Selection")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")
    
    # Create teacher interface
    logger.info(f"Loading teacher model: {args.teacher_model}")
    
    # For recoupling, we'll use a temporary teacher interface
    # We need it to extract latents across all layers
    teacher = TeacherInterface(
        model_name=args.teacher_model,
        start_layer=0,  # Doesn't matter, we'll extract all
        end_layer=1,
        device=args.device,
    )
    
    # Create dummy dataset
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dummy_texts = [
        "The Latent Flow Transformer uses optimal transport to select layers." * 10
        for _ in range(args.num_samples)
    ]
    
    dataset = TokenizedDataset(dummy_texts, tokenizer, max_length=512)
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Analyze recoupling ratios
    logger.info("Analyzing layer recoupling ratios...")
    
    results = analyze_layer_recoupling(
        teacher_interface=teacher,
        dataset=dataset,
        start_layer_range=(args.start_layer_min, args.start_layer_max),
        end_layer_range=(args.end_layer_min, args.end_layer_max),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    
    # Find best layer pairs
    logger.info("\nTop 10 layer pairs with lowest recoupling ratio:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    for (start, end), ratio in sorted_results[:10]:
        num_layers = end - start + 1
        logger.info(f"  Layers {start}->{end} ({num_layers} layers): R = {ratio:.4f}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "recoupling_ratios.txt")
    with open(results_file, 'w') as f:
        f.write("Layer Pair (start->end), Num Layers, Recoupling Ratio\n")
        f.write("=" * 60 + "\n")
        for (start, end), ratio in sorted_results:
            num_layers = end - start + 1
            f.write(f"{start:2d} -> {end:2d}, {num_layers:2d}, {ratio:.6f}\n")
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Generate heatmap if requested
    if args.plot:
        try:
            plot_file = os.path.join(args.output_dir, "recoupling_heatmap.png")
            plot_recoupling_heatmap(results, plot_file)
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")
    
    logger.info("Recoupling ratio computation complete!")


if __name__ == "__main__":
    main()
