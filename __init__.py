"""Main package initialization for Latent Flow Transformer."""

from .config import ModelConfig, TrainingConfig, InferenceConfig, RecouplingConfig, EvaluationConfig
from .models import VelocityEstimator, FlowLayer, LatentFlowTransformer, TimeEmbedding
from .training import train_lft, TeacherInterface, LatentPairDataset, TokenizedDataset
from .inference import LFTInference
from .evaluation import (
    compute_nmse,
    compute_kl_divergence,
    compute_perplexity,
    evaluate_lft,
    compute_recoupling_ratio,
    analyze_layer_recoupling,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "RecouplingConfig",
    "EvaluationConfig",
    # Models
    "VelocityEstimator",
    "FlowLayer",
    "LatentFlowTransformer",
    "TimeEmbedding",
    # Training
    "train_lft",
    "TeacherInterface",
    "LatentPairDataset",
    "TokenizedDataset",
    # Inference
    "LFTInference",
    # Evaluation
    "compute_nmse",
    "compute_kl_divergence",
    "compute_perplexity",
    "evaluate_lft",
    "compute_recoupling_ratio",
    "analyze_layer_recoupling",
]
