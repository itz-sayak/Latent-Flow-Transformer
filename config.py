"""Configuration dataclasses for Latent Flow Transformer."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for LFT model architecture."""
    
    # Teacher model
    teacher_name: str = "EleutherAI/pythia-410m"
    
    # Layer compression range
    start_layer: int = 6  # Layer m (inclusive)
    end_layer: int = 18   # Layer n (inclusive)
    
    # Model dimensions (inherited from teacher)
    hidden_dim: int = 1024
    num_attention_heads: int = 16
    intermediate_dim: int = 4096
    
    # Time embedding
    time_embed_dim: int = 1024  # Same as hidden_dim
    time_embed_type: str = "sinusoidal"
    
    # Velocity estimator
    num_velocity_layers: int = 1  # Single transformer block by default
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    
    # Architecture variant (Figure 1 in paper)
    # "sequential": Standard DiT (Figure 1a/b) - Attention then FFN
    # "parallel": Pythia-style (Figure 1c) - Parallel Attention and FFN
    architecture_variant: str = "parallel"  # Default to Pythia-style
    
    # Attention
    use_causal_mask: bool = True  # Autoregressive
    attention_dropout: float = 0.0
    
    # Device
    device: str = "cuda"
    dtype: str = "float32"  # or "bfloat16"


@dataclass
class TrainingConfig:
    """Configuration for training LFT."""
    
    # Training method
    method: Literal["sfm", "fw", "hybrid"] = "sfm"  # Standard Flow Matching, Flow Walking, or Hybrid
    
    # Data
    dataset_path: str = "pile"
    cache_dir: str = "./cache/latent_pairs"
    max_train_tokens: int = 2_600_000_000  # 2.6B as in paper
    max_eval_tokens: int = 10_000_000
    
    # Optimization
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 100_000
    warmup_steps: int = 1000
    
    # AdamW
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Flow Walking specific
    fw_num_steps: int = 3  # k=3 for training
    use_midpoint: bool = True  # Midpoint integration for FW
    sfm_regularization: float = 0.001  # alpha for hybrid loss (Equation 9)
    
    # Training infrastructure
    use_gradient_checkpointing: bool = False
    mixed_precision: bool = False  # Use AMP
    compile_model: bool = False  # torch.compile
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./outputs/lft"
    
    # Reproducibility
    seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for LFT inference."""
    
    # Inference mode
    mode: Literal["standalone", "full"] = "full"  # standalone: x0→x̂1, full: end-to-end LM
    
    # Number of discrete time steps
    num_steps: int = 1  # k (can vary at inference)
    
    # Integration method
    use_midpoint: bool = True
    
    # Generation
    max_length: int = 512
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Checkpoint
    checkpoint_path: Optional[str] = None
    
    # Device
    device: str = "cuda"
    batch_size: int = 1


@dataclass
class RecouplingConfig:
    """Configuration for recoupling ratio computation."""
    
    # OT method
    ot_method: Literal["sinkhorn", "emd"] = "sinkhorn"
    
    # Sinkhorn parameters
    sinkhorn_reg: float = 0.1  # Entropic regularization
    sinkhorn_num_iters: int = 100
    
    # Data sampling
    num_samples: int = 256  # Tokens to sample for OT
    
    # Device
    device: str = "cpu"  # OT computation on CPU
    
    # Layer range to analyze
    start_layer: int = 0
    end_layer: int = 23  # For Pythia-410m (24 layers)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Metrics to compute
    compute_nmse: bool = True
    compute_kl_latent: bool = True
    compute_kl_lm: bool = True
    compute_perplexity: bool = True
    
    # Evaluation data
    eval_dataset: str = "pile"
    num_eval_samples: int = 1000
    
    # Device
    device: str = "cuda"
    batch_size: int = 8
