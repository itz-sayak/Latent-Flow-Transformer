"""Quick test to verify all components work together."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

def test_components():
    """Test all major components."""
    print("=" * 60)
    print("Testing Latent Flow Transformer Components")
    print("=" * 60)
    
    # Test imports
    print("\n[1] Testing imports...")
    from config import ModelConfig, TrainingConfig, InferenceConfig
    from models import VelocityEstimator, FlowLayer, TimeEmbedding
    from training import FlowMatchingTrainer, FlowWalkingTrainer
    from evaluation import compute_nmse, compute_recoupling_ratio
    print("    ✓ All imports successful")
    
    # Test configuration
    print("\n[2] Testing configuration...")
    model_config = ModelConfig(
        hidden_dim=256,
        num_attention_heads=4,
        intermediate_dim=512,
        start_layer=2,
        end_layer=4,
    )
    train_config = TrainingConfig(
        method="fw",
        batch_size=2,
        max_steps=10,
    )
    print(f"    ✓ Model config: hidden_dim={model_config.hidden_dim}")
    print(f"    ✓ Train config: method={train_config.method}")
    
    # Test time embedding
    print("\n[3] Testing time embedding...")
    time_embed = TimeEmbedding(embedding_dim=256, hidden_dim=256)
    t = torch.rand(2)
    t_emb = time_embed(t)
    assert t_emb.shape == (2, 256), f"Expected (2, 256), got {t_emb.shape}"
    print(f"    ✓ Time embedding: {t.shape} -> {t_emb.shape}")
    
    # Test velocity estimator
    print("\n[4] Testing velocity estimator...")
    velocity_est = VelocityEstimator(
        hidden_dim=256,
        num_heads=4,
        intermediate_dim=512,
        num_layers=1,
        use_causal_mask=True,
    )
    x = torch.randn(2, 16, 256)  # (B, S, D)
    t = torch.rand(2)
    v = velocity_est(x, t)
    assert v.shape == x.shape, f"Expected {x.shape}, got {v.shape}"
    print(f"    ✓ Velocity estimator: {x.shape} -> {v.shape}")
    
    # Test flow layer
    print("\n[5] Testing flow layer...")
    flow_layer = FlowLayer(velocity_est, use_midpoint=True)
    x0 = torch.randn(2, 16, 256)
    x1_hat = flow_layer(x0, num_steps=3)
    assert x1_hat.shape == x0.shape, f"Expected {x0.shape}, got {x1_hat.shape}"
    print(f"    ✓ Flow layer (k=3): {x0.shape} -> {x1_hat.shape}")
    
    # Test trajectory
    trajectory = flow_layer.forward_trajectory(x0, num_steps=2)
    assert len(trajectory) == 3, f"Expected 3 states, got {len(trajectory)}"
    print(f"    ✓ Trajectory: {len(trajectory)} states")
    
    # Test SFM trainer
    print("\n[6] Testing SFM trainer...")
    optimizer = torch.optim.Adam(flow_layer.parameters(), lr=1e-4)
    sfm_trainer = FlowMatchingTrainer(
        flow_layer=flow_layer,
        optimizer=optimizer,
        device="cpu",
    )
    x0 = torch.randn(2, 16, 256)
    x1 = torch.randn(2, 16, 256)
    metrics = sfm_trainer.training_step(x0, x1)
    assert "loss" in metrics and isinstance(metrics["loss"], float)
    print(f"    ✓ SFM training step: loss={metrics['loss']:.4f}")
    
    # Test FW trainer
    print("\n[7] Testing FW trainer...")
    optimizer = torch.optim.Adam(flow_layer.parameters(), lr=1e-4)
    fw_trainer = FlowWalkingTrainer(
        flow_layer=flow_layer,
        optimizer=optimizer,
        device="cpu",
        num_steps=3,
        sfm_regularization=0.001,  # Test hybrid mode
    )
    metrics = fw_trainer.training_step(x0, x1)
    assert "loss" in metrics and isinstance(metrics["loss"], float)
    print(f"    ✓ FW training step: loss={metrics['loss']:.4f}")
    
    # Test evaluation metrics
    print("\n[8] Testing evaluation metrics...")
    x1_hat = flow_layer(x0, num_steps=3)
    nmse = compute_nmse(x1_hat, x1)
    print(f"    ✓ NMSE: {nmse:.4f}")
    
    # Test recoupling ratio
    print("\n[9] Testing recoupling ratio...")
    x0_flat = x0.view(-1, 256)[:32]  # Flatten and take subset
    x1_flat = x1.view(-1, 256)[:32]
    ratio, matching = compute_recoupling_ratio(x0_flat, x1_flat)
    print(f"    ✓ Recoupling ratio: {ratio:.4f}")
    
    # Test parameter counting
    print("\n[10] Testing utilities...")
    from utils import count_parameters
    num_params = count_parameters(flow_layer)
    print(f"    ✓ Flow layer parameters: {num_params:,}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return True


def test_full_pipeline():
    """Test a mini training pipeline."""
    print("\n" + "=" * 60)
    print("Testing Mini Training Pipeline")
    print("=" * 60)
    
    from config import ModelConfig, TrainingConfig
    from models import VelocityEstimator, FlowLayer
    from training import FlowWalkingTrainer
    
    # Create small model
    velocity_est = VelocityEstimator(
        hidden_dim=128,
        num_heads=4,
        intermediate_dim=256,
        num_layers=1,
    )
    flow_layer = FlowLayer(velocity_est, use_midpoint=True)
    
    # Create optimizer and trainer
    optimizer = torch.optim.AdamW(flow_layer.parameters(), lr=1e-3)
    trainer = FlowWalkingTrainer(
        flow_layer=flow_layer,
        optimizer=optimizer,
        device="cpu",
        num_steps=3,
    )
    
    # Generate dummy data
    batch_size, seq_len, hidden_dim = 4, 8, 128
    x0 = torch.randn(batch_size, seq_len, hidden_dim)
    x1 = x0 + 0.5 * torch.randn_like(x0)  # Small perturbation
    
    # Training loop
    print("\nTraining for 5 steps...")
    flow_layer.train()
    losses = []
    
    for step in range(5):
        metrics = trainer.training_step(x0, x1)
        losses.append(metrics["loss"])
        print(f"  Step {step+1}: loss={metrics['loss']:.4f}")
    
    # Check loss decreased
    if losses[-1] < losses[0]:
        print("\n✓ Loss decreased during training")
    else:
        print("\n⚠ Loss did not decrease (may need more steps)")
    
    # Test inference
    print("\nTesting inference...")
    flow_layer.eval()
    with torch.no_grad():
        x1_hat = flow_layer(x0, num_steps=3)
    
    mse = torch.nn.functional.mse_loss(x1_hat, x1).item()
    print(f"  Final MSE: {mse:.4f}")
    
    print("\n" + "=" * 60)
    print("Pipeline test complete! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_components()
    test_full_pipeline()
