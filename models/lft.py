"""Full Latent Flow Transformer model."""

import torch
import torch.nn as nn
from typing import Optional, Literal
from transformers import AutoModelForCausalLM, AutoConfig

from .flow_layer import FlowLayer
from .velocity_estimator import VelocityEstimator


class LatentFlowTransformer(nn.Module):
    """
    Full LFT model that integrates:
    - Teacher LLM layers [0, start_layer)
    - Flow layer (replaces layers [start_layer, end_layer])
    - Teacher LLM layers (end_layer, num_layers]
    - LM head
    
    Supports two modes:
    1. Standalone: x0 -> flow -> x1_hat (for training/debugging)
    2. Full: embeddings -> teacher -> flow -> teacher -> logits (for inference)
    """
    
    def __init__(
        self,
        teacher_model_name: str,
        start_layer: int,
        end_layer: int,
        flow_layer: FlowLayer,
        mode: Literal["standalone", "full"] = "full",
        freeze_teacher: bool = True,
    ):
        """
        Args:
            teacher_model_name: HuggingFace model identifier
            start_layer: First layer to replace (m)
            end_layer: Last layer to replace (n)
            flow_layer: Learned flow layer
            mode: "standalone" or "full"
            freeze_teacher: Whether to freeze teacher parameters
        """
        super().__init__()
        
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.mode = mode
        
        # Load teacher model
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            output_hidden_states=True,
        )
        
        # Freeze teacher if requested
        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        # Flow layer (trainable)
        self.flow_layer = flow_layer
        
        # Store model structure info
        config = self.teacher.config
        self.num_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size
        
        # Validate layer indices
        assert 0 <= start_layer < end_layer <= self.num_layers
    
    def forward_standalone(
        self,
        x0: torch.Tensor,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standalone mode: x0 -> flow -> x1_hat
        
        Used during training and for local evaluation.
        
        Args:
            x0: (B, S, D) input latent from layer start_layer
            num_steps: Number of discrete flow steps (k)
            attention_mask: Optional attention mask
            
        Returns:
            (B, S, D) predicted latent x1_hat
        """
        x1_hat = self.flow_layer(x0, num_steps, attention_mask)
        return x1_hat
    
    def forward_full(
        self,
        input_ids: torch.Tensor,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        Full end-to-end mode: embeddings -> teacher -> flow -> teacher -> logits
        
        Args:
            input_ids: (B, S) token IDs
            num_steps: Number of discrete flow steps
            attention_mask: (B, S) attention mask for padding
            return_hidden_states: Whether to return intermediate hidden states
            
        Returns:
            Dict with 'logits' and optionally 'hidden_states'
        """
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        # Extract hidden states
        all_hidden_states = teacher_outputs.hidden_states
        
        # Get input to flow layer (output of layer start_layer-1)
        # Note: all_hidden_states[0] is embeddings, all_hidden_states[i] is output of layer i-1
        x0 = all_hidden_states[self.start_layer]  # Output of layer (start_layer - 1)
        
        # Apply flow layer
        # Create causal mask for flow layer
        seq_len = x0.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x0.device),
            diagonal=1
        ).bool()
        
        x1_hat = self.flow_layer(x0, num_steps, causal_mask)
        
        # Continue with remaining teacher layers
        hidden = x1_hat
        
        # Access the transformer layers directly
        # For Pythia/GPT-NeoX models, layers are in model.gpt_neox.layers
        if hasattr(self.teacher, 'gpt_neox'):
            transformer_layers = self.teacher.gpt_neox.layers
            final_ln = self.teacher.gpt_neox.final_layer_norm
        elif hasattr(self.teacher, 'transformer'):
            # For GPT-2 style models
            transformer_layers = self.teacher.transformer.h
            final_ln = self.teacher.transformer.ln_f
        else:
            raise ValueError(f"Unknown model architecture: {type(self.teacher)}")
        
        # Apply remaining layers (end_layer+1 to num_layers-1)
        with torch.no_grad():
            for layer_idx in range(self.end_layer + 1, self.num_layers):
                layer = transformer_layers[layer_idx]
                hidden = layer(hidden, attention_mask=attention_mask)[0]
            
            # Apply final layer norm
            hidden = final_ln(hidden)
        
        # Get logits
        with torch.no_grad():
            logits = self.teacher.lm_head(hidden)
        
        output = {"logits": logits}
        
        if return_hidden_states:
            output["x0"] = x0
            output["x1_hat"] = x1_hat
            output["final_hidden"] = hidden
        
        return output
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass that dispatches to standalone or full mode.
        
        Args:
            input_ids: (B, S) token IDs (for full mode)
            x0: (B, S, D) input latent (for standalone mode)
            num_steps: Number of discrete flow steps
            attention_mask: Attention mask
            
        Returns:
            Depends on mode
        """
        if self.mode == "standalone":
            assert x0 is not None, "x0 required for standalone mode"
            return self.forward_standalone(x0, num_steps, attention_mask)
        else:
            assert input_ids is not None, "input_ids required for full mode"
            return self.forward_full(input_ids, num_steps, attention_mask, **kwargs)
    
    def set_mode(self, mode: Literal["standalone", "full"]):
        """Change forward mode."""
        self.mode = mode
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters (should be flow layer only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_compression_ratio(self) -> float:
        """
        Compute compression ratio.
        
        Returns ratio of parameters before/after compression.
        """
        num_replaced_layers = self.end_layer - self.start_layer + 1
        
        # Count parameters in one teacher layer (approximate)
        layer = self.teacher.gpt_neox.layers[0] if hasattr(self.teacher, 'gpt_neox') else self.teacher.transformer.h[0]
        params_per_layer = sum(p.numel() for p in layer.parameters())
        
        # Parameters in replaced layers
        params_replaced = params_per_layer * num_replaced_layers
        
        # Parameters in flow layer
        params_flow = sum(p.numel() for p in self.flow_layer.parameters())
        
        return params_replaced / params_flow if params_flow > 0 else 0.0
