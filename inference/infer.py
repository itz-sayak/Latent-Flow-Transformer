"""Inference utilities for Latent Flow Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging

from models import LatentFlowTransformer, FlowLayer
from config import InferenceConfig

logger = logging.getLogger(__name__)


class LFTInference:
    """
    Inference wrapper for Latent Flow Transformer.
    
    Supports:
    - Standalone mode: x0 -> x1_hat
    - Full end-to-end mode: tokens -> logits
    - Text generation
    """
    
    def __init__(
        self,
        lft_model: LatentFlowTransformer,
        config: InferenceConfig,
    ):
        """
        Args:
            lft_model: Trained LFT model
            config: Inference configuration
        """
        self.model = lft_model
        self.config = config
        self.device = config.device
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_latent(
        self,
        x0: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Standalone latent prediction: x0 -> x1_hat
        
        Args:
            x0: (B, S, D) source latent
            num_steps: Number of discrete flow steps (default from config)
            
        Returns:
            (B, S, D) predicted target latent
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        
        # Set model to standalone mode
        self.model.set_mode("standalone")
        
        x0 = x0.to(self.device)
        x1_hat = self.model(x0=x0, num_steps=num_steps)
        
        return x1_hat
    
    @torch.no_grad()
    def predict_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Full end-to-end prediction: tokens -> logits
        
        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) attention mask
            num_steps: Number of discrete flow steps
            
        Returns:
            (B, S, vocab_size) logits
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        
        # Set model to full mode
        self.model.set_mode("full")
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_steps=num_steps,
        )
        
        return outputs["logits"]
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: (B, S) initial tokens
            max_length: Maximum sequence length
            num_steps: Number of flow steps per token
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: End of sequence token ID
            
        Returns:
            (B, max_length) generated token IDs
        """
        if max_length is None:
            max_length = self.config.max_length
        if num_steps is None:
            num_steps = self.config.num_steps
        
        batch_size, seq_len = input_ids.shape
        device = self.device
        
        # Set model to full mode
        self.model.set_mode("full")
        
        # Move to device
        input_ids = input_ids.to(device)
        generated = input_ids.clone()
        
        # Generation loop
        for _ in range(max_length - seq_len):
            # Get logits for next token
            logits = self.predict_logits(generated, num_steps=num_steps)
            
            # Take logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return generated
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> float:
        """
        Compute perplexity on a batch of sequences.
        
        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) attention mask
            num_steps: Number of flow steps
            
        Returns:
            Perplexity value
        """
        # Get logits
        logits = self.predict_logits(input_ids, attention_mask, num_steps)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        )
        
        # Compute perplexity
        perplexity = torch.exp(loss).item()
        
        return perplexity
