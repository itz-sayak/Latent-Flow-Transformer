"""Teacher interface for extracting and caching latent pairs."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import os
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class TeacherInterface:
    """
    Interface for extracting (x0, x1) latent pairs from a teacher LLM.
    
    Extracts hidden states at layers m and n for each token position.
    """
    
    def __init__(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            start_layer: Layer m (source layer)
            end_layer: Layer n (target layer)
            device: Device to run teacher on
            dtype: Data type for computation
        """
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype
        
        # Load teacher model
        logger.info(f"Loading teacher model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=dtype,
        ).to(device)
        
        # Freeze teacher
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Teacher loaded. Will extract layers {start_layer} -> {end_layer}")
    
    @torch.no_grad()
    def extract_latent_pairs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract (x0, x1) pairs from teacher.
        
        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) attention mask
            
        Returns:
            x0: (B, S, D) hidden states at start_layer
            x1: (B, S, D) hidden states at end_layer
        """
        # Forward pass through teacher
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            output_hidden_states=True,
            use_cache=False,
        )
        
        # Extract hidden states
        # hidden_states[0] is embeddings, hidden_states[i] is output of layer i-1
        all_hidden_states = outputs.hidden_states
        
        x0 = all_hidden_states[self.start_layer]  # Output of layer (start_layer - 1)
        x1 = all_hidden_states[self.end_layer + 1]  # Output of layer end_layer
        
        return x0, x1
    
    def cache_dataset_latents(
        self,
        dataset,
        cache_dir: str,
        max_samples: Optional[int] = None,
        batch_size: int = 8,
    ):
        """
        Pre-compute and cache latent pairs for entire dataset.
        
        Args:
            dataset: Dataset yielding tokenized samples
            cache_dir: Directory to save cached latents
            max_samples: Maximum number of samples to process
            batch_size: Batch size for processing
        """
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(
            cache_dir,
            f"latents_l{self.start_layer}_to_l{self.end_layer}.pt"
        )
        
        if os.path.exists(cache_file):
            logger.info(f"Cache already exists: {cache_file}")
            return cache_file
        
        logger.info(f"Caching latent pairs to {cache_file}")
        
        all_x0 = []
        all_x1 = []
        
        num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Caching latents"):
            batch_end = min(i + batch_size, num_samples)
            batch = [dataset[j] for j in range(i, batch_end)]
            
            # Collate batch
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch]) if "attention_mask" in batch[0] else None
            
            # Extract latents
            x0, x1 = self.extract_latent_pairs(input_ids, attention_mask)
            
            # Move to CPU and store
            all_x0.append(x0.cpu())
            all_x1.append(x1.cpu())
        
        # Concatenate and save
        all_x0 = torch.cat(all_x0, dim=0)
        all_x1 = torch.cat(all_x1, dim=0)
        
        torch.save({
            "x0": all_x0,
            "x1": all_x1,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "model_name": self.model_name,
        }, cache_file)
        
        logger.info(f"Cached {all_x0.size(0)} latent pairs")
        logger.info(f"Shape: x0={all_x0.shape}, x1={all_x1.shape}")
        
        return cache_file


class LatentPairDataset(Dataset):
    """
    Dataset of cached (x0, x1) latent pairs.
    """
    
    def __init__(self, cache_file: str, device: str = "cuda"):
        """
        Args:
            cache_file: Path to cached latent pairs
            device: Device to load tensors to
        """
        self.device = device
        
        logger.info(f"Loading cached latents from {cache_file}")
        data = torch.load(cache_file, map_location="cpu")
        
        self.x0 = data["x0"]
        self.x1 = data["x1"]
        self.start_layer = data["start_layer"]
        self.end_layer = data["end_layer"]
        
        logger.info(f"Loaded {len(self)} latent pairs")
        logger.info(f"Layers: {self.start_layer} -> {self.end_layer}")
    
    def __len__(self) -> int:
        return self.x0.size(0)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with 'x0' and 'x1' tensors
        """
        return {
            "x0": self.x0[idx].to(self.device),
            "x1": self.x1[idx].to(self.device),
        }


class TokenizedDataset(Dataset):
    """
    Simple dataset wrapper for tokenized text data.
    """
    
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Tokenizing {len(texts)} samples...")
        
        # Tokenize all texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    
    def __len__(self) -> int:
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
