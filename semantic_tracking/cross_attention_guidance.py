"""
Diffusion Cross-Attention Semantic Guidance for MeshUp

This module extracts and uses cross-attention maps from the DeepFloyd diffusion
model to provide semantic guidance during mesh deformation. Unlike DINO which
uses visual similarity, this approach leverages the language-vision understanding
of the diffusion model.

Key Insight:
- Cross-attention maps show which image regions correspond to which text tokens
- Regions that don't match any prompt word get low attention → should transform/shrink
- Regions matching specific words (head, body, legs) get high attention → preserve structure

Usage:
    guidance = CrossAttentionGuidance(device='cuda')
    guidance.set_prompt_info(tokenizer, "a human")
    # During training (in loop_tracked.py):
    # 1. Get SDS with attention: result = guidance_model.SDS_with_attention(...)
    # 2. Compute attention loss: attn_loss = guidance(result['attention_maps'], epoch=epoch)

This is a STANDALONE module that can be used:
- Alone (without DINO)
- Together with DINO
- Disabled entirely
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import math


class CrossAttentionGuidance(nn.Module):
    """
    Cross-Attention based semantic guidance for mesh deformation.
    
    This module:
    1. Receives cross-attention maps from the UNet (via SDS_with_attention)
    2. Analyzes which image regions attend to which prompt words
    3. Creates guidance signals based on semantic attention patterns
    
    The key insight is that cross-attention directly reveals the diffusion
    model's understanding of "what should be where" based on the prompt.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        weight: float = 0.1,
        # Guidance settings
        use_consistency_guidance: bool = True,   # Keep attention patterns consistent with reference
        use_entropy_guidance: bool = True,       # Penalize uncertain (high entropy) regions
        use_coverage_guidance: bool = True,      # Ensure prompt words are well-covered
        consistency_weight: float = 0.5,
        entropy_weight: float = 0.3,
        coverage_weight: float = 0.2,
        # Warm-up settings
        warmup_epochs: int = 50,
        warmup_type: str = 'linear',
    ):
        super().__init__()
        
        self.device = device
        self.weight = weight
        self.use_consistency_guidance = use_consistency_guidance
        self.use_entropy_guidance = use_entropy_guidance
        self.use_coverage_guidance = use_coverage_guidance
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
        self.coverage_weight = coverage_weight
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        
        # Prompt token info
        self.num_tokens = 0
        self.token_to_word = {}
        
        # Reference attention maps (from source mesh at init)
        self.reference_attention = None
        self.initialized = False
    
    def get_warmup_factor(self, epoch: int) -> float:
        """Compute warmup factor for gradual loss introduction."""
        if epoch < self.warmup_epochs:
            if self.warmup_type == 'linear':
                return epoch / self.warmup_epochs
            elif self.warmup_type == 'cosine':
                return 0.5 * (1 - math.cos(math.pi * epoch / self.warmup_epochs))
            elif self.warmup_type == 'step':
                return 0.0
        return 1.0
    
    def set_prompt_info(self, tokenizer, prompt: str):
        """
        Set up token information for the prompt.
        
        Args:
            tokenizer: The tokenizer from the diffusion pipeline
            prompt: The text prompt
        """
        # Tokenize the prompt
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        prompt_tokens = tokens['input_ids'][0]
        self.num_tokens = (prompt_tokens != tokenizer.pad_token_id).sum().item()
        
        # Decode individual tokens for debugging
        self.token_to_word = {}
        for i in range(self.num_tokens):
            token_id = prompt_tokens[i].item()
            word = tokenizer.decode([token_id])
            self.token_to_word[i] = word
        
        print(f"Cross-attention guidance set for prompt: '{prompt}'")
        print(f"  Tokens ({self.num_tokens}): {[self.token_to_word[i] for i in range(min(10, self.num_tokens))]}")
    
    def process_attention_maps(
        self,
        attention_maps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw attention maps into useful guidance signals.
        
        Args:
            attention_maps: Raw cross-attention maps (B*heads, H*W, tokens)
        
        Returns:
            Dictionary with processed attention signals
        """
        if attention_maps is None:
            return None
        
        # Get spatial dimensions
        total_batch_heads = attention_maps.shape[0]
        seq_len = attention_maps.shape[1]
        num_tokens = attention_maps.shape[2]
        h = w = int(math.sqrt(seq_len))
        
        # Normalize attention over tokens
        attn_normalized = attention_maps / (attention_maps.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Average over batch*heads dimension
        # Shape: (H*W, num_tokens)
        attn_avg = attn_normalized.mean(dim=0)
        
        # Reshape to spatial: (H, W, num_tokens)
        attn_spatial = attn_avg.view(h, w, num_tokens)
        
        # Only consider actual tokens (not padding) - skip BOS token (index 0)
        if self.num_tokens > 1:
            attn_valid = attn_spatial[..., 1:self.num_tokens]
        else:
            attn_valid = attn_spatial[..., 1:]  # Skip BOS token
        
        # Compute attention entropy (uncertainty measure)
        # High entropy = uncertain about which token this region matches
        entropy = -(attn_valid * (attn_valid + 1e-8).log()).sum(dim=-1)
        
        # Compute max attention (confidence measure)
        max_attn, max_idx = attn_valid.max(dim=-1)
        
        # Compute total attention per token (coverage)
        # Which tokens are getting attention from the image?
        token_coverage = attn_valid.sum(dim=(0, 1))  # Sum over spatial dims
        
        return {
            'attention': attn_spatial,          # Full attention maps (H, W, tokens)
            'attention_valid': attn_valid,      # Only valid tokens (H, W, valid_tokens)
            'entropy': entropy,                 # Per-pixel entropy (H, W)
            'max_attention': max_attn,          # Max attention per pixel (H, W)
            'max_token_idx': max_idx,           # Which token each pixel attends to most (H, W)
            'token_coverage': token_coverage,   # Coverage per token (valid_tokens,)
        }
    
    @torch.no_grad()
    def set_reference(self, attention_maps: torch.Tensor):
        """
        Set reference attention maps from initial source mesh renders.
        
        This should be called early in training (e.g., first few iterations)
        to capture the "starting point" attention patterns.
        
        Args:
            attention_maps: Cross-attention from source mesh renders
        """
        if attention_maps is None:
            print("Warning: Cannot set reference - attention_maps is None")
            return
            
        processed = self.process_attention_maps(attention_maps)
        if processed is not None:
            self.reference_attention = {
                k: v.detach().clone() for k, v in processed.items()
            }
            self.initialized = True
            print("Cross-attention reference set from source mesh")
            print(f"  Reference entropy mean: {processed['entropy'].mean().item():.4f}")
            print(f"  Reference max_attention mean: {processed['max_attention'].mean().item():.4f}")
    
    def compute_consistency_loss(
        self,
        current: Dict[str, torch.Tensor],
        reference: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss encouraging attention patterns to remain consistent.
        
        This helps preserve "what attends to what" relationships - if a region
        attended to "body" initially, it should still attend to "body".
        """
        if reference is None:
            return torch.tensor(0.0, device=self.device)
        
        curr_attn = current['attention_valid']
        ref_attn = reference['attention_valid']
        
        # Handle size mismatch
        if curr_attn.shape != ref_attn.shape:
            # Resize reference to match current
            ref_attn = F.interpolate(
                ref_attn.permute(2, 0, 1).unsqueeze(0),  # (1, tokens, H, W)
                size=curr_attn.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # Back to (H, W, tokens)
        
        # KL divergence between attention distributions
        # Encourage current to match reference
        curr_log = (curr_attn + 1e-8).log()
        ref_prob = ref_attn + 1e-8
        
        # KL(ref || curr) = sum(ref * log(ref/curr))
        kl_div = (ref_prob * (ref_prob.log() - curr_log)).sum(dim=-1).mean()
        
        return kl_div
    
    def compute_entropy_loss(
        self,
        current: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute entropy regularization.
        
        Lower entropy = more confident attention = better semantic alignment.
        Regions should clearly correspond to specific prompt tokens.
        """
        entropy = current['entropy']
        
        # Penalize high entropy (uncertainty)
        loss = entropy.mean()
        
        return loss
    
    def compute_coverage_loss(
        self,
        current: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute coverage loss.
        
        Encourage all content tokens to be well-represented in the image.
        This helps ensure semantic concepts from the prompt are present.
        """
        coverage = current['token_coverage']
        
        # We want uniform coverage - each token should get similar attention
        # Compute variance of coverage
        coverage_mean = coverage.mean()
        coverage_var = ((coverage - coverage_mean) ** 2).mean()
        
        # Also penalize tokens with very low coverage
        min_coverage = coverage.min()
        low_coverage_penalty = F.relu(0.1 - min_coverage)
        
        return coverage_var + low_coverage_penalty
    
    def forward(
        self,
        attention_maps: Optional[torch.Tensor] = None,
        epoch: int = 0,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cross-attention guidance loss.
        
        Args:
            attention_maps: Cross-attention maps from UNet (via SDS_with_attention)
            epoch: Current epoch (for warmup)
            return_components: Whether to return loss components
        
        Returns:
            Total loss, or dict with components if return_components=True
        """
        # Get warmup factor
        warmup = self.get_warmup_factor(epoch)
        
        if warmup == 0 or attention_maps is None:
            if return_components:
                return {
                    'total': torch.tensor(0.0, device=self.device),
                    'consistency': torch.tensor(0.0, device=self.device),
                    'entropy': torch.tensor(0.0, device=self.device),
                    'coverage': torch.tensor(0.0, device=self.device),
                    'warmup': warmup
                }
            return torch.tensor(0.0, device=self.device)
        
        # Process attention maps
        current = self.process_attention_maps(attention_maps)
        
        if current is None:
            if return_components:
                return {
                    'total': torch.tensor(0.0, device=self.device),
                    'consistency': torch.tensor(0.0, device=self.device),
                    'entropy': torch.tensor(0.0, device=self.device),
                    'coverage': torch.tensor(0.0, device=self.device),
                    'warmup': warmup
                }
            return torch.tensor(0.0, device=self.device)
        
        # Compute loss components
        consistency_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        coverage_loss = torch.tensor(0.0, device=self.device)
        
        if self.use_consistency_guidance and self.reference_attention is not None:
            consistency_loss = self.compute_consistency_loss(current, self.reference_attention)
        
        if self.use_entropy_guidance:
            entropy_loss = self.compute_entropy_loss(current)
        
        if self.use_coverage_guidance:
            coverage_loss = self.compute_coverage_loss(current)
        
        # Combine losses
        total_loss = (
            self.consistency_weight * consistency_loss +
            self.entropy_weight * entropy_loss +
            self.coverage_weight * coverage_loss
        ) * self.weight * warmup
        
        if return_components:
            return {
                'total': total_loss,
                'consistency': consistency_loss * self.consistency_weight * self.weight * warmup,
                'entropy': entropy_loss * self.entropy_weight * self.weight * warmup,
                'coverage': coverage_loss * self.coverage_weight * self.weight * warmup,
                'warmup': warmup,
                'stats': {
                    'mean_entropy': current['entropy'].mean().item(),
                    'mean_max_attn': current['max_attention'].mean().item(),
                }
            }
        
        return total_loss


def create_cross_attention_guidance(
    device: str = 'cuda',
    weight: float = 0.1,
    warmup_epochs: int = 50,
    **kwargs
) -> CrossAttentionGuidance:
    """
    Convenience function to create cross-attention guidance.
    
    Args:
        device: Device to run on
        weight: Overall loss weight
        warmup_epochs: Epochs before loss fully activates
        **kwargs: Additional arguments (consistency_weight, entropy_weight, etc.)
    
    Returns:
        Configured CrossAttentionGuidance instance
    """
    return CrossAttentionGuidance(
        device=device,
        weight=weight,
        warmup_epochs=warmup_epochs,
        **kwargs
    )
