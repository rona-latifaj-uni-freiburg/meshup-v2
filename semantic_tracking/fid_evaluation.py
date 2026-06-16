"""
FID (Fréchet Inception Distance) Evaluation for MeshUp

This module provides FID computation between:
1. Rendered images of the deformed mesh vs reference images
2. Can compare against text-generated reference images from the diffusion model

FID measures the similarity between two image distributions using
Inception-v3 features. Lower FID = more similar distributions.

Usage:
    fid = FIDEvaluator(device='cuda')
    score = fid.compute_fid(generated_images, reference_images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
from scipy import linalg
import torchvision.transforms as transforms


class InceptionV3Features(nn.Module):
    """
    Inception-v3 feature extractor for FID computation.
    Uses the pool3 layer (2048-dimensional features).
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # Load pretrained Inception-v3
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception.eval()
        
        # Extract layers up to pool3
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        
        self.blocks = self.blocks.to(device)
        self.blocks.eval()
        
        for param in self.blocks.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for Inception-v3."""
        # Ensure [0, 1] range
        if images.max() > 1.5:
            images = images / 255.0
        
        # Resize to 299x299 (Inception input size)
        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        
        return images
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim features from images."""
        images = self.preprocess(images)
        features = self.blocks(images)
        return features.view(features.size(0), -1)  # (B, 2048)


class FIDEvaluator:
    """
    Compute FID between two sets of images.
    
    FID measures the Fréchet distance between two multivariate Gaussians
    fitted to the Inception-v3 feature distributions.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.inception = InceptionV3Features(device)
        print("FID Evaluator initialized with Inception-v3")
    
    def _compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def _compute_fid_from_stats(
        self, 
        mu1: np.ndarray, 
        sigma1: np.ndarray, 
        mu2: np.ndarray, 
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Compute FID given statistics.
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return float(fid)
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract Inception features from images."""
        # Handle different input formats
        if images.shape[-1] == 3:  # (B, H, W, 3)
            images = images.permute(0, 3, 1, 2)
        
        images = images.to(self.device)
        
        # Process in batches if too many images
        batch_size = 32
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            features = self.inception(batch)
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def compute_fid(
        self, 
        images1: Union[torch.Tensor, List[torch.Tensor]], 
        images2: Union[torch.Tensor, List[torch.Tensor]]
    ) -> float:
        """
        Compute FID between two sets of images.
        
        Args:
            images1: First set of images (B, C, H, W) or list of images
            images2: Second set of images (B, C, H, W) or list of images
        
        Returns:
            FID score (lower is better)
        """
        # Convert lists to tensors
        if isinstance(images1, list):
            images1 = torch.stack(images1)
        if isinstance(images2, list):
            images2 = torch.stack(images2)
        
        # Extract features
        features1 = self.extract_features(images1)
        features2 = self.extract_features(images2)
        
        # Need at least 2 samples for covariance
        if len(features1) < 2 or len(features2) < 2:
            print("Warning: Need at least 2 images per set for FID. Returning -1.")
            return -1.0
        
        # Compute statistics
        mu1, sigma1 = self._compute_statistics(features1)
        mu2, sigma2 = self._compute_statistics(features2)
        
        # Compute FID
        fid = self._compute_fid_from_stats(mu1, sigma1, mu2, sigma2)
        
        return fid
    
    def compute_fid_to_reference_stats(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        ref_mu: np.ndarray,
        ref_sigma: np.ndarray
    ) -> float:
        """Compute FID against pre-computed reference statistics."""
        if isinstance(images, list):
            images = torch.stack(images)
        
        features = self.extract_features(images)
        
        if len(features) < 2:
            return -1.0
        
        mu, sigma = self._compute_statistics(features)
        return self._compute_fid_from_stats(mu, sigma, ref_mu, ref_sigma)


def compute_mesh_fid(
    mesh_renders: torch.Tensor,
    reference_renders: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Convenience function to compute FID between mesh renders and references.
    
    Args:
        mesh_renders: Rendered images of the deformed mesh (N, C, H, W)
        reference_renders: Reference images (M, C, H, W)
        device: Device for computation
    
    Returns:
        FID score
    """
    evaluator = FIDEvaluator(device)
    return evaluator.compute_fid(mesh_renders, reference_renders)
