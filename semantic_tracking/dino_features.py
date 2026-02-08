"""
DINOv2 Feature Extraction and Correspondence for MeshUp

This module provides DINOv2-based feature extraction that can be used to:
1. Add semantic correspondence loss during optimization
2. Extract features for clustering vertices into semantic parts
3. Match semantic parts between source and target concepts

This is a lightweight addition to MeshUp that doesn't replace the main
DeepFloyd IF guidance, but augments it with semantic understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union


class DINOv2FeatureExtractor(nn.Module):
    """
    Extract semantic features from rendered mesh images using DINOv2.
    
    DINOv2 provides semantically meaningful features without any training,
    which can be used to establish correspondence between mesh parts.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        device: str = 'cuda',
        frozen: bool = True
    ):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name: DINOv2 model variant:
                - 'dinov2_vits14': Small (fastest)
                - 'dinov2_vitb14': Base
                - 'dinov2_vitl14': Large
                - 'dinov2_vitg14': Giant (best quality)
            device: Device to run on
            frozen: Whether to freeze the model weights
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load DINOv2 from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        
        if frozen:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model properties
        self.patch_size = 14  # DINOv2 uses 14x14 patches
        self.embed_dim = self.model.embed_dim
        
        # ImageNet normalization (DINOv2 expects this)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for DINOv2.
        
        Args:
            images: Input images, shape (B, C, H, W) in [0, 1]
        
        Returns:
            Preprocessed images
        """
        # Normalize with ImageNet stats
        images = (images - self.mean) / self.std
        
        # Resize to be divisible by patch size (DINOv2 requirement)
        H, W = images.shape[2:]
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        
        if new_H != H or new_W != W:
            images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        return images
    
    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        return_cls: bool = True,
        return_patches: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract DINOv2 features from images.
        
        Args:
            images: Input images, shape (B, C, H, W) in [0, 1]
            return_cls: Whether to return CLS token (global feature)
            return_patches: Whether to return patch features
        
        Returns:
            Dictionary containing:
                - 'cls': CLS token features, shape (B, D)
                - 'patches': Patch features, shape (B, H', W', D)
        """
        images = self.preprocess(images)
        
        # Get features
        features = self.model.forward_features(images)
        
        result = {}
        
        if return_cls:
            # CLS token is the first token
            result['cls'] = features['x_norm_clstoken']
        
        if return_patches:
            # Patch tokens are the rest
            patch_tokens = features['x_norm_patchtokens']
            B = images.shape[0]
            H = images.shape[2] // self.patch_size
            W = images.shape[3] // self.patch_size
            result['patches'] = patch_tokens.reshape(B, H, W, -1)
        
        return result
    
    @torch.no_grad()
    def extract_cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract global (CLS) features from images.
        
        Args:
            images: Input images, shape (B, C, H, W) in [0, 1]
        
        Returns:
            CLS features, shape (B, D)
        """
        return self.extract_features(images, return_cls=True, return_patches=False)['cls']
    
    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from images.
        
        Args:
            images: Input images, shape (B, C, H, W) in [0, 1]
        
        Returns:
            Patch features, shape (B, H', W', D)
        """
        return self.extract_features(images, return_cls=False, return_patches=True)['patches']
    
    def compute_feature_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute cosine similarity between feature sets.
        
        Args:
            features1: First features, shape (B, D) or (B, H, W, D)
            features2: Second features, same shape as features1
            normalize: Whether to L2-normalize features
        
        Returns:
            Similarity scores
        """
        if normalize:
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)
        
        return (features1 * features2).sum(dim=-1)


class SemanticCorrespondenceLoss(nn.Module):
    """
    Loss function that encourages semantic correspondence preservation.
    
    This can be added to the MeshUp optimization to ensure that
    semantically corresponding parts remain corresponding after deformation.
    """
    
    def __init__(
        self,
        dino_extractor: DINOv2FeatureExtractor,
        weight: float = 0.1,
        feature_type: str = 'patches'
    ):
        """
        Initialize semantic correspondence loss.
        
        Args:
            dino_extractor: DINOv2 feature extractor
            weight: Loss weight
            feature_type: Type of features to use ('cls' or 'patches')
        """
        super().__init__()
        self.dino = dino_extractor
        self.weight = weight
        self.feature_type = feature_type
        
        # Store reference features from initial render
        self.reference_features = None
    
    def set_reference(self, images: torch.Tensor):
        """
        Set reference features from initial mesh renders.
        
        Args:
            images: Rendered images of original mesh, shape (B, C, H, W)
        """
        with torch.no_grad():
            if self.feature_type == 'cls':
                self.reference_features = self.dino.extract_cls_features(images)
            else:
                self.reference_features = self.dino.extract_patch_features(images)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic correspondence loss.
        
        This compares current renders against reference renders
        and penalizes changes that break semantic correspondence.
        
        Args:
            images: Current rendered images, shape (B, C, H, W)
        
        Returns:
            Loss value
        """
        if self.reference_features is None:
            return torch.tensor(0.0, device=images.device)
        
        # Extract current features
        if self.feature_type == 'cls':
            current_features = self.dino.extract_cls_features(images)
        else:
            current_features = self.dino.extract_patch_features(images)
        
        # Compute cosine similarity loss
        similarity = self.dino.compute_feature_similarity(
            current_features, self.reference_features
        )
        
        # We want high similarity, so loss is 1 - similarity
        loss = (1 - similarity.mean()) * self.weight
        
        return loss


class PartClusteringWithDINO:
    """
    Cluster mesh vertices into semantic parts using DINOv2 features.
    
    This renders the mesh from multiple views, extracts DINOv2 patch features,
    and uses them to cluster vertices into semantically meaningful parts.
    """
    
    def __init__(
        self,
        dino_extractor: DINOv2FeatureExtractor,
        n_parts: int = 8
    ):
        """
        Initialize part clustering.
        
        Args:
            dino_extractor: DINOv2 feature extractor
            n_parts: Number of parts to cluster into
        """
        self.dino = dino_extractor
        self.n_parts = n_parts
    
    @torch.no_grad()
    def cluster_from_renders(
        self,
        rendered_images: torch.Tensor,
        vertex_visibility_maps: torch.Tensor
    ) -> np.ndarray:
        """
        Cluster vertices based on DINOv2 features from renders.
        
        Args:
            rendered_images: Multi-view renders, shape (N_views, C, H, W)
            vertex_visibility_maps: Per-vertex visibility per view,
                                   shape (N_views, H, W) with vertex indices
        
        Returns:
            Part labels per vertex
        """
        from sklearn.cluster import KMeans
        
        # Extract patch features for all views
        all_features = []
        for img in rendered_images:
            features = self.dino.extract_patch_features(img.unsqueeze(0))
            all_features.append(features.squeeze(0))
        
        # Aggregate features per vertex across views
        # This is a simplified version - in practice you'd need proper
        # visibility handling and feature aggregation
        
        # For now, just cluster the patch features
        all_patches = torch.cat([f.reshape(-1, f.shape[-1]) for f in all_features], dim=0)
        all_patches = all_patches.cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.n_parts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_patches)
        
        return labels


def create_dino_feature_loss(
    device: str = 'cuda',
    model_name: str = 'dinov2_vits14',
    weight: float = 0.1,
    feature_type: str = 'patches'
) -> Tuple[DINOv2FeatureExtractor, SemanticCorrespondenceLoss]:
    """
    Convenience function to create DINOv2 extractor and loss.
    
    Args:
        device: Device to run on
        model_name: DINOv2 model variant
        weight: Loss weight
        feature_type: Feature type to use
    
    Returns:
        Tuple of (extractor, loss_function)
    """
    extractor = DINOv2FeatureExtractor(model_name=model_name, device=device)
    loss_fn = SemanticCorrespondenceLoss(extractor, weight=weight, feature_type=feature_type)
    
    return extractor, loss_fn


# ============================================================================
# Simple ImageNet class label encoding using CLIP (Idea #2)
# ============================================================================

class CLIPClassLabelEncoder:
    """
    Encode ImageNet class labels using CLIP for use with DeepFloyd IF.
    
    This provides a simple way to use ImageNet classes instead of free-form
    text prompts, which can be more semantically precise for some applications.
    """
    
    # Common ImageNet classes useful for mesh deformation
    COMMON_CLASSES = {
        # Animals
        'cat': 'a cat, feline animal',
        'dog': 'a dog, canine animal',
        'hippo': 'a hippopotamus, large african mammal',
        'frog': 'a frog, amphibian',
        'elephant': 'an elephant, large mammal with trunk',
        'lion': 'a lion, big cat',
        'tiger': 'a tiger, striped big cat',
        'bear': 'a bear, large mammal',
        'rabbit': 'a rabbit, small mammal with long ears',
        'horse': 'a horse, equine animal',
        
        # Objects
        'chair': 'a chair, furniture for sitting',
        'table': 'a table, furniture surface',
        'car': 'a car, automobile vehicle',
        'airplane': 'an airplane, flying vehicle',
        'boat': 'a boat, water vessel',
        
        # Other
        'human': 'a human, person',
        'robot': 'a robot, mechanical humanoid',
    }
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize CLIP encoder.
        
        Args:
            device: Device to run on
        """
        self.device = device
        self._clip_model = None
        self._clip_processor = None
    
    def _load_clip(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            try:
                import clip
                self._clip_model, self._clip_processor = clip.load("ViT-B/32", device=self.device)
            except ImportError:
                print("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
                raise
    
    def get_class_prompt(self, class_name: str, style_suffix: str = ", a 3d rendering") -> str:
        """
        Get a descriptive prompt for a class name.
        
        Args:
            class_name: ImageNet class name or common name
            style_suffix: Suffix to add for 3D rendering style
        
        Returns:
            Descriptive text prompt
        """
        # Check if we have a predefined description
        class_lower = class_name.lower().strip()
        if class_lower in self.COMMON_CLASSES:
            base_prompt = self.COMMON_CLASSES[class_lower]
        else:
            # Use the class name directly
            base_prompt = f"a {class_name}"
        
        return base_prompt + style_suffix
    
    def encode_class(self, class_name: str) -> torch.Tensor:
        """
        Encode a class name using CLIP.
        
        Args:
            class_name: Class name to encode
        
        Returns:
            CLIP text embedding
        """
        self._load_clip()
        import clip
        
        prompt = self.get_class_prompt(class_name, style_suffix="")
        text = clip.tokenize([prompt]).to(self.device)
        
        with torch.no_grad():
            text_features = self._clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
