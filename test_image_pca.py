#!/usr/bin/env python3
"""
DINO PCA Visualization for Images

Generates PCA visualization of DINO features for input images.
Uses 2-step PCA: first to separate foreground/background, second for coloring.

KEY: Processes multiple images JOINTLY - fits PCA on combined features from
all images, which produces consistent and symmetric coloring across images.
This is the approach used in the original purnasai/Dino_V2 implementation.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_dino_model(model_name: str = 'dinov2_vitl14', device: str = 'cuda'):
    """
    Load DINOv2 model.
    
    Note: The original purnasai/Dino_V2 implementation uses dinov2_vitl14 (Large).
    - dinov2_vits14: Small, 384D features
    - dinov2_vitb14: Base, 768D features  
    - dinov2_vitl14: Large, 1024D features (RECOMMENDED for matching paper results)
    - dinov2_vitg14: Giant, 1536D features
    
    The _reg variants include register tokens and may produce different results.
    """
    print(f"Loading {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)
    model.eval()
    return model


def extract_features(model, image: torch.Tensor, device: str = 'cuda'):
    """
    Extract patch features from DINO model.
    
    Args:
        model: DINOv2 model
        image: Input image tensor (B, C, H, W), normalized
        device: Device
    
    Returns:
        Patch features (B, H', W', D)
    """
    with torch.no_grad():
        # Get patch size
        patch_size = model.patch_size  # Usually 14
        
        # Get features using forward_features
        outputs = model.forward_features(image)
        patch_tokens = outputs['x_norm_patchtokens']  # (B, N, D)
        
        # Reshape to spatial grid
        B, N, D = patch_tokens.shape
        H_patches = image.shape[2] // patch_size
        W_patches = image.shape[3] // patch_size
        
        features = patch_tokens.reshape(B, H_patches, W_patches, D)
        
    return features


def apply_two_step_pca_combined(features_list: list, n_components: int = 3):
    """
    Apply 2-step PCA to DINO features from MULTIPLE images combined.
    
    This is the key difference from single-image PCA:
    - By fitting PCA on combined features from all images, we capture
      common semantic structure across all views
    - This produces symmetric coloring because "wing" patches from different
      angles all cluster together
    
    Step 1: Use first PCA component to separate foreground/background
    Step 2: Apply PCA only to foreground for better semantic coloring
    
    Args:
        features_list: List of (H, W, D) feature maps, one per image
        n_components: Number of PCA components (3 for RGB)
    
    Returns:
        List of RGB visualizations, one per image (each H, W, 3)
    """
    n_images = len(features_list)
    
    # Get dimensions from first image
    H, W, D = features_list[0].shape
    n_patches_per_image = H * W
    
    # Combine all features into one array
    print(f"\n=== Combined PCA on {n_images} images ===")
    all_features = np.concatenate([f.reshape(-1, D) for f in features_list], axis=0)
    print(f"Combined features shape: {all_features.shape} ({n_images} x {H}x{W} patches, {D}D)")
    
    # Step 1: First PCA on ALL combined features
    print("Step 1: PCA on combined features...")
    pca1 = PCA(n_components=n_components)
    pca_features1 = pca1.fit_transform(all_features)  # (n_images*H*W, 3)
    
    print(f"  Explained variance: {pca1.explained_variance_ratio_}")
    
    # Use first component to separate fg/bg (across all images)
    pc1 = pca_features1[:, 0]
    
    # Min-max scale PC1 first (as in original implementation)
    pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    
    # Fixed threshold at 0.35 (as in original purnasai/Dino_V2 implementation)
    # This was determined empirically from histogram analysis
    # IMPORTANT: In the original, PC1 > 0.35 means BACKGROUND, not foreground!
    threshold = 0.35
    
    bg_mask = pc1_scaled > threshold  # Background: PC1 > threshold
    fg_mask = ~bg_mask                # Foreground: PC1 <= threshold
    
    n_fg = fg_mask.sum()
    n_bg = bg_mask.sum()
    print(f"  Foreground: {n_fg} pixels, Background: {n_bg} pixels")
    
    # If separation is too extreme, use simple PCA
    if n_fg < 100 or n_bg < 100:
        print("  WARNING: Using simple min-max scaling (poor fg/bg separation)")
        colors = np.zeros((len(all_features), 3), dtype=np.float32)
        for i in range(3):
            col = pca_features1[:, i]
            colors[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        
        # Split back into per-image results
        results = []
        for i in range(n_images):
            start_idx = i * n_patches_per_image
            end_idx = (i + 1) * n_patches_per_image
            img_colors = colors[start_idx:end_idx].reshape(H, W, 3)
            results.append(img_colors)
        return results
    
    # Step 2: PCA on foreground only (combined across all images)
    print("Step 2: PCA on combined foreground...")
    pca2 = PCA(n_components=n_components)
    fg_features = all_features[fg_mask]
    pca_features_fg = pca2.fit_transform(fg_features)
    
    print(f"  Explained variance: {pca2.explained_variance_ratio_}")
    
    # Normalize foreground features (GLOBAL normalization across all images)
    for i in range(n_components):
        col = pca_features_fg[:, i]
        pca_features_fg[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    
    # Build combined output
    colors = np.zeros((len(all_features), 3), dtype=np.float32)
    colors[bg_mask] = 0.0  # Black background
    colors[fg_mask] = pca_features_fg
    
    # Split back into per-image results
    results = []
    for i in range(n_images):
        start_idx = i * n_patches_per_image
        end_idx = (i + 1) * n_patches_per_image
        img_colors = colors[start_idx:end_idx].reshape(H, W, 3)
        results.append(img_colors)
    
    print(f"=== Done: generated {n_images} PCA visualizations ===\n")
    return results


def preprocess_image(image_path: str, image_size: int = 518):
    """
    Load and preprocess a single image for DINO.
    
    Args:
        image_path: Path to input image
        image_size: Size to resize image to (should be divisible by 14)
    
    Returns:
        (img_resized, img_normalized) - PIL image and normalized tensor
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Resize to be divisible by patch_size (14)
    new_size = (image_size, image_size)
    img_resized = img.resize(new_size, Image.BILINEAR)
    
    # Convert to tensor and normalize
    # Original purnasai/Dino_V2 uses: Normalize(mean=0.5, std=0.2)
    img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # Normalize with mean=0.5, std=0.2 (as in original implementation)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.2, 0.2, 0.2]).view(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    return img_resized, img_normalized, original_size


def save_visualization(img_resized, pca_colors, output_path: str, image_size: int = 518):
    """
    Save PCA visualization for a single image.
    
    Args:
        img_resized: Resized PIL image
        pca_colors: PCA colors (H', W', 3)
        output_path: Path to save visualization
        image_size: Image size
    """
    # Upsample PCA colors to image resolution
    pca_colors_tensor = torch.from_numpy(pca_colors).permute(2, 0, 1).unsqueeze(0)
    pca_colors_up = F.interpolate(
        pca_colors_tensor,
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    )
    pca_colors_up = pca_colors_up[0].permute(1, 2, 0).numpy()
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # PCA visualization
    axes[1].imshow(pca_colors_up)
    axes[1].set_title('DINO PCA (Combined)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # Also save just the PCA image
    pca_only_path = output_path.replace('.png', '_pca_only.png')
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_colors_up)
    plt.axis('off')
    plt.savefig(pca_only_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  Saved: {pca_only_path}")


def visualize_images_combined_pca(
    image_paths: list,
    model,
    output_dir: str,
    device: str = 'cuda',
    image_size: int = 518
):
    """
    Generate DINO PCA visualization for MULTIPLE images using combined PCA.
    
    This is the key function - it processes all images together:
    1. Extract features from all images
    2. Combine features and fit single PCA
    3. Apply same PCA to color all images consistently
    
    Args:
        image_paths: List of paths to input images
        model: DINOv2 model
        output_dir: Directory to save visualizations
        device: Device
        image_size: Size to resize images to
    """
    print(f"\n{'='*60}")
    print(f"Combined PCA Processing: {len(image_paths)} images")
    print(f"{'='*60}")
    
    # Step 1: Load and preprocess all images
    print("\nStep 1: Loading and preprocessing images...")
    images_resized = []
    images_normalized = []
    image_names = []
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found: {img_path}")
            continue
        
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  Loading: {img_path}")
        
        img_resized, img_normalized, original_size = preprocess_image(img_path, image_size)
        images_resized.append(img_resized)
        images_normalized.append(img_normalized)
        image_names.append(img_name)
        print(f"    Original: {original_size} -> Resized: {image_size}x{image_size}")
    
    if len(images_normalized) == 0:
        print("ERROR: No valid images found!")
        return
    
    # Step 2: Extract features from all images
    print(f"\nStep 2: Extracting DINO features from {len(images_normalized)} images...")
    features_list = []
    
    for i, img_normalized in enumerate(images_normalized):
        img_normalized = img_normalized.to(device)
        features = extract_features(model, img_normalized, device)  # (1, H', W', D)
        features = features[0].cpu().numpy()  # (H', W', D)
        features_list.append(features)
        print(f"  {image_names[i]}: features shape {features.shape}")
    
    # Step 3: Apply combined 2-step PCA
    print(f"\nStep 3: Applying combined 2-step PCA...")
    pca_colors_list = apply_two_step_pca_combined(features_list)
    
    # Step 4: Save visualizations
    print("Step 4: Saving visualizations...")
    for i, (img_resized, pca_colors, img_name) in enumerate(
        zip(images_resized, pca_colors_list, image_names)
    ):
        output_path = os.path.join(output_dir, f"{img_name}_pca.png")
        save_visualization(img_resized, pca_colors, output_path, image_size)
    
    # Also create a combined grid visualization
    n_images = len(images_resized)
    if n_images > 1:
        cols = min(n_images, 4)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 2, cols, figsize=(4 * cols, 8 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows * 2, cols)
        
        for i, (img_resized, pca_colors, img_name) in enumerate(
            zip(images_resized, pca_colors_list, image_names)
        ):
            row = (i // cols) * 2
            col = i % cols
            
            # Original
            axes[row, col].imshow(img_resized)
            axes[row, col].set_title(img_name)
            axes[row, col].axis('off')
            
            # PCA - upsample for display
            pca_tensor = torch.from_numpy(pca_colors).permute(2, 0, 1).unsqueeze(0)
            pca_up = F.interpolate(pca_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)
            pca_up = pca_up[0].permute(1, 2, 0).numpy()
            
            axes[row + 1, col].imshow(pca_up)
            axes[row + 1, col].set_title(f'{img_name} PCA')
            axes[row + 1, col].axis('off')
        
        # Hide unused axes
        for i in range(n_images, rows * cols):
            row = (i // cols) * 2
            col = i % cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
        
        plt.tight_layout()
        grid_path = os.path.join(output_dir, 'combined_pca_grid.png')
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved grid: {grid_path}")
    
    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='DINO PCA visualization for images (combined PCA)')
    parser.add_argument('--images', type=str, default='images/img1.jpg,images/img3.jpg,images/img5.jpg,images/img6.jpg',
                        help='Comma-separated list of image paths')
    parser.add_argument('--output_dir', type=str, default='outputs/image_pca',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='dinov2_vitl14',
                        help='DINOv2 model name (vitl14 recommended to match paper)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--image_size', type=int, default=518,
                        help='Image size (should be divisible by 14)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_dino_model(args.model, args.device)
    
    # Parse image paths
    image_paths = [p.strip() for p in args.images.split(',')]
    
    print(f"\nWill process {len(image_paths)} images with COMBINED PCA")
    print("(Features from all images are combined before fitting PCA)")
    print("(This produces consistent, symmetric coloring across all images)")
    
    # Process all images together with combined PCA
    visualize_images_combined_pca(
        image_paths,
        model,
        args.output_dir,
        device=args.device,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
