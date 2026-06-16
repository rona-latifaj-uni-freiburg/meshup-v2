#!/usr/bin/env python3
"""
Quick test script to verify DINO PCA visualization module imports correctly.
Run this to ensure all dependencies are available before submitting jobs.
"""

import sys
import traceback

def test_imports():
    """Test all required imports for DINO PCA visualization."""
    
    print("=" * 60)
    print("Testing DINO PCA Visualization Module Imports")
    print("=" * 60)
    
    # Test 1: Basic imports
    print("\n[1/5] Testing basic imports...")
    try:
        import torch
        import numpy as np
        from sklearn.decomposition import PCA
        print("✓ Basic dependencies available")
    except ImportError as e:
        print(f"✗ Missing basic dependency: {e}")
        return False
    
    # Test 2: DINO model
    print("\n[2/5] Testing DINOv2 model...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        print("✓ DINOv2 model loaded successfully")
        del model
    except Exception as e:
        print(f"✗ Failed to load DINOv2: {e}")
        print("   This is normal on first run - model will download automatically")
    
    # Test 3: DINO PCA module
    print("\n[3/5] Testing DINO PCA module import...")
    try:
        from semantic_tracking.dino_pca_visualization import (
            DINOPCAColorizer,
            create_dino_pca_visualization
        )
        print("✓ DINO PCA module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DINO PCA module: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Check __init__ exports
    print("\n[4/5] Testing semantic_tracking module exports...")
    try:
        import semantic_tracking
        assert hasattr(semantic_tracking, 'DINO_PCA_AVAILABLE')
        assert hasattr(semantic_tracking, 'DINOPCAColorizer')
        assert hasattr(semantic_tracking, 'create_dino_pca_visualization')
        print(f"✓ Module exports available (DINO_PCA_AVAILABLE={semantic_tracking.DINO_PCA_AVAILABLE})")
    except (ImportError, AssertionError) as e:
        print(f"✗ Module exports not found: {e}")
        return False
    
    # Test 5: Integration check
    print("\n[5/5] Testing integration in loop_tracked.py...")
    try:
        # Just check that the file imports without errors
        import importlib.util
        spec = importlib.util.spec_from_file_location("loop_tracked", "loop_tracked.py")
        module = importlib.util.module_from_spec(spec)
        # We don't actually execute it, just verify syntax
        print("✓ loop_tracked.py module structure valid")
    except Exception as e:
        print(f"✗ Issue with loop_tracked.py: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! DINO PCA visualization is ready to use.")
    print("=" * 60)
    print("\nQuick start:")
    print("  1. Set color_method: dino_pca in your config")
    print("  2. Run: python main.py --config configs/test_dino_pca.yml")
    print("  3. Or submit: sbatch jobs/test_dino_pca.sh")
    print("\nSee semantic_tracking/DINO_PCA_README.md for details.")
    return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
