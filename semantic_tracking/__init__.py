# Semantic tracking utilities for MeshUp
from .vertex_color_tracking import VertexColorTracker, initialize_semantic_colors, assign_part_colors
from .correspondence_export import export_mesh_with_colors, export_correspondence_map

# DINOv2 correspondence loss (optional, may require additional dependencies)
try:
    from .dino_correspondence_loss import DINOCorrespondenceLoss, create_dino_correspondence_loss
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    DINOCorrespondenceLoss = None
    create_dino_correspondence_loss = None

# Cross-Attention Semantic Guidance (optional)
try:
    from .cross_attention_guidance import CrossAttentionGuidance, create_cross_attention_guidance
    CROSS_ATTN_AVAILABLE = True
except ImportError:
    CROSS_ATTN_AVAILABLE = False
    CrossAttentionGuidance = None
    create_cross_attention_guidance = None

# DINO PCA Visualization (optional)
try:
    from .dino_pca_visualization import DINOPCAColorizer, create_dino_pca_visualization
    DINO_PCA_AVAILABLE = True
except ImportError:
    DINO_PCA_AVAILABLE = False
    DINOPCAColorizer = None
    create_dino_pca_visualization = None

# Target Mesh DINO Guidance (for mesh-to-mesh transformation)
try:
    from .target_mesh_dino_guidance import TargetMeshDINOGuidance, create_target_mesh_guidance
    TARGET_MESH_GUIDANCE_AVAILABLE = True
except ImportError:
    TARGET_MESH_GUIDANCE_AVAILABLE = False
    TargetMeshDINOGuidance = None
    create_target_mesh_guidance = None

# Target Mesh DINO Guidance V2 (viewpoint-aligned version)
try:
    from .target_mesh_dino_guidance_v2 import TargetMeshDINOGuidanceV2, create_target_mesh_guidance_v2
    TARGET_MESH_GUIDANCE_V2_AVAILABLE = True
except ImportError:
    TARGET_MESH_GUIDANCE_V2_AVAILABLE = False
    TargetMeshDINOGuidanceV2 = None
    create_target_mesh_guidance_v2 = None

# FID Evaluation
try:
    from .fid_evaluation import FIDEvaluator, compute_mesh_fid
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    FIDEvaluator = None
    compute_mesh_fid = None

# Multi-angle Visualization
try:
    from .multiangle_visualization import MultiAnglePCAVisualizer, create_multiangle_visualizer
    MULTIANGLE_VIS_AVAILABLE = True
except ImportError:
    MULTIANGLE_VIS_AVAILABLE = False
    MultiAnglePCAVisualizer = None
    create_multiangle_visualizer = None

__all__ = [
    'VertexColorTracker',
    'initialize_semantic_colors',
    'assign_part_colors',
    'export_mesh_with_colors',
    'export_correspondence_map',
    'DINOCorrespondenceLoss',
    'create_dino_correspondence_loss',
    'DINO_AVAILABLE',
    'CrossAttentionGuidance',
    'create_cross_attention_guidance',
    'CROSS_ATTN_AVAILABLE',
    'DINOPCAColorizer',
    'create_dino_pca_visualization',
    'DINO_PCA_AVAILABLE',
    'TargetMeshDINOGuidance',
    'create_target_mesh_guidance',
    'TARGET_MESH_GUIDANCE_AVAILABLE',
    'TargetMeshDINOGuidanceV2',
    'create_target_mesh_guidance_v2',
    'TARGET_MESH_GUIDANCE_V2_AVAILABLE',
    'FIDEvaluator',
    'compute_mesh_fid',
    'FID_AVAILABLE',
    'MultiAnglePCAVisualizer',
    'create_multiangle_visualizer',
    'MULTIANGLE_VIS_AVAILABLE',
]
