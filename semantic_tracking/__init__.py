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
]
