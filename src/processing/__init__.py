"""Processing functions for facial region manipulation."""

from .mask_creation import create_smooth_mask, create_region_masks
from .region_extension import (
    extend_forehead_with_segmentation,
    extend_lower_face_with_segmentation,
    detect_ears_from_segmentation
)
from .blending import blend_regions, apply_extended_region

__all__ = [
    'create_smooth_mask',
    'create_region_masks',
    'extend_forehead_with_segmentation',
    'extend_lower_face_with_segmentation',
    'detect_ears_from_segmentation',
    'blend_regions',
    'apply_extended_region',
]

