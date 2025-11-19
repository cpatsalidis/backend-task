"""Functions for blending facial regions onto images."""

import cv2 as cv
import numpy as np
import config


def blend_regions(original_img, region_masks):
    """
    Blend all facial region overlays onto the original image.
    
    Args:
        original_img: Original image
        region_masks: Dictionary of {region_name: mask}
    
    Returns:
        Blended result image
    """
    overlay = original_img.copy()
    
    # Apply each region color to the overlay
    for region_name, mask in region_masks.items():
        color = config.FACIAL_REGIONS[region_name]['color']
        overlay[mask > 0] = color
    
    # Blend overlay with original image
    result = cv.addWeighted(overlay, config.ALPHA_BLEND, original_img, 1 - config.ALPHA_BLEND, 0)
    
    return result


def apply_extended_region(result_img, original_img, extended_mask, color, alpha=None):
    """
    Apply an extended region overlay to the result image.
    
    Args:
        result_img: Current result image (will be modified)
        original_img: Original image (for blending)
        extended_mask: Mask for the extended region
        color: BGR color for the region
        alpha: Alpha blending factor
    
    Returns:
        Updated result image
    """
    if alpha is None:
        alpha = config.ALPHA_BLEND
    
    # Create overlay with extended region
    overlay_extended = original_img.copy()
    overlay_extended[extended_mask > 0] = color
    
    # Blend the extended region with the original image
    region_blended = cv.addWeighted(overlay_extended, alpha, original_img, 1 - alpha, 0)
    
    # Apply only the extended region area to result
    result_img[extended_mask > 0] = region_blended[extended_mask > 0]
    
    return result_img

