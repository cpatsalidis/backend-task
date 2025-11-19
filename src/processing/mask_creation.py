"""Functions for creating facial region masks."""

import cv2 as cv
import numpy as np
import config


def create_smooth_mask(img_shape, region_points):
    """
    Create smooth mask from region points.
    
    Args:
        img_shape: Shape of the image
        region_points: Points defining the region
    
    Returns:
        Binary mask (0 or 255)
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    if len(region_points) < 3:
        return mask
    
    # Fill polygon from landmarks
    cv.fillPoly(mask, [region_points], 255)
    
    # Smooth the mask edges
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.GaussianBlur(mask, config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_SIGMA)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    
    return mask


def create_region_masks(img_shape, landmarks):
    """
    Create masks for all facial regions.
    
    Args:
        img_shape: Shape of the image
        landmarks: Array of facial landmark points
    
    Returns:
        Dictionary of {region_name: mask}
    """
    region_masks = {}
    
    for region_name, region_info in config.FACIAL_REGIONS.items():
        indices = region_info['indices']
        region_points = np.array([landmarks[i] for i in indices], dtype=np.int32)
        mask = create_smooth_mask(img_shape, region_points)
        region_masks[region_name] = mask
    
    return region_masks

