"""Functions for cropping images to head regions."""

import cv2 as cv
import numpy as np


def calculate_head_crop_box(segmentation_img, padding_factor=0.03):
    """
    Calculate crop box based on segmentation map to include the full head.
    
    Args:
        segmentation_img: Segmentation map image
        padding_factor: Factor to add padding around head (default 0.03 = 3%)
    
    Returns:
        Tuple of (x, y, width, height) for the crop box
    """
    # Convert to grayscale
    seg_gray = cv.cvtColor(segmentation_img, cv.COLOR_BGR2GRAY)
    
    # Threshold to get the head region (anything non-black is head)
    _, head_mask = cv.threshold(seg_gray, 0, 255, cv.THRESH_BINARY)
    
    # Find contours of the head
    contours, _ = cv.findContours(head_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to full image if no contours found
        return 0, 0, segmentation_img.shape[1], segmentation_img.shape[0]
    
    # Get bounding rectangle of the largest contour (the head)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    
    # Calculate padding based on head size
    max_dimension = max(w, h)
    padding = int(max_dimension * padding_factor)
    
    # Apply padding with boundary checks
    img_height, img_width = segmentation_img.shape[:2]
    crop_x = max(0, x - padding)
    crop_y = max(0, y - padding)
    crop_x2 = min(img_width, x + w + padding)
    crop_y2 = min(img_height, y + h + padding)
    
    # Calculate final dimensions
    crop_width = crop_x2 - crop_x
    crop_height = crop_y2 - crop_y
    
    return crop_x, crop_y, crop_width, crop_height


def apply_crop(image, crop_box):
    """
    Apply crop to image using the calculated crop box.
    
    Args:
        image: Input image
        crop_box: Tuple of (x, y, width, height)
    
    Returns:
        Cropped image
    """
    x, y, w, h = crop_box
    return image[y:y+h, x:x+w].copy()

