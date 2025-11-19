"""Functions for drawing labels on images."""

import cv2 as cv
import numpy as np
from typing import Dict


def draw_region_labels(image: np.ndarray, region_masks: Dict[str, np.ndarray], 
                       region_labels: Dict[str, str], 
                       font_scale: float = 1.5, thickness: int = 3) -> None:
    """
    Draw numbered labels at the centroid of each region.
    
    Args:
        image: Image to draw labels on (modified in place)
        region_masks: Dictionary of region masks
        region_labels: Dictionary mapping region names to label text
        font_scale: Font scale for text
        thickness: Text thickness
    """
    for region_name, mask in region_masks.items():
        if region_name not in region_labels:
            continue
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Get the largest contour for centroid calculation
        largest_contour = max(contours, key=cv.contourArea)
        M = cv.moments(largest_contour)
        
        if M["m00"] != 0:
            # Calculate centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get label text
            label = region_labels[region_name]
            
            # Set up text properties
            font = cv.FONT_HERSHEY_SIMPLEX
            
            # Get text size for centering
            (text_width, text_height), baseline = cv.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Center the text at the centroid
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            # Draw text with thick black outline (for visibility)
            cv.putText(image, label, (text_x, text_y), font, font_scale,
                      (0, 0, 0), thickness + 4, cv.LINE_AA)
            
            # Draw white text on top
            cv.putText(image, label, (text_x, text_y), font, font_scale,
                      (255, 255, 255), thickness, cv.LINE_AA)

