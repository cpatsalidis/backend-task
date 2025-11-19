"""Utility functions for API operations."""

import base64
from typing import List, Dict, Tuple

import cv2 as cv
import numpy as np


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        OpenCV image (numpy array)
    
    Raises:
        ValueError: If base64 string is invalid or cannot be decoded
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image from base64 string")
        
        return image
    
    except Exception as e:
        raise ValueError(f"Failed to convert base64 to image: {str(e)}")


def image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert OpenCV image to base64 string.
    
    Args:
        image: OpenCV image (numpy array)
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64 encoded image string
    """
    try:
        # Encode image to specified format
        success, buffer = cv.imencode(f'.{format.lower()}', image)
        
        if not success:
            raise ValueError(f"Could not encode image to {format}")
        
        # Convert to base64
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        return base64_string
    
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")


def create_svg_overlay(image: np.ndarray, region_masks: Dict[str, np.ndarray], 
                       region_colors: Dict[str, Tuple[int, int, int]], 
                       region_labels: Dict[str, str] = None,
                       alpha: float = 0.4) -> str:
    """
    Create SVG representation of image with highlighted regions.
    
    Args:
        image: Original image
        region_masks: Dictionary of region masks
        region_colors: Dictionary of region colors (BGR format)
        region_labels: Dictionary mapping region names to label text (e.g., '1', '2')
        alpha: Transparency level
    
    Returns:
        SVG string with highlighted regions
    """
    height, width = image.shape[:2]
    
    # Start SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<defs>',
        f'  <style>',
        f'    .region {{ fill-opacity: {alpha}; stroke-width: 1; stroke-linejoin: round; stroke-dasharray: 3 3; }}',
        f'    .label {{ font-family: Arial, sans-serif; font-size: 48px; font-weight: bold; fill: white; ',
        f'             stroke: black; stroke-width: 4; paint-order: stroke; text-anchor: middle; dominant-baseline: middle; }}',
        f'  </style>',
        f'</defs>'
    ]
    
    # Add base image as background
    base64_img = image_to_base64(image, 'PNG')
    svg_parts.append(f'<image href="data:image/png;base64,{base64_img}" width="{width}" height="{height}"/>')
    
    # Add each region as a path with filled polygon
    region_centroids = {}
    for region_name, mask in region_masks.items():
        if region_name not in region_colors:
            continue
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Convert BGR to RGB
        bgr_color = region_colors[region_name]
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        color_hex = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        
        # Get the largest contour for centroid calculation
        largest_contour = max(contours, key=cv.contourArea)
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            region_centroids[region_name] = (cx, cy)
        
        # Create path for each contour
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Create SVG path
            points = contour.reshape(-1, 2)
            path_data = f"M {points[0][0]},{points[0][1]}"
            for point in points[1:]:
                path_data += f" L {point[0]},{point[1]}"
            path_data += " Z"
            
            svg_parts.append(
                f'<path class="region" d="{path_data}" fill="{color_hex}" '
                f'stroke="rgb(58,36,59)" data-region="{region_name}"/>'
            )
    
    # Add labels
    if region_labels:
        for region_name, (cx, cy) in region_centroids.items():
            if region_name in region_labels:
                label = region_labels[region_name]
                svg_parts.append(
                    f'<text class="label" x="{cx}" y="{cy}">{label}</text>'
                )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def svg_to_base64(svg_string: str) -> str:
    """
    Convert SVG string to base64.
    
    Args:
        svg_string: SVG content as string
    
    Returns:
        Base64 encoded SVG string
    """
    svg_bytes = svg_string.encode('utf-8')
    base64_string = base64.b64encode(svg_bytes).decode('utf-8')
    return base64_string


def extract_contours_from_masks(region_masks: Dict[str, np.ndarray], 
                                region_labels: Dict[str, str] = None) -> Dict[str, List[List[float]]]:
    """
    Extract contour points from region masks.
    
    Args:
        region_masks: Dictionary of region masks
        region_labels: Dictionary mapping region names to label numbers (e.g., '1', '2')
    
    Returns:
        Dictionary mapping label numbers to contour points
    """
    mask_contours = {}
    
    for region_name, mask in region_masks.items():
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Use label number as key if available, otherwise use region name
        key = region_labels.get(region_name, region_name) if region_labels else region_name
        
        if not contours:
            mask_contours[key] = []
            continue
        
        # Get the largest contour
        largest_contour = max(contours, key=cv.contourArea)
        
        # Convert to list of points
        points = largest_contour.reshape(-1, 2).tolist()
        
        # Convert to float for JSON serialization
        points = [[float(x), float(y)] for x, y in points]
        
        mask_contours[key] = points
    
    return mask_contours


def landmarks_to_numpy(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Convert landmark list to numpy array.
    
    Args:
        landmarks: List of landmark dictionaries with 'x' and 'y' keys
    
    Returns:
        Numpy array of shape (N, 2)
    """
    points = np.array([[lm['x'], lm['y']] for lm in landmarks], dtype=np.int32)
    return points

