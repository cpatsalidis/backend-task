"""Functions for drawing contours on images."""

import cv2 as cv
import numpy as np
import config


def draw_dashed_contour(img, contour, color, thickness=None, dash_length=None, gap_length=None):
    """
    Draw a dashed contour on an image.
    
    Args:
        img: Image to draw on
        contour: Contour points
        color: Line color (BGR)
        thickness: Line thickness
        dash_length: Length of each dash in pixels
        gap_length: Length of gap between dashes in pixels
    """
    if thickness is None:
        thickness = config.CONTOUR_THICKNESS
    if dash_length is None:
        dash_length = config.DASH_LENGTH
    if gap_length is None:
        gap_length = config.GAP_LENGTH
    
    # Flatten contour to list of points
    points = contour.reshape(-1, 2)
    
    # Calculate cumulative distances along the contour
    total_length = 0
    distances = [0]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i-1])
        total_length += dist
        distances.append(total_length)
    
    # Close the contour by adding distance back to start
    dist = np.linalg.norm(points[0] - points[-1])
    total_length += dist
    
    # Draw dashed line
    current_dist = 0
    pattern_length = dash_length + gap_length
    
    while current_dist < total_length:
        # Find start point
        start_dist = current_dist
        end_dist = min(current_dist + dash_length, total_length)
        
        # Find points at start and end distances
        start_pt = _get_point_at_distance(points, distances, start_dist, total_length)
        end_pt = _get_point_at_distance(points, distances, end_dist, total_length)
        
        # Draw dash
        cv.line(img, tuple(start_pt.astype(int)), tuple(end_pt.astype(int)), color, thickness, cv.LINE_AA)
        
        # Move to next dash
        current_dist += pattern_length


def _get_point_at_distance(points, distances, target_dist, total_length):
    """
    Get point on contour at specified distance.
    
    Args:
        points: Array of contour points
        distances: Cumulative distances at each point
        target_dist: Target distance along contour
        total_length: Total contour length
    
    Returns:
        Interpolated point at target distance
    """
    # Handle wrap-around for closed contours
    if target_dist >= distances[-1]:
        # Beyond last point, interpolate to first point
        remaining = target_dist - distances[-1]
        direction = points[0] - points[-1]
        dist = np.linalg.norm(direction)
        if dist > 0:
            return points[-1] + (direction / dist) * remaining
        return points[0]
    
    # Find segment containing target distance
    for i in range(len(distances) - 1):
        if distances[i] <= target_dist <= distances[i + 1]:
            # Interpolate between points[i] and points[i+1]
            segment_length = distances[i + 1] - distances[i]
            if segment_length > 0:
                t = (target_dist - distances[i]) / segment_length
                return points[i] + t * (points[i + 1] - points[i])
            return points[i]
    
    return points[-1]


def draw_all_contours(img, region_masks, color=None):
    """
    Draw dashed contours for all facial regions.
    
    Args:
        img: Image to draw on
        region_masks: Dictionary of {region_name: mask}
        color: Contour color (BGR)
    """
    if color is None:
        color = config.CONTOUR_COLOR
    
    for region_name, mask in region_masks.items():
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if contours:
            for contour in contours:
                draw_dashed_contour(img, contour, color)

