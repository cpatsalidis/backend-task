"""Functions for extending facial regions using segmentation data."""

import cv2 as cv
import numpy as np
import config


def extend_forehead_with_segmentation(forehead_mask, segmentation_img, landmarks):
    """
    Extend the forehead mask using segmentation map.
    Creates a mask from segmentation above eyebrow level and ORs it with landmark mask.
    
    Args:
        forehead_mask: Original forehead mask from landmarks
        segmentation_img: Segmentation map (rotated)
        landmarks: Facial landmarks (rotated)
    
    Returns:
        Extended forehead mask
    """
    # Convert segmentation to grayscale and create head mask
    seg_gray = cv.cvtColor(segmentation_img, cv.COLOR_BGR2GRAY)
    _, head_mask = cv.threshold(seg_gray, 1, 255, cv.THRESH_BINARY_INV)
    
    # Find eyebrow level using landmarks
    eyebrow_y_coords = [landmarks[i][1] for i in config.EYEBROW_LANDMARKS if i < len(landmarks)]
    
    if eyebrow_y_coords:
        # Get the highest (minimum y) eyebrow point
        eyebrow_level = int(np.min(eyebrow_y_coords))
    else:
        # Fallback: use forehead mask top
        forehead_points = np.where(forehead_mask > 0)
        if len(forehead_points[0]) > 0:
            eyebrow_level = int(np.min(forehead_points[0]))
        else:
            eyebrow_level = forehead_mask.shape[0] // 3
    
    # Create a mask from segmentation that's only above eyebrow level
    segmentation_above_eyebrows = head_mask.copy()
    segmentation_above_eyebrows[eyebrow_level:, :] = 0  # Zero out everything below eyebrow level

    _, full_head_mask = cv.threshold(seg_gray, 0, 255, cv.THRESH_BINARY)
    segmentation_above_eyebrows = cv.bitwise_and(segmentation_above_eyebrows, full_head_mask)
    
    # OR the segmentation mask with the landmark-based forehead mask
    extended_mask = cv.bitwise_or(forehead_mask, segmentation_above_eyebrows)
    
    return extended_mask


def extend_lower_face_with_segmentation(lower_face_mask, segmentation_img, landmarks):
    """
    Extend the lower face/chin mask using segmentation map.
    Creates a mask from segmentation below lip level and ORs it with landmark mask.
    
    Args:
        lower_face_mask: Original lower face mask from landmarks
        segmentation_img: Segmentation map (rotated)
        landmarks: Facial landmarks (rotated)
    
    Returns:
        Extended lower face mask
    """
    # Convert segmentation to grayscale and create head mask
    seg_gray = cv.cvtColor(segmentation_img, cv.COLOR_BGR2GRAY)
    _, head_mask = cv.threshold(seg_gray, 1, 255, cv.THRESH_BINARY_INV)
    
    # Find lip level using landmarks
    lip_y_coords = [landmarks[i][1] for i in config.LIP_LANDMARKS if i < len(landmarks)]
    
    if lip_y_coords:
        # Get the lowest (maximum y) lip point
        lip_level = int(np.min(lip_y_coords))
    else:
        # Fallback: use lower_face mask bottom
        lower_face_points = np.where(lower_face_mask > 0)
        if len(lower_face_points[0]) > 0:
            lip_level = int(np.min(lower_face_points[0]))
        else:
            lip_level = 2 * lower_face_mask.shape[0] // 3
    
    # Create a mask from segmentation that's only below lip level
    segmentation_below_lips = head_mask.copy()
    segmentation_below_lips[:lip_level, :] = 0  # Zero out everything above lip level

    _, full_head_mask = cv.threshold(seg_gray, 0, 255, cv.THRESH_BINARY)
    segmentation_below_lips = cv.bitwise_and(segmentation_below_lips, full_head_mask)
    
    # OR the segmentation mask with the landmark-based lower face mask
    extended_mask = cv.bitwise_or(lower_face_mask, segmentation_below_lips)
    
    return extended_mask


def detect_ears_from_segmentation(segmentation_img, landmarks, offset=None):
    """
    Detect ear regions using specific landmarks and segmentation mask.
    Creates ear masks with a gap between face boundary and ear region.
    
    Args:
        segmentation_img: Segmentation map (rotated)
        landmarks: Facial landmarks (rotated)
        offset: Pixel gap between face boundary and ear region
    
    Returns:
        Mask containing detected ear regions
    """
    if offset is None:
        offset = config.EAR_OFFSET
    
    # Convert segmentation to grayscale and create full head mask
    seg_gray = cv.cvtColor(segmentation_img, cv.COLOR_BGR2GRAY)
    _, head_mask = cv.threshold(seg_gray, 0, 255, cv.THRESH_BINARY)
    
    # Get left and right side landmark positions
    left_side_points = np.array([landmarks[i] for i in config.LEFT_SIDE_LANDMARKS], dtype=np.int32)
    right_side_points = np.array([landmarks[i] for i in config.RIGHT_SIDE_LANDMARKS], dtype=np.int32)
    
    # Get boundaries
    left_x = int(np.min(left_side_points[:, 0]))
    left_top_y = int(np.min(left_side_points[:, 1]))
    left_bottom_y = int(np.max(left_side_points[:, 1]))
    
    right_x = int(np.max(right_side_points[:, 0]))
    right_top_y = int(np.min(right_side_points[:, 1]))
    right_bottom_y = int(np.max(right_side_points[:, 1]))
    
    img_height, img_width = segmentation_img.shape[:2]
    
    # Create block masks for ear regions with offset
    ear_mask = np.zeros_like(head_mask)
    
    # Left ear block: from left edge (0) to (left_x - offset), with gap from face
    left_ear_boundary = max(0, left_x - offset)
    left_ear_block = np.zeros_like(head_mask)
    left_ear_block[left_top_y:left_bottom_y, 0:left_ear_boundary] = 255
    
    # Right ear block: from (right_x + offset) to right edge, with gap from face
    right_ear_boundary = min(img_width, right_x + offset)
    right_ear_block = np.zeros_like(head_mask)
    right_ear_block[right_top_y:right_bottom_y, right_ear_boundary:img_width] = 255
    
    # AND with head mask to get only actual head pixels in ear regions
    left_ear = cv.bitwise_and(left_ear_block, head_mask)
    right_ear = cv.bitwise_and(right_ear_block, head_mask)
    
    # Combine left and right ears
    ear_mask = cv.bitwise_or(left_ear, right_ear)
    
    return ear_mask

