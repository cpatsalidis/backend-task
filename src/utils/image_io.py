"""Functions for loading and saving images and landmarks."""

import cv2 as cv
import numpy as np
import ast


def load_images(original_path, segmentation_path):
    """
    Load original image and segmentation map.
    
    Args:
        original_path: Path to original image
        segmentation_path: Path to segmentation map
    
    Returns:
        Tuple of (original_img, segmented_img)
    """
    original_img = cv.imread(original_path)
    segmented_img = cv.imread(segmentation_path)
    
    if original_img is None:
        raise FileNotFoundError(f"Could not load original image from {original_path}")
    if segmented_img is None:
        raise FileNotFoundError(f"Could not load segmentation map from {segmentation_path}")
    
    # Resize segmentation map to match original image
    segmented_img = cv.resize(segmented_img, (original_img.shape[1], original_img.shape[0]))
    
    return original_img, segmented_img


def load_landmarks(landmarks_path):
    """
    Load facial landmarks from file.
    
    Args:
        landmarks_path: Path to landmarks file
    
    Returns:
        Numpy array of landmark points (N x 2)
    """
    with open(landmarks_path, 'r') as f:
        landmarks_data = ast.literal_eval(f.read())
    
    landmarks = landmarks_data['landmarks'][0]
    points = np.array([[int(lm['x']), int(lm['y'])] for lm in landmarks], dtype=np.int32)
    
    return points


def save_result(image, output_path):
    """
    Save processed result image.
    
    Args:
        image: Result image to save
        output_path: Path to save the image
    """
    cv.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")

