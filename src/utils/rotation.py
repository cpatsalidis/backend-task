"""Functions for detecting and correcting image rotation."""

import cv2 as cv
import numpy as np


def detect_face_tilt(landmarks):
    """
    Detect the tilt angle of the face using eye landmarks.
    
    Args:
        landmarks: Array of facial landmark points
    
    Returns:
        Angle in degrees (positive = clockwise tilt)
    """
    # MediaPipe landmark indices for eyes:
    # Left eye outer corner: 33
    # Right eye outer corner: 263
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    # Calculate angle between eyes
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    
    return angle


def rotate_image(image, angle, center=None):
    """
    Rotate image by the specified angle around center point.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        center: Center of rotation (default: image center)
    
    Returns:
        Rotated image
    """
    (height, width) = image.shape[:2]
    
    if center is None:
        center = (width // 2, height // 2)
    
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    dimensions = (width, height)
    
    return cv.warpAffine(image, rotation_matrix, dimensions)


def rotate_landmarks(landmarks, angle, image_shape):
    """
    Rotate landmarks to match rotated image.
    
    Args:
        landmarks: Array of landmark points
        angle: Rotation angle in degrees
        image_shape: Original image shape (height, width)
    
    Returns:
        Rotated landmarks
    """
    height, width = image_shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to each landmark
    rotated_landmarks = []
    for point in landmarks:
        # Convert to homogeneous coordinates
        point_homogeneous = np.array([point[0], point[1], 1.0])
        # Apply rotation
        rotated_point = rotation_matrix.dot(point_homogeneous)
        rotated_landmarks.append(rotated_point[:2])
    
    return np.array(rotated_landmarks, dtype=np.int32)

