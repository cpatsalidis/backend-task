"""Validation logic for API requests."""

from fastapi import HTTPException, status
from typing import List
import numpy as np

from .models import CropSubmitRequest, Landmark


class RequestValidator:
    """Validates API request data."""
    
    MIN_LANDMARKS = 100
    
    @staticmethod
    def validate_landmarks_count(landmarks: List[Landmark]) -> None:
        """
        Validate that sufficient landmarks are provided.
        
        Args:
            landmarks: List of landmark points
            
        Raises:
            HTTPException: If validation fails
        """
        if len(landmarks) < RequestValidator.MIN_LANDMARKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient landmarks: expected at least {RequestValidator.MIN_LANDMARKS}, got {len(landmarks)}"
            )
    
    @staticmethod
    def validate_landmarks_bounds(
        landmarks: np.ndarray,
        img_width: int,
        img_height: int
    ) -> None:
        """
        Validate that landmarks are within image bounds.
        
        Args:
            landmarks: Array of landmark points
            img_width: Image width
            img_height: Image height
            
        Note:
            This logs a warning but doesn't raise an exception, as some
            landmarks may be slightly outside bounds due to rounding.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if landmarks[:, 0].max() > img_width or landmarks[:, 1].max() > img_height:
            logger.warning("Some landmarks are outside image bounds")

