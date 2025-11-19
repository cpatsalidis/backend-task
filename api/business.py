"""Business logic layer for facial region processing."""

import logging
from typing import Dict, Tuple
import numpy as np

from .service import FacialRegionProcessor
from .utils import (
    base64_to_image,
    landmarks_to_numpy,
    create_svg_overlay,
    svg_to_base64,
    extract_contours_from_masks
)
from .models import CropSubmitRequest, CropSubmitResponse
from .validators import RequestValidator

logger = logging.getLogger(__name__)


class FacialProcessingService:
    """Business logic service for processing facial images."""
    
    def __init__(self, processor: FacialRegionProcessor):
        """
        Initialize service with processor dependency.
        
        Args:
            processor: FacialRegionProcessor instance (injected)
        """
        self.processor = processor
    
    def process_crop_submit(
        self,
        request: CropSubmitRequest
    ) -> CropSubmitResponse:
        """
        Process crop submit request - main business logic.
        
        This method orchestrates the entire processing pipeline:
        1. Decode images
        2. Convert landmarks
        3. Process facial regions
        4. Generate SVG overlay
        5. Extract contours
        
        Args:
            request: Crop submit request with image, landmarks, and segmentation
            
        Returns:
            CropSubmitResponse with SVG and mask contours
            
        Raises:
            ValueError: If image decoding fails
            RuntimeError: If processing fails
        """
        logger.info("Starting crop submit processing")
        
        # Step 1: Decode base64 images
        logger.info("Decoding base64 images...")
        original_img = base64_to_image(request.image)
        segmentation_img = base64_to_image(request.segmentation_map)
        
        # Step 2: Convert landmarks to numpy array
        logger.info(f"Processing {len(request.landmarks)} landmarks...")
        landmarks_dict = [{"x": lm.x, "y": lm.y} for lm in request.landmarks]
        landmarks = landmarks_to_numpy(landmarks_dict)
        
        # Validate landmarks bounds
        img_height, img_width = original_img.shape[:2]
        RequestValidator.validate_landmarks_bounds(landmarks, img_width, img_height)
        
        # Step 3: Process image
        logger.info("Processing facial regions...")
        processed_img, region_masks = self.processor.process_image(
            original_img,
            segmentation_img,
            landmarks
        )
        
        # Step 4: Create SVG overlay
        logger.info("Generating SVG overlay...")
        region_colors = self.processor.get_region_colors()
        region_labels = self.processor.get_region_labels()
        svg_string = create_svg_overlay(
            processed_img,
            region_masks,
            region_colors,
            region_labels=region_labels,
            alpha=self.processor.alpha_blend
        )
        svg_base64 = svg_to_base64(svg_string)
        
        # Step 5: Extract contours
        logger.info("Extracting contours...")
        mask_contours = extract_contours_from_masks(region_masks, region_labels)
        
        logger.info(f"Successfully processed image - Found {len(mask_contours)} regions")
        
        # Return response
        return CropSubmitResponse(
            svg=svg_base64,
            mask_contours=mask_contours
        )

