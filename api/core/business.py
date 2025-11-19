"""Business logic layer for facial region processing."""

import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from ..cache import PerceptualCache
from ..config import LOAD_TESTING_MODE
from ..models import CropSubmitRequest, CropSubmitResponse
from ..utils import (
    RequestValidator,
    base64_to_image,
    create_svg_overlay,
    extract_contours_from_masks,
    svg_to_base64,
)
from .service import FacialRegionProcessor

logger = logging.getLogger(__name__)


class FacialProcessingService:
    """Business logic service for processing facial images."""
    
    def __init__(
        self,
        processor: FacialRegionProcessor,
        db: Optional[Session] = None
    ):
        """
        Initialize service with processor dependency.
        
        Args:
            processor: FacialRegionProcessor instance (injected)
            db: Optional database session for caching
        """
        self.processor = processor
        self.cache = PerceptualCache(db) if db else None
    
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
        if not LOAD_TESTING_MODE:
            logger.info("🚀 Starting crop submit processing")
        
        # Check cache first if available
        # OPTIMIZATION: Cache stores decoded image to avoid re-decoding
        decoded_images = {}
        if self.cache:
            if not LOAD_TESTING_MODE:
                logger.info("🔍 Checking cache for similar image...")
            # Pass a callback to cache to store decoded image
            cached_result = self.cache.get_cached_result(
                request.image,
                decoded_image_callback=lambda img: decoded_images.update({'original': img})
            )
            if cached_result:
                if not LOAD_TESTING_MODE:
                    logger.info("✅ Using cached result - skipping processing")
                return CropSubmitResponse(
                    svg=cached_result["svg"],
                    mask_contours=cached_result["mask_contours"]
                )
        
        # Step 1: Decode base64 images
        # OPTIMIZATION: Reuse decoded image from cache lookup if available
        if not LOAD_TESTING_MODE:
            logger.info("🖼️  Decoding base64 images...")
        if 'original' in decoded_images:
            original_img = decoded_images['original']
        else:
            original_img = base64_to_image(request.image)
        segmentation_img = base64_to_image(request.segmentation_map)
        
        # Step 2: Convert landmarks to numpy array
        # OPTIMIZATION: Direct conversion from Pydantic models (skip dict intermediate)
        if not LOAD_TESTING_MODE:
            logger.info(f"📍 Processing [bold cyan]{len(request.landmarks)}[/bold cyan] landmarks...")
        landmarks = np.array([[lm.x, lm.y] for lm in request.landmarks], dtype=np.int32)
        
        # Get image dimensions (needed for processing)
        img_height, img_width = original_img.shape[:2]
        
        # Validate landmarks bounds (skip in load testing mode for performance)
        if not LOAD_TESTING_MODE:
            RequestValidator.validate_landmarks_bounds(landmarks, img_width, img_height)
        
        # Step 3: Process image
        if not LOAD_TESTING_MODE:
            logger.info("🎨 Processing facial regions...")
        processed_img, region_masks = self.processor.process_image(
            original_img,
            segmentation_img,
            landmarks
        )
        
        # Step 4: Create SVG overlay
        if not LOAD_TESTING_MODE:
            logger.info("📐 Generating SVG overlay...")
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
        if not LOAD_TESTING_MODE:
            logger.info("🔍 Extracting contours...")
        mask_contours = extract_contours_from_masks(region_masks, region_labels)
        
        if not LOAD_TESTING_MODE:
            logger.info(f"✅ Successfully processed image - Found [bold green]{len(mask_contours)}[/bold green] regions")
        
        # Store in cache if available
        if self.cache:
            try:
                self.cache.store_result(request.image, svg_base64, mask_contours)
                if not LOAD_TESTING_MODE:
                    logger.info("💾 Result stored in cache")
            except Exception as e:
                # Only log cache errors if not in load testing mode
                if not LOAD_TESTING_MODE:
                    logger.warning(f"⚠️  Failed to store in cache: {e}")
        
        # Return response
        return CropSubmitResponse(
            svg=svg_base64,
            mask_contours=mask_contours
        )

