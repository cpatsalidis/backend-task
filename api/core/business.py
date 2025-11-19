"""Business logic layer for facial region processing."""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..cache import PerceptualCache
from ..models import CropSubmitRequest, CropSubmitResponse
from ..utils import (
    RequestValidator,
    base64_to_image,
    create_svg_overlay,
    extract_contours_from_masks,
    landmarks_to_numpy,
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
        logger.info("🚀 Starting crop submit processing")
        
        # Check cache first if available
        if self.cache:
            logger.info("🔍 Checking cache for similar image...")
            cached_result = self.cache.get_cached_result(request.image)
            if cached_result:
                logger.info("✅ Using cached result - skipping processing")
                return CropSubmitResponse(
                    svg=cached_result["svg"],
                    mask_contours=cached_result["mask_contours"]
                )
        
        # Step 1: Decode base64 images
        logger.info("🖼️  Decoding base64 images...")
        original_img = base64_to_image(request.image)
        segmentation_img = base64_to_image(request.segmentation_map)
        
        # Step 2: Convert landmarks to numpy array
        logger.info(f"📍 Processing [bold cyan]{len(request.landmarks)}[/bold cyan] landmarks...")
        landmarks_dict = [{"x": lm.x, "y": lm.y} for lm in request.landmarks]
        landmarks = landmarks_to_numpy(landmarks_dict)
        
        # Validate landmarks bounds
        img_height, img_width = original_img.shape[:2]
        RequestValidator.validate_landmarks_bounds(landmarks, img_width, img_height)
        
        # Step 3: Process image
        logger.info("🎨 Processing facial regions...")
        processed_img, region_masks = self.processor.process_image(
            original_img,
            segmentation_img,
            landmarks
        )
        
        # Step 4: Create SVG overlay
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
        logger.info("🔍 Extracting contours...")
        mask_contours = extract_contours_from_masks(region_masks, region_labels)
        
        logger.info(f"✅ Successfully processed image - Found [bold green]{len(mask_contours)}[/bold green] regions")
        
        # Store in cache if available
        if self.cache:
            try:
                self.cache.store_result(request.image, svg_base64, mask_contours)
                logger.info("💾 Result stored in cache")
            except Exception as e:
                logger.warning(f"⚠️  Failed to store in cache: {e}")
        
        # Return response
        return CropSubmitResponse(
            svg=svg_base64,
            mask_contours=mask_contours
        )

