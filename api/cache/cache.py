"""Perceptual cache system for mask contours."""

import hashlib
import logging
from typing import Optional, Dict, Any

import imagehash
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from ..database.db_models import MaskContourCache
from ..utils.utils import base64_to_image

logger = logging.getLogger(__name__)


class PerceptualCache:
    """Perceptual cache for mask contours using image hashing."""
    
    def __init__(self, db: Session):
        """
        Initialize perceptual cache.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def _compute_perceptual_hash(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash for an image.
        
        Uses average hash (aHash) which is good for detecting similar images
        even with minor variations.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Perceptual hash string
        """
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        if len(image.shape) == 3:
            image_rgb = np.flip(image, axis=2)  # BGR to RGB
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb.astype('uint8'))
        
        # Compute average hash (perceptual hash)
        phash = imagehash.average_hash(pil_image, hash_size=16)
        
        return str(phash)
    
    def _compute_image_hash(self, image_base64: str) -> str:
        """
        Compute MD5 hash of the image for exact matching.
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            MD5 hash string
        """
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Compute MD5 hash
        image_bytes = image_base64.encode('utf-8')
        return hashlib.md5(image_bytes).hexdigest()
    
    def get_cached_result(
        self,
        image_base64: str,
        threshold: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if similar image exists.
        
        Args:
            image_base64: Base64 encoded image string
            threshold: Hash distance threshold for similarity (0-64, lower = stricter)
            
        Returns:
            Cached result dict with svg and mask_contours, or None if not found
        """
        try:
            # Decode image for perceptual hashing
            image = base64_to_image(image_base64)
            perceptual_hash = self._compute_perceptual_hash(image)
            image_hash = self._compute_image_hash(image_base64)
            
            # First try exact match (same image)
            exact_match = self.db.query(MaskContourCache).filter(
                MaskContourCache.image_hash == image_hash
            ).first()
            
            if exact_match:
                exact_match.access_count += 1
                self.db.commit()
                logger.info(f"💾 Cache hit (exact match) for hash {image_hash[:8]}...")
                return {
                    "svg": exact_match.svg,
                    "mask_contours": exact_match.mask_contours
                }
            
            # Try perceptual match (similar images)
            # Convert hash string to ImageHash object for comparison
            current_hash = imagehash.hex_to_hash(perceptual_hash)
            
            # Get all cached entries
            cached_entries = self.db.query(MaskContourCache).all()
            
            for entry in cached_entries:
                try:
                    cached_hash = imagehash.hex_to_hash(entry.perceptual_hash)
                    # Calculate Hamming distance
                    distance = current_hash - cached_hash
                    
                    if distance <= threshold:
                        entry.access_count += 1
                        self.db.commit()
                        logger.info(
                            f"💾 Cache hit (perceptual match, distance={distance}) "
                            f"for hash {entry.perceptual_hash[:8]}..."
                        )
                        return {
                            "svg": entry.svg,
                            "mask_contours": entry.mask_contours
                        }
                except Exception as e:
                    logger.warning(f"Error comparing hash: {e}")
                    continue
            
            logger.info(f"❌ Cache miss for hash {perceptual_hash[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error in cache lookup: {e}")
            return None
    
    def store_result(
        self,
        image_base64: str,
        svg: str,
        mask_contours: Dict[str, Any]
    ) -> None:
        """
        Store processing result in cache.
        
        Args:
            image_base64: Base64 encoded image string
            svg: Base64 encoded SVG string
            mask_contours: Dictionary of mask contours
        """
        try:
            # Decode image for perceptual hashing
            image = base64_to_image(image_base64)
            perceptual_hash = self._compute_perceptual_hash(image)
            image_hash = self._compute_image_hash(image_base64)
            
            # Check if entry already exists (shouldn't happen, but safety check)
            existing = self.db.query(MaskContourCache).filter(
                MaskContourCache.image_hash == image_hash
            ).first()
            
            if existing:
                # Update existing entry
                existing.svg = svg
                existing.mask_contours = mask_contours
                existing.perceptual_hash = perceptual_hash
                logger.info(f"🔄 Updated cache entry for hash {image_hash[:8]}...")
            else:
                # Create new entry
                cache_entry = MaskContourCache(
                    perceptual_hash=perceptual_hash,
                    image_hash=image_hash,
                    svg=svg,
                    mask_contours=mask_contours,
                    access_count=0
                )
                self.db.add(cache_entry)
                logger.info(f"💾 Stored new cache entry for hash {image_hash[:8]}...")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            self.db.rollback()
            raise

