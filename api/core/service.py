"""Service layer for processing facial images."""

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2 as cv
import numpy as np

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_dir))

import config
from processing.mask_creation import create_smooth_mask, create_region_masks
from processing.region_extension import (
    extend_forehead_with_segmentation,
    extend_lower_face_with_segmentation,
    detect_ears_from_segmentation
)
from processing.blending import blend_regions, apply_extended_region
from utils.rotation import detect_face_tilt, rotate_image, rotate_landmarks
from utils.cropping import calculate_head_crop_box, apply_crop
from visualization.contours import draw_all_contours, draw_dashed_contour


class FacialRegionProcessor:
    """Process facial images and extract region information."""
    
    def __init__(self):
        """Initialize processor with configuration."""
        self.facial_regions = config.FACIAL_REGIONS
        self.alpha_blend = config.ALPHA_BLEND
        self.contour_color = config.CONTOUR_COLOR
    
    def process_image(
        self,
        original_img: np.ndarray,
        segmentation_img: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Process facial image with full pipeline.
        
        Args:
            original_img: Original facial image
            segmentation_img: Segmentation map
            landmarks: Facial landmarks array
        
        Returns:
            Tuple of (processed_image, extended_region_masks)
        """
        # Resize segmentation to match original image
        if segmentation_img.shape[:2] != original_img.shape[:2]:
            segmentation_img = cv.resize(
                segmentation_img,
                (original_img.shape[1], original_img.shape[0])
            )
        
        # Step 1: Detect face tilt
        tilt_angle = detect_face_tilt(landmarks)
        
        # Step 2: Create initial region masks
        region_masks = create_region_masks(original_img.shape, landmarks)
        
        # Step 3: Blend regions and draw contours
        result_final = blend_regions(original_img, region_masks)
        draw_all_contours(result_final, region_masks, self.contour_color)
        
        # Step 4: Rotate if needed
        rotated_segmentation = segmentation_img.copy()
        rotated_landmarks = landmarks.copy()
        rotated_original = original_img.copy()
        
        if abs(tilt_angle) > 0.5:
            result_final = rotate_image(result_final, tilt_angle)
            rotated_segmentation = rotate_image(segmentation_img, tilt_angle)
            rotated_original = rotate_image(original_img, tilt_angle)
            rotated_landmarks = rotate_landmarks(landmarks, tilt_angle, result_final.shape)
        
        # Step 5: Extend regions
        extended_masks = {}
        
        # Extend forehead
        forehead_indices = self.facial_regions['forehead']['indices']
        forehead_points = np.array([rotated_landmarks[i] for i in forehead_indices], dtype=np.int32)
        forehead_mask = create_smooth_mask(result_final.shape, forehead_points)
        extended_forehead_mask = extend_forehead_with_segmentation(
            forehead_mask, rotated_segmentation, rotated_landmarks
        )
        
        # Apply extended forehead
        result_final[forehead_mask > 0] = rotated_original[forehead_mask > 0]
        forehead_color = self.facial_regions['forehead']['color']
        result_final = apply_extended_region(
            result_final, rotated_original, extended_forehead_mask, forehead_color
        )
        
        # Redraw forehead contours
        contours, _ = cv.findContours(extended_forehead_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if contours:
            for contour in contours:
                draw_dashed_contour(result_final, contour, self.contour_color)
        
        extended_masks['forehead'] = extended_forehead_mask
        
        # Extend lower face
        lower_face_indices = self.facial_regions['lower_face']['indices']
        lower_face_points = np.array([rotated_landmarks[i] for i in lower_face_indices], dtype=np.int32)
        lower_face_mask = create_smooth_mask(result_final.shape, lower_face_points)
        extended_lower_face_mask = extend_lower_face_with_segmentation(
            lower_face_mask, rotated_segmentation, rotated_landmarks
        )
        
        # Apply extended lower face
        result_final[lower_face_mask > 0] = rotated_original[lower_face_mask > 0]
        lower_face_color = self.facial_regions['lower_face']['color']
        result_final = apply_extended_region(
            result_final, rotated_original, extended_lower_face_mask, lower_face_color
        )
        
        # Redraw lower face contours
        contours, _ = cv.findContours(extended_lower_face_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if contours:
            for contour in contours:
                draw_dashed_contour(result_final, contour, self.contour_color)
        
        # Detect ears separately
        ear_mask = detect_ears_from_segmentation(rotated_segmentation, rotated_landmarks)
        extended_masks['lower_face'] = extended_lower_face_mask
        
        if ear_mask.any():
            # Split ear mask into left and right
            img_width = ear_mask.shape[1]
            mid_x = img_width // 2
            
            # Left ear (left half of image)
            left_ear_mask = ear_mask.copy()
            left_ear_mask[:, mid_x:] = 0
            
            # Right ear (right half of image)
            right_ear_mask = ear_mask.copy()
            right_ear_mask[:, :mid_x] = 0
            
            # Smooth ear masks for better appearance
            def smooth_mask(mask, aggressive=True):
                """Apply smoothing operations to mask.
                
                Args:
                    mask: Input mask to smooth
                    aggressive: If True, use full smoothing. If False, use lighter smoothing.
                """
                if not mask.any():
                    return mask
                # Morphological closing to fill gaps and smooth edges
                if aggressive:
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)
                else:
                    # Lighter smoothing for right ear - smaller kernel
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
                smoothed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
                # Gaussian blur for smoother edges
                if aggressive:
                    smoothed = cv.GaussianBlur(smoothed, config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_SIGMA)
                else:
                    # Lighter Gaussian blur for right ear
                    smoothed = cv.GaussianBlur(smoothed, (25, 25), 5)
                # Threshold back to binary
                _, smoothed = cv.threshold(smoothed, 127, 255, cv.THRESH_BINARY)
                return smoothed
            
            # Apply smoothing to ear masks - less aggressive for right ear
            left_ear_mask = smooth_mask(left_ear_mask, aggressive=True)
            right_ear_mask = smooth_mask(right_ear_mask, aggressive=False)
            
            # Apply left ear
            if left_ear_mask.any():
                result_final = apply_extended_region(
                    result_final, rotated_original, left_ear_mask, lower_face_color
                )
                contours, _ = cv.findContours(left_ear_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if contours:
                    for contour in contours:
                        draw_dashed_contour(result_final, contour, self.contour_color)
                extended_masks['left_ear'] = left_ear_mask
            
            # Apply right ear
            if right_ear_mask.any():
                result_final = apply_extended_region(
                    result_final, rotated_original, right_ear_mask, lower_face_color
                )
                contours, _ = cv.findContours(right_ear_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if contours:
                    for contour in contours:
                        draw_dashed_contour(result_final, contour, self.contour_color)
                extended_masks['right_ear'] = right_ear_mask
        
        # Add other regions (nose, eyes)
        for region_name in ['nose', 'left_under_eye', 'right_under_eye']:
            if region_name in region_masks:
                # Rotate the mask if needed
                if abs(tilt_angle) > 0.5:
                    rotated_mask = rotate_image(region_masks[region_name], tilt_angle)
                    extended_masks[region_name] = rotated_mask
                else:
                    extended_masks[region_name] = region_masks[region_name]
        
        # Step 6: Crop to head region
        crop_box = calculate_head_crop_box(rotated_segmentation, padding_factor=config.CROP_PADDING_FACTOR)
        result_final = apply_crop(result_final, crop_box)
        
        # Also crop the masks
        cropped_masks = {}
        for region_name, mask in extended_masks.items():
            cropped_masks[region_name] = apply_crop(mask, crop_box)
        
        return result_final, cropped_masks
    
    def get_region_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get region colors from configuration.
        
        Returns:
            Dictionary mapping region names to BGR colors
        """
        return {
            region_name: region_info['color']
            for region_name, region_info in self.facial_regions.items()
        }
    
    def get_region_labels(self) -> Dict[str, str]:
        """
        Get region labels from configuration.
        
        Returns:
            Dictionary mapping region names to label text
        """
        return {
            region_name: region_info['label']
            for region_name, region_info in self.facial_regions.items()
        }

