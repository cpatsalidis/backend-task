"""
Main entry point for facial region processing pipeline.

This script processes facial images by:
1. Loading images and landmarks
2. Detecting and correcting face tilt
3. Creating facial region masks
4. Extending regions using segmentation data
5. Blending regions onto the image
6. Drawing contours
7. Cropping to final result
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

import cv2 as cv
import numpy as np
import config
from utils.image_io import load_images, load_landmarks, save_result
from utils.rotation import detect_face_tilt, rotate_image, rotate_landmarks
from utils.cropping import calculate_head_crop_box, apply_crop
from processing.mask_creation import create_smooth_mask, create_region_masks
from processing.region_extension import (
    extend_forehead_with_segmentation,
    extend_lower_face_with_segmentation,
    detect_ears_from_segmentation
)
from processing.blending import blend_regions, apply_extended_region
from visualization.contours import draw_all_contours, draw_dashed_contour
from visualization.labels import draw_region_labels


def main():
    """Main processing pipeline."""
    
    print("="*60)
    print("FACIAL REGION PROCESSING PIPELINE")
    print("="*60)
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print("\n[1/6] Loading images and landmarks...")
    original_img, segmented_img = load_images(config.INPUT_ORIGINAL_IMAGE, config.INPUT_SEGMENTATION_MAP)
    points = load_landmarks(config.INPUT_LANDMARKS_FILE)
    print(f"      ✓ Loaded images: {original_img.shape[1]}x{original_img.shape[0]}")
    print(f"      ✓ Loaded {len(points)} landmarks")
    
    # ========================================================================
    # Step 2: Detect Face Tilt
    # ========================================================================
    print("\n[2/6] Detecting face tilt...")
    tilt_angle = detect_face_tilt(points)
    print(f"      Detected tilt: {tilt_angle:.2f}°")
    
    # ========================================================================
    # Step 3: Create Facial Region Masks
    # ========================================================================
    print("\n[3/6] Creating facial region masks...")
    region_masks = create_region_masks(original_img.shape, points)
    for region_name, region_info in config.FACIAL_REGIONS.items():
        print(f"      ✓ {region_info['name']}: {len(region_info['indices'])} landmarks")
    
    # ========================================================================
    # Step 4: Blend Regions and Draw Contours
    # ========================================================================
    print("\n[4/6] Blending regions and drawing contours...")
    result_final = blend_regions(original_img, region_masks)
    draw_all_contours(result_final, region_masks, config.CONTOUR_COLOR)
    print("      ✓ Regions blended with contours")
    
    # ========================================================================
    # Step 5: Post-Processing (Rotation & Extension)
    # ========================================================================
    print("\n[5/6] Post-processing...")
    
    # Rotate if needed
    rotated_segmentation = segmented_img.copy()
    rotated_landmarks = points.copy()
    rotated_original = original_img.copy()
    
    # Also prepare rotated masks for nose and eyes
    rotated_region_masks = region_masks.copy()
    
    if abs(tilt_angle) > 0.5:
        print(f"      Rotating by {-tilt_angle:.2f}°...")
        result_final = rotate_image(result_final, tilt_angle)
        rotated_segmentation = rotate_image(segmented_img, tilt_angle)
        rotated_original = rotate_image(original_img, tilt_angle)
        rotated_landmarks = rotate_landmarks(points, tilt_angle, result_final.shape)
        
        # Rotate the masks for nose and eyes too
        for region_name in ['nose', 'left_under_eye', 'right_under_eye']:
            if region_name in region_masks:
                rotated_region_masks[region_name] = rotate_image(region_masks[region_name], tilt_angle)
        
        print("      ✓ Images rotated")
    else:
        print("      ℹ No rotation needed")
    
    # Extend forehead
    print("      Extending forehead...")
    forehead_indices = config.FACIAL_REGIONS['forehead']['indices']
    forehead_points = np.array([rotated_landmarks[i] for i in forehead_indices], dtype=np.int32)
    forehead_mask = create_smooth_mask(result_final.shape, forehead_points)
    extended_forehead_mask = extend_forehead_with_segmentation(forehead_mask, rotated_segmentation, rotated_landmarks)
    
    # Remove old forehead and apply extended
    result_final[forehead_mask > 0] = rotated_original[forehead_mask > 0]
    forehead_color = config.FACIAL_REGIONS['forehead']['color']
    result_final = apply_extended_region(result_final, rotated_original, extended_forehead_mask, forehead_color)
    
    # Redraw contours
    contours, _ = cv.findContours(extended_forehead_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if contours:
        for contour in contours:
            draw_dashed_contour(result_final, contour, config.CONTOUR_COLOR)
    print("      ✓ Forehead extended")
    
    # Extend lower face
    print("      Extending lower face...")
    lower_face_indices = config.FACIAL_REGIONS['lower_face']['indices']
    lower_face_points = np.array([rotated_landmarks[i] for i in lower_face_indices], dtype=np.int32)
    lower_face_mask = create_smooth_mask(result_final.shape, lower_face_points)
    extended_lower_face_mask = extend_lower_face_with_segmentation(lower_face_mask, rotated_segmentation, rotated_landmarks)
    
    # Remove old lower face and apply extended
    result_final[lower_face_mask > 0] = rotated_original[lower_face_mask > 0]
    lower_face_color = config.FACIAL_REGIONS['lower_face']['color']
    result_final = apply_extended_region(result_final, rotated_original, extended_lower_face_mask, lower_face_color)
    
    # Redraw contours
    contours, _ = cv.findContours(extended_lower_face_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if contours:
        for contour in contours:
            draw_dashed_contour(result_final, contour, config.CONTOUR_COLOR)
    print("      ✓ Lower face extended")
    
    # Detect ears separately
    print("      Detecting ears...")
    ear_mask = detect_ears_from_segmentation(rotated_segmentation, rotated_landmarks)
    
    # Track all extended masks for label drawing (use rotated masks!)
    all_masks = {
        'forehead': extended_forehead_mask,
        'lower_face': extended_lower_face_mask,
        'nose': rotated_region_masks['nose'] if 'nose' in rotated_region_masks else None,
        'left_under_eye': rotated_region_masks['left_under_eye'] if 'left_under_eye' in rotated_region_masks else None,
        'right_under_eye': rotated_region_masks['right_under_eye'] if 'right_under_eye' in rotated_region_masks else None,
    }
    
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
        
        # Apply left ear
        if left_ear_mask.any():
            result_final = apply_extended_region(result_final, rotated_original, left_ear_mask, lower_face_color)
            contours, _ = cv.findContours(left_ear_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if contours:
                for contour in contours:
                    draw_dashed_contour(result_final, contour, config.CONTOUR_COLOR)
            all_masks['left_ear'] = left_ear_mask
        
        # Apply right ear
        if right_ear_mask.any():
            result_final = apply_extended_region(result_final, rotated_original, right_ear_mask, lower_face_color)
            contours, _ = cv.findContours(right_ear_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if contours:
                for contour in contours:
                    draw_dashed_contour(result_final, contour, config.CONTOUR_COLOR)
            all_masks['right_ear'] = right_ear_mask
        
        print("      ✓ Ears included")
    else:
        print("      ℹ No ears detected")
    
    # ========================================================================
    # Step 6: Final Cropping
    # ========================================================================
    print("\n[6/6] Cropping to head region...")
    crop_box = calculate_head_crop_box(rotated_segmentation, padding_factor=config.CROP_PADDING_FACTOR)
    original_size = f"{result_final.shape[1]}x{result_final.shape[0]}"
    result_final = apply_crop(result_final, crop_box)
    
    # Also crop all masks
    cropped_masks = {}
    for region_name, mask in all_masks.items():
        if mask is not None:
            cropped_masks[region_name] = apply_crop(mask, crop_box)
    
    cropped_size = f"{result_final.shape[1]}x{result_final.shape[0]}"
    print(f"      ✓ Cropped from {original_size} to {cropped_size}")
    
    # ========================================================================
    # Step 7: Draw Labels
    # ========================================================================
    print("\nDrawing region labels...")
    region_labels = {region_name: region_info['label'] 
                     for region_name, region_info in config.FACIAL_REGIONS.items()}
    draw_region_labels(result_final, cropped_masks, region_labels)
    print("      ✓ Labels drawn")
    
    # ========================================================================
    # Save Result
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING RESULT")
    print("="*60)
    save_result(result_final, config.OUTPUT_RESULT_IMAGE)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    
    # Display result
    cv.imshow('Final Result', result_final)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

