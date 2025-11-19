"""
Simple test script for the Facial Region Processing API.

Usage:
    python test_api.py
"""

import requests
import base64
import json
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_landmarks(landmarks_path: str) -> list:
    """Load landmarks from file."""
    with open(landmarks_path, 'r') as f:
        import ast
        landmarks_data = ast.literal_eval(f.read())
    
    landmarks = landmarks_data['landmarks'][0]
    return [{"x": float(lm['x']), "y": float(lm['y'])} for lm in landmarks]


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Health check failed"
    print("✓ Health check passed")


def test_root_endpoint(base_url: str = "http://localhost:8000"):
    """Test the root endpoint."""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)
    
    response = requests.get(f"{base_url}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Root endpoint failed"
    print("✓ Root endpoint passed")


def test_crop_submit(
    base_url: str = "http://localhost:8000",
    image_path: str = "Backend_Engineer/original_image.png",
    segmentation_path: str = "Backend_Engineer/segmentation_map.png",
    landmarks_path: str = "Backend_Engineer/landmarks.txt"
):
    """Test the crop/submit endpoint."""
    print("\n" + "="*60)
    print("Testing Crop/Submit Endpoint")
    print("="*60)
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return
    
    if not Path(segmentation_path).exists():
        print(f"❌ Segmentation file not found: {segmentation_path}")
        return
    
    if not Path(landmarks_path).exists():
        print(f"❌ Landmarks file not found: {landmarks_path}")
        return
    
    # Load data
    print("\nLoading test data...")
    image_base64 = image_to_base64(image_path)
    seg_base64 = image_to_base64(segmentation_path)
    landmarks = load_landmarks(landmarks_path)
    
    print(f"✓ Loaded image: {len(image_base64)} chars")
    print(f"✓ Loaded segmentation: {len(seg_base64)} chars")
    print(f"✓ Loaded landmarks: {len(landmarks)} points")
    
    # Prepare request
    payload = {
        "image": image_base64,
        "landmarks": landmarks,
        "segmentation_map": seg_base64
    }
    
    # Send request
    print("\nSending request to API...")
    response = requests.post(
        f"{base_url}/api/v1/frontal/crop/submit",
        json=payload,
        timeout=60  # 60 second timeout
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Save SVG
        svg_data = base64.b64decode(result['svg'])
        output_path = "api_test_result.svg"
        with open(output_path, 'wb') as f:
            f.write(svg_data)
        print(f"✓ SVG saved to: {output_path}")
        
        # Show contours info
        contours = result['mask_contours']
        print(f"\n✓ Found {len(contours)} regions:")
        for region_name, points in contours.items():
            print(f"  - {region_name}: {len(points)} contour points")
        
        print("\n✓ Crop/submit test passed!")
        
    else:
        print(f"❌ Request failed")
        try:
            error_data = response.json()
            print(f"Error: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Response: {response.text}")


def main():
    """Run all tests."""
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("FACIAL REGION PROCESSING API - TEST SUITE")
    print("="*60)
    print(f"Base URL: {base_url}")
    
    try:
        # Test health check
        test_health_check(base_url)
        
        # Test root endpoint
        test_root_endpoint(base_url)
        
        # Test crop/submit endpoint
        test_crop_submit(base_url)
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running:")
        print("  docker-compose up")
        print("  OR")
        print("  uvicorn api.main:app --reload")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

