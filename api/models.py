"""Pydantic models for API request/response validation."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Landmark(BaseModel):
    """Single facial landmark point."""
    x: float = Field(..., description="X coordinate of landmark")
    y: float = Field(..., description="Y coordinate of landmark")


class CropSubmitRequest(BaseModel):
    """Request model for /api/v1/frontal/crop/submit endpoint."""
    image: str = Field(..., description="Base64 encoded image string")
    landmarks: List[Landmark] = Field(..., description="Array of facial landmark points")
    segmentation_map: str = Field(..., description="Base64 encoded segmentation map string")

    class Config:
        json_schema_extra = {
            "example": {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "landmarks": [
                    {"x": 100.5, "y": 150.2},
                    {"x": 200.3, "y": 151.8}
                ],
                "segmentation_map": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
        }


class MaskContour(BaseModel):
    """Contour data for a single mask."""
    region_name: str = Field(..., description="Name of the facial region")
    points: List[List[float]] = Field(..., description="Array of contour points [[x1, y1], [x2, y2], ...]")


class CropSubmitResponse(BaseModel):
    """Response model for /api/v1/frontal/crop/submit endpoint."""
    svg: str = Field(..., description="Base64 encoded SVG with highlighted regions")
    mask_contours: Dict[str, List[List[float]]] = Field(..., description="Dictionary mapping region names to contour points")

    class Config:
        json_schema_extra = {
            "example": {
                "svg": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPi4uLjwvc3ZnPg==",
                "mask_contours": {
                    "forehead": [[100.0, 50.0], [110.0, 55.0], [120.0, 50.0]],
                    "nose": [[150.0, 200.0], [160.0, 210.0], [170.0, 200.0]]
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for categorization")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid image format",
                "error_code": "INVALID_IMAGE"
            }
        }

