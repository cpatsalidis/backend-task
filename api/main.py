"""FastAPI application for facial region processing."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict

from .models import CropSubmitRequest, CropSubmitResponse, ErrorResponse
from .dependencies import ProcessorDep
from .business import FacialProcessingService
from .validators import RequestValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Facial Region Processing API",
    description="API for processing facial images and extracting region information",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Facial Region Processing API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "submit": "/api/v1/frontal/crop/submit",
            "docs": "/api/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "facial-region-processing"
    }


@app.post(
    "/api/v1/frontal/crop/submit",
    response_model=CropSubmitResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request - Invalid input"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity - Validation error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Processing"]
)
async def crop_submit(
    request: CropSubmitRequest,
    processor: ProcessorDep
) -> CropSubmitResponse:
    """
    Process facial image and return highlighted regions overlay.
    
    This endpoint accepts a portrait image with facial landmarks and segmentation map,
    processes it to detect and highlight facial regions, and returns the result as an
    SVG overlay with contour information.
    
    **Processing Steps:**
    1. Decode base64 images
    2. Detect face tilt and rotate if needed
    3. Create and extend facial region masks (forehead, chin, ears, etc.)
    4. Generate SVG overlay with highlighted regions
    5. Extract contour points for each region
    
    **Parameters:**
    - **image**: Base64 encoded original facial image
    - **landmarks**: Array of facial landmark points (MediaPipe format, 478 points)
    - **segmentation_map**: Base64 encoded segmentation map
    
    **Returns:**
    - **svg**: Base64 encoded SVG with highlighted regions overlay
    - **mask_contours**: Dictionary mapping region names to contour point arrays
    """
    try:
        logger.info("Received crop/submit request")
        
        # Validate request data
        RequestValidator.validate_landmarks_count(request.landmarks)
        
        # Process request using business logic service
        # Processor is injected via dependency injection
        service = FacialProcessingService(processor)
        return service.process_crop_submit(request)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except ValueError as e:
        # Handle validation errors from business layer
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Log unexpected errors
        logger.exception(f"Unexpected error in crop_submit: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

