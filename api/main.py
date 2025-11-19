"""FastAPI application for facial region processing."""

import asyncio
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .cache import get_cache_entry, get_cache_stats, list_cache_entries
from .core import FacialProcessingService
from .database import init_db
from .dependencies import DatabaseDep, ProcessorDep
from .middleware import PrometheusMiddleware, get_metrics_response, setup_logging
from .models import (
    CropSubmitRequest,
    CropSubmitResponse,
    ErrorResponse,
    JobStatusResponse,
    JobSubmitResponse,
)
from .queue import job_queue
from .utils import RequestValidator

# Configure Rich logging
setup_logging(level="INFO")
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

# Add Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup."""
    try:
        init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"⚠️  Database initialization failed: {e}")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Facial Region Processing API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "submit": "/api/v1/frontal/crop/submit",
            "status": "/api/v1/frontal/crop/status/{job_id}",
            "docs": "/api/docs",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "facial-region-processing"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics_response()


@app.get("/api/v1/cache/stats", tags=["Cache"])
async def cache_stats(db: DatabaseDep):
    """
    Get cache statistics.
    
    Returns information about cached entries including:
    - Total number of cached entries
    - Total cache hits
    - Most accessed entry
    
    Note: Returns empty stats if database is not available.
    """
    if not db:
        return {
            "total_entries": 0,
            "total_cache_hits": 0,
            "most_accessed": None,
            "message": "Database not available - caching disabled"
        }
    return get_cache_stats(db)


@app.get("/api/v1/cache/entries", tags=["Cache"])
async def cache_entries(
    db: DatabaseDep,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of entries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List cache entries.
    
    Returns a paginated list of cache entries with metadata.
    
    Note: Returns empty list if database is not available.
    """
    if not db:
        return {
            "entries": [],
            "limit": limit,
            "offset": offset,
            "message": "Database not available - caching disabled"
        }
    return {
        "entries": list_cache_entries(db, limit=limit, offset=offset),
        "limit": limit,
        "offset": offset
    }


@app.get("/api/v1/cache/entries/{entry_id}", tags=["Cache"])
async def cache_entry_detail(entry_id: int, db: DatabaseDep):
    """
    Get detailed information about a specific cache entry.
    
    Returns the full cache entry including SVG and mask contours.
    
    Note: Returns 503 if database is not available.
    """
    if not db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available - caching disabled"
        )
    entry = get_cache_entry(db, entry_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache entry {entry_id} not found"
        )
    return entry


@app.post(
    "/api/v1/frontal/crop/submit",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request - Invalid input"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity - Validation error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Processing"]
)
async def crop_submit(
    request: CropSubmitRequest,
    processor: ProcessorDep,
    db: DatabaseDep,
    delay: float = Query(0.0, description="Optional delay in seconds to simulate complex processing")
) -> JobSubmitResponse:
    """
    Submit a job to process facial image asynchronously.
    
    This endpoint accepts a portrait image with facial landmarks and segmentation map,
    creates a processing job, and returns immediately with a job ID and status.
    
    The job is processed asynchronously in the background. Use the status endpoint
    to check job progress and retrieve results when completed.
    
    **Parameters:**
    - **image**: Base64 encoded original facial image
    - **landmarks**: Array of facial landmark points (MediaPipe format, 478 points)
    - **segmentation_map**: Base64 encoded segmentation map
    - **delay**: Optional delay in seconds to simulate complex processing (default: 0)
    
    **Returns:**
    - **id**: Job ID for status checking
    - **status**: Job status (pending)
    
    **Example:**
    ```json
    {
        "id": 123,
        "status": "pending"
    }
    ```
    """
    try:
        logger.info("📥 Received crop/submit request")
        
        # Validate request data
        RequestValidator.validate_landmarks_count(request.landmarks)
        
        # Create job
        job = await job_queue.create_job(request.dict())
        
        # Start background processing
        # Note: Service will be created in process_job with its own DB session
        asyncio.create_task(
            job_queue.process_job(
                job.id,
                processor,
                None,  # Service will be created in process_job
                delay_seconds=delay
            )
        )
        
        logger.info(f"✅ Created job [bold cyan]#{job.id}[/bold cyan], processing in background")
        
        return JobSubmitResponse(
            id=job.id,
            status=job.status.value
        )
    
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


@app.get(
    "/api/v1/frontal/crop/status/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Processing"]
)
async def crop_status(job_id: int) -> JobStatusResponse:
    """
    Check the status of a processing job.
    
    This endpoint returns the current status of a job and, if completed,
    includes the processing results (SVG and mask contours).
    
    **Parameters:**
    - **job_id**: Job ID returned from the submit endpoint
    
    **Returns:**
    - **id**: Job ID
    - **status**: Job status (pending, processing, completed, failed)
    - **svg**: Base64 encoded SVG (only when status is completed)
    - **mask_contours**: Dictionary of mask contours (only when status is completed)
    - **error**: Error message (only when status is failed)
    
    **Status Values:**
    - `pending`: Job is queued but not yet started
    - `processing`: Job is currently being processed
    - `completed`: Job completed successfully, results available
    - `failed`: Job failed with an error
    """
    try:
        job = await job_queue.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        response_dict = job.to_dict()
        return JobStatusResponse(**response_dict)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.exception(f"Unexpected error in crop_status: {str(e)}")
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

