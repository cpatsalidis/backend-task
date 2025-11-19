"""Job queue system for asynchronous processing."""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Represents a processing job."""
    
    def __init__(self, job_id: int, request_data: dict):
        """
        Initialize a new job.
        
        Args:
            job_id: Unique job identifier
            request_data: Request data to process
        """
        self.id = job_id
        self.status = JobStatus.PENDING
        self.request_data = request_data
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert job to dictionary for API response."""
        response = {
            "id": self.id,
            "status": self.status.value,
        }
        
        if self.status == JobStatus.COMPLETED and self.result:
            response["svg"] = self.result.get("svg")
            response["mask_contours"] = self.result.get("mask_contours")
        elif self.status == JobStatus.FAILED and self.error:
            response["error"] = self.error
        
        return response


class JobQueue:
    """In-memory job queue for managing asynchronous jobs."""
    
    def __init__(self):
        """Initialize the job queue."""
        self._jobs: Dict[int, Job] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()
    
    async def create_job(self, request_data: dict) -> Job:
        """
        Create a new job and add it to the queue.
        
        Args:
            request_data: Request data to process
            
        Returns:
            Created Job instance
        """
        async with self._lock:
            job_id = self._next_id
            self._next_id += 1
            
            job = Job(job_id, request_data)
            self._jobs[job_id] = job
            
            logger.info(f"Created job {job_id} with status {job.status.value}")
            return job
    
    async def get_job(self, job_id: int) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job instance or None if not found
        """
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update job status and result.
        
        Args:
            job_id: Job identifier
            status: New status
            result: Processing result (for completed jobs)
            error: Error message (for failed jobs)
            
        Returns:
            True if job was updated, False if not found
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            job.status = status
            if status == JobStatus.PROCESSING:
                job.started_at = datetime.utcnow()
            elif status == JobStatus.COMPLETED:
                job.completed_at = datetime.utcnow()
                job.result = result
            elif status == JobStatus.FAILED:
                job.completed_at = datetime.utcnow()
                job.error = error
            
            logger.info(f"Updated job {job_id} to status {status.value}")
            return True
    
    async def process_job(
        self,
        job_id: int,
        processor,
        service,
        delay_seconds: float = 0.0
    ) -> None:
        """
        Process a job asynchronously.
        
        Args:
            job_id: Job identifier
            processor: FacialRegionProcessor instance
            service: FacialProcessingService instance
            delay_seconds: Optional delay to simulate complex processing
        """
        try:
            # Update status to processing
            await self.update_job_status(job_id, JobStatus.PROCESSING)
            
            # Simulate delay if specified (for demonstration)
            if delay_seconds > 0:
                logger.info(f"Simulating {delay_seconds}s delay for job {job_id}")
                await asyncio.sleep(delay_seconds)
            
            # Get job and process
            job = await self.get_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return
            
            # Import here to avoid circular imports
            from .models import CropSubmitRequest
            
            # Process the request in executor to avoid blocking
            # The service.process_crop_submit is CPU-bound, so we run it in a thread
            loop = asyncio.get_event_loop()
            request = CropSubmitRequest(**job.request_data)
            
            # Run synchronous processing in executor
            result = await loop.run_in_executor(
                None,
                service.process_crop_submit,
                request
            )
            
            # Update job with result
            await self.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "svg": result.svg,
                    "mask_contours": result.mask_contours
                }
            )
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Error processing job {job_id}: {str(e)}")
            await self.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e)
            )


# Global job queue instance
job_queue = JobQueue()

