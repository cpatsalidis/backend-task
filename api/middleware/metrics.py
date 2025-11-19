"""Prometheus metrics for observability."""

import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Job metrics
jobs_total = Counter(
    'jobs_total',
    'Total number of jobs',
    ['status']
)

jobs_in_progress = Gauge(
    'jobs_in_progress',
    'Number of jobs currently in progress'
)

job_processing_duration_seconds = Histogram(
    'job_processing_duration_seconds',
    'Job processing duration in seconds',
    buckets=[1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP requests with Prometheus metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get endpoint path (simplified for metrics)
        endpoint = request.url.path
        # Normalize endpoint to avoid high cardinality
        if endpoint.startswith('/api/v1/frontal/crop/status/'):
            endpoint = '/api/v1/frontal/crop/status/{job_id}'
        
        method = request.method
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            raise


def get_metrics_response() -> Response:
    """Generate Prometheus metrics response."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

