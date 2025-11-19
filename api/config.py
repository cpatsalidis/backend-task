"""Application configuration."""

import os
from concurrent.futures import ThreadPoolExecutor

# Load testing mode - disables delay and reduces logging for maximum performance
LOAD_TESTING_MODE = os.getenv("LOAD_TESTING_MODE", "false").lower() == "true"

# Logging level - use WARNING or ERROR in load testing mode
if LOAD_TESTING_MODE:
    DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
else:
    DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Thread pool configuration for CPU-bound tasks
# Use more workers in load testing mode for better parallelism
if LOAD_TESTING_MODE:
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 4)))
else:
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Global thread pool executor for CPU-bound tasks
_thread_pool_executor = None


def get_thread_pool_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor for CPU-bound tasks."""
    global _thread_pool_executor
    if _thread_pool_executor is None:
        _thread_pool_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    return _thread_pool_executor

