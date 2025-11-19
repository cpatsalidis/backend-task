"""Application configuration."""

import os

# Load testing mode - disables delay and reduces logging for maximum performance
LOAD_TESTING_MODE = os.getenv("LOAD_TESTING_MODE", "false").lower() == "true"

# Logging level - use WARNING or ERROR in load testing mode
if LOAD_TESTING_MODE:
    DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
else:
    DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

