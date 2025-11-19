"""Rich logging configuration for beautiful console output."""

import logging
import sys
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

# Install Rich traceback handler for better error formatting
install(show_locals=True)

# Create console for Rich output
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging with Rich for beautiful console output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Create Rich handler with beautiful formatting
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_level=True,
    )
    
    # Set format (Rich handles most formatting, but we can add basic structure)
    rich_handler.setFormatter(
        logging.Formatter(
            "%(message)s",
            datefmt="[%X]"
        )
    )
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    rich_handler.setLevel(log_level)
    root_logger.setLevel(log_level)
    
    # Add handler
    root_logger.addHandler(rich_handler)
    
    # Prevent duplicate logs from uvicorn
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn").handlers.clear()
    
    # Set specific loggers to use Rich handler
    for logger_name in ["api", "uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(rich_handler)
        logger.setLevel(log_level)
        logger.propagate = False

