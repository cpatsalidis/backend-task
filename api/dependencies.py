"""Dependency injection for FastAPI application."""

from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import Session

from .service import FacialRegionProcessor
from .database import get_db


def get_processor() -> FacialRegionProcessor:
    """
    Factory function to create FacialRegionProcessor instance.
    
    This function is used for dependency injection, allowing the processor
    to be easily mocked in tests and configured per request if needed.
    
    Returns:
        FacialRegionProcessor instance
    """
    return FacialRegionProcessor()


# Type alias for dependency injection
ProcessorDep = Annotated[FacialRegionProcessor, Depends(get_processor)]
DatabaseDep = Annotated[Session, Depends(get_db)]

