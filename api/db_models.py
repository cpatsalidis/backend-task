"""Database models for storing mask contours."""

from sqlalchemy import Column, Integer, String, JSON, DateTime, Index
from sqlalchemy.sql import func
from .database import Base


class MaskContourCache(Base):
    """Model for caching mask contours with perceptual hash."""
    
    __tablename__ = "mask_contour_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    perceptual_hash = Column(String(64), nullable=False, index=True)
    image_hash = Column(String(64), nullable=False, index=True)
    svg = Column(String, nullable=False)  # Base64 encoded SVG
    mask_contours = Column(JSON, nullable=False)  # Dictionary of mask contours
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    access_count = Column(Integer, default=0)  # Track cache hits
    
    # Index for faster lookups
    __table_args__ = (
        Index('idx_perceptual_hash', 'perceptual_hash'),
        Index('idx_image_hash', 'image_hash'),
    )
    
    def __repr__(self):
        return f"<MaskContourCache(id={self.id}, hash={self.perceptual_hash[:8]}...)>"

