"""Views and utilities for inspecting cache entries."""

from typing import Any, Dict, List

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from ..database.db_models import MaskContourCache


def get_cache_stats(db: Session) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with cache statistics
    """
    total_entries = db.query(MaskContourCache).count()
    total_accesses = db.query(func.sum(MaskContourCache.access_count)).scalar() or 0
    
    most_accessed = db.query(MaskContourCache).order_by(
        desc(MaskContourCache.access_count)
    ).first()
    
    return {
        "total_entries": total_entries,
        "total_cache_hits": total_accesses,
        "most_accessed": {
            "id": most_accessed.id if most_accessed else None,
            "access_count": most_accessed.access_count if most_accessed else 0,
            "created_at": most_accessed.created_at.isoformat() if most_accessed else None
        } if most_accessed else None
    }


def list_cache_entries(
    db: Session,
    limit: int = 10,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List cache entries.
    
    Args:
        db: Database session
        limit: Maximum number of entries to return
        offset: Offset for pagination
        
    Returns:
        List of cache entry dictionaries
    """
    entries = db.query(MaskContourCache).order_by(
        desc(MaskContourCache.created_at)
    ).offset(offset).limit(limit).all()
    
    return [
        {
            "id": entry.id,
            "image_hash": entry.image_hash[:16] + "...",  # Truncate for display
            "perceptual_hash": entry.perceptual_hash[:16] + "...",
            "created_at": entry.created_at.isoformat(),
            "access_count": entry.access_count,
            "mask_contours_count": len(entry.mask_contours) if entry.mask_contours else 0,
            "svg_length": len(entry.svg) if entry.svg else 0
        }
        for entry in entries
    ]


def get_cache_entry(db: Session, entry_id: int) -> Dict[str, Any]:
    """
    Get a specific cache entry with full data.
    
    Args:
        db: Database session
        entry_id: Cache entry ID
        
    Returns:
        Dictionary with full cache entry data
    """
    entry = db.query(MaskContourCache).filter(
        MaskContourCache.id == entry_id
    ).first()
    
    if not entry:
        return None
    
    return {
        "id": entry.id,
        "image_hash": entry.image_hash,
        "perceptual_hash": entry.perceptual_hash,
        "svg": entry.svg,
        "mask_contours": entry.mask_contours,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
        "access_count": entry.access_count
    }

