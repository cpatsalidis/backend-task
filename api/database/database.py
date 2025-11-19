"""Database configuration and session management."""

import logging
import os
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from ..config import LOAD_TESTING_MODE

logger = logging.getLogger(__name__)

# Get database URL from environment or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/facial_processing"
)

# Check if database should be used (set to empty string to disable)
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() != "false"

# Initialize engine and session as None
engine = None
SessionLocal = None
Base = declarative_base()

# Try to create engine if database is enabled
if USE_DATABASE:
    try:
        # OPTIMIZATION: Increase pool size in load testing mode for better concurrency
        if LOAD_TESTING_MODE:
            pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
            max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "40"))
        else:
            pool_size = 10
            max_overflow = 20
        
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=pool_size,
            max_overflow=max_overflow,
            connect_args={"connect_timeout": 2}  # Quick timeout for connection test
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        if not LOAD_TESTING_MODE:
            logger.info("✅ Database connection available")
    except Exception as e:
        logger.warning(f"⚠️  Database not available: {e}. Running without caching.")
        engine = None
        SessionLocal = None
else:
    if not LOAD_TESTING_MODE:
        logger.info("ℹ️  Database disabled (USE_DATABASE=false). Running without caching.")


def get_db() -> Generator[Optional[Session], None, None]:
    """
    Dependency for getting database session.
    
    Returns None if database is not available.
    
    Yields:
        Database session or None
    """
    if not SessionLocal:
        yield None
        return
    
    db = SessionLocal()
    try:
        # Test connection
        db.execute(text("SELECT 1"))
        yield db
    except OperationalError as e:
        logger.warning(f"⚠️  Database connection failed: {e}. Continuing without cache.")
        yield None
    except Exception as e:
        logger.warning(f"⚠️  Database error: {e}. Continuing without cache.")
        yield None
    finally:
        if db:
            db.close()


def init_db() -> None:
    """Initialize database tables."""
    if not engine:
        logger.warning("⚠️  Database not available, skipping initialization")
        return
    
    try:
        Base.metadata.create_all(bind=engine)
        if not LOAD_TESTING_MODE:
            logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.warning(f"⚠️  Database initialization failed: {e}")

