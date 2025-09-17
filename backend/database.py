"""
Database configuration and session management
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator, Any

# Import models to ensure they are registered with the Base
from backend.models.auth import Base as AuthBase  # noqa: F401


# Database URL - using SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./wan_security.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args=({"check_same_thread": False} if "sqlite" in DATABASE_URL else {}),
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, Any, Any]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def init_database():
    """Initialize database with tables"""
    create_tables()
    print("Database initialized successfully")


def reset_database():
    """Reset database by dropping and recreating all tables"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset successfully")
