"""
Database configuration and models using SQLAlchemy
"""

import os
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

# Database URL - using SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./wan22_tasks.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class TaskStatusEnum(enum.Enum):
    """Task status enumeration for database"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelTypeEnum(enum.Enum):
    """Model type enumeration for database"""
    T2V_A14B = "T2V-A14B"
    I2V_A14B = "I2V-A14B"
    TI2V_5B = "TI2V-5B"

class GenerationTaskDB(Base):
    """Database model for generation tasks"""
    __tablename__ = "generation_tasks"
    
    id = Column(String, primary_key=True, index=True)
    model_type = Column(Enum(ModelTypeEnum), nullable=False)
    prompt = Column(Text, nullable=False)
    image_path = Column(String, nullable=True)
    end_image_path = Column(String, nullable=True)  # For I2V and TI2V interpolation
    resolution = Column(String, nullable=False, default="1280x720")
    steps = Column(Integer, nullable=False, default=50)
    lora_path = Column(String, nullable=True)
    lora_strength = Column(Float, nullable=False, default=1.0)
    status = Column(Enum(TaskStatusEnum), nullable=False, default=TaskStatusEnum.PENDING)
    progress = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    output_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    estimated_time_minutes = Column(Integer, nullable=True)
    generation_time_minutes = Column(Float, nullable=True)

class SystemStatsDB(Base):
    """Database model for system statistics (for historical data)"""
    __tablename__ = "system_stats"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cpu_percent = Column(Float, nullable=False)
    ram_used_gb = Column(Float, nullable=False)
    ram_total_gb = Column(Float, nullable=False)
    ram_percent = Column(Float, nullable=False)
    gpu_percent = Column(Float, nullable=False)
    vram_used_mb = Column(Float, nullable=False)
    vram_total_mb = Column(Float, nullable=False)
    vram_percent = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

def get_db():
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