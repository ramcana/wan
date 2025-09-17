import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from backend.models.auth import Base

# Create database engine
DATABASE_URL = "sqlite:///./wan_security.db"  # Default database URL
engine = create_engine(DATABASE_URL, echo=True)

# Create tables
Base.metadata.create_all(engine)

print("Security tables created successfully!")
