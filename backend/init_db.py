#!/usr/bin/env python3
"""
Initialize the database with required tables
"""

from backend.repositories.database import init_database

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    print("Database initialization complete!")