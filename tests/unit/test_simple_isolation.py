"""
Simple test for basic isolation functionality.
"""

import pytest
import tempfile
import sqlite3
import os
from pathlib import Path


def test_basic_database_isolation():
    """Test basic database isolation without complex dependencies."""
    # Create temporary database
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test.db"
    
    try:
        # Create database and table
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users (name) VALUES (?)", ("test_user",))
        conn.commit()
        
        # Verify data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        assert count == 1
        
        conn.close()
        
        # Verify database file exists
        assert db_path.exists()
        
    finally:
        # Cleanup
        if db_path.exists():
            db_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()


def test_basic_environment_isolation():
    """Test basic environment variable isolation."""
    # Store original value
    original_value = os.environ.get("TEST_ISOLATION_VAR")
    
    try:
        # Set test value
        os.environ["TEST_ISOLATION_VAR"] = "test_value"
        assert os.environ.get("TEST_ISOLATION_VAR") == "test_value"
        
        # Modify value
        os.environ["TEST_ISOLATION_VAR"] = "modified_value"
        assert os.environ.get("TEST_ISOLATION_VAR") == "modified_value"
        
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop("TEST_ISOLATION_VAR", None)
        else:
            os.environ["TEST_ISOLATION_VAR"] = original_value


def test_basic_file_isolation():
    """Test basic file system isolation."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test files
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "app.json"
        config_file.write_text('{"test": true}')
        
        # Verify files exist
        assert test_file.exists()
        assert config_dir.exists()
        assert config_file.exists()
        
        # Verify content
        assert test_file.read_text() == "test content"
        assert "test" in config_file.read_text()
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])