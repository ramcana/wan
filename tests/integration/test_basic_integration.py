"""Basic integration tests that should pass"""

import pytest
import json
from pathlib import Path


def test_project_structure():
    """Test that the project has the expected structure"""
    required_dirs = ["backend", "frontend", "tests", "scripts", "tools"]
    
    for dir_name in required_dirs:
        assert Path(dir_name).exists(), f"Directory {dir_name} should exist"


def test_config_file_valid():
    """Test that config.json is valid JSON"""
    config_path = Path("config.json")
    assert config_path.exists(), "config.json should exist"
    
    with open(config_path) as f:
        config = json.load(f)
    
    assert isinstance(config, dict), "Config should be a dictionary"


def test_requirements_file_readable():
    """Test that requirements.txt is readable"""
    req_path = Path("requirements.txt")
    assert req_path.exists(), "requirements.txt should exist"
    
    with open(req_path) as f:
        content = f.read()
    
    assert len(content) > 0, "Requirements file should not be empty"
    assert "pytest" in content, "Requirements should include pytest"


def test_basic_imports():
    """Test that basic imports work without errors"""
    import sys
    import os
    import json
    import pathlib
    
    # These should not raise any exceptions
    assert sys.version_info >= (3, 8), "Python version should be 3.8 or higher"
    assert os.path.exists("."), "Current directory should exist"
    assert json.dumps({"test": True}) == '{"test": true}', "JSON should work"
    assert pathlib.Path(".").exists(), "Pathlib should work"


@pytest.mark.integration
def test_test_framework_works():
    """Test that the test framework itself is working"""
    # This test validates that pytest is working correctly
    assert True, "Basic assertion should work"
    
    # Test that pytest markers work
    assert hasattr(pytest, "mark"), "Pytest marks should be available"
    
    # Test that we can create temporary files
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        assert tmp.name is not None, "Temporary files should work"