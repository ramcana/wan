"""Basic performance tests"""

import pytest
import time
import json
from pathlib import Path


@pytest.mark.performance
def test_json_parsing_performance():
    """Test that JSON parsing is reasonably fast"""
    # Create a moderately sized JSON object
    test_data = {
        "items": [{"id": i, "name": f"item_{i}", "value": i * 2} for i in range(1000)]
    }
    
    start_time = time.time()
    json_str = json.dumps(test_data)
    parsed_data = json.loads(json_str)
    end_time = time.time()
    
    # Should complete in less than 1 second
    duration = end_time - start_time
    assert duration < 1.0, f"JSON parsing took {duration:.2f}s, should be < 1.0s"
    assert parsed_data == test_data, "Parsed data should match original"


@pytest.mark.performance
def test_file_operations_performance():
    """Test that basic file operations are reasonably fast"""
    import tempfile
    
    start_time = time.time()
    
    # Create and write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write("test content\n" * 1000)
        tmp_path = tmp.name
    
    # Read the file back
    with open(tmp_path, 'r') as f:
        content = f.read()
    
    # Clean up
    Path(tmp_path).unlink()
    
    end_time = time.time()
    
    # Should complete in less than 1 second
    duration = end_time - start_time
    assert duration < 1.0, f"File operations took {duration:.2f}s, should be < 1.0s"
    assert len(content) > 0, "File content should not be empty"


@pytest.mark.performance
def test_import_performance():
    """Test that imports are reasonably fast"""
    start_time = time.time()
    
    # Import some common modules
    import os
    import sys
    import json
    import pathlib
    import tempfile
    import subprocess
    
    end_time = time.time()
    
    # Should complete in less than 2 seconds
    duration = end_time - start_time
    assert duration < 2.0, f"Imports took {duration:.2f}s, should be < 2.0s"