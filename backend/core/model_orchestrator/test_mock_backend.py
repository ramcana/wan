"""
Simple test to verify the mock backend works correctly.
"""

import tempfile
import shutil
from pathlib import Path
from backend.core.model_orchestrator.test_concurrent_ensure import MockStorageBackend
from backend.core.model_orchestrator.model_registry import FileSpec

def test_mock_backend_direct_call():
    """Test calling the mock backend directly."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        backend = MockStorageBackend(download_delay=0.1)
        
        # Create file specs
        file_specs = [
            FileSpec(path="config.json", size=1024, sha256="abc123"),
            FileSpec(path="model.bin", size=2048, sha256="def456")
        ]
        
        # Test direct call
        result = backend.download(
            source_url="mock://test.com/test-model",
            local_dir=temp_dir,
            file_specs=file_specs,
            allow_patterns=None,
            progress_callback=None
        )
        
        print(f"Direct call result: {result}")
        print(f"Success: {result.success}")
        print(f"Files downloaded: {result.files_downloaded}")
        print(f"Bytes downloaded: {result.bytes_downloaded}")
        
        # Check files exist
        config_file = temp_dir / "config.json"
        model_file = temp_dir / "model.bin"
        
        print(f"Config file exists: {config_file.exists()}")
        print(f"Model file exists: {model_file.exists()}")
        
        if config_file.exists():
            print(f"Config file size: {config_file.stat().st_size}")
        if model_file.exists():
            print(f"Model file size: {model_file.stat().st_size}")
        
        # Test keyword argument call (how retry_operation calls it)
        temp_dir2 = Path(tempfile.mkdtemp())
        try:
            result2 = backend.download(
                source_url="mock://test.com/test-model",
                local_dir=temp_dir2,
                file_specs=file_specs,
                allow_patterns=None,
                progress_callback=None
            )
            print(f"Keyword call result: {result2}")
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_mock_backend_direct_call()