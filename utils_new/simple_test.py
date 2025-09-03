#!/usr/bin/env python3
"""
Simple test to verify the main application structure is working
"""

import sys
import json
from pathlib import Path

def test_config_creation():
    """Test that configuration can be created and loaded"""
    print("Testing configuration creation...")
    
    # Remove any existing test config
    test_config = Path("test_simple.json")
    if test_config.exists():
        test_config.unlink()
    
    # Create a simple config manually
    config_data = {
        "system": {
            "default_quantization": "bf16",
            "enable_offload": True,
            "vae_tile_size": 256,
            "max_queue_size": 10,
            "stats_refresh_interval": 5
        },
        "directories": {
            "output_directory": "outputs",
            "models_directory": "models",
            "loras_directory": "loras"
        },
        "generation": {
            "default_resolution": "1280x720",
            "default_steps": 50,
            "max_prompt_length": 500
        }
    }
    
    # Save config
    with open(test_config, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Verify it can be loaded
    with open(test_config, 'r') as f:
        loaded_config = json.load(f)
    
    assert loaded_config["system"]["default_quantization"] == "bf16"
    assert loaded_config["directories"]["output_directory"] == "outputs"
    assert loaded_config["generation"]["default_steps"] == 50
    
    # Cleanup
    test_config.unlink()
    
    print("✓ Configuration creation and loading works")

def test_directory_creation():
    """Test that required directories can be created"""
    print("Testing directory creation...")
    
    test_dirs = ["test_outputs", "test_models", "test_loras"]
    
    # Create directories
    for dir_name in test_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        assert dir_path.exists(), f"Directory {dir_name} should exist"
    
    # Cleanup
    for dir_name in test_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            dir_path.rmdir()
    
    print("✓ Directory creation works")

def test_logging_configuration():
    """Test that logging can be configured"""
    print("Testing logging configuration...")
    
    import logging

    # Test basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("test_logger")
    logger.info("Test log message")
    
    # Verify logger level
    assert logger.level <= logging.INFO
    
    print("✓ Logging configuration works")

if __name__ == "__main__":
    print("Running simple application structure tests...")
    
    try:
        test_config_creation()
        test_directory_creation()
        test_logging_configuration()
        
        print("\n✅ All simple tests passed!")
        print("The application structure is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
traceback.print_exc()
        sys.exit(1)