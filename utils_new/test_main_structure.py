#!/usr/bin/env python3
"""
Test script to verify main.py structure without heavy dependencies
"""

import sys
import json
from pathlib import Path

# Test the ApplicationConfig class
sys.path.insert(0, '.')

def test_application_config():
    """Test the ApplicationConfig class"""
    print("Testing ApplicationConfig...")
    
    # Import just the config class
    from main import ApplicationConfig
    
    # Test with non-existent config file
    test_config_path = "test_config.json"
    if Path(test_config_path).exists():
        Path(test_config_path).unlink()
    
    config = ApplicationConfig(test_config_path)
    
    # Verify default config was created
    assert Path(test_config_path).exists(), "Config file should be created"
    
    # Verify config structure
    config_data = config.get_config()
    required_sections = ["system", "directories", "generation", "models", "optimization", "ui", "performance", "prompt_enhancement"]
    
    for section in required_sections:
        assert section in config_data, f"Missing config section: {section}"
    
    # Test config validation
    config.config["system"]["default_quantization"] = "invalid"
    config._validate_config()
    assert config.config["system"]["default_quantization"] == "bf16", "Invalid quantization should be corrected"
    
    # Test directory creation
    config._ensure_directories()
    for dir_name in ["outputs", "models", "loras"]:
        assert Path(dir_name).exists(), f"Directory {dir_name} should be created"
    
    # Cleanup
    Path(test_config_path).unlink()
    for dir_name in ["outputs", "models", "loras"]:
        if Path(dir_name).exists() and not any(Path(dir_name).iterdir()):
            Path(dir_name).rmdir()
    
    print("✓ ApplicationConfig tests passed")

def test_argument_parsing():
    """Test command-line argument parsing"""
    print("Testing argument parsing...")
    
    from main import parse_arguments
    
    # Mock sys.argv for testing
    original_argv = sys.argv
    
    try:
        # Test default arguments
        sys.argv = ["main.py"]
        args = parse_arguments()
        assert args.config == "config.json"
        assert args.host == "127.0.0.1"
        assert args.port == 7860
        assert not args.share
        assert not args.debug
        
        # Test custom arguments
        sys.argv = ["main.py", "--config", "custom.json", "--port", "8080", "--share", "--debug"]
        args = parse_arguments()
        assert args.config == "custom.json"
        assert args.port == 8080
        assert args.share
        assert args.debug
        
        print("✓ Argument parsing tests passed")
        
    finally:
        sys.argv = original_argv

def test_logging_setup():
    """Test logging setup"""
    print("Testing logging setup...")
    
    from main import setup_logging
    import logging
    
    # Test normal logging
    setup_logging(debug=False)
    assert logging.getLogger().level == logging.INFO
    
    # Test debug logging
    setup_logging(debug=True)
    assert logging.getLogger().level == logging.DEBUG
    
    print("✓ Logging setup tests passed")

if __name__ == "__main__":
    print("Testing main.py structure...")
    
    try:
        test_application_config()
        test_argument_parsing()
        test_logging_setup()
        
        print("\n✅ All main.py structure tests passed!")
        print("The application entry point is properly implemented.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)