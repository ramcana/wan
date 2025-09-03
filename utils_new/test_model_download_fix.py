#!/usr/bin/env python3
"""
Test script to verify model download fixes
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.services.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_download():
    """Test model download with fixes"""
    logger.info("Testing model download with fixes...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager("config.json")
        
        # Test downloading a small model first (TI2V is smaller)
        logger.info("Testing TI2V model download...")
        model_path = model_manager.download_model("ti2v-5B", timeout=600)
        
        logger.info(f"✅ Model downloaded successfully to: {model_path}")
        
        # Verify the model exists
        if Path(model_path).exists():
            logger.info("✅ Model path exists")
            
            # Check for key files
            model_dir = Path(model_path)
            key_files = ["config.json", "model_index.json"]
            
            for file in key_files:
                if (model_dir / file).exists():
                    logger.info(f"✅ Found {file}")
                else:
                    logger.warning(f"⚠️ Missing {file}")
            
            return True
        else:
            logger.error("❌ Model path does not exist")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model download test failed: {e}")
        return False

def test_model_loading():
    """Test model loading after download"""
    logger.info("Testing model loading...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager("config.json")
        
        # Try to load the model
        logger.info("Loading TI2V model...")
        pipeline, model_info = model_manager.load_model("ti2v-5B")
        
        logger.info(f"✅ Model loaded successfully: {model_info.model_type}")
        logger.info(f"Memory usage: {model_info.memory_usage_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run model download and loading tests"""
    logger.info("Starting model download and loading tests...")
    
    tests = [
        ("Model Download", test_model_download),
        ("Model Loading", test_model_loading)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running: {test_name} ---")
            result = test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n--- SUMMARY ---")
    logger.info(f"Passed: {passed}, Failed: {failed}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)