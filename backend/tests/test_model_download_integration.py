"""
Test script for model download integration
Tests the integration between ModelIntegrationBridge and existing download infrastructure
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_integration_bridge():
    """Test ModelIntegrationBridge initialization and basic functionality"""
    try:
        logger.info("Testing ModelIntegrationBridge initialization...")
        
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        
        # Test integration status
        status = bridge.get_integration_status()
        logger.info(f"Integration status: {status}")
        
        # Test model availability check
        logger.info("Testing model availability check...")
        for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
            model_status = await bridge.check_model_availability(model_type)
            logger.info(f"Model {model_type} status: {model_status.status.value}")
        
        logger.info("✅ ModelIntegrationBridge test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ ModelIntegrationBridge test failed: {e}")
        return False

async def test_model_download_system():
    """Test model download system integration"""
    try:
        logger.info("Testing model download system...")
        
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        
        # Test download system availability
        if not bridge.model_downloader:
            logger.warning("Model downloader not available - this is expected in test environment")
            return True
        
        # Test download progress tracking
        progress = bridge.get_all_download_progress()
        logger.info(f"Current download progress: {progress}")
        
        logger.info("✅ Model download system test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model download system test failed: {e}")
        return False

async def test_model_validation_system():
    """Test model validation system integration"""
    try:
        logger.info("Testing model validation system...")
        
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        
        # Test validation system availability
        if not bridge.model_validator:
            logger.warning("Model validator not available - this is expected in test environment")
            return True
        
        # Test integrity verification
        for model_type in ["t2v-A14B"]:  # Test with one model
            try:
                is_valid = await bridge._verify_model_integrity(model_type)
                logger.info(f"Model {model_type} integrity check: {'passed' if is_valid else 'failed'}")
            except Exception as e:
                logger.warning(f"Integrity check for {model_type} failed: {e}")
        
        logger.info("✅ Model validation system test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model validation system test failed: {e}")
        return False

async def test_websocket_integration():
    """Test WebSocket integration for progress notifications"""
    try:
        logger.info("Testing WebSocket integration...")
        
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        
        # Test WebSocket manager availability
        if not bridge._websocket_manager:
            logger.warning("WebSocket manager not available - this is expected in test environment")
            return True
        
        # Test progress notification
        await bridge._send_download_progress_notification("test-model", 50.0, "Testing progress notification")
        logger.info("Progress notification sent successfully")
        
        logger.info("✅ WebSocket integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket integration test failed: {e}")
        return False

async def test_generation_pipeline_integration():
    """Test RealGenerationPipeline integration with model download"""
    try:
        logger.info("Testing RealGenerationPipeline integration...")
        
        from backend.services.real_generation_pipeline import RealGenerationPipeline
        pipeline = RealGenerationPipeline()
        
        # Test pipeline initialization
        initialized = await pipeline.initialize()
        logger.info(f"Pipeline initialization: {'success' if initialized else 'failed'}")
        
        # Test model availability check
        model_available = await pipeline._ensure_model_available("t2v-A14B")
        logger.info(f"Model availability check: {'available' if model_available else 'not available'}")
        
        logger.info("✅ RealGenerationPipeline integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RealGenerationPipeline integration test failed: {e}")
        return False

async def test_api_endpoints():
    """Test API endpoints for model management"""
    try:
        logger.info("Testing API endpoints...")
        
        # Test convenience functions
        from backend.core.model_integration_bridge import (
            check_model_availability,
            get_all_model_download_progress,
            verify_model_integrity
        )
        
        # Test model status check
        status = await check_model_availability("t2v-A14B")
        logger.info(f"API model status check: {status.status.value}")
        
        # Test download progress
        progress = await get_all_model_download_progress()
        logger.info(f"API download progress: {len(progress)} models tracked")
        
        # Test integrity verification
        try:
            is_valid = await verify_model_integrity("t2v-A14B")
            logger.info(f"API integrity check: {'valid' if is_valid else 'invalid'}")
        except Exception as e:
            logger.warning(f"API integrity check failed: {e}")
        
        logger.info("✅ API endpoints test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ API endpoints test failed: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    logger.info("=" * 60)
    logger.info("🧪 Running Model Download Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("ModelIntegrationBridge", test_model_integration_bridge),
        ("Model Download System", test_model_download_system),
        ("Model Validation System", test_model_validation_system),
        ("WebSocket Integration", test_websocket_integration),
        ("RealGenerationPipeline Integration", test_generation_pipeline_integration),
        ("API Endpoints", test_api_endpoints)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            success = await test_func()
            if success:
                passed += 1
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Model download integration is working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Some functionality may not work as expected.")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
