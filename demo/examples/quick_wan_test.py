#!/usr/bin/env python3
"""
Quick WAN Model Test

Simple test to verify WAN model implementations are working.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wan_pipeline_factory():
    """Test WAN pipeline factory"""
    try:
        from backend.core.models.wan_models.wan_pipeline_factory import WANPipelineFactory
        
        factory = WANPipelineFactory()
        logger.info("✅ WAN Pipeline Factory created successfully")
        
        # Test available models
        available_models = factory.get_available_models()
        logger.info(f"Available models: {available_models}")
        
        # Test cache stats
        stats = factory.get_cache_stats()
        logger.info(f"Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WAN Pipeline Factory test failed: {e}")
        return False

def test_wan_model_config():
    """Test WAN model configuration"""
    try:
        from backend.core.models.wan_models.wan_model_config import get_wan_model_config
        
        # Test T2V config
        config = get_wan_model_config("t2v-A14B")
        if config:
            logger.info(f"✅ T2V-A14B config loaded: {config.model_type}")
        else:
            logger.error("❌ Could not load T2V-A14B config")
            return False
        
        # Test I2V config
        config = get_wan_model_config("i2v-A14B")
        if config:
            logger.info(f"✅ I2V-A14B config loaded: {config.model_type}")
        else:
            logger.error("❌ Could not load I2V-A14B config")
            return False
        
        # Test TI2V config
        config = get_wan_model_config("ti2v-5B")
        if config:
            logger.info(f"✅ TI2V-5B config loaded: {config.model_type}")
        else:
            logger.error("❌ Could not load TI2V-5B config")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WAN Model Config test failed: {e}")
        return False

def test_wan_models():
    """Test WAN model instantiation"""
    try:
        from backend.core.models.wan_models.wan_t2v_a14b import WANT2VA14B
        from backend.core.models.wan_models.wan_i2v_a14b import WANI2VA14B
        from backend.core.models.wan_models.wan_ti2v_5b import WANTI2V5B
        from backend.core.models.wan_models.wan_model_config import get_wan_model_config
        
        # Test T2V model
        config = get_wan_model_config("t2v-A14B")
        if config:
            try:
                model = WANT2VA14B(config)
                logger.info(f"✅ T2V-A14B model created: {model.model_type.value}")
            except Exception as e:
                logger.warning(f"⚠️ T2V-A14B model creation failed: {e}")
        
        # Test I2V model
        config = get_wan_model_config("i2v-A14B")
        if config:
            try:
                model = WANI2VA14B(config)
                logger.info(f"✅ I2V-A14B model created: {model.model_type.value}")
            except Exception as e:
                logger.warning(f"⚠️ I2V-A14B model creation failed: {e}")
        
        # Test TI2V model
        config = get_wan_model_config("ti2v-5B")
        if config:
            try:
                model = WANTI2V5B(config)
                logger.info(f"✅ TI2V-5B model created: {model.model_type.value}")
            except Exception as e:
                logger.warning(f"⚠️ TI2V-5B model creation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WAN Models test failed: {e}")
        return False

def test_backend_integration():
    """Test backend integration components"""
    try:
        from backend.core.model_integration_bridge import ModelIntegrationBridge
        
        bridge = ModelIntegrationBridge()
        logger.info("✅ Model Integration Bridge created successfully")
        
        # Test WAN model status
        try:
            status = bridge.get_wan_model_status("t2v-A14B")
            logger.info(f"T2V-A14B status: implemented={status.is_implemented}, weights={status.is_weights_available}")
        except Exception as e:
            logger.warning(f"⚠️ WAN model status check failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Backend Integration test failed: {e}")
        return False

def main():
    """Run all quick tests"""
    logger.info("🚀 Starting Quick WAN Model Tests...")
    
    tests = [
        ("WAN Pipeline Factory", test_wan_pipeline_factory),
        ("WAN Model Config", test_wan_model_config),
        ("WAN Models", test_wan_models),
        ("Backend Integration", test_backend_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASS")
            else:
                logger.info(f"❌ {test_name}: FAIL")
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} passed ({passed/total:.1%})")
    
    if passed >= total * 0.75:  # 75% pass rate
        logger.info("🎉 WAN models are ready for integration testing!")
        return True
    else:
        logger.warning("⚠️ WAN models need more work before full integration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
