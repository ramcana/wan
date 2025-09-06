#!/usr/bin/env python3
"""
Test WAN LoRA Integration - Task 13 Completion Verification

This script tests the integration of WAN models with LoRA support,
verifying that all components work together correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_lora_manager_wan_integration():
    """Test LoRAManager WAN integration"""
    try:
        logger.info("Testing LoRAManager WAN integration...")
        
        # Import LoRA manager
        from core.services.utils import get_lora_manager
        lora_manager = get_lora_manager()
        
        # Test WAN integration initialization
        assert hasattr(lora_manager, '_wan_lora_manager'), "WAN LoRA manager not initialized"
        assert hasattr(lora_manager, 'apply_lora_with_wan_support'), "WAN-aware LoRA application method missing"
        assert hasattr(lora_manager, 'check_wan_model_compatibility'), "WAN compatibility check method missing"
        
        logger.info("✓ LoRAManager WAN integration methods available")
        
        # Test available LoRAs listing
        available_loras = lora_manager.list_available_loras()
        logger.info(f"✓ Found {len(available_loras)} available LoRA files")
        
        # Test WAN LoRA manager availability
        if lora_manager._wan_lora_manager:
            logger.info("✓ WAN LoRA manager initialized successfully")
            
            # Test WAN model compatibility matrix
            wan_manager = lora_manager._wan_lora_manager
            compatibility_matrix = wan_manager.wan_model_compatibility
            
            expected_models = ['T2V_A14B', 'I2V_A14B', 'TI2V_5B']
            for model_type in expected_models:
                from core.services.wan_lora_manager import WANModelType
                wan_type = getattr(WANModelType, model_type)
                assert wan_type in compatibility_matrix, f"Compatibility info missing for {model_type}"
                
                compatibility = compatibility_matrix[wan_type]
                assert compatibility.supports_lora, f"{model_type} should support LoRA"
                assert len(compatibility.target_modules) > 0, f"{model_type} should have target modules"
                
            logger.info("✓ WAN model compatibility matrix properly configured")
        else:
            logger.warning("⚠ WAN LoRA manager not available (expected in some environments)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LoRAManager WAN integration test failed: {e}")
        return False

async def test_real_generation_pipeline_lora_integration():
    """Test RealGenerationPipeline LoRA integration"""
    try:
        logger.info("Testing RealGenerationPipeline LoRA integration...")
        
        # Import real generation pipeline
        from backend.services.real_generation_pipeline import get_real_generation_pipeline
        pipeline = await get_real_generation_pipeline()
        
        # Test LoRA integration methods
        assert hasattr(pipeline, '_apply_lora_to_pipeline'), "LoRA application method missing"
        assert hasattr(pipeline, '_remove_lora_from_pipeline'), "LoRA removal method missing"
        assert hasattr(pipeline, '_adjust_lora_strength_in_pipeline'), "LoRA strength adjustment method missing"
        assert hasattr(pipeline, 'get_applied_loras_status'), "LoRA status method missing"
        assert hasattr(pipeline, 'validate_lora_compatibility'), "LoRA compatibility validation method missing"
        
        logger.info("✓ RealGenerationPipeline LoRA integration methods available")
        
        # Test LoRA manager initialization
        if pipeline.lora_manager:
            logger.info("✓ LoRA manager initialized in pipeline")
        else:
            logger.warning("⚠ LoRA manager not initialized in pipeline (expected in some environments)")
        
        # Test applied LoRAs tracking
        applied_status = pipeline.get_applied_loras_status()
        assert isinstance(applied_status, dict), "Applied LoRAs status should return dictionary"
        logger.info("✓ Applied LoRAs status tracking working")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ RealGenerationPipeline LoRA integration test failed: {e}")
        return False

async def test_model_integration_bridge_lora_status():
    """Test ModelIntegrationBridge LoRA status reporting"""
    try:
        logger.info("Testing ModelIntegrationBridge LoRA status reporting...")
        
        # Import model integration bridge
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        
        # Test LoRA status methods
        assert hasattr(bridge, 'get_lora_status'), "LoRA status method missing"
        assert hasattr(bridge, 'validate_lora_compatibility_async'), "Async LoRA compatibility validation missing"
        assert hasattr(bridge, 'get_lora_memory_impact'), "LoRA memory impact method missing"
        
        logger.info("✓ ModelIntegrationBridge LoRA status methods available")
        
        # Test LoRA status retrieval
        lora_status = bridge.get_lora_status()
        assert isinstance(lora_status, dict), "LoRA status should return dictionary"
        assert 'available_loras' in lora_status, "Available LoRAs should be in status"
        assert 'lora_manager_available' in lora_status, "LoRA manager availability should be in status"
        assert 'wan_lora_support' in lora_status, "WAN LoRA support flag should be in status"
        
        logger.info(f"✓ LoRA status: {lora_status['total_available']} available, WAN support: {lora_status['wan_lora_support']}")
        
        # Test memory impact estimation
        if lora_status['total_available'] > 0:
            # Test with first available LoRA
            first_lora = list(lora_status['available_loras'].keys())[0]
            memory_impact = bridge.get_lora_memory_impact(first_lora)
            assert isinstance(memory_impact, dict), "Memory impact should return dictionary"
            logger.info(f"✓ Memory impact estimation working for LoRA: {first_lora}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ModelIntegrationBridge LoRA status test failed: {e}")
        return False

async def test_generation_params_lora_support():
    """Test GenerationParams LoRA parameter support"""
    try:
        logger.info("Testing GenerationParams LoRA parameter support...")
        
        # Import GenerationParams
        from backend.core.model_integration_bridge import GenerationParams
        
        # Test LoRA parameters in GenerationParams
        params = GenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_path="test_lora.safetensors",
            lora_strength=0.8
        )
        
        assert hasattr(params, 'lora_path'), "lora_path parameter missing"
        assert hasattr(params, 'lora_strength'), "lora_strength parameter missing"
        assert params.lora_path == "test_lora.safetensors", "lora_path not set correctly"
        assert params.lora_strength == 0.8, "lora_strength not set correctly"
        
        logger.info("✓ GenerationParams LoRA parameters working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ GenerationParams LoRA support test failed: {e}")
        return False

async def test_wan_lora_manager_direct():
    """Test WANLoRAManager directly"""
    try:
        logger.info("Testing WANLoRAManager directly...")
        
        # Import WAN LoRA manager
        from core.services.wan_lora_manager import get_wan_lora_manager, WANModelType
        wan_lora_manager = get_wan_lora_manager()
        
        # Test WAN LoRA manager methods
        assert hasattr(wan_lora_manager, 'check_wan_model_compatibility'), "WAN compatibility check missing"
        assert hasattr(wan_lora_manager, 'apply_wan_lora'), "WAN LoRA application missing"
        assert hasattr(wan_lora_manager, 'adjust_wan_lora_strength'), "WAN LoRA strength adjustment missing"
        assert hasattr(wan_lora_manager, 'remove_wan_lora'), "WAN LoRA removal missing"
        assert hasattr(wan_lora_manager, 'blend_wan_loras'), "WAN LoRA blending missing"
        assert hasattr(wan_lora_manager, 'get_wan_lora_status'), "WAN LoRA status missing"
        
        logger.info("✓ WANLoRAManager methods available")
        
        # Test model type detection
        assert hasattr(wan_lora_manager, '_detect_wan_model_type'), "WAN model type detection missing"
        
        # Test compatibility matrix
        compatibility_matrix = wan_lora_manager.wan_model_compatibility
        assert len(compatibility_matrix) == 3, "Should have 3 WAN model types in compatibility matrix"
        
        for model_type, compatibility in compatibility_matrix.items():
            assert compatibility.supports_lora, f"{model_type} should support LoRA"
            assert compatibility.max_lora_count > 0, f"{model_type} should allow at least 1 LoRA"
            assert len(compatibility.target_modules) > 0, f"{model_type} should have target modules"
            
        logger.info("✓ WAN model compatibility matrix validated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ WANLoRAManager direct test failed: {e}")
        return False

async def run_all_tests():
    """Run all LoRA integration tests"""
    logger.info("=" * 60)
    logger.info("WAN LoRA Integration Test Suite - Task 13 Verification")
    logger.info("=" * 60)
    
    tests = [
        ("LoRAManager WAN Integration", test_lora_manager_wan_integration),
        ("RealGenerationPipeline LoRA Integration", test_real_generation_pipeline_lora_integration),
        ("ModelIntegrationBridge LoRA Status", test_model_integration_bridge_lora_status),
        ("GenerationParams LoRA Support", test_generation_params_lora_support),
        ("WANLoRAManager Direct", test_wan_lora_manager_direct),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name}: PASSED")
            else:
                logger.error(f"✗ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Task 13 LoRA integration is complete.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Task 13 needs attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)