#!/usr/bin/env python3
"""
Test script to validate the Fallback and Recovery System implementation
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_fallback_recovery_system():
    """Test the fallback and recovery system implementation"""
    print("Testing Fallback and Recovery System Implementation...")
    
    try:
        # Test imports
        print("✓ Testing imports...")
        from backend.core.fallback_recovery_system import (
            FallbackRecoverySystem, FailureType, RecoveryAction, 
            get_fallback_recovery_system, initialize_fallback_recovery_system
        )
        print("  ✓ All imports successful")
        
        # Test system initialization
        print("✓ Testing system initialization...")
        recovery_system = FallbackRecoverySystem()
        assert recovery_system is not None
        print("  ✓ FallbackRecoverySystem created successfully")
        
        # Test failure types and recovery actions
        print("✓ Testing failure types and recovery actions...")
        assert len(list(FailureType)) == 6
        assert len(list(RecoveryAction)) == 8
        print(f"  ✓ {len(list(FailureType))} failure types defined")
        print(f"  ✓ {len(list(RecoveryAction))} recovery actions defined")
        
        # Test recovery strategies initialization
        print("✓ Testing recovery strategies...")
        strategies = recovery_system.recovery_strategies
        assert len(strategies) == len(list(FailureType))
        for failure_type in FailureType:
            assert failure_type in strategies
            assert len(strategies[failure_type]) > 0
        print("  ✓ All failure types have recovery strategies")
        
        # Test system health status
        print("✓ Testing system health status...")
        health_status = await recovery_system.get_system_health_status()
        assert health_status is not None
        assert health_status.overall_status in ["healthy", "degraded", "critical", "error"]
        print(f"  ✓ System health status: {health_status.overall_status}")
        
        # Test recovery statistics
        print("✓ Testing recovery statistics...")
        stats = recovery_system.get_recovery_statistics()
        assert "total_attempts" in stats
        assert "success_rate" in stats
        assert stats["total_attempts"] == 0  # No attempts yet
        print("  ✓ Recovery statistics generated successfully")
        
        # Test global instance functions
        print("✓ Testing global instance functions...")
        global_system = get_fallback_recovery_system()
        assert global_system is not None
        print("  ✓ Global fallback recovery system accessible")
        
        # Test mock generation fallback
        print("✓ Testing mock generation fallback...")
        # Create a mock generation service
        class MockGenerationService:
            def __init__(self):
                self.use_real_generation = True
                self.fallback_to_simulation = True
        
        mock_service = MockGenerationService()
        recovery_system.generation_service = mock_service
        
        success = await recovery_system._fallback_to_mock_generation()
        assert success is True
        assert mock_service.use_real_generation is False
        print("  ✓ Mock generation fallback works correctly")
        
        # Test GPU cache clearing (will work even without GPU)
        print("✓ Testing GPU cache clearing...")
        cache_success = await recovery_system._clear_gpu_cache()
        print(f"  ✓ GPU cache clearing: {'successful' if cache_success else 'no CUDA available'}")
        
        # Test parameter reduction
        print("✓ Testing parameter reduction...")
        context = {
            "generation_params": {
                "resolution": "1920x1080",
                "steps": 50,
                "num_frames": 32
            }
        }
        param_success = await recovery_system._reduce_generation_parameters(context)
        assert param_success is True
        assert context["generation_params"]["resolution"] == "1280x720"
        print("  ✓ Parameter reduction works correctly")
        
        # Test recovery state reset
        print("✓ Testing recovery state reset...")
        recovery_system.mock_generation_enabled = True
        recovery_system.reset_recovery_state()
        assert recovery_system.mock_generation_enabled is False
        print("  ✓ Recovery state reset works correctly")
        
        print("\n🎉 All Fallback and Recovery System tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_integration():
    """Test API integration"""
    print("\nTesting API Integration...")
    
    try:
        # Test FastAPI app imports
        print("✓ Testing FastAPI app integration...")
        from backend.app import app
        print("  ✓ FastAPI app imports successfully")
        
        # Test that recovery endpoints are available
        routes = [route.path for route in app.routes]
        recovery_routes = [route for route in routes if "/recovery/" in route]
        
        expected_routes = [
            "/api/v1/recovery/status",
            "/api/v1/recovery/health", 
            "/api/v1/recovery/trigger",
            "/api/v1/recovery/reset",
            "/api/v1/recovery/actions"
        ]
        
        for expected_route in expected_routes:
            assert expected_route in routes, f"Missing route: {expected_route}"
        
        print(f"  ✓ {len(recovery_routes)} recovery API endpoints available")
        
        print("\n🎉 API integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ API integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_integration_bridge():
    """Test model integration bridge enhancements"""
    print("\nTesting Model Integration Bridge Enhancements...")
    
    try:
        # Test retry logic imports
        print("✓ Testing retry logic imports...")
        from backend.core.model_integration_bridge import ModelIntegrationBridge
        
        bridge = ModelIntegrationBridge()
        assert hasattr(bridge, '_retry_config')
        assert hasattr(bridge, '_retry_with_exponential_backoff')
        assert hasattr(bridge, 'ensure_model_available_with_retry')
        print("  ✓ Retry logic methods available")
        
        # Test retry configuration
        retry_config = bridge._retry_config
        assert "model_download" in retry_config
        assert "model_loading" in retry_config
        print("  ✓ Retry configuration properly initialized")
        
        print("\n🎉 Model Integration Bridge enhancement tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model Integration Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_generation_service_integration():
    """Test generation service integration"""
    print("\nTesting Generation Service Integration...")
    
    try:
        # Test generation service imports
        print("✓ Testing generation service imports...")
        from backend.services.generation_service import GenerationService
        
        service = GenerationService()
        assert hasattr(service, 'fallback_recovery_system')
        assert hasattr(service, '_determine_failure_type')
        assert hasattr(service, '_run_mock_generation')
        print("  ✓ Generation service has fallback recovery integration")
        
        # Test failure type determination
        print("✓ Testing failure type determination...")
        model_loading_error = Exception("Model not found")
        failure_type = service._determine_failure_type(model_loading_error, "t2v-A14B")
        from backend.core.fallback_recovery_system import FailureType
        assert failure_type == FailureType.MODEL_LOADING_FAILURE
        print("  ✓ Failure type determination works correctly")
        
        print("\n🎉 Generation Service integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Generation Service integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("FALLBACK AND RECOVERY SYSTEM IMPLEMENTATION TEST")
    print("=" * 60)
    
    tests = [
        test_fallback_recovery_system,
        test_api_integration,
        test_model_integration_bridge,
        test_generation_service_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Fallback and Recovery System implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())