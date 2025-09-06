#!/usr/bin/env python3
"""
Test WAN Model Error Handler Requirements

This test validates that the WAN Model Error Handler fully meets the requirements:
- 7.1: Specific error messages about actual models and suggest solutions
- 7.2: CUDA memory errors with model-specific optimization strategies  
- 7.3: Categorize errors and provide recovery suggestions
- 7.4: Fall back to alternative models or mock generation with clear notifications

Requirements tested: 7.1, 7.2, 7.3, 7.4
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_requirement_7_1_specific_error_messages():
    """Test Requirement 7.1: Specific error messages about actual models"""
    print("\n=== Testing Requirement 7.1: Specific Error Messages ===")
    
    try:
        from core.models.wan_models.wan_model_error_handler import (
            WANModelErrorHandler, WANErrorContext, WANModelType
        )
        
        handler = WANModelErrorHandler()
        
        # Test model loading error with specific model
        context = WANErrorContext(
            model_type=WANModelType.T2V_A14B,
            model_loaded=False,
            weights_loaded=False,
            error_stage="model_loading"
        )
        
        error = Exception("Failed to load WAN T2V-A14B model weights")
        result = await handler.handle_wan_error(error, context)
        
        # Verify specific model mentioned in error message
        assert "T2V-A14B" in result.title, f"Expected T2V-A14B in title: {result.title}"
        assert "T2V-A14B" in result.message, f"Expected T2V-A14B in message: {result.message}"
        
        # Verify specific solutions are provided
        assert len(result.recovery_suggestions) > 0, "Expected recovery suggestions"
        assert any("download" in suggestion.lower() for suggestion in result.recovery_suggestions), \
            "Expected download suggestion for model loading error"
        
        print("✓ Requirement 7.1: Specific error messages about actual models - PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 7.1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_requirement_7_2_cuda_memory_optimization():
    """Test Requirement 7.2: CUDA memory errors with model-specific optimization strategies"""
    print("\n=== Testing Requirement 7.2: CUDA Memory Optimization ===")
    
    try:
        from core.models.wan_models.wan_model_error_handler import (
            WANModelErrorHandler, WANErrorContext, WANModelType
        )
        
        handler = WANModelErrorHandler()
        
        # Test CUDA memory error
        context = WANErrorContext(
            model_type=WANModelType.T2V_A14B,
            model_loaded=True,
            weights_loaded=True,
            error_stage="inference",
            vram_usage_gb=12.0,
            available_vram_gb=8.0
        )
        
        error = Exception("CUDA out of memory error during T2V-A14B inference")
        result = await handler.handle_wan_error(error, context)
        
        # Verify CUDA memory optimization strategies are provided
        cuda_suggestions = [s for s in result.recovery_suggestions if "cpu offload" in s.lower() or "quantization" in s.lower() or "vram" in s.lower()]
        assert len(cuda_suggestions) > 0, f"Expected CUDA memory optimization suggestions, got: {result.recovery_suggestions}"
        
        # Verify model-specific optimizations
        t2v_suggestions = [s for s in result.recovery_suggestions if "T2V-A14B" in s or "gradient checkpointing" in s.lower()]
        assert len(t2v_suggestions) > 0, f"Expected T2V-A14B specific optimizations, got: {result.recovery_suggestions}"
        
        print("✓ Requirement 7.2: CUDA memory errors with model-specific optimization - PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 7.2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_requirement_7_3_error_categorization():
    """Test Requirement 7.3: Categorize errors and provide recovery suggestions"""
    print("\n=== Testing Requirement 7.3: Error Categorization ===")
    
    try:
        from core.models.wan_models.wan_model_error_handler import (
            WANModelErrorHandler, WANErrorContext, WANModelType, WANErrorCategory
        )
        
        handler = WANModelErrorHandler()
        
        # Test different error types get categorized correctly
        test_cases = [
            ("model loading", "Failed to load checkpoint", "model_loading"),
            ("inference", "CUDA out of memory", "inference"),
            ("inference", "temporal attention failed", "inference"),
            ("parameter_validation", "invalid resolution", "parameter_validation")
        ]
        
        for stage, error_msg, expected_stage in test_cases:
            context = WANErrorContext(
                model_type=WANModelType.I2V_A14B,
                error_stage=stage
            )
            
            error = Exception(error_msg)
            result = await handler.handle_wan_error(error, context)
            
            # Verify error is categorized
            assert result.category is not None, f"Expected error category for {error_msg}"
            
            # Verify recovery suggestions are provided
            assert len(result.recovery_suggestions) > 0, f"Expected recovery suggestions for {error_msg}"
            
            # Verify error code is generated
            assert result.error_code is not None, f"Expected error code for {error_msg}"
            assert "WAN_" in result.error_code, f"Expected WAN prefix in error code: {result.error_code}"
        
        print("✓ Requirement 7.3: Error categorization and recovery suggestions - PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 7.3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_requirement_7_4_fallback_strategies():
    """Test Requirement 7.4: Fall back to alternative models or mock generation"""
    print("\n=== Testing Requirement 7.4: Fallback Strategies ===")
    
    try:
        from core.models.wan_models.wan_model_error_handler import (
            WANModelErrorHandler, WANErrorContext, WANModelType, WANErrorSeverity
        )
        
        handler = WANModelErrorHandler()
        
        # Test critical error that should trigger fallback
        context = WANErrorContext(
            model_type=WANModelType.T2V_A14B,
            model_loaded=False,
            weights_loaded=False,
            error_stage="model_loading"
        )
        
        error = Exception("Critical model corruption - T2V-A14B unusable")
        result = await handler.handle_wan_error(error, context)
        
        # Verify fallback suggestions are provided
        fallback_suggestions = [s for s in result.recovery_suggestions if "fallback" in s.lower() or "mock" in s.lower() or "alternative" in s.lower()]
        assert len(fallback_suggestions) > 0, f"Expected fallback suggestions, got: {result.recovery_suggestions}"
        
        # Verify clear notifications about fallback mode
        notification_suggestions = [s for s in result.recovery_suggestions if "mock generation" in s.lower() or "placeholder" in s.lower()]
        mock_mentioned = any("mock" in s.lower() for s in result.recovery_suggestions)
        
        if mock_mentioned:
            # Should have clear notification about what mock generation means
            assert len(notification_suggestions) > 0 or any("placeholder" in s.lower() or "temporarily" in s.lower() for s in result.recovery_suggestions), \
                f"Expected clear notification about mock generation, got: {result.recovery_suggestions}"
        
        print("✓ Requirement 7.4: Fallback to alternative models or mock generation - PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 7.4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_with_existing_error_handler():
    """Test integration with existing IntegratedErrorHandler system"""
    print("\n=== Testing Integration with IntegratedErrorHandler ===")
    
    try:
        from backend.core.integrated_error_handler import handle_wan_model_error
        
        # Test the integration function
        wan_context = {
            "model_type": "t2v-A14B",
            "model_loaded": False,
            "error_stage": "model_loading"
        }
        
        error = Exception("WAN model integration test error")
        result = await handle_wan_model_error(error, wan_context)
        
        # Verify result is a UserFriendlyError
        assert hasattr(result, 'category'), "Expected UserFriendlyError with category"
        assert hasattr(result, 'severity'), "Expected UserFriendlyError with severity"
        assert hasattr(result, 'recovery_suggestions'), "Expected UserFriendlyError with recovery_suggestions"
        
        # Verify WAN-specific handling
        assert len(result.recovery_suggestions) > 0, "Expected recovery suggestions from WAN handler"
        
        print("✓ Integration with IntegratedErrorHandler system - PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all requirement tests"""
    print("Testing WAN Model Error Handler Requirements Implementation")
    print("=" * 60)
    
    tests = [
        test_requirement_7_1_specific_error_messages,
        test_requirement_7_2_cuda_memory_optimization,
        test_requirement_7_3_error_categorization,
        test_requirement_7_4_fallback_strategies,
        test_integration_with_existing_error_handler
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("REQUIREMENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL REQUIREMENTS TESTS PASSED!")
        print("\nWAN Model Error Handler successfully implements:")
        print("✓ 7.1: Specific error messages about actual models")
        print("✓ 7.2: CUDA memory errors with model-specific optimization")
        print("✓ 7.3: Error categorization and recovery suggestions")
        print("✓ 7.4: Fallback to alternative models or mock generation")
        print("✓ Integration with existing IntegratedErrorHandler system")
    else:
        print("❌ Some requirement tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)