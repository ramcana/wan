"""
Test suite for WAN Model Error Handler

This test suite validates the WAN model error handling implementation
against requirements 7.1, 7.2, 7.3, and 7.4.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import the WAN error handler components
from backend.core.models.wan_models.wan_model_error_handler import (
    WANModelErrorHandler,
    WANErrorCategory,
    WANErrorSeverity,
    WANErrorContext,
    WANRecoveryAction,
    get_wan_error_handler,
    handle_wan_model_loading_error,
    handle_wan_inference_error,
    handle_wan_memory_error,
    handle_wan_parameter_validation_error
)

# Import WAN model types
try:
    from backend.core.models.wan_models.wan_base_model import WANModelType, HardwareProfile
    WAN_MODELS_AVAILABLE = True
except ImportError:
    # Create fallback for testing
    from enum import Enum
    class WANModelType(Enum):
        T2V_A14B = "t2v-A14B"
        I2V_A14B = "i2v-A14B"
        TI2V_5B = "ti2v-5B"
    
    class HardwareProfile:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    WAN_MODELS_AVAILABLE = False

# Import error handling infrastructure
try:
    from backend.core.integrated_error_handler import UserFriendlyError, ErrorCategory, ErrorSeverity
    INTEGRATED_ERROR_HANDLER_AVAILABLE = True
except ImportError:
    from dataclasses import dataclass
    from enum import Enum
    
    class ErrorCategory(Enum):
        VRAM_MEMORY = "vram_memory"
        MODEL_LOADING = "model_loading"
        GENERATION_PIPELINE = "generation_pipeline"
        INPUT_VALIDATION = "input_validation"
        SYSTEM_RESOURCE = "system_resource"
        UNKNOWN = "unknown"
    
    class ErrorSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    @dataclass
    class UserFriendlyError:
        category: ErrorCategory
        severity: ErrorSeverity
        title: str
        message: str
        recovery_suggestions: List[str]
        technical_details: str = ""
        error_code: str = ""
        recovery_actions: List = None
    
    INTEGRATED_ERROR_HANDLER_AVAILABLE = False


class TestWANModelErrorHandler:
    """Test the WAN Model Error Handler implementation"""
    
    @pytest.fixture
    def wan_error_handler(self):
        """Create a WAN error handler instance for testing"""
        return WANModelErrorHandler()
    
    @pytest.fixture
    def sample_wan_context(self):
        """Create a sample WAN error context for testing"""
        return WANErrorContext(
            model_type=WANModelType.T2V_A14B,
            model_loaded=True,
            weights_loaded=True,
            hardware_optimized=False,
            generation_params={"resolution": "1080p", "frames": 16, "steps": 25},
            vram_usage_gb=8.5,
            available_vram_gb=16.0,
            applied_optimizations=["fp16"],
            error_stage="inference",
            checkpoint_path="/models/wan-t2v-a14b/pytorch_model.bin"
        )
    
    @pytest.fixture
    def sample_hardware_profile(self):
        """Create a sample hardware profile for testing"""
        return HardwareProfile(
            gpu_name="RTX 4080",
            total_vram_gb=16.0,
            available_vram_gb=14.0,
            cpu_cores=24,
            total_ram_gb=64.0,
            architecture_type="cuda",
            supports_fp16=True,
            supports_bf16=True,
            tensor_cores_available=True
        )
    
    # Requirement 7.1: Model loading error handling with specific troubleshooting guidance
    @pytest.mark.asyncio
    async def test_model_loading_error_handling(self, wan_error_handler, sample_wan_context):
        """Test WAN model loading error handling with specific troubleshooting guidance"""
        
        # Test model not found error
        model_not_found_error = FileNotFoundError("WAN T2V-A14B model checkpoint not found")
        sample_wan_context.error_stage = "model_loading"
        sample_wan_context.model_loaded = False
        
        result = await wan_error_handler.handle_wan_error(model_not_found_error, sample_wan_context)
        
        # Verify error categorization
        assert result.category in [ErrorCategory.MODEL_LOADING]
        assert "WAN" in result.title
        assert "T2V-A14B" in result.title or "model" in result.title.lower()
        
        # Verify specific troubleshooting guidance
        suggestions = [s.lower() for s in result.recovery_suggestions]
        assert any("download" in s for s in suggestions), "Should suggest downloading model"
        assert any("weight" in s or "checkpoint" in s for s in suggestions), "Should mention weights/checkpoint"
        
        # Test architecture mismatch error
        arch_error = ValueError("WAN model architecture configuration mismatch")
        result = await wan_error_handler.handle_wan_error(arch_error, sample_wan_context)
        
        suggestions = [s.lower() for s in result.recovery_suggestions]
        assert any("config" in s or "architecture" in s for s in suggestions), "Should suggest config fixes"
    
    # Requirement 7.2: CUDA memory error handling with model-specific optimization strategies
    @pytest.mark.asyncio
    async def test_cuda_memory_error_handling(self, wan_error_handler, sample_wan_context):
        """Test CUDA memory error handling with model-specific optimization strategies"""
        
        # Test VRAM exhaustion error
        vram_error = RuntimeError("CUDA out of memory. Tried to allocate 2.50 GiB")
        sample_wan_context.error_stage = "inference"
        sample_wan_context.vram_usage_gb = 15.5
        sample_wan_context.available_vram_gb = 16.0
        
        result = await wan_error_handler.handle_wan_error(vram_error, sample_wan_context)
        
        # Verify error categorization
        assert result.category in [ErrorCategory.VRAM_MEMORY]
        
        # Verify model-specific optimization strategies
        suggestions = [s.lower() for s in result.recovery_suggestions]
        assert any("cpu offload" in s or "offload" in s for s in suggestions), "Should suggest CPU offloading"
        assert any("quantization" in s for s in suggestions), "Should suggest quantization"
        assert any("resolution" in s or "frames" in s for s in suggestions), "Should suggest parameter reduction"
        
        # Test WAN-specific optimizations
        wan_specific_suggestions = [s for s in result.recovery_suggestions if "wan" in s.lower()]
        assert len(wan_specific_suggestions) > 0, "Should include WAN-specific suggestions"
    
    # Requirement 7.3: Model inference error categorization and recovery suggestions
    @pytest.mark.asyncio
    async def test_inference_error_categorization(self, wan_error_handler, sample_wan_context):
        """Test model inference error categorization and recovery suggestions"""
        
        # Test temporal processing error
        temporal_error = RuntimeError("WAN temporal attention failed during inference")
        sample_wan_context.error_stage = "inference"
        
        result = await wan_error_handler.handle_wan_error(temporal_error, sample_wan_context)
        
        # Verify proper categorization
        assert result.category in [ErrorCategory.GENERATION_PIPELINE]
        assert "temporal" in result.message.lower() or "attention" in result.message.lower()
        
        # Verify recovery suggestions
        suggestions = [s.lower() for s in result.recovery_suggestions]
        assert any("reduce" in s and ("frames" in s or "complexity" in s) for s in suggestions)
        
        # Test conditioning error
        conditioning_error = ValueError("WAN text conditioning failed")
        result = await wan_error_handler.handle_wan_error(conditioning_error, sample_wan_context)
        
        suggestions = [s.lower() for s in result.recovery_suggestions]
        assert any("conditioning" in s or "prompt" in s for s in suggestions)
    
    # Requirement 7.4: Critical error fallback with clear notifications
    @pytest.mark.asyncio
    async def test_critical_error_fallback(self, wan_error_handler, sample_wan_context):
        """Test critical error fallback with clear notifications"""
        
        # Test system critical error
        critical_error = SystemError("WAN model system failure")
        sample_wan_context.error_stage = "inference"
        
        result = await wan_error_handler.handle_wan_error(critical_error, sample_wan_context)
        
        # Verify high severity
        assert result.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        
        # Verify fallback suggestions
        suggestions = [s.lower() for s in result.recovery_suggestions]
        assert any("fallback" in s or "mock" in s or "alternative" in s for s in suggestions)
        
        # Verify clear notifications
        assert "critical" in result.message.lower() or "system" in result.message.lower()
    
    def test_wan_error_categorization(self, wan_error_handler):
        """Test WAN-specific error categorization"""
        
        # Test various WAN error patterns
        test_cases = [
            ("WAN model checkpoint missing", WANErrorCategory.WAN_MODEL_LOADING),
            ("WAN architecture mismatch", WANErrorCategory.WAN_ARCHITECTURE),
            ("WAN weights download failed", WANErrorCategory.WAN_WEIGHTS_DOWNLOAD),
            ("WAN generation failed", WANErrorCategory.WAN_INFERENCE),
            ("WAN VRAM insufficient", WANErrorCategory.WAN_MEMORY_MANAGEMENT),
            ("WAN temporal attention failed", WANErrorCategory.WAN_TEMPORAL_PROCESSING),
            ("WAN text conditioning failed", WANErrorCategory.WAN_CONDITIONING),
            ("WAN frames exceed limit", WANErrorCategory.WAN_PARAMETER_VALIDATION)
        ]
        
        for error_message, expected_category in test_cases:
            error = RuntimeError(error_message)
            context = WANErrorContext(error_stage="inference")
            
            category = wan_error_handler._categorize_wan_error(error, context)
            assert category == expected_category, f"Failed to categorize '{error_message}' as {expected_category}"
    
    def test_model_specific_error_handling(self, wan_error_handler):
        """Test model-specific error handling configurations"""
        
        # Test T2V-A14B specific handling
        t2v_info = wan_error_handler.get_model_specific_info(WANModelType.T2V_A14B)
        assert "memory_requirements" in t2v_info
        assert "recovery_priorities" in t2v_info
        assert t2v_info["memory_requirements"]["minimum_vram_gb"] > 0
        
        # Test I2V-A14B specific handling
        i2v_info = wan_error_handler.get_model_specific_info(WANModelType.I2V_A14B)
        assert i2v_info["memory_requirements"]["minimum_vram_gb"] > 0
        
        # Test TI2V-5B specific handling (smaller model)
        ti2v_info = wan_error_handler.get_model_specific_info(WANModelType.TI2V_5B)
        assert ti2v_info["memory_requirements"]["minimum_vram_gb"] < t2v_info["memory_requirements"]["minimum_vram_gb"]
    
    def test_recovery_action_generation(self, wan_error_handler):
        """Test recovery action generation for different error categories"""
        
        # Test model loading recovery actions
        loading_actions = wan_error_handler.get_wan_recovery_actions(WANErrorCategory.WAN_MODEL_LOADING)
        assert len(loading_actions) > 0
        
        action_types = [action["action_type"] for action in loading_actions]
        assert "download_wan_model_weights" in action_types
        assert "clear_wan_model_cache" in action_types
        
        # Test memory management recovery actions
        memory_actions = wan_error_handler.get_wan_recovery_actions(WANErrorCategory.WAN_MEMORY_MANAGEMENT)
        assert len(memory_actions) > 0
        
        action_types = [action["action_type"] for action in memory_actions]
        assert "enable_wan_cpu_offload" in action_types
        assert "apply_wan_quantization" in action_types
    
    @pytest.mark.asyncio
    async def test_automatic_recovery_attempts(self, wan_error_handler, sample_wan_context):
        """Test automatic recovery attempts for recoverable errors"""
        
        # Mock the automatic recovery methods
        with patch.object(wan_error_handler, '_attempt_wan_automatic_recovery', return_value=True) as mock_recovery:
            
            # Test recoverable error
            recoverable_error = ValueError("WAN parameter validation failed")
            sample_wan_context.error_stage = "parameter_validation"
            
            result = await wan_error_handler.handle_wan_error(recoverable_error, sample_wan_context)
            
            # Verify automatic recovery was attempted
            mock_recovery.assert_called_once()
            
            # Verify recovery notification in suggestions
            assert any("automatic" in s.lower() and "recovery" in s.lower() for s in result.recovery_suggestions)
    
    def test_integration_with_existing_error_handler(self, wan_error_handler):
        """Test integration with existing IntegratedErrorHandler system"""
        
        # Verify integrated handler is available if possible
        if INTEGRATED_ERROR_HANDLER_AVAILABLE:
            assert wan_error_handler.integrated_handler is not None
        else:
            assert wan_error_handler.integrated_handler is None
        
        # Test error category mapping
        wan_category = WANErrorCategory.WAN_MODEL_LOADING
        standard_category = wan_error_handler._map_wan_to_standard_category(wan_category)
        assert standard_category == ErrorCategory.MODEL_LOADING
        
        # Test severity mapping
        wan_severity = WANErrorSeverity.SYSTEM_CRITICAL
        standard_severity = wan_error_handler._map_wan_to_standard_severity(wan_severity)
        assert standard_severity == ErrorSeverity.CRITICAL


class TestWANErrorHandlerConvenienceFunctions:
    """Test the convenience functions for WAN error handling"""
    
    @pytest.mark.asyncio
    async def test_handle_wan_model_loading_error(self):
        """Test the convenience function for model loading errors"""
        
        error = FileNotFoundError("WAN T2V model not found")
        model_type = WANModelType.T2V_A14B
        checkpoint_path = "/models/wan-t2v/pytorch_model.bin"
        
        result = await handle_wan_model_loading_error(error, model_type, checkpoint_path)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category in [ErrorCategory.MODEL_LOADING]
        assert len(result.recovery_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_handle_wan_inference_error(self):
        """Test the convenience function for inference errors"""
        
        error = RuntimeError("WAN inference failed")
        model_type = WANModelType.I2V_A14B
        generation_params = {"resolution": "1080p", "frames": 16}
        
        result = await handle_wan_inference_error(error, model_type, generation_params)
        
        assert isinstance(result, UserFriendlyError)
        assert len(result.recovery_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_handle_wan_memory_error(self):
        """Test the convenience function for memory errors"""
        
        error = RuntimeError("CUDA out of memory")
        model_type = WANModelType.T2V_A14B
        vram_usage_gb = 15.5
        available_vram_gb = 16.0
        
        result = await handle_wan_memory_error(error, model_type, vram_usage_gb, available_vram_gb)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category in [ErrorCategory.VRAM_MEMORY]
        assert len(result.recovery_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_handle_wan_parameter_validation_error(self):
        """Test the convenience function for parameter validation errors"""
        
        error = ValueError("Invalid WAN generation parameters")
        model_type = WANModelType.TI2V_5B
        generation_params = {"resolution": "4K", "frames": 100}  # Invalid params
        validation_errors = ["Resolution too high", "Too many frames"]
        
        result = await handle_wan_parameter_validation_error(
            error, model_type, generation_params, validation_errors
        )
        
        assert isinstance(result, UserFriendlyError)
        assert len(result.recovery_suggestions) > 0


class TestWANErrorHandlerIntegration:
    """Test WAN error handler integration with the broader system"""
    
    def test_global_error_handler_singleton(self):
        """Test that the global error handler works as a singleton"""
        
        handler1 = get_wan_error_handler()
        handler2 = get_wan_error_handler()
        
        assert handler1 is handler2
        assert isinstance(handler1, WANModelErrorHandler)
    
    @pytest.mark.asyncio
    async def test_error_handler_with_hardware_profile(self, sample_hardware_profile):
        """Test error handler with hardware profile context"""
        
        wan_context = WANErrorContext(
            model_type=WANModelType.T2V_A14B,
            hardware_profile=sample_hardware_profile,
            error_stage="optimization"
        )
        
        error = RuntimeError("WAN hardware optimization failed")
        handler = get_wan_error_handler()
        
        result = await handler.handle_wan_error(error, wan_context)
        
        assert isinstance(result, UserFriendlyError)
        # Should include RTX 4080 specific suggestions
        suggestions_text = " ".join(result.recovery_suggestions).lower()
        assert "rtx" in suggestions_text or "optimization" in suggestions_text
    
    def test_error_categories_completeness(self):
        """Test that all WAN error categories are properly defined"""
        
        handler = get_wan_error_handler()
        categories = handler.get_wan_error_categories()
        
        expected_categories = [
            "wan_model_loading",
            "wan_architecture", 
            "wan_weights_download",
            "wan_weights_integrity",
            "wan_inference",
            "wan_memory_management",
            "wan_hardware_optimization",
            "wan_parameter_validation",
            "wan_temporal_processing",
            "wan_conditioning"
        ]
        
        for expected in expected_categories:
            assert expected in categories, f"Missing error category: {expected}"
    
    @pytest.mark.asyncio
    async def test_fallback_error_handling(self):
        """Test fallback error handling when WAN handler fails"""
        
        handler = WANModelErrorHandler()
        
        # Mock a failure in WAN error handling
        with patch.object(handler, '_categorize_wan_error', side_effect=Exception("Handler failed")):
            
            error = RuntimeError("Test error")
            context = WANErrorContext(error_stage="test")
            
            result = await handler.handle_wan_error(error, context)
            
            # Should still return a valid error response
            assert isinstance(result, UserFriendlyError)
            assert len(result.recovery_suggestions) > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
