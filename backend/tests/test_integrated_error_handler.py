"""
Tests for Integrated Error Handler

Tests the enhanced error handling system that bridges FastAPI backend
with existing GenerationErrorHandler infrastructure.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional

# Import the integrated error handler
from backend.core.integrated_error_handler import (
    IntegratedErrorHandler,
    handle_model_loading_error,
    handle_vram_exhaustion_error,
    handle_generation_pipeline_error,
    get_integrated_error_handler,
    EXISTING_ERROR_HANDLER_AVAILABLE
)

# Import error types for testing
try:
    from infrastructure.hardware.error_handler import (
        ErrorCategory,
        ErrorSeverity,
        UserFriendlyError
    )
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    # Use fallback classes for testing
    from backend.core.integrated_error_handler import ErrorCategory, ErrorSeverity, UserFriendlyError


class TestIntegratedErrorHandler:
    """Test the IntegratedErrorHandler class"""
    
    @pytest.fixture
    def handler(self):
        """Create an IntegratedErrorHandler instance for testing"""
        return IntegratedErrorHandler()
    
    @pytest.fixture
    def mock_existing_handler(self):
        """Create a mock existing error handler"""
        mock_handler = Mock()
        mock_handler.handle_error.return_value = UserFriendlyError(
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            title="Model Loading Failed",
            message="Test error message",
            recovery_suggestions=["Test suggestion 1", "Test suggestion 2"],
            technical_details="Test technical details"
        )
        return mock_handler
    
    def test_handler_initialization(self, handler):
        """Test that the handler initializes correctly"""
        assert handler is not None
        assert hasattr(handler, 'existing_handler')
        assert hasattr(handler, 'logger')
        assert hasattr(handler, '_fastapi_error_patterns')
        assert hasattr(handler, '_recovery_strategies')
    
    def test_fastapi_error_patterns_initialization(self, handler):
        """Test that FastAPI-specific error patterns are initialized"""
        patterns = handler._fastapi_error_patterns
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        
        # Check for specific FastAPI patterns
        assert "model integration bridge" in patterns
        assert "validation error" in patterns
        assert "websocket connection" in patterns
    
    @pytest.mark.asyncio
    async def test_handle_error_basic(self, handler):
        """Test basic error handling functionality"""
        test_error = ValueError("Test error message")
        context = {"test_key": "test_value"}
        
        result = await handler.handle_error(test_error, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category in [category for category in ErrorCategory]
        assert result.severity in [severity for severity in ErrorSeverity]
        assert result.title is not None
        assert result.message is not None
        assert isinstance(result.recovery_suggestions, list)
    
    @pytest.mark.asyncio
    async def test_handle_model_loading_error(self, handler):
        """Test model loading error handling"""
        test_error = FileNotFoundError("Model file not found")
        model_type = "t2v-A14B"
        context = {"task_id": "test_task"}
        
        result = await handler.handle_model_loading_error(test_error, model_type, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category == ErrorCategory.MODEL_LOADING
        assert model_type in str(result.technical_details) or "model" in result.message.lower()
        assert len(result.recovery_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_handle_vram_exhaustion_error(self, handler):
        """Test VRAM exhaustion error handling"""
        test_error = RuntimeError("CUDA out of memory")
        generation_params = {
            "resolution": "1080p",
            "steps": 30,
            "model_type": "t2v-A14B"
        }
        context = {"task_id": "test_task"}
        
        result = await handler.handle_vram_exhaustion_error(test_error, generation_params, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category == ErrorCategory.VRAM_MEMORY
        assert "memory" in result.message.lower() or "vram" in result.message.lower()
        assert len(result.recovery_suggestions) > 0
        
        # Check for VRAM-specific suggestions
        suggestions_text = " ".join(result.recovery_suggestions).lower()
        assert any(keyword in suggestions_text for keyword in ["resolution", "memory", "optimization", "reduce"])
    
    @pytest.mark.asyncio
    async def test_handle_generation_pipeline_error(self, handler):
        """Test generation pipeline error handling"""
        test_error = RuntimeError("Generation pipeline failed")
        context = {"task_id": "test_task", "model_type": "i2v-A14B"}
        
        result = await handler.handle_generation_pipeline_error(test_error, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category == ErrorCategory.GENERATION_PIPELINE
        assert len(result.recovery_suggestions) > 0
    
    def test_enhance_context_for_fastapi(self, handler):
        """Test context enhancement for FastAPI integration"""
        original_context = {"test_key": "test_value"}
        
        enhanced_context = handler._enhance_context_for_fastapi(original_context)
        
        assert "fastapi_integration" in enhanced_context
        assert enhanced_context["fastapi_integration"] is True
        assert "error_handler_type" in enhanced_context
        assert enhanced_context["error_handler_type"] == "integrated"
        assert "existing_handler_available" in enhanced_context
        assert "test_key" in enhanced_context
        assert enhanced_context["test_key"] == "test_value"
    
    def test_get_fastapi_recovery_suggestions(self, handler):
        """Test FastAPI-specific recovery suggestions"""
        context = {"model_bridge_available": False, "wan22_optimizer_available": True}
        
        # Test model loading suggestions
        model_suggestions = handler._get_fastapi_recovery_suggestions(ErrorCategory.MODEL_LOADING, context)
        assert len(model_suggestions) > 0
        assert any("model" in suggestion.lower() for suggestion in model_suggestions)
        
        # Test VRAM suggestions
        vram_suggestions = handler._get_fastapi_recovery_suggestions(ErrorCategory.VRAM_MEMORY, context)
        assert len(vram_suggestions) > 0
        assert any("optimization" in suggestion.lower() or "vram" in suggestion.lower() for suggestion in vram_suggestions)
    
    def test_categorize_error_integrated(self, handler):
        """Test error categorization with integrated patterns"""
        # Test VRAM error
        vram_error = "cuda out of memory error"
        assert handler._categorize_error_integrated(vram_error) == ErrorCategory.VRAM_MEMORY
        
        # Test model error
        model_error = "model file not found"
        assert handler._categorize_error_integrated(model_error) == ErrorCategory.MODEL_LOADING
        
        # Test validation error
        validation_error = "validation failed for input"
        assert handler._categorize_error_integrated(validation_error) == ErrorCategory.INPUT_VALIDATION
        
        # Test FastAPI-specific error
        fastapi_error = "model integration bridge failed"
        assert handler._categorize_error_integrated(fastapi_error) == ErrorCategory.MODEL_LOADING
    
    def test_determine_severity_integrated(self, handler):
        """Test error severity determination"""
        # Test critical error
        critical_error = MemoryError("Out of memory")
        assert handler._determine_severity_integrated(critical_error, ErrorCategory.VRAM_MEMORY) == ErrorSeverity.CRITICAL
        
        # Test high severity error
        model_error = FileNotFoundError("Model not found")
        assert handler._determine_severity_integrated(model_error, ErrorCategory.MODEL_LOADING) == ErrorSeverity.HIGH
        
        # Test medium severity error
        pipeline_error = RuntimeError("Pipeline failed")
        assert handler._determine_severity_integrated(pipeline_error, ErrorCategory.GENERATION_PIPELINE) == ErrorSeverity.MEDIUM
        
        # Test low severity error
        validation_error = ValueError("Invalid input")
        assert handler._determine_severity_integrated(validation_error, ErrorCategory.INPUT_VALIDATION) == ErrorSeverity.LOW
    
    @pytest.mark.asyncio
    async def test_model_loading_recovery(self, handler):
        """Test automatic model loading recovery"""
        test_error = FileNotFoundError("Model not found")
        model_type = "t2v-A14B"
        context = {"task_id": "test_task"}
        
        # Mock torch.cuda for testing
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            result = await handler._attempt_model_loading_recovery(test_error, model_type, context)
            
            # Should attempt recovery for "not found" errors
            assert isinstance(result, bool)
            mock_empty_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vram_optimization(self, handler):
        """Test automatic VRAM optimization"""
        generation_params = {
            "resolution": "1080p",
            "steps": 30,
            "model_type": "t2v-A14B"
        }
        context = {"task_id": "test_task"}
        
        # Mock torch.cuda for testing
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            result = await handler._attempt_vram_optimization(generation_params, context)
            
            assert isinstance(result, bool)
            mock_empty_cache.assert_called_once()
            
            # Check if optimizations were applied
            if result:
                assert generation_params.get("resolution") == "720p"  # Should be reduced
                assert generation_params.get("steps") == 20  # Should be reduced
                assert generation_params.get("enable_quantization") is True
    
    def test_get_system_status(self, handler):
        """Test system status reporting"""
        status = handler.get_system_status()
        
        assert isinstance(status, dict)
        assert "existing_handler_available" in status
        assert "integrated_handler_active" in status
        assert status["integrated_handler_active"] is True
        
        # Should include system metrics
        assert "cpu_usage_percent" in status or "system_info_error" in status
    
    def test_get_error_categories(self, handler):
        """Test error categories retrieval"""
        categories = handler.get_error_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(category, str) for category in categories)
        assert "model_loading" in categories
        assert "vram_memory" in categories


class TestConvenienceFunctions:
    """Test convenience functions for common error scenarios"""
    
    @pytest.mark.asyncio
    async def test_handle_model_loading_error_function(self):
        """Test model loading error convenience function"""
        error = FileNotFoundError("Model not found")
        model_type = "t2v-A14B"
        context = {"test": "context"}
        
        result = await handle_model_loading_error(error, model_type, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category == ErrorCategory.MODEL_LOADING
    
    @pytest.mark.asyncio
    async def test_handle_vram_exhaustion_error_function(self):
        """Test VRAM exhaustion error convenience function"""
        error = RuntimeError("CUDA out of memory")
        generation_params = {"resolution": "1080p", "steps": 30}
        context = {"test": "context"}
        
        result = await handle_vram_exhaustion_error(error, generation_params, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category == ErrorCategory.VRAM_MEMORY
    
    @pytest.mark.asyncio
    async def test_handle_generation_pipeline_error_function(self):
        """Test generation pipeline error convenience function"""
        error = RuntimeError("Pipeline failed")
        context = {"test": "context"}
        
        result = await handle_generation_pipeline_error(error, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.category == ErrorCategory.GENERATION_PIPELINE


class TestGlobalErrorHandler:
    """Test global error handler instance management"""
    
    def test_get_integrated_error_handler(self):
        """Test global error handler retrieval"""
        handler1 = get_integrated_error_handler()
        handler2 = get_integrated_error_handler()
        
        # Should return the same instance (singleton pattern)
        assert handler1 is handler2
        assert isinstance(handler1, IntegratedErrorHandler)


class TestErrorHandlerIntegration:
    """Test integration with existing error handling infrastructure"""
    
    @pytest.mark.skipif(not INFRASTRUCTURE_AVAILABLE, reason="Infrastructure error handler not available")
    @pytest.mark.asyncio
    async def test_existing_handler_integration(self):
        """Test integration with existing GenerationErrorHandler"""
        handler = IntegratedErrorHandler()
        
        # Should have existing handler if infrastructure is available
        if EXISTING_ERROR_HANDLER_AVAILABLE:
            assert handler.existing_handler is not None
        
        # Test error handling with existing infrastructure
        test_error = ValueError("Test error")
        context = {"test": "context"}
        
        result = await handler.handle_error(test_error, context)
        
        assert isinstance(result, UserFriendlyError)
        assert result.message is not None
        assert isinstance(result.recovery_suggestions, list)
    
    @pytest.mark.asyncio
    async def test_fallback_when_existing_handler_fails(self):
        """Test fallback to integrated handling when existing handler fails"""
        handler = IntegratedErrorHandler()
        
        # Mock existing handler to fail
        if handler.existing_handler:
            with patch.object(handler.existing_handler, 'handle_error', side_effect=Exception("Handler failed")):
                test_error = ValueError("Test error")
                context = {"test": "context"}
                
                result = await handler.handle_error(test_error, context)
                
                # Should still return a valid error result
                assert isinstance(result, UserFriendlyError)
                assert result.message is not None


class TestErrorRecovery:
    """Test automatic error recovery functionality"""
    
    @pytest.mark.asyncio
    async def test_model_loading_recovery_success(self):
        """Test successful model loading recovery"""
        handler = IntegratedErrorHandler()
        
        # Test with "not found" error (should trigger recovery)
        error = FileNotFoundError("Model file not found")
        model_type = "t2v-A14B"
        context = {"task_id": "test"}
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'):
            
            result = await handler._attempt_model_loading_recovery(error, model_type, context)
            assert result is True  # Should succeed for "not found" errors
    
    @pytest.mark.asyncio
    async def test_vram_optimization_success(self):
        """Test successful VRAM optimization"""
        handler = IntegratedErrorHandler()
        
        generation_params = {
            "resolution": "1080p",
            "steps": 35,
            "model_type": "t2v-A14B"
        }
        context = {"task_id": "test"}
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'):
            
            result = await handler._attempt_vram_optimization(generation_params, context)
            
            # Should apply optimizations
            assert result is True
            assert generation_params["resolution"] == "720p"
            assert generation_params["steps"] == 20
            assert generation_params["enable_quantization"] is True


class TestErrorContextEnhancement:
    """Test error context enhancement for FastAPI integration"""
    
    def test_context_enhancement_with_generation_service(self):
        """Test context enhancement when generation service is provided"""
        handler = IntegratedErrorHandler()
        
        # Mock generation service
        mock_service = Mock()
        mock_service.model_integration_bridge = Mock()
        mock_service.real_generation_pipeline = Mock()
        mock_service.wan22_system_optimizer = None
        
        context = {"generation_service": mock_service}
        
        enhanced_context = handler._enhance_context_for_fastapi(context)
        
        assert enhanced_context["fastapi_integration"] is True
        assert enhanced_context["model_bridge_available"] is True
        assert enhanced_context["real_pipeline_available"] is True
        assert enhanced_context["wan22_optimizer_available"] is False
    
    def test_context_enhancement_without_generation_service(self):
        """Test context enhancement without generation service"""
        handler = IntegratedErrorHandler()
        
        context = {"test_key": "test_value"}
        
        enhanced_context = handler._enhance_context_for_fastapi(context)
        
        assert enhanced_context["fastapi_integration"] is True
        assert enhanced_context["error_handler_type"] == "integrated"
        assert "test_key" in enhanced_context


if __name__ == "__main__":
    pytest.main([__file__])
