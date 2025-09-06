"""
Integration tests for error handler with existing codebase
"""

import pytest
from unittest.mock import Mock, patch
from error_handler import GenerationErrorHandler, handle_validation_error, handle_vram_error


class TestErrorHandlerIntegrationWithExistingCode:
    """Test error handler integration with existing video generation code"""
    
    def test_integration_with_validation_framework(self):
        """Test integration with existing validation framework"""
        # This would test integration with input_validation.py if it exists
        handler = GenerationErrorHandler()
        
        # Simulate validation error from existing validation code
        validation_error = ValueError("Invalid input provided: prompt exceeds maximum length")
        context = {
            "prompt": "x" * 600,
            "resolution": "1080p",
            "model_type": "wan22"
        }
        
        user_error = handler.handle_error(validation_error, context)
        
        assert user_error.category.value == "input_validation"
        assert "Input Validation Error" in user_error.title
        assert len(user_error.recovery_suggestions) > 0
        
        # Test automatic recovery
        success, message = handler.attempt_automatic_recovery(user_error, context)
        if success:
            # Verify prompt was truncated
            assert len(context.get("prompt", "")) <= 512
    
    def test_integration_with_model_loading(self):
        """Test integration with model loading errors"""
        handler = GenerationErrorHandler()
        
        # Simulate model loading error
        model_error = FileNotFoundError("Model file not found: /models/wan22/model.safetensors")
        context = {
            "model_path": "/models/wan22/model.safetensors",
            "model_type": "wan22"
        }
        
        user_error = handler.handle_error(model_error, context)
        
        assert user_error.category.value == "model_loading"
        assert "Model Loading Failed" in user_error.title
        assert any("download" in suggestion.lower() for suggestion in user_error.recovery_suggestions)
    
    def test_integration_with_generation_pipeline(self):
        """Test integration with generation pipeline errors"""
        handler = GenerationErrorHandler()
        
        # Simulate generation pipeline error
        pipeline_error = RuntimeError("Generation failed: tensor dimension mismatch")
        context = {
            "prompt": "A beautiful landscape",
            "resolution": "720p",
            "steps": 25,
            "model": "wan22"
        }
        
        user_error = handler.handle_error(pipeline_error, context)
        
        assert user_error.category.value == "generation_pipeline"
        assert "Generation Failed" in user_error.title
        assert user_error.technical_details is not None
    
    def test_convenience_functions_usage(self):
        """Test using convenience functions for common scenarios"""
        
        # Test validation error convenience function
        validation_error = ValueError("invalid input provided")
        context = {"prompt": "test", "resolution": "1080p"}
        
        user_error = handle_validation_error(validation_error, context)
        assert user_error.category.value == "input_validation"
        
        # Test VRAM error convenience function
        vram_error = RuntimeError("cuda out of memory")
        generation_params = {"resolution": "1080p", "steps": 50}
        
        user_error = handle_vram_error(vram_error, generation_params)
        assert user_error.category.value == "vram_memory"
    
    def test_error_logging_and_debugging(self):
        """Test error logging for debugging purposes"""
        handler = GenerationErrorHandler()
        
        with patch.object(handler.logger, 'error') as mock_logger:
            error = ValueError("test error")
            context = {"test_param": "test_value"}
            
            user_error = handler.handle_error(error, context)
            
            # Verify error was logged
            mock_logger.assert_called_once()
            log_call_args = mock_logger.call_args[0][0]
            assert "Generation Error:" in log_call_args
    
    def test_html_output_for_ui_integration(self):
        """Test HTML output for UI integration"""
        handler = GenerationErrorHandler()
        
        error = RuntimeError("CUDA out of memory")
        context = {"resolution": "1080p"}
        
        user_error = handler.handle_error(error, context)
        html_output = user_error.to_html()
        
        # Verify HTML is properly formatted for UI display
        assert '<div class="error-container"' in html_output
        assert user_error.title in html_output
        assert user_error.message in html_output
        assert "Suggested Solutions:" in html_output
        
        # Verify color coding
        assert "#fd7e14" in html_output  # HIGH severity color for VRAM errors
    
    def test_recovery_action_execution(self):
        """Test that recovery actions can be executed"""
        handler = GenerationErrorHandler()
        
        # Test VRAM optimization recovery
        vram_error = RuntimeError("cuda out of memory")
        context = {"resolution": "1080p", "steps": 50}
        
        user_error = handler.handle_error(vram_error, context)
        
        # Find automatic recovery actions
        automatic_actions = [
            action for action in user_error.recovery_actions 
            if action.automatic and action.success_probability > 0.5
        ]
        
        assert len(automatic_actions) > 0
        
        # Test recovery attempt
        success, message = handler.attempt_automatic_recovery(user_error, context)
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    def test_error_categorization_accuracy(self):
        """Test that errors are categorized correctly"""
        handler = GenerationErrorHandler()
        
        test_cases = [
            ("Invalid input provided", "input_validation"),
            ("Model not found", "model_loading"),
            ("CUDA out of memory", "vram_memory"),
            ("Generation failed", "generation_pipeline"),
            ("Permission denied", "file_system"),
            ("Network connection failed", "network"),
            ("Some unknown error", "unknown")
        ]
        
        for error_message, expected_category in test_cases:
            error = Exception(error_message)
            user_error = handler.handle_error(error)
            assert user_error.category.value == expected_category
    
    def test_context_specific_recovery_suggestions(self):
        """Test that recovery suggestions are context-aware"""
        handler = GenerationErrorHandler()
        
        # Test high resolution VRAM error
        vram_error = RuntimeError("cuda out of memory")
        high_res_context = {"resolution": "1080p", "steps": 50}
        
        user_error = handler.handle_error(vram_error, high_res_context)
        suggestions = user_error.recovery_suggestions
        
        # Should suggest reducing resolution
        assert any("720p" in suggestion for suggestion in suggestions)
        assert any("steps" in suggestion for suggestion in suggestions)
        
        # Test long prompt validation error
        validation_error = ValueError("invalid input provided")
        long_prompt_context = {"prompt": "x" * 600, "prompt_length": 600}
        
        user_error = handler.handle_error(validation_error, long_prompt_context)
        suggestions = user_error.recovery_suggestions
        
        # Should suggest shortening prompt
        assert any("300 characters" in suggestion for suggestion in suggestions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])