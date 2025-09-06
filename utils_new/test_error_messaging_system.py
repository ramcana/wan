"""
Tests for Comprehensive Error Messaging System

Tests cover:
- User-friendly error message generation
- Progressive error disclosure (basic → detailed → diagnostic)
- Specific guidance for common compatibility issues
- Error recovery suggestions with actionable steps
- Requirements: 1.4, 2.4, 3.4, 4.4, 6.4, 7.4
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from error_messaging_system import (
    ErrorMessageGenerator,
    ErrorMessageFormatter,
    ProgressiveErrorDisclosure,
    ErrorGuidanceSystem,
    ErrorAnalytics,
    EnhancedErrorHandler,
    ErrorContext,
    ErrorMessage,
    RecoveryAction,
    ErrorSeverity,
    ErrorCategory,
    create_architecture_error,
    create_pipeline_error,
    create_vae_error,
    create_resource_error,
    create_dependency_error,
    create_video_error
)


class TestErrorMessageGenerator:
    """Test error message generation functionality"""
    
    def setup_method(self):
        self.generator = ErrorMessageGenerator()
    
    def test_generate_missing_pipeline_error(self):
        """Test generation of missing pipeline error message"""
        context = ErrorContext(
            model_path="/path/to/wan/model",
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            pipeline_class="WanPipeline",
            attempted_operation="pipeline_loading"
        )
        
        error_msg = self.generator.generate_error_message(
            "missing_pipeline_class", context
        )
        
        assert error_msg.title == "Custom Pipeline Not Available"
        assert "WanPipeline class is not installed" in error_msg.summary
        assert error_msg.category == ErrorCategory.PIPELINE_LOADING
        assert error_msg.severity == ErrorSeverity.ERROR
        assert error_msg.detailed_description is not None
        assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in error_msg.detailed_description
        assert len(error_msg.recovery_actions) > 0
        
        # Check recovery actions
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Install WanPipeline Package" in action_titles
        assert "Enable Remote Code Download" in action_titles
    
    def test_generate_vae_shape_mismatch_error(self):
        """Test generation of VAE shape mismatch error message"""
        context = ErrorContext(
            model_path="/path/to/wan/model",
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            attempted_operation="vae_loading"
        )
        
        error_msg = self.generator.generate_error_message(
            "vae_shape_mismatch", context
        )
        
        assert error_msg.title == "VAE Architecture Incompatible"
        assert error_msg.category == ErrorCategory.VAE_COMPATIBILITY
        assert error_msg.severity == ErrorSeverity.ERROR
        assert "3D architecture" in error_msg.detailed_description
        assert "shape [384, ...]" in error_msg.detailed_description
        
        # Check recovery actions
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Update Model Components" in action_titles
    
    def test_generate_insufficient_vram_error(self):
        """Test generation of insufficient VRAM error message"""
        context = ErrorContext(
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            system_info={"available_vram": "8GB", "required_vram": "12GB"},
            attempted_operation="model_loading"
        )
        
        error_msg = self.generator.generate_error_message(
            "insufficient_vram", context
        )
        
        assert error_msg.title == "Insufficient GPU Memory"
        assert error_msg.category == ErrorCategory.RESOURCE_CONSTRAINTS
        assert error_msg.severity == ErrorSeverity.ERROR
        assert "12-16 GB" in error_msg.detailed_description
        
        # Check recovery actions include optimization strategies
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Enable CPU Offloading" in action_titles
        assert "Use Mixed Precision" in action_titles
        assert "Enable Chunked Processing" in action_titles
    
    def test_generate_remote_code_blocked_error(self):
        """Test generation of remote code blocked error message"""
        context = ErrorContext(
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            pipeline_class="WanPipeline",
            attempted_operation="remote_code_fetch"
        )
        
        error_msg = self.generator.generate_error_message(
            "remote_code_blocked", context
        )
        
        assert error_msg.title == "Remote Code Execution Blocked"
        assert error_msg.category == ErrorCategory.SECURITY_VALIDATION
        assert error_msg.severity == ErrorSeverity.WARNING
        assert "trust_remote_code=False" in error_msg.detailed_description
        
        # Check recovery actions
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Enable Trust Remote Code" in action_titles
        assert "Install Pipeline Locally" in action_titles
    
    def test_generate_encoding_failed_error(self):
        """Test generation of video encoding failed error message"""
        context = ErrorContext(
            user_inputs={"output_path": "/path/to/output.mp4"},
            attempted_operation="video_encoding"
        )
        
        error_msg = self.generator.generate_error_message(
            "encoding_failed", context
        )
        
        assert error_msg.title == "Video Encoding Failed"
        assert error_msg.category == ErrorCategory.VIDEO_PROCESSING
        assert error_msg.severity == ErrorSeverity.WARNING
        
        # Check recovery actions
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Install FFmpeg" in action_titles
        assert "Use Frame Sequence Output" in action_titles
    
    def test_generate_generic_error(self):
        """Test generation of generic error for unknown types"""
        context = ErrorContext(
            model_path="/path/to/model",
            attempted_operation="unknown_operation"
        )
        
        error_msg = self.generator.generate_error_message(
            "unknown_error_type", context
        )
        
        assert error_msg.title == "Unexpected Error"
        assert "unknown_error_type" in error_msg.summary
        assert error_msg.severity == ErrorSeverity.ERROR
        assert len(error_msg.recovery_actions) > 0
    
    def test_add_technical_details_with_exception(self):
        """Test adding technical details when exception is provided"""
        context = ErrorContext(
            model_path="/path/to/model",
            pipeline_class="WanPipeline"
        )
        
        exception = ValueError("Test exception message")
        
        error_msg = self.generator.generate_error_message(
            "missing_pipeline_class", context, exception
        )
        
        assert error_msg.technical_details is not None
        assert "ValueError" in error_msg.technical_details
        assert "Test exception message" in error_msg.technical_details
        assert "/path/to/model" in error_msg.technical_details
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_collect_system_context(self, mock_get_props, mock_cuda_available):
        """Test system context collection"""
        mock_cuda_available.return_value = True
        mock_props = Mock()
        mock_props.total_memory = 12 * 1024**3  # 12GB
        mock_get_props.return_value = mock_props
        
        context = self.generator._collect_system_context()
        
        assert "torch_cuda_available" in context
        assert context["torch_cuda_available"] is True
        assert "total_vram" in context


class TestErrorMessageFormatter:
    """Test error message formatting functionality"""
    
    def setup_method(self):
        self.formatter = ErrorMessageFormatter()
        self.sample_error = ErrorMessage(
            title="Test Error",
            summary="This is a test error",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.PIPELINE_LOADING,
            detailed_description="Detailed description of the error",
            recovery_actions=[
                RecoveryAction(
                    title="Fix Action",
                    description="This will fix the error",
                    command="pip install fix",
                    priority=1,
                    estimated_time="5 minutes"
                )
            ],
            error_code="test_error",
            timestamp="2024-01-01T00:00:00"
        )
    
    def test_format_for_console_basic(self):
        """Test basic console formatting"""
        output = self.formatter.format_for_console(self.sample_error, "basic")
        
        assert "❌ Test Error" in output
        assert "This is a test error" in output
        assert "Recommended Actions:" in output
        assert "Fix Action" in output
        assert "pip install fix" in output
        assert "Estimated time: 5 minutes" in output
        
        # Should not include detailed description in basic mode
        assert "Detailed description" not in output
    
    def test_format_for_console_detailed(self):
        """Test detailed console formatting"""
        output = self.formatter.format_for_console(self.sample_error, "detailed")
        
        assert "❌ Test Error" in output
        assert "This is a test error" in output
        assert "Details:" in output
        assert "Detailed description of the error" in output
        assert "Recommended Actions:" in output
    
    def test_format_for_console_diagnostic(self):
        """Test diagnostic console formatting"""
        self.sample_error.technical_details = "Technical details here"
        self.sample_error.debug_info = {"debug": "info"}
        
        output = self.formatter.format_for_console(self.sample_error, "diagnostic")
        
        assert "Technical Details:" in output
        assert "Technical details here" in output
        assert "Debug Information:" in output
        assert '"debug": "info"' in output
    
    def test_format_for_ui(self):
        """Test UI formatting"""
        ui_format = self.formatter.format_for_ui(self.sample_error)
        
        assert ui_format["title"] == "Test Error"
        assert ui_format["summary"] == "This is a test error"
        assert ui_format["severity"] == "error"
        assert ui_format["category"] == "pipeline_loading"
        assert len(ui_format["recovery_actions"]) == 1
        
        action = ui_format["recovery_actions"][0]
        assert action["title"] == "Fix Action"
        assert action["command"] == "pip install fix"
        assert action["priority"] == 1
    
    def test_format_for_json_basic(self):
        """Test basic JSON formatting"""
        json_output = self.formatter.format_for_json(self.sample_error, include_diagnostics=False)
        data = json.loads(json_output)
        
        assert data["title"] == "Test Error"
        assert data["error_code"] == "test_error"
        assert "stack_trace" not in data
        assert "debug_info" not in data
    
    def test_format_for_json_diagnostic(self):
        """Test diagnostic JSON formatting"""
        self.sample_error.debug_info = {"debug": "info"}
        
        json_output = self.formatter.format_for_json(self.sample_error, include_diagnostics=True)
        data = json.loads(json_output)
        
        assert data["title"] == "Test Error"
        assert "debug_info" in data
        assert data["debug_info"]["debug"] == "info"


class TestProgressiveErrorDisclosure:
    """Test progressive error disclosure functionality"""
    
    def setup_method(self):
        self.disclosure = ProgressiveErrorDisclosure()
    
    def test_handle_error_console_basic(self):
        """Test handling error with basic console output"""
        context = ErrorContext(
            model_path="/path/to/model",
            attempted_operation="test"
        )
        
        output = self.disclosure.handle_error(
            "missing_pipeline_class", context, 
            output_format="console", detail_level="basic"
        )
        
        assert isinstance(output, str)
        assert "Custom Pipeline Not Available" in output
        assert "Recommended Actions:" in output
    
    def test_handle_error_ui_format(self):
        """Test handling error with UI format"""
        context = ErrorContext(
            model_path="/path/to/model",
            attempted_operation="test"
        )
        
        output = self.disclosure.handle_error(
            "missing_pipeline_class", context,
            output_format="ui"
        )
        
        assert isinstance(output, dict)
        assert "title" in output
        assert "recovery_actions" in output
    
    def test_handle_error_json_format(self):
        """Test handling error with JSON format"""
        context = ErrorContext(
            model_path="/path/to/model",
            attempted_operation="test"
        )
        
        output = self.disclosure.handle_error(
            "missing_pipeline_class", context,
            output_format="json"
        )
        
        assert isinstance(output, str)
        data = json.loads(output)
        assert "title" in data
        assert "recovery_actions" in data
    
    def test_invalid_output_format(self):
        """Test handling invalid output format"""
        context = ErrorContext()
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            self.disclosure.handle_error(
                "test_error", context,
                output_format="invalid"
            )


        assert True  # TODO: Add proper assertion

class TestErrorGuidanceSystem:
    """Test error guidance system functionality"""
    
    def setup_method(self):
        self.guidance = ErrorGuidanceSystem()
    
    def test_get_guided_resolution(self):
        """Test getting guided resolution steps"""
        context = ErrorContext(
            model_path="/path/to/model",
            pipeline_class="WanPipeline"
        )
        
        resolution = self.guidance.get_guided_resolution("missing_pipeline_class", context)
        
        assert "error_summary" in resolution
        assert "guided_steps" in resolution
        assert "alternative_solutions" in resolution
        assert "prevention_tips" in resolution
        
        # Check guided steps structure
        assert len(resolution["guided_steps"]) > 0
        step = resolution["guided_steps"][0]
        assert "title" in step
        assert "description" in step
        assert "type" in step
        assert "validation" in step
    
    def test_determine_step_type(self):
        """Test step type determination"""
        command_action = RecoveryAction(
            title="Install Package",
            description="Install the package",
            command="pip install package"
        )
        assert self.guidance._determine_step_type(command_action) == "command"
        
        url_action = RecoveryAction(
            title="Download File",
            description="Download the file",
            url="https://example.com/file"
        )
        assert self.guidance._determine_step_type(url_action) == "download"
        
        install_action = RecoveryAction(
            title="Install Dependencies",
            description="Install required dependencies"
        )
        assert self.guidance._determine_step_type(install_action) == "installation"
    
    def test_get_alternative_solutions(self):
        """Test getting alternative solutions"""
        alternatives = self.guidance._get_alternative_solutions("missing_pipeline_class")
        
        assert len(alternatives) > 0
        assert any("Docker" in alt["title"] for alt in alternatives)
        assert any("Cloud" in alt["title"] for alt in alternatives)
    
    def test_get_prevention_tips(self):
        """Test getting prevention tips"""
        tips = self.guidance._get_prevention_tips("insufficient_vram")
        
        assert len(tips) > 0
        assert any("VRAM" in tip for tip in tips)
        assert any("GPU" in tip for tip in tips)


class TestErrorAnalytics:
    """Test error analytics functionality"""
    
    def setup_method(self):
        # Use temporary file for analytics
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.analytics = ErrorAnalytics(self.temp_file.name)
    
    def teardown_method(self):
        # Clean up temporary file
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_record_error(self):
        """Test recording error occurrences"""
        context = ErrorContext(
            model_path="/path/to/model.bin",
            attempted_operation="test"
        )
        
        # Record multiple errors
        self.analytics.record_error("missing_pipeline_class", context)
        self.analytics.record_error("missing_pipeline_class", context, resolved=True)
        self.analytics.record_error("vae_shape_mismatch", context)
        
        # Check error counts
        assert self.analytics.analytics_data["error_counts"]["missing_pipeline_class"] == 2
        assert self.analytics.analytics_data["error_counts"]["vae_shape_mismatch"] == 1
        
        # Check resolution tracking
        resolution_data = self.analytics.analytics_data["resolution_success"]["missing_pipeline_class"]
        assert resolution_data["attempts"] == 2
        assert resolution_data["successes"] == 1
    
    def test_get_error_statistics(self):
        """Test getting error statistics"""
        context = ErrorContext(model_path="/path/to/model.bin")
        
        # Record some errors
        for _ in range(3):
            self.analytics.record_error("missing_pipeline_class", context)
        for _ in range(2):
            self.analytics.record_error("vae_shape_mismatch", context, resolved=True)
        
        stats = self.analytics.get_error_statistics()
        
        assert stats["total_errors"] == 5
        assert len(stats["most_common_errors"]) > 0
        assert stats["most_common_errors"][0][0] == "missing_pipeline_class"
        assert stats["most_common_errors"][0][1] == 3
        
        # Check resolution rates
        assert "missing_pipeline_class" in stats["resolution_rates"]
        assert stats["resolution_rates"]["vae_shape_mismatch"] == 1.0  # All resolved
    
    def test_get_problematic_models(self):
        """Test identifying problematic models"""
        context1 = ErrorContext(model_path="/path/to/model1.bin")
        context2 = ErrorContext(model_path="/path/to/model2.bin")
        
        # Record more errors for model1
        for _ in range(5):
            self.analytics.record_error("missing_pipeline_class", context1)
        for _ in range(2):
            self.analytics.record_error("vae_shape_mismatch", context2)
        
        stats = self.analytics.get_error_statistics()
        problematic = stats["problematic_models"]
        
        assert len(problematic) > 0
        assert problematic[0]["model"] == "model1.bin"
        assert problematic[0]["total_errors"] == 5


class TestEnhancedErrorHandler:
    """Test enhanced error handler functionality"""
    
    def setup_method(self):
        # Use temporary file for analytics
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        with patch('error_messaging_system.ErrorAnalytics') as mock_analytics:
            mock_analytics.return_value = ErrorAnalytics(self.temp_file.name)
            self.handler = EnhancedErrorHandler()
    
    def teardown_method(self):
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_handle_compatibility_error_interactive(self):
        """Test handling compatibility error in interactive mode"""
        context = ErrorContext(
            model_path="/path/to/model",
            pipeline_class="WanPipeline"
        )
        
        response = self.handler.handle_compatibility_error(
            "missing_pipeline_class", context, interactive=True
        )
        
        assert "error_message" in response
        assert "guided_resolution" in response
        assert "error_id" in response
        assert "support_info" in response
        
        # Check error message structure
        error_msg = response["error_message"]
        assert "title" in error_msg
        assert "recovery_actions" in error_msg
        
        # Check guided resolution
        guided = response["guided_resolution"]
        assert "guided_steps" in guided
        assert "alternative_solutions" in guided
        assert "prevention_tips" in guided
    
    def test_handle_compatibility_error_non_interactive(self):
        """Test handling compatibility error in non-interactive mode"""
        context = ErrorContext(
            model_path="/path/to/model",
            pipeline_class="WanPipeline"
        )
        
        response = self.handler.handle_compatibility_error(
            "missing_pipeline_class", context, interactive=False
        )
        
        assert "error_message" in response
        assert response["guided_resolution"] is None
    
    def test_mark_error_resolved(self):
        """Test marking error as resolved"""
        context = ErrorContext(model_path="/path/to/model")
        
        # This should not raise an exception
        self.handler.mark_error_resolved("missing_pipeline_class", context)

        assert True  # TODO: Add proper assertion
    
    def test_get_system_health(self):
        """Test getting system health information"""
        health = self.handler.get_system_health()
        
        assert "health_score" in health
        assert "error_statistics" in health
        assert "recommendations" in health
        assert 0 <= health["health_score"] <= 100


class TestConvenienceFunctions:
    """Test convenience functions for common error scenarios"""
    
    def test_create_architecture_error(self):
        """Test creating architecture detection error"""
        output = create_architecture_error(
            "/path/to/model", "corrupted_model_index"
        )
        
        assert isinstance(output, str)
        assert "Model Configuration Corrupted" in output
    
    def test_create_pipeline_error(self):
        """Test creating pipeline loading error"""
        output = create_pipeline_error(
            "/path/to/model", "WanPipeline", "missing_pipeline_class"
        )
        
        assert isinstance(output, str)
        assert "Custom Pipeline Not Available" in output
    
    def test_create_vae_error(self):
        """Test creating VAE compatibility error"""
        output = create_vae_error(
            "/path/to/model", "vae_shape_mismatch"
        )
        
        assert isinstance(output, str)
        assert "VAE Architecture Incompatible" in output
    
    def test_create_resource_error(self):
        """Test creating resource constraint error"""
        system_info = {"available_vram": "8GB", "required_vram": "12GB"}
        output = create_resource_error(
            "insufficient_vram", system_info
        )
        
        assert isinstance(output, str)
        assert "Insufficient GPU Memory" in output
    
    def test_create_dependency_error(self):
        """Test creating dependency management error"""
        missing_deps = ["torch>=2.0.0", "diffusers>=0.21.0"]
        output = create_dependency_error(
            "dependency_missing", missing_deps
        )
        
        assert isinstance(output, str)
        assert "Required Dependencies Missing" in output
    
    def test_create_video_error(self):
        """Test creating video processing error"""
        output = create_video_error(
            "encoding_failed", "/path/to/output.mp4"
        )
        
        assert isinstance(output, str)
        assert "Video Encoding Failed" in output


class TestRequirementsCoverage:
    """Test coverage of specific requirements"""
    
    def setup_method(self):
        self.generator = ErrorMessageGenerator()
        self.disclosure = ProgressiveErrorDisclosure()
    
    def test_requirement_1_4_clear_instructions(self):
        """Test Requirement 1.4: Clear instructions for obtaining pipeline code"""
        context = ErrorContext(
            model_path="/path/to/wan/model",
            pipeline_class="WanPipeline"
        )
        
        error_msg = self.generator.generate_error_message(
            "missing_pipeline_class", context
        )
        
        # Should provide clear instructions
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Install WanPipeline Package" in action_titles
        assert "Manual Pipeline Installation" in action_titles
        
        # Should have commands and URLs
        install_action = next(
            action for action in error_msg.recovery_actions 
            if "Install WanPipeline Package" in action.title
        )
        assert install_action.command is not None
        assert "pip install" in install_action.command
    
    def test_requirement_2_4_specific_vae_error_messages(self):
        """Test Requirement 2.4: Specific error messages about VAE compatibility"""
        context = ErrorContext(
            model_path="/path/to/wan/model",
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        )
        
        error_msg = self.generator.generate_error_message(
            "vae_shape_mismatch", context
        )
        
        # Should mention specific VAE requirements
        assert "3D architecture" in error_msg.detailed_description
        assert "shape [384, ...]" in error_msg.detailed_description
        assert "random initialization" in error_msg.detailed_description
    
    def test_requirement_3_4_clear_argument_error_messages(self):
        """Test Requirement 3.4: Clear error messages about required arguments"""
        context = ErrorContext(
            pipeline_class="WanPipeline",
            attempted_operation="pipeline_initialization"
        )
        
        error_msg = self.generator.generate_error_message(
            "pipeline_args_mismatch", context
        )
        
        assert "arguments" in error_msg.summary.lower()
        assert error_msg.category == ErrorCategory.PIPELINE_INITIALIZATION
        
        # Should provide guidance on fixing arguments
        action_descriptions = [action.description for action in error_msg.recovery_actions]
        assert any("argument" in desc.lower() for desc in action_descriptions)
    
    def test_requirement_4_4_diagnostic_information(self):
        """Test Requirement 4.4: Diagnostic information about compatibility issues"""
        context = ErrorContext(
            model_path="/path/to/model",
            model_name="TestModel",
            system_info={"gpu": "RTX 4090", "vram": "24GB"}
        )
        
        exception = RuntimeError("Model loading failed")
        
        error_msg = self.generator.generate_error_message(
            "missing_components", context, exception
        )
        
        # Should include diagnostic information
        assert error_msg.technical_details is not None
        assert error_msg.debug_info is not None
        assert "RuntimeError" in error_msg.technical_details
        assert "Model loading failed" in error_msg.technical_details
    
    def test_requirement_6_4_local_installation_alternatives(self):
        """Test Requirement 6.4: Local installation alternatives for security restrictions"""
        context = ErrorContext(
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            pipeline_class="WanPipeline"
        )
        
        error_msg = self.generator.generate_error_message(
            "remote_code_blocked", context
        )
        
        # Should provide local installation alternatives
        action_titles = [action.title for action in error_msg.recovery_actions]
        assert "Install Pipeline Locally" in action_titles
        assert "Use Pre-installed Pipeline" in action_titles
        
        # Should have specific commands
        local_install_action = next(
            action for action in error_msg.recovery_actions 
            if "Install Pipeline Locally" in action.title
        )
        assert local_install_action.command is not None
        assert "git clone" in local_install_action.command
    
    def test_requirement_7_4_clear_installation_guidance(self):
        """Test Requirement 7.4: Clear installation guidance for encoding dependencies"""
        context = ErrorContext(
            user_inputs={"output_path": "/path/to/video.mp4"},
            attempted_operation="video_encoding"
        )
        
        error_msg = self.generator.generate_error_message(
            "ffmpeg_missing", context
        )
        
        # Should provide clear installation guidance
        action_titles = [action.title for action in error_msg.recovery_actions]
        ffmpeg_actions = [title for title in action_titles if "FFmpeg" in title]
        assert len(ffmpeg_actions) > 0
        
        # Should have URL for installation
        ffmpeg_action = next(
            action for action in error_msg.recovery_actions 
            if "FFmpeg" in action.title
        )
        assert ffmpeg_action.url is not None
        assert "ffmpeg.org" in ffmpeg_action.url


class TestProgressiveDisclosure:
    """Test progressive error disclosure functionality"""
    
    def setup_method(self):
        self.formatter = ErrorMessageFormatter()
        self.sample_error = ErrorMessage(
            title="Test Error",
            summary="Basic error summary",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.PIPELINE_LOADING,
            detailed_description="Detailed explanation of the error",
            technical_details="Technical details for developers",
            debug_info={"debug": "information"},
            recovery_actions=[
                RecoveryAction(
                    title="Basic Fix",
                    description="Simple fix description",
                    priority=1
                )
            ]
        )
    
    def test_basic_disclosure_level(self):
        """Test basic disclosure level shows only essential information"""
        output = self.formatter.format_for_console(self.sample_error, "basic")
        
        # Should include basic information
        assert "Test Error" in output
        assert "Basic error summary" in output
        assert "Basic Fix" in output
        
        # Should not include detailed information
        assert "Detailed explanation" not in output
        assert "Technical details" not in output
        assert "Debug Information" not in output
    
    def test_detailed_disclosure_level(self):
        """Test detailed disclosure level shows additional context"""
        output = self.formatter.format_for_console(self.sample_error, "detailed")
        
        # Should include basic and detailed information
        assert "Test Error" in output
        assert "Basic error summary" in output
        assert "Detailed explanation of the error" in output
        assert "Basic Fix" in output
        
        # Should not include diagnostic information
        assert "Technical details for developers" not in output
        assert "Debug Information" not in output
    
    def test_diagnostic_disclosure_level(self):
        """Test diagnostic disclosure level shows all information"""
        output = self.formatter.format_for_console(self.sample_error, "diagnostic")
        
        # Should include all information
        assert "Test Error" in output
        assert "Basic error summary" in output
        assert "Detailed explanation of the error" in output
        assert "Technical details for developers" in output
        assert "Debug Information" in output
        assert '"debug": "information"' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])