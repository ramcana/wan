"""
Integration Tests for Generation Pipeline Improvements
Tests complete generation workflows with validation, orchestration, and retry mechanisms
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from PIL import Image
import numpy as np

# Import the components we're testing
from enhanced_generation_pipeline import (
    EnhancedGenerationPipeline, GenerationContext, PipelineResult,
    GenerationStage, RetryStrategy, get_enhanced_pipeline
)
from generation_orchestrator import GenerationRequest, GenerationMode
from generation_mode_router import (
    GenerationModeRouter, GenerationModeType, get_generation_mode_router
)
from input_validation import ValidationResult
from error_handler import ErrorCategory, ErrorSeverity, UserFriendlyError
from utils import generate_video, generate_video_enhanced

class TestGenerationPipelineIntegration:
    """Test suite for complete generation pipeline integration"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "directories": {
                "models_directory": "models",
                "outputs_directory": "outputs",
                "loras_directory": "loras"
            },
            "generation": {
                "max_retry_attempts": 3,
                "enable_auto_optimization": True,
                "enable_preflight_checks": True,
                "max_prompt_length": 512
            },
            "optimization": {
                "max_vram_usage_gb": 12,
                "default_quantization": "bf16"
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay_seconds": 1,
                "generation_timeout_seconds": 300
            }
        }
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def t2v_request(self):
        """Sample T2V generation request"""
        return GenerationRequest(
            model_type="t2v-A14B",
            prompt="A beautiful sunset over the ocean with waves",
            resolution="720p",
            steps=50,
            guidance_scale=7.5,
            fps=24,
            duration=4
        )
    
    @pytest.fixture
    def i2v_request(self, sample_image):
        """Sample I2V generation request"""
        return GenerationRequest(
            model_type="i2v-A14B",
            prompt="",  # Optional for I2V
            image=sample_image,
            resolution="720p",
            steps=40,
            guidance_scale=7.5,
            fps=24,
            duration=4
        )
    
    @pytest.fixture
    def ti2v_request(self, sample_image):
        """Sample TI2V generation request"""
        return GenerationRequest(
            model_type="ti2v-5B",
            prompt="A cinematic scene with dramatic lighting",
            image=sample_image,
            resolution="720p",
            steps=30,
            guidance_scale=7.5,
            fps=24,
            duration=4
        )

class TestPreflightChecksIntegration:
    """Test pre-flight checks integration"""
    
    def test_preflight_checks_enabled(self, config):
        """Test that pre-flight checks are properly enabled"""
        pipeline = EnhancedGenerationPipeline(config)
        assert pipeline.enable_preflight_checks == True
        
        # Test configuration override
        config["generation"]["enable_preflight_checks"] = False
        pipeline = EnhancedGenerationPipeline(config)
        assert pipeline.enable_preflight_checks == False
    
    @pytest.mark.asyncio
    async def test_preflight_validation_integration(self, config, t2v_request):
        """Test integration of preflight checks with validation"""
        pipeline = EnhancedGenerationPipeline(config)
        context = GenerationContext(request=t2v_request, task_id="test_preflight")
        
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            # Mock successful preflight with optimization recommendations
            mock_preflight.return_value = PreflightResult(
                can_proceed=True,
                model_status=ModelStatus(is_available=True, is_loaded=False),
                resource_estimate=ResourceEstimate(6000, 3000, 45, 85),
                optimization_recommendations=[
                    "Enable gradient checkpointing",
                    "Use CPU offloading for VAE"
                ],
                blocking_issues=[],
                warnings=["High VRAM usage expected"]
            )
            
            result = await pipeline._run_preflight_checks(context)
            
            assert result.success
            assert "preflight_result" in context.metadata
            
            # Check that optimization recommendations are stored
            preflight_result = context.metadata["preflight_result"]
            assert len(preflight_result.optimization_recommendations) == 2
            assert "Enable gradient checkpointing" in preflight_result.optimization_recommendations
    
    @pytest.mark.asyncio
    async def test_preflight_blocking_issues(self, config, t2v_request):
        """Test handling of preflight blocking issues"""
        pipeline = EnhancedGenerationPipeline(config)
        context = GenerationContext(request=t2v_request, task_id="test_blocking")
        
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            # Mock preflight with blocking issues
            mock_preflight.return_value = PreflightResult(
                can_proceed=False,
                model_status=ModelStatus(
                    is_available=False, 
                    is_loaded=False,
                    loading_error="Model not found"
                ),
                resource_estimate=ResourceEstimate(0, 0, 0, 0),
                blocking_issues=[
                    "Model not available: Model not found",
                    "Insufficient VRAM for generation"
                ],
                warnings=[]
            )
            
            result = await pipeline._run_preflight_checks(context)
            
            assert not result.success
            assert result.error.category == ErrorCategory.SYSTEM_RESOURCE
            assert "Model not found" in result.error.message
            assert "Insufficient VRAM" in result.error.message

class TestGenerationModeRouting:
    """Test generation mode routing integration"""
    
    def test_mode_router_initialization(self, config):
        """Test that mode router is properly initialized"""
        router = get_generation_mode_router(config)
        assert router is not None
        
        # Test available modes
        modes = router.list_available_modes()
        assert len(modes) == 3
        
        mode_names = [mode["mode"] for mode in modes]
        assert "t2v-A14B" in mode_names
        assert "i2v-A14B" in mode_names
        assert "ti2v-5B" in mode_names
    
    def test_t2v_mode_routing(self, config, t2v_request):
        """Test T2V mode routing and validation"""
        router = get_generation_mode_router(config)
        result = router.route_request(t2v_request)
        
        assert result.is_valid
        assert result.mode == GenerationModeType.TEXT_TO_VIDEO
        assert result.optimized_request.model_type == "t2v-A14B"
        
        # Check that optimization was applied
        assert result.optimized_request.steps >= 30  # T2V minimum optimization
    
    def test_i2v_mode_routing(self, config, i2v_request):
        """Test I2V mode routing and validation"""
        router = get_generation_mode_router(config)
        result = router.route_request(i2v_request)
        
        assert result.is_valid
        assert result.mode == GenerationModeType.IMAGE_TO_VIDEO
        assert result.optimized_request.model_type == "i2v-A14B"
        
        # Check I2V specific optimizations
        assert result.optimized_request.steps <= 60  # I2V maximum optimization
        assert result.optimized_request.strength <= 0.8  # Strength adjustment
    
    def test_ti2v_mode_routing(self, config, ti2v_request):
        """Test TI2V mode routing and validation"""
        router = get_generation_mode_router(config)
        result = router.route_request(ti2v_request)
        
        assert result.is_valid
        assert result.mode == GenerationModeType.TEXT_IMAGE_TO_VIDEO
        assert result.optimized_request.model_type == "ti2v-5B"
        
        # Check TI2V specific optimizations
        assert result.optimized_request.lora_config == {}  # LoRA removed
        assert result.optimized_request.steps <= 40  # TI2V efficiency optimization
    
    def test_mode_validation_failures(self, config):
        """Test mode validation failure scenarios"""
        router = get_generation_mode_router(config)
        
        # Test T2V without prompt
        invalid_t2v = GenerationRequest(
            model_type="t2v-A14B",
            prompt="",  # Empty prompt should fail
            resolution="720p",
            steps=50
        )
        
        result = router.route_request(invalid_t2v)
        assert not result.is_valid
        assert "requires a text prompt" in " ".join(result.validation_issues)
        
        # Test I2V without image
        invalid_i2v = GenerationRequest(
            model_type="i2v-A14B",
            prompt="test",
            image=None,  # Missing image should fail
            resolution="720p",
            steps=40
        )
        
        result = router.route_request(invalid_i2v)
        assert not result.is_valid
        assert "requires an input image" in " ".join(result.validation_issues)
        
        # Test unsupported resolution
        invalid_resolution = GenerationRequest(
            model_type="ti2v-5B",
            prompt="test",
            image=Image.new("RGB", (512, 512)),
            resolution="4K",  # Unsupported for TI2V
            steps=30
        )
        
        result = router.route_request(invalid_resolution)
        assert not result.is_valid
        assert "not supported" in " ".join(result.validation_issues)

class TestRetryMechanisms:
    """Test automatic retry mechanisms with optimization"""
    
    @pytest.mark.asyncio
    async def test_vram_error_retry_optimization(self, config, t2v_request):
        """Test retry optimization for VRAM errors"""
        pipeline = EnhancedGenerationPipeline(config)
        
        # Test VRAM error optimization
        vram_error = UserFriendlyError(
            category=ErrorCategory.VRAM_MEMORY,
            severity=ErrorSeverity.HIGH,
            title="VRAM Error",
            message="CUDA out of memory",
            recovery_suggestions=[],
            recovery_actions=[]
        )
        
        # Test first retry (attempt 2)
        optimized_request = pipeline._apply_retry_optimizations(t2v_request, vram_error, 2)
        
        assert optimized_request.steps < t2v_request.steps  # Steps reduced
        if t2v_request.resolution == "1080p":
            assert optimized_request.resolution == "720p"  # Resolution reduced
        
        # Test second retry (attempt 3)
        optimized_request_2 = pipeline._apply_retry_optimizations(t2v_request, vram_error, 3)
        
        assert optimized_request_2.steps < optimized_request.steps  # Further reduction
        assert optimized_request_2.resolution == "720p"  # Forced to 720p
        assert optimized_request_2.lora_config == {}  # LoRAs removed
    
    @pytest.mark.asyncio
    async def test_generation_error_retry_optimization(self, config, t2v_request):
        """Test retry optimization for generation pipeline errors"""
        pipeline = EnhancedGenerationPipeline(config)
        
        generation_error = UserFriendlyError(
            category=ErrorCategory.GENERATION_PIPELINE,
            severity=ErrorSeverity.HIGH,
            title="Generation Error",
            message="Pipeline execution failed",
            recovery_suggestions=[],
            recovery_actions=[]
        )
        
        # Test optimization for generation errors
        optimized_request = pipeline._apply_retry_optimizations(t2v_request, generation_error, 2)
        
        assert optimized_request.guidance_scale != t2v_request.guidance_scale  # Guidance scale adjusted
        
        # Test aggressive optimization
        optimized_request_2 = pipeline._apply_retry_optimizations(t2v_request, generation_error, 3)
        
        assert optimized_request_2.steps < t2v_request.steps  # Steps reduced
        assert optimized_request_2.guidance_scale == 7.5  # Reset to default
    
    def test_retry_decision_logic(self, config):
        """Test retry decision logic for different error types"""
        pipeline = EnhancedGenerationPipeline(config)
        context = GenerationContext(request=GenerationRequest(model_type="t2v-A14B", prompt="test"), task_id="test")
        
        # Should retry VRAM errors
        vram_error = UserFriendlyError(category=ErrorCategory.VRAM_MEMORY, severity=ErrorSeverity.HIGH, title="VRAM", message="OOM", recovery_suggestions=[], recovery_actions=[])
        assert pipeline._should_retry(vram_error, context) == True
        
        # Should retry system resource errors
        system_error = UserFriendlyError(category=ErrorCategory.SYSTEM_RESOURCE, severity=ErrorSeverity.HIGH, title="System", message="Resource issue", recovery_suggestions=[], recovery_actions=[])
        assert pipeline._should_retry(system_error, context) == True
        
        # Should retry generation pipeline errors
        gen_error = UserFriendlyError(category=ErrorCategory.GENERATION_PIPELINE, severity=ErrorSeverity.HIGH, title="Generation", message="Pipeline failed", recovery_suggestions=[], recovery_actions=[])
        assert pipeline._should_retry(gen_error, context) == True
        
        # Should NOT retry validation errors
        val_error = UserFriendlyError(category=ErrorCategory.INPUT_VALIDATION, severity=ErrorSeverity.HIGH, title="Validation", message="Invalid input", recovery_suggestions=[], recovery_actions=[])
        assert pipeline._should_retry(val_error, context) == False
        
        # Should NOT retry file system errors
        fs_error = UserFriendlyError(category=ErrorCategory.FILE_SYSTEM, severity=ErrorSeverity.HIGH, title="File", message="File not found", recovery_suggestions=[], recovery_actions=[])
        assert pipeline._should_retry(fs_error, context) == False

class TestEndToEndWorkflows:
    """Test complete end-to-end generation workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_t2v_workflow_with_pipeline(self, config, t2v_request):
        """Test complete T2V workflow using enhanced pipeline"""
        with patch('enhanced_generation_pipeline.get_generation_mode_router') as mock_router_getter, \
             patch.object(EnhancedGenerationPipeline, '_execute_legacy_generation') as mock_legacy_gen, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Mock mode router
            mock_router = Mock()
            from generation_mode_router import ModeValidationResult, GenerationModeType
            mock_router.route_request.return_value = ModeValidationResult(
                is_valid=True,
                mode=GenerationModeType.TEXT_TO_VIDEO,
                validation_issues=[],
                warnings=[],
                optimized_request=t2v_request
            )
            mock_router_getter.return_value = mock_router
            
            # Mock legacy generation
            mock_legacy_gen.return_value = {
                "success": True,
                "output_path": "/tmp/test_t2v_video.mp4",
                "generation_time": 45.0
            }
            
            # Mock file system
            mock_exists.return_value = True
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB file
            
            # Create pipeline and execute
            pipeline = EnhancedGenerationPipeline(config)
            
            # Mock all validation and preflight stages
            with patch.object(pipeline, '_validate_inputs') as mock_validate, \
                 patch.object(pipeline, '_run_preflight_checks') as mock_preflight, \
                 patch.object(pipeline, '_prepare_generation') as mock_prepare, \
                 patch.object(pipeline, '_post_process_output') as mock_post:
                
                mock_validate.return_value = PipelineResult(success=True, context=None)
                mock_preflight.return_value = PipelineResult(success=True, context=None)
                mock_prepare.return_value = PipelineResult(success=True, context=None)
                mock_post.return_value = PipelineResult(success=True, output_path="/tmp/test_t2v_video.mp4", context=None)
                
                result = await pipeline.generate_video(t2v_request, "test_t2v_complete")
                
                assert result.success
                assert result.output_path == "/tmp/test_t2v_video.mp4"
                assert result.generation_time is not None
                assert result.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_complete_i2v_workflow_with_retry(self, config, i2v_request):
        """Test complete I2V workflow with retry mechanism"""
        with patch('enhanced_generation_pipeline.get_generation_mode_router') as mock_router_getter, \
             patch.object(EnhancedGenerationPipeline, '_execute_legacy_generation') as mock_legacy_gen, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Mock mode router
            mock_router = Mock()
            from generation_mode_router import ModeValidationResult, GenerationModeType
            mock_router.route_request.return_value = ModeValidationResult(
                is_valid=True,
                mode=GenerationModeType.IMAGE_TO_VIDEO,
                validation_issues=[],
                warnings=[],
                optimized_request=i2v_request
            )
            mock_router_getter.return_value = mock_router
            
            # Mock legacy generation - first attempt fails, second succeeds
            call_count = 0
            def mock_generation_with_retry(context, progress_callback):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    return {
                        "success": False,
                        "error": "CUDA out of memory",
                        "recovery_suggestions": ["Reduce resolution", "Enable CPU offloading"]
                    }
                else:
                    return {
                        "success": True,
                        "output_path": "/tmp/test_i2v_video_retry.mp4",
                        "generation_time": 35.0
                    }
            
            mock_legacy_gen.side_effect = mock_generation_with_retry
            
            # Mock file system
            mock_exists.return_value = True
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB file
            
            # Create pipeline and execute
            pipeline = EnhancedGenerationPipeline(config)
            
            # Mock validation and preflight stages
            with patch.object(pipeline, '_validate_inputs') as mock_validate, \
                 patch.object(pipeline, '_run_preflight_checks') as mock_preflight, \
                 patch.object(pipeline, '_prepare_generation') as mock_prepare, \
                 patch.object(pipeline, '_post_process_output') as mock_post:
                
                mock_validate.return_value = PipelineResult(success=True, context=None)
                mock_preflight.return_value = PipelineResult(success=True, context=None)
                mock_prepare.return_value = PipelineResult(success=True, context=None)
                mock_post.return_value = PipelineResult(success=True, output_path="/tmp/test_i2v_video_retry.mp4", context=None)
                
                result = await pipeline.generate_video(i2v_request, "test_i2v_retry")
                
                assert result.success
                assert result.output_path == "/tmp/test_i2v_video_retry.mp4"
                assert result.retry_count == 1  # One retry occurred
                assert call_count == 2  # Two generation attempts
    
    def test_enhanced_generation_function_integration(self, config, t2v_request):
        """Test integration with enhanced generation function"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline, \
             patch('asyncio.run') as mock_asyncio_run:
            
            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.add_progress_callback = Mock()
            
            # Mock successful generation result
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_path = "/tmp/enhanced_test.mp4"
            mock_result.generation_time = 42.0
            mock_result.retry_count = 0
            mock_result.context = Mock()
            mock_result.context.metadata = {"test": "metadata"}
            
            mock_asyncio_run.return_value = mock_result
            mock_get_pipeline.return_value = mock_pipeline
            
            # Test the enhanced generation function
            result = generate_video_enhanced(
                model_type="t2v-A14B",
                prompt="Test prompt",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == True
            assert result["output_path"] == "/tmp/enhanced_test.mp4"
            assert result["generation_time"] == 42.0
            assert result["retry_count"] == 0
            assert "metadata" in result
    
    def test_fallback_to_legacy_generation(self, config):
        """Test fallback to legacy generation when enhanced pipeline fails"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline, \
             patch('utils.generate_video_legacy') as mock_legacy_gen:
            
            # Mock pipeline returning None (not available)
            mock_get_pipeline.return_value = None
            
            # Mock legacy generation success
            mock_legacy_gen.return_value = {
                "success": True,
                "output_path": "/tmp/legacy_test.mp4",
                "generation_time": None,
                "retry_count": 0
            }
            
            # Test fallback
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test prompt",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == True
            assert result["output_path"] == "/tmp/legacy_test.mp4"
            
            # Verify legacy generation was called
            mock_legacy_gen.assert_called_once()

class TestErrorHandlingIntegration:
    """Test error handling integration across the pipeline"""
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, config, t2v_request):
        """Test handling of validation errors in the pipeline"""
        pipeline = EnhancedGenerationPipeline(config)
        
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_validate:
            # Mock validation failure
            validation_result = ValidationResult(is_valid=False)
            validation_result.add_error("Prompt too long", "prompt", "Shorten the prompt")
            mock_validate.return_value = validation_result
            
            result = await pipeline.generate_video(t2v_request, "test_validation_error")
            
            assert not result.success
            assert result.error.category == ErrorCategory.INPUT_VALIDATION
            assert "Prompt validation failed" in result.error.message
            assert len(result.error.recovery_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_model_loading_error_handling(self, config, t2v_request):
        """Test handling of model loading errors"""
        pipeline = EnhancedGenerationPipeline(config)
        
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            # Mock model loading failure
            mock_preflight.return_value = PreflightResult(
                can_proceed=False,
                model_status=ModelStatus(
                    is_available=False,
                    is_loaded=False,
                    loading_error="Model directory not found: /path/to/model"
                ),
                resource_estimate=ResourceEstimate(0, 0, 0, 0),
                blocking_issues=["Model not available: Model directory not found"],
                warnings=[]
            )
            
            result = await pipeline.generate_video(t2v_request, "test_model_error")
            
            assert not result.success
            assert result.error.category == ErrorCategory.SYSTEM_RESOURCE
            assert "Model directory not found" in result.error.message
    
    @pytest.mark.asyncio
    async def test_generation_failure_with_recovery_suggestions(self, config, t2v_request):
        """Test generation failure with proper recovery suggestions"""
        pipeline = EnhancedGenerationPipeline(config)
        
        with patch('enhanced_generation_pipeline.get_generation_mode_router') as mock_router_getter, \
             patch.object(pipeline, '_execute_legacy_generation') as mock_legacy_gen:
            
            # Mock mode router
            mock_router = Mock()
            from generation_mode_router import ModeValidationResult, GenerationModeType
            mock_router.route_request.return_value = ModeValidationResult(
                is_valid=True,
                mode=GenerationModeType.TEXT_TO_VIDEO,
                validation_issues=[],
                warnings=[],
                optimized_request=t2v_request
            )
            mock_router_getter.return_value = mock_router
            
            # Mock generation failure with recovery suggestions
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Insufficient VRAM for current settings",
                "recovery_suggestions": [
                    "Reduce resolution to 720p",
                    "Decrease number of inference steps",
                    "Enable CPU offloading",
                    "Use int8 quantization"
                ]
            }
            
            # Mock other pipeline stages
            with patch.object(pipeline, '_validate_inputs') as mock_validate, \
                 patch.object(pipeline, '_run_preflight_checks') as mock_preflight, \
                 patch.object(pipeline, '_prepare_generation') as mock_prepare:
                
                mock_validate.return_value = PipelineResult(success=True, context=None)
                mock_preflight.return_value = PipelineResult(success=True, context=None)
                mock_prepare.return_value = PipelineResult(success=True, context=None)
                
                result = await pipeline.generate_video(t2v_request, "test_recovery_suggestions")
                
                assert not result.success
                assert result.error.category == ErrorCategory.GENERATION_PIPELINE
                assert "Insufficient VRAM" in result.error.message
                assert len(result.error.recovery_suggestions) == 4
                assert "Reduce resolution to 720p" in result.error.recovery_suggestions

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
