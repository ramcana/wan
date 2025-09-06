"""
Integration Tests for Enhanced Generation Pipeline
Tests complete generation workflows for T2V, I2V, and TI2V modes
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from enhanced_generation_pipeline import (
    EnhancedGenerationPipeline, GenerationContext, PipelineResult,
    GenerationStage, RetryStrategy
)
from generation_orchestrator import GenerationRequest, GenerationMode
from generation_mode_router import GenerationModeRouter, GenerationModeType
from input_validation import ValidationResult
from error_handler import ErrorCategory, ErrorSeverity, UserFriendlyError

class TestEnhancedGenerationPipeline:
    """Test suite for the enhanced generation pipeline"""
    
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
            }
        }
    
    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline instance for testing"""
        return EnhancedGenerationPipeline(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple RGB image
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

class TestGenerationModeRouting:
    """Test generation mode routing and validation"""
    
    @pytest.fixture
    def router(self, config):
        """Create router instance for testing"""
        return GenerationModeRouter(config)
    
    def test_t2v_mode_detection(self, router, t2v_request):
        """Test T2V mode detection and validation"""
        result = router.route_request(t2v_request)
        
        assert result.is_valid
        assert result.mode == GenerationModeType.TEXT_TO_VIDEO
        assert result.optimized_request.model_type == "t2v-A14B"
    
    def test_i2v_mode_detection(self, router, i2v_request):
        """Test I2V mode detection and validation"""
        result = router.route_request(i2v_request)
        
        assert result.is_valid
        assert result.mode == GenerationModeType.IMAGE_TO_VIDEO
        assert result.optimized_request.model_type == "i2v-A14B"
    
    def test_ti2v_mode_detection(self, router, ti2v_request):
        """Test TI2V mode detection and validation"""
        result = router.route_request(ti2v_request)
        
        assert result.is_valid
        assert result.mode == GenerationModeType.TEXT_IMAGE_TO_VIDEO
        assert result.optimized_request.model_type == "ti2v-5B"
        # TI2V should remove LoRA config
        assert result.optimized_request.lora_config == {}
    
    def test_invalid_prompt_validation(self, router, config):
        """Test validation of invalid prompts"""
        invalid_request = GenerationRequest(
            model_type="t2v-A14B",
            prompt="",  # Empty prompt for T2V should fail
            resolution="720p",
            steps=50
        )
        
        result = router.route_request(invalid_request)
        assert not result.is_valid
        assert "requires a text prompt" in " ".join(result.validation_issues)
    
    def test_unsupported_resolution_validation(self, router, t2v_request):
        """Test validation of unsupported resolutions"""
        t2v_request.resolution = "4K"  # Unsupported resolution
        
        result = router.route_request(t2v_request)
        assert not result.is_valid
        assert "not supported" in " ".join(result.validation_issues)
    
    def test_lora_compatibility_validation(self, router, ti2v_request):
        """Test LoRA compatibility validation"""
        ti2v_request.lora_config = {"test_lora": 1.0}  # TI2V doesn't support LoRA
        
        result = router.route_request(ti2v_request)
        # Should still be valid but LoRA should be removed in optimization
        assert result.is_valid
        assert result.optimized_request.lora_config == {}

class TestPipelineValidation:
    """Test pipeline validation stages"""
    
    @pytest.mark.asyncio
    async def test_successful_validation(self, pipeline, t2v_request):
        """Test successful input validation"""
        context = GenerationContext(request=t2v_request, task_id="test_001")
        
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_prompt_val, \
             patch.object(pipeline.orchestrator.config_validator, 'validate') as mock_config_val:
            
            # Mock successful validation
            mock_prompt_val.return_value = ValidationResult(is_valid=True)
            mock_config_val.return_value = ValidationResult(is_valid=True)
            
            result = await pipeline._validate_inputs(context)
            
            assert result.success
            assert result.context == context
    
    @pytest.mark.asyncio
    async def test_prompt_validation_failure(self, pipeline, t2v_request):
        """Test prompt validation failure"""
        context = GenerationContext(request=t2v_request, task_id="test_002")
        
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_val:
            # Mock validation failure
            validation_result = ValidationResult(is_valid=False)
            validation_result.add_error("Prompt too long", "prompt", "Shorten the prompt")
            mock_val.return_value = validation_result
            
            result = await pipeline._validate_inputs(context)
            
            assert not result.success
            assert result.error.category == ErrorCategory.INPUT_VALIDATION
            assert "Prompt validation failed" in result.error.message
    
    @pytest.mark.asyncio
    async def test_image_validation_failure(self, pipeline, i2v_request):
        """Test image validation failure"""
        context = GenerationContext(request=i2v_request, task_id="test_003")
        
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_prompt_val, \
             patch.object(pipeline.orchestrator.image_validator, 'validate') as mock_image_val, \
             patch.object(pipeline.orchestrator.config_validator, 'validate') as mock_config_val:
            
            # Mock successful prompt and config validation
            mock_prompt_val.return_value = ValidationResult(is_valid=True)
            mock_config_val.return_value = ValidationResult(is_valid=True)
            
            # Mock image validation failure
            validation_result = ValidationResult(is_valid=False)
            validation_result.add_error("Invalid image format", "image", "Use PNG or JPG")
            mock_image_val.return_value = validation_result
            
            result = await pipeline._validate_inputs(context)
            
            assert not result.success
            assert result.error.category == ErrorCategory.INPUT_VALIDATION
            assert "Image validation failed" in result.error.message

class TestPipelinePreflightChecks:
    """Test pipeline preflight checks"""
    
    @pytest.mark.asyncio
    async def test_successful_preflight_checks(self, pipeline, t2v_request):
        """Test successful preflight checks"""
        context = GenerationContext(request=t2v_request, task_id="test_004")
        
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            # Mock successful preflight
            mock_preflight.return_value = PreflightResult(
                can_proceed=True,
                model_status=ModelStatus(is_available=True, is_loaded=False),
                resource_estimate=ResourceEstimate(6000, 3000, 45, 85),
                optimization_recommendations=[],
                blocking_issues=[],
                warnings=[]
            )
            
            result = await pipeline._run_preflight_checks(context)
            
            assert result.success
            assert "preflight_result" in context.metadata
    
    @pytest.mark.asyncio
    async def test_preflight_checks_failure(self, pipeline, t2v_request):
        """Test preflight checks failure"""
        context = GenerationContext(request=t2v_request, task_id="test_005")
        
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            # Mock preflight failure
            mock_preflight.return_value = PreflightResult(
                can_proceed=False,
                model_status=ModelStatus(is_available=False, is_loaded=False, loading_error="Model not found"),
                resource_estimate=ResourceEstimate(0, 0, 0, 0),
                blocking_issues=["Model not available", "Insufficient VRAM"],
                warnings=[]
            )
            
            result = await pipeline._run_preflight_checks(context)
            
            assert not result.success
            assert result.error.category == ErrorCategory.SYSTEM_RESOURCE
            assert "Pre-flight Check Failed" in result.error.title

class TestPipelineGeneration:
    """Test actual generation execution"""
    
    @pytest.mark.asyncio
    async def test_successful_generation(self, pipeline, t2v_request):
        """Test successful video generation"""
        context = GenerationContext(request=t2v_request, task_id="test_006")
        
        with patch('utils.generate_video') as mock_generate:
            # Mock successful generation
            mock_generate.return_value = {
                "success": True,
                "output_path": "/tmp/test_video.mp4"
            }
            
            result = await pipeline._execute_generation(context)
            
            assert result.success
            assert result.output_path == "/tmp/test_video.mp4"
    
    @pytest.mark.asyncio
    async def test_generation_failure(self, pipeline, t2v_request):
        """Test generation failure handling"""
        context = GenerationContext(request=t2v_request, task_id="test_007")
        
        with patch('utils.generate_video') as mock_generate:
            # Mock generation failure
            mock_generate.return_value = {
                "success": False,
                "error": "CUDA out of memory"
            }
            
            result = await pipeline._execute_generation(context)
            
            assert not result.success
            assert result.error.category == ErrorCategory.GENERATION_PIPELINE
            assert "Generation Failed" in result.error.title

class TestPipelineRetryMechanism:
    """Test retry mechanisms and optimization"""
    
    @pytest.mark.asyncio
    async def test_retry_with_optimization(self, pipeline, t2v_request):
        """Test retry with parameter optimization"""
        context = GenerationContext(
            request=t2v_request,
            task_id="test_008",
            max_attempts=2
        )
        
        # Mock first attempt failure, second attempt success
        call_count = 0
        async def mock_single_pipeline(ctx):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First attempt fails with VRAM error
                error = UserFriendlyError(
                    category=ErrorCategory.VRAM_MEMORY,
                    severity=ErrorSeverity.HIGH,
                    title="VRAM Error",
                    message="Out of memory",
                    recovery_suggestions=[],
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=ctx)
            else:
                # Second attempt succeeds
                return PipelineResult(
                    success=True,
                    output_path="/tmp/test_video.mp4",
                    context=ctx
                )
        
        with patch.object(pipeline, '_execute_single_pipeline', side_effect=mock_single_pipeline):
            result = await pipeline._execute_pipeline_with_retry(context)
            
            assert result.success
            assert result.retry_count == 1
            assert call_count == 2
    
    def test_retry_optimization_vram_error(self, pipeline, t2v_request):
        """Test retry optimization for VRAM errors"""
        error = UserFriendlyError(
            category=ErrorCategory.VRAM_MEMORY,
            severity=ErrorSeverity.HIGH,
            title="VRAM Error",
            message="Out of memory",
            recovery_suggestions=[],
            recovery_actions=[]
        )
        
        optimized = pipeline._apply_retry_optimizations(t2v_request, error, 2)
        
        # Should reduce steps and potentially resolution
        assert optimized.steps < t2v_request.steps
        if t2v_request.resolution == "1080p":
            assert optimized.resolution == "720p"
    
    def test_retry_optimization_generation_error(self, pipeline, t2v_request):
        """Test retry optimization for generation errors"""
        error = UserFriendlyError(
            category=ErrorCategory.GENERATION_PIPELINE,
            severity=ErrorSeverity.HIGH,
            title="Generation Error",
            message="Pipeline failed",
            recovery_suggestions=[],
            recovery_actions=[]
        )
        
        optimized = pipeline._apply_retry_optimizations(t2v_request, error, 2)
        
        # Should adjust guidance scale
        assert optimized.guidance_scale != t2v_request.guidance_scale

class TestEndToEndWorkflows:
    """Test complete end-to-end generation workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_t2v_workflow(self, pipeline, t2v_request):
        """Test complete T2V generation workflow"""
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_prompt_val, \
             patch.object(pipeline.orchestrator.config_validator, 'validate') as mock_config_val, \
             patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight, \
             patch.object(pipeline.orchestrator, 'prepare_generation') as mock_prepare, \
             patch('utils.generate_video') as mock_generate, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Mock all stages to succeed
            mock_prompt_val.return_value = ValidationResult(is_valid=True)
            mock_config_val.return_value = ValidationResult(is_valid=True)
            
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            mock_preflight.return_value = PreflightResult(
                can_proceed=True,
                model_status=ModelStatus(is_available=True, is_loaded=False),
                resource_estimate=ResourceEstimate(6000, 3000, 45, 85),
                optimization_recommendations=[],
                blocking_issues=[],
                warnings=[]
            )
            
            mock_prepare.return_value = (True, "Success")
            mock_generate.return_value = {
                "success": True,
                "output_path": "/tmp/test_video.mp4"
            }
            
            mock_exists.return_value = True
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB file
            
            result = await pipeline.generate_video(t2v_request, "test_t2v")
            
            assert result.success
            assert result.output_path == "/tmp/test_video.mp4"
            assert result.generation_time is not None
    
    @pytest.mark.asyncio
    async def test_complete_i2v_workflow(self, pipeline, i2v_request):
        """Test complete I2V generation workflow"""
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_prompt_val, \
             patch.object(pipeline.orchestrator.image_validator, 'validate') as mock_image_val, \
             patch.object(pipeline.orchestrator.config_validator, 'validate') as mock_config_val, \
             patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight, \
             patch.object(pipeline.orchestrator, 'prepare_generation') as mock_prepare, \
             patch('utils.generate_video') as mock_generate, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Mock all stages to succeed
            mock_prompt_val.return_value = ValidationResult(is_valid=True)
            mock_image_val.return_value = ValidationResult(is_valid=True)
            mock_config_val.return_value = ValidationResult(is_valid=True)
            
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            mock_preflight.return_value = PreflightResult(
                can_proceed=True,
                model_status=ModelStatus(is_available=True, is_loaded=False),
                resource_estimate=ResourceEstimate(5500, 2500, 35, 80),
                optimization_recommendations=[],
                blocking_issues=[],
                warnings=[]
            )
            
            mock_prepare.return_value = (True, "Success")
            mock_generate.return_value = {
                "success": True,
                "output_path": "/tmp/test_i2v_video.mp4"
            }
            
            mock_exists.return_value = True
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB file
            
            result = await pipeline.generate_video(i2v_request, "test_i2v")
            
            assert result.success
            assert result.output_path == "/tmp/test_i2v_video.mp4"
    
    @pytest.mark.asyncio
    async def test_complete_ti2v_workflow(self, pipeline, ti2v_request):
        """Test complete TI2V generation workflow"""
        with patch.object(pipeline.orchestrator.prompt_validator, 'validate') as mock_prompt_val, \
             patch.object(pipeline.orchestrator.image_validator, 'validate') as mock_image_val, \
             patch.object(pipeline.orchestrator.config_validator, 'validate') as mock_config_val, \
             patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight, \
             patch.object(pipeline.orchestrator, 'prepare_generation') as mock_prepare, \
             patch('utils.generate_video') as mock_generate, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Mock all stages to succeed
            mock_prompt_val.return_value = ValidationResult(is_valid=True)
            mock_image_val.return_value = ValidationResult(is_valid=True)
            mock_config_val.return_value = ValidationResult(is_valid=True)
            
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            mock_preflight.return_value = PreflightResult(
                can_proceed=True,
                model_status=ModelStatus(is_available=True, is_loaded=False),
                resource_estimate=ResourceEstimate(4000, 2000, 25, 75),
                optimization_recommendations=[],
                blocking_issues=[],
                warnings=[]
            )
            
            mock_prepare.return_value = (True, "Success")
            mock_generate.return_value = {
                "success": True,
                "output_path": "/tmp/test_ti2v_video.mp4"
            }
            
            mock_exists.return_value = True
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB file
            
            result = await pipeline.generate_video(ti2v_request, "test_ti2v")
            
            assert result.success
            assert result.output_path == "/tmp/test_ti2v_video.mp4"

class TestErrorScenarios:
    """Test various error scenarios and recovery"""
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, pipeline, t2v_request):
        """Test handling of model not found errors"""
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            mock_preflight.return_value = PreflightResult(
                can_proceed=False,
                model_status=ModelStatus(
                    is_available=False,
                    is_loaded=False,
                    loading_error="Model directory not found"
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
    async def test_insufficient_vram_error(self, pipeline, t2v_request):
        """Test handling of insufficient VRAM errors"""
        with patch.object(pipeline.orchestrator, 'run_preflight_checks') as mock_preflight:
            from generation_orchestrator import PreflightResult, ModelStatus, ResourceEstimate
            
            mock_preflight.return_value = PreflightResult(
                can_proceed=False,
                model_status=ModelStatus(is_available=True, is_loaded=False),
                resource_estimate=ResourceEstimate(16000, 4000, 60, 95),  # High VRAM requirement
                blocking_issues=["Insufficient VRAM for generation"],
                optimization_recommendations=[
                    "Reduce resolution to 720p",
                    "Decrease generation steps",
                    "Enable CPU offloading"
                ],
                warnings=[]
            )
            
            result = await pipeline.generate_video(t2v_request, "test_vram_error")
            
            assert not result.success
            assert result.error.category == ErrorCategory.SYSTEM_RESOURCE
            assert "Insufficient VRAM" in result.error.message
            assert len(result.error.recovery_suggestions) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])