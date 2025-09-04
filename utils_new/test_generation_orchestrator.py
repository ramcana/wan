"""
Unit tests for Generation Orchestrator Components
Tests PreflightChecker, ResourceManager, PipelineRouter, and GenerationRequest
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import psutil
from pathlib import Path
import tempfile
import shutil

from generation_orchestrator import (
    GenerationRequest, PreflightChecker, ResourceManager, PipelineRouter,
    GenerationOrchestrator, GenerationMode, ResourceStatus, ModelStatus,
    PreflightResult, ResourceEstimate, GenerationPipeline,
    create_generation_request
)
from input_validation import ValidationResult


class TestGenerationRequest(unittest.TestCase):
    """Test GenerationRequest data model"""
    
    def test_generation_request_creation(self):
        """Test basic GenerationRequest creation"""
        request = GenerationRequest(
            model_type="t2v",
            prompt="Test prompt",
            resolution="720p",
            steps=50
        )
        
        self.assertEqual(request.model_type, GenerationMode.TEXT_TO_VIDEO.value)
        self.assertEqual(request.prompt, "Test prompt")
        self.assertEqual(request.resolution, "720p")
        self.assertEqual(request.steps, 50)

        assert True  # TODO: Add proper assertion
    
    def test_model_type_normalization(self):
        """Test model type normalization in __post_init__"""
        # Test t2v normalization
        request = GenerationRequest(model_type="t2v", prompt="test")
        self.assertEqual(request.model_type, GenerationMode.TEXT_TO_VIDEO.value)
        
        # Test i2v normalization
        request = GenerationRequest(model_type="image-to-video", prompt="test")
        self.assertEqual(request.model_type, GenerationMode.IMAGE_TO_VIDEO.value)
        
        # Test ti2v normalization
        request = GenerationRequest(model_type="ti2v", prompt="test")
        self.assertEqual(request.model_type, GenerationMode.TEXT_IMAGE_TO_VIDEO.value)

        assert True  # TODO: Add proper assertion
    
    def test_parameter_validation(self):
        """Test parameter validation and normalization"""
        request = GenerationRequest(
            model_type="t2v",
            prompt="test",
            resolution="invalid",  # Should default to 720p
            steps=-5,  # Should be clamped to 1
            guidance_scale=-1,  # Should be clamped to 0.1
            strength=2.0,  # Should be clamped to 1.0
            fps=0,  # Should be clamped to 1
            duration=-1  # Should be clamped to 1
        )
        
        self.assertEqual(request.resolution, "720p")
        self.assertEqual(request.steps, 1)
        self.assertEqual(request.guidance_scale, 0.1)
        self.assertEqual(request.strength, 1.0)
        self.assertEqual(request.fps, 1)
        self.assertEqual(request.duration, 1)

        assert True  # TODO: Add proper assertion
    
    def test_factory_function(self):
        """Test create_generation_request factory function"""
        request = create_generation_request(
            model_type="t2v",
            prompt="Factory test",
            resolution="1080p"
        )
        
        self.assertEqual(request.model_type, GenerationMode.TEXT_TO_VIDEO.value)
        self.assertEqual(request.prompt, "Factory test")
        self.assertEqual(request.resolution, "1080p")


        assert True  # TODO: Add proper assertion

class TestPreflightChecker(unittest.TestCase):
    """Test PreflightChecker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {"model_path": "./models", "output_path": "./outputs"}
        self.checker = PreflightChecker(self.config)
        
        # Mock validators
        self.checker.prompt_validator = Mock()
        self.checker.image_validator = Mock()
        self.checker.config_validator = Mock()
    
    def test_initialization(self):
        """Test PreflightChecker initialization"""
        self.assertIsNotNone(self.checker.prompt_validator)
        self.assertIsNotNone(self.checker.image_validator)
        self.assertIsNotNone(self.checker.config_validator)

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.Path')
    def test_check_model_availability_success(self, mock_path):
        """Test successful model availability check"""
        # Mock path existence and required files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Mock required files exist
        def mock_file_exists(file_path):
            return True
        
        mock_path_instance.__truediv__ = Mock(side_effect=lambda x: Mock(exists=Mock(return_value=True)))
        
        result = self.checker.check_model_availability(GenerationMode.TEXT_TO_VIDEO.value)
        
        self.assertTrue(result.is_available)
        self.assertFalse(result.is_loaded)
        self.assertIsNone(result.loading_error)

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.Path')
    def test_check_model_availability_missing_directory(self, mock_path):
        """Test model availability check with missing directory"""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        result = self.checker.check_model_availability(GenerationMode.TEXT_TO_VIDEO.value)
        
        self.assertFalse(result.is_available)
        self.assertFalse(result.is_loaded)
        self.assertIn("Model directory not found", result.loading_error)

        assert True  # TODO: Add proper assertion
    
    def test_check_model_availability_unknown_type(self):
        """Test model availability check with unknown model type"""
        result = self.checker.check_model_availability("unknown_model")
        
        self.assertFalse(result.is_available)
        self.assertFalse(result.is_loaded)
        self.assertIn("Unknown model type", result.loading_error)

        assert True  # TODO: Add proper assertion
    
    def test_estimate_resource_requirements(self):
        """Test resource requirement estimation"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_TO_VIDEO.value,
            prompt="test",
            resolution="720p",
            steps=50,
            duration=4
        )
        
        estimate = self.checker.estimate_resource_requirements(request)
        
        self.assertIsInstance(estimate, ResourceEstimate)
        self.assertGreater(estimate.vram_mb, 0)
        self.assertGreater(estimate.system_ram_mb, 0)
        self.assertGreater(estimate.estimated_time_seconds, 0)
        self.assertGreater(estimate.gpu_utilization_percent, 0)

        assert True  # TODO: Add proper assertion
    
    def test_estimate_resource_requirements_with_lora(self):
        """Test resource estimation with LoRA configuration"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_TO_VIDEO.value,
            prompt="test",
            resolution="720p",
            steps=50,
            lora_config={"lora1": 0.8, "lora2": 0.6}
        )
        
        estimate_with_lora = self.checker.estimate_resource_requirements(request)
        
        # Create same request without LoRA
        request_no_lora = GenerationRequest(
            model_type=GenerationMode.TEXT_TO_VIDEO.value,
            prompt="test",
            resolution="720p",
            steps=50
        )
        
        estimate_no_lora = self.checker.estimate_resource_requirements(request_no_lora)
        
        # LoRA should increase resource requirements
        self.assertGreater(estimate_with_lora.vram_mb, estimate_no_lora.vram_mb)
        self.assertGreater(estimate_with_lora.estimated_time_seconds, estimate_no_lora.estimated_time_seconds)

        assert True  # TODO: Add proper assertion
    
    def test_validate_inputs(self):
        """Test input validation"""
        # Mock successful validation
        self.checker.prompt_validator.validate.return_value = ValidationResult(is_valid=True)
        self.checker.config_validator.validate.return_value = ValidationResult(is_valid=True)
        
        request = GenerationRequest(model_type="t2v", prompt="test")
        result = self.checker._validate_inputs(request)
        
        self.assertTrue(result.is_valid)
        self.checker.prompt_validator.validate.assert_called_once()
        self.checker.config_validator.validate.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    def test_validate_inputs_with_image(self):
        """Test input validation with image"""
        # Mock successful validation
        self.checker.prompt_validator.validate.return_value = ValidationResult(is_valid=True)
        self.checker.image_validator.validate.return_value = ValidationResult(is_valid=True)
        self.checker.config_validator.validate.return_value = ValidationResult(is_valid=True)
        
        mock_image = Mock()
        request = GenerationRequest(model_type="i2v", prompt="test", image=mock_image)
        result = self.checker._validate_inputs(request)
        
        self.assertTrue(result.is_valid)
        self.checker.image_validator.validate.assert_called_once_with(mock_image)

        assert True  # TODO: Add proper assertion
    
    def test_get_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        request = GenerationRequest(model_type="t2v", prompt="test", resolution="1080p")
        estimate = ResourceEstimate(vram_mb=15000, system_ram_mb=4000, estimated_time_seconds=300, gpu_utilization_percent=90)
        
        recommendations = self.checker._get_optimization_recommendations(request, estimate)
        
        self.assertIsInstance(recommendations, list)
        # Should have recommendations for high VRAM and long time
        self.assertTrue(any("VRAM" in rec for rec in recommendations))
        self.assertTrue(any("time" in rec or "steps" in rec for rec in recommendations))

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.ResourceManager')
    @patch('generation_orchestrator.Path')
    def test_run_preflight_checks_success(self, mock_path, mock_resource_manager):
        """Test successful preflight checks"""
        # Mock model availability
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = Mock(side_effect=lambda x: Mock(exists=Mock(return_value=True)))
        
        # Mock resource manager
        mock_rm_instance = Mock()
        mock_rm_instance.check_vram_availability.return_value = True
        mock_resource_manager.return_value = mock_rm_instance
        
        # Mock validation
        self.checker.prompt_validator.validate.return_value = ValidationResult(is_valid=True)
        self.checker.config_validator.validate.return_value = ValidationResult(is_valid=True)
        
        request = GenerationRequest(model_type=GenerationMode.TEXT_TO_VIDEO.value, prompt="test")
        result = self.checker.run_preflight_checks(request)
        
        self.assertIsInstance(result, PreflightResult)
        self.assertTrue(result.can_proceed)
        self.assertEqual(len(result.blocking_issues), 0)


        assert True  # TODO: Add proper assertion

class TestResourceManager(unittest.TestCase):
    """Test ResourceManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {"model_path": "./models"}
        
    @patch('generation_orchestrator.torch')
    @patch('generation_orchestrator.psutil')
    def _create_resource_manager(self, mock_psutil, mock_torch):
        """Helper to create ResourceManager with mocked dependencies"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = Mock(total_memory=8 * 1024**3)
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_psutil.virtual_memory.return_value = Mock(total=16 * 1024**3)
        return ResourceManager(self.config)
        
    @patch('generation_orchestrator.torch')
    @patch('generation_orchestrator.psutil')
    def test_initialization_with_gpu(self, mock_psutil, mock_torch):
        """Test ResourceManager initialization with GPU available"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = Mock(total_memory=8 * 1024**3)  # 8GB
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_psutil.virtual_memory.return_value = Mock(total=16 * 1024**3)  # 16GB RAM
        
        manager = ResourceManager(self.config)
        
        self.assertTrue(manager.gpu_available)
        self.assertEqual(manager.gpu_count, 1)
        self.assertEqual(manager.total_vram, 8 * 1024**3)
        self.assertEqual(manager.gpu_name, "Test GPU")

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.torch')
    @patch('generation_orchestrator.psutil')
    def test_initialization_without_gpu(self, mock_psutil, mock_torch):
        """Test ResourceManager initialization without GPU"""
        mock_torch.cuda.is_available.return_value = False
        mock_psutil.virtual_memory.return_value = Mock(total=16 * 1024**3)
        
        manager = ResourceManager(self.config)
        
        self.assertFalse(manager.gpu_available)
        self.assertEqual(manager.gpu_count, 0)
        self.assertEqual(manager.total_vram, 0)

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.torch')
    def test_check_vram_availability_sufficient(self, mock_torch):
        """Test VRAM availability check with sufficient memory"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB allocated
        mock_torch.cuda.memory_reserved.return_value = 2.5 * 1024**3  # 2.5GB reserved
        
        manager = ResourceManager(self.config)
        manager.gpu_available = True
        manager.total_vram = 8 * 1024**3  # 8GB total
        
        # Request 4GB (4096MB)
        result = manager.check_vram_availability(4096)
        
        self.assertTrue(result)

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.torch')
    def test_check_vram_availability_insufficient(self, mock_torch):
        """Test VRAM availability check with insufficient memory"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 6 * 1024**3  # 6GB allocated
        mock_torch.cuda.memory_reserved.return_value = 6 * 1024**3  # 6GB reserved
        
        manager = ResourceManager(self.config)
        manager.gpu_available = True
        manager.total_vram = 8 * 1024**3  # 8GB total
        
        # Request 4GB (4096MB) - should fail due to insufficient available memory
        result = manager.check_vram_availability(4096)
        
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion
    
    def test_check_vram_availability_no_gpu(self):
        """Test VRAM availability check without GPU"""
        manager = ResourceManager(self.config)
        manager.gpu_available = False
        
        result = manager.check_vram_availability(1000)
        
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.torch')
    def test_optimize_for_available_resources(self, mock_torch):
        """Test resource optimization"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB allocated
        
        manager = ResourceManager(self.config)
        manager.gpu_available = True
        manager.total_vram = 6 * 1024**3  # 6GB total (4GB available)
        
        # Create request that should be optimized
        request = GenerationRequest(
            model_type="t2v",
            prompt="test",
            resolution="1080p",
            steps=80
        )
        
        optimized = manager.optimize_for_available_resources(request)
        
        # Should optimize for low VRAM
        self.assertEqual(optimized.resolution, "720p")  # Downgraded from 1080p
        self.assertLessEqual(optimized.steps, 30)  # Reduced steps
        self.assertTrue(optimized.optimization_settings.get("enable_memory_efficient_attention", False))

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.torch')
    @patch('generation_orchestrator.gc')
    def test_prepare_generation_environment(self, mock_gc, mock_torch):
        """Test generation environment preparation"""
        mock_torch.cuda.is_available.return_value = True
        
        manager = ResourceManager(self.config)
        manager.gpu_available = True
        
        manager.prepare_generation_environment("t2v")
        
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_gc.collect.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.torch')
    @patch('generation_orchestrator.psutil')
    def test_get_resource_status(self, mock_psutil, mock_torch):
        """Test resource status reporting"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3
        mock_torch.cuda.memory_reserved.return_value = 2.5 * 1024**3
        mock_psutil.virtual_memory.return_value = Mock(
            total=16 * 1024**3,
            available=12 * 1024**3,
            percent=25.0
        )
        
        manager = ResourceManager(self.config)
        manager.gpu_available = True
        manager.total_vram = 8 * 1024**3
        manager.gpu_name = "Test GPU"
        
        status = manager.get_resource_status()
        
        self.assertTrue(status["gpu_available"])
        self.assertEqual(status["gpu_name"], "Test GPU")
        self.assertIn("total_vram_gb", status)
        self.assertIn("allocated_vram_gb", status)
        self.assertIn("available_vram_gb", status)
        self.assertIn("total_ram_gb", status)
        self.assertIn("available_ram_gb", status)
        self.assertEqual(status["ram_usage_percent"], 25.0)


        assert True  # TODO: Add proper assertion

class TestPipelineRouter(unittest.TestCase):
    """Test PipelineRouter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {"model_path": "./models"}
        self.router = PipelineRouter(self.config)
    
    def test_initialization(self):
        """Test PipelineRouter initialization"""
        self.assertIn(GenerationMode.TEXT_TO_VIDEO.value, self.router.pipelines)
        self.assertIn(GenerationMode.IMAGE_TO_VIDEO.value, self.router.pipelines)
        self.assertIn(GenerationMode.TEXT_IMAGE_TO_VIDEO.value, self.router.pipelines)
    
    def test_route_generation_request_success(self):
        """Test successful generation request routing"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_TO_VIDEO.value,
            prompt="test",
            resolution="720p",
            steps=50,
            guidance_scale=7.5
        )
        
        pipeline = self.router.route_generation_request(request)
        
        self.assertIsInstance(pipeline, GenerationPipeline)
        self.assertEqual(pipeline.pipeline_type, "text_to_video")
        self.assertIn("720p", pipeline.supported_resolutions)

        assert True  # TODO: Add proper assertion
    
    def test_route_generation_request_unknown_model(self):
        """Test routing with unknown model type"""
        request = GenerationRequest(
            model_type="unknown_model",
            prompt="test"
        )
        
        # Should return default pipeline (T2V)
        pipeline = self.router.route_generation_request(request)
        self.assertEqual(pipeline.pipeline_type, "text_to_video")

        assert True  # TODO: Add proper assertion
    
    def test_validate_request_compatibility_success(self):
        """Test successful request compatibility validation"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_TO_VIDEO.value,
            prompt="test",
            resolution="720p",
            steps=50,
            guidance_scale=7.5
        )
        
        pipeline = self.router.pipelines[GenerationMode.TEXT_TO_VIDEO.value]
        
        # Should not raise exception
        self.router._validate_request_compatibility(request, pipeline)

        assert True  # TODO: Add proper assertion
    
    def test_validate_request_compatibility_invalid_resolution(self):
        """Test compatibility validation with invalid resolution"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_IMAGE_TO_VIDEO.value,
            prompt="test",
            resolution="1080p"  # TI2V doesn't support 1080p
        )
        
        pipeline = self.router.pipelines[GenerationMode.TEXT_IMAGE_TO_VIDEO.value]
        
        with self.assertRaises(ValueError) as context:
            self.router._validate_request_compatibility(request, pipeline)
        
        self.assertIn("Resolution", str(context.exception))

        assert True  # TODO: Add proper assertion
    
    def test_validate_request_compatibility_invalid_guidance_scale(self):
        """Test compatibility validation with invalid guidance scale"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_TO_VIDEO.value,
            prompt="test",
            guidance_scale=25.0  # Outside valid range
        )
        
        pipeline = self.router.pipelines[GenerationMode.TEXT_TO_VIDEO.value]
        
        with self.assertRaises(ValueError) as context:
            self.router._validate_request_compatibility(request, pipeline)
        
        self.assertIn("Guidance scale", str(context.exception))

        assert True  # TODO: Add proper assertion
    
    def test_validate_request_compatibility_lora_not_supported(self):
        """Test compatibility validation with LoRA on unsupported pipeline"""
        request = GenerationRequest(
            model_type=GenerationMode.TEXT_IMAGE_TO_VIDEO.value,
            prompt="test",
            lora_config={"lora1": 0.8}  # TI2V doesn't support LoRA
        )
        
        pipeline = self.router.pipelines[GenerationMode.TEXT_IMAGE_TO_VIDEO.value]
        
        with self.assertRaises(ValueError) as context:
            self.router._validate_request_compatibility(request, pipeline)
        
        self.assertIn("LoRA not supported", str(context.exception))

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.ResourceManager')
    @patch('generation_orchestrator.torch')
    def test_select_optimal_pipeline_memory_optimized(self, mock_torch, mock_resource_manager):
        """Test optimal pipeline selection for low memory"""
        mock_rm_instance = Mock()
        mock_rm_instance.gpu_available = True
        mock_rm_instance.total_vram = 6 * 1024**3  # 6GB
        mock_resource_manager.return_value = mock_rm_instance
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB allocated
        
        request = GenerationRequest(model_type=GenerationMode.TEXT_TO_VIDEO.value, prompt="test")
        
        config = self.router.select_optimal_pipeline(GenerationMode.TEXT_TO_VIDEO.value, request)
        
        self.assertEqual(config, "memory_optimized")

        assert True  # TODO: Add proper assertion
    
    def test_get_pipeline_info(self):
        """Test pipeline information retrieval"""
        info = self.router.get_pipeline_info(GenerationMode.TEXT_TO_VIDEO.value)
        
        self.assertIsNotNone(info)
        self.assertEqual(info["pipeline_type"], "text_to_video")
        self.assertIn("supported_resolutions", info)
        self.assertIn("memory_requirements", info)
        self.assertIn("optimization_flags", info)

        assert True  # TODO: Add proper assertion
    
    def test_get_pipeline_info_unknown_model(self):
        """Test pipeline info for unknown model"""
        info = self.router.get_pipeline_info("unknown_model")
        
        self.assertIsNone(info)

        assert True  # TODO: Add proper assertion
    
    def test_list_available_pipelines(self):
        """Test listing all available pipelines"""
        pipelines = self.router.list_available_pipelines()
        
        self.assertIsInstance(pipelines, dict)
        self.assertIn(GenerationMode.TEXT_TO_VIDEO.value, pipelines)
        self.assertIn(GenerationMode.IMAGE_TO_VIDEO.value, pipelines)
        self.assertIn(GenerationMode.TEXT_IMAGE_TO_VIDEO.value, pipelines)


        assert True  # TODO: Add proper assertion

class TestGenerationOrchestrator(unittest.TestCase):
    """Test GenerationOrchestrator integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {"model_path": "./models", "output_path": "./outputs"}
        
    @patch('generation_orchestrator.PreflightChecker')
    @patch('generation_orchestrator.ResourceManager')
    @patch('generation_orchestrator.PipelineRouter')
    def test_initialization(self, mock_pipeline_router, mock_resource_manager, mock_preflight_checker):
        """Test GenerationOrchestrator initialization"""
        orchestrator = GenerationOrchestrator(self.config)
        
        self.assertIsNotNone(orchestrator.preflight_checker)
        self.assertIsNotNone(orchestrator.resource_manager)
        self.assertIsNotNone(orchestrator.pipeline_router)
        
        # Test legacy compatibility
        self.assertIsNotNone(orchestrator.prompt_validator)
        self.assertIsNotNone(orchestrator.image_validator)
        self.assertIsNotNone(orchestrator.config_validator)
    
    @patch('generation_orchestrator.PreflightChecker')
    @patch('generation_orchestrator.ResourceManager')
    @patch('generation_orchestrator.PipelineRouter')
    def test_prepare_generation_success(self, mock_pipeline_router, mock_resource_manager, mock_preflight_checker):
        """Test successful generation preparation"""
        # Mock preflight checker
        mock_preflight_instance = Mock()
        mock_preflight_result = PreflightResult(
            can_proceed=True,
            model_status=ModelStatus(is_available=True, is_loaded=False),
            resource_estimate=ResourceEstimate(6000, 3000, 45, 85),
            optimization_recommendations=["Test recommendation"],
            blocking_issues=[],
            warnings=[]
        )
        mock_preflight_instance.run_preflight_checks.return_value = mock_preflight_result
        mock_preflight_checker.return_value = mock_preflight_instance
        
        # Mock resource manager
        mock_resource_instance = Mock()
        mock_resource_instance.optimize_for_available_resources.return_value = GenerationRequest(
            model_type="t2v", prompt="test"
        )
        mock_resource_manager.return_value = mock_resource_instance
        
        # Mock pipeline router
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.route_generation_request.return_value = GenerationPipeline(
            pipeline_type="text_to_video",
            model_path="test_path",
            config={},
            memory_requirements=ResourceEstimate(6000, 3000, 45, 85),
            supported_resolutions=["720p"]
        )
        mock_pipeline_router.return_value = mock_pipeline_instance
        
        orchestrator = GenerationOrchestrator(self.config)
        request = GenerationRequest(model_type="t2v", prompt="test")
        
        success, message = orchestrator.prepare_generation(request)
        
        self.assertTrue(success)
        self.assertIn("successful", message)
        mock_preflight_instance.run_preflight_checks.assert_called_once()
        mock_resource_instance.optimize_for_available_resources.assert_called_once()
        mock_resource_instance.prepare_generation_environment.assert_called_once()
        mock_pipeline_instance.route_generation_request.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    @patch('generation_orchestrator.PreflightChecker')
    @patch('generation_orchestrator.ResourceManager')
    @patch('generation_orchestrator.PipelineRouter')
    def test_prepare_generation_preflight_failure(self, mock_pipeline_router, mock_resource_manager, mock_preflight_checker):
        """Test generation preparation with preflight failure"""
        # Mock preflight checker with failure
        mock_preflight_instance = Mock()
        mock_preflight_result = PreflightResult(
            can_proceed=False,
            model_status=ModelStatus(is_available=False, is_loaded=False, loading_error="Model not found"),
            resource_estimate=ResourceEstimate(6000, 3000, 45, 85),
            optimization_recommendations=[],
            blocking_issues=["Model not found", "Insufficient VRAM"],
            warnings=[]
        )
        mock_preflight_instance.run_preflight_checks.return_value = mock_preflight_result
        mock_preflight_checker.return_value = mock_preflight_instance
        
        orchestrator = GenerationOrchestrator(self.config)
        request = GenerationRequest(model_type="t2v", prompt="test")
        
        success, message = orchestrator.prepare_generation(request)
        
        self.assertFalse(success)
        self.assertIn("Preflight checks failed", message)
        self.assertIn("Model not found", message)
        self.assertIn("Insufficient VRAM", message)


        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)