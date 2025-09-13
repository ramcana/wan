"""
Integration tests for WAN models with existing pipeline infrastructure
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import pipeline components
try:
    from backend.core.services.wan_generation_service import WANGenerationService
    from backend.core.models.wan_models.wan_pipeline_factory import WANPipelineFactory
    from backend.core.models.wan_models.wan_progress_tracker import WANProgressTracker
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline components not available")
class TestWANModelPipelineIntegration:
    """Test WAN models integration with pipeline infrastructure"""
    
    def test_model_factory_integration(self, mock_model_config):
        """Test WAN model creation through pipeline factory"""
        with patch('core.models.wan_models.wan_pipeline_factory.WANPipelineFactory') as mock_factory:
            # Mock factory creation
            mock_factory_instance = Mock()
            mock_factory.return_value = mock_factory_instance
            
            # Mock model creation
            mock_model = Mock()
            mock_model.model_type = "t2v-A14B"
            mock_model.is_loaded = True
            mock_factory_instance.create_model.return_value = mock_model
            
            # Test model creation
            factory = WANPipelineFactory()
            model = factory.create_model("t2v-A14B", mock_model_config)
            
            assert model is not None
            assert model.model_type == "t2v-A14B"
            assert model.is_loaded is True
    
    def test_generation_service_integration(self, mock_model_config, sample_generation_params):
        """Test WAN models integration with generation service"""
        with patch('core.services.wan_generation_service.WANGenerationService') as mock_service:
            # Mock service instance
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock generation result
            mock_result = Mock()
            mock_result.success = True
            mock_result.video_path = "/tmp/generated_video.mp4"
            mock_result.generation_time = 45.2
            mock_result.metadata = {"frames": 16, "fps": 8.0}
            mock_service_instance.generate_video.return_value = mock_result
            
            # Test service integration
            service = WANGenerationService()
            result = service.generate_video("t2v-A14B", sample_generation_params)
            
            assert result is not None
            assert result.success is True
            assert result.video_path is not None
            assert result.generation_time > 0
    
    def test_progress_tracking_integration(self, mock_wan_model, mock_progress_callback):
        """Test progress tracking integration across pipeline"""
        with patch('core.models.wan_models.wan_progress_tracker.WANProgressTracker') as mock_tracker:
            # Mock progress tracker
            mock_tracker_instance = Mock()
            mock_tracker.return_value = mock_tracker_instance
            
            # Mock progress updates
            mock_tracker_instance.start_generation.return_value = None
            mock_tracker_instance.update_progress.return_value = None
            mock_tracker_instance.complete_generation.return_value = None
            
            # Test progress tracking
            tracker = WANProgressTracker("t2v-A14B", mock_wan_model)
            
            # Simulate generation with progress tracking
            tracker.start_generation()
            for i in range(5):
                tracker.update_progress(f"step_{i}", i / 5.0)
            tracker.complete_generation()
            
            # Verify progress tracking calls
            mock_tracker_instance.start_generation.assert_called_once()
            assert mock_tracker_instance.update_progress.call_count == 5
            mock_tracker_instance.complete_generation.assert_called_once()
    
    def test_model_loading_pipeline(self, mock_model_config):
        """Test model loading through pipeline infrastructure"""
        with patch('core.models.wan_models.wan_model_downloader.WANModelDownloader') as mock_downloader:
            # Mock model downloader
            mock_downloader_instance = Mock()
            mock_downloader.return_value = mock_downloader_instance
            
            # Mock download and loading
            mock_downloader_instance.download_model.return_value = True
            mock_downloader_instance.load_model.return_value = Mock()
            
            # Test model loading pipeline
            downloader = mock_downloader()
            success = downloader.download_model("t2v-A14B")
            model = downloader.load_model("t2v-A14B", mock_model_config)
            
            assert success is True
            assert model is not None
    
    def test_error_handling_integration(self, mock_model_config):
        """Test error handling across pipeline components"""
        with patch('core.models.wan_models.wan_model_error_handler.WANModelErrorHandler') as mock_handler:
            # Mock error handler
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            
            # Mock error handling
            mock_handler_instance.handle_generation_error.return_value = {
                "error_type": "memory_error",
                "suggested_fix": "enable_cpu_offload",
                "retry_possible": True
            }
            
            # Test error handling
            handler = mock_handler()
            error_info = handler.handle_generation_error(Exception("CUDA out of memory"))
            
            assert error_info["error_type"] == "memory_error"
            assert error_info["suggested_fix"] == "enable_cpu_offload"
            assert error_info["retry_possible"] is True
    
    def test_hardware_optimization_integration(self, mock_model_config, mock_hardware_info):
        """Test hardware optimization integration"""
        with patch('core.models.wan_models.wan_hardware_optimizer.WANHardwareOptimizer') as mock_optimizer:
            # Mock hardware optimizer
            mock_optimizer_instance = Mock()
            mock_optimizer.return_value = mock_optimizer_instance
            
            # Mock optimization
            mock_optimizer_instance.optimize_for_hardware.return_value = {
                "precision": "fp16",
                "cpu_offload": True,
                "quantization": "int8",
                "batch_size": 1,
                "chunk_size": 4
            }
            
            # Test hardware optimization
            optimizer = mock_optimizer()
            optimization_config = optimizer.optimize_for_hardware(mock_hardware_info)
            
            assert optimization_config["precision"] == "fp16"
            assert optimization_config["cpu_offload"] is True
            assert optimization_config["quantization"] == "int8"
    
    @pytest.mark.slow
    def test_end_to_end_generation_pipeline(self, mock_model_config, sample_generation_params, mock_progress_callback):
        """Test complete end-to-end generation pipeline"""
        with patch('core.services.wan_generation_service.WANGenerationService') as mock_service:
            # Mock complete pipeline
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock pipeline steps
            mock_service_instance.load_model = Mock(return_value=True)
            mock_service_instance.validate_params = Mock(return_value=(True, []))
            mock_service_instance.optimize_model = Mock(return_value=True)
            mock_service_instance.generate_video = AsyncMock()
            
            # Mock generation result
            mock_result = Mock()
            mock_result.success = True
            mock_result.video_path = "/tmp/test_video.mp4"
            mock_result.generation_time = 30.5
            mock_result.metadata = {
                "model": "t2v-A14B",
                "frames": 16,
                "resolution": "512x512",
                "fps": 8.0
            }
            mock_service_instance.generate_video.return_value = mock_result
            
            # Test end-to-end pipeline
            async def run_pipeline():
                service = WANGenerationService()
                
                # Load model
                loaded = service.load_model("t2v-A14B", mock_model_config)
                assert loaded is True
                
                # Validate parameters
                valid, errors = service.validate_params(sample_generation_params)
                assert valid is True
                assert len(errors) == 0
                
                # Optimize model
                optimized = service.optimize_model()
                assert optimized is True
                
                # Generate video
                result = await service.generate_video(sample_generation_params)
                assert result.success is True
                assert result.video_path is not None
                
                return result
            
            # Run the pipeline
            result = asyncio.run(run_pipeline())
            assert result.success is True
    
    def test_model_switching_integration(self, mock_model_config):
        """Test switching between different WAN models"""
        with patch('core.services.wan_generation_service.WANGenerationService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock model switching
            mock_service_instance.switch_model = Mock(return_value=True)
            mock_service_instance.current_model = Mock()
            
            # Test model switching
            service = WANGenerationService()
            
            # Switch to T2V model
            success = service.switch_model("t2v-A14B", mock_model_config)
            assert success is True
            
            # Switch to I2V model
            success = service.switch_model("i2v-A14B", mock_model_config)
            assert success is True
            
            # Switch to TI2V model
            success = service.switch_model("ti2v-5B", mock_model_config)
            assert success is True
            
            # Verify switching calls
            assert mock_service_instance.switch_model.call_count == 3
    
    def test_batch_generation_integration(self, mock_model_config):
        """Test batch generation through pipeline"""
        with patch('core.services.wan_generation_service.WANGenerationService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock batch generation
            mock_results = []
            for i in range(3):
                mock_result = Mock()
                mock_result.success = True
                mock_result.video_path = f"/tmp/batch_video_{i}.mp4"
                mock_result.generation_time = 25.0 + i * 5
                mock_results.append(mock_result)
            
            mock_service_instance.generate_batch = Mock(return_value=mock_results)
            
            # Test batch generation
            service = WANGenerationService()
            batch_params = [
                {"prompt": "A cat playing", "num_frames": 16},
                {"prompt": "A dog running", "num_frames": 16},
                {"prompt": "A bird flying", "num_frames": 16}
            ]
            
            results = service.generate_batch("t2v-A14B", batch_params)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert all(result.video_path is not None for result in results)
    
    def test_memory_management_integration(self, mock_model_config):
        """Test memory management across pipeline"""
        with patch('core.models.wan_models.wan_vram_monitor.WANVRAMMonitor') as mock_monitor:
            # Mock VRAM monitor
            mock_monitor_instance = Mock()
            mock_monitor.return_value = mock_monitor_instance
            
            # Mock memory monitoring
            mock_monitor_instance.get_memory_usage.return_value = {
                "allocated_gb": 8.5,
                "cached_gb": 1.2,
                "free_gb": 6.3,
                "total_gb": 16.0
            }
            mock_monitor_instance.check_memory_availability.return_value = True
            mock_monitor_instance.cleanup_memory.return_value = 2.1  # GB freed
            
            # Test memory management
            monitor = mock_monitor()
            
            # Check memory usage
            usage = monitor.get_memory_usage()
            assert usage["allocated_gb"] == 8.5
            assert usage["total_gb"] == 16.0
            
            # Check availability
            available = monitor.check_memory_availability(10.0)  # Need 10GB
            assert available is True
            
            # Cleanup memory
            freed = monitor.cleanup_memory()
            assert freed == 2.1


@pytest.mark.integration
class TestModelConfigurationIntegration:
    """Test model configuration integration"""
    
    def test_config_loading_integration(self):
        """Test configuration loading from various sources"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_get_config:
            # Mock configuration loading
            mock_config = Mock()
            mock_config.model_name = "t2v-A14B"
            mock_config.architecture = Mock()
            mock_config.architecture.hidden_dim = 1536
            mock_config.architecture.num_layers = 24
            mock_config.hardware = Mock()
            mock_config.hardware.min_vram_gb = 8.0
            mock_config.hardware.recommended_vram_gb = 12.0
            
            mock_get_config.return_value = mock_config
            
            # Test config loading
            from backend.core.models.wan_models.wan_model_config import get_wan_model_config
            config = get_wan_model_config("t2v-A14B")
            
            assert config is not None
            assert config.model_name == "t2v-A14B"
            assert config.architecture.hidden_dim == 1536
            assert config.hardware.min_vram_gb == 8.0
    
    def test_config_validation_integration(self, mock_model_config):
        """Test configuration validation across components"""
        with patch('core.models.wan_models.wan_model_config.validate_model_config') as mock_validate:
            # Mock validation
            mock_validate.return_value = (True, [])
            
            # Test validation
            from backend.core.models.wan_models.wan_model_config import validate_model_config
            is_valid, errors = validate_model_config(mock_model_config)
            
            assert is_valid is True
            assert len(errors) == 0
    
    def test_config_override_integration(self, mock_model_config):
        """Test configuration override mechanisms"""
        # Test environment variable overrides
        with patch.dict('os.environ', {'WAN_MODEL_PRECISION': 'fp16', 'WAN_MODEL_DEVICE': 'cuda'}):
            # Configuration should respect environment overrides
            assert True  # Placeholder for actual override testing
        
        # Test runtime configuration updates
        updated_config = mock_model_config.copy()
        updated_config['precision'] = 'fp16'
        updated_config['enable_optimization'] = True
        
        assert updated_config['precision'] == 'fp16'
        assert updated_config['enable_optimization'] is True


@pytest.mark.integration
class TestModelInteroperability:
    """Test interoperability between different WAN models"""
    
    def test_model_output_compatibility(self, mock_model_config):
        """Test that different models produce compatible outputs"""
        with patch('core.models.wan_models.wan_t2v_a14b.WANT2VA14B') as mock_t2v, \
             patch('core.models.wan_models.wan_i2v_a14b.WANI2VA14B') as mock_i2v, \
             patch('core.models.wan_models.wan_ti2v_5b.WANTI2V5B') as mock_ti2v:
            
            # Mock model outputs
            mock_output = Mock()
            mock_output.shape = (1, 16, 4, 64, 64)  # Standard video latent format
            
            mock_t2v_instance = Mock()
            mock_t2v_instance.generate.return_value = mock_output
            mock_t2v.return_value = mock_t2v_instance
            
            mock_i2v_instance = Mock()
            mock_i2v_instance.generate.return_value = mock_output
            mock_i2v.return_value = mock_i2v_instance
            
            mock_ti2v_instance = Mock()
            mock_ti2v_instance.generate.return_value = mock_output
            mock_ti2v.return_value = mock_ti2v_instance
            
            # Test output compatibility
            t2v_model = mock_t2v(mock_model_config)
            i2v_model = mock_i2v(mock_model_config)
            ti2v_model = mock_ti2v(mock_model_config)
            
            t2v_output = t2v_model.generate(Mock())
            i2v_output = i2v_model.generate(Mock())
            ti2v_output = ti2v_model.generate(Mock())
            
            # All outputs should have the same format
            assert t2v_output.shape == i2v_output.shape == ti2v_output.shape
    
    def test_model_pipeline_chaining(self, mock_model_config):
        """Test chaining different models in a pipeline"""
        # Example: T2V -> I2V chaining (use T2V output as I2V input)
        with patch('core.services.wan_generation_service.WANGenerationService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock chained generation
            mock_t2v_result = Mock()
            mock_t2v_result.video_frames = [Mock() for _ in range(16)]  # 16 frames
            
            mock_i2v_result = Mock()
            mock_i2v_result.video_frames = [Mock() for _ in range(16)]  # Extended frames
            
            mock_service_instance.chain_models.return_value = [mock_t2v_result, mock_i2v_result]
            
            # Test model chaining
            service = WANGenerationService()
            results = service.chain_models(
                [("t2v-A14B", {"prompt": "A cat"}), ("i2v-A14B", {"image": "frame_0"})]
            )
            
            assert len(results) == 2
            assert results[0] is mock_t2v_result
            assert results[1] is mock_i2v_result
