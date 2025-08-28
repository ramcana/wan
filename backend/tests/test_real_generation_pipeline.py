"""
Tests for Real Generation Pipeline
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import torch

from backend.services.real_generation_pipeline import (
    RealGenerationPipeline, GenerationStage, ProgressUpdate,
    get_real_generation_pipeline
)
from backend.core.model_integration_bridge import GenerationParams, GenerationResult

class TestRealGenerationPipeline:
    """Test suite for RealGenerationPipeline"""
    
    @pytest.fixture
    def mock_wan_pipeline_loader(self):
        """Mock WanPipelineLoader"""
        mock_loader = Mock()
        
        # Mock pipeline wrapper
        mock_wrapper = Mock()
        mock_wrapper.generate.return_value = Mock(
            success=True,
            frames=[torch.randn(3, 256, 256) for _ in range(16)],
            peak_memory_mb=1024,
            memory_used_mb=512,
            applied_optimizations=["fp16", "cpu_offload"]
        )
        
        mock_loader.load_wan_pipeline.return_value = mock_wrapper
        return mock_loader
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Mock WebSocket manager"""
        mock_manager = AsyncMock()
        return mock_manager
    
    @pytest.fixture
    def sample_params(self):
        """Sample generation parameters"""
        return GenerationParams(
            prompt="A beautiful sunset over mountains",
            model_type="t2v-A14B",
            resolution="1280x720",
            steps=20,
            num_frames=16,
            fps=8.0,
            guidance_scale=7.5
        )
    
    @pytest.fixture
    def pipeline(self, mock_wan_pipeline_loader, mock_websocket_manager):
        """Create pipeline instance with mocks"""
        return RealGenerationPipeline(
            wan_pipeline_loader=mock_wan_pipeline_loader,
            websocket_manager=mock_websocket_manager
        )
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = RealGenerationPipeline()
        
        with patch('backend.core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_integration.get_wan_pipeline_loader.return_value = Mock()
            mock_get_integration.return_value = mock_integration
            
            with patch('backend.websocket.manager.get_connection_manager') as mock_get_manager:
                mock_get_manager.return_value = Mock()
                
                result = await pipeline.initialize()
                assert result is True
                assert pipeline.wan_pipeline_loader is not None
                assert pipeline.websocket_manager is not None
    
    @pytest.mark.asyncio
    async def test_t2v_generation_success(self, pipeline, sample_params):
        """Test successful T2V generation"""
        # Mock the _load_pipeline method
        with patch.object(pipeline, '_load_pipeline') as mock_load:
            mock_wrapper = Mock()
            mock_wrapper.generate.return_value = Mock(
                success=True,
                frames=[torch.randn(3, 256, 256) for _ in range(16)],
                peak_memory_mb=1024,
                memory_used_mb=512,
                applied_optimizations=["fp16"],
                errors=[]
            )
            mock_load.return_value = mock_wrapper
            
            # Mock save method
            with patch.object(pipeline, '_save_generated_video') as mock_save:
                mock_save.return_value = "outputs/test_video.mp4"
                
                result = await pipeline.generate_t2v(
                    "A beautiful sunset", sample_params
                )
                
                assert result.success is True
                assert result.model_used == "t2v-A14B"
                assert result.output_path == "outputs/test_video.mp4"
                assert result.generation_time_seconds > 0
                assert "fp16" in result.optimizations_applied
    
    @pytest.mark.asyncio
    async def test_t2v_generation_validation_failure(self, pipeline, sample_params):
        """Test T2V generation with validation failure"""
        # Test with empty prompt
        result = await pipeline.generate_t2v("", sample_params)
        
        assert result.success is False
        assert result.error_category == "parameter_validation"
        assert "Prompt cannot be empty" in result.error_message
    
    @pytest.mark.asyncio
    async def test_i2v_generation_success(self, pipeline, sample_params):
        """Test successful I2V generation"""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Mock image loading
            with patch.object(pipeline, '_load_and_validate_image') as mock_load_image:
                from PIL import Image
                mock_image = Image.new('RGB', (512, 512), color='red')
                mock_load_image.return_value = mock_image
                
                # Mock pipeline loading
                with patch.object(pipeline, '_load_pipeline') as mock_load:
                    mock_wrapper = Mock()
                    mock_wrapper.generate.return_value = Mock(
                        success=True,
                        frames=[torch.randn(3, 256, 256) for _ in range(16)],
                        peak_memory_mb=1024,
                        memory_used_mb=512,
                        applied_optimizations=["fp16"],
                        errors=[]
                    )
                    mock_load.return_value = mock_wrapper
                    
                    # Mock save method
                    with patch.object(pipeline, '_save_generated_video') as mock_save:
                        mock_save.return_value = "outputs/test_i2v.mp4"
                        
                        result = await pipeline.generate_i2v(
                            tmp_path, "Animate this image", sample_params
                        )
                        
                        assert result.success is True
                        assert result.model_used == "i2v-A14B"
                        assert result.output_path == "outputs/test_i2v.mp4"
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_i2v_generation_missing_image(self, pipeline, sample_params):
        """Test I2V generation with missing image"""
        result = await pipeline.generate_i2v(
            "/nonexistent/image.jpg", "Animate this", sample_params
        )
        
        assert result.success is False
        assert result.error_category == "parameter_validation"
        assert "must exist" in result.error_message
    
    @pytest.mark.asyncio
    async def test_ti2v_generation_success(self, pipeline, sample_params):
        """Test successful TI2V generation"""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Mock image loading
            with patch.object(pipeline, '_load_and_validate_image') as mock_load_image:
                from PIL import Image
                mock_image = Image.new('RGB', (512, 512), color='blue')
                mock_load_image.return_value = mock_image
                
                # Mock pipeline loading
                with patch.object(pipeline, '_load_pipeline') as mock_load:
                    mock_wrapper = Mock()
                    mock_wrapper.generate.return_value = Mock(
                        success=True,
                        frames=[torch.randn(3, 256, 256) for _ in range(16)],
                        peak_memory_mb=1024,
                        memory_used_mb=512,
                        applied_optimizations=["bf16"],
                        errors=[]
                    )
                    mock_load.return_value = mock_wrapper
                    
                    # Mock save method
                    with patch.object(pipeline, '_save_generated_video') as mock_save:
                        mock_save.return_value = "outputs/test_ti2v.mp4"
                        
                        result = await pipeline.generate_ti2v(
                            tmp_path, "Transform this image with text", sample_params
                        )
                        
                        assert result.success is True
                        assert result.model_used == "ti2v-5B"
                        assert result.output_path == "outputs/test_ti2v.mp4"
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_ti2v_generation_empty_prompt(self, pipeline, sample_params):
        """Test TI2V generation with empty prompt"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = await pipeline.generate_ti2v(tmp_path, "", sample_params)
            
            assert result.success is False
            assert result.error_category == "parameter_validation"
            assert "Text prompt is required" in result.error_message
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_parse_resolution(self, pipeline):
        """Test resolution parsing"""
        assert pipeline._parse_resolution("1280x720") == (1280, 720)
        assert pipeline._parse_resolution("1920x1080") == (1920, 1080)
        assert pipeline._parse_resolution("720p") == (1280, 720)
        assert pipeline._parse_resolution("1080p") == (1920, 1080)
        assert pipeline._parse_resolution("invalid") == (1280, 720)  # Default
    
    def test_get_model_path(self, pipeline):
        """Test model path resolution"""
        path = pipeline._get_model_path("t2v-A14B")
        assert path is not None
        assert "Wan2.2-T2V-A14B-Diffusers" in path
        
        path = pipeline._get_model_path("i2v-A14B")
        assert path is not None
        assert "Wan2.2-I2V-A14B-Diffusers" in path
        
        path = pipeline._get_model_path("ti2v-5B")
        assert path is not None
        assert "Wan2.2-TI2V-5B-Diffusers" in path
        
        path = pipeline._get_model_path("nonexistent")
        assert path is None
    
    def test_create_optimization_config(self, pipeline, sample_params):
        """Test optimization config creation"""
        sample_params.quantization_level = "fp16"
        sample_params.max_vram_usage_gb = 8.0
        sample_params.enable_offload = True
        
        config = pipeline._create_optimization_config(sample_params)
        
        assert config["precision"] == "fp16"
        assert config["min_vram_mb"] == 8192
        assert config["enable_cpu_offload"] is True
        assert "chunk_size" in config
    
    @pytest.mark.asyncio
    async def test_progress_updates(self, pipeline, mock_websocket_manager):
        """Test progress update functionality"""
        task_id = "test_task"
        
        await pipeline._send_progress_update(
            task_id, GenerationStage.LOADING_MODEL, 25, "Loading model"
        )
        
        mock_websocket_manager.send_generation_progress.assert_called_once_with(
            task_id=task_id,
            progress=25,
            status="loading_model",
            message="Loading model"
        )
    
    @pytest.mark.asyncio
    async def test_progress_callback_integration(self, pipeline):
        """Test custom progress callback integration"""
        callback_calls = []
        
        def progress_callback(update: ProgressUpdate):
            callback_calls.append(update)
        
        pipeline._progress_callback = progress_callback
        
        await pipeline._send_progress_update(
            "test", GenerationStage.GENERATING, 50, "Generating frames"
        )
        
        assert len(callback_calls) == 1
        assert callback_calls[0].stage == GenerationStage.GENERATING
        assert callback_calls[0].progress_percent == 50
    
    @pytest.mark.asyncio
    async def test_pipeline_caching(self, pipeline, sample_params):
        """Test pipeline caching functionality"""
        with patch.object(pipeline, 'wan_pipeline_loader') as mock_loader:
            mock_wrapper = Mock()
            mock_loader.load_wan_pipeline.return_value = mock_wrapper
            
            # First call should load pipeline
            result1 = await pipeline._load_pipeline("t2v-A14B", sample_params)
            assert result1 == mock_wrapper
            assert mock_loader.load_wan_pipeline.call_count == 1
            
            # Second call should use cache
            result2 = await pipeline._load_pipeline("t2v-A14B", sample_params)
            assert result2 == mock_wrapper
            assert mock_loader.load_wan_pipeline.call_count == 1  # No additional calls
    
    def test_generation_stats(self, pipeline):
        """Test generation statistics"""
        # Simulate some generations
        pipeline._generation_count = 5
        pipeline._total_generation_time = 150.0
        
        stats = pipeline.get_generation_stats()
        
        assert stats["total_generations"] == 5
        assert stats["total_generation_time"] == 150.0
        assert stats["average_generation_time"] == 30.0
        assert "cached_pipelines" in stats
        assert "wan_pipeline_loader_available" in stats
    
    def test_clear_pipeline_cache(self, pipeline):
        """Test pipeline cache clearing"""
        # Add something to cache
        pipeline._pipeline_cache["test"] = Mock()
        assert len(pipeline._pipeline_cache) == 1
        
        pipeline.clear_pipeline_cache()
        assert len(pipeline._pipeline_cache) == 0
    
    @pytest.mark.asyncio
    async def test_error_result_creation(self, pipeline):
        """Test error result creation with recovery suggestions"""
        result = pipeline._create_error_result(
            "test_task", "model_loading", "Failed to load model"
        )
        
        assert result.success is False
        assert result.task_id == "test_task"
        assert result.error_category == "model_loading"
        assert result.error_message == "Failed to load model"
        assert len(result.recovery_suggestions) > 0
        assert "model files are downloaded" in result.recovery_suggestions[0]
    
    @pytest.mark.asyncio
    async def test_image_validation(self, pipeline):
        """Test image loading and validation"""
        # Test with non-existent file
        result = await pipeline._load_and_validate_image("/nonexistent.jpg")
        assert result is None
        
        # Test with valid image (mocked)
        with patch('PIL.Image.open') as mock_open:
            mock_image = Mock()
            mock_image.mode = 'RGB'
            mock_image.size = (512, 512)
            mock_open.return_value = mock_image
            
            result = await pipeline._load_and_validate_image("test.jpg")
            assert result == mock_image
    
    @pytest.mark.asyncio
    async def test_save_generated_video(self, pipeline, sample_params):
        """Test video saving functionality"""
        frames = [torch.randn(3, 256, 256) for _ in range(16)]
        task_id = "test_save"
        
        with patch('pathlib.Path.mkdir'), \
             patch('builtins.open', create=True) as mock_open:
            
            result = await pipeline._save_generated_video(frames, task_id, sample_params)
            
            assert "outputs/generated_" in result
            assert task_id in result
            assert result.endswith(".mp4")
    
    @pytest.mark.asyncio
    async def test_global_pipeline_instance(self):
        """Test global pipeline instance creation"""
        with patch('backend.services.real_generation_pipeline.RealGenerationPipeline') as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance
            
            # Clear global instance
            import backend.services.real_generation_pipeline as pipeline_module
            pipeline_module._real_generation_pipeline = None
            
            pipeline = await get_real_generation_pipeline()
            
            assert pipeline == mock_instance
            mock_instance.initialize.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])