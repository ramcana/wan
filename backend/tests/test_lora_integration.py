"""
Test LoRA integration in the real generation pipeline
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import the classes we're testing
try:
    from backend.services.real_generation_pipeline import RealGenerationPipeline
    from backend.core.model_integration_bridge import GenerationParams, ModelIntegrationBridge
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Required imports not available: {import_error if not IMPORTS_AVAILABLE else ''}")
class TestLoRAIntegration:
    """Test LoRA integration functionality"""
    
    @pytest.fixture
    def temp_loras_dir(self):
        """Create temporary LoRAs directory for testing"""
        temp_dir = tempfile.mkdtemp()
        loras_dir = Path(temp_dir) / "loras"
        loras_dir.mkdir(exist_ok=True)
        
        # Create mock LoRA files
        mock_lora_files = [
            "anime_style.safetensors",
            "realistic_photo.pt",
            "detail_enhancer.pth"
        ]
        
        for lora_file in mock_lora_files:
            (loras_dir / lora_file).write_bytes(b"mock_lora_data")
        
        yield str(loras_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_loras_dir):
        """Mock configuration with LoRA directory"""
        return {
            "directories": {
                "loras_directory": temp_loras_dir,
                "models_directory": "models",
                "outputs_directory": "outputs"
            },
            "lora_max_file_size_mb": 500,
            "optimization": {
                "max_vram_usage_gb": 12
            }
        }
    
    @pytest.fixture
    def generation_params(self, temp_loras_dir):
        """Sample generation parameters with LoRA"""
        return GenerationParams(
            prompt="a beautiful landscape",
            model_type="t2v-A14B",
            resolution="1280x720",
            steps=20,
            lora_path=str(Path(temp_loras_dir) / "anime_style.safetensors"),
            lora_strength=0.8,
            num_frames=16
        )
    
    @pytest.mark.asyncio
    async def test_lora_validation_valid_params(self, temp_loras_dir):
        """Test LoRA parameter validation with valid parameters"""
        pipeline = RealGenerationPipeline()
        
        # Mock the LoRA manager initialization
        with patch('backend.services.real_generation_pipeline.LORA_MANAGER_AVAILABLE', True):
            await pipeline._initialize_lora_manager()
        
        params = GenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_path=str(Path(temp_loras_dir) / "anime_style.safetensors"),
            lora_strength=0.8
        )
        
        validation_result = pipeline._validate_lora_params(params)
        
        assert validation_result["valid"] == True
        assert len(validation_result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_lora_validation_invalid_strength(self):
        """Test LoRA parameter validation with invalid strength"""
        pipeline = RealGenerationPipeline()
        
        params = GenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_strength=2.5  # Invalid - too high
        )
        
        validation_result = pipeline._validate_lora_params(params)
        
        assert validation_result["valid"] == False
        assert any("strength must be between 0.0 and 2.0" in error for error in validation_result["errors"])
    
    @pytest.mark.asyncio
    async def test_lora_validation_missing_file(self):
        """Test LoRA parameter validation with missing file"""
        pipeline = RealGenerationPipeline()
        
        params = GenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_path="/nonexistent/path/missing_lora.safetensors",
            lora_strength=1.0
        )
        
        validation_result = pipeline._validate_lora_params(params)
        
        assert validation_result["valid"] == False
        assert any("LoRA file not found" in error for error in validation_result["errors"])
    
    @pytest.mark.asyncio
    async def test_lora_fallback_enhancement(self):
        """Test LoRA fallback prompt enhancement"""
        pipeline = RealGenerationPipeline()
        
        # Test basic fallback enhancement
        enhanced_prompt = pipeline._get_basic_lora_fallback("a beautiful scene", "anime_style")
        assert "anime style" in enhanced_prompt.lower()
        assert "a beautiful scene" in enhanced_prompt
        
        # Test realistic LoRA fallback
        enhanced_prompt = pipeline._get_basic_lora_fallback("portrait", "realistic_photo")
        assert "photorealistic" in enhanced_prompt.lower()
        
        # Test detail LoRA fallback
        enhanced_prompt = pipeline._get_basic_lora_fallback("landscape", "detail_enhancer")
        assert "detailed" in enhanced_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_lora_application_with_manager(self, mock_config, temp_loras_dir):
        """Test LoRA application when LoRA manager is available"""
        pipeline = RealGenerationPipeline()
        
        # Mock LoRA manager
        mock_lora_manager = Mock()
        mock_lora_manager.load_lora.return_value = {
            "name": "anime_style",
            "num_layers": 16,
            "size_mb": 50.0
        }
        mock_lora_manager.apply_lora.return_value = Mock()  # Mock modified pipeline
        
        pipeline.lora_manager = mock_lora_manager
        
        # Mock pipeline wrapper
        mock_pipeline_wrapper = Mock()
        mock_pipeline_wrapper.pipeline = Mock()
        
        params = GenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_path=str(Path(temp_loras_dir) / "anime_style.safetensors"),
            lora_strength=0.8
        )
        
        # Test LoRA application
        result = await pipeline._apply_lora_to_pipeline(mock_pipeline_wrapper, params, "test_task")
        
        assert result == True
        mock_lora_manager.load_lora.assert_called_once_with("anime_style")
        mock_lora_manager.apply_lora.assert_called_once()
        assert "test_task" in pipeline._applied_loras
    
    @pytest.mark.asyncio
    async def test_lora_application_fallback(self, temp_loras_dir):
        """Test LoRA application fallback when manager is not available"""
        pipeline = RealGenerationPipeline()
        pipeline.lora_manager = None  # No LoRA manager available
        
        mock_pipeline_wrapper = Mock()
        
        params = GenerationParams(
            prompt="original prompt",
            model_type="t2v-A14B",
            lora_path=str(Path(temp_loras_dir) / "anime_style.safetensors"),
            lora_strength=0.8
        )
        
        # Test fallback application
        result = await pipeline._apply_lora_to_pipeline(mock_pipeline_wrapper, params, "test_task")
        
        assert result == True
        # Check that prompt was enhanced
        assert params.prompt != "original prompt"
        assert "anime style" in params.prompt.lower()
    
    @pytest.mark.asyncio
    async def test_lora_cleanup(self):
        """Test LoRA cleanup after task completion"""
        pipeline = RealGenerationPipeline()
        
        # Add a mock applied LoRA
        task_id = "test_task_123"
        pipeline._applied_loras[task_id] = {
            "name": "test_lora",
            "strength": 1.0,
            "path": "/path/to/lora.safetensors"
        }
        
        # Test cleanup
        await pipeline._cleanup_lora_for_task(task_id)
        
        # Verify LoRA was removed from tracking
        assert task_id not in pipeline._applied_loras
    
    @pytest.mark.asyncio
    async def test_lora_status_reporting(self):
        """Test LoRA status reporting"""
        pipeline = RealGenerationPipeline()
        
        # Mock LoRA manager
        mock_lora_manager = Mock()
        mock_lora_manager.list_available_loras.return_value = {
            "anime_style": {"size_mb": 50.0, "is_loaded": True},
            "realistic": {"size_mb": 75.0, "is_loaded": False}
        }
        mock_lora_manager.loaded_loras = {"anime_style": {}}
        
        pipeline.lora_manager = mock_lora_manager
        pipeline._applied_loras = {
            "task_1": {"name": "anime_style", "strength": 0.8}
        }
        
        # Get status
        status = pipeline.get_lora_status()
        
        assert status["lora_manager_available"] == True
        assert len(status["available_loras"]) == 2
        assert "anime_style" in status["available_loras"]
        assert len(status["loaded_loras"]) == 1
        assert len(status["applied_loras"]) == 1
    
    @pytest.mark.asyncio
    async def test_model_integration_bridge_lora_status(self):
        """Test LoRA status from model integration bridge"""
        bridge = ModelIntegrationBridge()
        
        with patch('backend.core.model_integration_bridge.get_system_integration') as mock_get_integration:
            # Mock system integration
            mock_integration = Mock()
            mock_integration.config = {
                "directories": {"loras_directory": "loras"}
            }
            mock_integration.scan_available_loras.return_value = ["test_lora.safetensors"]
            mock_get_integration.return_value = mock_integration
            
            # Mock Path.exists to return True
            with patch('pathlib.Path.exists', return_value=True):
                status = await bridge.get_lora_status()
            
            assert status["lora_support_available"] == True
            assert len(status["available_loras"]) == 1
            assert "test_lora.safetensors" in status["available_loras"]
    
    def test_lora_file_extension_validation(self):
        """Test LoRA file extension validation"""
        pipeline = RealGenerationPipeline()
        
        valid_extensions = ['.safetensors', '.pt', '.pth', '.bin']
        invalid_extensions = ['.txt', '.json', '.pkl', '.ckpt']
        
        for ext in valid_extensions:
            params = GenerationParams(
                prompt="test",
                model_type="t2v-A14B",
                lora_path=f"/path/to/lora{ext}",
                lora_strength=1.0
            )
            
            # Mock file existence
            with patch('pathlib.Path.exists', return_value=True):
                validation = pipeline._validate_lora_params(params)
                # Should not have extension-related errors
                extension_errors = [e for e in validation["errors"] if "Invalid LoRA file format" in e]
                assert len(extension_errors) == 0
        
        for ext in invalid_extensions:
            params = GenerationParams(
                prompt="test",
                model_type="t2v-A14B", 
                lora_path=f"/path/to/lora{ext}",
                lora_strength=1.0
            )
            
            # Mock file existence
            with patch('pathlib.Path.exists', return_value=True):
                validation = pipeline._validate_lora_params(params)
                # Should have extension-related errors
                extension_errors = [e for e in validation["errors"] if "Invalid LoRA file format" in e]
                assert len(extension_errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])