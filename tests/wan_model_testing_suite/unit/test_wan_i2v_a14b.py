"""
Unit tests for WAN I2V-A14B model implementation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the model under test
try:
    from backend.core.models.wan_models.wan_i2v_a14b import (
        WANI2VA14B, 
        I2VGenerationParams,
        ImageEncoder,
        ImageConditioningBlock,
        TemporalImageAttention,
        I2VTransformerBlock
    )
    WAN_MODELS_AVAILABLE = True
except ImportError:
    WAN_MODELS_AVAILABLE = False
    # Create mock classes for testing
    class WANI2VA14B:
        pass
    class I2VGenerationParams:
        pass


@pytest.mark.unit
@pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
class TestWANI2VA14B:
    """Test suite for WAN I2V-A14B model"""
    
    def test_model_initialization(self, mock_model_config):
        """Test model initialization"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            # Mock configuration
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            # Initialize model
            model = WANI2VA14B(mock_model_config)
            
            # Verify initialization
            assert model.hidden_dim == 1536
            assert model.num_layers == 24
            assert model.num_heads == 24
            assert model.max_frames == 16
            assert model.max_width == 1280
            assert model.max_height == 720
    
    def test_model_capabilities(self, mock_model_config):
        """Test model capabilities"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            capabilities = model._get_model_capabilities()
            
            # Verify I2V-specific capabilities
            assert capabilities.supports_text_conditioning is True  # Optional
            assert capabilities.supports_image_conditioning is True  # Primary
            assert capabilities.supports_dual_conditioning is True   # Both
            assert capabilities.supports_lora is True
            assert capabilities.max_frames == 16
            assert capabilities.max_resolution == (1280, 720)
            assert capabilities.parameter_count == 14_000_000_000
            assert capabilities.estimated_vram_gb == 11.0  # Higher due to image encoder
    
    def test_image_preprocessing(self, mock_model_config, temp_dir):
        """Test image preprocessing functionality"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            
            # Test tensor input preprocessing
            with patch('torch.randn') as mock_randn:
                mock_tensor = Mock()
                mock_tensor.shape = (1, 3, 512, 512)
                mock_tensor.max.return_value = 255.0
                mock_tensor.permute.return_value = mock_tensor
                mock_tensor.unsqueeze.return_value = mock_tensor
                mock_randn.return_value = mock_tensor
                
                # Mock F.interpolate
                with patch('torch.nn.functional.interpolate') as mock_interpolate:
                    mock_interpolate.return_value = mock_tensor
                    
                    result = model.preprocess_image(mock_tensor)
                    assert result is not None
    
    def test_image_validation(self, mock_model_config, temp_dir):
        """Test image input validation"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            
            # Test valid tensor input
            mock_tensor = Mock()
            mock_tensor.shape = (1, 3, 512, 512)
            is_valid, errors = model.validate_image_input(mock_tensor)
            assert is_valid is True
            assert len(errors) == 0
            
            # Test invalid tensor shape
            mock_tensor.shape = (2, 3, 512, 512)  # Batch size > 1
            is_valid, errors = model.validate_image_input(mock_tensor)
            assert is_valid is False
            assert len(errors) > 0
            
            # Test non-existent file
            fake_path = temp_dir / "nonexistent.jpg"
            is_valid, errors = model.validate_image_input(str(fake_path))
            assert is_valid is False
            assert "not found" in str(errors[0])
    
    def test_image_encoding(self, mock_model_config, mock_torch):
        """Test image encoding functionality"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            
            # Mock image encoder
            model.image_encoder = Mock()
            global_features = mock_torch.randn(1, 1536)
            spatial_features = mock_torch.randn(1, 1536, 16, 16)
            model.image_encoder.return_value = (global_features, spatial_features)
            
            # Test image encoding
            image = mock_torch.randn(1, 3, 512, 512)
            global_feat, spatial_feat = model._encode_image(image)
            
            # Verify encoding was called
            model.image_encoder.assert_called_once()
            assert global_feat is not None
            assert spatial_feat is not None
    
    def test_generation_params_validation(self, mock_model_config, temp_dir):
        """Test I2V generation parameters validation"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            
            # Create a mock image file
            image_path = temp_dir / "test_image.jpg"
            image_path.write_text("mock image data")
            
            # Test valid parameters
            mock_tensor = Mock()
            mock_tensor.shape = (1, 3, 512, 512)
            
            valid_params = I2VGenerationParams(
                image=mock_tensor,
                prompt="A cat playing in a garden",
                num_frames=16,
                width=512,
                height=512,
                num_inference_steps=50
            )
            
            is_valid, errors = model.validate_generation_params(valid_params)
            assert is_valid is True
            assert len(errors) == 0
            
            # Test invalid parameters
            invalid_params = I2VGenerationParams(
                image=None,  # No image
                prompt="",   # Empty prompt (optional but tested)
                num_frames=100,  # Too many frames
                width=2000,  # Too wide
                height=2000,  # Too tall
                num_inference_steps=0  # Invalid steps
            )
            
            is_valid, errors = model.validate_generation_params(invalid_params)
            assert is_valid is False
            assert len(errors) > 0
    
    def test_dual_conditioning_forward_pass(self, mock_model_config, mock_torch):
        """Test forward pass with dual conditioning (image + text)"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            
            # Mock model components
            model.input_projection = Mock(return_value=mock_torch.randn(1, 16, 64, 64, 1536))
            model.temporal_pos_encoding = Mock(return_value=mock_torch.randn(16, 4096, 1536))
            model.time_embedding = Mock(return_value=mock_torch.randn(1, 1536))
            model.image_condition_projection = Mock(return_value=mock_torch.randn(1, 1536))
            model.transformer_blocks = [Mock(return_value=mock_torch.randn(1, 16, 64, 64, 1536))]
            model.output_projection = Mock(return_value=mock_torch.randn(1, 16, 64, 64, 4))
            
            # Test forward pass with dual conditioning
            latents = mock_torch.randn(1, 16, 4, 64, 64)
            timestep = mock_torch.tensor([500])
            text_embeddings = mock_torch.randn(1, 256, 1536)
            image_features = mock_torch.randn(1, 256, 1536)
            
            output = model.forward(latents, timestep, text_embeddings, image_features)
            
            # Verify forward pass components were called
            model.input_projection.assert_called_once()
            model.output_projection.assert_called_once()
            assert output is not None
    
    @pytest.mark.slow
    def test_i2v_generation_pipeline(self, mock_model_config, mock_torch, mock_progress_callback):
        """Test complete I2V generation pipeline"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1536
            mock_config.return_value.architecture.num_layers = 24
            mock_config.return_value.architecture.attention_heads = 24
            mock_config.return_value.architecture.temporal_layers = 12
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANI2VA14B(mock_model_config)
            
            # Mock generation pipeline components
            model.preprocess_image = Mock(return_value=mock_torch.randn(1, 3, 512, 512))
            model._encode_image = Mock(return_value=(mock_torch.randn(1, 1536), mock_torch.randn(1, 256, 1536)))
            model._encode_text = Mock(return_value=(mock_torch.randn(1, 256, 1536), mock_torch.randn(1, 256, 1536)))
            model.scheduler = Mock()
            model.scheduler.set_timesteps = Mock()
            model.scheduler.timesteps = [500, 400, 300, 200, 100]
            model.scheduler.step = Mock(return_value=mock_torch.randn(1, 16, 4, 64, 64))
            model.forward = Mock(return_value=mock_torch.randn(1, 16, 4, 64, 64))
            
            # Create generation parameters
            params = I2VGenerationParams(
                image=mock_torch.randn(1, 3, 512, 512),
                prompt="A cat playing in a garden",
                num_frames=16,
                width=512,
                height=512,
                num_inference_steps=5,
                callback=mock_progress_callback
            )
            
            # Test generation
            result = model.generate(params)
            
            # Verify generation components were called
            model.preprocess_image.assert_called_once()
            model._encode_image.assert_called_once()
            model.scheduler.set_timesteps.assert_called_once()
            assert mock_progress_callback.call_count > 0


@pytest.mark.unit
class TestI2VComponents:
    """Test individual I2V model components"""
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_image_encoder(self, mock_torch):
        """Test image encoder component"""
        image_encoder = ImageEncoder(input_channels=3, embed_dim=1536, image_size=512)
        
        # Test forward pass
        image = mock_torch.randn(1, 3, 512, 512)
        global_features, spatial_features = image_encoder(image)
        
        assert global_features is not None
        assert spatial_features is not None
        # In real implementation, would verify feature dimensions and quality
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_image_conditioning_block(self, mock_torch):
        """Test image conditioning block"""
        conditioning_block = ImageConditioningBlock(dim=512, num_heads=8)
        
        # Test forward pass
        video_features = mock_torch.randn(2, 1024, 512)  # (batch, tokens, dim)
        image_features = mock_torch.randn(2, 256, 512)   # (batch, image_tokens, dim)
        
        output = conditioning_block(video_features, image_features)
        
        assert output is not None
        # In real implementation, would verify conditioning effectiveness
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_temporal_image_attention(self, mock_torch):
        """Test temporal attention with image conditioning"""
        temporal_attn = TemporalImageAttention(dim=512, num_heads=8)
        
        # Test forward pass
        x = mock_torch.randn(2, 16, 32, 32, 512)  # (batch, frames, height, width, channels)
        image_features = mock_torch.randn(2, 256, 512)  # (batch, image_tokens, dim)
        
        output = temporal_attn(x, image_features)
        
        assert output is not None
        # In real implementation, would verify temporal-image attention patterns
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_i2v_transformer_block(self, mock_torch):
        """Test I2V transformer block with image conditioning"""
        transformer_block = I2VTransformerBlock(dim=512, num_heads=8)
        
        # Test forward pass
        x = mock_torch.randn(2, 16, 32, 32, 512)  # (batch, frames, height, width, channels)
        image_features = mock_torch.randn(2, 256, 512)  # (batch, image_tokens, dim)
        
        output = transformer_block(x, image_features)
        
        assert output is not None
        # In real implementation, would verify image conditioning integration


@pytest.mark.unit
class TestI2VGenerationParams:
    """Test I2V generation parameters"""
    
    def test_params_creation(self, mock_torch):
        """Test parameter creation and validation"""
        image_tensor = mock_torch.randn(1, 3, 512, 512)
        
        params = I2VGenerationParams(
            image=image_tensor,
            prompt="A cat playing in a garden",
            negative_prompt="blurry",
            num_frames=16,
            width=512,
            height=512,
            fps=8.0,
            num_inference_steps=50,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            seed=42
        )
        
        assert params.image is image_tensor
        assert params.prompt == "A cat playing in a garden"
        assert params.negative_prompt == "blurry"
        assert params.num_frames == 16
        assert params.width == 512
        assert params.height == 512
        assert params.fps == 8.0
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 7.5
        assert params.image_guidance_scale == 1.5
        assert params.seed == 42
    
    def test_params_defaults(self, mock_torch):
        """Test parameter defaults"""
        image_tensor = mock_torch.randn(1, 3, 512, 512)
        params = I2VGenerationParams(image=image_tensor)
        
        assert params.prompt is None
        assert params.negative_prompt is None
        assert params.num_frames == 16
        assert params.width == 1280
        assert params.height == 720
        assert params.fps == 8.0
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 7.5
        assert params.image_guidance_scale == 1.5
        assert params.seed is None
    
    def test_params_with_file_path(self, temp_dir):
        """Test parameters with image file path"""
        # Create a mock image file
        image_path = temp_dir / "test_image.jpg"
        image_path.write_text("mock image data")
        
        params = I2VGenerationParams(image=str(image_path))
        
        assert params.image == str(image_path)
        assert isinstance(params.image, str)
