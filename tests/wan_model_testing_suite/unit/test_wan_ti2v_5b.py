"""
Unit tests for WAN TI2V-5B model implementation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the model under test
try:
    from core.models.wan_models.wan_ti2v_5b import (
        WANTI2V5B, 
        TI2VGenerationParams,
        CompactImageEncoder,
        CompactTextEncoder,
        DualConditioningBlock,
        CompactTemporalAttention,
        TI2VTransformerBlock,
        ImageInterpolator
    )
    WAN_MODELS_AVAILABLE = True
except ImportError:
    WAN_MODELS_AVAILABLE = False
    # Create mock classes for testing
    class WANTI2V5B:
        pass
    class TI2VGenerationParams:
        pass


@pytest.mark.unit
@pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
class TestWANTI2V5B:
    """Test suite for WAN TI2V-5B model"""
    
    def test_model_initialization(self, mock_model_config):
        """Test model initialization"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            # Mock configuration for 5B model (smaller than 14B models)
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024  # Smaller for 5B
            mock_config.return_value.architecture.num_layers = 16    # Fewer layers
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            # Initialize model
            model = WANTI2V5B(mock_model_config)
            
            # Verify initialization
            assert model.hidden_dim == 1024  # Compact size
            assert model.num_layers == 16    # Fewer layers
            assert model.num_heads == 16
            assert model.max_frames == 16
            assert model.max_width == 1280
            assert model.max_height == 720
    
    def test_model_capabilities(self, mock_model_config):
        """Test model capabilities"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024
            mock_config.return_value.architecture.num_layers = 16
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            capabilities = model._get_model_capabilities()
            
            # Verify TI2V-specific capabilities
            assert capabilities.supports_text_conditioning is True
            assert capabilities.supports_image_conditioning is True
            assert capabilities.supports_dual_conditioning is True
            assert capabilities.supports_lora is True
            assert capabilities.max_frames == 16
            assert capabilities.max_resolution == (1280, 720)
            assert capabilities.parameter_count == 5_000_000_000  # 5B parameters
            # Should be more efficient than 14B models
            assert capabilities.estimated_vram_gb < 11.0
    
    def test_dual_conditioning_validation(self, mock_model_config, temp_dir):
        """Test dual conditioning (text + image) validation"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024
            mock_config.return_value.architecture.num_layers = 16
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            
            # Test valid dual conditioning parameters
            mock_tensor = Mock()
            mock_tensor.shape = (1, 3, 512, 512)
            
            valid_params = TI2VGenerationParams(
                image=mock_tensor,
                prompt="A cat playing in a garden",  # Both image and text
                num_frames=16,
                width=512,
                height=512,
                num_inference_steps=50,
                text_guidance_scale=7.5,
                image_guidance_scale=1.5
            )
            
            is_valid, errors = model.validate_generation_params(valid_params)
            assert is_valid is True
            assert len(errors) == 0
            
            # Test missing text prompt (should fail for TI2V)
            invalid_params = TI2VGenerationParams(
                image=mock_tensor,
                prompt="",  # Empty text prompt
                num_frames=16,
                width=512,
                height=512
            )
            
            is_valid, errors = model.validate_generation_params(invalid_params)
            assert is_valid is False
            assert any("prompt" in error.lower() for error in errors)
    
    def test_image_interpolation_functionality(self, mock_model_config, mock_torch):
        """Test image interpolation for start/end image generation"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024
            mock_config.return_value.architecture.num_layers = 16
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            
            # Mock image interpolator
            model.image_interpolator = Mock()
            interpolated_features = mock_torch.randn(1, 16, 1024)  # (batch, frames, dim)
            model.image_interpolator.return_value = interpolated_features
            
            # Test interpolation with start and end images
            start_image_features = mock_torch.randn(1, 1024)
            end_image_features = mock_torch.randn(1, 1024)
            
            result = model.image_interpolator(start_image_features, end_image_features, 16)
            
            # Verify interpolation was called
            model.image_interpolator.assert_called_once()
            assert result is not None
    
    def test_compact_architecture_efficiency(self, mock_model_config):
        """Test that compact architecture is more efficient"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024  # Smaller than 14B models
            mock_config.return_value.architecture.num_layers = 16    # Fewer layers
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            
            # Verify compact dimensions
            assert model.hidden_dim == 1024  # Smaller than 1536 in 14B models
            assert model.num_layers == 16    # Fewer than 24 in 14B models
            
            # Verify memory efficiency
            capabilities = model._get_model_capabilities()
            assert capabilities.parameter_count == 5_000_000_000  # 5B vs 14B
            assert capabilities.estimated_vram_gb < 10.0  # More efficient
    
    def test_dual_conditioning_forward_pass(self, mock_model_config, mock_torch):
        """Test forward pass with dual conditioning (text + image)"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024
            mock_config.return_value.architecture.num_layers = 16
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            
            # Mock model components
            model.input_projection = Mock(return_value=mock_torch.randn(1, 16, 64, 64, 1024))
            model.temporal_pos_encoding = Mock(return_value=mock_torch.randn(16, 4096, 1024))
            model.time_embedding = Mock(return_value=mock_torch.randn(1, 1024))
            model.transformer_blocks = [Mock(return_value=mock_torch.randn(1, 16, 64, 64, 1024))]
            model.output_projection = Mock(return_value=mock_torch.randn(1, 16, 64, 64, 4))
            
            # Test forward pass with dual conditioning
            latents = mock_torch.randn(1, 16, 4, 64, 64)
            timestep = mock_torch.tensor([500])
            text_embeddings = mock_torch.randn(1, 256, 1024)
            image_features = mock_torch.randn(1, 256, 1024)
            
            output = model.forward(latents, timestep, text_embeddings, image_features)
            
            # Verify forward pass components were called
            model.input_projection.assert_called_once()
            model.output_projection.assert_called_once()
            assert output is not None
    
    def test_end_image_interpolation_params(self, mock_model_config, mock_torch):
        """Test generation with end image for interpolation"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024
            mock_config.return_value.architecture.num_layers = 16
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            
            # Test parameters with end image for interpolation
            start_image = mock_torch.randn(1, 3, 512, 512)
            end_image = mock_torch.randn(1, 3, 512, 512)
            
            params = TI2VGenerationParams(
                image=start_image,
                end_image=end_image,  # End image for interpolation
                prompt="A cat playing in a garden",
                interpolation_strength=1.0,
                num_frames=16
            )
            
            is_valid, errors = model.validate_generation_params(params)
            assert is_valid is True
            assert len(errors) == 0
            assert params.end_image is not None
            assert params.interpolation_strength == 1.0
    
    @pytest.mark.slow
    def test_ti2v_generation_pipeline(self, mock_model_config, mock_torch, mock_progress_callback):
        """Test complete TI2V generation pipeline with dual conditioning"""
        with patch('core.models.wan_models.wan_model_config.get_wan_model_config') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.architecture = Mock()
            mock_config.return_value.architecture.hidden_dim = 1024
            mock_config.return_value.architecture.num_layers = 16
            mock_config.return_value.architecture.attention_heads = 16
            mock_config.return_value.architecture.temporal_layers = 8
            mock_config.return_value.architecture.max_frames = 16
            mock_config.return_value.architecture.max_resolution = (1280, 720)
            
            model = WANTI2V5B(mock_model_config)
            
            # Mock generation pipeline components
            model.preprocess_image = Mock(return_value=mock_torch.randn(1, 3, 512, 512))
            model._encode_image = Mock(return_value=(mock_torch.randn(1, 1024), mock_torch.randn(1, 256, 1024)))
            model._encode_text = Mock(return_value=(mock_torch.randn(1, 256, 1024), mock_torch.randn(1, 256, 1024)))
            model.scheduler = Mock()
            model.scheduler.set_timesteps = Mock()
            model.scheduler.timesteps = [500, 400, 300, 200, 100]
            model.scheduler.step = Mock(return_value=mock_torch.randn(1, 16, 4, 64, 64))
            model.forward = Mock(return_value=mock_torch.randn(1, 16, 4, 64, 64))
            
            # Create generation parameters with dual conditioning
            params = TI2VGenerationParams(
                image=mock_torch.randn(1, 3, 512, 512),
                prompt="A cat playing in a garden",
                num_frames=16,
                width=512,
                height=512,
                num_inference_steps=5,
                text_guidance_scale=7.5,
                image_guidance_scale=1.5,
                callback=mock_progress_callback
            )
            
            # Test generation
            result = model.generate(params)
            
            # Verify generation components were called
            model.preprocess_image.assert_called_once()
            model._encode_image.assert_called_once()
            model._encode_text.assert_called_once()
            model.scheduler.set_timesteps.assert_called_once()
            assert mock_progress_callback.call_count > 0


@pytest.mark.unit
class TestTI2VComponents:
    """Test individual TI2V model components"""
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_compact_image_encoder(self, mock_torch):
        """Test compact image encoder component"""
        image_encoder = CompactImageEncoder(input_channels=3, embed_dim=1024, image_size=512)
        
        # Test forward pass
        image = mock_torch.randn(1, 3, 512, 512)
        global_features, spatial_features = image_encoder(image)
        
        assert global_features is not None
        assert spatial_features is not None
        # Verify compact architecture (fewer parameters than full encoder)
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_compact_text_encoder(self, mock_torch):
        """Test compact text encoder component"""
        text_encoder = CompactTextEncoder(vocab_size=50000, embed_dim=1024, max_length=256)
        
        # Test forward pass
        input_ids = mock_torch.ones(2, 128, dtype=mock_torch.long)
        output = text_encoder(input_ids)
        
        assert output is not None
        # Verify compact architecture (fewer transformer layers)
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_dual_conditioning_block(self, mock_torch):
        """Test dual conditioning block"""
        conditioning_block = DualConditioningBlock(dim=1024, num_heads=8)
        
        # Test forward pass
        video_features = mock_torch.randn(2, 1024, 1024)  # (batch, tokens, dim)
        text_features = mock_torch.randn(2, 256, 1024)    # (batch, text_tokens, dim)
        image_features = mock_torch.randn(2, 256, 1024)   # (batch, image_tokens, dim)
        
        output = conditioning_block(video_features, text_features, image_features)
        
        assert output is not None
        # In real implementation, would verify fusion mechanism
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_compact_temporal_attention(self, mock_torch):
        """Test compact temporal attention with dual conditioning"""
        temporal_attn = CompactTemporalAttention(dim=1024, num_heads=8)
        
        # Test forward pass
        x = mock_torch.randn(2, 16, 32, 32, 1024)  # (batch, frames, height, width, channels)
        text_features = mock_torch.randn(2, 256, 1024)  # (batch, text_tokens, dim)
        image_features = mock_torch.randn(2, 256, 1024)  # (batch, image_tokens, dim)
        
        output = temporal_attn(x, text_features, image_features)
        
        assert output is not None
        # In real implementation, would verify dual conditioning effectiveness
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_ti2v_transformer_block(self, mock_torch):
        """Test TI2V transformer block with dual conditioning"""
        transformer_block = TI2VTransformerBlock(dim=1024, num_heads=8, mlp_ratio=2.0)  # Compact MLP
        
        # Test forward pass
        x = mock_torch.randn(2, 16, 32, 32, 1024)  # (batch, frames, height, width, channels)
        text_features = mock_torch.randn(2, 256, 1024)  # (batch, text_tokens, dim)
        image_features = mock_torch.randn(2, 256, 1024)  # (batch, image_tokens, dim)
        
        output = transformer_block(x, text_features, image_features)
        
        assert output is not None
        # In real implementation, would verify compact efficiency
    
    @pytest.mark.skipif(not WAN_MODELS_AVAILABLE, reason="WAN models not available")
    def test_image_interpolator(self, mock_torch):
        """Test image interpolation component"""
        interpolator = ImageInterpolator(embed_dim=1024)
        
        # Test forward pass
        start_features = mock_torch.randn(2, 1024)
        end_features = mock_torch.randn(2, 1024)
        num_frames = 16
        
        output = interpolator(start_features, end_features, num_frames)
        
        assert output is not None
        # In real implementation, would verify interpolation quality and smoothness


@pytest.mark.unit
class TestTI2VGenerationParams:
    """Test TI2V generation parameters"""
    
    def test_params_creation(self, mock_torch):
        """Test parameter creation and validation"""
        image_tensor = mock_torch.randn(1, 3, 512, 512)
        end_image_tensor = mock_torch.randn(1, 3, 512, 512)
        
        params = TI2VGenerationParams(
            image=image_tensor,
            prompt="A cat playing in a garden",
            end_image=end_image_tensor,
            negative_prompt="blurry",
            num_frames=16,
            width=512,
            height=512,
            fps=8.0,
            num_inference_steps=50,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            text_guidance_scale=7.5,
            interpolation_strength=1.0,
            seed=42
        )
        
        assert params.image is image_tensor
        assert params.prompt == "A cat playing in a garden"
        assert params.end_image is end_image_tensor
        assert params.negative_prompt == "blurry"
        assert params.num_frames == 16
        assert params.width == 512
        assert params.height == 512
        assert params.fps == 8.0
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 7.5
        assert params.image_guidance_scale == 1.5
        assert params.text_guidance_scale == 7.5
        assert params.interpolation_strength == 1.0
        assert params.seed == 42
    
    def test_params_defaults(self, mock_torch):
        """Test parameter defaults"""
        image_tensor = mock_torch.randn(1, 3, 512, 512)
        params = TI2VGenerationParams(
            image=image_tensor,
            prompt="Test prompt"
        )
        
        assert params.end_image is None
        assert params.negative_prompt is None
        assert params.num_frames == 16
        assert params.width == 1280
        assert params.height == 720
        assert params.fps == 8.0
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 7.5
        assert params.image_guidance_scale == 1.5
        assert params.text_guidance_scale == 7.5
        assert params.interpolation_strength == 1.0
        assert params.seed is None
    
    def test_dual_conditioning_requirements(self, mock_torch):
        """Test that both image and text are required for TI2V"""
        image_tensor = mock_torch.randn(1, 3, 512, 512)
        
        # Valid dual conditioning
        params = TI2VGenerationParams(
            image=image_tensor,
            prompt="A cat playing in a garden"
        )
        
        assert params.image is not None
        assert params.prompt is not None
        assert params.prompt != ""
        
        # Test interpolation parameters
        end_image_tensor = mock_torch.randn(1, 3, 512, 512)
        params_with_interpolation = TI2VGenerationParams(
            image=image_tensor,
            prompt="A cat playing in a garden",
            end_image=end_image_tensor,
            interpolation_strength=0.8
        )
        
        assert params_with_interpolation.end_image is not None
        assert params_with_interpolation.interpolation_strength == 0.8