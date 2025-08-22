"""
Tests for VAE Compatibility Handler

Tests VAE shape detection, validation, and loading for different dimensional configurations.
"""

import json
import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from vae_compatibility_handler import (
    VAECompatibilityHandler,
    VAEDimensions,
    VAECompatibilityResult,
    VAELoadingResult,
    create_vae_compatibility_handler
)


class TestVAEDimensions:
    """Test VAE dimensions data class"""
    
    def test_2d_vae_dimensions(self):
        """Test 2D VAE dimensions"""
        dims = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
        
        assert dims.channels == 4
        assert dims.height == 64
        assert dims.width == 64
        assert dims.depth is None
        assert not dims.is_3d
        assert dims.shape == (4, 64, 64)
    
    def test_3d_vae_dimensions(self):
        """Test 3D VAE dimensions"""
        dims = VAEDimensions(channels=4, height=64, width=64, depth=16, is_3d=True)
        
        assert dims.channels == 4
        assert dims.height == 64
        assert dims.width == 64
        assert dims.depth == 16
        assert dims.is_3d
        assert dims.shape == (4, 16, 64, 64)


class TestVAECompatibilityHandler:
    """Test VAE compatibility handler functionality"""
    
    @pytest.fixture
    def handler(self):
        """Create VAE compatibility handler"""
        return VAECompatibilityHandler()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def create_vae_config(self, temp_dir: Path, config_data: dict) -> Path:
        """Create a temporary VAE config file"""
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        return config_path
    
    def test_detect_2d_vae_architecture(self, handler, temp_dir):
        """Test detection of 2D VAE architecture"""
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64
        }
        
        config_path = self.create_vae_config(temp_dir, config_data)
        result = handler.detect_vae_architecture(config_path)
        
        assert result.is_compatible
        assert not result.detected_dimensions.is_3d
        assert result.detected_dimensions.channels == 4
        assert result.detected_dimensions.height == 64
        assert result.detected_dimensions.width == 64
        assert result.detected_dimensions.depth is None
        assert result.loading_strategy == "standard"
    
    def test_detect_3d_vae_architecture(self, handler, temp_dir):
        """Test detection of 3D VAE architecture"""
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": [16, 64, 64],
            "temporal_layers": True,
            "temporal_depth": 16
        }
        
        config_path = self.create_vae_config(temp_dir, config_data)
        result = handler.detect_vae_architecture(config_path)
        
        assert result.is_compatible
        assert result.detected_dimensions.is_3d
        assert result.detected_dimensions.channels == 4
        assert result.detected_dimensions.height == 64
        assert result.detected_dimensions.width == 64
        assert result.detected_dimensions.depth == 16
        assert result.loading_strategy == "standard"
    
    def test_detect_problematic_vae_shape(self, handler, temp_dir):
        """Test detection of problematic VAE shape (384x384)"""
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384  # Problematic size
        }
        
        config_path = self.create_vae_config(temp_dir, config_data)
        result = handler.detect_vae_architecture(config_path)
        
        assert result.detected_dimensions.height == 384
        assert result.detected_dimensions.width == 384
        assert result.loading_strategy == "reshape"
        assert any("384" in issue for issue in result.compatibility_issues)
    
    def test_detect_3d_vae_missing_depth(self, handler, temp_dir):
        """Test detection of 3D VAE with missing depth parameter"""
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64,
            "temporal_layers": True  # Indicates 3D but no depth specified
        }
        
        config_path = self.create_vae_config(temp_dir, config_data)
        result = handler.detect_vae_architecture(config_path)
        
        assert result.detected_dimensions.is_3d
        assert result.loading_strategy == "custom"
        assert any("temporal depth not specified" in issue for issue in result.compatibility_issues)
    
    def test_detect_vae_missing_config(self, handler, temp_dir):
        """Test handling of missing VAE config file"""
        missing_path = temp_dir / "nonexistent_config.json"
        result = handler.detect_vae_architecture(missing_path)
        
        assert not result.is_compatible
        assert "not found" in result.error_message
    
    def test_detect_vae_invalid_config(self, handler, temp_dir):
        """Test handling of invalid VAE config file"""
        config_path = temp_dir / "invalid_config.json"
        with open(config_path, 'w') as f:
            f.write("invalid json content")
        
        result = handler.detect_vae_architecture(config_path)
        
        assert not result.is_compatible
        assert "Failed to analyze VAE config" in result.error_message
    
    def test_validate_vae_weights_2d(self, handler, temp_dir):
        """Test validation of 2D VAE weights"""
        # Create mock weights file
        weights_path = temp_dir / "pytorch_model.bin"
        
        # Mock weight shapes for 2D VAE
        mock_weights = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),  # 2D conv
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3),  # 2D conv
            "quant_conv.weight": torch.randn(8, 8, 1, 1)
        }
        
        torch.save(mock_weights, weights_path)
        
        expected_dims = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
        result = handler.validate_vae_weights(weights_path, expected_dims)
        
        assert result.is_compatible
        assert not result.detected_dimensions.is_3d
        assert result.loading_strategy == "standard"
    
    def test_validate_vae_weights_3d(self, handler, temp_dir):
        """Test validation of 3D VAE weights"""
        # Create mock weights file
        weights_path = temp_dir / "pytorch_model.bin"
        
        # Mock weight shapes for 3D VAE
        mock_weights = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3, 3),  # 3D conv (5 dimensions)
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3, 3),  # 3D conv
            "temporal_conv.weight": torch.randn(64, 64, 1, 3, 3)
        }
        
        torch.save(mock_weights, weights_path)
        
        expected_dims = VAEDimensions(channels=4, height=64, width=64, depth=16, is_3d=True)
        result = handler.validate_vae_weights(weights_path, expected_dims)
        
        assert result.detected_dimensions.is_3d
        # May not be fully compatible due to architecture mismatch detection
    
    def test_validate_vae_weights_missing_file(self, handler, temp_dir):
        """Test handling of missing VAE weights file"""
        missing_path = temp_dir / "nonexistent_weights.bin"
        expected_dims = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
        
        result = handler.validate_vae_weights(missing_path, expected_dims)
        
        assert not result.is_compatible
        assert "not found" in result.error_message
    
    @patch('vae_compatibility_handler.AutoencoderKL')
    def test_load_vae_standard_success(self, mock_autoencoder, handler, temp_dir):
        """Test successful standard VAE loading"""
        # Mock successful VAE loading
        mock_vae = Mock()
        mock_autoencoder.from_pretrained.return_value = mock_vae
        
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, is_3d=False),
            loading_strategy="standard"
        )
        
        result = handler.load_vae_with_compatibility(temp_dir, compatibility_result)
        
        assert result.success
        assert result.vae_model == mock_vae
        assert result.loading_strategy_used == "standard"
        assert not result.fallback_used
    
    @patch('vae_compatibility_handler.AutoencoderKL')
    def test_load_vae_with_reshape(self, mock_autoencoder, handler, temp_dir):
        """Test VAE loading with reshape handling"""
        # Mock VAE with problematic config
        mock_vae = Mock()
        mock_vae.config = Mock()
        mock_vae.config.sample_size = [384, 384]
        mock_autoencoder.from_pretrained.return_value = mock_vae
        
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 384, 384, is_3d=False),
            loading_strategy="reshape"
        )
        
        result = handler.load_vae_with_compatibility(temp_dir, compatibility_result)
        
        assert result.success
        assert result.loading_strategy_used == "reshape"
        assert len(result.warnings) > 0
        assert any("reshape" in warning.lower() for warning in result.warnings)
        # Check that config was modified
        assert mock_vae.config.sample_size == [64, 64]
    
    @patch('vae_compatibility_handler.AutoencoderKL')
    def test_load_vae_3d_with_trust_remote_code(self, mock_autoencoder, handler, temp_dir):
        """Test 3D VAE loading with trust_remote_code"""
        # Mock successful 3D VAE loading
        mock_vae = Mock()
        mock_autoencoder.from_pretrained.return_value = mock_vae
        
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, depth=16, is_3d=True),
            loading_strategy="custom"
        )
        
        result = handler.load_vae_with_compatibility(temp_dir, compatibility_result)
        
        assert result.success
        assert result.loading_strategy_used == "custom_3d"
        assert any("trust_remote_code" in warning for warning in result.warnings)
        
        # Verify trust_remote_code was used
        mock_autoencoder.from_pretrained.assert_called_with(
            str(temp_dir), 
            trust_remote_code=True
        )
    
    @patch('vae_compatibility_handler.AutoencoderKL')
    def test_load_vae_3d_fallback(self, mock_autoencoder, handler, temp_dir):
        """Test 3D VAE loading with fallback to standard method"""
        # Mock trust_remote_code failure, then standard success
        mock_vae = Mock()
        mock_autoencoder.from_pretrained.side_effect = [
            Exception("trust_remote_code failed"),  # First call fails
            mock_vae  # Second call succeeds
        ]
        
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, depth=16, is_3d=True),
            loading_strategy="custom"
        )
        
        result = handler.load_vae_with_compatibility(temp_dir, compatibility_result)
        
        assert result.success
        assert result.loading_strategy_used == "custom_fallback"
        assert result.fallback_used
        assert len(result.warnings) > 0
        assert any("compatibility issues" in warning for warning in result.warnings)
    
    @patch('vae_compatibility_handler.AutoencoderKL')
    def test_load_vae_complete_failure(self, mock_autoencoder, handler, temp_dir):
        """Test complete VAE loading failure"""
        # Mock complete loading failure
        mock_autoencoder.from_pretrained.side_effect = Exception("Loading failed")
        
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, is_3d=False),
            loading_strategy="standard"
        )
        
        result = handler.load_vae_with_compatibility(temp_dir, compatibility_result)
        
        assert not result.success
        assert len(result.errors) > 0
        assert any("Loading failed" in error for error in result.errors)
    
    def test_get_vae_error_guidance_shape_mismatch(self, handler):
        """Test error guidance for shape mismatch issues"""
        compatibility_result = VAECompatibilityResult(
            is_compatible=False,
            detected_dimensions=VAEDimensions(4, 384, 384, is_3d=False),
            compatibility_issues=["Detected problematic VAE shape (4, 384, 384), may need reshaping"]
        )
        
        guidance = handler.get_vae_error_guidance(compatibility_result)
        
        assert len(guidance) > 0
        assert any("Shape Mismatch" in line for line in guidance)
        assert any("384x384 instead of 64x64" in line for line in guidance)
        assert any("trust_remote_code=True" in line for line in guidance)
    
    def test_get_vae_error_guidance_3d_vae(self, handler):
        """Test error guidance for 3D VAE issues"""
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, depth=16, is_3d=True),
            loading_strategy="custom"
        )
        
        guidance = handler.get_vae_error_guidance(compatibility_result)
        
        assert any("3D VAE Detected" in line for line in guidance)
        assert any("video generation" in line for line in guidance)
        assert any("WanPipeline" in line for line in guidance)
        assert any("8GB+ recommended" in line for line in guidance)
    
    def test_get_vae_error_guidance_fallback_used(self, handler):
        """Test error guidance when fallback loading was used"""
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, is_3d=False),
            loading_strategy="standard"
        )
        
        loading_result = VAELoadingResult(
            success=True,
            loading_strategy_used="custom_fallback",
            fallback_used=True,
            warnings=["Fallback loading used"]
        )
        
        guidance = handler.get_vae_error_guidance(compatibility_result, loading_result)
        
        assert any("Fallback Loading Used" in line for line in guidance)
        assert any("reduced functionality" in line for line in guidance)
        assert any("updating dependencies" in line for line in guidance)
    
    def test_get_vae_error_guidance_loading_failure(self, handler):
        """Test error guidance for loading failures"""
        compatibility_result = VAECompatibilityResult(
            is_compatible=True,
            detected_dimensions=VAEDimensions(4, 64, 64, is_3d=False),
            loading_strategy="standard"
        )
        
        loading_result = VAELoadingResult(
            success=False,
            errors=["Failed to load VAE: Model not found"]
        )
        
        guidance = handler.get_vae_error_guidance(compatibility_result, loading_result)
        
        assert any("VAE Loading Failed" in line for line in guidance)
        assert any("Model not found" in line for line in guidance)


class TestVAECompatibilityIntegration:
    """Integration tests for VAE compatibility system"""
    
    @pytest.fixture
    def handler(self):
        """Create VAE compatibility handler"""
        return create_vae_compatibility_handler()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def create_mock_vae_model(self, temp_dir: Path, config_data: dict, 
                             weights_data: dict = None) -> Path:
        """Create a complete mock VAE model directory"""
        vae_dir = temp_dir / "vae"
        vae_dir.mkdir()
        
        # Create config.json
        config_path = vae_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Create weights file if provided
        if weights_data:
            weights_path = vae_dir / "pytorch_model.bin"
            torch.save(weights_data, weights_path)
        
        return vae_dir
    
    def test_end_to_end_2d_vae_workflow(self, handler, temp_dir):
        """Test complete workflow for 2D VAE"""
        # Create mock 2D VAE
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64
        }
        
        weights_data = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3)
        }
        
        vae_dir = self.create_mock_vae_model(temp_dir, config_data, weights_data)
        
        # Step 1: Detect architecture
        config_path = vae_dir / "config.json"
        compatibility_result = handler.detect_vae_architecture(config_path)
        
        assert compatibility_result.is_compatible
        assert not compatibility_result.detected_dimensions.is_3d
        assert compatibility_result.loading_strategy == "standard"
        
        # Step 2: Validate weights
        weights_path = vae_dir / "pytorch_model.bin"
        validation_result = handler.validate_vae_weights(
            weights_path, 
            compatibility_result.detected_dimensions
        )
        
        assert validation_result.is_compatible
        assert validation_result.loading_strategy == "standard"
        
        # Step 3: Get guidance (should be minimal for compatible VAE)
        guidance = handler.get_vae_error_guidance(compatibility_result, None)
        
        # Should have minimal guidance for compatible VAE
        assert len(guidance) == 0 or all("Issue" not in line for line in guidance)
    
    def test_end_to_end_3d_vae_workflow(self, handler, temp_dir):
        """Test complete workflow for 3D VAE"""
        # Create mock 3D VAE
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": [16, 64, 64],
            "temporal_layers": True,
            "temporal_depth": 16
        }
        
        weights_data = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3, 3),  # 3D conv
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3, 3),
            "temporal_conv.weight": torch.randn(64, 64, 1, 3, 3)
        }
        
        vae_dir = self.create_mock_vae_model(temp_dir, config_data, weights_data)
        
        # Step 1: Detect architecture
        config_path = vae_dir / "config.json"
        compatibility_result = handler.detect_vae_architecture(config_path)
        
        assert compatibility_result.is_compatible
        assert compatibility_result.detected_dimensions.is_3d
        assert compatibility_result.detected_dimensions.depth == 16
        
        # Step 2: Validate weights
        weights_path = vae_dir / "pytorch_model.bin"
        validation_result = handler.validate_vae_weights(
            weights_path, 
            compatibility_result.detected_dimensions
        )
        
        assert validation_result.detected_dimensions.is_3d
        
        # Step 3: Get guidance for 3D VAE
        guidance = handler.get_vae_error_guidance(compatibility_result, None)
        
        assert any("3D VAE Detected" in line for line in guidance)
        assert any("WanPipeline" in line for line in guidance)
    
    def test_end_to_end_problematic_vae_workflow(self, handler, temp_dir):
        """Test complete workflow for problematic VAE (384x384)"""
        # Create mock problematic VAE
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384  # Problematic size
        }
        
        weights_data = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3)
        }
        
        vae_dir = self.create_mock_vae_model(temp_dir, config_data, weights_data)
        
        # Step 1: Detect architecture
        config_path = vae_dir / "config.json"
        compatibility_result = handler.detect_vae_architecture(config_path)
        
        assert compatibility_result.detected_dimensions.height == 384
        assert compatibility_result.loading_strategy == "reshape"
        assert len(compatibility_result.compatibility_issues) > 0
        
        # Step 2: Get guidance for problematic VAE
        guidance = handler.get_vae_error_guidance(compatibility_result, None)
        
        assert any("Shape Mismatch" in line for line in guidance)
        assert any("384x384 instead of 64x64" in line for line in guidance)


def test_create_vae_compatibility_handler():
    """Test factory function"""
    handler = create_vae_compatibility_handler()
    assert isinstance(handler, VAECompatibilityHandler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])