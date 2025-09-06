"""
Unit tests for the architecture detection system.

Tests cover various Wan model variants, standard SD models, edge cases,
and error handling scenarios.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from architecture_detector import (
    ArchitectureDetector,
    ArchitectureType,
    VAEType,
    ArchitectureSignature,
    ModelArchitecture,
    ComponentInfo,
    CompatibilityReport,
    ModelRequirements
)


class TestArchitectureSignature:
    """Test ArchitectureSignature functionality."""
    
    def test_is_wan_architecture_with_transformer_2(self):
        """Test Wan detection with transformer_2."""
        signature = ArchitectureSignature(has_transformer_2=True)
        assert signature.is_wan_architecture()
    
    def test_is_wan_architecture_with_boundary_ratio(self):
        """Test Wan detection with boundary_ratio."""
        signature = ArchitectureSignature(has_boundary_ratio=True)
        assert signature.is_wan_architecture()
    
    def test_is_wan_architecture_with_3d_vae(self):
        """Test Wan detection with 3D VAE."""
        signature = ArchitectureSignature(vae_dimensions=3)
        assert signature.is_wan_architecture()
    
    def test_is_wan_architecture_with_wan_pipeline(self):
        """Test Wan detection with WanPipeline class."""
        signature = ArchitectureSignature(pipeline_class="WanPipeline")
        assert signature.is_wan_architecture()
    
    def test_is_wan_architecture_with_wan_components(self):
        """Test Wan detection with Wan component classes."""
        signature = ArchitectureSignature(
            component_classes={"transformer": "WanTransformer"}
        )
        assert signature.is_wan_architecture()
    
    def test_is_not_wan_architecture(self):
        """Test non-Wan model detection."""
        signature = ArchitectureSignature(
            vae_dimensions=2,
            component_classes={"unet": "UNet2DConditionModel"}
        )
        assert not signature.is_wan_architecture()
    
    def test_get_architecture_type_wan_t2v(self):
        """Test T2V architecture type detection."""
        signature = ArchitectureSignature(
            has_transformer_2=True,
            vae_dimensions=3
        )
        assert signature.get_architecture_type() == ArchitectureType.WAN_T2V
    
    def test_get_architecture_type_wan_t2i(self):
        """Test T2I architecture type detection."""
        signature = ArchitectureSignature(
            has_transformer=True,
            has_transformer_2=False,
            vae_dimensions=2,
            pipeline_class="WanT2IPipeline"
        )
        assert signature.get_architecture_type() == ArchitectureType.WAN_T2I
    
    def test_get_architecture_type_wan_i2v(self):
        """Test I2V architecture type detection."""
        signature = ArchitectureSignature(
            pipeline_class="WanI2VPipeline",
            vae_dimensions=3
        )
        assert signature.get_architecture_type() == ArchitectureType.WAN_I2V
    
    def test_get_architecture_type_stable_diffusion(self):
        """Test Stable Diffusion architecture type detection."""
        signature = ArchitectureSignature(
            component_classes={"unet": "StableDiffusionUNet"}
        )
        assert signature.get_architecture_type() == ArchitectureType.STABLE_DIFFUSION
    
    def test_get_architecture_type_unknown(self):
        """Test unknown architecture type detection."""
        signature = ArchitectureSignature()
        assert signature.get_architecture_type() == ArchitectureType.UNKNOWN


class TestComponentInfo:
    """Test ComponentInfo validation."""
    
    def test_valid_component_info(self):
        """Test valid component info creation."""
        component = ComponentInfo(
            class_name="TestClass",
            config_path="/path/to/config.json",
            weight_path="/path/to/weights"
        )
        assert component.class_name == "TestClass"
        assert not component.is_custom
        assert component.dependencies == []
    
    def test_component_info_empty_class_name(self):
        """Test component info with empty class name."""
        with pytest.raises(ValueError, match="class_name cannot be empty"):
            ComponentInfo(
                class_name="",
                config_path="/path/to/config.json",
                weight_path="/path/to/weights"
            )

        assert True  # TODO: Add proper assertion
    
    def test_component_info_no_paths(self):
        """Test component info without config or weight paths."""
        with pytest.raises(ValueError, match="must have either config_path or weight_path"):
            ComponentInfo(
                class_name="TestClass",
                config_path="",
                weight_path=""
            )


        assert True  # TODO: Add proper assertion

class TestArchitectureDetector:
    """Test ArchitectureDetector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ArchitectureDetector()
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model"
        self.model_path.mkdir(parents=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_model_index(self, content: dict):
        """Helper to create model_index.json."""
        model_index_path = self.model_path / "model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump(content, f)
    
    def create_component_config(self, component_name: str, config: dict):
        """Helper to create component config."""
        component_path = self.model_path / component_name
        component_path.mkdir(exist_ok=True)
        config_path = component_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    def test_detect_model_architecture_nonexistent_path(self):
        """Test detection with nonexistent model path."""
        with pytest.raises(FileNotFoundError):
            self.detector.detect_model_architecture("/nonexistent/path")

        assert True  # TODO: Add proper assertion
    
    def test_detect_wan_t2v_model(self):
        """Test detection of Wan T2V model."""
        # Create Wan T2V model structure
        self.create_model_index({
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer_2": ["diffusers", "WanTransformer2D"],
            "vae": ["diffusers", "AutoencoderKLTemporalDecoder"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "boundary_ratio": 0.5
        })
        
        # Create 3D VAE config
        self.create_component_config("vae", {
            "_class_name": "AutoencoderKLTemporalDecoder",
            "latent_channels": 16,
            "temporal_compression_ratio": 4
        })
        
        self.create_component_config("transformer_2", {
            "_class_name": "WanTransformer2D"
        })
        
        self.create_component_config("scheduler", {
            "_class_name": "DDIMScheduler"
        })
        
        architecture = self.detector.detect_model_architecture(self.model_path)
        
        assert architecture.architecture_type == ArchitectureType.WAN_T2V
        assert architecture.signature.has_transformer_2
        assert architecture.signature.has_boundary_ratio
        assert architecture.signature.vae_dimensions == 3
        assert architecture.requirements.requires_trust_remote_code
        assert architecture.requirements.min_vram_mb == 8192
        assert "text_to_video" in architecture.capabilities
    
    def test_detect_wan_t2i_model(self):
        """Test detection of Wan T2I model."""
        self.create_model_index({
            "_class_name": "WanT2IPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "WanTransformer"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "DDIMScheduler"]
        })
        
        self.create_component_config("vae", {
            "_class_name": "AutoencoderKL",
            "latent_channels": 4
        })
        
        self.create_component_config("transformer", {
            "_class_name": "WanTransformer"
        })
        
        architecture = self.detector.detect_model_architecture(self.model_path)
        
        assert architecture.architecture_type == ArchitectureType.WAN_T2I
        assert architecture.signature.has_transformer
        assert not architecture.signature.has_transformer_2
        assert architecture.signature.vae_dimensions == 2
        assert "text_to_image" in architecture.capabilities
    
    def test_detect_stable_diffusion_model(self):
        """Test detection of standard Stable Diffusion model."""
        self.create_model_index({
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.21.0",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "PNDMScheduler"]
        })
        
        self.create_component_config("vae", {
            "_class_name": "AutoencoderKL",
            "latent_channels": 4
        })
        
        self.create_component_config("unet", {
            "_class_name": "UNet2DConditionModel"
        })
        
        architecture = self.detector.detect_model_architecture(self.model_path)
        
        assert architecture.architecture_type == ArchitectureType.STABLE_DIFFUSION
        assert not architecture.signature.is_wan_architecture()
        assert architecture.signature.vae_dimensions == 2
        assert not architecture.requirements.requires_trust_remote_code
        assert architecture.requirements.min_vram_mb == 4096
    
    def test_analyze_model_index_missing_file(self):
        """Test model index analysis with missing file."""
        signature = self.detector.analyze_model_index(self.model_path)
        
        assert not signature.has_transformer
        assert not signature.has_transformer_2
        assert not signature.has_boundary_ratio
        assert signature.pipeline_class is None
    
    def test_analyze_model_index_invalid_json(self):
        """Test model index analysis with invalid JSON."""
        model_index_path = self.model_path / "model_index.json"
        with open(model_index_path, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(ValueError, match="Invalid model_index.json"):
            self.detector.analyze_model_index(self.model_path)

        assert True  # TODO: Add proper assertion
    
    def test_check_vae_dimensions_3d(self):
        """Test 3D VAE dimension detection."""
        vae_path = self.model_path / "vae"
        vae_path.mkdir()
        
        config = {
            "_class_name": "AutoencoderKLTemporalDecoder",
            "latent_channels": 16,
            "temporal_compression_ratio": 4,
            "out_channels": 3
        }
        
        with open(vae_path / "config.json", 'w') as f:
            json.dump(config, f)
        
        vae_type = self.detector.check_vae_dimensions(vae_path)
        assert vae_type == VAEType.VAE_3D
    
    def test_check_vae_dimensions_2d(self):
        """Test 2D VAE dimension detection."""
        vae_path = self.model_path / "vae"
        vae_path.mkdir()
        
        config = {
            "_class_name": "AutoencoderKL",
            "latent_channels": 4,
            "out_channels": 3
        }
        
        with open(vae_path / "config.json", 'w') as f:
            json.dump(config, f)
        
        vae_type = self.detector.check_vae_dimensions(vae_path)
        assert vae_type == VAEType.VAE_2D
    
    def test_check_vae_dimensions_missing_config(self):
        """Test VAE dimension detection with missing config."""
        vae_path = self.model_path / "vae"
        vae_path.mkdir()
        
        vae_type = self.detector.check_vae_dimensions(vae_path)
        assert vae_type == VAEType.UNKNOWN
    
    def test_validate_component_compatibility_wan_model(self):
        """Test component compatibility validation for Wan model."""
        components = {
            "transformer_2": ComponentInfo(
                class_name="WanTransformer2D",
                config_path="/path/config.json",
                weight_path="/path/weights",
                is_custom=True
            ),
            "vae": ComponentInfo(
                class_name="AutoencoderKLTemporalDecoder",
                config_path="/path/config.json",
                weight_path="/path/weights",
                is_custom=True
            ),
            "scheduler": ComponentInfo(
                class_name="DDIMScheduler",
                config_path="/path/config.json",
                weight_path="/path/weights"
            )
        }
        
        report = self.detector.validate_component_compatibility(components)
        
        assert report.is_compatible
        assert "transformer_2" in report.compatible_components
        assert "Use WanPipeline for optimal performance" in report.recommendations
        assert any("Custom component detected" in warning for warning in report.warnings)
    
    def test_validate_component_compatibility_sd_model(self):
        """Test component compatibility validation for SD model."""
        components = {
            "unet": ComponentInfo(
                class_name="UNet2DConditionModel",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "text_encoder": ComponentInfo(
                class_name="CLIPTextModel",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "tokenizer": ComponentInfo(
                class_name="CLIPTokenizer",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "vae": ComponentInfo(
                class_name="AutoencoderKL",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "scheduler": ComponentInfo(
                class_name="PNDMScheduler",
                config_path="/path/config.json",
                weight_path="/path/weights"
            )
        }
        
        report = self.detector.validate_component_compatibility(components)
        
        assert report.is_compatible
        assert "unet" in report.compatible_components
        assert "text_encoder" in report.compatible_components
        assert "tokenizer" in report.compatible_components
        assert "Standard StableDiffusionPipeline compatible" in report.recommendations
    
    def test_validate_component_compatibility_missing_required(self):
        """Test component compatibility with missing required components."""
        components = {
            "transformer": ComponentInfo(
                class_name="WanTransformer",
                config_path="/path/config.json",
                weight_path="/path/weights"
            )
            # Missing scheduler and vae
        }
        
        report = self.detector.validate_component_compatibility(components)
        
        assert not report.is_compatible
        assert "scheduler" in report.missing_components
        assert "vae" in report.missing_components
        assert any("Missing required components" in rec for rec in report.recommendations)
    
    def test_validate_component_compatibility_hybrid_model(self):
        """Test component compatibility with hybrid Wan/SD model."""
        components = {
            "transformer": ComponentInfo(
                class_name="WanTransformer",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "unet": ComponentInfo(
                class_name="UNet2DConditionModel",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "vae": ComponentInfo(
                class_name="AutoencoderKL",
                config_path="/path/config.json",
                weight_path="/path/weights"
            ),
            "scheduler": ComponentInfo(
                class_name="DDIMScheduler",
                config_path="/path/config.json",
                weight_path="/path/weights"
            )
        }
        
        report = self.detector.validate_component_compatibility(components)
        
        assert report.is_compatible
        assert any("hybrid model" in warning for warning in report.warnings)


class TestModelArchitecture:
    """Test ModelArchitecture functionality."""
    
    def test_model_architecture_default_capabilities_wan_t2v(self):
        """Test default capabilities for Wan T2V model."""
        architecture = ModelArchitecture(architecture_type=ArchitectureType.WAN_T2V)
        assert "text_to_video" in architecture.capabilities
    
    def test_model_architecture_default_capabilities_wan_t2i(self):
        """Test default capabilities for Wan T2I model."""
        architecture = ModelArchitecture(architecture_type=ArchitectureType.WAN_T2I)
        assert "text_to_image" in architecture.capabilities
    
    def test_model_architecture_default_capabilities_wan_i2v(self):
        """Test default capabilities for Wan I2V model."""
        architecture = ModelArchitecture(architecture_type=ArchitectureType.WAN_I2V)
        assert "image_to_video" in architecture.capabilities
    
    def test_model_architecture_default_capabilities_sd(self):
        """Test default capabilities for SD model."""
        architecture = ModelArchitecture(architecture_type=ArchitectureType.STABLE_DIFFUSION)
        assert "text_to_image" in architecture.capabilities
    
    def test_model_architecture_custom_capabilities(self):
        """Test custom capabilities override."""
        custom_capabilities = ["custom_capability"]
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.WAN_T2V,
            capabilities=custom_capabilities
        )
        assert architecture.capabilities == custom_capabilities


class TestIntegration:
    """Integration tests for the complete architecture detection system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ArchitectureDetector()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_complete_wan_model(self, model_name: str) -> Path:
        """Create a complete Wan model structure for testing."""
        model_path = Path(self.temp_dir) / model_name
        model_path.mkdir(parents=True)
        
        # Create model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer_2": ["diffusers", "WanTransformer2D"],
            "vae": ["diffusers", "AutoencoderKLTemporalDecoder"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "boundary_ratio": 0.5
        }
        
        with open(model_path / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        # Create component configs
        components = {
            "transformer_2": {
                "_class_name": "WanTransformer2D",
                "num_attention_heads": 16,
                "attention_head_dim": 88
            },
            "vae": {
                "_class_name": "AutoencoderKLTemporalDecoder",
                "latent_channels": 16,
                "temporal_compression_ratio": 4,
                "out_channels": 3
            },
            "scheduler": {
                "_class_name": "DDIMScheduler",
                "num_train_timesteps": 1000
            }
        }
        
        for comp_name, config in components.items():
            comp_path = model_path / comp_name
            comp_path.mkdir()
            with open(comp_path / "config.json", 'w') as f:
                json.dump(config, f)
        
        return model_path
    
    def test_complete_wan_model_detection(self):
        """Test complete Wan model detection workflow."""
        model_path = self.create_complete_wan_model("wan_t2v_test")
        
        architecture = self.detector.detect_model_architecture(model_path)
        
        # Verify architecture detection
        assert architecture.architecture_type == ArchitectureType.WAN_T2V
        assert architecture.signature.is_wan_architecture()
        assert architecture.signature.has_transformer_2
        assert architecture.signature.has_boundary_ratio
        assert architecture.signature.vae_dimensions == 3
        
        # Verify requirements
        assert architecture.requirements.requires_trust_remote_code
        assert architecture.requirements.min_vram_mb >= 8192
        assert "transformers>=4.25.0" in architecture.requirements.required_dependencies
        
        # Verify components
        assert "transformer_2" in architecture.components
        assert "vae" in architecture.components
        assert "scheduler" in architecture.components
        
        # Verify component compatibility
        report = self.detector.validate_component_compatibility(architecture.components)
        assert report.is_compatible
        
        # Verify capabilities
        assert "text_to_video" in architecture.capabilities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])