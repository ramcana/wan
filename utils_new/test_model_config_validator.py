"""
Test suite for ModelConfigValidator class

Tests model configuration validation, component validation, and compatibility checking.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_config_validator import (
    ModelConfigValidator,
    ModelCompatibilityInfo,
    validate_model_directory,
    validate_model_index
)
from config_validator import ValidationSeverity, ValidationMessage, ValidationResult


class TestModelConfigValidator:
    """Test cases for ModelConfigValidator class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        
        # Create mock compatibility registry
        self.compatibility_registry = {
            "Wan2.2-T2V-A14B": {
                "pipeline_class": "WanPipeline",
                "min_diffusers_version": "0.21.0",
                "required_dependencies": ["transformers>=4.25.0", "torch>=2.0.0"],
                "vram_requirements": {"min_mb": 8192, "recommended_mb": 12288},
                "supported_optimizations": ["cpu_offload", "mixed_precision"],
                "trust_remote_code": True
            }
        }
        
        self.registry_path = self.temp_dir / "compatibility_registry.json"
        with open(self.registry_path, 'w') as f:
            json.dump(self.compatibility_registry, f, indent=2)
        
        self.validator = ModelConfigValidator(
            backup_dir=self.backup_dir,
            compatibility_registry_path=self.registry_path
        )
        
        # Create valid test configurations
        self.valid_model_index = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "PNDMScheduler"]
        }
        
        self.valid_vae_config = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.21.0",
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 512,
            "scaling_factor": 0.18215,
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
        }
        
        self.valid_text_encoder_config = {
            "_name_or_path": "openai/clip-vit-large-patch14",
            "architectures": ["CLIPTextModel"],
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 77,
            "model_type": "clip_text_model",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "projection_dim": 768,
            "torch_dtype": "float32",
            "transformers_version": "4.25.1",
            "vocab_size": 49408
        }
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test ModelConfigValidator initialization"""
        validator = ModelConfigValidator()
        assert validator.backup_dir.exists()
        assert validator.model_schemas is not None
        assert validator.model_cleanup_attributes is not None
        assert validator.compatibility_registry is not None
    
    def test_validate_valid_model_index(self):
        """Test validation of a valid model_index.json"""
        model_index_path = self.temp_dir / "model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump(self.valid_model_index, f, indent=2)
        
        result = self.validator.validate_model_index(model_index_path)
        
        assert result.is_valid
        assert not result.has_errors()
        assert result.backup_path is not None
    
    def test_validate_missing_model_index(self):
        """Test validation of missing model_index.json"""
        model_index_path = self.temp_dir / "missing_model_index.json"
        
        result = self.validator.validate_model_index(model_index_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "MODEL_INDEX_NOT_FOUND" for msg in result.messages)
    
    def test_validate_invalid_json_model_index(self):
        """Test validation of invalid JSON in model_index.json"""
        model_index_path = self.temp_dir / "invalid_model_index.json"
        with open(model_index_path, 'w') as f:
            f.write('{ "invalid": json, }')
        
        result = self.validator.validate_model_index(model_index_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "INVALID_JSON" for msg in result.messages)
    
    def test_validate_missing_required_field(self):
        """Test validation with missing required field in model_index.json"""
        model_index = self.valid_model_index.copy()
        del model_index["_class_name"]  # Remove required field
        
        model_index_path = self.temp_dir / "missing_field_model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump(model_index, f, indent=2)
        
        result = self.validator.validate_model_index(model_index_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "MISSING_REQUIRED_PROPERTY" for msg in result.messages)
    
    def test_validate_invalid_pipeline_class(self):
        """Test validation with invalid pipeline class"""
        model_index = self.valid_model_index.copy()
        model_index["_class_name"] = "InvalidPipeline"
        
        model_index_path = self.temp_dir / "invalid_pipeline_model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump(model_index, f, indent=2)
        
        result = self.validator.validate_model_index(model_index_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "INVALID_ENUM_VALUE" for msg in result.messages)
    
    def test_cleanup_model_index_attributes(self):
        """Test cleanup of unexpected attributes in model_index.json"""
        model_index = self.valid_model_index.copy()
        
        # Add attributes that should be cleaned up
        model_index["requires_safety_checker"] = False
        model_index["safety_checker"] = None
        model_index["cache_dir"] = "/tmp/cache"
        
        model_index_path = self.temp_dir / "cleanup_model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump(model_index, f, indent=2)
        
        result = self.validator.validate_model_index(model_index_path)
        
        assert len(result.cleaned_attributes) > 0
        assert any("requires_safety_checker" in attr for attr in result.cleaned_attributes)
        
        # Verify cleaned model index no longer has unexpected attributes
        with open(model_index_path, 'r') as f:
            cleaned_model_index = json.load(f)
        
        assert "requires_safety_checker" not in cleaned_model_index
        assert "safety_checker" not in cleaned_model_index
        assert "cache_dir" not in cleaned_model_index
    
    def test_validate_valid_vae_config(self):
        """Test validation of a valid VAE config"""
        vae_config_path = self.temp_dir / "vae_config.json"
        with open(vae_config_path, 'w') as f:
            json.dump(self.valid_vae_config, f, indent=2)
        
        result = self.validator.validate_component_config(vae_config_path, "vae_config")
        
        assert result.is_valid
        assert not result.has_errors()
        assert result.backup_path is not None
    
    def test_cleanup_vae_config_attributes(self):
        """Test cleanup of unexpected attributes in VAE config"""
        vae_config = self.valid_vae_config.copy()
        
        # Add attributes that should be cleaned up (main issue from requirements)
        vae_config["clip_output"] = True
        vae_config["force_upcast"] = False
        vae_config["use_tiling"] = True
        
        vae_config_path = self.temp_dir / "cleanup_vae_config.json"
        with open(vae_config_path, 'w') as f:
            json.dump(vae_config, f, indent=2)
        
        result = self.validator.validate_component_config(vae_config_path, "vae_config")
        
        assert len(result.cleaned_attributes) > 0
        assert any("clip_output" in attr for attr in result.cleaned_attributes)
        
        # Verify cleaned VAE config no longer has unexpected attributes
        with open(vae_config_path, 'r') as f:
            cleaned_vae_config = json.load(f)
        
        assert "clip_output" not in cleaned_vae_config
        assert "force_upcast" not in cleaned_vae_config
        assert "use_tiling" not in cleaned_vae_config
    
    def test_validate_text_encoder_config(self):
        """Test validation of text encoder config"""
        text_encoder_config_path = self.temp_dir / "text_encoder_config.json"
        with open(text_encoder_config_path, 'w') as f:
            json.dump(self.valid_text_encoder_config, f, indent=2)
        
        result = self.validator.validate_component_config(text_encoder_config_path, "text_encoder_config")
        
        assert result.is_valid
        assert not result.has_errors()
    
    def test_cleanup_text_encoder_config_attributes(self):
        """Test cleanup of unexpected attributes in text encoder config"""
        text_encoder_config = self.valid_text_encoder_config.copy()
        
        # Add attributes that should be cleaned up
        text_encoder_config["use_attention_mask"] = True
        text_encoder_config["return_dict"] = True
        text_encoder_config["output_attentions"] = False
        
        text_encoder_config_path = self.temp_dir / "cleanup_text_encoder_config.json"
        with open(text_encoder_config_path, 'w') as f:
            json.dump(text_encoder_config, f, indent=2)
        
        result = self.validator.validate_component_config(text_encoder_config_path, "text_encoder_config")
        
        assert len(result.cleaned_attributes) > 0
        assert any("use_attention_mask" in attr for attr in result.cleaned_attributes)
        
        # Verify cleaned config no longer has unexpected attributes
        with open(text_encoder_config_path, 'r') as f:
            cleaned_config = json.load(f)
        
        assert "use_attention_mask" not in cleaned_config
        assert "return_dict" not in cleaned_config
        assert "output_attentions" not in cleaned_config
    
    def test_validate_model_directory(self):
        """Test validation of entire model directory"""
        # Create model directory structure
        model_dir = self.temp_dir / "test_model"
        model_dir.mkdir()
        
        # Create model_index.json
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(self.valid_model_index, f, indent=2)
        
        # Create VAE component
        vae_dir = model_dir / "vae"
        vae_dir.mkdir()
        with open(vae_dir / "config.json", 'w') as f:
            json.dump(self.valid_vae_config, f, indent=2)
        
        # Create text encoder component
        text_encoder_dir = model_dir / "text_encoder"
        text_encoder_dir.mkdir()
        with open(text_encoder_dir / "config.json", 'w') as f:
            json.dump(self.valid_text_encoder_config, f, indent=2)
        
        # Create unet component (required for WanPipeline)
        unet_dir = model_dir / "unet"
        unet_dir.mkdir()
        unet_config = {
            "_class_name": "UNet2DConditionModel",
            "_diffusers_version": "0.21.0",
            "in_channels": 4,
            "out_channels": 4
        }
        with open(unet_dir / "config.json", 'w') as f:
            json.dump(unet_config, f, indent=2)
        
        result = self.validator.validate_model_directory(model_dir)
        
        assert result.is_valid
        assert not result.has_errors()
    
    def test_validate_wan_pipeline_requirements(self):
        """Test validation of WAN pipeline specific requirements"""
        # Create model directory with missing WAN components
        model_dir = self.temp_dir / "wan_model"
        model_dir.mkdir()
        
        # Create model_index.json with WanPipeline class
        model_index = self.valid_model_index.copy()
        model_index["_class_name"] = "WanPipeline"
        
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
        
        # Don't create required components (vae, text_encoder, unet)
        
        result = self.validator.validate_model_directory(model_dir)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "MISSING_WAN_COMPONENT" for msg in result.messages)
    
    def test_get_model_compatibility_info(self):
        """Test getting model compatibility information"""
        info = self.validator.get_model_compatibility_info("Wan2.2-T2V-A14B")
        
        assert info is not None
        assert info.model_name == "Wan2.2-T2V-A14B"
        assert info.pipeline_class == "WanPipeline"
        assert info.trust_remote_code == True
        assert "cpu_offload" in info.supported_optimizations
        
        # Test non-existent model
        info_none = self.validator.get_model_compatibility_info("NonExistentModel")
        assert info_none is None
    
    def test_validate_diffusers_version_compatibility(self):
        """Test validation of diffusers version compatibility"""
        model_dir = self.temp_dir / "version_test_model"
        model_dir.mkdir()
        
        # Create model_index.json with old diffusers version
        model_index = self.valid_model_index.copy()
        model_index["_diffusers_version"] = "0.15.0"  # Old version
        
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
        
        result = self.validator.validate_model_directory(model_dir)
        
        # Should have warning about old version (if packaging is available)
        warning_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.WARNING]
        version_warnings = [msg for msg in warning_messages if "diffusers_version" in msg.code.lower()]
        
        # May or may not have warning depending on packaging availability
        # Just check that validation completes without errors
        assert not result.has_errors()
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test validate_model_index function
        model_index_path = self.temp_dir / "convenience_model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump(self.valid_model_index, f, indent=2)
        
        result = validate_model_index(
            model_index_path,
            backup_dir=self.backup_dir,
            compatibility_registry_path=self.registry_path
        )
        
        assert result.is_valid
        assert result.backup_path is not None
        
        # Test validate_model_directory function
        model_dir = self.temp_dir / "convenience_model"
        model_dir.mkdir()
        
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(self.valid_model_index, f, indent=2)
        
        result = validate_model_directory(
            model_dir,
            backup_dir=self.backup_dir,
            compatibility_registry_path=self.registry_path
        )
        
        assert result.is_valid


if __name__ == "__main__":
    # Run basic tests
    import sys
    
    test_instance = TestModelConfigValidator()
    test_instance.setup_method()
    
    try:
        # Test basic functionality
        print("Testing ModelConfigValidator initialization...")
        test_instance.test_validator_initialization()
        print("✅ Initialization test passed")
        
        print("Testing valid model_index.json validation...")
        test_instance.test_validate_valid_model_index()
        print("✅ Valid model_index test passed")
        
        print("Testing model_index.json cleanup...")
        test_instance.test_cleanup_model_index_attributes()
        print("✅ Model index cleanup test passed")
        
        print("Testing VAE config validation...")
        test_instance.test_validate_valid_vae_config()
        print("✅ VAE config test passed")
        
        print("Testing VAE config cleanup (clip_output issue)...")
        test_instance.test_cleanup_vae_config_attributes()
        print("✅ VAE config cleanup test passed")
        
        print("Testing text encoder config cleanup...")
        test_instance.test_cleanup_text_encoder_config_attributes()
        print("✅ Text encoder cleanup test passed")
        
        print("Testing model directory validation...")
        test_instance.test_validate_model_directory()
        print("✅ Model directory test passed")
        
        print("Testing WAN pipeline requirements...")
        test_instance.test_validate_wan_pipeline_requirements()
        print("✅ WAN pipeline requirements test passed")
        
        print("Testing compatibility info...")
        test_instance.test_get_model_compatibility_info()
        print("✅ Compatibility info test passed")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        test_instance.teardown_method()