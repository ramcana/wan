"""
Test suite for ConfigValidator class

Tests configuration validation, schema checking, attribute cleanup, and backup functionality.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from config_validator import (
    ConfigValidator,
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    CleanupResult,
    validate_config_file,
    format_validation_result
)


class TestConfigValidator:
    """Test cases for ConfigValidator class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        self.validator = ConfigValidator(backup_dir=self.backup_dir)
        
        # Create a valid test configuration
        self.valid_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "stats_refresh_interval": 5
            },
            "directories": {
                "output_directory": "outputs",
                "models_directory": "models",
                "loras_directory": "loras"
            },
            "generation": {
                "default_resolution": "1280x720",
                "default_steps": 50,
                "default_duration": 4,
                "default_fps": 24,
                "max_prompt_length": 500,
                "supported_resolutions": ["854x480", "1280x720", "1920x1080"]
            },
            "models": {
                "t2v_model": "Wan2.2-T2V-A14B",
                "i2v_model": "Wan2.2-I2V-A14B",
                "ti2v_model": "Wan2.2-TI2V-5B"
            },
            "optimization": {
                "default_quantization": "bf16",
                "quantization_levels": ["fp16", "bf16", "int8"],
                "vae_tile_size_range": [128, 512],
                "max_vram_usage_gb": 12,
                "enable_offload": True,
                "vae_tile_size": 256
            },
            "ui": {
                "max_file_size_mb": 10,
                "supported_image_formats": ["PNG", "JPG", "JPEG", "WebP"],
                "gallery_thumbnail_size": 256
            },
            "performance": {
                "target_720p_time_minutes": 9,
                "target_1080p_time_minutes": 17,
                "vram_warning_threshold": 0.9,
                "cpu_warning_percent": 95,
                "memory_warning_percent": 85,
                "vram_warning_percent": 90,
                "sample_interval_seconds": 30.0,
                "max_history_samples": 100,
                "cpu_monitoring_enabled": False,
                "disk_io_monitoring_enabled": False,
                "network_monitoring_enabled": False
            },
            "prompt_enhancement": {
                "max_prompt_length": 500,
                "enable_basic_quality": True,
                "enable_vace_detection": True,
                "enable_cinematic_enhancement": True,
                "enable_style_detection": True,
                "max_quality_keywords": 3,
                "max_cinematic_keywords": 3,
                "max_style_keywords": 2
            }
        }
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test ConfigValidator initialization"""
        validator = ConfigValidator()
        assert validator.backup_dir.exists()
        assert validator.expected_schema is not None
        assert validator.cleanup_attributes is not None
        
        # Test custom backup directory
        custom_backup_dir = self.temp_dir / "custom_backups"
        validator_custom = ConfigValidator(backup_dir=custom_backup_dir)
        assert validator_custom.backup_dir == custom_backup_dir
        assert custom_backup_dir.exists()
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration"""
        config_path = self.temp_dir / "valid_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert result.is_valid
        assert not result.has_errors()
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()
    
    def test_validate_missing_file(self):
        """Test validation of non-existent file"""
        config_path = self.temp_dir / "missing_config.json"
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "FILE_NOT_FOUND" for msg in result.messages)
        assert result.backup_path is None
    
    def test_validate_invalid_json(self):
        """Test validation of invalid JSON"""
        config_path = self.temp_dir / "invalid_config.json"
        with open(config_path, 'w') as f:
            f.write('{ "invalid": json, }')
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "INVALID_JSON" for msg in result.messages)
    
    def test_validate_missing_required_section(self):
        """Test validation with missing required section"""
        config = self.valid_config.copy()
        del config["system"]  # Remove required section
        
        config_path = self.temp_dir / "missing_section_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "MISSING_REQUIRED_SECTION" for msg in result.messages)
    
    def test_validate_missing_required_property(self):
        """Test validation with missing required property"""
        config = self.valid_config.copy()
        del config["system"]["default_quantization"]  # Remove required property
        
        config_path = self.temp_dir / "missing_property_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "MISSING_REQUIRED_PROPERTY" for msg in result.messages)
    
    def test_validate_invalid_type(self):
        """Test validation with invalid property type"""
        config = self.valid_config.copy()
        config["system"]["enable_offload"] = "true"  # Should be boolean, not string
        
        config_path = self.temp_dir / "invalid_type_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "INVALID_TYPE" for msg in result.messages)
    
    def test_validate_invalid_enum_value(self):
        """Test validation with invalid enum value"""
        config = self.valid_config.copy()
        config["system"]["default_quantization"] = "invalid_quantization"
        
        config_path = self.temp_dir / "invalid_enum_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "INVALID_ENUM_VALUE" for msg in result.messages)
    
    def test_validate_value_range(self):
        """Test validation of value ranges"""
        config = self.valid_config.copy()
        config["system"]["vae_tile_size"] = 50  # Below minimum of 128
        
        config_path = self.temp_dir / "range_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert result.has_warnings()
        assert any(msg.code == "VALUE_TOO_LOW" for msg in result.messages)
    
    def test_cleanup_unexpected_attributes(self):
        """Test cleanup of unexpected attributes"""
        config = self.valid_config.copy()
        
        # Add unexpected attributes that should be cleaned up
        config["clip_output"] = True  # General cleanup
        config["system"]["clip_output"] = True  # VAE config cleanup
        config["optimization"]["force_upcast"] = False  # Model config cleanup
        
        config_path = self.temp_dir / "cleanup_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert len(result.cleaned_attributes) > 0
        assert any("clip_output" in attr for attr in result.cleaned_attributes)
        
        # Verify cleaned config no longer has unexpected attributes
        with open(config_path, 'r') as f:
            cleaned_config = json.load(f)
        
        assert "clip_output" not in cleaned_config
        assert "clip_output" not in cleaned_config.get("system", {})
        assert "force_upcast" not in cleaned_config.get("optimization", {})
    
    def test_create_backup(self):
        """Test backup creation functionality"""
        config_path = self.temp_dir / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f, indent=2)
        
        backup_path = self.validator.create_backup(config_path)
        
        assert Path(backup_path).exists()
        assert backup_path.endswith(".backup.json")
        assert self.backup_dir.name in backup_path
        
        # Verify backup content matches original
        with open(backup_path, 'r') as f:
            backup_config = json.load(f)
        
        assert backup_config == self.valid_config
    
    def test_validate_pattern_matching(self):
        """Test pattern validation for strings"""
        config = self.valid_config.copy()
        config["generation"]["default_resolution"] = "invalid_resolution"  # Should match \d+x\d+
        
        config_path = self.temp_dir / "pattern_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        result = self.validator.validate_config_file(config_path)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(msg.code == "INVALID_PATTERN" for msg in result.messages)
    
    def test_convenience_function(self):
        """Test convenience function validate_config_file"""
        config_path = self.temp_dir / "convenience_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f, indent=2)
        
        result = validate_config_file(config_path, backup_dir=self.backup_dir)
        
        assert result.is_valid
        assert result.backup_path is not None
    
    def test_format_validation_result(self):
        """Test formatting of validation results"""
        # Create a result with various message types
        messages = [
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="TEST_ERROR",
                message="Test error message",
                field_path="test.field",
                current_value="invalid",
                suggested_value="valid",
                help_text="This is help text"
            ),
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TEST_WARNING",
                message="Test warning message",
                field_path="test.warning"
            )
        ]
        
        result = ValidationResult(
            is_valid=False,
            messages=messages,
            cleaned_attributes=["test.cleaned_attr"],
            backup_path="/path/to/backup.json"
        )
        
        formatted = format_validation_result(result)
        
        assert "❌ Configuration has errors" in formatted
        assert "📁 Backup created" in formatted
        assert "🧹 Cleaned 1 attributes" in formatted
        assert "❌ ERROR" in formatted
        assert "⚠️ WARNING" in formatted
        assert "TEST_ERROR: Test error message" in formatted
        assert "Path: test.field" in formatted
        assert "Current: invalid" in formatted
        assert "Suggested: valid" in formatted
        assert "Help: This is help text" in formatted
    
    def test_validation_result_methods(self):
        """Test ValidationResult helper methods"""
        messages = [
            ValidationMessage(ValidationSeverity.ERROR, "E1", "Error 1", "path1"),
            ValidationMessage(ValidationSeverity.WARNING, "W1", "Warning 1", "path2"),
            ValidationMessage(ValidationSeverity.INFO, "I1", "Info 1", "path3")
        ]
        
        result = ValidationResult(
            is_valid=False,
            messages=messages,
            cleaned_attributes=[]
        )
        
        assert result.has_errors()
        assert result.has_warnings()
        
        # Test valid result
        valid_result = ValidationResult(
            is_valid=True,
            messages=[ValidationMessage(ValidationSeverity.INFO, "I1", "Info", "path")],
            cleaned_attributes=[]
        )
        
        assert not valid_result.has_errors()
        assert not valid_result.has_warnings()


if __name__ == "__main__":
    # Run basic tests
    import sys
    
    test_instance = TestConfigValidator()
    test_instance.setup_method()
    
    try:
        # Test basic functionality
        print("Testing ConfigValidator initialization...")
        test_instance.test_validator_initialization()
        print("✅ Initialization test passed")
        
        print("Testing valid configuration validation...")
        test_instance.test_validate_valid_config()
        print("✅ Valid config test passed")
        
        print("Testing missing file handling...")
        test_instance.test_validate_missing_file()
        print("✅ Missing file test passed")
        
        print("Testing attribute cleanup...")
        test_instance.test_cleanup_unexpected_attributes()
        print("✅ Cleanup test passed")
        
        print("Testing backup creation...")
        test_instance.test_create_backup()
        print("✅ Backup test passed")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        test_instance.teardown_method()