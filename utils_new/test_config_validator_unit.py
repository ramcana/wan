#!/usr/bin/env python3
"""
Unit tests for ConfigValidator component
Tests configuration validation, cleanup, and backup functionality
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from config_validator import (
    ConfigValidator, ValidationResult, ValidationMessage, CleanupResult,
    ValidationSeverity, validate_config_file, format_validation_result
)


class TestConfigValidator(unittest.TestCase):
    """Test cases for ConfigValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_dir = Path(self.temp_dir) / "backups"
        self.validator = ConfigValidator(backup_dir=self.backup_dir)
        
        # Create test config file
        self.test_config_path = Path(self.temp_dir) / "test_config.json"
        self.valid_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256
            },
            "directories": {
                "output_directory": "/path/to/output",
                "models_directory": "/path/to/models"
            },
            "generation": {
                "default_resolution": "512x512",
                "default_steps": 20
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test ConfigValidator initialization"""
        self.assertIsInstance(self.validator, ConfigValidator)
        self.assertTrue(self.backup_dir.exists())
        self.assertIn("system", self.validator.expected_schema)
        self.assertIn("clip_output", self.validator.cleanup_attributes["vae_config"])
    
    def test_validate_valid_config_file(self):
        """Test validation of valid configuration file"""
        with open(self.test_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.backup_path)
        # Should have minimal messages for valid config
        error_messages = [msg for msg in result.messages if msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        self.assertEqual(len(error_messages), 0)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file"""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.json"
        
        result = self.validator.validate_config_file(nonexistent_path)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].severity, ValidationSeverity.CRITICAL)
        self.assertEqual(result.messages[0].code, "FILE_NOT_FOUND")
    
    def test_validate_invalid_json(self):
        """Test validation of file with invalid JSON"""
        invalid_json = '{"system": {"default_quantization": "bf16",}}'  # Trailing comma
        
        with open(self.test_config_path, 'w') as f:
            f.write(invalid_json)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any(msg.code == "INVALID_JSON" for msg in result.messages))
    
    def test_validate_config_with_missing_required_section(self):
        """Test validation of config missing required section"""
        incomplete_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True
            }
            # Missing required "directories" section
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(incomplete_config, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have error for missing required section
        error_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.ERROR]
        self.assertTrue(any("directories" in msg.message.lower() for msg in error_messages))
    
    def test_validate_config_with_invalid_property_type(self):
        """Test validation of config with invalid property type"""
        invalid_config = self.valid_config.copy()
        invalid_config["system"]["enable_offload"] = "true"  # Should be boolean, not string
        
        with open(self.test_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have error for invalid type
        error_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.ERROR]
        self.assertTrue(any("boolean" in msg.message.lower() for msg in error_messages))
    
    def test_validate_config_with_invalid_enum_value(self):
        """Test validation of config with invalid enum value"""
        invalid_config = self.valid_config.copy()
        invalid_config["system"]["default_quantization"] = "invalid_quantization"
        
        with open(self.test_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have error for invalid enum value
        error_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.ERROR]
        self.assertTrue(any("invalid_enum_value" in msg.code.lower() for msg in error_messages))
    
    def test_validate_config_with_out_of_range_value(self):
        """Test validation of config with out-of-range value"""
        invalid_config = self.valid_config.copy()
        invalid_config["system"]["vae_tile_size"] = 2048  # Above maximum of 1024
        
        with open(self.test_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have warning for value too high
        warning_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.WARNING]
        self.assertTrue(any("too high" in msg.message.lower() for msg in warning_messages))
    
    def test_validate_config_with_invalid_pattern(self):
        """Test validation of config with invalid pattern"""
        invalid_config = self.valid_config.copy()
        invalid_config["generation"]["default_resolution"] = "invalid_resolution"  # Should match pattern
        
        with open(self.test_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have error for invalid pattern
        error_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.ERROR]
        self.assertTrue(any("pattern" in msg.message.lower() for msg in error_messages))
    
    def test_cleanup_config_with_unexpected_attributes(self):
        """Test cleanup of config with unexpected attributes"""
        config_with_cleanup = self.valid_config.copy()
        config_with_cleanup["clip_output"] = True  # Should be removed
        config_with_cleanup["force_upcast"] = False  # Should be removed
        config_with_cleanup["system"]["unknown_attribute"] = "value"  # Should be removed
        
        with open(self.test_config_path, 'w') as f:
            json.dump(config_with_cleanup, f)
        
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have cleaned attributes
        self.assertGreater(len(result.cleaned_attributes), 0)
        self.assertTrue(any("clip_output" in attr for attr in result.cleaned_attributes))
        
        # Verify file was actually cleaned
        with open(self.test_config_path, 'r') as f:
            cleaned_config = json.load(f)
        
        self.assertNotIn("clip_output", cleaned_config)
        self.assertNotIn("force_upcast", cleaned_config)
    
    def test_create_backup(self):
        """Test backup creation"""
        with open(self.test_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        backup_path = self.validator.create_backup(self.test_config_path)
        
        self.assertTrue(Path(backup_path).exists())
        
        # Verify backup content matches original
        with open(backup_path, 'r') as f:
            backup_config = json.load(f)
        
        self.assertEqual(backup_config, self.valid_config)
    
    def test_validate_property_string_type(self):
        """Test property validation for string type"""
        messages = self.validator._validate_property(
            "test_string",
            {"type": "string"},
            "test.property"
        )
        
        self.assertEqual(len(messages), 0)  # Should be valid
        
        # Test invalid string type
        messages = self.validator._validate_property(
            123,
            {"type": "string"},
            "test.property"
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].severity, ValidationSeverity.ERROR)
        self.assertEqual(messages[0].code, "INVALID_TYPE")
    
    def test_validate_property_integer_type(self):
        """Test property validation for integer type"""
        messages = self.validator._validate_property(
            42,
            {"type": "integer"},
            "test.property"
        )
        
        self.assertEqual(len(messages), 0)  # Should be valid
        
        # Test invalid integer type
        messages = self.validator._validate_property(
            "not_an_integer",
            {"type": "integer"},
            "test.property"
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].code, "INVALID_TYPE")
    
    def test_validate_property_number_range(self):
        """Test property validation for number range"""
        # Valid number within range
        messages = self.validator._validate_property(
            50,
            {"type": "number", "minimum": 10, "maximum": 100},
            "test.property"
        )
        
        self.assertEqual(len(messages), 0)
        
        # Number below minimum
        messages = self.validator._validate_property(
            5,
            {"type": "number", "minimum": 10, "maximum": 100},
            "test.property"
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].code, "VALUE_TOO_LOW")
        self.assertEqual(messages[0].severity, ValidationSeverity.WARNING)
        
        # Number above maximum
        messages = self.validator._validate_property(
            150,
            {"type": "number", "minimum": 10, "maximum": 100},
            "test.property"
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].code, "VALUE_TOO_HIGH")
        self.assertEqual(messages[0].severity, ValidationSeverity.WARNING)
    
    def test_validate_property_enum(self):
        """Test property validation for enum values"""
        # Valid enum value
        messages = self.validator._validate_property(
            "bf16",
            {"enum": ["fp16", "bf16", "int8", "none"]},
            "test.property"
        )
        
        self.assertEqual(len(messages), 0)
        
        # Invalid enum value
        messages = self.validator._validate_property(
            "invalid_value",
            {"enum": ["fp16", "bf16", "int8", "none"]},
            "test.property"
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].code, "INVALID_ENUM_VALUE")
        self.assertEqual(messages[0].severity, ValidationSeverity.ERROR)
    
    def test_validate_property_pattern(self):
        """Test property validation for pattern matching"""
        # Valid pattern
        messages = self.validator._validate_property(
            "512x512",
            {"type": "string", "pattern": r"^\d+x\d+$"},
            "test.property"
        )
        
        self.assertEqual(len(messages), 0)
        
        # Invalid pattern
        messages = self.validator._validate_property(
            "invalid_resolution",
            {"type": "string", "pattern": r"^\d+x\d+$"},
            "test.property"
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].code, "INVALID_PATTERN")
        self.assertEqual(messages[0].severity, ValidationSeverity.ERROR)
    
    def test_cleanup_config_method(self):
        """Test _cleanup_config method"""
        config_data = {
            "clip_output": True,  # Should be removed
            "force_upcast": False,  # Should be removed
            "system": {
                "default_quantization": "bf16",
                "clip_output": True  # Should be removed from section
            },
            "valid_section": {
                "valid_property": "value"
            }
        }
        
        result = self.validator._cleanup_config(config_data)
        
        self.assertIsInstance(result, CleanupResult)
        self.assertGreater(len(result.cleaned_attributes), 0)
        self.assertTrue(result.backup_created)
        
        # Verify cleanup was applied
        self.assertNotIn("clip_output", config_data)
        self.assertNotIn("force_upcast", config_data)
        self.assertNotIn("clip_output", config_data["system"])
        self.assertIn("valid_section", config_data)  # Valid sections should remain


class TestValidationMessage(unittest.TestCase):
    """Test cases for ValidationMessage dataclass"""
    
    def test_validation_message_creation(self):
        """Test ValidationMessage creation"""
        message = ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="TEST_ERROR",
            message="Test error message",
            field_path="test.field",
            current_value="invalid",
            suggested_value="valid",
            help_text="This is help text"
        )
        
        self.assertEqual(message.severity, ValidationSeverity.ERROR)
        self.assertEqual(message.code, "TEST_ERROR")
        self.assertEqual(message.message, "Test error message")
        self.assertEqual(message.field_path, "test.field")
        self.assertEqual(message.current_value, "invalid")
        self.assertEqual(message.suggested_value, "valid")
        self.assertEqual(message.help_text, "This is help text")


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult dataclass"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        messages = [
            ValidationMessage(ValidationSeverity.ERROR, "ERROR1", "Error 1", "field1"),
            ValidationMessage(ValidationSeverity.WARNING, "WARN1", "Warning 1", "field2")
        ]
        cleaned_attributes = ["attr1", "attr2"]
        
        result = ValidationResult(
            is_valid=False,
            messages=messages,
            cleaned_attributes=cleaned_attributes,
            backup_path="/path/to/backup"
        )
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(len(result.cleaned_attributes), 2)
        self.assertEqual(result.backup_path, "/path/to/backup")
    
    def test_has_errors(self):
        """Test has_errors method"""
        # Result with errors
        error_result = ValidationResult(
            is_valid=False,
            messages=[
                ValidationMessage(ValidationSeverity.ERROR, "ERROR1", "Error", "field"),
                ValidationMessage(ValidationSeverity.WARNING, "WARN1", "Warning", "field")
            ],
            cleaned_attributes=[]
        )
        
        self.assertTrue(error_result.has_errors())
        
        # Result without errors
        warning_result = ValidationResult(
            is_valid=True,
            messages=[
                ValidationMessage(ValidationSeverity.WARNING, "WARN1", "Warning", "field"),
                ValidationMessage(ValidationSeverity.INFO, "INFO1", "Info", "field")
            ],
            cleaned_attributes=[]
        )
        
        self.assertFalse(warning_result.has_errors())
    
    def test_has_warnings(self):
        """Test has_warnings method"""
        # Result with warnings
        warning_result = ValidationResult(
            is_valid=True,
            messages=[
                ValidationMessage(ValidationSeverity.WARNING, "WARN1", "Warning", "field"),
                ValidationMessage(ValidationSeverity.INFO, "INFO1", "Info", "field")
            ],
            cleaned_attributes=[]
        )
        
        self.assertTrue(warning_result.has_warnings())
        
        # Result without warnings
        info_result = ValidationResult(
            is_valid=True,
            messages=[
                ValidationMessage(ValidationSeverity.INFO, "INFO1", "Info", "field")
            ],
            cleaned_attributes=[]
        )
        
        self.assertFalse(info_result.has_warnings())


class TestCleanupResult(unittest.TestCase):
    """Test cases for CleanupResult dataclass"""
    
    def test_cleanup_result_creation(self):
        """Test CleanupResult creation"""
        result = CleanupResult(
            cleaned_attributes=["attr1", "attr2", "attr3"],
            backup_created=True,
            backup_path="/path/to/backup"
        )
        
        self.assertEqual(len(result.cleaned_attributes), 3)
        self.assertTrue(result.backup_created)
        self.assertEqual(result.backup_path, "/path/to/backup")


class TestValidationSeverity(unittest.TestCase):
    """Test cases for ValidationSeverity enum"""
    
    def test_validation_severity_values(self):
        """Test ValidationSeverity enum values"""
        self.assertEqual(ValidationSeverity.INFO.value, "info")
        self.assertEqual(ValidationSeverity.WARNING.value, "warning")
        self.assertEqual(ValidationSeverity.ERROR.value, "error")
        self.assertEqual(ValidationSeverity.CRITICAL.value, "critical")


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.temp_dir) / "test_config.json"
        
        valid_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(valid_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_config_file_function(self):
        """Test validate_config_file convenience function"""
        result = validate_config_file(self.test_config_path)
        
        self.assertIsInstance(result, ValidationResult)
        # Should create backup
        self.assertIsNotNone(result.backup_path)
    
    def test_format_validation_result_valid(self):
        """Test format_validation_result for valid result"""
        result = ValidationResult(
            is_valid=True,
            messages=[
                ValidationMessage(ValidationSeverity.INFO, "INFO1", "Info message", "field")
            ],
            cleaned_attributes=["attr1"],
            backup_path="/path/to/backup"
        )
        
        formatted = format_validation_result(result)
        
        self.assertIn("✅ Configuration is valid", formatted)
        self.assertIn("📁 Backup created", formatted)
        self.assertIn("🧹 Cleaned 1 attributes", formatted)
        self.assertIn("ℹ️ INFO", formatted)
    
    def test_format_validation_result_invalid(self):
        """Test format_validation_result for invalid result"""
        result = ValidationResult(
            is_valid=False,
            messages=[
                ValidationMessage(
                    ValidationSeverity.ERROR, 
                    "ERROR1", 
                    "Error message", 
                    "field",
                    current_value="invalid",
                    suggested_value="valid",
                    help_text="Fix this error"
                ),
                ValidationMessage(ValidationSeverity.WARNING, "WARN1", "Warning message", "field2")
            ],
            cleaned_attributes=[],
            backup_path="/path/to/backup"
        )
        
        formatted = format_validation_result(result)
        
        self.assertIn("❌ Configuration has errors", formatted)
        self.assertIn("❌ ERROR (1):", formatted)
        self.assertIn("⚠️ WARNING (1):", formatted)
        self.assertIn("Current: invalid", formatted)
        self.assertIn("Suggested: valid", formatted)
        self.assertIn("Help: Fix this error", formatted)


class TestConfigValidatorIntegration(unittest.TestCase):
    """Integration tests for ConfigValidator"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_dir = Path(self.temp_dir) / "backups"
        self.validator = ConfigValidator(backup_dir=self.backup_dir)
        self.test_config_path = Path(self.temp_dir) / "integration_config.json"
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_validation_and_cleanup(self):
        """Test complete validation and cleanup workflow"""
        # Create config with various issues
        problematic_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 2048,  # Too high
                "unknown_attribute": "should_be_removed"
            },
            "directories": {
                "output_directory": "/path/to/output",
                "models_directory": "/path/to/models"
            },
            "generation": {
                "default_resolution": "invalid_resolution",  # Invalid pattern
                "default_steps": 20
            },
            "clip_output": True,  # Should be cleaned
            "force_upcast": False  # Should be cleaned
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(problematic_config, f)
        
        # Validate and clean
        result = self.validator.validate_config_file(self.test_config_path)
        
        # Should have created backup
        self.assertIsNotNone(result.backup_path)
        self.assertTrue(Path(result.backup_path).exists())
        
        # Should have cleaned attributes
        self.assertGreater(len(result.cleaned_attributes), 0)
        self.assertTrue(any("clip_output" in attr for attr in result.cleaned_attributes))
        
        # Should have validation messages
        self.assertGreater(len(result.messages), 0)
        
        # Should have warnings for out-of-range values
        warning_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.WARNING]
        self.assertTrue(any("too high" in msg.message.lower() for msg in warning_messages))
        
        # Should have errors for invalid patterns
        error_messages = [msg for msg in result.messages if msg.severity == ValidationSeverity.ERROR]
        self.assertTrue(any("pattern" in msg.message.lower() for msg in error_messages))
        
        # Verify file was actually cleaned
        with open(self.test_config_path, 'r') as f:
            cleaned_config = json.load(f)
        
        self.assertNotIn("clip_output", cleaned_config)
        self.assertNotIn("force_upcast", cleaned_config)
        self.assertNotIn("unknown_attribute", cleaned_config["system"])
    
    def test_backup_and_restore_workflow(self):
        """Test backup creation and potential restore workflow"""
        original_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(original_config, f)
        
        # Validate (creates backup)
        result = self.validator.validate_config_file(self.test_config_path)
        backup_path = result.backup_path
        
        # Modify original file
        modified_config = original_config.copy()
        modified_config["system"]["new_attribute"] = "new_value"
        
        with open(self.test_config_path, 'w') as f:
            json.dump(modified_config, f)
        
        # Verify backup still has original content
        with open(backup_path, 'r') as f:
            backup_config = json.load(f)
        
        self.assertEqual(backup_config, original_config)
        self.assertNotIn("new_attribute", backup_config["system"])


if __name__ == '__main__':
    unittest.main()