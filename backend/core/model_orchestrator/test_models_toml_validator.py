#!/usr/bin/env python3
"""
Test suite for the models.toml validator.

Tests various validation scenarios including:
- Schema version issues
- Duplicate detection
- Path traversal vulnerabilities
- Windows case sensitivity issues
- Malformed TOML structures
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the backend directory to the path for imports
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, backend_dir)

from validate_models_toml import ModelsTomlValidator


class TestModelsTomlValidator(unittest.TestCase):
    """Test cases for the ModelsTomlValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_manifest(self, content: str) -> str:
        """Create a temporary manifest file with the given content."""
        manifest_path = os.path.join(self.temp_dir, "test_models.toml")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return manifest_path
    
    def test_valid_manifest(self):
        """Test validation of a valid manifest."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"
resolution_caps = ["720p24"]
optional_components = []
lora_required = false
allow_patterns = ["*.safetensors", "*.json"]
required_components = ["unet", "vae"]

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
component = "config"

[models."test-model@1.0.0".sources]
priority = ["local://test", "hf://test/model"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertTrue(is_valid, f"Valid manifest should pass validation. Errors: {errors}")
        self.assertEqual(len(errors), 0)
    
    def test_missing_schema_version(self):
        """Test detection of missing schema version."""
        content = '''
[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("schema_version" in error for error in errors))
    
    def test_unsupported_schema_version(self):
        """Test detection of unsupported schema version."""
        content = '''
schema_version = 999

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Unsupported schema version" in error for error in errors))
    
    def test_invalid_model_id_format(self):
        """Test detection of invalid model ID formats."""
        content = '''
schema_version = 1

[models."invalid-model-id"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."invalid-model-id".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."invalid-model-id".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid model ID format" in error for error in errors))
    
    def test_duplicate_file_paths(self):
        """Test detection of duplicate file paths within a model."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 2048
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("duplicate file paths" in error for error in errors))
    
    def test_path_traversal_detection(self):
        """Test detection of path traversal vulnerabilities."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "../../../etc/passwd"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("path traversal" in error for error in errors))
    
    def test_absolute_path_detection(self):
        """Test detection of absolute paths."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "/absolute/path/config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("absolute path" in error for error in errors))
    
    def test_windows_reserved_names(self):
        """Test detection of Windows reserved names."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "CON.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Windows reserved name" in error for error in errors))
    
    def test_case_collision_detection(self):
        """Test detection of case collisions."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "Config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 2048
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("case collision" in error for error in errors))
    
    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
# Missing version, variants, default_variant, files, sources

[[models."test-model@1.0.0".files]]
path = "config.json"
# Missing size, sha256
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("missing required field" in error for error in errors))
    
    def test_invalid_sha256_length(self):
        """Test detection of invalid SHA256 length."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "invalid_short_hash"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("sha256 must be a 64-character string" in error for error in errors))
    
    def test_invalid_file_size(self):
        """Test detection of invalid file sizes."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = -1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("size must be a non-negative integer" in error for error in errors))
    
    def test_default_variant_not_in_variants(self):
        """Test detection when default_variant is not in variants list."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16", "bf16"]
default_variant = "int8"

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("default_variant" in error and "not in variants list" in error for error in errors))
    
    def test_malformed_toml(self):
        """Test handling of malformed TOML."""
        content = '''
schema_version = 1
[models."test-model@1.0.0"
# Missing closing bracket - invalid TOML
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Failed to parse TOML" in error for error in errors))
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent manifest file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.toml")
        validator = ModelsTomlValidator(nonexistent_path)
        is_valid, errors, warnings = validator.validate()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Manifest file not found" in error for error in errors))
    
    def test_long_path_warning(self):
        """Test warning for very long paths."""
        long_path = "very/long/path/" + "a" * 200 + "/config.json"
        content = f'''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "{long_path}"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        # Should be valid but with warnings
        self.assertTrue(is_valid)
        self.assertTrue(any("very long path" in warning for warning in warnings))
    
    def test_trailing_space_warning(self):
        """Test warning for paths with trailing spaces."""
        content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "config.json "
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_test_manifest(content)
        validator = ModelsTomlValidator(manifest_path)
        is_valid, errors, warnings = validator.validate()
        
        # Should be valid but with warnings
        self.assertTrue(is_valid)
        self.assertTrue(any("ends with space or dot" in warning for warning in warnings))


def main():
    """Run the test suite."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()