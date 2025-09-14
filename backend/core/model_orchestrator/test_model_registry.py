"""
Unit tests for the Model Registry system.

Tests manifest parsing, validation, model ID normalization, and error handling.
"""

import os
import tempfile
from pathlib import Path
from typing import List
import pytest

from .model_registry import ModelRegistry, ModelSpec, FileSpec, NormalizedModelId
from .exceptions import (
    ModelNotFoundError,
    VariantNotFoundError,
    InvalidModelIdError,
    ManifestValidationError,
    SchemaVersionError,
)


class TestModelIdNormalization:
    """Test model ID normalization functionality."""
    
    def test_normalize_basic_model_id(self):
        """Test basic model ID normalization."""
        result = ModelRegistry.normalize_model_id("t2v-A14B@2.2.0")
        
        assert result.model_id == "t2v-A14B"
        assert result.version == "2.2.0"
        assert result.variant is None
    
    def test_normalize_model_id_with_variant(self):
        """Test model ID normalization with variant."""
        result = ModelRegistry.normalize_model_id("t2v-A14B@2.2.0", "fp16")
        
        assert result.model_id == "t2v-A14B"
        assert result.version == "2.2.0"
        assert result.variant == "fp16"
    
    def test_normalize_model_id_with_embedded_variant(self):
        """Test model ID normalization with embedded variant."""
        result = ModelRegistry.normalize_model_id("t2v-A14B@2.2.0#fp16")
        
        assert result.model_id == "t2v-A14B"
        assert result.version == "2.2.0"
        assert result.variant == "fp16"
    
    def test_normalize_variant_override(self):
        """Test that explicit variant overrides embedded variant."""
        result = ModelRegistry.normalize_model_id("t2v-A14B@2.2.0#fp16", "bf16")
        
        assert result.model_id == "t2v-A14B"
        assert result.version == "2.2.0"
        assert result.variant == "bf16"
    
    def test_normalize_invalid_model_id_format(self):
        """Test error handling for invalid model ID formats."""
        with pytest.raises(InvalidModelIdError) as exc_info:
            ModelRegistry.normalize_model_id("invalid-model-id")
        
        assert "must include version" in str(exc_info.value)
        assert exc_info.value.error_code.value == "INVALID_MODEL_ID"
    
    def test_normalize_invalid_version_format(self):
        """Test error handling for invalid version formats."""
        with pytest.raises(InvalidModelIdError) as exc_info:
            ModelRegistry.normalize_model_id("t2v-A14B@invalid-version")
        
        assert "semantic versioning" in str(exc_info.value)
    
    def test_normalize_invalid_model_name(self):
        """Test error handling for invalid model names."""
        with pytest.raises(InvalidModelIdError) as exc_info:
            ModelRegistry.normalize_model_id("-invalid@2.2.0")
        
        assert "must start with alphanumeric" in str(exc_info.value)
    
    def test_normalize_invalid_variant(self):
        """Test error handling for invalid variants."""
        with pytest.raises(InvalidModelIdError) as exc_info:
            ModelRegistry.normalize_model_id("t2v-A14B@2.2.0", "-invalid")
        
        assert "must start with alphanumeric" in str(exc_info.value)
    
    def test_normalize_empty_model_id(self):
        """Test error handling for empty model ID."""
        with pytest.raises(InvalidModelIdError) as exc_info:
            ModelRegistry.normalize_model_id("")
        
        assert "non-empty string" in str(exc_info.value)


class TestManifestParsing:
    """Test manifest parsing and validation."""
    
    def create_temp_manifest(self, content: str) -> str:
        """Create a temporary manifest file with given content."""
        fd, path = tempfile.mkstemp(suffix=".toml")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            return path
        except:
            os.close(fd)
            raise
    
    def test_load_valid_manifest(self):
        """Test loading a valid manifest."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"
resolution_caps = ["720p"]
optional_components = []
lora_required = false
allow_patterns = ["*.safetensors"]

[[models."test-model@1.0.0".files]]
path = "model.safetensors"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_temp_manifest(manifest_content)
        try:
            registry = ModelRegistry(manifest_path)
            
            assert registry.get_schema_version() == "1"
            assert len(registry.list_models()) == 1
            assert "test-model@1.0.0" in registry.list_models()
            
            spec = registry.spec("test-model@1.0.0")
            assert spec.model_id == "test-model@1.0.0"
            assert spec.version == "1.0.0"
            assert spec.variants == ["fp16", "bf16"]
            assert spec.default_variant == "fp16"
            assert len(spec.files) == 1
            assert spec.files[0].path == "model.safetensors"
            assert spec.files[0].size == 1024
            
        finally:
            os.unlink(manifest_path)
    
    def test_load_manifest_missing_file(self):
        """Test error handling for missing manifest file."""
        with pytest.raises(FileNotFoundError):
            ModelRegistry("/nonexistent/path/models.toml")
    
    def test_load_manifest_invalid_toml(self):
        """Test error handling for invalid TOML syntax."""
        manifest_content = '''
schema_version = 1
[models."test-model@1.0.0"
# Missing closing bracket
'''
        
        manifest_path = self.create_temp_manifest(manifest_content)
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "Failed to parse TOML" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_load_manifest_unsupported_schema(self):
        """Test error handling for unsupported schema version."""
        manifest_content = '''
schema_version = 999

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "model.safetensors"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_temp_manifest(manifest_content)
        try:
            with pytest.raises(SchemaVersionError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "999" in str(exc_info.value)
            assert "Supported versions: 1" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_load_manifest_missing_required_fields(self):
        """Test error handling for missing required fields."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
# Missing required fields: variants, default_variant, files, sources
'''
        
        manifest_path = self.create_temp_manifest(manifest_content)
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "variants field is required" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_load_manifest_invalid_model_id(self):
        """Test error handling for invalid model ID format."""
        manifest_content = '''
schema_version = 1

[models."invalid-model-id"]
description = "Test model"
variants = ["fp16"]
default_variant = "fp16"

[[models."invalid-model-id".files]]
path = "model.safetensors"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."invalid-model-id".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_temp_manifest(manifest_content)
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "Invalid model ID format" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_load_manifest_invalid_file_spec(self):
        """Test error handling for invalid file specifications."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "model.safetensors"
size = -1  # Invalid negative size
sha256 = "invalid-hash"  # Invalid hash length

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = self.create_temp_manifest(manifest_content)
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            error_msg = str(exc_info.value)
            assert "non-negative integer" in error_msg or "64 characters long" in error_msg
            
        finally:
            os.unlink(manifest_path)


class TestPathSafetyValidation:
    """Test path safety validation."""
    
    def create_temp_manifest_with_paths(self, file_paths: List[str]) -> str:
        """Create a temporary manifest with specified file paths."""
        files_section = ""
        for i, path in enumerate(file_paths):
            files_section += f'''
[[models."test-model@1.0.0".files]]
path = "{path}"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
'''
        
        manifest_content = f'''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"
{files_section}

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        fd, path = tempfile.mkstemp(suffix=".toml")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(manifest_content)
            return path
        except:
            os.close(fd)
            raise
    
    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts."""
        manifest_path = self.create_temp_manifest_with_paths([
            "model.safetensors",
            "../../../etc/passwd",  # Path traversal attempt
        ])
        
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "Path traversal detected" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_absolute_path_detection(self):
        """Test detection of absolute paths."""
        manifest_path = self.create_temp_manifest_with_paths([
            "model.safetensors",
            "/absolute/path/model.safetensors",  # Absolute path
        ])
        
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "Absolute path not allowed" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_windows_reserved_name_detection(self):
        """Test detection of Windows reserved names."""
        manifest_path = self.create_temp_manifest_with_paths([
            "model.safetensors",
            "CON.txt",  # Windows reserved name
        ])
        
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "Windows reserved name" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)
    
    def test_case_collision_detection(self):
        """Test detection of case collisions."""
        manifest_path = self.create_temp_manifest_with_paths([
            "Model.safetensors",
            "model.safetensors",  # Case collision
        ])
        
        try:
            with pytest.raises(ManifestValidationError) as exc_info:
                ModelRegistry(manifest_path)
            
            assert "Case collision detected" in str(exc_info.value)
            
        finally:
            os.unlink(manifest_path)


class TestModelSpecRetrieval:
    """Test model specification retrieval."""
    
    def setup_method(self):
        """Set up test registry with sample models."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model"
version = "1.0.0"
variants = ["fp16", "bf16", "int8"]
default_variant = "fp16"
resolution_caps = ["720p"]
optional_components = []
lora_required = false

[[models."test-model@1.0.0".files]]
path = "test_model.safetensors"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]

[models."another-model@2.0.0"]
description = "Another test model"
version = "2.0.0"
variants = ["fp32"]
default_variant = "fp32"

[[models."another-model@2.0.0".files]]
path = "another_model.safetensors"
size = 2048
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

[models."another-model@2.0.0".sources]
priority = ["local://test"]
'''
        
        fd, self.manifest_path = tempfile.mkstemp(suffix=".toml")
        with os.fdopen(fd, 'w') as f:
            f.write(manifest_content)
        self.registry = ModelRegistry(self.manifest_path)
    
    def teardown_method(self):
        """Clean up test files."""
        if hasattr(self, 'manifest_path'):
            os.unlink(self.manifest_path)
    
    def test_get_model_spec_basic(self):
        """Test basic model specification retrieval."""
        spec = self.registry.spec("test-model@1.0.0")
        
        assert spec.model_id == "test-model@1.0.0"
        assert spec.version == "1.0.0"
        assert spec.variants == ["fp16", "bf16", "int8"]
        assert spec.default_variant == "fp16"
        assert len(spec.files) == 1
        assert spec.files[0].path == "test_model.safetensors"
    
    def test_get_model_spec_with_variant(self):
        """Test model specification retrieval with variant."""
        spec = self.registry.spec("test-model@1.0.0", "bf16")
        
        assert spec.model_id == "test-model@1.0.0"
        assert "bf16" in spec.variants
    
    def test_get_model_spec_nonexistent_model(self):
        """Test error handling for nonexistent model."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            self.registry.spec("nonexistent-model@1.0.0")
        
        assert "nonexistent-model@1.0.0" in str(exc_info.value)
        assert exc_info.value.error_code.value == "MODEL_NOT_FOUND"
        assert "test-model@1.0.0" in str(exc_info.value)  # Available models listed
    
    def test_get_model_spec_invalid_variant(self):
        """Test error handling for invalid variant."""
        with pytest.raises(VariantNotFoundError) as exc_info:
            self.registry.spec("test-model@1.0.0", "invalid-variant")
        
        assert "invalid-variant" in str(exc_info.value)
        assert "test-model@1.0.0" in str(exc_info.value)
        assert exc_info.value.error_code.value == "VARIANT_NOT_FOUND"
        assert "fp16" in str(exc_info.value)  # Available variants listed
    
    def test_list_models(self):
        """Test listing all available models."""
        models = self.registry.list_models()
        
        assert len(models) == 2
        assert "test-model@1.0.0" in models
        assert "another-model@2.0.0" in models


if __name__ == "__main__":
    pytest.main([__file__])