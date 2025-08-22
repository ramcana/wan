"""
Tests for Model Index Schema Validation

This module contains comprehensive tests for the model_index_schema module,
including tests for valid and invalid model configurations.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from model_index_schema import (
    ModelIndexSchema, 
    SchemaValidator, 
    ModelType, 
    SchemaValidationResult
)


class TestModelIndexSchema:
    """Test cases for ModelIndexSchema"""
    
    def test_valid_wan_t2v_schema(self):
        """Test valid Wan T2V model schema"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "transformer_2": ["diffusers", "Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "boundary_ratio": 0.5
        }
        
        schema = ModelIndexSchema(**data)
        
        assert schema.class_name == "WanPipeline"
        assert schema.diffusers_version == "0.21.0"
        assert schema.transformer is not None
        assert schema.transformer_2 is not None
        assert schema.boundary_ratio == 0.5
        assert schema.is_wan_architecture() is True
        assert schema.detect_model_type() == ModelType.WAN_T2V
    
    def test_valid_stable_diffusion_schema(self):
        """Test valid Stable Diffusion model schema"""
        data = {
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.21.0",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "PNDMScheduler"]
        }
        
        schema = ModelIndexSchema(**data)
        
        assert schema.class_name == "StableDiffusionPipeline"
        assert schema.is_wan_architecture() is False
        assert schema.detect_model_type() == ModelType.STABLE_DIFFUSION
    
    def test_wan_i2v_schema(self):
        """Test Wan I2V model schema"""
        data = {
            "_class_name": "WanI2VPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "boundary_ratio": 0.3
        }
        
        schema = ModelIndexSchema(**data)
        
        assert schema.detect_model_type() == ModelType.WAN_I2V
        assert schema.is_wan_architecture() is True
    
    def test_missing_required_fields(self):
        """Test schema with missing required fields"""
        from pydantic import ValidationError
        
        data = {
            "_diffusers_version": "0.21.0"
            # Missing _class_name
        }
        
        with pytest.raises(ValidationError):  # Should raise validation error
            ModelIndexSchema(**data)
    
    def test_invalid_boundary_ratio(self):
        """Test schema with invalid boundary_ratio"""
        from pydantic import ValidationError
        
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "boundary_ratio": 1.5  # Invalid: > 1
        }
        
        with pytest.raises(ValidationError):  # Should raise validation error
            ModelIndexSchema(**data)
    
    def test_get_required_components_wan(self):
        """Test getting required components for Wan model"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "boundary_ratio": 0.5
        }
        
        schema = ModelIndexSchema(**data)
        required = schema.get_required_components()
        
        assert "transformer" in required
        assert "vae" in required
        assert "text_encoder" in required
        assert "tokenizer" in required
        assert "scheduler" in required
    
    def test_get_required_components_sd(self):
        """Test getting required components for SD model"""
        data = {
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.21.0",
            "unet": ["diffusers", "UNet2DConditionModel"]
        }
        
        schema = ModelIndexSchema(**data)
        required = schema.get_required_components()
        
        assert "unet" in required
        assert "vae" in required
        assert "text_encoder" in required
        assert "tokenizer" in required
        assert "scheduler" in required
    
    def test_get_missing_components(self):
        """Test detection of missing components"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            # Missing vae, text_encoder, tokenizer, scheduler
        }
        
        schema = ModelIndexSchema(**data)
        missing = schema.get_missing_components()
        
        assert "vae" in missing
        assert "text_encoder" in missing
        assert "tokenizer" in missing
        assert "scheduler" in missing
    
    def test_validate_wan_specific_attributes(self):
        """Test validation of Wan-specific attributes"""
        # Test missing transformer components
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "boundary_ratio": 0.5
            # Missing transformer/transformer_2
        }
        
        schema = ModelIndexSchema(**data)
        issues = schema.validate_wan_specific_attributes()
        
        assert any("transformer" in issue for issue in issues)
    
    def test_validate_wan_pipeline_class_name(self):
        """Test validation of pipeline class name for Wan models"""
        data = {
            "_class_name": "StableDiffusionPipeline",  # Wrong class name for Wan
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "boundary_ratio": 0.5
        }
        
        schema = ModelIndexSchema(**data)
        issues = schema.validate_wan_specific_attributes()
        
        assert any("should start with 'Wan'" in issue for issue in issues)


class TestSchemaValidator:
    """Test cases for SchemaValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = SchemaValidator()
    
    def test_validate_valid_model_index_file(self):
        """Test validation of valid model_index.json file"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "transformer_2": ["diffusers", "Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "boundary_ratio": 0.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = self.validator.validate_model_index(temp_path)
            
            assert result.is_valid is True
            assert result.model_type == ModelType.WAN_T2V
            assert result.schema is not None
            assert len(result.errors) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_validate_missing_file(self):
        """Test validation of non-existent file"""
        result = self.validator.validate_model_index("non_existent_file.json")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0]
        assert "Ensure model_index.json exists" in result.suggested_fixes[0]
    
    def test_validate_invalid_json(self):
        """Test validation of invalid JSON file"""
        invalid_json = '{"_class_name": "WanPipeline", "invalid": json}'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(invalid_json)
            temp_path = f.name
        
        try:
            result = self.validator.validate_model_index(temp_path)
            
            assert result.is_valid is False
            assert any("Invalid JSON format" in error for error in result.errors)
            assert any("Fix JSON syntax" in fix for fix in result.suggested_fixes)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_model_index_dict(self):
        """Test validation of model index data from dictionary"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "DDIMScheduler"]
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        assert result.is_valid is True
        assert result.model_type == ModelType.WAN_T2V
        assert result.schema is not None
    
    def test_validate_missing_required_field(self):
        """Test validation with missing required field"""
        data = {
            "_diffusers_version": "0.21.0"
            # Missing _class_name
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        assert result.is_valid is False
        assert any("class_name" in error for error in result.errors)
        assert any("_class_name" in fix for fix in result.suggested_fixes)
    
    def test_validate_old_diffusers_version(self):
        """Test validation with old diffusers version"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.19.0",  # Old version
            "transformer": ["diffusers", "Transformer2DModel"]
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        assert any("Old diffusers version" in warning for warning in result.warnings)
        assert any("upgrading to diffusers" in fix for fix in result.suggested_fixes)
    
    def test_validate_component_consistency(self):
        """Test validation of component consistency"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "text_encoder": ["transformers", "CLIPTextModel"]
            # Missing tokenizer
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        assert any("tokenizer missing" in warning for warning in result.warnings)
        assert any("Add tokenizer component" in fix for fix in result.suggested_fixes)
    
    def test_generate_schema_report(self):
        """Test generation of schema validation report"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "boundary_ratio": 0.5
        }
        
        result = self.validator.validate_model_index_dict(data)
        report = self.validator.generate_schema_report(result)
        
        assert "MODEL INDEX SCHEMA VALIDATION REPORT" in report
        assert "WanPipeline" in report
        assert "wan_t2v" in report
        
        if result.warnings:
            assert "WARNINGS:" in report
        if result.suggested_fixes:
            assert "SUGGESTED FIXES:" in report
    
    def test_validation_history(self):
        """Test validation history tracking"""
        data1 = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0"
        }
        
        data2 = {
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.21.0"
        }
        
        result1 = self.validator.validate_model_index_dict(data1)
        result2 = self.validator.validate_model_index_dict(data2)
        
        history = self.validator.get_validation_history()
        
        assert len(history) == 2
        assert history[0] == result1
        assert history[1] == result2
        
        self.validator.clear_validation_history()
        assert len(self.validator.get_validation_history()) == 0
    
    def test_result_to_dict(self):
        """Test conversion of validation result to dictionary"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"]
        }
        
        result = self.validator.validate_model_index_dict(data)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "is_valid" in result_dict
        assert "model_type" in result_dict
        assert "errors" in result_dict
        assert "warnings" in result_dict
        assert "suggested_fixes" in result_dict
        assert "schema_data" in result_dict


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = SchemaValidator()
    
    def test_empty_model_index(self):
        """Test validation of empty model index"""
        result = self.validator.validate_model_index_dict({})
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_model_index_with_extra_fields(self):
        """Test model index with extra custom fields"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "custom_field": "custom_value",
            "another_custom": {"nested": "data"}
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        # Should still be valid due to extra="allow"
        assert result.is_valid is True
        assert result.schema is not None
    
    def test_boundary_ratio_edge_values(self):
        """Test boundary_ratio with edge values"""
        # Test boundary_ratio = 0
        data1 = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "boundary_ratio": 0.0
        }
        
        result1 = self.validator.validate_model_index_dict(data1)
        assert result1.schema.boundary_ratio == 0.0
        
        # Test boundary_ratio = 1
        data2 = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "boundary_ratio": 1.0
        }
        
        result2 = self.validator.validate_model_index_dict(data2)
        assert result2.schema.boundary_ratio == 1.0
        
        # Test invalid boundary_ratio
        data3 = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "boundary_ratio": -0.1
        }
        
        result3 = self.validator.validate_model_index_dict(data3)
        assert result3.is_valid is False
    
    def test_unknown_model_type(self):
        """Test detection of unknown model type"""
        data = {
            "_class_name": "CustomPipeline",
            "_diffusers_version": "0.21.0"
            # No recognizable components
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        assert result.model_type == ModelType.UNKNOWN
    
    def test_malformed_version_string(self):
        """Test handling of malformed version strings"""
        data = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "invalid.version.string",
            "transformer": ["diffusers", "Transformer2DModel"]
        }
        
        result = self.validator.validate_model_index_dict(data)
        
        assert any("Invalid diffusers version format" in warning for warning in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])