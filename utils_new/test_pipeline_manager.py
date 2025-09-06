"""
Tests for the pipeline management system.

This module tests pipeline selection, loading, validation, and compatibility
checking for different model architectures.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipeline_manager import (
    PipelineManager, PipelineRequirements, ValidationResult, 
    PipelineLoadResult, PipelineLoadStatus
)
from architecture_detector import (
    ArchitectureSignature, ArchitectureType, ModelArchitecture,
    ModelRequirements, ComponentInfo
)


class TestPipelineManager:
    """Test cases for PipelineManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PipelineManager()
    
    def test_init(self):
        """Test PipelineManager initialization."""
        assert self.manager is not None
        assert hasattr(self.manager, 'logger')
        assert hasattr(self.manager, '_pipeline_cache')
        assert isinstance(self.manager.PIPELINE_MAPPINGS, dict)
        assert isinstance(self.manager.PIPELINE_REQUIREMENTS, dict)
    
    def test_select_pipeline_class_explicit(self):
        """Test pipeline selection with explicit pipeline class."""
        signature = ArchitectureSignature(
            pipeline_class="CustomWanPipeline",
            has_transformer_2=True,
            vae_dimensions=3
        )
        
        result = self.manager.select_pipeline_class(signature)
        assert result == "CustomWanPipeline"
    
    def test_select_pipeline_class_wan_t2v(self):
        """Test pipeline selection for Wan T2V architecture."""
        signature = ArchitectureSignature(
            has_transformer_2=True,
            vae_dimensions=3,
            has_boundary_ratio=True
        )
        
        result = self.manager.select_pipeline_class(signature)
        assert result == "WanPipeline"
    
    def test_select_pipeline_class_wan_t2i(self):
        """Test pipeline selection for Wan T2I architecture."""
        signature = ArchitectureSignature(
            has_transformer=True,
            has_transformer_2=False,
            component_classes={"transformer": "WanTransformer"}
        )
        
        result = self.manager.select_pipeline_class(signature)
        assert result == "WanPipeline"
    
    def test_select_pipeline_class_stable_diffusion(self):
        """Test pipeline selection for Stable Diffusion architecture."""
        signature = ArchitectureSignature(
            component_classes={
                "unet": "UNet2DConditionModel",
                "text_encoder": "CLIPTextModel"
            }
        )
        
        result = self.manager.select_pipeline_class(signature)
        assert result == "StableDiffusionPipeline"
    
    def test_select_pipeline_class_unknown(self):
        """Test pipeline selection for unknown architecture."""
        signature = ArchitectureSignature()
        
        result = self.manager.select_pipeline_class(signature)
        assert result == "DiffusionPipeline"
    
    def test_get_pipeline_requirements_wan(self):
        """Test getting requirements for WanPipeline."""
        requirements = self.manager.get_pipeline_requirements("WanPipeline")
        
        assert isinstance(requirements, PipelineRequirements)
        assert "transformer" in requirements.required_args
        assert "scheduler" in requirements.required_args
        assert "vae" in requirements.required_args
        assert requirements.requires_trust_remote_code is True
        assert requirements.min_vram_mb >= 8192
        assert "transformers>=4.25.0" in requirements.dependencies
    
    def test_get_pipeline_requirements_stable_diffusion(self):
        """Test getting requirements for StableDiffusionPipeline."""
        requirements = self.manager.get_pipeline_requirements("StableDiffusionPipeline")
        
        assert isinstance(requirements, PipelineRequirements)
        assert "unet" in requirements.required_args
        assert "text_encoder" in requirements.required_args
        assert "tokenizer" in requirements.required_args
        assert requirements.requires_trust_remote_code is False
        assert requirements.min_vram_mb >= 4096
    
    def test_get_pipeline_requirements_unknown(self):
        """Test getting requirements for unknown pipeline."""
        requirements = self.manager.get_pipeline_requirements("UnknownPipeline")
        
        assert isinstance(requirements, PipelineRequirements)
        # Should return default requirements
        assert requirements.min_vram_mb > 0
    
    def test_get_pipeline_requirements_inferred_wan(self):
        """Test inferring requirements for Wan-like pipeline names."""
        requirements = self.manager.get_pipeline_requirements("CustomWanVideoPipeline")
        
        assert requirements.requires_trust_remote_code is True
        assert requirements.min_vram_mb >= 8192
        assert "transformer" in requirements.required_args
    
    def test_validate_pipeline_args_valid(self):
        """Test validation with valid arguments."""
        provided_args = {
            "transformer": Mock(),
            "scheduler": Mock(),
            "vae": Mock(),
            "torch_dtype": "torch.float16"
        }
        
        result = self.manager.validate_pipeline_args("WanPipeline", provided_args)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.missing_required) == 0
    
    def test_validate_pipeline_args_missing_required(self):
        """Test validation with missing required arguments."""
        provided_args = {
            "scheduler": Mock(),
            "vae": Mock()
            # Missing transformer
        }
        
        result = self.manager.validate_pipeline_args("WanPipeline", provided_args)
        
        assert result.is_valid is False
        assert "transformer" in result.missing_required
    
    def test_validate_pipeline_args_with_warnings(self):
        """Test validation that generates warnings."""
        provided_args = {
            "transformer": Mock(),
            "scheduler": Mock(),
            "vae": Mock(),
            "optional_arg": None  # This should generate a warning
        }
        
        result = self.manager.validate_pipeline_args("WanPipeline", provided_args)
        
        assert len(result.warnings) > 0
        assert any("None" in warning for warning in result.warnings)
    
    def test_validate_pipeline_args_with_suggestions(self):
        """Test validation that generates suggestions."""
        provided_args = {
            "transformer": Mock(),
            "scheduler": Mock(), 
            "vae": Mock()
            # Missing trust_remote_code for WanPipeline
        }
        
        result = self.manager.validate_pipeline_args("WanPipeline", provided_args)
        
        assert len(result.suggestions) > 0
        assert any("trust_remote_code" in suggestion for suggestion in result.suggestions)
    
    @patch('diffusers.DiffusionPipeline')
    def test_load_custom_pipeline_success(self, mock_diffusion_pipeline):
        """Test successful pipeline loading."""
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "WanPipeline"
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline
        
        result = self.manager.load_custom_pipeline(
            model_path="/fake/path",
            pipeline_class="WanPipeline",
            trust_remote_code=True
        )
        
        assert result.status == PipelineLoadStatus.SUCCESS
        assert result.pipeline is mock_pipeline
        assert result.pipeline_class == "WanPipeline"
        assert result.error_message is None
    
    @patch('diffusers.DiffusionPipeline')
    def test_load_custom_pipeline_missing_class(self, mock_diffusion_pipeline):
        """Test pipeline loading with missing class."""
        mock_diffusion_pipeline.from_pretrained.side_effect = AttributeError("Pipeline not found")
        
        result = self.manager.load_custom_pipeline(
            model_path="/fake/path",
            pipeline_class="NonExistentPipeline",
            trust_remote_code=False
        )
        
        assert result.status == PipelineLoadStatus.FAILED_MISSING_CLASS
        assert result.pipeline is None
        assert "not found" in result.error_message
    
    @patch('diffusers.DiffusionPipeline')
    def test_load_custom_pipeline_dependencies_error(self, mock_diffusion_pipeline):
        """Test pipeline loading with missing dependencies."""
        mock_diffusion_pipeline.from_pretrained.side_effect = ImportError("Missing transformers")
        
        result = self.manager.load_custom_pipeline(
            model_path="/fake/path",
            pipeline_class="WanPipeline",
            trust_remote_code=True
        )
        
        assert result.status == PipelineLoadStatus.FAILED_DEPENDENCIES
        assert "Missing dependencies" in result.error_message
    
    @patch('diffusers.DiffusionPipeline')
    def test_load_custom_pipeline_unknown_error(self, mock_diffusion_pipeline):
        """Test pipeline loading with unknown error."""
        mock_diffusion_pipeline.from_pretrained.side_effect = RuntimeError("Unknown error")
        
        result = self.manager.load_custom_pipeline(
            model_path="/fake/path",
            pipeline_class="WanPipeline",
            trust_remote_code=True
        )
        
        assert result.status == PipelineLoadStatus.FAILED_UNKNOWN
        assert "Unknown error" in result.error_message
    
    def test_create_pipeline_mapping_wan(self):
        """Test creating pipeline mapping for Wan architecture."""
        components = {
            "transformer": ComponentInfo("WanTransformer", "config.json", "weights"),
            "transformer_2": ComponentInfo("WanTransformer2", "config.json", "weights"),
            "scheduler": ComponentInfo("DDIMScheduler", "config.json", "weights"),
            "vae": ComponentInfo("WanVAE", "config.json", "weights")
        }
        
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.WAN_T2V,
            components=components
        )
        
        mapping = self.manager.create_pipeline_mapping(architecture)
        
        assert "transformer" in mapping
        assert "transformer_2" in mapping
        assert "scheduler" in mapping
        assert "vae" in mapping
        assert "unet" not in mapping  # Wan models don't use unet
    
    def test_create_pipeline_mapping_stable_diffusion(self):
        """Test creating pipeline mapping for Stable Diffusion architecture."""
        components = {
            "unet": ComponentInfo("UNet2DConditionModel", "config.json", "weights"),
            "text_encoder": ComponentInfo("CLIPTextModel", "config.json", "weights"),
            "tokenizer": ComponentInfo("CLIPTokenizer", "config.json", "weights"),
            "scheduler": ComponentInfo("DDIMScheduler", "config.json", "weights"),
            "vae": ComponentInfo("AutoencoderKL", "config.json", "weights")
        }
        
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.STABLE_DIFFUSION,
            components=components
        )
        
        mapping = self.manager.create_pipeline_mapping(architecture)
        
        assert "unet" in mapping
        assert "text_encoder" in mapping
        assert "tokenizer" in mapping
        assert "scheduler" in mapping
        assert "vae" in mapping
        assert "transformer" not in mapping  # SD models don't use transformer
    
    def test_suggest_pipeline_alternatives_wan_failed(self):
        """Test suggesting alternatives when WanPipeline fails."""
        signature = ArchitectureSignature(has_transformer_2=True)
        
        alternatives = self.manager.suggest_pipeline_alternatives("WanPipeline", signature)
        
        assert "DiffusionPipeline" in alternatives
        assert "StableDiffusionPipeline" in alternatives
        assert "WanPipeline" not in alternatives  # Should not suggest the failed one
    
    def test_suggest_pipeline_alternatives_sd_failed(self):
        """Test suggesting alternatives when StableDiffusionPipeline fails."""
        signature = ArchitectureSignature()
        
        alternatives = self.manager.suggest_pipeline_alternatives("StableDiffusionPipeline", signature)
        
        assert "DiffusionPipeline" in alternatives
        assert len(alternatives) > 0
        assert "StableDiffusionPipeline" not in alternatives
    
    def test_validate_pipeline_compatibility_wan_with_wan_model(self):
        """Test compatibility validation for WanPipeline with Wan model."""
        components = {
            "transformer": ComponentInfo("WanTransformer", "config.json", "weights"),
            "scheduler": ComponentInfo("DDIMScheduler", "config.json", "weights"),
            "vae": ComponentInfo("WanVAE", "config.json", "weights")
        }
        
        signature = ArchitectureSignature(
            has_transformer_2=True,
            vae_dimensions=3
        )
        
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.WAN_T2V,
            components=components,
            signature=signature
        )
        
        result = self.manager.validate_pipeline_compatibility("WanPipeline", architecture)
        
        assert result.is_valid is True
        assert len(result.warnings) == 0  # Should be compatible
    
    def test_validate_pipeline_compatibility_sd_with_wan_model(self):
        """Test compatibility validation for StableDiffusionPipeline with Wan model."""
        components = {
            "transformer": ComponentInfo("WanTransformer", "config.json", "weights"),
            "scheduler": ComponentInfo("DDIMScheduler", "config.json", "weights"),
            "vae": ComponentInfo("WanVAE", "config.json", "weights")
        }
        
        signature = ArchitectureSignature(
            has_transformer_2=True,
            vae_dimensions=3
        )
        
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.WAN_T2V,
            components=components,
            signature=signature
        )
        
        result = self.manager.validate_pipeline_compatibility("StableDiffusionPipeline", architecture)
        
        assert len(result.warnings) > 0  # Should warn about incompatibility
        assert any("Wan model" in warning for warning in result.warnings)
        assert any("WanPipeline" in suggestion for suggestion in result.suggestions)
    
    def test_validate_pipeline_compatibility_missing_components(self):
        """Test compatibility validation with missing components."""
        components = {
            "scheduler": ComponentInfo("DDIMScheduler", "config.json", "weights"),
            "vae": ComponentInfo("AutoencoderKL", "config.json", "weights")
            # Missing required unet, text_encoder, tokenizer
        }
        
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.STABLE_DIFFUSION,
            components=components
        )
        
        result = self.manager.validate_pipeline_compatibility("StableDiffusionPipeline", architecture)
        
        assert result.is_valid is False
        assert len(result.missing_required) > 0
        assert "unet" in result.missing_required
        assert "text_encoder" in result.missing_required
        assert "tokenizer" in result.missing_required
    
    def test_validate_pipeline_compatibility_vram_mismatch(self):
        """Test compatibility validation with VRAM requirements mismatch."""
        components = {
            "transformer": ComponentInfo("WanTransformer", "config.json", "weights"),
            "scheduler": ComponentInfo("DDIMScheduler", "config.json", "weights"),
            "vae": ComponentInfo("WanVAE", "config.json", "weights")
        }
        
        # Model requires more VRAM than pipeline typically supports
        requirements = ModelRequirements(min_vram_mb=16384)
        
        architecture = ModelArchitecture(
            architecture_type=ArchitectureType.WAN_T2V,
            components=components,
            requirements=requirements
        )
        
        result = self.manager.validate_pipeline_compatibility("StableDiffusionPipeline", architecture)
        
        assert len(result.warnings) > 0
        assert any("VRAM" in warning for warning in result.warnings)


class TestPipelineRequirements:
    """Test cases for PipelineRequirements dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of PipelineRequirements."""
        req = PipelineRequirements()
        
        assert req.required_args == []
        assert req.optional_args == []
        assert req.dependencies == []
        assert req.min_vram_mb == 4096
        assert req.supports_cpu_offload is True
        assert req.supports_mixed_precision is True
        assert req.requires_trust_remote_code is False
        assert req.pipeline_source is None
    
    def test_custom_initialization(self):
        """Test custom initialization of PipelineRequirements."""
        req = PipelineRequirements(
            required_args=["transformer", "vae"],
            dependencies=["transformers>=4.25.0"],
            min_vram_mb=8192,
            requires_trust_remote_code=True
        )
        
        assert req.required_args == ["transformer", "vae"]
        assert req.dependencies == ["transformers>=4.25.0"]
        assert req.min_vram_mb == 8192
        assert req.requires_trust_remote_code is True


class TestValidationResult:
    """Test cases for ValidationResult dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of ValidationResult."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.missing_required == []
        assert result.invalid_args == []
        assert result.warnings == []
        assert result.suggestions == []
    
    def test_validation_failure(self):
        """Test ValidationResult for validation failure."""
        result = ValidationResult(
            is_valid=False,
            missing_required=["transformer"],
            warnings=["VRAM may be insufficient"],
            suggestions=["Add trust_remote_code=True"]
        )
        
        assert result.is_valid is False
        assert "transformer" in result.missing_required
        assert len(result.warnings) == 1
        assert len(result.suggestions) == 1


class TestPipelineLoadResult:
    """Test cases for PipelineLoadResult dataclass."""
    
    def test_success_result(self):
        """Test successful pipeline load result."""
        mock_pipeline = Mock()
        
        result = PipelineLoadResult(
            status=PipelineLoadStatus.SUCCESS,
            pipeline=mock_pipeline,
            pipeline_class="WanPipeline"
        )
        
        assert result.status == PipelineLoadStatus.SUCCESS
        assert result.pipeline is mock_pipeline
        assert result.pipeline_class == "WanPipeline"
        assert result.error_message is None
        assert result.warnings == []
        assert result.applied_optimizations == []
    
    def test_failure_result(self):
        """Test failed pipeline load result."""
        result = PipelineLoadResult(
            status=PipelineLoadStatus.FAILED_MISSING_CLASS,
            error_message="Pipeline class not found"
        )
        
        assert result.status == PipelineLoadStatus.FAILED_MISSING_CLASS
        assert result.pipeline is None
        assert result.error_message == "Pipeline class not found"


class TestIntegrationScenarios:
    """Integration test scenarios for pipeline management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PipelineManager()
    
    def test_end_to_end_wan_pipeline_selection(self):
        """Test complete pipeline selection workflow for Wan model."""
        # Create Wan architecture signature
        signature = ArchitectureSignature(
            has_transformer_2=True,
            has_boundary_ratio=True,
            vae_dimensions=3,
            component_classes={
                "transformer": "WanTransformer",
                "transformer_2": "WanTransformer2",
                "vae": "WanVAE"
            }
        )
        
        # Select pipeline class
        pipeline_class = self.manager.select_pipeline_class(signature)
        assert pipeline_class == "WanPipeline"
        
        # Get requirements
        requirements = self.manager.get_pipeline_requirements(pipeline_class)
        assert requirements.requires_trust_remote_code is True
        
        # Validate arguments
        provided_args = {
            "transformer": Mock(),
            "transformer_2": Mock(),
            "scheduler": Mock(),
            "vae": Mock(),
            "trust_remote_code": True
        }
        
        validation = self.manager.validate_pipeline_args(pipeline_class, provided_args)
        assert validation.is_valid is True
    
    def test_end_to_end_stable_diffusion_pipeline_selection(self):
        """Test complete pipeline selection workflow for Stable Diffusion model."""
        # Create SD architecture signature
        signature = ArchitectureSignature(
            component_classes={
                "unet": "UNet2DConditionModel",
                "text_encoder": "CLIPTextModel",
                "tokenizer": "CLIPTokenizer"
            }
        )
        
        # Select pipeline class
        pipeline_class = self.manager.select_pipeline_class(signature)
        assert pipeline_class == "StableDiffusionPipeline"
        
        # Get requirements
        requirements = self.manager.get_pipeline_requirements(pipeline_class)
        assert requirements.requires_trust_remote_code is False
        
        # Validate arguments
        provided_args = {
            "unet": Mock(),
            "text_encoder": Mock(),
            "tokenizer": Mock(),
            "scheduler": Mock(),
            "vae": Mock()
        }
        
        validation = self.manager.validate_pipeline_args(pipeline_class, provided_args)
        assert validation.is_valid is True
    
    def test_fallback_pipeline_selection(self):
        """Test fallback pipeline selection when primary fails."""
        signature = ArchitectureSignature(
            has_transformer_2=True,
            vae_dimensions=3
        )
        
        # Primary selection
        primary_pipeline = self.manager.select_pipeline_class(signature)
        assert primary_pipeline == "WanPipeline"
        
        # Get alternatives if primary fails
        alternatives = self.manager.suggest_pipeline_alternatives(primary_pipeline, signature)
        assert len(alternatives) > 0
        assert "DiffusionPipeline" in alternatives
        assert primary_pipeline not in alternatives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])