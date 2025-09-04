"""
Tests for Compatibility Registry System

Tests cover registry operations, model-pipeline compatibility checking,
batch updates, and validation functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from compatibility_registry import (
    CompatibilityRegistry,
    PipelineRequirements,
    CompatibilityCheck,
    get_compatibility_registry
)


class TestPipelineRequirements:
    """Test PipelineRequirements data class"""
    
    def test_pipeline_requirements_creation(self):
        """Test creating PipelineRequirements instance"""
        req = PipelineRequirements(
            pipeline_class="WanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0"],
            pipeline_source="https://example.com",
            vram_requirements={"min_mb": 8192},
            supported_optimizations=["cpu_offload"]
        )
        
        assert req.pipeline_class == "WanPipeline"
        assert req.trust_remote_code is True  # default value
        assert len(req.required_dependencies) == 1
    
    def test_pipeline_requirements_serialization(self):
        """Test to_dict and from_dict methods"""
        req = PipelineRequirements(
            pipeline_class="WanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0"],
            pipeline_source="https://example.com",
            vram_requirements={"min_mb": 8192},
            supported_optimizations=["cpu_offload"],
            trust_remote_code=False
        )
        
        # Test serialization
        req_dict = req.to_dict()
        assert isinstance(req_dict, dict)
        assert req_dict["pipeline_class"] == "WanPipeline"
        assert req_dict["trust_remote_code"] is False
        
        # Test deserialization
        req_restored = PipelineRequirements.from_dict(req_dict)
        assert req_restored.pipeline_class == req.pipeline_class
        assert req_restored.trust_remote_code == req.trust_remote_code


class TestCompatibilityCheck:
    """Test CompatibilityCheck data class"""
    
    def test_compatibility_check_creation(self):
        """Test creating CompatibilityCheck instance"""
        check = CompatibilityCheck(
            is_compatible=True,
            compatibility_score=0.9,
            issues=[],
            warnings=["Low VRAM"],
            recommendations=["Use CPU offload"]
        )
        
        assert check.is_compatible is True
        assert check.compatibility_score == 0.9
        assert len(check.warnings) == 1
        assert len(check.recommendations) == 1
    
    def test_compatibility_check_serialization(self):
        """Test to_dict method"""
        check = CompatibilityCheck(
            is_compatible=False,
            compatibility_score=0.3,
            issues=["Pipeline mismatch"],
            warnings=[],
            recommendations=["Update pipeline"]
        )
        
        check_dict = check.to_dict()
        assert isinstance(check_dict, dict)
        assert check_dict["is_compatible"] is False
        assert check_dict["compatibility_score"] == 0.3
        assert len(check_dict["issues"]) == 1


class TestCompatibilityRegistry:
    """Test CompatibilityRegistry class"""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_requirements(self):
        """Sample pipeline requirements for testing"""
        return PipelineRequirements(
            pipeline_class="WanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0", "transformers>=4.25.0"],
            pipeline_source="https://huggingface.co/test-model",
            vram_requirements={"min_mb": 8192, "recommended_mb": 12288},
            supported_optimizations=["cpu_offload", "mixed_precision"]
        )
    
    def test_registry_initialization_new_file(self, temp_registry_file):
        """Test registry initialization with new file"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Should create default registry
        assert len(registry.registry) > 0
        assert Path(temp_registry_file).exists()
        
        # Check default entries
        assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in registry.registry
    
    def test_registry_initialization_existing_file(self, temp_registry_file):
        """Test registry initialization with existing file"""
        # Create test registry data
        test_data = {
            "test-model": {
                "pipeline_class": "TestPipeline",
                "min_diffusers_version": "0.20.0",
                "required_dependencies": ["torch>=1.0.0"],
                "pipeline_source": "https://test.com",
                "vram_requirements": {"min_mb": 4096},
                "supported_optimizations": ["test_opt"],
                "trust_remote_code": True
            }
        }
        
        with open(temp_registry_file, 'w') as f:
            json.dump(test_data, f)
        
        registry = CompatibilityRegistry(temp_registry_file)
        
        assert len(registry.registry) == 1
        assert "test-model" in registry.registry
        assert registry.registry["test-model"].pipeline_class == "TestPipeline"
    
    def test_get_pipeline_requirements_direct_match(self, temp_registry_file, sample_requirements):
        """Test getting pipeline requirements with direct model name match"""
        registry = CompatibilityRegistry(temp_registry_file)
        registry.register_model_compatibility("test-model", sample_requirements)
        
        result = registry.get_pipeline_requirements("test-model")
        
        assert result is not None
        assert result.pipeline_class == "WanPipeline"
        assert result.min_diffusers_version == "0.21.0"
    
    def test_get_pipeline_requirements_variant_match(self, temp_registry_file, sample_requirements):
        """Test getting pipeline requirements with variant matching"""
        registry = CompatibilityRegistry(temp_registry_file)
        registry.register_model_compatibility("test-model-diffusers", sample_requirements)
        
        # Should match variant
        result = registry.get_pipeline_requirements("test-model-pytorch")
        
        assert result is not None
        assert result.pipeline_class == "WanPipeline"
    
    def test_get_pipeline_requirements_no_match(self, temp_registry_file):
        """Test getting pipeline requirements with no match"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        result = registry.get_pipeline_requirements("nonexistent-model")
        
        assert result is None
    
    def test_register_model_compatibility(self, temp_registry_file, sample_requirements):
        """Test registering new model compatibility"""
        registry = CompatibilityRegistry(temp_registry_file)
        initial_count = len(registry.registry)
        
        registry.register_model_compatibility("new-model", sample_requirements)
        
        assert len(registry.registry) == initial_count + 1
        assert "new-model" in registry.registry
        assert registry.registry["new-model"].pipeline_class == "WanPipeline"
        
        # Verify file was saved
        assert Path(temp_registry_file).exists()
    
    def test_update_registry_batch(self, temp_registry_file):
        """Test batch registry updates"""
        registry = CompatibilityRegistry(temp_registry_file)
        initial_count = len(registry.registry)
        
        updates = {
            "model1": PipelineRequirements(
                pipeline_class="Pipeline1",
                min_diffusers_version="0.20.0",
                required_dependencies=[],
                pipeline_source="https://test1.com",
                vram_requirements={"min_mb": 4096},
                supported_optimizations=[]
            ),
            "model2": PipelineRequirements(
                pipeline_class="Pipeline2",
                min_diffusers_version="0.21.0",
                required_dependencies=[],
                pipeline_source="https://test2.com",
                vram_requirements={"min_mb": 8192},
                supported_optimizations=[]
            )
        }
        
        registry.update_registry(updates)
        
        assert len(registry.registry) == initial_count + 2
        assert "model1" in registry.registry
        assert "model2" in registry.registry
    
    def test_validate_model_pipeline_compatibility_perfect_match(self, temp_registry_file, sample_requirements):
        """Test compatibility validation with perfect match"""
        registry = CompatibilityRegistry(temp_registry_file)
        registry.register_model_compatibility("test-model", sample_requirements)
        
        result = registry.validate_model_pipeline_compatibility("test-model", "WanPipeline")
        
        assert result.is_compatible is True
        assert result.compatibility_score == 1.0
        assert len(result.issues) == 0
    
    def test_validate_model_pipeline_compatibility_pipeline_mismatch(self, temp_registry_file, sample_requirements):
        """Test compatibility validation with pipeline mismatch"""
        registry = CompatibilityRegistry(temp_registry_file)
        registry.register_model_compatibility("test-model", sample_requirements)
        
        result = registry.validate_model_pipeline_compatibility("test-model", "StableDiffusionPipeline")
        
        assert result.is_compatible is False
        assert result.compatibility_score < 1.0
        assert len(result.issues) > 0
        assert "Pipeline mismatch" in result.issues[0]
    
    def test_validate_model_pipeline_compatibility_no_registry_entry(self, temp_registry_file):
        """Test compatibility validation with no registry entry"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        result = registry.validate_model_pipeline_compatibility("unknown-model", "WanPipeline")
        
        assert result.is_compatible is False
        assert result.compatibility_score == 0.0
        assert "No compatibility information found" in result.issues[0]
        assert len(result.recommendations) > 0
    
    def test_list_registered_models(self, temp_registry_file, sample_requirements):
        """Test listing registered models"""
        registry = CompatibilityRegistry(temp_registry_file)
        registry.register_model_compatibility("test-model", sample_requirements)
        
        models = registry.list_registered_models()
        
        assert isinstance(models, list)
        assert "test-model" in models
        assert len(models) > 0
    
    def test_get_models_by_pipeline(self, temp_registry_file):
        """Test getting models by pipeline class"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Add models with different pipeline classes
        req1 = PipelineRequirements(
            pipeline_class="WanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=[],
            pipeline_source="https://test.com",
            vram_requirements={"min_mb": 8192},
            supported_optimizations=[]
        )
        
        req2 = PipelineRequirements(
            pipeline_class="StableDiffusionPipeline",
            min_diffusers_version="0.20.0",
            required_dependencies=[],
            pipeline_source="https://test.com",
            vram_requirements={"min_mb": 4096},
            supported_optimizations=[]
        )
        
        registry.register_model_compatibility("wan-model", req1)
        registry.register_model_compatibility("sd-model", req2)
        
        wan_models = registry.get_models_by_pipeline("WanPipeline")
        sd_models = registry.get_models_by_pipeline("StableDiffusionPipeline")
        
        assert "wan-model" in wan_models
        assert "sd-model" in sd_models
        assert "sd-model" not in wan_models
    
    def test_export_registry(self, temp_registry_file, sample_requirements):
        """Test registry export functionality"""
        registry = CompatibilityRegistry(temp_registry_file)
        registry.register_model_compatibility("test-model", sample_requirements)
        
        export_path = temp_registry_file + ".export"
        registry.export_registry(export_path)
        
        # Verify export file exists and has correct structure
        assert Path(export_path).exists()
        
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert "export_timestamp" in export_data
        assert "registry_version" in export_data
        assert "models" in export_data
        assert "test-model" in export_data["models"]
        
        # Cleanup
        Path(export_path).unlink()
    
    def test_import_registry_merge(self, temp_registry_file, sample_requirements):
        """Test registry import with merge"""
        registry = CompatibilityRegistry(temp_registry_file)
        initial_count = len(registry.registry)
        
        # Create import data
        import_data = {
"export_timestamp": "2024-01-01T00:00:00",
            "registry_version": "1.0",
            "models": {
                "imported-model": sample_requirements.to_dict()
            }
        }
        
        import_path = temp_registry_file + ".import"
        with open(import_path, 'w') as f:
            json.dump(import_data, f)
        
        registry.import_registry(import_path, merge=True)
        
        assert len(registry.registry) == initial_count + 1
        assert "imported-model" in registry.registry
        
        # Cleanup
        Path(import_path).unlink()
    
    def test_import_registry_replace(self, temp_registry_file, sample_requirements):
        """Test registry import with replace"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Create import data
        import_data = {
"export_timestamp": "2024-01-01T00:00:00",
            "registry_version": "1.0",
            "models": {
                "only-model": sample_requirements.to_dict()
            }
        }
        
        import_path = temp_registry_file + ".import"
        with open(import_path, 'w') as f:
            json.dump(import_data, f)
        
        registry.import_registry(import_path, merge=False)
        
        assert len(registry.registry) == 1
        assert "only-model" in registry.registry
        
        # Cleanup
        Path(import_path).unlink()
    
    def test_validate_registry_integrity(self, temp_registry_file):
        """Test registry integrity validation"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Add valid entry
        valid_req = PipelineRequirements(
            pipeline_class="WanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=[],
            pipeline_source="https://test.com",
            vram_requirements={"min_mb": 8192},
            supported_optimizations=[]
        )
        registry.register_model_compatibility("valid-model", valid_req)
        
        # Add invalid entry (manually to bypass validation)
        invalid_req = PipelineRequirements(
            pipeline_class="",  # Invalid: empty pipeline class
            min_diffusers_version="0.21.0",
            required_dependencies=[],
            pipeline_source="",  # Warning: empty source
            vram_requirements={"min_mb": 8192},
            supported_optimizations=[]
        )
        registry.registry["invalid-model"] = invalid_req
        
        report = registry.validate_registry_integrity()
        
        assert report["total_models"] == len(registry.registry)
        assert len(report["validation_errors"]) > 0
        assert len(report["validation_warnings"]) > 0
        assert "pipeline_classes" in report
        assert "WanPipeline" in report["pipeline_classes"]


class TestGlobalRegistry:
    """Test global registry access"""
    
    def test_get_compatibility_registry_singleton(self):
        """Test that global registry returns same instance"""
        registry1 = get_compatibility_registry()
        registry2 = get_compatibility_registry()
        
        assert registry1 is registry2
        assert isinstance(registry1, CompatibilityRegistry)


class TestRegistryIntegration:
    """Integration tests for registry system"""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_end_to_end_registry_workflow(self, temp_registry_file):
        """Test complete registry workflow"""
        # Create registry
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Register model
        requirements = PipelineRequirements(
            pipeline_class="WanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0"],
            pipeline_source="https://huggingface.co/test-model",
            vram_requirements={"min_mb": 8192, "recommended_mb": 12288},
            supported_optimizations=["cpu_offload", "mixed_precision"]
        )
        
        registry.register_model_compatibility("test-model", requirements)
        
        # Retrieve and validate
        retrieved = registry.get_pipeline_requirements("test-model")
        assert retrieved is not None
        assert retrieved.pipeline_class == "WanPipeline"
        
        # Check compatibility
        compat_check = registry.validate_model_pipeline_compatibility("test-model", "WanPipeline")
        assert compat_check.is_compatible is True
        
        # Export and import
        export_path = temp_registry_file + ".export"
        registry.export_registry(export_path)
        
        new_registry = CompatibilityRegistry(temp_registry_file + ".new")
        new_registry.import_registry(export_path, merge=False)
        
        # Verify import worked
        imported = new_registry.get_pipeline_requirements("test-model")
        assert imported is not None
        assert imported.pipeline_class == "WanPipeline"
        
        # Cleanup
        Path(export_path).unlink()
        Path(temp_registry_file + ".new").unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])