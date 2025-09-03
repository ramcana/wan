"""
Integration tests for Compatibility Registry System

Tests integration with other components and real-world usage scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from compatibility_registry import (
    CompatibilityRegistry,
    PipelineRequirements,
    get_compatibility_registry
)


class TestRegistryIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_model_loading_workflow_integration(self, temp_registry_file):
        """Test integration with model loading workflow"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Simulate model loading workflow
        model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Step 1: Check if model is registered
        requirements = registry.get_pipeline_requirements(model_path)
        assert requirements is not None
        assert requirements.pipeline_class == "WanPipeline"
        
        # Step 2: Validate compatibility with available pipeline
        available_pipeline = "WanPipeline"
        compat_check = registry.validate_model_pipeline_compatibility(
            model_path, available_pipeline
        )
        assert compat_check.is_compatible is True
        
        # Step 3: Get optimization recommendations
        optimizations = requirements.supported_optimizations
        assert "cpu_offload" in optimizations
        assert "mixed_precision" in optimizations
        
        # Step 4: Check VRAM requirements
        vram_req = requirements.vram_requirements
        assert vram_req["min_mb"] == 8192
        assert vram_req["recommended_mb"] == 12288
    
    def test_fallback_pipeline_selection(self, temp_registry_file):
        """Test fallback pipeline selection based on compatibility"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Test with incompatible pipeline
        compat_check = registry.validate_model_pipeline_compatibility(
            model_path, "StableDiffusionPipeline"
        )
        
        assert compat_check.is_compatible is False
        assert len(compat_check.issues) > 0
        assert "Pipeline mismatch" in compat_check.issues[0]
        
        # Recommendations should suggest correct pipeline
        assert len(compat_check.recommendations) > 0
    
    def test_custom_model_registration_workflow(self, temp_registry_file):
        """Test workflow for registering custom models"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Simulate discovering a new custom model
        custom_model = "user/custom-wan-model"
        
        # Initially not registered
        requirements = registry.get_pipeline_requirements(custom_model)
        assert requirements is None
        
        # Register the model
        custom_requirements = PipelineRequirements(
            pipeline_class="CustomWanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0", "custom-lib>=1.0.0"],
            pipeline_source="https://huggingface.co/user/custom-wan-model",
            vram_requirements={"min_mb": 6144, "recommended_mb": 8192},
            supported_optimizations=["cpu_offload", "mixed_precision"]
        )
        
        registry.register_model_compatibility(custom_model, custom_requirements)
        
        # Verify registration
        retrieved = registry.get_pipeline_requirements(custom_model)
        assert retrieved is not None
        assert retrieved.pipeline_class == "CustomWanPipeline"
        assert len(retrieved.required_dependencies) == 2
    
    def test_model_variant_detection(self, temp_registry_file):
        """Test detection of model variants"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Test variant matching for different model formats
        base_model = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        variants = [
            "Wan-AI/Wan2.2-T2V-A14B-PyTorch",
            "Wan-AI/Wan2.2-T2V-A14B-SafeTensors",
            "Wan-AI/Wan2.2-T2V-A14B-FP16",
            "local/Wan2.2-T2V-A14B-Custom"
        ]
        
        # Base model should be registered
        base_requirements = registry.get_pipeline_requirements(base_model)
        assert base_requirements is not None
        
        # Variants should match to base model
        for variant in variants:
            variant_requirements = registry.get_pipeline_requirements(variant)
            if variant_requirements:  # Some variants might match
                assert variant_requirements.pipeline_class == base_requirements.pipeline_class
    
    def test_optimization_recommendation_integration(self, temp_registry_file):
        """Test integration with optimization system"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        requirements = registry.get_pipeline_requirements(model_path)
        
        # Simulate different VRAM scenarios
        min_vram = requirements.vram_requirements["min_mb"]
        recommended_vram = requirements.vram_requirements["recommended_mb"]
        
        # Test optimization recommendations based on VRAM
        optimizations = requirements.supported_optimizations
        
        # Should include memory-saving optimizations
        assert "cpu_offload" in optimizations
        assert "mixed_precision" in optimizations
        
        # For T2V models, should include chunked processing
        if "T2V" in model_path:
            assert "chunked_processing" in optimizations
    
    def test_batch_model_update_scenario(self, temp_registry_file):
        """Test batch update scenario for model releases"""
        registry = CompatibilityRegistry(temp_registry_file)
        initial_count = len(registry.registry)
        
        # Simulate batch update for new model release
        new_models = {
            "Wan-AI/Wan2.3-T2V-A16B-Diffusers": PipelineRequirements(
                pipeline_class="Wan23Pipeline",
                min_diffusers_version="0.22.0",
                required_dependencies=["torch>=2.1.0", "transformers>=4.30.0"],
                pipeline_source="https://huggingface.co/Wan-AI/Wan2.3-T2V-A16B-Diffusers",
                vram_requirements={"min_mb": 10240, "recommended_mb": 16384},
                supported_optimizations=["cpu_offload", "mixed_precision", "chunked_processing"]
            ),
            "Wan-AI/Wan2.3-T2I-A16B-Diffusers": PipelineRequirements(
                pipeline_class="Wan23Pipeline",
                min_diffusers_version="0.22.0",
                required_dependencies=["torch>=2.1.0", "transformers>=4.30.0"],
                pipeline_source="https://huggingface.co/Wan-AI/Wan2.3-T2I-A16B-Diffusers",
                vram_requirements={"min_mb": 8192, "recommended_mb": 12288},
                supported_optimizations=["cpu_offload", "mixed_precision"]
            )
        }
        
        registry.update_registry(new_models)
        
        # Verify batch update
        assert len(registry.registry) == initial_count + 2
        
        # Verify new models are accessible
        for model_name in new_models.keys():
            requirements = registry.get_pipeline_requirements(model_name)
            assert requirements is not None
            assert requirements.pipeline_class == "Wan23Pipeline"
    
    def test_registry_persistence_across_sessions(self, temp_registry_file):
        """Test registry persistence across different sessions"""
        # Session 1: Create and populate registry
        registry1 = CompatibilityRegistry(temp_registry_file)
        
        custom_model = "session-test/model"
        custom_requirements = PipelineRequirements(
            pipeline_class="SessionTestPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0"],
            pipeline_source="https://test.com",
            vram_requirements={"min_mb": 4096},
            supported_optimizations=["cpu_offload"]
        )
        
        registry1.register_model_compatibility(custom_model, custom_requirements)
        
        # Session 2: Load existing registry
        registry2 = CompatibilityRegistry(temp_registry_file)
        
        # Verify data persisted
        retrieved = registry2.get_pipeline_requirements(custom_model)
        assert retrieved is not None
        assert retrieved.pipeline_class == "SessionTestPipeline"
        assert retrieved.vram_requirements["min_mb"] == 4096
    
    def test_error_handling_integration(self, temp_registry_file):
        """Test error handling in integration scenarios"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Test with corrupted model path
        corrupted_path = "definitely-nonexistent-model-12345"
        requirements = registry.get_pipeline_requirements(corrupted_path)
        assert requirements is None
        
        # Test compatibility check with None model
        compat_check = registry.validate_model_pipeline_compatibility(
            "nonexistent-model", "WanPipeline"
        )
        assert compat_check.is_compatible is False
        assert len(compat_check.issues) > 0
        assert "No compatibility information found" in compat_check.issues[0]
        
        # Test with invalid pipeline class
        compat_check = registry.validate_model_pipeline_compatibility(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", ""
        )
        assert compat_check.is_compatible is False


class TestRegistryPerformance:
    """Test registry performance characteristics"""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_large_registry_performance(self, temp_registry_file):
        """Test performance with large number of models"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Add many models to test performance
        batch_size = 100
        batch_updates = {}
        
        for i in range(batch_size):
            model_name = f"test-org/model-{i:03d}"
            requirements = PipelineRequirements(
                pipeline_class=f"TestPipeline{i % 5}",  # 5 different pipeline types
                min_diffusers_version="0.21.0",
                required_dependencies=[f"dep{i % 3}>=1.0.0"],  # 3 different deps
                pipeline_source=f"https://test.com/model-{i}",
                vram_requirements={"min_mb": 4096 + (i % 4) * 2048},  # Varying VRAM
                supported_optimizations=["cpu_offload", "mixed_precision"]
            )
            batch_updates[model_name] = requirements
        
        # Measure batch update performance
        import time
start_time = time.time()
        registry.update_registry(batch_updates)
        update_time = time.time() - start_time
        
        # Should complete reasonably quickly (less than 1 second for 100 models)
        assert update_time < 1.0
        assert len(registry.registry) >= batch_size
        
        # Test lookup performance
        start_time = time.time()
        for i in range(10):  # Test 10 lookups
            model_name = f"test-org/model-{i:03d}"
            requirements = registry.get_pipeline_requirements(model_name)
            assert requirements is not None
        lookup_time = time.time() - start_time
        
        # Lookups should be very fast
        assert lookup_time < 0.1
    
    def test_registry_memory_usage(self, temp_registry_file):
        """Test registry memory usage characteristics"""
        registry = CompatibilityRegistry(temp_registry_file)
        
        # Registry should not consume excessive memory
        import sys
initial_size = sys.getsizeof(registry.registry)
        
        # Add some models
        for i in range(10):
            model_name = f"memory-test/model-{i}"
            requirements = PipelineRequirements(
                pipeline_class="TestPipeline",
                min_diffusers_version="0.21.0",
                required_dependencies=["torch>=2.0.0"],
                pipeline_source="https://test.com",
                vram_requirements={"min_mb": 4096},
                supported_optimizations=["cpu_offload"]
            )
            registry.register_model_compatibility(model_name, requirements)
        
        final_size = sys.getsizeof(registry.registry)
        
        # Memory growth should be reasonable
        growth_ratio = final_size / initial_size if initial_size > 0 else 1
        assert growth_ratio < 10  # Should not grow more than 10x for 10 models


if __name__ == "__main__":
    pytest.main([__file__])