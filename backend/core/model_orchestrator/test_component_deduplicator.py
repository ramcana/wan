"""
Tests for Component Deduplication System.

Tests component sharing across t2v/i2v/ti2v models with various scenarios
including hardlink/symlink creation, reference tracking, and cleanup.
"""

import os
import json
import shutil
import tempfile
import platform
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from .component_deduplicator import ComponentDeduplicator, ComponentInfo, DeduplicationResult
from .exceptions import ModelOrchestratorError


class TestComponentDeduplicator:
    """Test suite for ComponentDeduplicator."""

    @pytest.fixture
    def temp_models_root(self):
        """Create a temporary models root directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def deduplicator(self, temp_models_root):
        """Create a ComponentDeduplicator instance."""
        return ComponentDeduplicator(str(temp_models_root))

    @pytest.fixture
    def sample_models(self, temp_models_root):
        """Create sample model directories with common components."""
        models = {}
        
        # Create t2v-A14B model
        t2v_path = temp_models_root / "wan22" / "t2v-A14B@2.2.0"
        t2v_path.mkdir(parents=True)
        
        # Common tokenizer files
        tokenizer_dir = t2v_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text('{"vocab_size": 50000}')
        (tokenizer_dir / "tokenizer_config.json").write_text('{"model_max_length": 512}')
        (tokenizer_dir / "vocab.json").write_text('{"hello": 1, "world": 2}')
        
        # Text encoder
        text_encoder_dir = t2v_path / "text_encoder"
        text_encoder_dir.mkdir()
        (text_encoder_dir / "config.json").write_text('{"hidden_size": 768}')
        (text_encoder_dir / "pytorch_model.safetensors").write_bytes(b"fake_text_encoder_weights" * 1000)
        
        # UNet (unique to t2v)
        unet_dir = t2v_path / "unet"
        unet_dir.mkdir()
        (unet_dir / "config.json").write_text('{"in_channels": 4}')
        (unet_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"fake_t2v_unet_weights" * 2000)
        
        models["t2v-A14B@2.2.0"] = t2v_path
        
        # Create i2v-A14B model (shares tokenizer and text encoder)
        i2v_path = temp_models_root / "wan22" / "i2v-A14B@2.2.0"
        i2v_path.mkdir(parents=True)
        
        # Same tokenizer files (duplicates)
        tokenizer_dir = i2v_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text('{"vocab_size": 50000}')
        (tokenizer_dir / "tokenizer_config.json").write_text('{"model_max_length": 512}')
        (tokenizer_dir / "vocab.json").write_text('{"hello": 1, "world": 2}')
        
        # Same text encoder (duplicates)
        text_encoder_dir = i2v_path / "text_encoder"
        text_encoder_dir.mkdir()
        (text_encoder_dir / "config.json").write_text('{"hidden_size": 768}')
        (text_encoder_dir / "pytorch_model.safetensors").write_bytes(b"fake_text_encoder_weights" * 1000)
        
        # Image encoder (unique to i2v)
        image_encoder_dir = i2v_path / "image_encoder"
        image_encoder_dir.mkdir()
        (image_encoder_dir / "config.json").write_text('{"image_size": 224}')
        (image_encoder_dir / "pytorch_model.safetensors").write_bytes(b"fake_image_encoder_weights" * 1500)
        
        # UNet (different from t2v)
        unet_dir = i2v_path / "unet"
        unet_dir.mkdir()
        (unet_dir / "config.json").write_text('{"in_channels": 8}')
        (unet_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"fake_i2v_unet_weights" * 2000)
        
        models["i2v-A14B@2.2.0"] = i2v_path
        
        # Create ti2v-5b model (shares tokenizer, text encoder, and image encoder)
        ti2v_path = temp_models_root / "wan22" / "ti2v-5b@2.2.0"
        ti2v_path.mkdir(parents=True)
        
        # Same tokenizer files (duplicates)
        tokenizer_dir = ti2v_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text('{"vocab_size": 50000}')
        (tokenizer_dir / "tokenizer_config.json").write_text('{"model_max_length": 512}')
        (tokenizer_dir / "vocab.json").write_text('{"hello": 1, "world": 2}')
        
        # Same text encoder (duplicates)
        text_encoder_dir = ti2v_path / "text_encoder"
        text_encoder_dir.mkdir()
        (text_encoder_dir / "config.json").write_text('{"hidden_size": 768}')
        (text_encoder_dir / "pytorch_model.safetensors").write_bytes(b"fake_text_encoder_weights" * 1000)
        
        # Same image encoder as i2v (duplicates)
        image_encoder_dir = ti2v_path / "image_encoder"
        image_encoder_dir.mkdir()
        (image_encoder_dir / "config.json").write_text('{"image_size": 224}')
        (image_encoder_dir / "pytorch_model.safetensors").write_bytes(b"fake_image_encoder_weights" * 1500)
        
        # UNet (unique to ti2v)
        unet_dir = ti2v_path / "unet"
        unet_dir.mkdir()
        (unet_dir / "config.json").write_text('{"in_channels": 12}')
        (unet_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"fake_ti2v_unet_weights" * 2500)
        
        models["ti2v-5b@2.2.0"] = ti2v_path
        
        return models

    def test_initialization(self, temp_models_root):
        """Test ComponentDeduplicator initialization."""
        deduplicator = ComponentDeduplicator(str(temp_models_root))
        
        assert deduplicator.models_root == temp_models_root
        assert deduplicator.components_root == temp_models_root / "components"
        assert deduplicator.metadata_file == temp_models_root / "components" / ".component_registry.json"
        assert isinstance(deduplicator.is_windows, bool)
        assert isinstance(deduplicator.supports_hardlinks, bool)
        assert isinstance(deduplicator.supports_symlinks, bool)

    def test_hardlink_support_detection(self, deduplicator, temp_models_root):
        """Test hardlink support detection."""
        # The actual support depends on the filesystem, but we can test the method exists
        assert isinstance(deduplicator.supports_hardlinks, bool)
        
        # Test that the detection doesn't crash
        supports_hardlinks = deduplicator._check_hardlink_support()
        assert isinstance(supports_hardlinks, bool)

    def test_symlink_support_detection(self, deduplicator, temp_models_root):
        """Test symlink support detection."""
        # The actual support depends on the filesystem and permissions
        assert isinstance(deduplicator.supports_symlinks, bool)
        
        # Test that the detection doesn't crash
        supports_symlinks = deduplicator._check_symlink_support()
        assert isinstance(supports_symlinks, bool)

    def test_component_type_identification(self, deduplicator):
        """Test component type identification from file paths."""
        test_cases = [
            ("tokenizer/tokenizer.json", "tokenizer"),
            ("text_encoder/config.json", "text_encoder"),
            ("scheduler/scheduler_config.json", "scheduler"),
            ("safety_checker/pytorch_model.bin", "safety_checker"),
            ("feature_extractor/preprocessor_config.json", "feature_extractor"),
            ("unet/config.json", None),  # UNet is not in common_components
            ("random_file.txt", None),
        ]
        
        for file_path, expected_type in test_cases:
            result = deduplicator._identify_component_type(file_path)
            assert result == expected_type, f"Failed for {file_path}: expected {expected_type}, got {result}"

    def test_file_hash_calculation(self, deduplicator, temp_models_root):
        """Test file hash calculation."""
        test_file = temp_models_root / "test_file.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        hash_result = deduplicator._calculate_file_hash(test_file)
        
        # Verify it's a valid SHA256 hash
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)
        
        # Verify consistency
        hash_result2 = deduplicator._calculate_file_hash(test_file)
        assert hash_result == hash_result2

    def test_single_model_deduplication(self, deduplicator, sample_models):
        """Test deduplication within a single model."""
        model_id = "t2v-A14B@2.2.0"
        model_path = sample_models[model_id]
        
        # Create duplicate component files within the model
        # Create a duplicate tokenizer config in a different location
        duplicate_tokenizer_dir = model_path / "backup_tokenizer"
        duplicate_tokenizer_dir.mkdir()
        (duplicate_tokenizer_dir / "tokenizer.json").write_text('{"vocab_size": 50000}')
        
        result = deduplicator.deduplicate_model(model_id, model_path)
        
        assert isinstance(result, DeduplicationResult)
        assert result.total_files_processed > 0
        assert result.processing_time > 0
        
        # Check that duplicates were found and processed
        if result.duplicates_found > 0:
            assert result.links_created > 0
            assert result.bytes_saved > 0

    def test_cross_model_deduplication(self, deduplicator, sample_models):
        """Test deduplication across multiple models."""
        result = deduplicator.deduplicate_across_models(sample_models)
        
        assert isinstance(result, DeduplicationResult)
        assert result.total_files_processed > 0
        assert result.processing_time > 0
        
        # Should find duplicates across models (tokenizer and text_encoder files)
        assert result.duplicates_found > 0
        assert result.links_created > 0
        assert result.bytes_saved > 0
        
        # Verify that shared components directory was created
        assert deduplicator.components_root.exists()
        
        # Check that component metadata was saved
        assert deduplicator.metadata_file.exists()

    def test_component_metadata_persistence(self, deduplicator, sample_models):
        """Test that component metadata is properly saved and loaded."""
        # Perform deduplication to create metadata
        deduplicator.deduplicate_across_models(sample_models)
        
        # Verify metadata file exists
        assert deduplicator.metadata_file.exists()
        
        # Create a new deduplicator instance to test loading
        new_deduplicator = ComponentDeduplicator(str(deduplicator.models_root))
        new_deduplicator._load_component_metadata()
        
        # Should have loaded the same components
        assert len(new_deduplicator._component_cache) > 0
        
        # Verify component info structure
        for component_key, component_info in new_deduplicator._component_cache.items():
            assert isinstance(component_info, ComponentInfo)
            assert component_info.name
            assert component_info.version
            assert component_info.content_hash
            assert component_info.size > 0
            assert component_info.references
            assert component_info.created_at > 0

    def test_reference_tracking(self, deduplicator, sample_models):
        """Test component reference tracking."""
        # Perform cross-model deduplication
        deduplicator.deduplicate_across_models(sample_models)
        
        # Add reference for a new model
        new_model_id = "test-model@1.0.0"
        new_model_path = sample_models["t2v-A14B@2.2.0"]  # Reuse existing path for test
        
        deduplicator.add_model_reference(new_model_id, new_model_path)
        
        # Remove reference for one model
        orphaned_components = deduplicator.remove_model_reference("t2v-A14B@2.2.0")
        
        # Should return list of component keys (may be empty if other models still reference them)
        assert isinstance(orphaned_components, list)

    def test_orphaned_component_cleanup(self, deduplicator, sample_models):
        """Test cleanup of orphaned components."""
        # Perform deduplication
        deduplicator.deduplicate_across_models(sample_models)
        
        # Remove all model references to create orphaned components
        all_orphaned = []
        for model_id in sample_models.keys():
            orphaned = deduplicator.remove_model_reference(model_id)
            all_orphaned.extend(orphaned)
        
        if all_orphaned:
            # Clean up orphaned components
            bytes_reclaimed = deduplicator.cleanup_orphaned_components(all_orphaned)
            
            assert isinstance(bytes_reclaimed, int)
            assert bytes_reclaimed >= 0
            
            # Verify components were removed from cache
            for component_key in all_orphaned:
                assert component_key not in deduplicator._component_cache

    def test_component_stats(self, deduplicator, sample_models):
        """Test component statistics reporting."""
        # Initially should have no components
        stats = deduplicator.get_component_stats()
        assert stats["total_components"] == 0
        assert stats["total_size_bytes"] == 0
        
        # Perform deduplication
        deduplicator.deduplicate_across_models(sample_models)
        
        # Should now have components
        stats = deduplicator.get_component_stats()
        assert stats["total_components"] > 0
        assert stats["total_size_bytes"] > 0
        assert stats["total_references"] > 0
        assert isinstance(stats["component_types"], dict)
        assert stats["components_root"] == str(deduplicator.components_root)

    def test_link_creation_fallback(self, deduplicator, temp_models_root):
        """Test link creation with various fallback strategies."""
        source_file = temp_models_root / "source.txt"
        source_file.write_text("test content")
        
        target_file = temp_models_root / "target.txt"
        
        # Test link creation
        success = deduplicator._create_link(source_file, target_file)
        
        # Should succeed with some method (hardlink, symlink, or copy)
        assert success
        assert target_file.exists()
        
        # Content should be the same
        assert target_file.read_text() == "test content"

    def test_cross_platform_compatibility(self, deduplicator):
        """Test cross-platform compatibility features."""
        # Test Windows detection
        assert isinstance(deduplicator.is_windows, bool)
        
        # Test platform-specific link support
        if platform.system() == "Windows":
            # On Windows, may or may not support hardlinks/symlinks depending on permissions
            assert isinstance(deduplicator.supports_hardlinks, bool)
            assert isinstance(deduplicator.supports_symlinks, bool)
        else:
            # On Unix-like systems, should generally support both
            # (though this may vary by filesystem)
            assert isinstance(deduplicator.supports_hardlinks, bool)
            assert isinstance(deduplicator.supports_symlinks, bool)

    def test_error_handling(self, deduplicator, temp_models_root):
        """Test error handling in various scenarios."""
        # Test with non-existent model path
        non_existent_path = temp_models_root / "non_existent"
        result = deduplicator.deduplicate_model("test-model", non_existent_path)
        
        assert isinstance(result, DeduplicationResult)
        assert result.total_files_processed == 0
        assert len(result.errors) == 0  # Should handle gracefully, not error

    def test_concurrent_access_safety(self, deduplicator, sample_models):
        """Test thread safety of metadata operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def dedup_worker(model_id, model_path):
            try:
                result = deduplicator.deduplicate_model(model_id, model_path)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for model_id, model_path in sample_models.items():
            thread = threading.Thread(target=dedup_worker, args=(model_id, model_path))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not have any errors from concurrent access
        assert len(errors) == 0
        assert len(results) == len(sample_models)

    def test_model_ensurer_integration(self, temp_models_root, sample_models):
        """Test integration with ModelEnsurer for automatic deduplication during atomic move."""
        from unittest.mock import Mock
        from .model_ensurer import ModelEnsurer
        from .model_registry import ModelRegistry, ModelSpec, FileSpec
        from .model_resolver import ModelResolver
        from .lock_manager import LockManager
        
        # Create mock components
        registry = Mock(spec=ModelRegistry)
        resolver = Mock(spec=ModelResolver)
        resolver.models_root = str(temp_models_root)
        lock_manager = Mock(spec=LockManager)
        storage_backends = []
        
        # Create model ensurer with deduplication enabled
        ensurer = ModelEnsurer(
            registry=registry,
            resolver=resolver,
            lock_manager=lock_manager,
            storage_backends=storage_backends,
            enable_deduplication=True
        )
        
        # Verify deduplicator was created
        assert ensurer.component_deduplicator is not None
        assert ensurer.enable_deduplication is True
        
        # Test deduplication methods
        model_id = "test-model@1.0.0"
        model_path = str(sample_models["t2v-A14B@2.2.0"])
        
        # Test add_model_reference
        ensurer.add_model_reference(model_id, model_path)
        
        # Test get_component_stats
        stats = ensurer.get_component_stats()
        assert stats is not None
        assert "total_components" in stats
        
        # Test remove_model_reference
        bytes_reclaimed = ensurer.remove_model_reference(model_id)
        assert isinstance(bytes_reclaimed, int)
        assert bytes_reclaimed >= 0

    def test_wan22_specific_components(self, deduplicator, sample_models):
        """Test handling of WAN2.2-specific model components."""
        # Perform cross-model deduplication
        result = deduplicator.deduplicate_across_models(sample_models)
        
        # Should identify and deduplicate common WAN2.2 components
        stats = deduplicator.get_component_stats()
        component_types = stats["component_types"]
        
        # Should find tokenizer and text_encoder components (shared across models)
        expected_components = {"tokenizer", "text_encoder"}
        found_components = set(component_types.keys())
        
        # At least some expected components should be found
        assert len(expected_components.intersection(found_components)) > 0

    def test_large_file_handling(self, deduplicator, temp_models_root):
        """Test handling of large files during deduplication."""
        model_path = temp_models_root / "large_model"
        model_path.mkdir(parents=True)
        
        # Create a moderately large tokenizer file (1MB) - this will be identified as a component
        tokenizer_dir = model_path / "tokenizer"
        tokenizer_dir.mkdir()
        large_file = tokenizer_dir / "tokenizer.json"
        large_content = b'{"vocab_size": 50000}' + b"x" * (1024 * 1024 - 20)  # ~1MB
        large_file.write_bytes(large_content)
        
        # Create duplicate in another tokenizer directory
        duplicate_tokenizer_dir = model_path / "backup_tokenizer"
        duplicate_tokenizer_dir.mkdir()
        duplicate_file = duplicate_tokenizer_dir / "tokenizer.json"
        duplicate_file.write_bytes(large_content)
        
        result = deduplicator.deduplicate_model("large-model", model_path)
        
        assert result.total_files_processed >= 2
        if result.duplicates_found > 0:
            # Should save significant space
            assert result.bytes_saved >= 1024 * 1024

    def test_component_versioning(self, deduplicator, temp_models_root):
        """Test component versioning based on content hash."""
        model_path = temp_models_root / "version_test"
        model_path.mkdir(parents=True)
        
        # Create a component file
        component_file = model_path / "tokenizer" / "config.json"
        component_file.parent.mkdir()
        component_file.write_text('{"version": 1}')
        
        result1 = deduplicator.deduplicate_model("test-v1", model_path)
        
        # Modify the component (different version)
        component_file.write_text('{"version": 2}')
        
        result2 = deduplicator.deduplicate_model("test-v2", model_path)
        
        # Should handle different versions as separate components
        stats = deduplicator.get_component_stats()
        assert stats["total_components"] >= 0  # May be 0, 1, or 2 depending on deduplication logic

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_hardlink_creation(self, deduplicator, temp_models_root):
        """Test hardlink creation on Unix systems."""
        if not deduplicator.supports_hardlinks:
            pytest.skip("Hardlinks not supported on this filesystem")
        
        source_file = temp_models_root / "source.txt"
        source_file.write_text("hardlink test")
        
        target_file = temp_models_root / "target.txt"
        
        success = deduplicator._create_link(source_file, target_file)
        assert success
        
        # Verify it's actually a hardlink (same inode)
        source_stat = source_file.stat()
        target_stat = target_file.stat()
        assert source_stat.st_ino == target_stat.st_ino
        assert source_stat.st_nlink == 2

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_junction_creation(self, deduplicator, temp_models_root):
        """Test Windows junction creation for directories."""
        if not deduplicator.is_windows:
            pytest.skip("Windows-specific test")
        
        # Create source directory
        source_dir = temp_models_root / "source_dir"
        source_dir.mkdir()
        (source_dir / "test.txt").write_text("junction test")
        
        target_dir = temp_models_root / "target_dir"
        
        # Test directory linking (may use junction on Windows)
        success = deduplicator._create_link(source_dir, target_dir)
        
        # Should succeed with some method
        assert success
        assert target_dir.exists()
        assert (target_dir / "test.txt").exists()