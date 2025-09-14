"""
Tests for the Garbage Collector.
"""

import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from .garbage_collector import (
    GarbageCollector, GCConfig, GCResult, GCTrigger, 
    ModelInfo, DiskUsage
)
from .model_registry import ModelRegistry
from .model_resolver import ModelResolver
from .exceptions import ModelOrchestratorError


@pytest.fixture
def temp_models_dir():
    """Create a temporary models directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_registry():
    """Create a mock model registry."""
    registry = Mock(spec=ModelRegistry)
    return registry


@pytest.fixture
def mock_resolver(temp_models_dir):
    """Create a mock model resolver."""
    resolver = Mock(spec=ModelResolver)
    resolver.models_root = str(temp_models_dir)
    return resolver


@pytest.fixture
def gc_config():
    """Create a test garbage collection configuration."""
    return GCConfig(
        max_total_size=1024 * 1024 * 1024,  # 1GB
        max_model_age=7 * 24 * 3600,  # 7 days
        low_disk_threshold=0.1,  # 10% free space
        safety_margin_bytes=100 * 1024 * 1024,  # 100MB
        pin_recent_models=True,
        recent_access_threshold=24 * 3600,  # 24 hours
        enable_auto_gc=True
    )


@pytest.fixture
def garbage_collector(mock_registry, mock_resolver, gc_config):
    """Create a garbage collector instance."""
    return GarbageCollector(mock_registry, mock_resolver, gc_config)


@pytest.fixture
def sample_models(temp_models_dir):
    """Create sample model directories for testing."""
    models_dir = temp_models_dir / "wan22"
    models_dir.mkdir(parents=True)
    
    # Create test models with different sizes and access times
    models = []
    
    # Recent model (should be auto-pinned)
    recent_model = models_dir / "t2v-A14B"
    recent_model.mkdir()
    (recent_model / "model.safetensors").write_bytes(b"x" * 1000)
    models.append(("t2v-A14B", None, recent_model, time.time() - 3600))  # 1 hour ago
    
    # Old model (candidate for removal)
    old_model = models_dir / "i2v-A14B@fp16"
    old_model.mkdir()
    (old_model / "model.safetensors").write_bytes(b"x" * 2000)
    models.append(("i2v-A14B", "fp16", old_model, time.time() - 10 * 24 * 3600))  # 10 days ago
    
    # Medium age model
    medium_model = models_dir / "ti2v-5b"
    medium_model.mkdir()
    (medium_model / "model.safetensors").write_bytes(b"x" * 1500)
    models.append(("ti2v-5b", None, medium_model, time.time() - 3 * 24 * 3600))  # 3 days ago
    
    # Set access times
    for model_id, variant, path, access_time in models:
        os.utime(path, (access_time, access_time))
        
        # Create verification file with access time
        verification_file = path / ".verified.json"
        verification_data = {
            "model_id": model_id,
            "verified_at": access_time,
            "files": []
        }
        with open(verification_file, 'w') as f:
            json.dump(verification_data, f)
        os.utime(verification_file, (access_time, access_time))
    
    return models


class TestGarbageCollector:
    """Test cases for the GarbageCollector class."""
    
    def test_init(self, mock_registry, mock_resolver, gc_config):
        """Test garbage collector initialization."""
        gc = GarbageCollector(mock_registry, mock_resolver, gc_config)
        
        assert gc.registry == mock_registry
        assert gc.resolver == mock_resolver
        assert gc.config == gc_config
        assert isinstance(gc._pinned_models, set)
    
    def test_pin_unpin_model(self, garbage_collector):
        """Test pinning and unpinning models."""
        model_id = "test-model"
        variant = "fp16"
        
        # Initially not pinned
        assert not garbage_collector.is_pinned(model_id, variant)
        
        # Pin the model
        garbage_collector.pin_model(model_id, variant)
        assert garbage_collector.is_pinned(model_id, variant)
        
        # Unpin the model
        garbage_collector.unpin_model(model_id, variant)
        assert not garbage_collector.is_pinned(model_id, variant)
    
    def test_pin_model_without_variant(self, garbage_collector):
        """Test pinning models without variants."""
        model_id = "test-model"
        
        garbage_collector.pin_model(model_id)
        assert garbage_collector.is_pinned(model_id)
        assert garbage_collector.is_pinned(model_id, None)
    
    @patch('shutil.disk_usage')
    def test_get_disk_usage(self, mock_disk_usage, garbage_collector, temp_models_dir):
        """Test disk usage calculation."""
        # Mock disk usage - shutil.disk_usage returns a named tuple
        from collections import namedtuple
        DiskUsageTuple = namedtuple('usage', ['total', 'used', 'free'])
        mock_disk_usage.return_value = DiskUsageTuple(
            total=1000000000,  # total
            used=800000000,    # used
            free=200000000     # free
        )
        
        # Create some test files
        test_file = temp_models_dir / "test_file.txt"
        test_file.write_bytes(b"x" * 1000)
        
        usage = garbage_collector.get_disk_usage()
        
        assert usage.total_bytes == 1000000000
        assert usage.free_bytes == 200000000
        assert usage.used_bytes == 800000000
        assert usage.usage_percentage == 80.0
        assert usage.models_bytes >= 1000  # At least the test file
    
    def test_discover_models(self, garbage_collector, sample_models):
        """Test model discovery."""
        models = garbage_collector._discover_models()
        
        assert len(models) == 3
        
        # Check that all models were discovered
        model_keys = {garbage_collector._get_model_key(m.model_id, m.variant) for m in models}
        expected_keys = {"t2v-A14B", "i2v-A14B@fp16", "ti2v-5b"}
        assert model_keys == expected_keys
        
        # Check model info
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.size_bytes > 0
            assert model.last_accessed > 0
    
    def test_auto_pin_recent_models(self, garbage_collector, sample_models):
        """Test automatic pinning of recently accessed models."""
        models = garbage_collector._discover_models()
        
        # Auto-pin recent models
        garbage_collector._auto_pin_recent_models(models)
        
        # Recent model should be pinned
        assert garbage_collector.is_pinned("t2v-A14B")
        
        # Old models should not be pinned
        assert not garbage_collector.is_pinned("i2v-A14B", "fp16")
        assert not garbage_collector.is_pinned("ti2v-5b")
    
    def test_select_removal_candidates_by_age(self, garbage_collector, sample_models):
        """Test selection of removal candidates based on age."""
        models = garbage_collector._discover_models()
        
        # Set max age to 5 days
        garbage_collector.config.max_model_age = 5 * 24 * 3600
        
        candidates = garbage_collector._select_removal_candidates(models, dry_run=True)
        
        # Only the 10-day-old model should be selected
        assert len(candidates) == 1
        assert candidates[0].model_id == "i2v-A14B"
        assert candidates[0].variant == "fp16"
    
    def test_select_removal_candidates_by_size(self, garbage_collector, sample_models):
        """Test selection of removal candidates based on total size."""
        models = garbage_collector._discover_models()
        
        # Set max total size very small to force removal
        garbage_collector.config.max_total_size = 2000  # Smaller than total size
        garbage_collector.config.max_model_age = None  # Disable age-based removal
        
        candidates = garbage_collector._select_removal_candidates(models, dry_run=True)
        
        # Should select oldest models first (LRU)
        assert len(candidates) >= 1
        # Oldest model should be first candidate
        oldest_candidate = min(candidates, key=lambda x: x.last_accessed)
        assert oldest_candidate.model_id == "i2v-A14B"
    
    def test_select_removal_candidates_respects_pinned(self, garbage_collector, sample_models):
        """Test that pinned models are not selected for removal."""
        # Pin the oldest model first
        garbage_collector.pin_model("i2v-A14B", "fp16")
        
        # Then discover models (so pinned status is correct)
        models = garbage_collector._discover_models()
        
        # Set max total size very small
        garbage_collector.config.max_total_size = 1000
        garbage_collector.config.max_model_age = None
        
        candidates = garbage_collector._select_removal_candidates(models, dry_run=True)
        
        # Pinned model should not be in candidates
        pinned_in_candidates = any(
            c.model_id == "i2v-A14B" and c.variant == "fp16" 
            for c in candidates
        )
        assert not pinned_in_candidates
    
    @patch('shutil.disk_usage')
    def test_select_removal_candidates_by_disk_space(self, mock_disk_usage, garbage_collector, sample_models):
        """Test selection based on low disk space."""
        # Mock low disk space (95% used)
        from collections import namedtuple
        DiskUsageTuple = namedtuple('usage', ['total', 'used', 'free'])
        mock_disk_usage.return_value = DiskUsageTuple(
            total=1000000000,  # total
            used=950000000,    # used
            free=50000000      # free (5%)
        )
        
        models = garbage_collector._discover_models()
        
        # Disable other removal criteria
        garbage_collector.config.max_total_size = None
        garbage_collector.config.max_model_age = None
        
        candidates = garbage_collector._select_removal_candidates(models, dry_run=True)
        
        # Should select models to free up space
        assert len(candidates) > 0
    
    def test_dry_run_collection(self, garbage_collector, sample_models):
        """Test dry run garbage collection."""
        # Set aggressive removal policy
        garbage_collector.config.max_model_age = 5 * 24 * 3600  # 5 days
        
        result = garbage_collector.collect(dry_run=True)
        
        assert result.dry_run is True
        assert result.trigger == GCTrigger.MANUAL
        assert len(result.models_removed) >= 1  # Should identify old model for removal
        assert result.bytes_reclaimed > 0
        
        # Verify models still exist on disk
        models = garbage_collector._discover_models()
        assert len(models) == 3  # No models actually removed
    
    def test_actual_collection(self, garbage_collector, sample_models):
        """Test actual garbage collection (not dry run)."""
        # Set aggressive removal policy
        garbage_collector.config.max_model_age = 5 * 24 * 3600  # 5 days
        
        # Get initial model count
        initial_models = garbage_collector._discover_models()
        initial_count = len(initial_models)
        
        result = garbage_collector.collect(dry_run=False)
        
        assert result.dry_run is False
        assert len(result.models_removed) >= 1
        assert result.bytes_reclaimed > 0
        
        # Verify models were actually removed
        remaining_models = garbage_collector._discover_models()
        assert len(remaining_models) < initial_count
    
    def test_estimate_reclaimable_space(self, garbage_collector, sample_models):
        """Test estimation of reclaimable space."""
        # Set policy that would remove old models
        garbage_collector.config.max_model_age = 5 * 24 * 3600  # 5 days
        
        reclaimable = garbage_collector.estimate_reclaimable_space()
        
        assert reclaimable > 0
        # Should be at least the size of the old model
        assert reclaimable >= 2000  # Size of old model
    
    @patch('shutil.disk_usage')
    def test_should_trigger_gc_low_disk_space(self, mock_disk_usage, garbage_collector):
        """Test automatic GC trigger for low disk space."""
        # Mock low disk space (95% used)
        from collections import namedtuple
        DiskUsageTuple = namedtuple('usage', ['total', 'used', 'free'])
        mock_disk_usage.return_value = DiskUsageTuple(
            total=1000000000,  # total
            used=950000000,    # used
            free=50000000      # free (5%)
        )
        
        should_trigger, trigger_reason = garbage_collector.should_trigger_gc()
        
        assert should_trigger is True
        assert trigger_reason == GCTrigger.LOW_DISK_SPACE
    
    def test_should_trigger_gc_quota_exceeded(self, garbage_collector, sample_models):
        """Test automatic GC trigger for quota exceeded."""
        # Set very small quota
        garbage_collector.config.max_total_size = 1000  # Smaller than actual usage
        
        should_trigger, trigger_reason = garbage_collector.should_trigger_gc()
        
        assert should_trigger is True
        assert trigger_reason == GCTrigger.QUOTA_EXCEEDED
    
    def test_should_not_trigger_gc_when_disabled(self, garbage_collector, sample_models):
        """Test that GC doesn't trigger when disabled."""
        garbage_collector.config.enable_auto_gc = False
        garbage_collector.config.max_total_size = 1000  # Would normally trigger
        
        should_trigger, trigger_reason = garbage_collector.should_trigger_gc()
        
        assert should_trigger is False
        assert trigger_reason is None
    
    def test_pin_file_persistence(self, garbage_collector, temp_models_dir):
        """Test that pinned models persist across instances."""
        model_id = "test-model"
        variant = "fp16"
        
        # Pin a model
        garbage_collector.pin_model(model_id, variant)
        
        # Create new instance
        new_gc = GarbageCollector(
            garbage_collector.registry,
            garbage_collector.resolver,
            garbage_collector.config
        )
        
        # Should still be pinned
        assert new_gc.is_pinned(model_id, variant)
    
    def test_calculate_directory_size(self, garbage_collector, temp_models_dir):
        """Test directory size calculation."""
        # Create test directory with files
        test_dir = temp_models_dir / "test_dir"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").write_bytes(b"x" * 1000)
        (test_dir / "file2.txt").write_bytes(b"y" * 2000)
        
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_bytes(b"z" * 500)
        
        size = garbage_collector._calculate_directory_size(test_dir)
        assert size == 3500  # 1000 + 2000 + 500
    
    def test_get_last_access_time_from_verification_file(self, garbage_collector, temp_models_dir):
        """Test getting last access time from verification file."""
        model_dir = temp_models_dir / "test_model"
        model_dir.mkdir()
        
        # Create verification file with specific time
        verification_time = time.time() - 3600  # 1 hour ago
        verification_file = model_dir / ".verified.json"
        verification_data = {
            "model_id": "test_model",
            "verified_at": verification_time,
            "files": []
        }
        with open(verification_file, 'w') as f:
            json.dump(verification_data, f)
        
        # Set file modification time
        os.utime(verification_file, (verification_time, verification_time))
        
        access_time = garbage_collector._get_last_access_time(model_dir)
        assert abs(access_time - verification_time) < 1  # Within 1 second
    
    def test_get_last_access_time_fallback(self, garbage_collector, temp_models_dir):
        """Test fallback to directory modification time."""
        model_dir = temp_models_dir / "test_model"
        model_dir.mkdir()
        
        # Set directory modification time
        dir_time = time.time() - 7200  # 2 hours ago
        os.utime(model_dir, (dir_time, dir_time))
        
        access_time = garbage_collector._get_last_access_time(model_dir)
        assert abs(access_time - dir_time) < 1  # Within 1 second
    
    def test_error_handling_in_collection(self, garbage_collector, sample_models):
        """Test error handling during garbage collection."""
        # Mock removal to raise an exception
        with patch.object(garbage_collector, '_remove_model', side_effect=Exception("Test error")):
            garbage_collector.config.max_model_age = 5 * 24 * 3600  # Force removal
            
            result = garbage_collector.collect(dry_run=False)
            
            # Should have errors but not crash
            assert len(result.errors) > 0
            assert "Test error" in result.errors[0]
    
    def test_model_key_generation(self, garbage_collector):
        """Test model key generation for different scenarios."""
        # With variant
        key1 = garbage_collector._get_model_key("test-model", "fp16")
        assert key1 == "test-model@fp16"
        
        # Without variant
        key2 = garbage_collector._get_model_key("test-model", None)
        assert key2 == "test-model"
        
        # Empty variant should be treated as None (no @ suffix)
        key3 = garbage_collector._get_model_key("test-model", "")
        assert key3 == "test-model"


class TestGCConfig:
    """Test cases for GCConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GCConfig()
        
        assert config.max_total_size is None
        assert config.max_model_age is None
        assert config.low_disk_threshold == 0.1
        assert config.safety_margin_bytes == 1024 * 1024 * 1024
        assert config.pin_recent_models is True
        assert config.recent_access_threshold == 24 * 3600
        assert config.enable_auto_gc is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GCConfig(
            max_total_size=500 * 1024 * 1024,
            max_model_age=3 * 24 * 3600,
            low_disk_threshold=0.05,
            enable_auto_gc=False
        )
        
        assert config.max_total_size == 500 * 1024 * 1024
        assert config.max_model_age == 3 * 24 * 3600
        assert config.low_disk_threshold == 0.05
        assert config.enable_auto_gc is False


class TestModelInfo:
    """Test cases for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation and attributes."""
        model_info = ModelInfo(
            model_id="test-model",
            variant="fp16",
            path=Path("/test/path"),
            size_bytes=1000000,
            last_accessed=time.time(),
            is_pinned=True
        )
        
        assert model_info.model_id == "test-model"
        assert model_info.variant == "fp16"
        assert model_info.path == Path("/test/path")
        assert model_info.size_bytes == 1000000
        assert model_info.is_pinned is True
    
    def test_model_info_defaults(self):
        """Test ModelInfo default values."""
        model_info = ModelInfo(
            model_id="test-model",
            variant=None,
            path=Path("/test/path"),
            size_bytes=1000000,
            last_accessed=time.time()
        )
        
        assert model_info.is_pinned is False
        assert model_info.verification_time is None


class TestDiskUsage:
    """Test cases for DiskUsage dataclass."""
    
    def test_disk_usage_creation(self):
        """Test DiskUsage creation and calculations."""
        usage = DiskUsage(
            total_bytes=1000000000,
            used_bytes=800000000,
            free_bytes=200000000,
            models_bytes=500000000,
            usage_percentage=80.0
        )
        
        assert usage.total_bytes == 1000000000
        assert usage.used_bytes == 800000000
        assert usage.free_bytes == 200000000
        assert usage.models_bytes == 500000000
        assert usage.usage_percentage == 80.0


class TestGCResult:
    """Test cases for GCResult dataclass."""
    
    def test_gc_result_creation(self):
        """Test GCResult creation and default values."""
        result = GCResult(
            trigger=GCTrigger.MANUAL,
            dry_run=True
        )
        
        assert result.trigger == GCTrigger.MANUAL
        assert result.dry_run is True
        assert result.models_removed == []
        assert result.models_preserved == []
        assert result.bytes_reclaimed == 0
        assert result.bytes_preserved == 0
        assert result.errors == []
        assert result.duration_seconds == 0.0
    
    def test_gc_result_with_data(self):
        """Test GCResult with actual data."""
        result = GCResult(
            trigger=GCTrigger.QUOTA_EXCEEDED,
            dry_run=False,
            models_removed=["model1", "model2"],
            bytes_reclaimed=1000000,
            errors=["Error 1"],
            duration_seconds=5.5
        )
        
        assert result.trigger == GCTrigger.QUOTA_EXCEEDED
        assert result.dry_run is False
        assert len(result.models_removed) == 2
        assert result.bytes_reclaimed == 1000000
        assert len(result.errors) == 1
        assert result.duration_seconds == 5.5