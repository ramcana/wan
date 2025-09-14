"""
Tests for migration and compatibility tools.

This module contains comprehensive tests for configuration migration,
validation, rollback functionality, and backward compatibility.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from .migration_manager import (
    ConfigurationMigrator,
    ManifestValidator, 
    RollbackManager,
    LegacyPathAdapter,
    FeatureFlags,
    LegacyConfig,
    MigrationResult
)
from .exceptions import MigrationError, ValidationError


class TestLegacyConfig:
    """Test LegacyConfig data class."""
    
    def test_legacy_config_creation(self):
        """Test creating LegacyConfig from data."""
        config = LegacyConfig(
            system={"default_quantization": "bf16"},
            directories={"models_directory": "models"},
            models={"t2v_model": "t2v-A14B"},
            optimization={"max_vram_usage_gb": 12}
        )
        
        assert config.system["default_quantization"] == "bf16"
        assert config.directories["models_directory"] == "models"
        assert config.models["t2v_model"] == "t2v-A14B"
        assert config.optimization["max_vram_usage_gb"] == 12


class TestFeatureFlags:
    """Test FeatureFlags functionality."""
    
    def test_default_feature_flags(self):
        """Test default feature flag values."""
        flags = FeatureFlags()
        
        assert flags.enable_orchestrator is False
        assert flags.enable_manifest_validation is True
        assert flags.enable_legacy_fallback is True
        assert flags.enable_path_migration is False
        assert flags.enable_automatic_download is False
        assert flags.strict_validation is False
    
    def test_feature_flags_from_env(self):
        """Test loading feature flags from environment variables."""
        env_vars = {
            'WAN_ENABLE_ORCHESTRATOR': 'true',
            'WAN_ENABLE_MANIFEST_VALIDATION': 'false',
            'WAN_ENABLE_LEGACY_FALLBACK': 'false',
            'WAN_ENABLE_PATH_MIGRATION': 'true',
            'WAN_ENABLE_AUTO_DOWNLOAD': 'true',
            'WAN_STRICT_VALIDATION': 'true',
        }
        
        with patch.dict(os.environ, env_vars):
            flags = FeatureFlags.from_env()
            
            assert flags.enable_orchestrator is True
            assert flags.enable_manifest_validation is False
            assert flags.enable_legacy_fallback is False
            assert flags.enable_path_migration is True
            assert flags.enable_automatic_download is True
            assert flags.strict_validation is True


class TestLegacyPathAdapter:
    """Test LegacyPathAdapter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.legacy_dir = Path(self.temp_dir) / "legacy_models"
        self.orchestrator_dir = Path(self.temp_dir) / "orchestrator_models"
        
        self.legacy_dir.mkdir(parents=True)
        self.orchestrator_dir.mkdir(parents=True)
        
        self.adapter = LegacyPathAdapter(
            str(self.legacy_dir),
            str(self.orchestrator_dir)
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_map_legacy_path(self):
        """Test mapping legacy model names to new paths."""
        # Test known mappings
        assert self.adapter.map_legacy_path("t2v-A14B") == str(
            self.orchestrator_dir / "wan22" / "t2v-A14B@2.2.0"
        )
        assert self.adapter.map_legacy_path("i2v-A14B") == str(
            self.orchestrator_dir / "wan22" / "i2v-A14B@2.2.0"
        )
        assert self.adapter.map_legacy_path("ti2v-5B") == str(
            self.orchestrator_dir / "wan22" / "ti2v-5b@2.2.0"
        )
        
        # Test unknown mapping
        assert self.adapter.map_legacy_path("unknown-model") is None
    
    def test_get_legacy_path(self):
        """Test getting legacy path for a model."""
        path = self.adapter.get_legacy_path("t2v-A14B")
        assert path == str(self.legacy_dir / "t2v-A14B")
    
    def test_path_exists_in_legacy(self):
        """Test checking if model exists in legacy location."""
        # Create a legacy model directory
        legacy_model_dir = self.legacy_dir / "t2v-A14B"
        legacy_model_dir.mkdir()
        
        assert self.adapter.path_exists_in_legacy("t2v-A14B") is True
        assert self.adapter.path_exists_in_legacy("nonexistent-model") is False
    
    def test_migrate_model_files_dry_run(self):
        """Test dry run migration."""
        # Create a legacy model directory with files
        legacy_model_dir = self.legacy_dir / "t2v-A14B"
        legacy_model_dir.mkdir()
        (legacy_model_dir / "model.safetensors").write_text("fake model data")
        
        # Test dry run
        result = self.adapter.migrate_model_files("t2v-A14B", dry_run=True)
        assert result is True
        
        # Verify no actual migration occurred
        new_path = Path(self.adapter.map_legacy_path("t2v-A14B"))
        assert not new_path.exists()
    
    def test_migrate_model_files_actual(self):
        """Test actual migration of model files."""
        # Create a legacy model directory with files
        legacy_model_dir = self.legacy_dir / "t2v-A14B"
        legacy_model_dir.mkdir()
        test_file = legacy_model_dir / "model.safetensors"
        test_content = "fake model data"
        test_file.write_text(test_content)
        
        # Test actual migration
        result = self.adapter.migrate_model_files("t2v-A14B", dry_run=False)
        assert result is True
        
        # Verify migration occurred
        new_path = Path(self.adapter.map_legacy_path("t2v-A14B"))
        assert new_path.exists()
        assert (new_path / "model.safetensors").read_text() == test_content
    
    def test_migrate_nonexistent_model(self):
        """Test migration of nonexistent model."""
        result = self.adapter.migrate_model_files("nonexistent-model", dry_run=False)
        assert result is False


class TestConfigurationMigrator:
    """Test ConfigurationMigrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.migrator = ConfigurationMigrator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_legacy_config(self) -> str:
        """Create a test legacy config file."""
        config_data = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True
            },
            "directories": {
                "output_directory": "outputs",
                "models_directory": "models",
                "loras_directory": "loras"
            },
            "models": {
                "t2v_model": "t2v-A14B",
                "i2v_model": "i2v-A14B",
                "ti2v_model": "ti2v-5B"
            },
            "optimization": {
                "default_quantization": "fp16",
                "enable_offload": True,
                "max_vram_usage_gb": 12
            }
        }
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return str(config_path)
    
    def test_load_legacy_config(self):
        """Test loading legacy configuration."""
        config_path = self.create_legacy_config()
        
        legacy_config = self.migrator.load_legacy_config(config_path)
        
        assert legacy_config.system["default_quantization"] == "bf16"
        assert legacy_config.models["t2v_model"] == "t2v-A14B"
        assert legacy_config.optimization["max_vram_usage_gb"] == 12
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent config file."""
        with pytest.raises(MigrationError):
            self.migrator.load_legacy_config("nonexistent.json")
    
    def test_generate_manifest_from_legacy(self):
        """Test generating manifest from legacy config."""
        config_path = self.create_legacy_config()
        legacy_config = self.migrator.load_legacy_config(config_path)
        
        manifest = self.migrator.generate_manifest_from_legacy(legacy_config)
        
        # Check basic structure
        assert manifest["schema_version"] == 1
        assert "models" in manifest
        
        # Check model entries
        models = manifest["models"]
        assert "t2v-A14B@2.2.0" in models
        assert "i2v-A14B@2.2.0" in models
        assert "ti2v-5b@2.2.0" in models
        
        # Check model specifications
        t2v_model = models["t2v-A14B@2.2.0"]
        assert t2v_model["description"] == "WAN2.2 Text-to-Video A14B Model"
        assert t2v_model["default_variant"] == "fp16"  # From optimization.default_quantization
        assert t2v_model["required_components"] == ["text_encoder", "unet", "vae"]
    
    def test_determine_component_type(self):
        """Test component type determination from file paths."""
        # Test various file path patterns
        assert self.migrator._determine_component_type("Wan2.1_VAE.pth") == "vae"
        assert self.migrator._determine_component_type("models_t5_umt5-xxl-enc-bf16.pth") == "text_encoder"
        assert self.migrator._determine_component_type("image_encoder/pytorch_model.bin") == "image_encoder"
        assert self.migrator._determine_component_type("diffusion_pytorch_model.safetensors") == "unet"
        assert self.migrator._determine_component_type("configuration.json") == "config"
        assert self.migrator._determine_component_type("unknown_file.bin") is None
    
    def test_scan_legacy_model_files(self):
        """Test scanning legacy model files."""
        # Create a mock model directory structure
        models_dir = Path(self.temp_dir) / "models"
        model_dir = models_dir / "t2v-A14B"
        model_dir.mkdir(parents=True)
        
        # Create some test files
        (model_dir / "config.json").write_text('{"test": "config"}')
        (model_dir / "model.safetensors").write_text("fake model data")
        
        files = self.migrator.scan_legacy_model_files(str(models_dir), "t2v-A14B")
        
        assert len(files) == 2
        
        # Check file specifications
        config_file = next(f for f in files if f["path"] == "config.json")
        assert config_file["component"] == "config"
        assert config_file["size"] > 0
        assert "sha256" in config_file
    
    @patch('backend.core.model_orchestrator.migration_manager.tomli_w')
    def test_write_manifest(self, mock_tomli_w):
        """Test writing manifest to TOML file."""
        manifest_data = {"schema_version": 1, "models": {}}
        output_path = Path(self.temp_dir) / "models.toml"
        
        self.migrator.write_manifest(manifest_data, str(output_path))
        
        # Verify tomli_w.dump was called
        mock_tomli_w.dump.assert_called_once()
    
    def test_write_manifest_without_tomli_w(self):
        """Test writing manifest when tomli_w is not available."""
        with patch('backend.core.model_orchestrator.migration_manager.tomli_w', None):
            with pytest.raises(MigrationError):
                self.migrator.write_manifest({}, "test.toml")
    
    def test_migrate_configuration_success(self):
        """Test successful configuration migration."""
        config_path = self.create_legacy_config()
        output_path = Path(self.temp_dir) / "models.toml"
        
        with patch.object(self.migrator, 'write_manifest') as mock_write:
            result = self.migrator.migrate_configuration(
                legacy_config_path=config_path,
                output_manifest_path=str(output_path),
                backup=False
            )
        
        assert result.success is True
        assert result.manifest_path == str(output_path)
        assert len(result.errors) == 0
        mock_write.assert_called_once()
    
    def test_migrate_configuration_with_backup(self):
        """Test migration with backup creation."""
        config_path = self.create_legacy_config()
        output_path = Path(self.temp_dir) / "models.toml"
        
        # Create existing manifest to trigger backup
        output_path.write_text("existing manifest")
        
        with patch.object(self.migrator, 'write_manifest'):
            result = self.migrator.migrate_configuration(
                legacy_config_path=config_path,
                output_manifest_path=str(output_path),
                backup=True
            )
        
        assert result.success is True
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()


class TestManifestValidator:
    """Test ManifestValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ManifestValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_valid_manifest(self) -> str:
        """Create a valid test manifest file."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test Model"
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
priority = ["local://test-model@1.0.0"]
'''
        
        manifest_path = Path(self.temp_dir) / "models.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def create_invalid_manifest(self) -> str:
        """Create an invalid test manifest file."""
        manifest_content = '''
# Missing schema_version

[models."invalid-model"]
# Missing required fields
description = "Invalid Model"
'''
        
        manifest_path = Path(self.temp_dir) / "invalid_models.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def test_validate_valid_manifest(self):
        """Test validation of a valid manifest."""
        manifest_path = self.create_valid_manifest()
        
        errors = self.validator.validate_manifest_file(manifest_path)
        
        # Should have no errors for valid manifest
        assert len(errors) == 0
    
    def test_validate_invalid_manifest(self):
        """Test validation of an invalid manifest."""
        manifest_path = self.create_invalid_manifest()
        
        errors = self.validator.validate_manifest_file(manifest_path)
        
        # Should have errors for invalid manifest
        assert len(errors) > 0
    
    def test_validate_nonexistent_manifest(self):
        """Test validation of nonexistent manifest."""
        errors = self.validator.validate_manifest_file("nonexistent.toml")
        
        assert len(errors) > 0
        assert any("Failed to load manifest" in str(e) for e in errors)
    
    def test_validate_configuration_compatibility(self):
        """Test compatibility validation between manifest and legacy config."""
        manifest_path = self.create_valid_manifest()
        
        # Create legacy config
        legacy_config = {
            "models": {
                "t2v_model": "t2v-A14B",
                "i2v_model": "i2v-A14B"
            }
        }
        legacy_path = Path(self.temp_dir) / "config.json"
        with open(legacy_path, 'w') as f:
            json.dump(legacy_config, f)
        
        errors = self.validator.validate_configuration_compatibility(
            manifest_path, str(legacy_path)
        )
        
        # Should have compatibility errors since manifest doesn't contain legacy models
        assert len(errors) > 0


class TestRollbackManager:
    """Test RollbackManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.rollback_manager = RollbackManager()
        self.rollback_dir = Path(self.temp_dir) / "rollbacks"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_rollback_point(self):
        """Test creating a rollback point."""
        # Create test config files
        config1 = Path(self.temp_dir) / "config1.json"
        config2 = Path(self.temp_dir) / "config2.toml"
        
        config1.write_text('{"test": "config1"}')
        config2.write_text('test = "config2"')
        
        # Create rollback point
        rollback_id = self.rollback_manager.create_rollback_point(
            config_paths=[str(config1), str(config2)],
            rollback_dir=str(self.rollback_dir)
        )
        
        # Verify rollback point was created
        assert rollback_id.startswith("rollback_")
        rollback_path = self.rollback_dir / rollback_id
        assert rollback_path.exists()
        
        # Verify metadata file exists
        metadata_file = rollback_path / "rollback_metadata.json"
        assert metadata_file.exists()
        
        # Verify backed up files exist
        assert (rollback_path / "config1.json").exists()
        assert (rollback_path / "config2.toml").exists()
    
    def test_execute_rollback(self):
        """Test executing a rollback."""
        # Create test config files
        config1 = Path(self.temp_dir) / "config1.json"
        config1.write_text('{"original": "content"}')
        
        # Create rollback point
        rollback_id = self.rollback_manager.create_rollback_point(
            config_paths=[str(config1)],
            rollback_dir=str(self.rollback_dir)
        )
        
        # Modify the original file
        config1.write_text('{"modified": "content"}')
        
        # Execute rollback
        success = self.rollback_manager.execute_rollback(
            rollback_id=rollback_id,
            rollback_dir=str(self.rollback_dir)
        )
        
        # Verify rollback was successful
        assert success is True
        
        # Verify original content was restored
        restored_content = config1.read_text()
        assert '{"original": "content"}' == restored_content
    
    def test_execute_nonexistent_rollback(self):
        """Test executing rollback for nonexistent rollback point."""
        success = self.rollback_manager.execute_rollback(
            rollback_id="nonexistent_rollback",
            rollback_dir=str(self.rollback_dir)
        )
        
        assert success is False
    
    def test_list_rollback_points(self):
        """Test listing rollback points."""
        # Initially no rollback points
        points = self.rollback_manager.list_rollback_points(str(self.rollback_dir))
        assert len(points) == 0
        
        # Create a rollback point
        config1 = Path(self.temp_dir) / "config1.json"
        config1.write_text('{"test": "config"}')
        
        rollback_id = self.rollback_manager.create_rollback_point(
            config_paths=[str(config1)],
            rollback_dir=str(self.rollback_dir)
        )
        
        # List rollback points
        points = self.rollback_manager.list_rollback_points(str(self.rollback_dir))
        assert len(points) == 1
        assert points[0]["rollback_id"] == rollback_id
        assert len(points[0]["backed_up_files"]) == 1


class TestMigrationIntegration:
    """Integration tests for migration tools."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_migration_workflow(self):
        """Test complete migration workflow from legacy to orchestrator."""
        # Create legacy configuration
        legacy_config = {
            "system": {"default_quantization": "bf16"},
            "directories": {"models_directory": "models"},
            "models": {"t2v_model": "t2v-A14B"},
            "optimization": {"max_vram_usage_gb": 12}
        }
        
        legacy_config_path = Path(self.temp_dir) / "config.json"
        with open(legacy_config_path, 'w') as f:
            json.dump(legacy_config, f, indent=2)
        
        # Create legacy model directory
        legacy_models_dir = Path(self.temp_dir) / "models"
        legacy_model_dir = legacy_models_dir / "t2v-A14B"
        legacy_model_dir.mkdir(parents=True)
        (legacy_model_dir / "model.safetensors").write_text("fake model data")
        
        # Create rollback point
        rollback_manager = RollbackManager()
        rollback_dir = Path(self.temp_dir) / "rollbacks"
        rollback_id = rollback_manager.create_rollback_point(
            config_paths=[str(legacy_config_path)],
            rollback_dir=str(rollback_dir)
        )
        
        # Migrate configuration
        migrator = ConfigurationMigrator()
        manifest_path = Path(self.temp_dir) / "models.toml"
        
        with patch.object(migrator, 'write_manifest') as mock_write:
            result = migrator.migrate_configuration(
                legacy_config_path=str(legacy_config_path),
                output_manifest_path=str(manifest_path),
                legacy_models_dir=str(legacy_models_dir),
                scan_files=True
            )
        
        # Verify migration succeeded
        assert result.success is True
        mock_write.assert_called_once()
        
        # Migrate model files
        orchestrator_dir = Path(self.temp_dir) / "orchestrator"
        adapter = LegacyPathAdapter(
            str(legacy_models_dir),
            str(orchestrator_dir)
        )
        
        migration_success = adapter.migrate_model_files("t2v-A14B", dry_run=False)
        assert migration_success is True
        
        # Verify new model location exists
        new_model_path = Path(adapter.map_legacy_path("t2v-A14B"))
        assert new_model_path.exists()
        assert (new_model_path / "model.safetensors").exists()
        
        # Test rollback
        rollback_success = rollback_manager.execute_rollback(
            rollback_id=rollback_id,
            rollback_dir=str(rollback_dir)
        )
        assert rollback_success is True
    
    def test_migration_error_handling(self):
        """Test error handling in migration workflow."""
        migrator = ConfigurationMigrator()
        
        # Test migration with invalid legacy config
        with pytest.raises(MigrationError):
            migrator.load_legacy_config("nonexistent.json")
        
        # Test path adapter with invalid paths
        adapter = LegacyPathAdapter("/nonexistent/legacy", "/nonexistent/orchestrator")
        result = adapter.migrate_model_files("nonexistent-model")
        assert result is False
        
        # Test validator with invalid manifest
        validator = ManifestValidator()
        errors = validator.validate_manifest_file("nonexistent.toml")
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])