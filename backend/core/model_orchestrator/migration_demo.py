#!/usr/bin/env python3
"""
Migration Demo - Demonstration script for migration and compatibility tools.

This script demonstrates the complete migration workflow from legacy
configuration to the new model orchestrator system.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from .migration_manager import (
    ConfigurationMigrator,
    ManifestValidator,
    RollbackManager,
    LegacyPathAdapter
)
from .validation_tools import ComprehensiveValidator
from .feature_flags import FeatureFlagManager, OrchestratorFeatureFlags


def create_demo_legacy_config(temp_dir: str) -> str:
    """Create a demo legacy configuration for testing."""
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
    
    config_path = Path(temp_dir) / "legacy_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✓ Created demo legacy config: {config_path}")
    return str(config_path)


def create_demo_legacy_models(temp_dir: str) -> str:
    """Create demo legacy model directories."""
    models_dir = Path(temp_dir) / "legacy_models"
    
    # Create model directories with sample files
    for model_name in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        model_dir = models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample model files
        (model_dir / "config.json").write_text('{"model_type": "' + model_name + '"}')
        (model_dir / "model.safetensors").write_text("fake model weights")
        (model_dir / "tokenizer.json").write_text('{"vocab_size": 32000}')
        
        print(f"✓ Created demo model directory: {model_dir}")
    
    return str(models_dir)


def demonstrate_feature_flags():
    """Demonstrate feature flag functionality."""
    print("\n" + "="*60)
    print("FEATURE FLAGS DEMONSTRATION")
    print("="*60)
    
    # Create feature flags from environment
    flags = OrchestratorFeatureFlags.from_env()
    print(f"✓ Loaded feature flags from environment")
    print(f"  - Orchestrator enabled: {flags.enable_orchestrator}")
    print(f"  - Legacy fallback: {flags.enable_legacy_fallback}")
    print(f"  - Manifest validation: {flags.enable_manifest_validation}")
    
    # Demonstrate rollout stage detection
    print(f"  - Current rollout stage: {flags.get_rollout_stage()}")
    
    # Validate configuration
    issues = flags.validate_configuration()
    if issues:
        print(f"  - Configuration issues found: {len(issues)}")
        for issue_key, issue_msg in issues.items():
            print(f"    * {issue_key}: {issue_msg}")
    else:
        print(f"  - Configuration is valid")
    
    # Demonstrate feature flag manager
    manager = FeatureFlagManager()
    status = manager.get_status_report()
    print(f"  - Total features: {status['total_features']}")
    print(f"  - Features enabled: {status['total_features_enabled']}")


def demonstrate_migration_workflow():
    """Demonstrate the complete migration workflow."""
    print("\n" + "="*60)
    print("MIGRATION WORKFLOW DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Create demo legacy configuration
        print("\n1. Creating demo legacy configuration...")
        legacy_config_path = create_demo_legacy_config(temp_dir)
        legacy_models_dir = create_demo_legacy_models(temp_dir)
        
        # Step 2: Create rollback point
        print("\n2. Creating rollback point...")
        rollback_manager = RollbackManager()
        rollback_dir = Path(temp_dir) / "rollbacks"
        
        rollback_id = rollback_manager.create_rollback_point(
            config_paths=[legacy_config_path],
            rollback_dir=str(rollback_dir)
        )
        print(f"✓ Created rollback point: {rollback_id}")
        
        # Step 3: Migrate configuration
        print("\n3. Migrating configuration...")
        migrator = ConfigurationMigrator()
        manifest_path = Path(temp_dir) / "models.toml"
        
        # Mock the write_manifest method for demo
        original_write = migrator.write_manifest
        def mock_write_manifest(manifest_data: Dict[str, Any], output_path: str) -> None:
            print(f"✓ Would write manifest to: {output_path}")
            print(f"  - Schema version: {manifest_data.get('schema_version')}")
            print(f"  - Models defined: {len(manifest_data.get('models', {}))}")
            
            # Actually write a simple manifest for validation demo
            simple_manifest = '''
schema_version = 1

[models."t2v-A14B@2.2.0"]
description = "WAN2.2 Text-to-Video A14B Model"
version = "2.2.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"

[[models."t2v-A14B@2.2.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."t2v-A14B@2.2.0".sources]
priority = ["local://wan22/t2v-A14B@2.2.0"]
'''
            Path(output_path).write_text(simple_manifest)
        
        migrator.write_manifest = mock_write_manifest
        
        result = migrator.migrate_configuration(
            legacy_config_path=legacy_config_path,
            output_manifest_path=str(manifest_path),
            legacy_models_dir=legacy_models_dir,
            scan_files=True
        )
        
        if result.success:
            print(f"✓ Migration completed successfully")
            if result.warnings:
                for warning in result.warnings:
                    print(f"  ⚠ {warning}")
        else:
            print(f"✗ Migration failed")
            for error in result.errors:
                print(f"  ✗ {error}")
        
        # Step 4: Validate migrated manifest
        print("\n4. Validating migrated manifest...")
        validator = ComprehensiveValidator()
        
        if Path(manifest_path).exists():
            validation_report = validator.validate_manifest(str(manifest_path))
            
            print(f"✓ Validation completed")
            print(f"  - Valid: {validation_report.valid}")
            print(f"  - Total issues: {validation_report.summary['total']}")
            print(f"  - Errors: {validation_report.summary['errors']}")
            print(f"  - Warnings: {validation_report.summary['warnings']}")
            
            if validation_report.issues:
                print("  Issues found:")
                for issue in validation_report.issues[:3]:  # Show first 3 issues
                    print(f"    * [{issue.severity.upper()}] {issue.message}")
                if len(validation_report.issues) > 3:
                    print(f"    ... and {len(validation_report.issues) - 3} more")
        
        # Step 5: Demonstrate path migration
        print("\n5. Demonstrating path migration...")
        orchestrator_dir = Path(temp_dir) / "orchestrator_models"
        adapter = LegacyPathAdapter(
            legacy_models_dir=legacy_models_dir,
            orchestrator_models_root=str(orchestrator_dir)
        )
        
        for model_name in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
            if adapter.path_exists_in_legacy(model_name):
                new_path = adapter.map_legacy_path(model_name)
                print(f"  - {model_name}: {adapter.get_legacy_path(model_name)} → {new_path}")
                
                # Demonstrate dry run
                success = adapter.migrate_model_files(model_name, dry_run=True)
                print(f"    Dry run: {'✓' if success else '✗'}")
        
        # Step 6: List rollback points
        print("\n6. Listing rollback points...")
        rollback_points = rollback_manager.list_rollback_points(str(rollback_dir))
        
        for point in rollback_points:
            import datetime
            timestamp = datetime.datetime.fromtimestamp(point['timestamp'])
            print(f"  - {point['rollback_id']}: {timestamp} ({len(point['backed_up_files'])} files)")
        
        print(f"\n✓ Migration workflow demonstration completed!")


def demonstrate_validation_tools():
    """Demonstrate validation tools functionality."""
    print("\n" + "="*60)
    print("VALIDATION TOOLS DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a manifest with various issues for demonstration
        problematic_manifest = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test Model with Issues"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

# Security issue: executable file
[[models."test-model@1.0.0".files]]
path = "malware.exe"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Compatibility issue: invalid characters
[[models."test-model@1.0.0".files]]
path = "invalid<>name.txt"
size = 512
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

# Performance issue: very large file
[[models."test-model@1.0.0".files]]
path = "huge_model.safetensors"
size = 107374182400
sha256 = "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78"

[models."test-model@1.0.0".vram_estimation]
base_vram_gb = 32.0

[models."test-model@1.0.0".sources]
priority = [
    "http://insecure-site.com/model",
    "local://test-model@1.0.0"
]
'''
        
        manifest_path = Path(temp_dir) / "problematic_manifest.toml"
        manifest_path.write_text(problematic_manifest)
        
        print(f"✓ Created test manifest with intentional issues")
        
        # Run comprehensive validation
        validator = ComprehensiveValidator()
        report = validator.validate_manifest(str(manifest_path))
        
        print(f"\nValidation Results:")
        print(f"  - Valid: {report.valid}")
        print(f"  - Total issues: {report.summary['total']}")
        print(f"  - Errors: {report.summary['errors']}")
        print(f"  - Warnings: {report.summary['warnings']}")
        print(f"  - Info: {report.summary['info']}")
        
        # Show issues by category
        categories = set(issue.category for issue in report.issues)
        for category in sorted(categories):
            category_issues = report.get_issues_by_category(category)
            print(f"\n  {category.title()} Issues ({len(category_issues)}):")
            for issue in category_issues[:2]:  # Show first 2 per category
                print(f"    * [{issue.severity.upper()}] {issue.message}")
            if len(category_issues) > 2:
                print(f"    ... and {len(category_issues) - 2} more")


def main():
    """Main demonstration function."""
    print("WAN Model Orchestrator - Migration and Compatibility Tools Demo")
    print("=" * 70)
    
    try:
        # Demonstrate feature flags
        demonstrate_feature_flags()
        
        # Demonstrate migration workflow
        demonstrate_migration_workflow()
        
        # Demonstrate validation tools
        demonstrate_validation_tools()
        
        print("\n" + "="*70)
        print("✓ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("  1. Set environment variables to enable features:")
        print("     export WAN_ENABLE_ORCHESTRATOR=true")
        print("     export WAN_ENABLE_MANIFEST_VALIDATION=true")
        print("  2. Run migration on your actual configuration:")
        print("     python -m backend.core.model_orchestrator.migration_cli migrate-config config.json config/models.toml")
        print("  3. Validate the migrated manifest:")
        print("     python -m backend.core.model_orchestrator.migration_cli validate-manifest config/models.toml")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()