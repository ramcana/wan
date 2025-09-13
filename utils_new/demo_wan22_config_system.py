"""
Demo script for WAN22 Configuration System

This script demonstrates the comprehensive configuration management system
including presets, validation, migration, and customization options.
"""

import tempfile
import shutil
from pathlib import Path

from wan22_config_manager import (
    ConfigurationManager,
    WAN22Config,
    OptimizationStrategy,
    PipelineSelectionMode,
    SecurityLevel
)
from wan22_config_presets import (
    ConfigurationPresets,
    apply_preset,
    get_preset_comparison,
    recommend_preset
)
from wan22_config_validation import (
    validate_config,
    get_validation_summary
)
from wan22_config_migration import (
    MigrationManager,
    migrate_configuration
)


def demo_basic_configuration():
    """Demonstrate basic configuration management"""
    print("=" * 60)
    print("DEMO: Basic Configuration Management")
    print("=" * 60)
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager(temp_dir)
        
        # Load default configuration
        print("1. Loading default configuration...")
        config = config_manager.load_config()
        print(f"   Default strategy: {config.optimization.strategy}")
        print(f"   Default security level: {config.security.security_level}")
        print(f"   Config file created at: {config_manager.config_path}")
        
        # Update configuration
        print("\n2. Updating configuration...")
        updates = {
            "optimization": {
                "strategy": "memory",
                "enable_cpu_offload": True,
                "max_chunk_size": 4
            },
            "user_preferences": {
                "default_fps": 30.0,
                "verbose_logging": True
            }
        }
        
        success = config_manager.update_config(updates)
        print(f"   Update successful: {success}")
        
        # Verify updates
        updated_config = config_manager.get_config()
        print(f"   New strategy: {updated_config.optimization.strategy}")
        print(f"   CPU offload enabled: {updated_config.optimization.enable_cpu_offload}")
        print(f"   New FPS: {updated_config.user_preferences.default_fps}")
        
        # Export configuration
        print("\n3. Exporting configuration...")
        export_path = Path(temp_dir) / "exported_config.json"
        success = config_manager.export_config(str(export_path))
        print(f"   Export successful: {success}")
        print(f"   Exported to: {export_path}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_configuration_presets():
    """Demonstrate configuration presets"""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Presets")
    print("=" * 60)
    
    # Show available presets
    print("1. Available presets:")
    preset_names = ConfigurationPresets.get_preset_names()
    for name in preset_names[:6]:  # Show first 6
        description = ConfigurationPresets.get_preset_description(name)
        print(f"   - {name}: {description}")
    print(f"   ... and {len(preset_names) - 6} more")
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    try:
        config_manager = ConfigurationManager(temp_dir)
        
        # Apply high performance preset
        print("\n2. Applying high performance preset...")
        success = apply_preset(config_manager, "high_performance")
        print(f"   Applied successfully: {success}")
        
        config = config_manager.get_config()
        print(f"   Strategy: {config.optimization.strategy}")
        print(f"   CPU offload: {config.optimization.enable_cpu_offload}")
        print(f"   Fallback enabled: {config.pipeline.enable_fallback}")
        
        # Compare presets
        print("\n3. Comparing presets...")
        comparison = get_preset_comparison("high_performance", "memory_optimized")
        print(f"   Comparing: {comparison['preset1']} vs {comparison['preset2']}")
        print(f"   Found {len(comparison['differences'])} differences")
        
        # Show a few key differences
        for key, diff in list(comparison['differences'].items())[:3]:
            print(f"   - {key}: {diff['preset1']} → {diff['preset2']}")
        
        # Get recommendations
        print("\n4. Getting recommendations...")
        low_vram_rec = recommend_preset(4096, "general")
        high_vram_rec = recommend_preset(20480, "general")
        dev_rec = recommend_preset(8192, "development")
        
        print(f"   Low VRAM (4GB): {low_vram_rec}")
        print(f"   High VRAM (20GB): {high_vram_rec}")
        print(f"   Development use: {dev_rec}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_configuration_validation():
    """Demonstrate configuration validation"""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Validation")
    print("=" * 60)
    
    # Validate a good configuration
    print("1. Validating good configuration...")
    good_config = WAN22Config()
    result = validate_config(good_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Score: {result.score:.2f}/1.00")
    print(f"   Messages: {len(result.messages)}")
    
    # Create problematic configuration
    print("\n2. Validating problematic configuration...")
    bad_config = WAN22Config()
    bad_config.optimization.max_chunk_size = -1  # Invalid
    bad_config.optimization.vram_threshold_mb = 100  # Too low
    bad_config.pipeline.max_retry_attempts = 50  # Too high
    bad_config.user_preferences.default_fps = -5.0  # Invalid
    
    result = validate_config(bad_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Score: {result.score:.2f}/1.00")
    print(f"   Errors: {len(result.get_messages_by_severity('error'))}")
    print(f"   Warnings: {len(result.get_messages_by_severity('warning'))}")
    
    # Show validation summary
    print("\n3. Validation summary:")
    summary = get_validation_summary(result)
    print(summary)
    
    # Show specific issues
    print("\n4. Specific validation issues:")
    for msg in result.messages[:3]:  # Show first 3
        print(f"   {msg.severity.value.upper()}: {msg.message}")
        if msg.suggested_value:
            print(f"      Suggested: {msg.suggested_value}")


def demo_configuration_migration():
    """Demonstrate configuration migration"""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Migration")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    try:
        config_manager = ConfigurationManager(temp_dir)
        
        # Create old version configuration
        print("1. Creating old version configuration...")
        old_config_data = {
            "version": "0.2.0",
            "optimization": {"strategy": "performance"},
            "security": {"trust_remote_code": False},
            "output_format": "webm",  # Old setting
            "verbose": True  # Old setting
        }
        
        import json
        with open(config_manager.config_path, 'w') as f:
            json.dump(old_config_data, f)
        
        print(f"   Created config with version: {old_config_data['version']}")
        
        # Check migration info
        print("\n2. Checking migration info...")
        migration_manager = MigrationManager(config_manager)
        info = migration_manager.get_migration_info()
        
        print(f"   Current version: {info['current_version']}")
        print(f"   Target version: {info['target_version']}")
        print(f"   Migration needed: {info['migration_needed']}")
        print(f"   Migration path: {info['migration_path']}")
        
        # Perform migration
        print("\n3. Performing migration...")
        success = migration_manager.migrate_if_needed("1.0.0")
        print(f"   Migration successful: {success}")
        
        # Verify migration results
        print("\n4. Verifying migration results...")
        config = config_manager.get_config()
        print(f"   New version: {config.version}")
        print(f"   Strategy preserved: {config.optimization.strategy}")
        print(f"   Security preserved: {config.security.trust_remote_code}")
        print(f"   Old setting migrated: {config.user_preferences.default_output_format}")
        print(f"   Verbose migrated: {config.user_preferences.verbose_logging}")
        
        # Show backup files
        info = migration_manager.get_migration_info()
        print(f"   Backup files created: {len(info['backup_files'])}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_advanced_customization():
    """Demonstrate advanced customization options"""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Customization")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    try:
        config_manager = ConfigurationManager(temp_dir)
        
        # Start with a preset
        print("1. Starting with development preset...")
        apply_preset(config_manager, "development")
        config = config_manager.get_config()
        print(f"   Base strategy: {config.optimization.strategy}")
        print(f"   Verbose logging: {config.user_preferences.verbose_logging}")
        
        # Add custom optimizations
        print("\n2. Adding custom optimizations...")
        updates = {
            "optimization": {
                "custom_optimizations": {
                    "use_flash_attention": True,
                    "gradient_checkpointing": True,
                    "custom_scheduler": "cosine_annealing"
                }
            },
            "experimental_features": {
                "advanced_diagnostics": True,
                "performance_profiling": True,
                "beta_optimizations": True
            },
            "custom_settings": {
                "project_name": "my_video_project",
                "output_template": "{project}_{timestamp}_{prompt_hash}",
                "custom_model_paths": {
                    "lora_dir": "/path/to/loras",
                    "checkpoint_dir": "/path/to/checkpoints"
                }
            }
        }
        
        success = config_manager.update_config(updates)
        print(f"   Custom settings applied: {success}")
        
        # Validate customized configuration
        print("\n3. Validating customized configuration...")
        config = config_manager.get_config()
        result = validate_config(config)
        print(f"   Still valid: {result.is_valid}")
        print(f"   Validation score: {result.score:.2f}/1.00")
        
        # Show custom settings
        print("\n4. Custom settings summary:")
        print(f"   Custom optimizations: {len(config.optimization.custom_optimizations)}")
        print(f"   Experimental features: {len(config.experimental_features)}")
        print(f"   Custom settings: {len(config.custom_settings)}")
        
        # Export for sharing
        print("\n5. Exporting customized configuration...")
        export_path = Path(temp_dir) / "custom_config.json"
        success = config_manager.export_config(str(export_path), include_sensitive=False)
        print(f"   Exported (safe): {success}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_integration_workflow():
    """Demonstrate complete integration workflow"""
    print("\n" + "=" * 60)
    print("DEMO: Complete Integration Workflow")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Initialize with recommended preset
        print("1. Getting hardware-based recommendation...")
        vram_mb = 12288  # 12GB VRAM
        use_case = "content_creation"
        recommended = recommend_preset(vram_mb, use_case)
        print(f"   Recommended preset: {recommended}")
        
        # 2. Apply preset and customize
        print("\n2. Applying and customizing configuration...")
        config_manager = ConfigurationManager(temp_dir)
        apply_preset(config_manager, recommended)
        
        # Add project-specific customizations
        updates = {
            "user_preferences": {
                "default_fps": 60.0,  # High quality for content creation
                "preferred_video_codec": "h265",  # Better compression
                "max_concurrent_generations": 2  # Utilize hardware
            },
            "optimization": {
                "vram_threshold_mb": vram_mb,
                "enable_vae_tiling": False  # Better quality with sufficient VRAM
            }
        }
        config_manager.update_config(updates)
        
        # 3. Validate final configuration
        print("\n3. Validating final configuration...")
        config = config_manager.get_config()
        result = validate_config(config)
        
        if result.is_valid:
            print("   ✅ Configuration is valid and ready to use!")
            print(f"   Quality score: {result.score:.2f}/1.00")
        else:
            print("   ❌ Configuration has issues:")
            for msg in result.get_messages_by_severity('error')[:2]:
                print(f"      - {msg.message}")
        
        # 4. Show final configuration summary
        print("\n4. Final configuration summary:")
        print(f"   Optimization strategy: {config.optimization.strategy}")
        print(f"   VRAM threshold: {config.optimization.vram_threshold_mb} MB")
        print(f"   Security level: {config.security.security_level}")
        print(f"   Output format: {config.user_preferences.default_output_format}")
        print(f"   FPS: {config.user_preferences.default_fps}")
        print(f"   Concurrent generations: {config.user_preferences.max_concurrent_generations}")
        
        # 5. Save for production use
        print("\n5. Saving production configuration...")
        success = config_manager.save_config(config)
        print(f"   Saved successfully: {success}")
        print(f"   Configuration file: {config_manager.config_path}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all configuration system demos"""
    print("WAN22 Configuration System Demo")
    print("This demo showcases the comprehensive configuration management system")
    print("for WAN22 model compatibility, including presets, validation, and migration.")
    
    try:
        demo_basic_configuration()
        demo_configuration_presets()
        demo_configuration_validation()
        demo_configuration_migration()
        demo_advanced_customization()
        demo_integration_workflow()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("The WAN22 configuration system provides:")
        print("✅ Comprehensive configuration management")
        print("✅ 12+ predefined presets for different use cases")
        print("✅ Advanced validation with detailed feedback")
        print("✅ Automatic migration between versions")
        print("✅ Extensive customization options")
        print("✅ Hardware-based recommendations")
        print("✅ Import/export capabilities")
        print("✅ Full integration with the WAN22 compatibility system")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
