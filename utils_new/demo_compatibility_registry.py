"""
Demo script for Compatibility Registry System

This script demonstrates the key features of the compatibility registry:
- Model registration and lookup
- Pipeline compatibility checking
- Batch updates and validation
- Registry export/import functionality
"""

import json
import logging
from pathlib import Path

from compatibility_registry import (
    CompatibilityRegistry,
    PipelineRequirements,
    get_compatibility_registry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_registry_operations():
    """Demonstrate basic registry operations"""
    print("\n" + "="*60)
    print("DEMO: Basic Registry Operations")
    print("="*60)
    
    # Create registry with demo file
    registry = CompatibilityRegistry("demo_compatibility_registry.json")
    
    print(f"Registry initialized with {len(registry.registry)} default models")
    
    # List registered models
    models = registry.list_registered_models()
    print(f"\nRegistered models:")
    for model in models[:3]:  # Show first 3
        print(f"  - {model}")
    if len(models) > 3:
        print(f"  ... and {len(models) - 3} more")
    
    # Get requirements for a specific model
    model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    requirements = registry.get_pipeline_requirements(model_name)
    
    if requirements:
        print(f"\nRequirements for {model_name}:")
        print(f"  Pipeline Class: {requirements.pipeline_class}")
        print(f"  Min Diffusers Version: {requirements.min_diffusers_version}")
        print(f"  VRAM Requirements: {requirements.vram_requirements}")
        print(f"  Supported Optimizations: {requirements.supported_optimizations}")


def demo_model_registration():
    """Demonstrate registering new models"""
    print("\n" + "="*60)
    print("DEMO: Model Registration")
    print("="*60)
    
    registry = CompatibilityRegistry("demo_compatibility_registry.json")
    
    # Register a custom model
    custom_requirements = PipelineRequirements(
        pipeline_class="CustomWanPipeline",
        min_diffusers_version="0.22.0",
        required_dependencies=[
            "torch>=2.1.0",
            "transformers>=4.30.0",
            "accelerate>=0.21.0",
            "custom-wan-package>=1.0.0"
        ],
        pipeline_source="https://huggingface.co/community/custom-wan-model",
        vram_requirements={"min_mb": 6144, "recommended_mb": 10240},
        supported_optimizations=[
            "cpu_offload",
            "mixed_precision",
            "chunked_processing",
            "gradient_checkpointing"
        ],
        trust_remote_code=True
    )
    
    model_name = "community/custom-wan-t2v-v2"
    registry.register_model_compatibility(model_name, custom_requirements)
    
    print(f"Registered custom model: {model_name}")
    
    # Verify registration
    retrieved = registry.get_pipeline_requirements(model_name)
    if retrieved:
        print(f"✓ Model successfully registered")
        print(f"  Pipeline: {retrieved.pipeline_class}")
        print(f"  Dependencies: {len(retrieved.required_dependencies)} packages")


def demo_compatibility_checking():
    """Demonstrate pipeline compatibility checking"""
    print("\n" + "="*60)
    print("DEMO: Pipeline Compatibility Checking")
    print("="*60)
    
    registry = CompatibilityRegistry("demo_compatibility_registry.json")
    
    test_cases = [
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "WanPipeline"),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "StableDiffusionPipeline"),
        ("nonexistent-model", "WanPipeline"),
        ("Wan-AI/Wan2.2-T2I-A14B-Diffusers", "WanPipeline")
    ]
    
    for model_name, pipeline_class in test_cases:
        print(f"\nChecking compatibility: {model_name} + {pipeline_class}")
        
        compat_check = registry.validate_model_pipeline_compatibility(
            model_name, pipeline_class
        )
        
        status = "✓ COMPATIBLE" if compat_check.is_compatible else "✗ INCOMPATIBLE"
        print(f"  Status: {status} (Score: {compat_check.compatibility_score:.2f})")
        
        if compat_check.issues:
            print(f"  Issues:")
            for issue in compat_check.issues:
                print(f"    - {issue}")
        
        if compat_check.warnings:
            print(f"  Warnings:")
            for warning in compat_check.warnings:
                print(f"    - {warning}")
        
        if compat_check.recommendations:
            print(f"  Recommendations:")
            for rec in compat_check.recommendations[:2]:  # Show first 2
                print(f"    - {rec}")


def demo_batch_operations():
    """Demonstrate batch registry operations"""
    print("\n" + "="*60)
    print("DEMO: Batch Operations")
    print("="*60)
    
    registry = CompatibilityRegistry("demo_compatibility_registry.json")
    
    # Batch update with multiple models
    batch_updates = {
        "research/experimental-wan-v1": PipelineRequirements(
            pipeline_class="ExperimentalWanPipeline",
            min_diffusers_version="0.23.0",
            required_dependencies=["torch>=2.1.0", "experimental-lib>=0.1.0"],
            pipeline_source="https://github.com/research/experimental-wan",
            vram_requirements={"min_mb": 4096, "recommended_mb": 8192},
            supported_optimizations=["mixed_precision"]
        ),
        "research/lightweight-wan": PipelineRequirements(
            pipeline_class="LightweightWanPipeline",
            min_diffusers_version="0.21.0",
            required_dependencies=["torch>=2.0.0"],
            pipeline_source="https://github.com/research/lightweight-wan",
            vram_requirements={"min_mb": 2048, "recommended_mb": 4096},
            supported_optimizations=["cpu_offload", "mixed_precision", "chunked_processing"]
        )
    }
    
    initial_count = len(registry.registry)
    registry.update_registry(batch_updates)
    final_count = len(registry.registry)
    
    print(f"Batch update completed:")
    print(f"  Models before: {initial_count}")
    print(f"  Models after: {final_count}")
    print(f"  Added: {final_count - initial_count} models")
    
    # Show models by pipeline type
    wan_models = registry.get_models_by_pipeline("WanPipeline")
    experimental_models = registry.get_models_by_pipeline("ExperimentalWanPipeline")
    
    print(f"\nModels by pipeline type:")
    print(f"  WanPipeline: {len(wan_models)} models")
    print(f"  ExperimentalWanPipeline: {len(experimental_models)} models")


def demo_export_import():
    """Demonstrate registry export/import functionality"""
    print("\n" + "="*60)
    print("DEMO: Export/Import Functionality")
    print("="*60)
    
    # Create source registry
    source_registry = CompatibilityRegistry("demo_compatibility_registry.json")
    
    # Export registry
    export_path = "demo_registry_export.json"
    source_registry.export_registry(export_path)
    print(f"Registry exported to: {export_path}")
    
    # Show export file structure
    with open(export_path, 'r') as f:
        export_data = json.load(f)
    
    print(f"Export contains:")
    print(f"  Timestamp: {export_data.get('export_timestamp', 'N/A')}")
    print(f"  Version: {export_data.get('registry_version', 'N/A')}")
    print(f"  Models: {len(export_data.get('models', {}))}")
    
    # Create new registry and import
    target_registry = CompatibilityRegistry("demo_target_registry.json")
    initial_count = len(target_registry.registry)
    
    target_registry.import_registry(export_path, merge=True)
    final_count = len(target_registry.registry)
    
    print(f"\nImport completed:")
    print(f"  Target registry models before: {initial_count}")
    print(f"  Target registry models after: {final_count}")
    
    # Cleanup
    Path(export_path).unlink(missing_ok=True)
    Path("demo_target_registry.json").unlink(missing_ok=True)


def demo_registry_validation():
    """Demonstrate registry integrity validation"""
    print("\n" + "="*60)
    print("DEMO: Registry Validation")
    print("="*60)
    
    registry = CompatibilityRegistry("demo_compatibility_registry.json")
    
    # Run integrity validation
    validation_report = registry.validate_registry_integrity()
    
    print(f"Registry Validation Report:")
    print(f"  Total models: {validation_report['total_models']}")
    print(f"  Validation errors: {len(validation_report['validation_errors'])}")
    print(f"  Validation warnings: {len(validation_report['validation_warnings'])}")
    
    # Show pipeline class distribution
    pipeline_classes = validation_report['pipeline_classes']
    print(f"\nPipeline class distribution:")
    for pipeline_class, models in pipeline_classes.items():
        print(f"  {pipeline_class}: {len(models)} models")
    
    # Show any validation issues
    if validation_report['validation_errors']:
        print(f"\nValidation Errors:")
        for error in validation_report['validation_errors']:
            print(f"  - {error}")
    
    if validation_report['validation_warnings']:
        print(f"\nValidation Warnings:")
        for warning in validation_report['validation_warnings'][:3]:  # Show first 3
            print(f"  - {warning}")


def demo_global_registry():
    """Demonstrate global registry access"""
    print("\n" + "="*60)
    print("DEMO: Global Registry Access")
    print("="*60)
    
    # Get global registry instance
    global_registry = get_compatibility_registry()
    
    print(f"Global registry loaded with {len(global_registry.registry)} models")
    
    # Show that it's a singleton
    another_instance = get_compatibility_registry()
    is_same = global_registry is another_instance
    
    print(f"Singleton pattern working: {is_same}")
    
    # Use global registry for compatibility check
    model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    requirements = global_registry.get_pipeline_requirements(model_name)
    
    if requirements:
        print(f"\nGlobal registry lookup successful:")
        print(f"  Model: {model_name}")
        print(f"  Pipeline: {requirements.pipeline_class}")
        print(f"  Min VRAM: {requirements.vram_requirements.get('min_mb', 'N/A')}MB")


def cleanup_demo_files():
    """Clean up demo files"""
    demo_files = [
        "demo_compatibility_registry.json",
        "demo_registry_export.json",
        "demo_target_registry.json"
    ]
    
    for file_path in demo_files:
        Path(file_path).unlink(missing_ok=True)


def main():
    """Run all compatibility registry demos"""
    print("Compatibility Registry System Demo")
    print("This demo shows the key features of the model-pipeline compatibility registry")
    
    try:
        demo_basic_registry_operations()
        demo_model_registration()
        demo_compatibility_checking()
        demo_batch_operations()
        demo_export_import()
        demo_registry_validation()
        demo_global_registry()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Model registration and lookup")
        print("✓ Pipeline compatibility validation")
        print("✓ Batch registry operations")
        print("✓ Export/import functionality")
        print("✓ Registry integrity validation")
        print("✓ Global registry access")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        cleanup_demo_files()


if __name__ == "__main__":
    main()
