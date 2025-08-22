"""
Demo script for Model Index Schema Validation

This script demonstrates the model_index_schema validation system with various
model configurations including valid and invalid examples.
"""

from model_index_schema import SchemaValidator, ModelType
import json


def demo_schema_validation():
    """Demonstrate schema validation with various model configurations"""
    
    print("=" * 70)
    print("MODEL INDEX SCHEMA VALIDATION DEMO")
    print("=" * 70)
    
    validator = SchemaValidator()
    
    # Test cases
    test_cases = [
        {
            "name": "Valid Wan T2V Model",
            "data": {
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
        },
        {
            "name": "Valid Stable Diffusion Model",
            "data": {
                "_class_name": "StableDiffusionPipeline",
                "_diffusers_version": "0.21.0",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "scheduler": ["diffusers", "PNDMScheduler"]
            }
        },
        {
            "name": "Invalid - Missing Required Fields",
            "data": {
                "_diffusers_version": "0.21.0"
                # Missing _class_name
            }
        },
        {
            "name": "Invalid - Bad Boundary Ratio",
            "data": {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.21.0",
                "transformer": ["diffusers", "Transformer2DModel"],
                "boundary_ratio": 1.5  # Invalid: > 1
            }
        },
        {
            "name": "Wan Model with Missing Components",
            "data": {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.21.0",
                "transformer": ["diffusers", "Transformer2DModel"],
                "boundary_ratio": 0.3
                # Missing vae, text_encoder, etc.
            }
        },
        {
            "name": "Old Diffusers Version",
            "data": {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.19.0",  # Old version
                "transformer": ["diffusers", "Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "scheduler": ["diffusers", "DDIMScheduler"]
            }
        }
    ]
    
    # Run validation tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 50)
        
        result = validator.validate_model_index_dict(test_case['data'])
        
        # Print basic results
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"Status: {status}")
        print(f"Model Type: {result.model_type.value}")
        
        if result.schema:
            print(f"Pipeline Class: {result.schema.class_name}")
            print(f"Is Wan Architecture: {result.schema.is_wan_architecture()}")
        
        # Print errors
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  • {error}")
        
        # Print warnings
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        # Print suggested fixes
        if result.suggested_fixes:
            print("\nSuggested Fixes:")
            for fix in result.suggested_fixes:
                print(f"  • {fix}")
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION REPORT EXAMPLE")
    print("=" * 70)
    
    # Generate a comprehensive report for one of the test cases
    wan_model_data = {
        "_class_name": "WanPipeline",
        "_diffusers_version": "0.21.0",
        "transformer": ["diffusers", "Transformer2DModel"],
        "boundary_ratio": 0.4
        # Missing some components
    }
    
    result = validator.validate_model_index_dict(wan_model_data)
    report = validator.generate_schema_report(result)
    print(report)
    
    print("\n" + "=" * 70)
    print("VALIDATION HISTORY")
    print("=" * 70)
    
    history = validator.get_validation_history()
    print(f"Total validations performed: {len(history)}")
    
    valid_count = sum(1 for r in history if r.is_valid)
    invalid_count = len(history) - valid_count
    
    print(f"Valid models: {valid_count}")
    print(f"Invalid models: {invalid_count}")
    
    # Model type distribution
    model_types = {}
    for result in history:
        model_type = result.model_type.value
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    print("\nModel Type Distribution:")
    for model_type, count in model_types.items():
        print(f"  {model_type}: {count}")


def demo_file_validation():
    """Demonstrate file-based validation"""
    
    print("\n" + "=" * 70)
    print("FILE-BASED VALIDATION DEMO")
    print("=" * 70)
    
    validator = SchemaValidator()
    
    # Create sample model_index.json files
    sample_files = [
        {
            "filename": "valid_wan_model_index.json",
            "data": {
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
        },
        {
            "filename": "invalid_model_index.json",
            "data": {
                "_diffusers_version": "0.21.0"
                # Missing _class_name
            }
        }
    ]
    
    # Create and validate files
    for sample in sample_files:
        filename = sample["filename"]
        
        # Write sample file
        with open(filename, 'w') as f:
            json.dump(sample["data"], f, indent=2)
        
        print(f"\nValidating file: {filename}")
        print("-" * 40)
        
        # Validate file
        result = validator.validate_model_index(filename)
        
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"Status: {status}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  • {error}")
        
        if result.suggested_fixes:
            print("Suggested Fixes:")
            for fix in result.suggested_fixes:
                print(f"  • {fix}")
    
    # Clean up sample files
    import os
    for sample in sample_files:
        try:
            os.remove(sample["filename"])
        except:
            pass


def demo_component_analysis():
    """Demonstrate component analysis features"""
    
    print("\n" + "=" * 70)
    print("COMPONENT ANALYSIS DEMO")
    print("=" * 70)
    
    validator = SchemaValidator()
    
    # Test different model configurations
    models = [
        {
            "name": "Complete Wan Model",
            "data": {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.21.0",
                "transformer": ["diffusers", "Transformer2DModel"],
                "transformer_2": ["diffusers", "Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "CLIPTokenizer"],
                "scheduler": ["diffusers", "DDIMScheduler"],
                "boundary_ratio": 0.5
            }
        },
        {
            "name": "Incomplete SD Model",
            "data": {
                "_class_name": "StableDiffusionPipeline",
                "_diffusers_version": "0.21.0",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"]
                # Missing text_encoder, tokenizer, scheduler
            }
        }
    ]
    
    for model in models:
        print(f"\nAnalyzing: {model['name']}")
        print("-" * 40)
        
        result = validator.validate_model_index_dict(model['data'])
        
        if result.schema:
            schema = result.schema
            
            print(f"Model Type: {schema.detect_model_type().value}")
            print(f"Is Wan Architecture: {schema.is_wan_architecture()}")
            
            required = schema.get_required_components()
            missing = schema.get_missing_components()
            
            print(f"Required Components: {', '.join(required) if required else 'None'}")
            print(f"Missing Components: {', '.join(missing) if missing else 'None'}")
            
            if schema.is_wan_architecture():
                wan_issues = schema.validate_wan_specific_attributes()
                if wan_issues:
                    print("Wan-specific Issues:")
                    for issue in wan_issues:
                        print(f"  • {issue}")


if __name__ == "__main__":
    demo_schema_validation()
    demo_file_validation()
    demo_component_analysis()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThe model index schema validation system provides:")
    print("• Comprehensive validation of model_index.json files")
    print("• Support for both standard Diffusers and Wan architectures")
    print("• Detailed error reporting and suggested fixes")
    print("• Component analysis and missing dependency detection")
    print("• Validation history tracking")
    print("• Flexible validation for both files and data dictionaries")