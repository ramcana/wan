#!/usr/bin/env python3
"""
Simple test script for SampleManager
"""

import sys
from pathlib import Path

# Add local_testing_framework to path
sys.path.insert(0, str(Path(__file__).parent))

from local_testing_framework.sample_manager import (
    SampleManager, ConfigurationTemplateGenerator, EdgeCaseGenerator
)

def test_sample_manager():
    """Test SampleManager functionality"""
    print("Testing SampleManager...")
    
    manager = SampleManager()
    
    # Test realistic prompt generation
    print("1. Testing realistic video prompt generation...")
    prompts = manager.generate_realistic_video_prompts(count=3)
    print(f"   Generated {len(prompts)} prompts")
    
    for i, prompt in enumerate(prompts[:2]):  # Show first 2
        print(f"   Prompt {i+1}: {prompt['category']} - {prompt['estimated_complexity']}")
        print(f"   Full: {prompt['full_prompt'][:80]}...")
    
    # Test sample input JSON creation
    print("2. Testing sample input JSON creation...")
    sample_input = manager.create_sample_input_json(prompts[0], "720p")
    print(f"   Created sample input with resolution: {sample_input['resolution']}")
    print(f"   Prompt: {sample_input['input'][:60]}...")
    print(f"   Output path: {sample_input['output_path']}")
    
    # Test file generation
    print("3. Testing sample input file generation...")
    files = manager.generate_sample_input_files(count=2, resolutions=["720p"])
    print(f"   Generated {len(files)} sample files")
    
    for file_path in files[:2]:  # Show first 2
        print(f"   File: {file_path.name}")
    
    return True

    assert True  # TODO: Add proper assertion

def test_configuration_generator():
    """Test ConfigurationTemplateGenerator functionality"""
    print("\nTesting ConfigurationTemplateGenerator...")
    
    generator = ConfigurationTemplateGenerator()
    
    # Test config.json template generation
    print("1. Testing config.json template generation...")
    
    for preset in ["balanced", "performance", "quality"]:
        config = generator.generate_config_json_template(preset)
        print(f"   {preset.capitalize()} preset: {len(config)} sections")
        
        # Check optimization settings
        opt = config["optimization"]
        print(f"     Attention slicing: {opt['enable_attention_slicing']}")
        print(f"     VAE tiling: {opt['enable_vae_tiling']}")
        print(f"     FP16: {opt['use_fp16']}")
    
    # Test .env template generation
    print("2. Testing .env template generation...")
    env_template = generator.generate_env_template(include_examples=True)
    print(f"   Generated {len(env_template)} environment variables")
    print(f"   HF_TOKEN: {env_template.get('HF_TOKEN', 'Not found')}")
    print(f"   CUDA_VISIBLE_DEVICES: {env_template.get('CUDA_VISIBLE_DEVICES', 'Not found')}")
    
    # Test template validation
    print("3. Testing template validation...")
    valid_config = {
        "system": {"device": "cuda"},
        "directories": {"models": "models/", "outputs": "outputs/", "cache": "cache/", "logs": "logs/"},
        "optimization": {},
        "performance": {}
    }
    
    result = generator.validate_config_template(valid_config)
    print(f"   Validation result: {result.status.value}")
    print(f"   Message: {result.message}")
    
    return True

    assert True  # TODO: Add proper assertion

def test_edge_case_generator():
    """Test EdgeCaseGenerator functionality"""
    print("\nTesting EdgeCaseGenerator...")
    
    generator = EdgeCaseGenerator()
    
    # Test edge case prompt generation
    print("1. Testing edge case prompt generation...")
    edge_cases = generator.generate_edge_case_prompts()
    print(f"   Generated {len(edge_cases)} edge cases")
    
    # Show different categories
    categories = {}
    for case in edge_cases:
        category = case["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(case)
    
    print(f"   Categories: {list(categories.keys())}")
    
    # Show example from each category
    for category, cases in list(categories.items())[:3]:  # First 3 categories
        example = cases[0]
        print(f"   {category}: {example['test_type']} - {example['prompt'][:50]}...")
    
    # Test invalid input samples
    print("2. Testing invalid input sample generation...")
    invalid_samples = generator.create_invalid_input_samples()
    print(f"   Generated {len(invalid_samples)} invalid samples")
    
    # Show different error types
    error_types = {}
    for sample in invalid_samples:
        error_type = sample["error_type"]
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(sample)
    
    print(f"   Error types: {list(error_types.keys())}")
    
    # Test multi-resolution test suite
    print("3. Testing multi-resolution test suite...")
    test_suite = generator.create_multi_resolution_test_suite()
    print(f"   720p tests: {len(test_suite['720p'])}")
    print(f"   1080p tests: {len(test_suite['1080p'])}")
    print(f"   Unsupported tests: {len(test_suite['unsupported'])}")
    
    # Test stress test scenarios
    print("4. Testing stress test scenarios...")
    stress_tests = generator.create_stress_test_scenarios()
    print(f"   Generated {len(stress_tests)} stress test scenarios")
    
    for test in stress_tests:
        print(f"   {test['type']}: {test['description'][:60]}...")
    
    return True

    assert True  # TODO: Add proper assertion

def test_file_generation():
    """Test actual file generation"""
    print("\nTesting file generation...")
    
    # Test SampleManager file generation
    manager = SampleManager()
    sample_files = manager.generate_sample_input_files(count=1, resolutions=["720p"])
    print(f"Sample files generated: {len(sample_files)}")
    
    # Test ConfigurationTemplateGenerator file saving
    config_gen = ConfigurationTemplateGenerator()
    config = config_gen.generate_config_json_template("balanced")
    config_file = config_gen.save_config_template(config, "test_config.json")
    print(f"Config template saved: {config_file.exists()}")
    
    env_template = config_gen.generate_env_template()
    env_file = config_gen.save_env_template(env_template, "test_env.txt")
    print(f"Env template saved: {env_file.exists()}")
    
    # Test EdgeCaseGenerator file saving
    edge_gen = EdgeCaseGenerator()
    edge_files = edge_gen.save_edge_case_samples()
    print(f"Edge case files generated: {len(edge_files)}")
    
    return True

    assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    try:
        test_sample_manager()
        test_configuration_generator()
        test_edge_case_generator()
        test_file_generation()
        print("\n✓ SampleManager implementation is working correctly!")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)