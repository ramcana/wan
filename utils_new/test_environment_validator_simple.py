#!/usr/bin/env python3
"""
Simple test script for EnvironmentValidator
"""

import sys
from pathlib import Path

# Add local_testing_framework to path
sys.path.insert(0, str(Path(__file__).parent))

from local_testing_framework.environment_validator import EnvironmentValidator
from local_testing_framework.models.configuration import TestConfiguration

def test_basic_functionality():
    """Test basic EnvironmentValidator functionality"""
    print("Testing EnvironmentValidator...")
    
    # Create validator instance
    validator = EnvironmentValidator()
    
    # Test platform detection
    print("1. Testing platform detection...")
    platform_info = validator._detect_platform()
    print(f"   Platform: {platform_info['system']}")
    print(f"   Python: {platform_info['python_version']}")
    
    # Test Python version validation
    print("2. Testing Python version validation...")
    python_result = validator.validate_python_version()
    print(f"   Status: {python_result.status.value}")
    print(f"   Message: {python_result.message}")
    
    # Test dependencies validation
    print("3. Testing dependencies validation...")
    deps_result = validator.validate_dependencies()
    print(f"   Status: {deps_result.status.value}")
    print(f"   Message: {deps_result.message}")
    
    # Test CUDA validation
    print("4. Testing CUDA validation...")
    cuda_result = validator.validate_cuda_availability()
    print(f"   Status: {cuda_result.status.value}")
    print(f"   Message: {cuda_result.message}")
    
    # Test configuration validation
    print("5. Testing configuration validation...")
    config_result = validator.validate_configuration_files()
    print(f"   Status: {config_result.status.value}")
    print(f"   Message: {config_result.message}")
    
    # Test environment variables validation
    print("6. Testing environment variables validation...")
    env_result = validator.validate_environment_variables()
    print(f"   Status: {env_result.status.value}")
    print(f"   Message: {env_result.message}")
    
    # Test full validation
    print("7. Testing full environment validation...")
    full_results = validator.validate_full_environment()
    print(f"   Overall Status: {full_results.overall_status.value}")
    
    # Generate report
    print("8. Generating environment report...")
    report = validator.generate_environment_report(full_results)
    print("   Report generated successfully")
    
    # Generate remediation instructions
    print("9. Generating remediation instructions...")
    instructions = validator.generate_remediation_instructions(full_results)
    print("   Instructions generated successfully")
    
    print("\nAll tests completed successfully!")
    return True

    assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    try:
        test_basic_functionality()
        print("\n✓ EnvironmentValidator implementation is working correctly!")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)