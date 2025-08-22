#!/usr/bin/env python3
"""
Simple test script for ProductionValidator functionality
"""

import json
import tempfile
import os
from pathlib import Path

from local_testing_framework.production_validator import ProductionValidator


def test_production_validator():
    """Test basic ProductionValidator functionality"""
    print("🧪 Testing ProductionValidator...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create test configuration
            config = {
                "system": {"gpu_enabled": True},
                "directories": {"models": "models", "outputs": "outputs"},
                "optimization": {
                    "enable_attention_slicing": True,
                    "enable_vae_tiling": True
                },
                "performance": {
                    "stats_refresh_interval": 5,
                    "vram_warning_threshold": 0.8,
                    "cpu_warning_percent": 75,
                    "max_queue_size": 5
                },
                "security": {
                    "enable_https": False,  # Disabled for test
                    "enable_auth": False,   # Disabled for test
                    "cors_origins": ["https://example.com"]
                }
            }
            
            config_path = "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create test .env file
            env_path = ".env"
            with open(env_path, 'w') as f:
                f.write("HF_TOKEN=test_token_12345\n")
                f.write("CUDA_VISIBLE_DEVICES=0\n")
            
            print("✅ Test files created")
            
            # Initialize validator
            validator = ProductionValidator(config_path)
            print("✅ ProductionValidator initialized")
            
            # Test individual validation methods
            print("\n🔍 Testing individual validation methods:")
            
            # Test config validation
            config_result = validator._validate_config_file()
            print(f"   Config validation: {config_result.status.value} - {config_result.message}")
            
            # Test env validation
            env_result = validator._validate_env_file()
            print(f"   Env validation: {env_result.status.value} - {env_result.message}")
            
            # Test security config validation
            security_result = validator._validate_security_config()
            print(f"   Security config: {security_result.status.value} - {security_result.message}")
            
            # Test performance config validation
            perf_result = validator._validate_performance_config()
            print(f"   Performance config: {perf_result.status.value} - {perf_result.message}")
            
            # Test file permissions validation
            perm_result = validator._validate_file_permissions()
            print(f"   File permissions: {perm_result.status.value} - {perm_result.message}")
            
            # Test load test simulation (short duration)
            print("\n🚀 Testing load test simulation:")
            validator.load_test_concurrent_users = 2
            validator.load_test_duration_minutes = 0.05  # 3 seconds
            
            load_results = validator._run_concurrent_load_test()
            print(f"   Concurrent users: {load_results['concurrent_requests']}")
            print(f"   Successful requests: {load_results['successful_requests']}")
            print(f"   Failed requests: {load_results['failed_requests']}")
            print(f"   Success rate: {load_results['success_rate']:.2%}")
            
            # Test queue management validation
            queue_result = validator._validate_queue_management(load_results)
            print(f"   Queue management: {queue_result.status.value} - {queue_result.message}")
            
            print("\n✅ All basic tests completed successfully!")
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_production_validator()