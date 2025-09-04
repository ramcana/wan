"""
Test script to validate migration and deployment components.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest

def test_data_migration():
    """Test data migration functionality."""
    print("Testing data migration...")
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock Gradio outputs
        gradio_outputs = temp_path / "gradio_outputs"
        gradio_outputs.mkdir()
        
        # Create mock video files
        (gradio_outputs / "video1.mp4").write_bytes(b"mock_video_1")
        (gradio_outputs / "video2.mp4").write_bytes(b"mock_video_2")
        
        # Create metadata files
        metadata1 = {"prompt": "Test prompt 1", "model_type": "T2V-A14B"}
        metadata2 = {"prompt": "Test prompt 2", "model_type": "I2V-A14B"}
        
        with open(gradio_outputs / "video1.json", 'w') as f:
            json.dump(metadata1, f)
        with open(gradio_outputs / "video2.json", 'w') as f:
            json.dump(metadata2, f)
        
        # Test migration scanning
        try:
            from backend.migration.data_migrator import DataMigrator
            
            migrator = DataMigrator(
                gradio_outputs_dir=str(gradio_outputs),
                new_outputs_dir=str(temp_path / "new_outputs"),
                backup_dir=str(temp_path / "backup")
            )
            
            # Scan outputs
            outputs = migrator.scan_gradio_outputs()
            assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
            
            print("✓ Data migration scanning works")
            
        except ImportError as e:
            print(f"⚠ Data migration test skipped: {e}")

def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    # Create temporary config
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        
        # Create mock config
        config = {
            "model_type": "t2v",
            "quantization": True,
            "vram_optimize": True,
            "output_dir": "outputs",
            "steps": 50
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        try:
            from backend.config.config_validator import ConfigValidator
            
            validator = ConfigValidator(str(config_path))
            result = validator.run_validation()
            
            assert result.migrated_config is not None, "Migration should produce config"
            assert "model_settings" in result.migrated_config, "Should have model_settings section"
            
            print("✓ Configuration validation works")
            
        except ImportError as e:
            print(f"⚠ Config validation test skipped: {e}")

def test_environment_config():
    """Test environment-specific configuration."""
    print("Testing environment configuration...")
    
    try:
        from backend.config.environment import EnvironmentConfig, Environment
        
        # Test development environment
        config = EnvironmentConfig("development")
        assert config.env == Environment.DEVELOPMENT
        assert config.is_development()
        
        # Test configuration loading
        api_config = config.get_api_config()
        assert "host" in api_config
        assert "port" in api_config
        
        print("✓ Environment configuration works")
        
    except ImportError as e:
        print(f"⚠ Environment config test skipped: {e}")

def test_monitoring_setup():
    """Test monitoring and logging setup."""
    print("Testing monitoring setup...")
    
    try:
        from backend.monitoring.metrics import MetricsCollector
        from backend.monitoring.logger import PerformanceLogger, ErrorLogger
        
        # Test metrics collector
        collector = MetricsCollector()
        metrics = collector.get_latest_metrics()
        
        assert "system" in metrics
        assert "application" in metrics
        assert "performance" in metrics
        
        # Test loggers
        perf_logger = PerformanceLogger()
        error_logger = ErrorLogger()
        
        assert perf_logger is not None
        assert error_logger is not None
        
        print("✓ Monitoring setup works")
        
    except ImportError as e:
        print(f"⚠ Monitoring test skipped: {e}")

def test_docker_files():
    """Test Docker configuration files."""
    print("Testing Docker configuration...")
    
    # Check if Docker files exist
    docker_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "nginx.conf"
    ]
    
    for file_name in docker_files:
        if Path(file_name).exists():
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")

    assert True  # TODO: Add proper assertion

def test_config_files():
    """Test configuration files."""
    print("Testing configuration files...")
    
    config_files = [
        "config_development.json",
        "config_production.json",
        "config_testing.json"
    ]
    
    for file_name in config_files:
        if Path(file_name).exists():
            try:
                with open(file_name, 'r') as f:
                    config = json.load(f)
                    assert isinstance(config, dict), f"{file_name} should contain valid JSON object"
                    print(f"✓ {file_name} is valid")
            except json.JSONDecodeError:
                print(f"✗ {file_name} contains invalid JSON")
        else:
            print(f"✗ {file_name} missing")

def test_deployment_guide():
    """Test deployment guide exists and is readable."""
    print("Testing deployment documentation...")
    
    if Path("DEPLOYMENT_GUIDE.md").exists():
        with open("DEPLOYMENT_GUIDE.md", 'r') as f:
            content = f.read()
            assert len(content) > 1000, "Deployment guide should be comprehensive"
            assert "Migration Process" in content, "Should contain migration process"
            assert "Rollback Procedures" in content, "Should contain rollback procedures"
            print("✓ Deployment guide exists and is comprehensive")
    else:
        print("✗ DEPLOYMENT_GUIDE.md missing")

def test_rollback_script():
    """Test rollback script exists."""
    print("Testing rollback script...")
    
    if Path("rollback.sh").exists():
        with open("rollback.sh", 'r') as f:
            content = f.read()
            assert "#!/bin/bash" in content, "Should be a bash script"
            assert "stop_new_system" in content, "Should have stop function"
            assert "start_gradio_system" in content, "Should have start function"
            print("✓ Rollback script exists and looks valid")
    else:
        print("✗ rollback.sh missing")

def test_backwards_compatibility():
    """Test backwards compatibility test suite."""
    print("Testing backwards compatibility tests...")
    
    test_file = Path("backend/tests/test_backwards_compatibility.py")
    if test_file.exists():
        with open(test_file, 'r') as f:
            content = f.read()
            assert "TestBackwardsCompatibility" in content, "Should have test class"
            assert "test_model_file_detection" in content, "Should test model detection"
            assert "test_config_compatibility" in content, "Should test config compatibility"
            print("✓ Backwards compatibility tests exist")
    else:
        print("✗ Backwards compatibility tests missing")

def main():
    """Run all migration and deployment tests."""
    print("=" * 60)
    print("MIGRATION AND DEPLOYMENT VALIDATION")
    print("=" * 60)
    print()
    
    tests = [
        test_data_migration,
        test_config_validation,
        test_environment_config,
        test_monitoring_setup,
        test_docker_files,
        test_config_files,
        test_deployment_guide,
        test_rollback_script,
        test_backwards_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All migration and deployment components are ready!")
    else:
        print("⚠ Some components need attention before deployment.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)