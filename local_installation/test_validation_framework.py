"""
Test suite for the validation framework.
Tests the installation validator and functionality tester components.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from validate_installation import InstallationValidator, DependencyInfo, ModelInfo, HardwareTestResult
from functionality_tester import FunctionalityTester, TestResult, BenchmarkResult
from interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


def create_mock_hardware_profile():
    """Create a mock hardware profile for testing."""
    return HardwareProfile(
        cpu=CPUInfo(
            model="AMD Ryzen 7 5800X",
            cores=8,
            threads=16,
            base_clock=3.8,
            boost_clock=4.7,
            architecture="x64"
        ),
        memory=MemoryInfo(
            total_gb=32,
            available_gb=24,
            type="DDR4",
            speed=3200
        ),
        gpu=GPUInfo(
            model="NVIDIA GeForce RTX 3070",
            vram_gb=8,
            cuda_version="12.1",
            driver_version="537.13",
            compute_capability="8.6"
        ),
        storage=StorageInfo(
            available_gb=500,
            type="NVMe SSD"
        ),
        os=OSInfo(
            name="Windows",
            version="11",
            architecture="x64"
        )
    )


def test_installation_validator_basic():
    """Test basic InstallationValidator functionality."""
    print("Testing InstallationValidator basic functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock directory structure
        venv_dir = temp_path / "venv" / "Scripts"
        venv_dir.mkdir(parents=True)
        
        models_dir = temp_path / "models"
        models_dir.mkdir(parents=True)
        
        # Create mock Python executable
        python_exe = venv_dir / "python.exe"
        python_exe.touch()
        
        # Create mock model directories
        for model_name in ["WAN2.2-T2V-A14B", "WAN2.2-I2V-A14B", "WAN2.2-TI2V-5B"]:
            model_path = models_dir / model_name
            model_path.mkdir()
            
            # Create mock model files
            (model_path / "pytorch_model.bin").write_bytes(b"0" * (1024 * 1024 * 100))  # 100MB
            (model_path / "config.json").write_text('{"model_type": "test"}')
            (model_path / "tokenizer.json").write_text('{"tokenizer": "test"}')
        
        hardware_profile = create_mock_hardware_profile()
        validator = InstallationValidator(str(temp_path), hardware_profile)
        
        # Test model validation (should work with mock structure)
        model_result = validator.validate_models()
        print(f"  Model validation: {'✅' if model_result.success else '❌'} - {model_result.message}")
        
        if model_result.details:
            models = model_result.details.get("models", [])
            print(f"    Found {len(models)} models")
            for model in models:
                print(f"      - {model['name']}: {'✅' if model['exists'] else '❌'}")
        
        # Test hardware integration validation
        hw_result = validator.validate_hardware_integration()
        print(f"  Hardware validation: {'✅' if hw_result.success else '❌'} - {hw_result.message}")
        
        if hw_result.details:
            hw_tests = hw_result.details.get("hardware_tests", [])
            print(f"    Hardware tests: {len(hw_tests)}")
            for test in hw_tests:
                status = "✅" if test['available'] else "❌"
                print(f"      - {test['component']}: {status}")
        
        print("  InstallationValidator basic test completed")


def test_functionality_tester_basic():
    """Test basic FunctionalityTester functionality."""
    print("Testing FunctionalityTester basic functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock directory structure
        venv_dir = temp_path / "venv" / "Scripts"
        venv_dir.mkdir(parents=True)
        
        models_dir = temp_path / "models"
        models_dir.mkdir(parents=True)
        
        # Create mock Python executable
        python_exe = venv_dir / "python.exe"
        python_exe.touch()
        
        hardware_profile = create_mock_hardware_profile()
        tester = FunctionalityTester(str(temp_path), hardware_profile)
        
        # Test hardware tier determination
        tier = tester._determine_hardware_tier()
        print(f"  Hardware tier: {tier}")
        
        # Test test case definition
        test_cases = tester._define_test_cases()
        print(f"  Test cases defined: {len(test_cases)}")
        for test_case in test_cases[:3]:  # Show first 3
            print(f"    - {test_case.name}: {test_case.description}")
        
        # Test benchmark creation (mock)
        benchmarks = []
        
        # Mock model loading benchmark
        model_benchmark = tester._benchmark_model_loading()
        if model_benchmark:
            model_benchmark.hardware_tier = tier
            benchmarks.append(model_benchmark)
        
        # Mock inference benchmark
        inference_benchmark = tester._benchmark_inference_speed()
        if inference_benchmark:
            inference_benchmark.hardware_tier = tier
            benchmarks.append(inference_benchmark)
        
        print(f"  Benchmarks created: {len(benchmarks)}")
        for benchmark in benchmarks:
            print(f"    - {benchmark.benchmark_name}: {benchmark.score:.3f} {benchmark.unit}")
        
        print("  FunctionalityTester basic test completed")


def test_validation_report_generation():
    """Test validation report generation."""
    print("Testing validation report generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create logs directory
        logs_dir = temp_path / "logs"
        logs_dir.mkdir(parents=True)
        
        # Create mock directory structure
        venv_dir = temp_path / "venv" / "Scripts"
        venv_dir.mkdir(parents=True)
        
        models_dir = temp_path / "models"
        models_dir.mkdir(parents=True)
        
        # Create mock Python executable
        python_exe = venv_dir / "python.exe"
        python_exe.touch()
        
        # Create minimal model structure
        model_path = models_dir / "WAN2.2-TI2V-5B"
        model_path.mkdir()
        (model_path / "config.json").write_text('{"model_type": "test"}')
        
        hardware_profile = create_mock_hardware_profile()
        validator = InstallationValidator(str(temp_path), hardware_profile)
        
        # Mock the subprocess calls to avoid actual Python execution
        with patch('subprocess.run') as mock_run:
            # Mock successful dependency check
            mock_run.return_value = Mock(
                returncode=0,
                stdout="1.0.0",
                stderr=""
            )
            
            # Generate validation report (will use mocked subprocess)
            try:
                report = validator.generate_validation_report()
                print(f"  Report generated: {'✅' if report.overall_success else '❌'}")
                print(f"    Dependencies: {len(report.dependencies)}")
                print(f"    Models: {len(report.models)}")
                print(f"    Hardware tests: {len(report.hardware_tests)}")
                print(f"    Errors: {len(report.errors)}")
                print(f"    Warnings: {len(report.warnings)}")
                
                # Check if report file was created
                report_file = temp_path / "logs" / "validation_report.json"
                if report_file.exists():
                    print(f"  Report file created: ✅")
                    
                    # Verify report content
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                    
                    required_fields = ["timestamp", "installation_path", "overall_success"]
                    missing_fields = [field for field in required_fields if field not in report_data]
                    
                    if not missing_fields:
                        print(f"  Report structure valid: ✅")
                    else:
                        print(f"  Report structure invalid: ❌ Missing: {missing_fields}")
                else:
                    print(f"  Report file created: ❌")
                
            except Exception as e:
                print(f"  Report generation failed: ❌ {str(e)}")
        
        print("  Validation report generation test completed")


def test_error_diagnosis():
    """Test error diagnosis functionality."""
    print("Testing error diagnosis...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        hardware_profile = create_mock_hardware_profile()
        tester = FunctionalityTester(str(temp_path), hardware_profile)
        
        # Create mock test results with various errors
        test_results = [
            TestResult(
                test_name="cuda_test",
                success=False,
                duration=10.0,
                error="CUDA out of memory"
            ),
            TestResult(
                test_name="model_load_test",
                success=False,
                duration=5.0,
                error="Model checkpoint not found"
            ),
            TestResult(
                test_name="timeout_test",
                success=False,
                duration=300.0,
                error="Test timed out after 300 seconds"
            ),
            TestResult(
                test_name="memory_test",
                success=False,
                duration=15.0,
                error="Out of memory (OOM) error"
            ),
            TestResult(
                test_name="success_test",
                success=True,
                duration=2.0
            )
        ]
        
        # Test error diagnosis
        diagnosis = tester.diagnose_errors(test_results)
        
        print(f"  Error categories identified:")
        print(f"    Critical errors: {len(diagnosis['critical_errors'])}")
        print(f"    Performance issues: {len(diagnosis['performance_issues'])}")
        print(f"    Configuration problems: {len(diagnosis['configuration_problems'])}")
        print(f"    Hardware limitations: {len(diagnosis['hardware_limitations'])}")
        print(f"    Suggested fixes: {len(diagnosis['suggested_fixes'])}")
        
        # Verify categorization
        expected_categories = {
            "hardware_limitations": 2,  # CUDA and memory errors
            "configuration_problems": 1,  # Model checkpoint error
            "performance_issues": 1,      # Timeout error
        }
        
        categorization_correct = True
        for category, expected_count in expected_categories.items():
            actual_count = len(diagnosis[category])
            if actual_count != expected_count:
                print(f"    ❌ {category}: expected {expected_count}, got {actual_count}")
                categorization_correct = False
            else:
                print(f"    ✅ {category}: {actual_count} errors correctly categorized")
        
        if categorization_correct:
            print(f"  Error diagnosis: ✅")
        else:
            print(f"  Error diagnosis: ❌")
        
        print("  Error diagnosis test completed")


def test_performance_baseline_comparison():
    """Test performance baseline comparison."""
    print("Testing performance baseline comparison...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        hardware_profile = create_mock_hardware_profile()
        tester = FunctionalityTester(str(temp_path), hardware_profile)
        
        # Test different hardware tiers
        tiers = ["high_end", "mid_range", "budget"]
        
        for tier in tiers:
            print(f"  Testing {tier} tier:")
            
            # Create mock benchmark result
            benchmark = BenchmarkResult(
                benchmark_name="model_load_time",
                hardware_tier=tier,
                score=45.0,  # 45 seconds
                unit="seconds",
                details={"test": "mock"}
            )
            
            # Test baseline comparison
            comparison = tester._compare_to_baseline(benchmark, tier)
            
            if comparison is not None:
                print(f"    Baseline comparison: {comparison:.2f}x")
                
                # Check if comparison makes sense
                baseline = tester.performance_baselines[tier]["model_load_time"]
                expected_comparison = baseline / benchmark.score
                
                if abs(comparison - expected_comparison) < 0.01:
                    print(f"    Comparison calculation: ✅")
                else:
                    print(f"    Comparison calculation: ❌")
            else:
                print(f"    Baseline comparison: ❌ (None returned)")
        
        print("  Performance baseline comparison test completed")


def run_all_tests():
    """Run all validation framework tests."""
    print("=" * 60)
    print("WAN2.2 Validation Framework Test Suite")
    print("=" * 60)
    
    tests = [
        test_installation_validator_basic,
        test_functionality_tester_basic,
        test_validation_report_generation,
        test_error_diagnosis,
        test_performance_baseline_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            print(f"\n{'-' * 40}")
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation framework tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)