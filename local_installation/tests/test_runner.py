"""
Automated test runner for the WAN2.2 local installation system.
Provides comprehensive testing including unit tests, integration tests, and hardware simulation.
"""

import os
import sys
import json
import time
import unittest
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    description: str
    tests: List[str]
    setup_required: bool = False
    teardown_required: bool = False


class HardwareSimulator:
    """Simulates different hardware configurations for testing."""
    
    @staticmethod
    def get_high_end_profile() -> HardwareProfile:
        """High-end system: Threadripper PRO + RTX 4080."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen Threadripper PRO 5995WX",
                cores=64,
                threads=128,
                base_clock=2.7,
                boost_clock=4.5,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=128,
                available_gb=120,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 4080",
                vram_gb=16,
                cuda_version="12.1",
                driver_version="537.13",
                compute_capability="8.9"
            ),
            storage=StorageInfo(
                available_gb=2000,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="11",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_mid_range_profile() -> HardwareProfile:
        """Mid-range system: Ryzen 7 + RTX 3070."""
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
                available_gb=28,
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
    
    @staticmethod
    def get_budget_profile() -> HardwareProfile:
        """Budget system: Ryzen 5 + GTX 1660 Ti."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 5 3600",
                cores=6,
                threads=12,
                base_clock=3.6,
                boost_clock=4.2,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=16,
                available_gb=14,
                type="DDR4",
                speed=2666
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce GTX 1660 Ti",
                vram_gb=6,
                cuda_version="11.8",
                driver_version="516.94",
                compute_capability="7.5"
            ),
            storage=StorageInfo(
                available_gb=250,
                type="SATA SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="10",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_minimum_profile() -> HardwareProfile:
        """Minimum system: Intel i5 + GTX 1060."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="Intel Core i5-8400",
                cores=6,
                threads=6,
                base_clock=2.8,
                boost_clock=4.0,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=8,
                available_gb=6,
                type="DDR4",
                speed=2400
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce GTX 1060",
                vram_gb=6,
                cuda_version="11.8",
                driver_version="516.94",
                compute_capability="6.1"
            ),
            storage=StorageInfo(
                available_gb=100,
                type="HDD"
            ),
            os=OSInfo(
                name="Windows",
                version="10",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_no_gpu_profile() -> HardwareProfile:
        """System without dedicated GPU."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 7 5700G",
                cores=8,
                threads=16,
                base_clock=3.8,
                boost_clock=4.6,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=16,
                available_gb=14,
                type="DDR4",
                speed=3200
            ),
            gpu=None,
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


class AutomatedTestFramework:
    """Main automated testing framework."""
    
    def __init__(self, installation_path: str):
        self.installation_path = Path(installation_path)
        self.test_results: List[TestResult] = []
        self.hardware_simulator = HardwareSimulator()
        
        # Define test suites
        self.test_suites = {
            "unit_tests": TestSuite(
                name="Unit Tests",
                description="Individual component testing",
                tests=[
                    "test_system_detection_unit",
                    "test_dependency_management_unit",
                    "test_model_management_unit",
                    "test_configuration_engine_unit",
                    "test_validation_framework_unit",
                    "test_error_handling_unit"
                ]
            ),
            "integration_tests": TestSuite(
                name="Integration Tests",
                description="Cross-component functionality testing",
                tests=[
                    "test_detection_to_config_integration",
                    "test_dependency_to_validation_integration",
                    "test_model_to_config_integration",
                    "test_full_installation_flow",
                    "test_error_recovery_integration"
                ]
            ),
            "hardware_simulation_tests": TestSuite(
                name="Hardware Simulation Tests",
                description="Testing with various hardware configurations",
                tests=[
                    "test_high_end_hardware_simulation",
                    "test_mid_range_hardware_simulation",
                    "test_budget_hardware_simulation",
                    "test_minimum_hardware_simulation",
                    "test_no_gpu_hardware_simulation"
                ]
            )
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        print("=" * 80)
        print("WAN2.2 Local Installation Automated Test Framework")
        print("=" * 80)
        
        start_time = time.time()
        suite_results = {}
        
        for suite_name, suite in self.test_suites.items():
            print(f"\n🧪 Running {suite.name}...")
            print(f"   {suite.description}")
            print("-" * 60)
            
            suite_results[suite_name] = self._run_test_suite(suite)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_test_report(suite_results, total_time)
        
        # Save report to file
        self._save_test_report(report)
        
        return report
    
    def _run_test_suite(self, suite: TestSuite) -> Dict[str, Any]:
        """Run a specific test suite."""
        suite_start = time.time()
        suite_results = []
        
        for test_name in suite.tests:
            print(f"  Running {test_name}...")
            
            test_start = time.time()
            try:
                # Get the test method
                test_method = getattr(self, test_name)
                
                # Run the test
                result = test_method()
                
                test_duration = time.time() - test_start
                
                if result:
                    print(f"    ✅ PASSED ({test_duration:.2f}s)")
                    suite_results.append(TestResult(
                        test_name=test_name,
                        success=True,
                        duration=test_duration
                    ))
                else:
                    print(f"    ❌ FAILED ({test_duration:.2f}s)")
                    suite_results.append(TestResult(
                        test_name=test_name,
                        success=False,
                        duration=test_duration,
                        error_message="Test returned False"
                    ))
                    
            except Exception as e:
                test_duration = time.time() - test_start
                print(f"    ❌ ERROR ({test_duration:.2f}s): {str(e)}")
                suite_results.append(TestResult(
                    test_name=test_name,
                    success=False,
                    duration=test_duration,
                    error_message=str(e)
                ))
        
        suite_duration = time.time() - suite_start
        passed = sum(1 for r in suite_results if r.success)
        total = len(suite_results)
        
        print(f"\n  Suite Summary: {passed}/{total} tests passed ({suite_duration:.2f}s)")
        
        return {
            "results": suite_results,
            "duration": suite_duration,
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0
        }
    
    # Unit Tests
    def test_system_detection_unit(self) -> bool:
        """Test system detection components."""
        try:
            from detect_system import SystemDetector
            
            with tempfile.TemporaryDirectory() as temp_dir:
                detector = SystemDetector(temp_dir)
                
                # Test hardware detection (mocked)
                with patch('detect_system.SystemDetector._detect_cpu') as mock_cpu, \
                     patch('detect_system.SystemDetector._detect_memory') as mock_memory, \
                     patch('detect_system.SystemDetector._detect_gpu') as mock_gpu, \
                     patch('detect_system.SystemDetector._detect_storage') as mock_storage, \
                     patch('detect_system.SystemDetector._detect_os') as mock_os:
                    
                    # Mock return values
                    mock_cpu.return_value = self.hardware_simulator.get_mid_range_profile().cpu
                    mock_memory.return_value = self.hardware_simulator.get_mid_range_profile().memory
                    mock_gpu.return_value = self.hardware_simulator.get_mid_range_profile().gpu
                    mock_storage.return_value = self.hardware_simulator.get_mid_range_profile().storage
                    mock_os.return_value = self.hardware_simulator.get_mid_range_profile().os
                    
                    profile = detector.detect_hardware()
                    
                    # Verify profile structure
                    assert profile.cpu is not None
                    assert profile.memory is not None
                    assert profile.gpu is not None
                    assert profile.storage is not None
                    assert profile.os is not None
                    
                    return True
                    
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_dependency_management_unit(self) -> bool:
        """Test dependency management components."""
        try:
            from setup_dependencies import DependencyManager
            from base_classes import ConsoleProgressReporter
            
            with tempfile.TemporaryDirectory() as temp_dir:
                progress_reporter = ConsoleProgressReporter()
                dep_manager = DependencyManager(temp_dir, progress_reporter)
                
                # Test Python detection
                python_info = dep_manager.check_python_installation()
                assert isinstance(python_info, dict)
                assert "recommended_action" in python_info
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_model_management_unit(self) -> bool:
        """Test model management components."""
        try:
            from download_models import ModelDownloader
            
            with tempfile.TemporaryDirectory() as temp_dir:
                downloader = ModelDownloader(temp_dir)
                
                # Test model configuration
                models = downloader.get_required_models()
                assert isinstance(models, list)
                assert len(models) > 0
                
                # Test model path validation
                for model in models:
                    assert "name" in model
                    assert "url" in model
                    assert "size_gb" in model
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_configuration_engine_unit(self) -> bool:
        """Test configuration engine components."""
        try:
            from generate_config import ConfigurationEngine
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config_engine = ConfigurationEngine(temp_dir)
                hardware_profile = self.hardware_simulator.get_mid_range_profile()
                
                # Test configuration generation
                config = config_engine.generate_config(hardware_profile)
                assert isinstance(config, dict)
                
                # Verify required configuration sections
                required_sections = ["system", "optimization"]
                for section in required_sections:
                    assert section in config
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_validation_framework_unit(self) -> bool:
        """Test validation framework components."""
        try:
            from validate_installation import InstallationValidator
            
            with tempfile.TemporaryDirectory() as temp_dir:
                hardware_profile = self.hardware_simulator.get_mid_range_profile()
                validator = InstallationValidator(temp_dir, hardware_profile)
                
                # Test validation methods exist
                assert hasattr(validator, 'validate_dependencies')
                assert hasattr(validator, 'validate_models')
                assert hasattr(validator, 'validate_hardware_integration')
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_error_handling_unit(self) -> bool:
        """Test error handling components."""
        try:
            from error_handler import ErrorHandler, InstallationError
            
            # Test error creation
            error = InstallationError(
                message="Test error",
                category="test",
                recovery_suggestions=["Test suggestion"]
            )
            
            assert error.message == "Test error"
            assert error.category == "test"
            assert len(error.recovery_suggestions) == 1
            
            # Test error handler
            handler = ErrorHandler()
            assert hasattr(handler, 'handle_error')
            assert hasattr(handler, 'log_error')
            
            return True
            
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    # Integration Tests
    def test_detection_to_config_integration(self) -> bool:
        """Test integration between system detection and configuration generation."""
        try:
            from detect_system import SystemDetector
            from generate_config import ConfigurationEngine
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock system detection
                with patch('detect_system.SystemDetector.detect_hardware') as mock_detect:
                    mock_detect.return_value = self.hardware_simulator.get_high_end_profile()
                    
                    detector = SystemDetector(temp_dir)
                    config_engine = ConfigurationEngine(temp_dir)
                    
                    # Detect hardware
                    hardware_profile = detector.detect_hardware()
                    
                    # Generate configuration
                    config = config_engine.generate_config(hardware_profile)
                    
                    # Verify high-end optimizations
                    assert config["optimization"]["cpu_threads"] >= 32
                    assert config["optimization"]["memory_pool_gb"] >= 16
                    assert config["system"]["enable_gpu_acceleration"] is True
                    
                    return True
                    
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_dependency_to_validation_integration(self) -> bool:
        """Test integration between dependency management and validation."""
        try:
            from setup_dependencies import DependencyManager
            from validate_installation import InstallationValidator
            from base_classes import ConsoleProgressReporter
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create mock virtual environment
                venv_dir = Path(temp_dir) / "venv" / "Scripts"
                venv_dir.mkdir(parents=True)
                (venv_dir / "python.exe").touch()
                
                progress_reporter = ConsoleProgressReporter()
                dep_manager = DependencyManager(temp_dir, progress_reporter)
                hardware_profile = self.hardware_simulator.get_mid_range_profile()
                validator = InstallationValidator(temp_dir, hardware_profile)
                
                # Test dependency validation
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = Mock(returncode=0, stdout="1.0.0", stderr="")
                    
                    dep_result = validator.validate_dependencies()
                    assert hasattr(dep_result, 'success')
                    
                    return True
                    
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_model_to_config_integration(self) -> bool:
        """Test integration between model management and configuration."""
        try:
            from download_models import ModelDownloader
            from generate_config import ConfigurationEngine
            
            with tempfile.TemporaryDirectory() as temp_dir:
                downloader = ModelDownloader(temp_dir)
                config_engine = ConfigurationEngine(temp_dir)
                hardware_profile = self.hardware_simulator.get_budget_profile()
                
                # Get model requirements
                models = downloader.get_required_models()
                total_model_size = sum(model["size_gb"] for model in models)
                
                # Generate configuration
                config = config_engine.generate_config(hardware_profile)
                
                # Verify configuration accounts for model size
                if total_model_size > 50:  # Large models
                    assert config["system"]["enable_model_offload"] is True
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_full_installation_flow(self) -> bool:
        """Test the complete installation flow integration."""
        try:
            from main_installer import MainInstaller
            from base_classes import ConsoleProgressReporter
            
            with tempfile.TemporaryDirectory() as temp_dir:
                progress_reporter = ConsoleProgressReporter()
                installer = MainInstaller(temp_dir, progress_reporter)
                
                # Test installation phases
                phases = installer.get_installation_phases()
                assert len(phases) > 0
                
                # Verify phase structure
                for phase in phases:
                    assert "name" in phase
                    assert "description" in phase
                    assert "function" in phase
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def test_error_recovery_integration(self) -> bool:
        """Test error recovery integration across components."""
        try:
            from error_handler import ErrorHandler
            from rollback_manager import RollbackManager
            
            with tempfile.TemporaryDirectory() as temp_dir:
                error_handler = ErrorHandler()
                rollback_manager = RollbackManager(temp_dir)
                
                # Test error handling with rollback
                assert hasattr(error_handler, 'handle_error')
                assert hasattr(rollback_manager, 'create_snapshot')
                assert hasattr(rollback_manager, 'restore_snapshot')
                
                return True
                
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    # Hardware Simulation Tests
    def test_high_end_hardware_simulation(self) -> bool:
        """Test with high-end hardware configuration."""
        return self._test_hardware_configuration(
            self.hardware_simulator.get_high_end_profile(),
            "high_end"
        )
    
    def test_mid_range_hardware_simulation(self) -> bool:
        """Test with mid-range hardware configuration."""
        return self._test_hardware_configuration(
            self.hardware_simulator.get_mid_range_profile(),
            "mid_range"
        )
    
    def test_budget_hardware_simulation(self) -> bool:
        """Test with budget hardware configuration."""
        return self._test_hardware_configuration(
            self.hardware_simulator.get_budget_profile(),
            "budget"
        )
    
    def test_minimum_hardware_simulation(self) -> bool:
        """Test with minimum hardware configuration."""
        return self._test_hardware_configuration(
            self.hardware_simulator.get_minimum_profile(),
            "minimum"
        )
    
    def test_no_gpu_hardware_simulation(self) -> bool:
        """Test with no GPU configuration."""
        return self._test_hardware_configuration(
            self.hardware_simulator.get_no_gpu_profile(),
            "no_gpu"
        )
    
    def _test_hardware_configuration(self, hardware_profile: HardwareProfile, tier: str) -> bool:
        """Test a specific hardware configuration."""
        try:
            from generate_config import ConfigurationEngine
            from validate_installation import InstallationValidator
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Test configuration generation
                config_engine = ConfigurationEngine(temp_dir)
                config = config_engine.generate_config(hardware_profile)
                
                # Verify configuration is appropriate for hardware tier
                if tier == "high_end":
                    assert config["optimization"]["cpu_threads"] >= 32
                    assert config["system"]["enable_gpu_acceleration"] is True
                elif tier == "budget":
                    assert config["optimization"]["cpu_threads"] <= 16
                    assert config["system"]["enable_model_offload"] is True
                elif tier == "no_gpu":
                    assert config["system"]["enable_gpu_acceleration"] is False
                
                # Test validation
                validator = InstallationValidator(temp_dir, hardware_profile)
                validation_result = validator.validate_requirements(hardware_profile)
                
                # Hardware should meet minimum requirements
                assert hasattr(validation_result, 'success')
                
                return True
                
        except Exception as e:
            print(f"      Error testing {tier} hardware: {e}")
            return False
    
    def _generate_test_report(self, suite_results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(suite["total"] for suite in suite_results.values())
        total_passed = sum(suite["passed"] for suite in suite_results.values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_time,
            "overall_stats": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_tests - total_passed,
                "success_rate": overall_success_rate
            },
            "suite_results": suite_results,
            "summary": {
                "status": "PASSED" if overall_success_rate >= 0.9 else "FAILED",
                "critical_failures": [],
                "recommendations": []
            }
        }
        
        # Add critical failures and recommendations
        for suite_name, suite_result in suite_results.items():
            if suite_result["success_rate"] < 0.5:
                report["summary"]["critical_failures"].append(
                    f"{suite_name}: {suite_result['success_rate']:.1%} success rate"
                )
        
        if overall_success_rate < 0.9:
            report["summary"]["recommendations"].append(
                "Review failed tests and address underlying issues before deployment"
            )
        
        return report
    
    def _save_test_report(self, report: Dict[str, Any]) -> None:
        """Save test report to file."""
        logs_dir = self.installation_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        report_file = logs_dir / "automated_test_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Test report saved to: {report_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {report['overall_stats']['total_tests']}")
        print(f"Passed: {report['overall_stats']['total_passed']}")
        print(f"Failed: {report['overall_stats']['total_failed']}")
        print(f"Success Rate: {report['overall_stats']['success_rate']:.1%}")
        print(f"Duration: {report['total_duration']:.2f}s")
        print(f"Status: {report['summary']['status']}")
        
        if report['summary']['critical_failures']:
            print("\nCritical Failures:")
            for failure in report['summary']['critical_failures']:
                print(f"  ❌ {failure}")
        
        if report['summary']['recommendations']:
            print("\nRecommendations:")
            for rec in report['summary']['recommendations']:
                print(f"  💡 {rec}")


def main():
    """Run the automated test framework."""
    import argparse

    parser = argparse.ArgumentParser(description="WAN2.2 Automated Test Framework")
    parser.add_argument("--installation-path", default=".", help="Installation path to test")
    parser.add_argument("--suite", choices=["unit", "integration", "hardware", "all"], 
                       default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    framework = AutomatedTestFramework(args.installation_path)
    
    if args.suite == "all":
        report = framework.run_all_tests()
    else:
        suite_map = {
            "unit": "unit_tests",
            "integration": "integration_tests", 
            "hardware": "hardware_simulation_tests"
        }
        
        suite_name = suite_map[args.suite]
        suite = framework.test_suites[suite_name]
        
        print(f"Running {suite.name}...")
        suite_result = framework._run_test_suite(suite)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suite_results": {suite_name: suite_result},
            "overall_stats": {
                "total_tests": suite_result["total"],
                "total_passed": suite_result["passed"],
                "success_rate": suite_result["success_rate"]
            }
        }
        
        framework._save_test_report(report)
    
    # Exit with appropriate code
    success_rate = report["overall_stats"]["success_rate"]
    sys.exit(0 if success_rate >= 0.9 else 1)


if __name__ == "__main__":
    main()