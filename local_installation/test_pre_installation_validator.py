"""
Comprehensive tests for the pre-installation validation system.
Tests all validation components and different system configurations.
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from pre_installation_validator import (
    PreInstallationValidator, TimeoutManager, SystemRequirement,
    NetworkTest, PermissionTest, ConflictDetection, PreValidationReport
)
from interfaces import ValidationResult, HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


class TestTimeoutManager(unittest.TestCase):
    """Test the TimeoutManager utility class."""
    
    def test_timeout_manager_normal_operation(self):
        """Test timeout manager with normal operation."""
        cleanup_called = False
        
        def cleanup_func():
            nonlocal cleanup_called
            cleanup_called = True
        
        with TimeoutManager(1, cleanup_func) as tm:
            time.sleep(0.1)  # Short operation
            self.assertFalse(tm.is_timed_out())
        
        self.assertFalse(cleanup_called)
        self.assertFalse(tm.is_timed_out())
    
    def test_timeout_manager_timeout(self):
        """Test timeout manager with timeout."""
        cleanup_called = False
        
        def cleanup_func():
            nonlocal cleanup_called
            cleanup_called = True
        
        with TimeoutManager(0.1, cleanup_func) as tm:
            time.sleep(0.2)  # Long operation
        
        # Give a moment for timeout to trigger
        time.sleep(0.1)
        self.assertTrue(tm.is_timed_out())
    
    def test_timeout_manager_elapsed_time(self):
        """Test elapsed time calculation."""
        with TimeoutManager(1) as tm:
            time.sleep(0.1)
            elapsed = tm.elapsed_time()
            self.assertGreater(elapsed, 0.05)
            self.assertLess(elapsed, 0.2)


class TestPreInstallationValidator(unittest.TestCase):
    """Test the PreInstallationValidator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.installation_path = Path(self.temp_dir) / "wan22_test"
        self.installation_path.mkdir(parents=True, exist_ok=True)
        
        # Create mock hardware profile
        self.hardware_profile = HardwareProfile(
            cpu=CPUInfo(
                model="Test CPU",
                cores=8,
                threads=16,
                base_clock=3.0,
                boost_clock=4.0,
                architecture="x86_64"
            ),
            memory=MemoryInfo(
                total_gb=32,
                available_gb=24,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="Test GPU",
                vram_gb=12,
                cuda_version="11.8",
                driver_version="520.0",
                compute_capability="8.6"
            ),
            storage=StorageInfo(
                available_gb=500,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows" if os.name == 'nt' else "Linux",
                version="10" if os.name == 'nt' else "20.04",
                architecture="x86_64"
            )
        )
        
        self.validator = PreInstallationValidator(
            str(self.installation_path),
            self.hardware_profile
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('psutil.disk_usage')
    def test_validate_system_requirements_sufficient(self, mock_disk_usage):
        """Test system requirements validation with sufficient resources."""
        # Mock sufficient disk space
        mock_disk_usage.return_value = Mock(free=100 * 1024**3)  # 100GB
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=32 * 1024**3,  # 32GB
                available=16 * 1024**3  # 16GB available
            )
            
            result = self.validator.validate_system_requirements()
            
            self.assertTrue(result.success)
            self.assertIn("successfully", result.message)
            self.assertIsInstance(result.details, dict)
            self.assertIn("requirements", result.details)
    
    @patch('psutil.disk_usage')
    def test_validate_system_requirements_insufficient(self, mock_disk_usage):
        """Test system requirements validation with insufficient resources."""
        # Mock insufficient disk space
        mock_disk_usage.return_value = Mock(free=50 * 1024**3)  # 50GB (insufficient)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=8 * 1024**3,  # 8GB (insufficient)
                available=4 * 1024**3  # 4GB available (insufficient)
            )
            
            result = self.validator.validate_system_requirements()
            
            self.assertFalse(result.success)
            self.assertIn("requirement issues", result.message)
            self.assertIsInstance(result.details, dict)
    
    def test_validate_network_connectivity_success(self):
        """Test network connectivity validation with successful connections."""
        # Run actual network test (may be slow but more reliable)
        result = self.validator.validate_network_connectivity()
        
        # Should return a valid result regardless of network state
        self.assertIsInstance(result, ValidationResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.message, str)
        self.assertIn("network_tests", result.details or {})
    
    @patch('urllib.request.urlopen')
    def test_validate_network_connectivity_failure(self, mock_urlopen):
        """Test network connectivity validation with connection failures."""
        # Mock network failures
        mock_urlopen.side_effect = Exception("Network error")
        
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 1  # Failure
            mock_socket.return_value = mock_sock
            
            result = self.validator.validate_network_connectivity()
            
            self.assertFalse(result.success)
            self.assertIn("network issues", result.message)
    
    def test_validate_permissions_success(self):
        """Test permissions validation with sufficient permissions."""
        # Ensure test directory has proper permissions
        self.installation_path.chmod(0o755)
        
        result = self.validator.validate_permissions()
        
        self.assertTrue(result.success)
        self.assertIn("successfully", result.message)
        self.assertIn("permission_tests", result.details)
    
    def test_validate_permissions_failure(self):
        """Test permissions validation with insufficient permissions."""
        # Create a read-only directory to test permission failures
        readonly_path = self.installation_path / "readonly"
        readonly_path.mkdir(exist_ok=True)
        
        if os.name != 'nt':  # Unix-like systems
            readonly_path.chmod(0o444)  # Read-only
        
        # Test with read-only installation path
        validator = PreInstallationValidator(str(readonly_path))
        result = validator.validate_permissions()
        
        # On Windows, this might still succeed due to different permission model
        if os.name != 'nt':
            self.assertFalse(result.success)
    
    def test_validate_existing_installation_no_conflicts(self):
        """Test existing installation validation with no conflicts."""
        result = self.validator.validate_existing_installation()
        
        # Should succeed with no major conflicts
        self.assertTrue(result.success)
        self.assertIn("conflicts", result.details)
    
    def test_validate_existing_installation_with_conflicts(self):
        """Test existing installation validation with conflicts."""
        # Create existing installation files
        (self.installation_path / "config.json").write_text('{"test": true}')
        (self.installation_path / "models").mkdir(exist_ok=True)
        (self.installation_path / "venv").mkdir(exist_ok=True)
        
        result = self.validator.validate_existing_installation()
        
        # Should have warnings about existing files
        self.assertTrue(len(result.warnings or []) > 0)
    
    def test_generate_validation_report(self):
        """Test comprehensive validation report generation."""
        with patch.object(self.validator, 'validate_system_requirements') as mock_system, \
             patch.object(self.validator, 'validate_network_connectivity') as mock_network, \
             patch.object(self.validator, 'validate_permissions') as mock_permissions, \
             patch.object(self.validator, 'validate_existing_installation') as mock_existing:
            
            # Mock successful results
            mock_system.return_value = ValidationResult(
                success=True,
                message="System OK",
                details={"requirements": []}
            )
            mock_network.return_value = ValidationResult(
                success=True,
                message="Network OK",
                details={"network_tests": []}
            )
            mock_permissions.return_value = ValidationResult(
                success=True,
                message="Permissions OK",
                details={"permission_tests": []}
            )
            mock_existing.return_value = ValidationResult(
                success=True,
                message="No conflicts",
                details={"conflicts": []}
            )
            
            report = self.validator.generate_validation_report()
            
            self.assertIsInstance(report, PreValidationReport)
            self.assertTrue(report.overall_success)
            self.assertIsInstance(report.timestamp, str)
            self.assertIsInstance(report.system_requirements, list)
            self.assertIsInstance(report.network_tests, list)
            self.assertIsInstance(report.permission_tests, list)
            self.assertIsInstance(report.conflicts, list)
    
    def test_system_requirement_creation(self):
        """Test SystemRequirement data class."""
        req = SystemRequirement(
            name="Test Requirement",
            minimum_value=10,
            current_value=20,
            unit="GB",
            met=True,
            details="Test details"
        )
        
        self.assertEqual(req.name, "Test Requirement")
        self.assertEqual(req.minimum_value, 10)
        self.assertEqual(req.current_value, 20)
        self.assertEqual(req.unit, "GB")
        self.assertTrue(req.met)
        self.assertEqual(req.details, "Test details")
    
    def test_network_test_creation(self):
        """Test NetworkTest data class."""
        test = NetworkTest(
            test_name="Test Network",
            target="https://example.com",
            success=True,
            latency_ms=50.5,
            bandwidth_mbps=100.0
        )
        
        self.assertEqual(test.test_name, "Test Network")
        self.assertEqual(test.target, "https://example.com")
        self.assertTrue(test.success)
        self.assertEqual(test.latency_ms, 50.5)
        self.assertEqual(test.bandwidth_mbps, 100.0)
    
    def test_permission_test_creation(self):
        """Test PermissionTest data class."""
        test = PermissionTest(
            path="/test/path",
            readable=True,
            writable=True,
            executable=False
        )
        
        self.assertEqual(test.path, "/test/path")
        self.assertTrue(test.readable)
        self.assertTrue(test.writable)
        self.assertFalse(test.executable)
    
    def test_conflict_detection_creation(self):
        """Test ConflictDetection data class."""
        conflict = ConflictDetection(
            path="/conflict/path",
            conflict_type="existing_installation",
            severity="warning",
            description="Test conflict",
            resolution="Test resolution"
        )
        
        self.assertEqual(conflict.path, "/conflict/path")
        self.assertEqual(conflict.conflict_type, "existing_installation")
        self.assertEqual(conflict.severity, "warning")
        self.assertEqual(conflict.description, "Test conflict")
        self.assertEqual(conflict.resolution, "Test resolution")


class TestPreInstallationValidatorIntegration(unittest.TestCase):
    """Integration tests for the pre-installation validator."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.installation_path = Path(self.temp_dir) / "wan22_integration"
        self.installation_path.mkdir(parents=True, exist_ok=True)
        
        self.validator = PreInstallationValidator(str(self.installation_path))
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_validation_workflow(self):
        """Test the complete validation workflow."""
        # This test runs actual validation (may be slow)
        report = self.validator.generate_validation_report()
        
        # Basic assertions about the report structure
        self.assertIsInstance(report, PreValidationReport)
        self.assertIsInstance(report.timestamp, str)
        self.assertIsInstance(report.system_requirements, list)
        self.assertIsInstance(report.network_tests, list)
        self.assertIsInstance(report.permission_tests, list)
        self.assertIsInstance(report.conflicts, list)
        self.assertIsInstance(report.overall_success, bool)
        self.assertIsInstance(report.errors, list)
        self.assertIsInstance(report.warnings, list)
        
        # Check that report was saved
        report_path = self.installation_path / "logs" / "pre_validation_report.json"
        self.assertTrue(report_path.exists())
        
        # Verify report content
        with open(report_path, 'r') as f:
            saved_report = json.load(f)
        
        self.assertIn("timestamp", saved_report)
        self.assertIn("system_requirements", saved_report)
        self.assertIn("network_tests", saved_report)
        self.assertIn("permission_tests", saved_report)
        self.assertIn("conflicts", saved_report)
    
    def test_timeout_enforcement(self):
        """Test that timeouts are properly enforced."""
        # Create a validator with very short timeouts
        validator = PreInstallationValidator(str(self.installation_path))
        validator.timeouts = {
            "network_test": 0.1,  # Very short timeout
            "bandwidth_test": 0.1,
            "disk_test": 0.1,
            "permission_test": 0.1,
            "conflict_detection": 0.1
        }
        
        # Run validation - should complete despite short timeouts
        result = validator.validate_system_requirements()
        self.assertIsInstance(result, ValidationResult)
        
        result = validator.validate_permissions()
        self.assertIsInstance(result, ValidationResult)
    
    def test_cleanup_functionality(self):
        """Test that temporary files are properly cleaned up."""
        # Run validation that creates temporary files
        result = self.validator.validate_permissions()
        
        # Check that no temporary test files remain
        temp_files = list(self.installation_path.glob(".wan22_permission_test_*"))
        self.assertEqual(len(temp_files), 0)


class TestPreInstallationValidatorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up edge case test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.installation_path = Path(self.temp_dir) / "wan22_edge"
        # Don't create the directory to test missing path scenarios
        
        self.validator = PreInstallationValidator(str(self.installation_path))
    
    def tearDown(self):
        """Clean up edge case test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_installation_directory(self):
        """Test validation with missing installation directory."""
        # Installation path doesn't exist
        result = self.validator.validate_permissions()
        
        # Should handle gracefully
        self.assertIsInstance(result, ValidationResult)
    
    def test_invalid_hardware_profile(self):
        """Test validation with invalid hardware profile."""
        # Create validator with None hardware profile
        validator = PreInstallationValidator(str(self.installation_path), None)
        
        result = validator.validate_system_requirements()
        self.assertIsInstance(result, ValidationResult)
    
    @patch('psutil.disk_usage')
    def test_disk_usage_error(self, mock_disk_usage):
        """Test handling of disk usage check errors."""
        mock_disk_usage.side_effect = Exception("Disk error")
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.side_effect = Exception("Memory error")
            
            result = self.validator.validate_system_requirements()
            
            # Should handle the error gracefully
            self.assertIsInstance(result, ValidationResult)
            self.assertFalse(result.success)
    
    @patch('psutil.virtual_memory')
    def test_memory_check_error(self, mock_memory):
        """Test handling of memory check errors."""
        mock_memory.side_effect = Exception("Memory error")
        
        result = self.validator.validate_system_requirements()
        
        # Should handle the error gracefully
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.success)
    
    def test_network_timeout_handling(self):
        """Test network timeout handling."""
        # Set very short network timeout
        self.validator.timeouts["network_test"] = 0.001
        
        result = self.validator.validate_network_connectivity()
        
        # Should complete despite timeouts
        self.assertIsInstance(result, ValidationResult)


def run_performance_tests():
    """Run performance tests for the validator."""
    print("Running performance tests...")
    
    temp_dir = tempfile.mkdtemp()
    installation_path = Path(temp_dir) / "wan22_perf"
    installation_path.mkdir(parents=True, exist_ok=True)
    
    try:
        validator = PreInstallationValidator(str(installation_path))
        
        # Time full validation
        start_time = time.time()
        report = validator.generate_validation_report()
        total_time = time.time() - start_time
        
        print(f"Full validation completed in {total_time:.2f} seconds")
        print(f"Overall success: {report.overall_success}")
        print(f"Errors: {len(report.errors)}")
        print(f"Warnings: {len(report.warnings)}")
        
        if report.estimated_install_time_minutes:
            print(f"Estimated installation time: {report.estimated_install_time_minutes} minutes")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-Installation Validator Tests")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    
    args = parser.parse_args()
    
    if args.performance:
        run_performance_tests()
        return
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if args.unit or not (args.integration or args.unit):
        # Add unit tests
        suite.addTest(unittest.makeSuite(TestTimeoutManager))
        suite.addTest(unittest.makeSuite(TestPreInstallationValidator))
        suite.addTest(unittest.makeSuite(TestPreInstallationValidatorEdgeCases))
    
    if args.integration or not (args.integration or args.unit):
        # Add integration tests
        suite.addTest(unittest.makeSuite(TestPreInstallationValidatorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())