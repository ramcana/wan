"""
Unit tests for ProductionValidator
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from local_testing_framework.production_validator import (
    ProductionValidator, ProductionReadinessResults, ConsistencyValidationResults,
    ConfigurationValidationResults, SecurityValidationResults, ScalabilityValidationResults,
    ProductionCertificate
)
from local_testing_framework.models.test_results import (
    ValidationResult, ValidationStatus, TestStatus, PerformanceTestResults,
    BenchmarkResult, OptimizationResult, ResourceMetrics
)


class TestProductionValidator(unittest.TestCase):
    """Test cases for ProductionValidator"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.env_path = os.path.join(self.temp_dir, ".env")
        
        # Create test config
        self.test_config = {
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
                "enable_https": True,
                "enable_auth": True,
                "auth_method": "token",
                "auth_token": "test_token",
                "ssl": {
                    "cert_file": "cert.pem",
                    "key_file": "key.pem"
                },
                "cors_origins": ["https://example.com"]
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Create test .env file
        with open(self.env_path, 'w') as f:
            f.write("HF_TOKEN=test_token\n")
        
        # Set restrictive permissions
        os.chmod(self.env_path, 0o600)
        
        # Change to temp directory for tests
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test ProductionValidator initialization"""
        validator = ProductionValidator(self.config_path)
        
        self.assertEqual(validator.config_path, self.config_path)
        self.assertEqual(validator.config, self.test_config)
        self.assertEqual(validator.consistency_runs, 3)
        self.assertEqual(validator.load_test_concurrent_users, 5)

    @patch('local_testing_framework.production_validator.PerformanceTester')
    def test_validate_performance_consistency_success(self, mock_perf_tester_class):
        """Test successful performance consistency validation"""
        # Mock performance tester
        mock_perf_tester = Mock()
        mock_perf_tester_class.return_value = mock_perf_tester
        
        # Create mock benchmark results
        mock_benchmark_720p = BenchmarkResult(
            resolution="720p",
            generation_time=8.0,
            target_time=9.0,
            meets_target=True,
            vram_usage=10.0,
            cpu_usage=60.0,
            memory_usage=50.0,
            optimization_level="high"
        )
        
        mock_benchmark_1080p = BenchmarkResult(
            resolution="1080p",
            generation_time=15.0,
            target_time=17.0,
            meets_target=True,
            vram_usage=11.0,
            cpu_usage=65.0,
            memory_usage=55.0,
            optimization_level="high"
        )
        
        mock_vram_opt = OptimizationResult(
            baseline_vram_mb=20000,
            optimized_vram_mb=4000,
            reduction_percent=80.0,
            target_reduction_percent=80.0,
            meets_target=True,
            optimizations_applied=["attention_slicing", "vae_tiling"]
        )
        
        mock_perf_results = PerformanceTestResults(
            benchmark_720p=mock_benchmark_720p,
            benchmark_1080p=mock_benchmark_1080p,
            vram_optimization=mock_vram_opt,
            overall_status=TestStatus.PASSED
        )
        
        mock_perf_tester.run_performance_tests.return_value = mock_perf_results
        
        validator = ProductionValidator(self.config_path)
        
        with patch('time.sleep'):  # Skip delays in tests
            result = validator._validate_performance_consistency()
        
        self.assertEqual(result.runs_completed, 3)
        self.assertEqual(result.target_runs, 3)
        self.assertEqual(len(result.benchmark_results), 3)
        self.assertEqual(result.overall_status, TestStatus.PASSED)
        self.assertTrue(result.consistency_metrics["all_targets_met"])

    def test_validate_config_file_success(self):
        """Test successful config file validation"""
        validator = ProductionValidator(self.config_path)
        result = validator._validate_config_file()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "config.json")

    def test_validate_config_file_missing_sections(self):
        """Test config file validation with missing sections"""
        # Create config with missing sections
        incomplete_config = {"system": {"gpu_enabled": True}}
        
        with open(self.config_path, 'w') as f:
            json.dump(incomplete_config, f)
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_config_file()
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("Missing required sections", result.message)

    def test_validate_env_file_success(self):
        """Test successful .env file validation"""
        validator = ProductionValidator(self.config_path)
        result = validator._validate_env_file()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, ".env")

    def test_validate_env_file_missing(self):
        """Test .env file validation when file is missing"""
        os.remove(self.env_path)
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_env_file()
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("not found", result.message)

    def test_validate_env_file_world_readable(self):
        """Test .env file validation with insecure permissions"""
        # Make file world-readable
        os.chmod(self.env_path, 0o644)
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_env_file()
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("world-readable", result.message)

    def test_validate_https_configuration_enabled(self):
        """Test HTTPS configuration validation when enabled"""
        # Create dummy certificate files
        cert_path = Path("cert.pem")
        key_path = Path("key.pem")
        cert_path.touch()
        key_path.touch()
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_https_configuration()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "https_config")

    def test_validate_https_configuration_disabled(self):
        """Test HTTPS configuration validation when disabled"""
        self.test_config["security"]["enable_https"] = False
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_https_configuration()
        
        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertIn("not enabled", result.message)

    def test_validate_authentication_enabled(self):
        """Test authentication validation when enabled"""
        validator = ProductionValidator(self.config_path)
        result = validator._validate_authentication()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "authentication")

    def test_validate_authentication_disabled(self):
        """Test authentication validation when disabled"""
        self.test_config["security"]["enable_auth"] = False
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_authentication()
        
        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertIn("not enabled", result.message)

    def test_validate_file_permissions_secure(self):
        """Test file permissions validation with secure permissions"""
        validator = ProductionValidator(self.config_path)
        result = validator._validate_file_permissions()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "file_permissions")

    def test_validate_file_permissions_insecure(self):
        """Test file permissions validation with insecure permissions"""
        # Make .env world-readable
        os.chmod(self.env_path, 0o644)
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_file_permissions()
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("security issues", result.message)

    @patch('subprocess.run')
    def test_validate_ssl_certificates_valid(self, mock_subprocess):
        """Test SSL certificate validation with valid certificate"""
        # Create dummy certificate file
        cert_path = Path("cert.pem")
        cert_path.touch()
        
        # Mock openssl command success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "notBefore=Jan 1 00:00:00 2024 GMT\nnotAfter=Jan 1 00:00:00 2025 GMT"
        mock_subprocess.return_value = mock_result
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_ssl_certificates()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "ssl_certificates")

    @patch('subprocess.run')
    def test_validate_ssl_certificates_invalid(self, mock_subprocess):
        """Test SSL certificate validation with invalid certificate"""
        # Create dummy certificate file
        cert_path = Path("cert.pem")
        cert_path.touch()
        
        # Mock openssl command failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_ssl_certificates()
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("Invalid SSL certificate", result.message)

    @patch('local_testing_framework.production_validator.psutil')
    def test_collect_resource_metrics(self, mock_psutil):
        """Test resource metrics collection"""
        # Mock psutil
        mock_cpu = Mock()
        mock_cpu.cpu_percent.return_value = 50.0
        
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.used = 8 * (1024**3)  # 8GB
        mock_memory.total = 16 * (1024**3)  # 16GB
        
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        validator = ProductionValidator(self.config_path)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=4 * (1024**3)), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_device = Mock()
            mock_device.total_memory = 12 * (1024**3)  # 12GB
            mock_props.return_value = mock_device
            
            metrics = validator._collect_resource_metrics()
        
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.memory_used_gb, 8.0)
        self.assertEqual(metrics.memory_total_gb, 16.0)

    def test_run_concurrent_load_test(self):
        """Test concurrent load test execution"""
        validator = ProductionValidator(self.config_path)
        validator.load_test_concurrent_users = 2
        validator.load_test_duration_minutes = 0.1  # 6 seconds for test
        
        with patch('time.sleep'):  # Speed up test
            result = validator._run_concurrent_load_test()
        
        self.assertIn("concurrent_requests", result)
        self.assertIn("successful_requests", result)
        self.assertIn("failed_requests", result)
        self.assertIn("success_rate", result)
        self.assertEqual(result["concurrent_requests"], 2)

    def test_validate_queue_management_success(self):
        """Test queue management validation with good performance"""
        load_test_results = {
            "success_rate": 0.98,
            "failed_requests": 2,
            "average_response_time": 5.0
        }
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_queue_management(load_test_results)
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "queue_management")

    def test_validate_queue_management_low_success_rate(self):
        """Test queue management validation with low success rate"""
        load_test_results = {
            "success_rate": 0.90,
            "failed_requests": 10,
            "average_response_time": 5.0
        }
        
        validator = ProductionValidator(self.config_path)
        result = validator._validate_queue_management(load_test_results)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("Low success rate", result.message)

    def test_generate_production_certificate(self):
        """Test production certificate generation"""
        validator = ProductionValidator(self.config_path)
        
        # Create mock results
        consistency_results = ConsistencyValidationResults(
            runs_completed=3,
            target_runs=3,
            benchmark_results=[],
            consistency_metrics={"consistency_score": 95.0, "all_targets_met": True},
            overall_status=TestStatus.PASSED
        )
        
        config_results = ConfigurationValidationResults(
            config_file_validation=ValidationResult("config", ValidationStatus.PASSED, "OK"),
            env_file_validation=ValidationResult("env", ValidationStatus.PASSED, "OK"),
            security_config_validation=ValidationResult("security", ValidationStatus.PASSED, "OK"),
            performance_config_validation=ValidationResult("performance", ValidationStatus.PASSED, "OK"),
            overall_status=TestStatus.PASSED
        )
        
        security_results = SecurityValidationResults(
            https_validation=ValidationResult("https", ValidationStatus.PASSED, "OK"),
            auth_validation=ValidationResult("auth", ValidationStatus.PASSED, "OK"),
            file_permissions_validation=ValidationResult("permissions", ValidationStatus.PASSED, "OK"),
            certificate_validation=None,
            overall_status=TestStatus.PASSED
        )
        
        scalability_results = ScalabilityValidationResults(
            concurrent_users_tested=5,
            load_test_duration_minutes=10,
            performance_under_load={},
            resource_usage_under_load=[],
            queue_management_validation=ValidationResult("queue", ValidationStatus.PASSED, "OK"),
            overall_status=TestStatus.PASSED
        )
        
        results = ProductionReadinessResults(
            start_time=datetime.now(),
            consistency_validation=consistency_results,
            configuration_validation=config_results,
            security_validation=security_results,
            scalability_validation=scalability_results,
            overall_status=TestStatus.PASSED
        )
        
        certificate = validator._generate_production_certificate(results)
        
        self.assertIsInstance(certificate, ProductionCertificate)
        self.assertTrue(certificate.security_compliance)
        self.assertTrue(certificate.scalability_validated)
        self.assertEqual(len(certificate.certificate_id), 16)
        
        # Check certificate file was created
        cert_file = Path("production_readiness_certificate.json")
        self.assertTrue(cert_file.exists())

    @patch('local_testing_framework.production_validator.PerformanceTester')
    @patch('local_testing_framework.production_validator.EnvironmentValidator')
    @patch('local_testing_framework.production_validator.IntegrationTester')
    def test_validate_production_readiness_full(self, mock_integration, mock_env, mock_perf):
        """Test full production readiness validation"""
        # Create dummy certificate files
        cert_path = Path("cert.pem")
        key_path = Path("key.pem")
        cert_path.touch()
        key_path.touch()
        
        validator = ProductionValidator(self.config_path)
        validator.consistency_runs = 1  # Reduce for test speed
        validator.load_test_duration_minutes = 0.05  # 3 seconds
        
        # Mock performance results
        mock_benchmark = BenchmarkResult(
            resolution="720p",
            generation_time=8.0,
            target_time=9.0,
            meets_target=True,
            vram_usage=10.0,
            cpu_usage=60.0,
            memory_usage=50.0,
            optimization_level="high"
        )
        
        mock_perf_results = PerformanceTestResults(
            benchmark_720p=mock_benchmark,
            benchmark_1080p=mock_benchmark,
            vram_optimization=OptimizationResult(
                baseline_vram_mb=20000,
                optimized_vram_mb=4000,
                reduction_percent=80.0,
                target_reduction_percent=80.0,
                meets_target=True,
                optimizations_applied=[]
            ),
            overall_status=TestStatus.PASSED
        )
        
        mock_perf.return_value.run_performance_tests.return_value = mock_perf_results
        
        with patch('time.sleep'), \
             patch('subprocess.run') as mock_subprocess, \
             patch('local_testing_framework.production_validator.psutil') as mock_psutil:
            
            # Mock SSL validation
            mock_ssl_result = Mock()
            mock_ssl_result.returncode = 0
            mock_ssl_result.stdout = "notAfter=Jan 1 00:00:00 2025 GMT"
            mock_subprocess.return_value = mock_ssl_result
            
            # Mock psutil
            mock_memory = Mock()
            mock_memory.percent = 60.0
            mock_memory.used = 8 * (1024**3)
            mock_memory.total = 16 * (1024**3)
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value = mock_memory
            
            result = validator.validate_production_readiness()
        
        self.assertIsInstance(result, ProductionReadinessResults)
        self.assertIsNotNone(result.consistency_validation)
        self.assertIsNotNone(result.configuration_validation)
        self.assertIsNotNone(result.security_validation)
        self.assertIsNotNone(result.scalability_validation)
        
        # Should generate certificate if all validations pass
        if result.overall_status == TestStatus.PASSED:
            self.assertIsNotNone(result.certificate)


if __name__ == '__main__':
    unittest.main()