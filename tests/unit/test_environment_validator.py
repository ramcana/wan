"""
Tests for the test environment validator
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from tests.config.environment_validator import (
    EnvironmentValidator, ValidationStatus, ValidationLevel,
    EnvironmentRequirement, validate_test_environment
)


class TestEnvironmentValidator:
    """Test cases for EnvironmentValidator class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = EnvironmentValidator()
    
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_loads_builtin_requirements(self):
        """Test that initialization loads built-in requirements"""
        assert len(self.validator.requirements) > 0
        
        # Check for some expected built-in requirements
        requirement_names = [req.name for req in self.validator.requirements]
        assert "python_version" in requirement_names
        assert "pytest" in requirement_names
        assert "memory_check" in requirement_names
    
    def test_validate_python_package_success(self):
        """Test successful Python package validation"""
        # Test with a package that should exist (sys is built-in)
        requirement = EnvironmentRequirement(
            name="sys",
            type="python_package",
            required=True
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
        assert result.level == ValidationLevel.INFO
    
    def test_validate_python_package_not_found(self):
        """Test Python package validation when package not found"""
        requirement = EnvironmentRequirement(
            name="nonexistent_package_12345",
            type="python_package",
            required=True
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.FAILED
        assert result.level == ValidationLevel.CRITICAL
        assert "not installed" in result.message
        assert len(result.suggestions) > 0
    
    def test_validate_python_version(self):
        """Test Python version validation"""
        requirement = EnvironmentRequirement(
            name="python_version",
            type="python_version",
            required=True,
            version_check=">=3.6"  # Should pass for most systems
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
        assert "version" in result.details
    
    def test_validate_directory_exists(self):
        """Test directory validation when directory exists"""
        test_dir = self.temp_dir / "test_directory"
        test_dir.mkdir()
        
        requirement = EnvironmentRequirement(
            name="test_dir",
            type="directory",
            required=True,
            config={"path": str(test_dir)}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
        assert result.level == ValidationLevel.INFO
    
    def test_validate_directory_not_exists_required(self):
        """Test directory validation when required directory doesn't exist"""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        requirement = EnvironmentRequirement(
            name="missing_dir",
            type="directory",
            required=True,
            config={"path": str(nonexistent_dir)}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.FAILED
        assert result.level == ValidationLevel.CRITICAL
        assert len(result.suggestions) > 0
    
    def test_validate_directory_not_exists_optional(self):
        """Test directory validation when optional directory doesn't exist"""
        nonexistent_dir = self.temp_dir / "optional_dir"
        
        requirement = EnvironmentRequirement(
            name="optional_dir",
            type="directory",
            required=False,
            config={"path": str(nonexistent_dir)}
        )
        
        result = self.validator._validate_requirement(requirement)
        # Should auto-create optional directories
        assert result.status == ValidationStatus.PASSED
        assert nonexistent_dir.exists()
    
    def test_validate_file_exists(self):
        """Test file validation when file exists"""
        test_file = self.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        requirement = EnvironmentRequirement(
            name="test_file",
            type="file",
            required=True,
            config={"path": str(test_file)}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
        assert "size" in result.details
    
    def test_validate_file_not_exists(self):
        """Test file validation when file doesn't exist"""
        nonexistent_file = self.temp_dir / "nonexistent.txt"
        
        requirement = EnvironmentRequirement(
            name="missing_file",
            type="file",
            required=True,
            config={"path": str(nonexistent_file)}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.FAILED
        assert result.level == ValidationLevel.CRITICAL
    
    def test_validate_environment_variable_set(self):
        """Test environment variable validation when variable is set"""
        with patch.dict('os.environ', {'TEST_VAR': 'test_value'}):
            requirement = EnvironmentRequirement(
                name="test_var",
                type="env_var",
                required=True,
                config={"name": "TEST_VAR"}
            )
            
            result = self.validator._validate_requirement(requirement)
            assert result.status == ValidationStatus.PASSED
            assert result.details["value"] == "test_value"
    
    def test_validate_environment_variable_not_set(self):
        """Test environment variable validation when variable is not set"""
        requirement = EnvironmentRequirement(
            name="missing_var",
            type="env_var",
            required=True,
            config={"name": "NONEXISTENT_VAR_12345"}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.FAILED
        assert result.level == ValidationLevel.CRITICAL
    
    @patch('subprocess.run')
    def test_validate_system_command_success(self, mock_run):
        """Test system command validation when command exists"""
        mock_run.return_value = Mock(returncode=0, stdout="version 1.0", stderr="")
        
        requirement = EnvironmentRequirement(
            name="test_command",
            type="system_command",
            required=True
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_validate_system_command_not_found(self, mock_run):
        """Test system command validation when command not found"""
        mock_run.side_effect = FileNotFoundError()
        
        requirement = EnvironmentRequirement(
            name="nonexistent_command",
            type="system_command",
            required=True
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.FAILED
        assert result.level == ValidationLevel.CRITICAL
    
    @patch('psutil.virtual_memory')
    def test_validate_memory_sufficient(self, mock_memory):
        """Test memory validation when sufficient memory available"""
        # Mock 4GB available memory
        mock_memory.return_value = Mock(available=4 * 1024**3)
        
        requirement = EnvironmentRequirement(
            name="memory_check",
            type="system_resource",
            required=True,
            config={"min_memory_gb": 2}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
    
    @patch('psutil.virtual_memory')
    def test_validate_memory_insufficient(self, mock_memory):
        """Test memory validation when insufficient memory available"""
        # Mock 1GB available memory
        mock_memory.return_value = Mock(available=1 * 1024**3)
        
        requirement = EnvironmentRequirement(
            name="memory_check",
            type="system_resource",
            required=True,
            config={"min_memory_gb": 2}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.FAILED
        assert result.level == ValidationLevel.WARNING
    
    @patch('psutil.disk_usage')
    def test_validate_disk_space_sufficient(self, mock_disk):
        """Test disk space validation when sufficient space available"""
        # Mock 10GB free space
        mock_disk.return_value = Mock(free=10 * 1024**3)
        
        requirement = EnvironmentRequirement(
            name="disk_space_check",
            type="system_resource",
            required=True,
            config={"min_disk_gb": 5}
        )
        
        result = self.validator._validate_requirement(requirement)
        assert result.status == ValidationStatus.PASSED
    
    def test_validate_environment_full(self):
        """Test full environment validation"""
        results = self.validator.validate_environment()
        
        assert len(results) > 0
        assert all(hasattr(result, 'name') for result in results)
        assert all(hasattr(result, 'status') for result in results)
        assert all(hasattr(result, 'level') for result in results)
    
    def test_get_validation_summary(self):
        """Test validation summary generation"""
        # Run validation first
        self.validator.validate_environment()
        
        summary = self.validator.get_validation_summary()
        
        assert "status" in summary
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "skipped" in summary
        assert "critical_failures" in summary
        assert "ready_for_testing" in summary
        
        assert summary["total_checks"] == len(self.validator.validation_results)
    
    def test_generate_text_report(self):
        """Test text report generation"""
        # Run validation first
        self.validator.validate_environment()
        
        report = self.validator.generate_report("text")
        
        assert isinstance(report, str)
        assert "Test Environment Validation Report" in report
        assert "Overall Status:" in report
        assert "Total Checks:" in report
    
    def test_generate_json_report(self):
        """Test JSON report generation"""
        # Run validation first
        self.validator.validate_environment()
        
        report = self.validator.generate_report("json")
        
        assert isinstance(report, str)
        
        # Verify it's valid JSON
        report_data = json.loads(report)
        assert "summary" in report_data
        assert "results" in report_data
    
    def test_version_check_methods(self):
        """Test version checking methods"""
        # Test various version check formats
        assert self.validator._check_version("1.2.3", ">=1.0.0")
        assert self.validator._check_version("2.0.0", ">1.9.0")
        assert not self.validator._check_version("1.0.0", ">1.0.0")
        assert self.validator._check_version("1.0.0", "<=1.0.0")
    
    def test_convenience_function(self):
        """Test convenience validation function"""
        validator = validate_test_environment()
        
        assert isinstance(validator, EnvironmentValidator)
        assert len(validator.validation_results) > 0
        
        summary = validator.get_validation_summary()
        assert summary["status"] in ["passed", "failed"]


if __name__ == "__main__":
    pytest.main([__file__])