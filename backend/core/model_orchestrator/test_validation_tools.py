"""
Tests for validation tools.

This module contains comprehensive tests for manifest validation,
security checks, performance analysis, and compatibility validation.
"""

import hashlib
import tempfile
import shutil
from pathlib import Path
import pytest

from .validation_tools import (
    ValidationIssue,
    ValidationReport,
    ManifestSchemaValidator,
    SecurityValidator,
    PerformanceValidator,
    CompatibilityValidator,
    ComprehensiveValidator
)


class TestValidationIssue:
    """Test ValidationIssue functionality."""
    
    def test_validation_issue_creation(self):
        """Test creating validation issues."""
        issue = ValidationIssue(
            severity="error",
            category="schema",
            message="Test error message",
            context="test_context",
            suggestion="Test suggestion"
        )
        
        assert issue.severity == "error"
        assert issue.category == "schema"
        assert issue.message == "Test error message"
        assert issue.context == "test_context"
        assert issue.suggestion == "Test suggestion"
    
    def test_validation_issue_string_representation(self):
        """Test string representation of validation issues."""
        issue = ValidationIssue(
            severity="warning",
            category="security",
            message="Test warning",
            context="test_context",
            suggestion="Fix this"
        )
        
        str_repr = str(issue)
        assert "[WARNING]" in str_repr
        assert "(test_context)" in str_repr
        assert "Test warning" in str_repr
        assert "Suggestion: Fix this" in str_repr


class TestValidationReport:
    """Test ValidationReport functionality."""
    
    def test_empty_validation_report(self):
        """Test empty validation report."""
        report = ValidationReport(valid=True, issues=[], summary={})
        
        assert report.valid is True
        assert len(report.issues) == 0
        assert report.summary["total"] == 0
        assert report.summary["errors"] == 0
        assert report.summary["warnings"] == 0
        assert report.summary["info"] == 0
    
    def test_validation_report_with_issues(self):
        """Test validation report with various issues."""
        issues = [
            ValidationIssue("error", "schema", "Error message"),
            ValidationIssue("warning", "security", "Warning message"),
            ValidationIssue("info", "performance", "Info message")
        ]
        
        report = ValidationReport(valid=True, issues=issues, summary={})
        
        assert report.valid is False  # Should be False due to error
        assert len(report.issues) == 3
        assert report.summary["total"] == 3
        assert report.summary["errors"] == 1
        assert report.summary["warnings"] == 1
        assert report.summary["info"] == 1
    
    def test_add_issue_to_report(self):
        """Test adding issues to report."""
        report = ValidationReport(valid=True, issues=[], summary={})
        
        issue = ValidationIssue("error", "schema", "New error")
        report.add_issue(issue)
        
        assert len(report.issues) == 1
        assert report.valid is False
        assert report.summary["errors"] == 1
    
    def test_filter_issues_by_severity(self):
        """Test filtering issues by severity."""
        issues = [
            ValidationIssue("error", "schema", "Error 1"),
            ValidationIssue("error", "schema", "Error 2"),
            ValidationIssue("warning", "security", "Warning 1")
        ]
        
        report = ValidationReport(valid=True, issues=issues, summary={})
        
        errors = report.get_issues_by_severity("error")
        warnings = report.get_issues_by_severity("warning")
        
        assert len(errors) == 2
        assert len(warnings) == 1
    
    def test_filter_issues_by_category(self):
        """Test filtering issues by category."""
        issues = [
            ValidationIssue("error", "schema", "Schema error"),
            ValidationIssue("warning", "schema", "Schema warning"),
            ValidationIssue("error", "security", "Security error")
        ]
        
        report = ValidationReport(valid=True, issues=issues, summary={})
        
        schema_issues = report.get_issues_by_category("schema")
        security_issues = report.get_issues_by_category("security")
        
        assert len(schema_issues) == 2
        assert len(security_issues) == 1


class TestManifestSchemaValidator:
    """Test ManifestSchemaValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ManifestSchemaValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_valid_manifest(self) -> str:
        """Create a valid test manifest."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test Model"
version = "1.0.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"
resolution_caps = ["720p24"]
optional_components = []
lora_required = false
allow_patterns = ["*.safetensors", "*.json"]

[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test-model@1.0.0"]
'''
        
        manifest_path = Path(self.temp_dir) / "valid_manifest.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def create_invalid_manifest(self) -> str:
        """Create an invalid test manifest."""
        manifest_content = '''
# Missing schema_version

[models."invalid-model-id"]
# Missing required fields
description = "Invalid Model"
'''
        
        manifest_path = Path(self.temp_dir) / "invalid_manifest.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def test_validate_valid_manifest(self):
        """Test validation of valid manifest."""
        manifest_path = self.create_valid_manifest()
        
        report = self.validator.validate_manifest_schema(manifest_path)
        
        assert report.valid is True
        assert report.summary["errors"] == 0
    
    def test_validate_invalid_manifest(self):
        """Test validation of invalid manifest."""
        manifest_path = self.create_invalid_manifest()
        
        report = self.validator.validate_manifest_schema(manifest_path)
        
        assert report.valid is False
        assert report.summary["errors"] > 0
        
        # Check for specific errors
        error_messages = [issue.message for issue in report.get_issues_by_severity("error")]
        assert any("schema_version" in msg for msg in error_messages)
    
    def test_validate_missing_required_fields(self):
        """Test validation of manifest with missing required fields."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test Model"
# Missing version, variants, default_variant, files, sources
'''
        
        manifest_path = Path(self.temp_dir) / "missing_fields.toml"
        manifest_path.write_text(manifest_content)
        
        report = self.validator.validate_manifest_schema(str(manifest_path))
        
        assert report.valid is False
        
        # Should have errors for missing required fields
        error_messages = [issue.message for issue in report.get_issues_by_severity("error")]
        required_fields = ["version", "variants", "default_variant", "files", "sources"]
        
        for field in required_fields:
            assert any(field in msg for msg in error_messages)
    
    def test_validate_invalid_model_id_format(self):
        """Test validation of invalid model ID format."""
        manifest_content = '''
schema_version = 1

[models."invalid-model-id-format"]
description = "Test Model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"
files = []
sources = {priority = []}
'''
        
        manifest_path = Path(self.temp_dir) / "invalid_id.toml"
        manifest_path.write_text(manifest_content)
        
        report = self.validator.validate_manifest_schema(str(manifest_path))
        
        assert report.valid is False
        
        # Should have error for invalid model ID format
        error_messages = [issue.message for issue in report.get_issues_by_severity("error")]
        assert any("Invalid model ID format" in msg for msg in error_messages)
    
    def test_validate_file_security_issues(self):
        """Test validation of file security issues."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test Model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "../../../etc/passwd"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."test-model@1.0.0".sources]
priority = ["local://test"]
'''
        
        manifest_path = Path(self.temp_dir) / "security_issues.toml"
        manifest_path.write_text(manifest_content)
        
        report = self.validator.validate_manifest_schema(str(manifest_path))
        
        assert report.valid is False
        
        # Should have security error for path traversal
        security_issues = report.get_issues_by_category("security")
        assert len(security_issues) > 0
        assert any("Path traversal detected" in issue.message for issue in security_issues)


class TestSecurityValidator:
    """Test SecurityValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = SecurityValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_manifest_with_security_issues(self) -> str:
        """Create manifest with various security issues."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Test Model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."test-model@1.0.0".files]]
path = "malware.exe"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[models."test-model@1.0.0".files]]
path = "CON.txt"
size = 512
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

[[models."test-model@1.0.0".files]]
path = ".hidden_file"
size = 256
sha256 = "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78"

[models."test-model@1.0.0".sources]
priority = [
    "http://insecure-site.com/model",
    "https://bit.ly/suspicious-link"
]
'''
        
        manifest_path = Path(self.temp_dir) / "security_manifest.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def test_validate_security_issues(self):
        """Test detection of various security issues."""
        manifest_path = self.create_manifest_with_security_issues()
        
        report = self.validator.validate_security(manifest_path)
        
        # Should detect multiple security issues
        assert len(report.issues) > 0
        
        issue_messages = [issue.message for issue in report.issues]
        
        # Check for executable file detection
        assert any("Executable file detected" in msg for msg in issue_messages)
        
        # Check for Windows reserved name detection
        assert any("Windows reserved name" in msg for msg in issue_messages)
        
        # Check for hidden file detection
        assert any("Hidden file detected" in msg for msg in issue_messages)
        
        # Check for insecure HTTP URL detection
        assert any("Insecure HTTP URL" in msg for msg in issue_messages)
        
        # Check for suspicious URL detection
        assert any("suspicious URL" in msg for msg in issue_messages)


class TestPerformanceValidator:
    """Test PerformanceValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = PerformanceValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_large_model_manifest(self) -> str:
        """Create manifest with performance issues."""
        manifest_content = '''
schema_version = 1

[models."large-model@1.0.0"]
description = "Large Test Model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

# Large file (60GB)
[[models."large-model@1.0.0".files]]
path = "huge_model.safetensors"
size = 64424509440
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."large-model@1.0.0".vram_estimation]
params_billion = 70.0
family_size = "huge"
base_vram_gb = 32.0
per_frame_vram_mb = 512.0

[models."large-model@1.0.0".sources]
priority = ["local://large-model@1.0.0"]
'''
        
        # Add many small files
        for i in range(150):
            manifest_content += f'''
[[models."large-model@1.0.0".files]]
path = "small_file_{i:03d}.json"
size = 512
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
'''
        
        manifest_path = Path(self.temp_dir) / "large_manifest.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def test_validate_performance_issues(self):
        """Test detection of performance issues."""
        manifest_path = self.create_large_model_manifest()
        
        report = self.validator.validate_performance(manifest_path)
        
        # Should detect performance issues
        assert len(report.issues) > 0
        
        issue_messages = [issue.message for issue in report.issues]
        
        # Check for large model size detection
        assert any("Large model size" in msg or "Very large model size" in msg for msg in issue_messages)
        
        # Check for many small files detection
        assert any("Many small files detected" in msg for msg in issue_messages)
        
        # Check for high VRAM requirement detection
        assert any("High VRAM requirement" in msg for msg in issue_messages)


class TestCompatibilityValidator:
    """Test CompatibilityValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = CompatibilityValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_compatibility_issues_manifest(self) -> str:
        """Create manifest with compatibility issues."""
        manifest_content = '''
schema_version = 1

[models."compat-model@1.0.0"]
description = "Compatibility Test Model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

# Case collision files
[[models."compat-model@1.0.0".files]]
path = "Model.safetensors"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[models."compat-model@1.0.0".files]]
path = "model.safetensors"
size = 1024
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

# Long path
[[models."compat-model@1.0.0".files]]
path = "very/long/path/with/many/nested/directories/and/a/very/long/filename/that/exceeds/windows/path/limits/model_with_extremely_long_name_that_causes_issues.safetensors"
size = 2048
sha256 = "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78"

# Invalid characters for Windows
[[models."compat-model@1.0.0".files]]
path = "invalid<>chars.txt"
size = 512
sha256 = "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2"

[models."compat-model@1.0.0".sources]
priority = ["local://compat-model@1.0.0"]
'''
        
        manifest_path = Path(self.temp_dir) / "compat_manifest.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def test_validate_compatibility_issues(self):
        """Test detection of compatibility issues."""
        manifest_path = self.create_compatibility_issues_manifest()
        
        report = self.validator.validate_compatibility(manifest_path)
        
        # Should detect compatibility issues
        assert len(report.issues) > 0
        
        issue_messages = [issue.message for issue in report.issues]
        
        # Check for case collision detection
        assert any("Case collision detected" in msg for msg in issue_messages)
        
        # Check for long path detection
        assert any("Path may exceed Windows limit" in msg for msg in issue_messages)
        
        # Check for invalid character detection
        assert any("Invalid character" in msg for msg in issue_messages)


class TestComprehensiveValidator:
    """Test ComprehensiveValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ComprehensiveValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_comprehensive_test_manifest(self) -> str:
        """Create manifest with various types of issues."""
        manifest_content = '''
schema_version = 1

[models."test-model@1.0.0"]
description = "Comprehensive Test Model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

# Valid file
[[models."test-model@1.0.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Security issue: executable file
[[models."test-model@1.0.0".files]]
path = "script.exe"
size = 2048
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

# Performance issue: large file
[[models."test-model@1.0.0".files]]
path = "large_model.safetensors"
size = 53687091200
sha256 = "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78"

# Compatibility issue: invalid characters
[[models."test-model@1.0.0".files]]
path = "invalid|name.txt"
size = 512
sha256 = "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2"

[models."test-model@1.0.0".vram_estimation]
params_billion = 50.0
base_vram_gb = 28.0
per_frame_vram_mb = 400.0

[models."test-model@1.0.0".sources]
priority = [
    "http://insecure-site.com/model",
    "local://test-model@1.0.0"
]
'''
        
        manifest_path = Path(self.temp_dir) / "comprehensive_manifest.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation with all validators."""
        manifest_path = self.create_comprehensive_test_manifest()
        
        report = self.validator.validate_manifest(manifest_path)
        
        # Should detect issues from all categories
        assert len(report.issues) > 0
        
        # Check that we have issues from different categories
        categories = set(issue.category for issue in report.issues)
        expected_categories = {"security", "performance", "compatibility"}
        
        # Should have at least some of the expected categories
        assert len(categories.intersection(expected_categories)) > 0
    
    def test_selective_validation(self):
        """Test running only specific validation types."""
        manifest_path = self.create_comprehensive_test_manifest()
        
        # Run only schema validation
        schema_report = self.validator.validate_manifest(
            manifest_path,
            include_schema=True,
            include_security=False,
            include_performance=False,
            include_compatibility=False
        )
        
        # Should have fewer issues than comprehensive validation
        comprehensive_report = self.validator.validate_manifest(manifest_path)
        assert len(schema_report.issues) <= len(comprehensive_report.issues)
    
    def test_file_integrity_validation(self):
        """Test file integrity validation."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        # Calculate expected values
        expected_size = len(test_content.encode())
        expected_sha256 = hashlib.sha256(test_content.encode()).hexdigest()
        
        # Test with correct values
        report = self.validator.validate_file_integrity(
            str(test_file), expected_sha256, expected_size
        )
        
        assert report.valid is True
        assert len(report.issues) == 0
        
        # Test with incorrect values
        wrong_report = self.validator.validate_file_integrity(
            str(test_file), "wrong_hash", expected_size + 1
        )
        
        assert wrong_report.valid is False
        assert len(wrong_report.issues) > 0
        
        # Should have both size and hash mismatch errors
        error_messages = [issue.message for issue in wrong_report.issues]
        assert any("Size mismatch" in msg for msg in error_messages)
        assert any("SHA256 mismatch" in msg for msg in error_messages)
    
    def test_nonexistent_file_integrity(self):
        """Test integrity validation of nonexistent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        
        report = self.validator.validate_file_integrity(
            str(nonexistent_file), "dummy_hash", 1024
        )
        
        assert report.valid is False
        assert len(report.issues) == 1
        assert "File not found" in report.issues[0].message


if __name__ == "__main__":
    pytest.main([__file__])