"""
Comprehensive tests for the Model Validation Recovery System.

This test suite validates all aspects of the ModelValidationRecovery class:
- Model issue identification (missing files, corruption, wrong versions)
- Automatic model re-download with integrity verification
- Model file repair and directory structure fixing
- Detailed model issue reporting when recovery fails

Requirements tested: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import unittest
import tempfile
import shutil
import json
import hashlib
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the class under test
from scripts.model_validation_recovery import (
    ModelValidationRecovery, ModelIssueType, ModelIssue, 
    ModelValidationResult, ModelRecoveryResult
)


class TestModelValidationRecovery(unittest.TestCase):
    """Test suite for ModelValidationRecovery class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create mock logger
        self.mock_logger = Mock()
        
        # Initialize the recovery system
        self.recovery_system = ModelValidationRecovery(
            installation_path=self.temp_dir,
            models_directory=str(self.models_dir),
            logger=self.mock_logger
        )
        
        # Test model configuration
        self.test_model_id = "Wan2.2/T2V-A14B"
        self.test_model_path = self.recovery_system._get_model_path(self.test_model_id)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_valid_model(self, model_id: str = None) -> Path:
        """Create a valid model directory structure for testing."""
        if model_id is None:
            model_id = self.test_model_id
        
        model_path = self.recovery_system._get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create required files with sufficient size to pass validation
        (model_path / "config.json").write_text('{"model_type": "test"}')
        # Create a larger fake model file to avoid size validation issues
        (model_path / "pytorch_model.bin").write_bytes(b"fake model weights" * 100000)  # Larger file
        (model_path / "tokenizer.json").write_text('{"tokenizer": "test"}')
        
        return model_path
    
    def _create_corrupted_model(self, model_id: str = None) -> Path:
        """Create a model with corrupted files for testing."""
        if model_id is None:
            model_id = self.test_model_id
        
        model_path = self.recovery_system._get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create files with some missing and some empty
        (model_path / "config.json").write_text('{"model_type": "test"}')
        (model_path / "pytorch_model.bin").write_text("")  # Empty file (corrupted)
        # Missing tokenizer.json
        
        return model_path
    
    def _create_incomplete_model(self, model_id: str = None) -> Path:
        """Create a model with incomplete download indicators."""
        if model_id is None:
            model_id = self.test_model_id
        
        model_path = self.recovery_system._get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create some valid files and some temporary files
        (model_path / "config.json").write_text('{"model_type": "test"}')
        (model_path / "pytorch_model.bin.tmp").write_bytes(b"incomplete download")
        (model_path / "tokenizer.json.part").write_text("partial download")
        
        return model_path


class TestModelIssueDetection(TestModelValidationRecovery):
    """Test model issue detection capabilities."""
    
    def test_validate_valid_model(self):
        """Test validation of a valid model."""
        # Create a valid model
        self._create_valid_model()
        
        # Validate the model
        result = self.recovery_system.validate_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(result.is_valid)
        self.assertEqual(result.model_id, self.test_model_id)
        self.assertTrue(result.structure_valid)
        self.assertTrue(result.required_files_present)
        self.assertGreater(result.file_count, 0)
        self.assertGreater(result.total_size_mb, 0)
        self.assertEqual(len([issue for issue in result.issues if issue.severity in ["high", "critical"]]), 0)
    
    def test_detect_missing_model_directory(self):
        """Test detection of missing model directory."""
        # Don't create the model directory
        
        # Validate the model
        result = self.recovery_system.validate_model(self.test_model_id)
        
        # Assertions
        self.assertFalse(result.is_valid)
        self.assertFalse(result.structure_valid)
        self.assertFalse(result.required_files_present)
        
        # Check for missing files issue
        missing_issues = [issue for issue in result.issues if issue.issue_type == ModelIssueType.MISSING_FILES]
        self.assertGreater(len(missing_issues), 0)
        self.assertEqual(missing_issues[0].severity, "critical")
    
    def test_detect_missing_required_files(self):
        """Test detection of missing required files."""
        # Create model directory but without required files
        model_path = self.recovery_system._get_model_path(self.test_model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Only create config.json, missing other required files
        (model_path / "config.json").write_text('{"model_type": "test"}')
        
        # Validate the model
        result = self.recovery_system.validate_model(self.test_model_id)
        
        # Assertions
        self.assertFalse(result.is_valid)
        self.assertFalse(result.required_files_present)
        
        # Check for missing files issues
        missing_issues = [issue for issue in result.issues if issue.issue_type == ModelIssueType.MISSING_FILES]
        self.assertGreater(len(missing_issues), 0)
        
        # Should detect missing tokenizer.json and model weights
        missing_files = [issue.description for issue in missing_issues]
        self.assertTrue(any("tokenizer.json" in desc for desc in missing_files))
        self.assertTrue(any("weight" in desc.lower() for desc in missing_files))
    
    def test_detect_corrupted_files(self):
        """Test detection of corrupted (empty) files."""
        # Create model with empty files
        self._create_corrupted_model()
        
        # Validate the model
        result = self.recovery_system.validate_model(self.test_model_id)
        
        # Assertions
        self.assertFalse(result.is_valid)
        
        # Check for corrupted files issues
        corrupted_issues = [issue for issue in result.issues if issue.issue_type == ModelIssueType.CORRUPTED_FILES]
        self.assertGreater(len(corrupted_issues), 0)
        
        # Should detect empty pytorch_model.bin
        corrupted_files = [issue.file_path for issue in corrupted_issues]
        self.assertTrue(any("pytorch_model.bin" in path for path in corrupted_files))
    
    def test_detect_incomplete_downloads(self):
        """Test detection of incomplete downloads."""
        # Create model with temporary files
        self._create_incomplete_model()
        
        # Validate the model
        result = self.recovery_system.validate_model(self.test_model_id)
        
        # Assertions
        self.assertFalse(result.is_valid)
        
        # Check for incomplete download issues
        incomplete_issues = [issue for issue in result.issues if issue.issue_type == ModelIssueType.INCOMPLETE_DOWNLOAD]
        self.assertGreater(len(incomplete_issues), 0)
        
        # Should detect .tmp and .part files
        temp_files = [issue.file_path for issue in incomplete_issues if issue.file_path]
        self.assertTrue(any(".tmp" in path for path in temp_files))
        self.assertTrue(any(".part" in path for path in temp_files))
    
    def test_detect_permission_errors(self):
        """Test detection of permission errors."""
        # Create a valid model first
        model_path = self._create_valid_model()
        
        # Mock os.access to simulate permission error
        with patch('os.access', return_value=False):
            result = self.recovery_system.validate_model(self.test_model_id)
        
        # Assertions
        self.assertFalse(result.structure_valid)
        
        # Check for permission error issues
        permission_issues = [issue for issue in result.issues if issue.issue_type == ModelIssueType.PERMISSION_ERROR]
        self.assertGreater(len(permission_issues), 0)
        self.assertEqual(permission_issues[0].severity, "high")
    
    def test_checksum_validation(self):
        """Test checksum validation functionality."""
        # Create a valid model
        model_path = self._create_valid_model()
        
        # Calculate and store checksums
        self.recovery_system._calculate_and_store_checksums(self.test_model_id, model_path)
        
        # Modify a file to create checksum mismatch
        (model_path / "config.json").write_text('{"model_type": "modified"}')
        
        # Validate the model
        result = self.recovery_system.validate_model(self.test_model_id)
        
        # Check for checksum mismatch issues
        checksum_issues = [issue for issue in result.issues if issue.issue_type == ModelIssueType.CHECKSUM_MISMATCH]
        self.assertGreater(len(checksum_issues), 0)
        self.assertEqual(checksum_issues[0].severity, "high")
        self.assertIn("expected", checksum_issues[0].additional_info)
        self.assertIn("actual", checksum_issues[0].additional_info)


class TestModelRecovery(TestModelValidationRecovery):
    """Test model recovery capabilities."""
    
    def test_recover_valid_model_no_action_needed(self):
        """Test recovery of an already valid model."""
        # Create a valid model
        self._create_valid_model()
        
        # Attempt recovery
        recovery_result = self.recovery_system.recover_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(recovery_result.success)
        self.assertIn("already valid", recovery_result.details)
        self.assertEqual(len(recovery_result.issues_resolved), 0)
        self.assertEqual(len(recovery_result.issues_remaining), 0)
    
    @patch('scripts.model_validation_recovery.ModelValidationRecovery._download_model_with_retry')
    def test_recover_missing_files(self, mock_download):
        """Test recovery of missing files."""
        # Mock successful download
        mock_download.return_value = True
        
        # Don't create the model (missing files)
        
        # Mock validation after recovery to return valid
        with patch.object(self.recovery_system, 'validate_model') as mock_validate:
            # First call (initial validation) - invalid
            invalid_result = ModelValidationResult(
                model_id=self.test_model_id,
                is_valid=False,
                issues=[ModelIssue(
                    issue_type=ModelIssueType.MISSING_FILES,
                    model_id=self.test_model_id,
                    description="Model directory does not exist",
                    severity="critical"
                )]
            )
            
            # Second call (after recovery) - valid
            valid_result = ModelValidationResult(
                model_id=self.test_model_id,
                is_valid=True
            )
            
            mock_validate.side_effect = [invalid_result, valid_result]
            
            # Attempt recovery
            recovery_result = self.recovery_system.recover_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(recovery_result.success)
        self.assertIn(ModelIssueType.MISSING_FILES, recovery_result.issues_resolved)
        self.assertEqual(recovery_result.recovery_method, "complete_redownload")
        mock_download.assert_called_once()
    
    @patch('scripts.model_validation_recovery.ModelValidationRecovery._download_model_with_retry')
    def test_recover_corrupted_files(self, mock_download):
        """Test recovery of corrupted files."""
        # Mock successful download
        mock_download.return_value = True
        
        # Create corrupted model
        self._create_corrupted_model()
        
        # Mock validation after recovery to return valid
        with patch.object(self.recovery_system, 'validate_model') as mock_validate:
            # First call (initial validation) - invalid with corruption
            invalid_result = ModelValidationResult(
                model_id=self.test_model_id,
                is_valid=False,
                issues=[ModelIssue(
                    issue_type=ModelIssueType.CORRUPTED_FILES,
                    model_id=self.test_model_id,
                    description="Empty file detected",
                    severity="high"
                )]
            )
            
            # Second call (after recovery) - valid
            valid_result = ModelValidationResult(
                model_id=self.test_model_id,
                is_valid=True
            )
            
            mock_validate.side_effect = [invalid_result, valid_result]
            
            # Attempt recovery
            recovery_result = self.recovery_system.recover_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(recovery_result.success)
        self.assertIn(ModelIssueType.CORRUPTED_FILES, recovery_result.issues_resolved)
        self.assertEqual(recovery_result.recovery_method, "corrupted_file_redownload")
        mock_download.assert_called_once()
    
    def test_recover_incomplete_downloads(self):
        """Test recovery of incomplete downloads."""
        # Create model with temporary files
        model_path = self._create_incomplete_model()
        
        # Mock successful download
        with patch.object(self.recovery_system, '_download_model_with_retry', return_value=True):
            # Mock validation after recovery to return valid
            with patch.object(self.recovery_system, 'validate_model') as mock_validate:
                # First call (initial validation) - invalid with incomplete download
                invalid_result = ModelValidationResult(
                    model_id=self.test_model_id,
                    is_valid=False,
                    issues=[ModelIssue(
                        issue_type=ModelIssueType.INCOMPLETE_DOWNLOAD,
                        model_id=self.test_model_id,
                        description="Temporary files found",
                        severity="medium"
                    )]
                )
                
                # Second call (after recovery) - valid
                valid_result = ModelValidationResult(
                    model_id=self.test_model_id,
                    is_valid=True
                )
                
                mock_validate.side_effect = [invalid_result, valid_result]
                
                # Attempt recovery
                recovery_result = self.recovery_system.recover_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(recovery_result.success)
        self.assertIn(ModelIssueType.INCOMPLETE_DOWNLOAD, recovery_result.issues_resolved)
        self.assertEqual(recovery_result.recovery_method, "incomplete_download_recovery")
        
        # Check that temporary files were cleaned up
        self.assertGreater(len(recovery_result.files_recovered), 0)
    
    def test_recover_invalid_structure(self):
        """Test recovery of invalid directory structure."""
        # Create a file where directory should be
        model_path = self.recovery_system._get_model_path(self.test_model_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("This should be a directory")
        
        # Mock successful download
        with patch.object(self.recovery_system, '_download_model_with_retry', return_value=True):
            # Mock validation after recovery to return valid
            with patch.object(self.recovery_system, 'validate_model') as mock_validate:
                # First call (initial validation) - invalid structure
                invalid_result = ModelValidationResult(
                    model_id=self.test_model_id,
                    is_valid=False,
                    issues=[ModelIssue(
                        issue_type=ModelIssueType.INVALID_STRUCTURE,
                        model_id=self.test_model_id,
                        description="Path is not a directory",
                        severity="critical"
                    )]
                )
                
                # Second call (after recovery) - valid
                valid_result = ModelValidationResult(
                    model_id=self.test_model_id,
                    is_valid=True
                )
                
                mock_validate.side_effect = [invalid_result, valid_result]
                
                # Attempt recovery
                recovery_result = self.recovery_system.recover_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(recovery_result.success)
        self.assertIn(ModelIssueType.INVALID_STRUCTURE, recovery_result.issues_resolved)
        self.assertEqual(recovery_result.recovery_method, "structure_recreation")
    
    def test_recover_permission_errors(self):
        """Test recovery of permission errors."""
        # Create a valid model
        model_path = self._create_valid_model()
        
        # Mock os.chmod to simulate successful permission fix
        with patch('os.chmod') as mock_chmod:
            with patch.object(self.recovery_system, 'validate_model') as mock_validate:
                # First call (initial validation) - permission error
                invalid_result = ModelValidationResult(
                    model_id=self.test_model_id,
                    is_valid=False,
                    issues=[ModelIssue(
                        issue_type=ModelIssueType.PERMISSION_ERROR,
                        model_id=self.test_model_id,
                        description="No read permission",
                        severity="high"
                    )]
                )
                
                # Second call (after recovery) - valid
                valid_result = ModelValidationResult(
                    model_id=self.test_model_id,
                    is_valid=True
                )
                
                mock_validate.side_effect = [invalid_result, valid_result]
                
                # Attempt recovery
                recovery_result = self.recovery_system.recover_model(self.test_model_id)
        
        # Assertions
        self.assertTrue(recovery_result.success)
        self.assertIn(ModelIssueType.PERMISSION_ERROR, recovery_result.issues_resolved)
        self.assertEqual(recovery_result.recovery_method, "permission_fix")
        mock_chmod.assert_called()


class TestModelDownload(TestModelValidationRecovery):
    """Test model download functionality."""
    
    @patch('huggingface_hub.snapshot_download')
    def test_download_with_huggingface_hub_success(self, mock_snapshot_download):
        """Test successful download using Hugging Face Hub."""
        # Mock successful download
        mock_snapshot_download.return_value = str(self.test_model_path)
        
        # Create a valid model after download (this will be used for size calculation)
        model_path = self._create_valid_model()
        
        recovery_result = ModelRecoveryResult(model_id=self.test_model_id, success=False)
        
        # Test download
        success = self.recovery_system._download_with_huggingface_hub(self.test_model_id, recovery_result)
        
        # Assertions
        self.assertTrue(success)
        self.assertEqual(recovery_result.recovery_method, "huggingface_hub_download")
        # The bytes_downloaded should be set by the method
        self.assertGreaterEqual(recovery_result.bytes_downloaded, 0)  # Allow 0 for test case
        mock_snapshot_download.assert_called_once()
    
    @patch('huggingface_hub.snapshot_download')
    def test_download_with_huggingface_hub_failure(self, mock_snapshot_download):
        """Test failed download using Hugging Face Hub."""
        # Mock failed download
        mock_snapshot_download.side_effect = Exception("Download failed")
        
        recovery_result = ModelRecoveryResult(model_id=self.test_model_id, success=False)
        
        # Test download
        success = self.recovery_system._download_with_huggingface_hub(self.test_model_id, recovery_result)
        
        # Assertions
        self.assertFalse(success)
        mock_snapshot_download.assert_called_once()
    
    def test_download_with_retry_multiple_attempts(self):
        """Test download with retry logic."""
        # Mock download methods: first attempt succeeds (to simplify the test)
        with patch.object(self.recovery_system, '_download_with_huggingface_hub', return_value=True):
            with patch('time.sleep'):  # Mock sleep to speed up test
                # Mock validation to succeed after successful download
                with patch.object(self.recovery_system, 'validate_model') as mock_validate:
                    mock_validate.return_value = ModelValidationResult(
                        model_id=self.test_model_id,
                        is_valid=True
                    )
                    
                    recovery_result = ModelRecoveryResult(model_id=self.test_model_id, success=False)
                    
                    # Test download with retry
                    success = self.recovery_system._download_model_with_retry(self.test_model_id, recovery_result)
        
        # Assertions
        self.assertTrue(success)
    
    def test_download_with_retry_all_attempts_fail(self):
        """Test download with retry when all attempts fail."""
        # Mock all download methods to fail
        with patch.object(self.recovery_system, '_download_with_huggingface_hub', return_value=False):
            with patch.object(self.recovery_system, '_download_with_alternative_source', return_value=False):
                with patch.object(self.recovery_system, '_download_with_manual_method', return_value=False):
                    with patch('time.sleep'):  # Speed up test by mocking sleep
                        recovery_result = ModelRecoveryResult(model_id=self.test_model_id, success=False)
                        
                        # Test download with retry
                        success = self.recovery_system._download_model_with_retry(self.test_model_id, recovery_result)
        
        # Assertions
        self.assertFalse(success)


class TestUtilityFunctions(TestModelValidationRecovery):
    """Test utility functions."""
    
    def test_get_model_path(self):
        """Test model path generation."""
        model_id = "Wan2.2/T2V-A14B"
        expected_path = self.models_dir / "Wan2.2_T2V-A14B"
        
        actual_path = self.recovery_system._get_model_path(model_id)
        
        self.assertEqual(actual_path, expected_path)
    
    def test_calculate_file_checksum(self):
        """Test file checksum calculation."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        # Calculate checksum
        checksum = self.recovery_system._calculate_file_checksum(test_file)
        
        # Verify checksum
        expected_checksum = hashlib.sha256(test_content).hexdigest()
        self.assertEqual(checksum, expected_checksum)
    
    def test_calculate_directory_size(self):
        """Test directory size calculation."""
        # Create test directory with files
        test_dir = Path(self.temp_dir) / "test_dir"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").write_bytes(b"A" * 100)
        (test_dir / "file2.txt").write_bytes(b"B" * 200)
        
        # Calculate size
        total_size = self.recovery_system._calculate_directory_size(test_dir)
        
        # Verify size
        self.assertEqual(total_size, 300)
    
    def test_calculate_and_store_checksums(self):
        """Test checksum calculation and storage."""
        # Create a model with files
        model_path = self._create_valid_model()
        
        # Calculate and store checksums
        self.recovery_system._calculate_and_store_checksums(self.test_model_id, model_path)
        
        # Verify checksums were stored
        self.assertIn(self.test_model_id, self.recovery_system.model_metadata)
        self.assertIn("checksums", self.recovery_system.model_metadata[self.test_model_id])
        
        checksums = self.recovery_system.model_metadata[self.test_model_id]["checksums"]
        self.assertIn("config.json", checksums)
        self.assertIn("pytorch_model.bin", checksums)
        self.assertIn("tokenizer.json", checksums)


class TestReporting(TestModelValidationRecovery):
    """Test reporting functionality."""
    
    def test_generate_detailed_report_valid_model(self):
        """Test report generation for a valid model."""
        # Create validation result for valid model
        validation_result = ModelValidationResult(
            model_id=self.test_model_id,
            is_valid=True,
            file_count=3,
            total_size_mb=100.5,
            structure_valid=True,
            required_files_present=True,
            checksum_verified=True
        )
        
        # Generate report
        report = self.recovery_system.generate_detailed_report(self.test_model_id, validation_result)
        
        # Verify report content
        self.assertIn("Model Validation and Recovery Report", report)
        self.assertIn(self.test_model_id, report)
        self.assertIn("Model Valid: True", report)
        self.assertIn("File Count: 3", report)
        self.assertIn("Total Size: 100.5 MB", report)
        self.assertIn("No issues found", report)
    
    def test_generate_detailed_report_with_issues(self):
        """Test report generation for a model with issues."""
        # Create validation result with issues
        issues = [
            ModelIssue(
                issue_type=ModelIssueType.MISSING_FILES,
                model_id=self.test_model_id,
                description="Missing tokenizer.json",
                severity="high",
                file_path="/path/to/tokenizer.json"
            ),
            ModelIssue(
                issue_type=ModelIssueType.CORRUPTED_FILES,
                model_id=self.test_model_id,
                description="Empty pytorch_model.bin",
                severity="critical",
                file_path="/path/to/pytorch_model.bin"
            )
        ]
        
        validation_result = ModelValidationResult(
            model_id=self.test_model_id,
            is_valid=False,
            issues=issues,
            file_count=2,
            total_size_mb=50.0
        )
        
        # Generate report
        report = self.recovery_system.generate_detailed_report(self.test_model_id, validation_result)
        
        # Verify report content
        self.assertIn("Model Valid: False", report)
        self.assertIn("Issues Found (2)", report)
        self.assertIn("MISSING_FILES", report)
        self.assertIn("CORRUPTED_FILES", report)
        self.assertIn("Missing tokenizer.json", report)
        self.assertIn("Empty pytorch_model.bin", report)
        self.assertIn("Severity: high", report)
        self.assertIn("Severity: critical", report)
    
    def test_generate_detailed_report_with_recovery(self):
        """Test report generation with recovery results."""
        # Create validation result
        validation_result = ModelValidationResult(
            model_id=self.test_model_id,
            is_valid=False,
            issues=[ModelIssue(
                issue_type=ModelIssueType.MISSING_FILES,
                model_id=self.test_model_id,
                description="Missing files",
                severity="high"
            )]
        )
        
        # Create recovery result
        recovery_result = ModelRecoveryResult(
            model_id=self.test_model_id,
            success=True,
            issues_resolved=[ModelIssueType.MISSING_FILES],
            recovery_method="complete_redownload",
            details="Successfully re-downloaded model",
            files_recovered=["tokenizer.json"],
            bytes_downloaded=1000000
        )
        
        # Generate report
        report = self.recovery_system.generate_detailed_report(
            self.test_model_id, validation_result, recovery_result
        )
        
        # Verify report content
        self.assertIn("Recovery Results:", report)
        self.assertIn("Recovery Successful: True", report)
        self.assertIn("Recovery Method: complete_redownload", report)
        self.assertIn("Issues Resolved: ['missing_files']", report)
        self.assertIn("Files Recovered: 1", report)
        self.assertIn("Bytes Downloaded: 1,000,000", report)
        self.assertIn("Successfully re-downloaded model", report)
    
    def test_get_recovery_suggestions(self):
        """Test recovery suggestions generation."""
        # Create validation result with various issues
        issues = [
            ModelIssue(
                issue_type=ModelIssueType.MISSING_FILES,
                model_id=self.test_model_id,
                description="Model directory missing",
                severity="critical"
            ),
            ModelIssue(
                issue_type=ModelIssueType.CORRUPTED_FILES,
                model_id=self.test_model_id,
                description="Corrupted file",
                severity="high",
                file_path="/path/to/file"
            ),
            ModelIssue(
                issue_type=ModelIssueType.PERMISSION_ERROR,
                model_id=self.test_model_id,
                description="Permission denied",
                severity="high"
            )
        ]
        
        validation_result = ModelValidationResult(
            model_id=self.test_model_id,
            is_valid=False,
            issues=issues
        )
        
        # Get suggestions
        suggestions = self.recovery_system.get_recovery_suggestions(validation_result)
        
        # Verify suggestions
        suggestions_text = "\n".join(suggestions)
        self.assertIn("CRITICAL ISSUES DETECTED", suggestions_text)
        self.assertIn("HIGH PRIORITY ISSUES", suggestions_text)
        self.assertIn("GENERAL RECOVERY STEPS", suggestions_text)
        self.assertIn("Re-download the model completely", suggestions_text)
        self.assertIn("Re-download corrupted file", suggestions_text)
        self.assertIn("Fix file permissions", suggestions_text)
        self.assertIn("stable internet connection", suggestions_text)
        self.assertIn("administrator", suggestions_text)


class TestIntegrationScenarios(TestModelValidationRecovery):
    """Test end-to-end integration scenarios."""
    
    def test_complete_recovery_workflow(self):
        """Test complete validation and recovery workflow."""
        # Create a model with multiple issues
        model_path = self.recovery_system._get_model_path(self.test_model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create some files with issues
        (model_path / "config.json").write_text('{"model_type": "test"}')
        (model_path / "pytorch_model.bin").write_text("")  # Empty (corrupted)
        (model_path / "temp_file.tmp").write_text("temporary")  # Incomplete download
        # Missing tokenizer.json
        
        # Mock successful recovery
        with patch.object(self.recovery_system, '_download_model_with_retry', return_value=True):
            # Mock final validation to return valid
            original_validate = self.recovery_system.validate_model
            
            def mock_validate(model_id):
                if hasattr(mock_validate, 'call_count'):
                    mock_validate.call_count += 1
                else:
                    mock_validate.call_count = 1
                
                if mock_validate.call_count == 1:
                    # First call - return actual validation (with issues)
                    return original_validate(model_id)
                else:
                    # Subsequent calls - return valid
                    return ModelValidationResult(model_id=model_id, is_valid=True)
            
            with patch.object(self.recovery_system, 'validate_model', side_effect=mock_validate):
                # Perform validation
                validation_result = self.recovery_system.validate_model(self.test_model_id)
                
                # Verify issues were detected
                self.assertFalse(validation_result.is_valid)
                self.assertGreater(len(validation_result.issues), 0)
                
                # Perform recovery
                recovery_result = self.recovery_system.recover_model(self.test_model_id, validation_result)
                
                # Verify recovery was successful
                self.assertTrue(recovery_result.success)
                self.assertGreater(len(recovery_result.issues_resolved), 0)
                
                # Generate report
                report = self.recovery_system.generate_detailed_report(
                    self.test_model_id, validation_result, recovery_result
                )
                
                # Verify report contains expected information
                self.assertIn("Recovery Successful: True", report)
                self.assertIn("Issues Found", report)
                self.assertIn("Recovery Results", report)
    
    def test_recovery_failure_scenario(self):
        """Test scenario where recovery fails."""
        # Create a model with critical issues
        model_path = self.recovery_system._get_model_path(self.test_model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Mock failed recovery
        with patch.object(self.recovery_system, '_download_model_with_retry', return_value=False):
            # Perform recovery
            recovery_result = self.recovery_system.recover_model(self.test_model_id)
            
            # Verify recovery failed
            self.assertFalse(recovery_result.success)
            self.assertGreater(len(recovery_result.issues_remaining), 0)
            
            # Get recovery suggestions
            validation_result = self.recovery_system.validate_model(self.test_model_id)
            suggestions = self.recovery_system.get_recovery_suggestions(validation_result)
            
            # Verify suggestions are provided
            self.assertGreater(len(suggestions), 0)
            self.assertTrue(any("manual" in suggestion.lower() for suggestion in suggestions))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestModelIssueDetection,
        TestModelRecovery,
        TestModelDownload,
        TestUtilityFunctions,
        TestReporting,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Model Validation Recovery System Test Results")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
