"""
Unit tests for ModelResolver class.

Tests cross-platform path handling, atomic operations, and Windows long path scenarios.
"""

import os
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from .model_resolver import ModelResolver, PathIssue
from .exceptions import ModelOrchestratorError, ErrorCode


class TestModelResolver(unittest.TestCase):
    """Test cases for ModelResolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_root = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_root, exist_ok=True)
        self.resolver = ModelResolver(self.models_root)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_valid_models_root(self):
        """Test ModelResolver initialization with valid models_root."""
        resolver = ModelResolver(self.models_root)
        self.assertEqual(str(resolver.get_models_root()), str(Path(self.models_root).resolve()))
    
    def test_init_empty_models_root(self):
        """Test ModelResolver initialization with empty models_root."""
        with self.assertRaises(ModelOrchestratorError) as cm:
            ModelResolver("")
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_CONFIG)
        self.assertIn("cannot be empty", str(cm.exception))
    
    def test_init_none_models_root(self):
        """Test ModelResolver initialization with None models_root."""
        with self.assertRaises(ModelOrchestratorError) as cm:
            ModelResolver(None)
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_CONFIG)
    
    def test_local_dir_basic(self):
        """Test basic local directory path generation."""
        model_id = "t2v-A14B@2.2.0"
        expected_path = os.path.join(self.models_root, "wan22", "t2v-A14B@2.2.0")
        result = self.resolver.local_dir(model_id)
        self.assertEqual(result, str(Path(expected_path).resolve()))
    
    def test_local_dir_with_variant(self):
        """Test local directory path generation with variant."""
        model_id = "t2v-A14B@2.2.0"
        variant = "fp16"
        expected_path = os.path.join(self.models_root, "wan22", "t2v-A14B@2.2.0@fp16")
        result = self.resolver.local_dir(model_id, variant)
        self.assertEqual(result, str(Path(expected_path).resolve()))
    
    def test_local_dir_empty_model_id(self):
        """Test local directory with empty model_id."""
        with self.assertRaises(ModelOrchestratorError) as cm:
            self.resolver.local_dir("")
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_MODEL_ID)
    
    def test_local_dir_none_model_id(self):
        """Test local directory with None model_id."""
        with self.assertRaises(ModelOrchestratorError) as cm:
            self.resolver.local_dir(None)
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_MODEL_ID)
    
    def test_temp_dir_basic(self):
        """Test temporary directory path generation."""
        model_id = "t2v-A14B@2.2.0"
        result = self.resolver.temp_dir(model_id)
        
        # Should be under .tmp directory
        self.assertIn(".tmp", result)
        self.assertTrue(result.startswith(str(self.resolver.get_models_root())))
        
        # Should contain model ID and end with .partial
        self.assertIn("t2v-A14B@2.2.0", result)
        self.assertTrue(result.endswith(".partial"))
    
    def test_temp_dir_with_variant(self):
        """Test temporary directory path generation with variant."""
        model_id = "t2v-A14B@2.2.0"
        variant = "fp16"
        result = self.resolver.temp_dir(model_id, variant)
        
        # Should contain both model ID and variant
        self.assertIn("t2v-A14B@2.2.0@fp16", result)
        self.assertTrue(result.endswith(".partial"))
    
    def test_temp_dir_uniqueness(self):
        """Test that temp directories are unique."""
        model_id = "t2v-A14B@2.2.0"
        result1 = self.resolver.temp_dir(model_id)
        result2 = self.resolver.temp_dir(model_id)
        
        # Should be different due to UUID
        self.assertNotEqual(result1, result2)
    
    def test_temp_dir_empty_model_id(self):
        """Test temp directory with empty model_id."""
        with self.assertRaises(ModelOrchestratorError) as cm:
            self.resolver.temp_dir("")
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_MODEL_ID)
    
    def test_normalize_model_id(self):
        """Test model ID normalization."""
        # Test path separator replacement
        self.assertEqual(
            self.resolver._normalize_model_id("model/with/slashes"),
            "model_with_slashes"
        )
        self.assertEqual(
            self.resolver._normalize_model_id("model\\with\\backslashes"),
            "model_with_backslashes"
        )
        
        # Test colon replacement
        self.assertEqual(
            self.resolver._normalize_model_id("model:with:colons"),
            "model_with_colons"
        )
    
    def test_validate_path_constraints_valid_path(self):
        """Test path validation with valid path."""
        valid_path = os.path.join(self.models_root, "valid", "path")
        issues = self.resolver.validate_path_constraints(valid_path)
        self.assertEqual(len(issues), 0)
    
    @unittest.skipUnless(platform.system() == "Windows", "Windows-specific test")
    def test_validate_path_constraints_windows_reserved_names(self):
        """Test path validation with Windows reserved names."""
        reserved_path = os.path.join(self.models_root, "CON", "file.txt")
        issues = self.resolver.validate_path_constraints(reserved_path)
        
        # Should have at least one issue for reserved name
        reserved_issues = [issue for issue in issues if issue.issue_type == "RESERVED_NAME"]
        self.assertGreater(len(reserved_issues), 0)
    
    def test_validate_path_constraints_long_path(self):
        """Test path validation with very long path."""
        # Create a path that's definitely too long for Windows (>260 chars)
        # Start with a base that ensures we exceed the limit
        base_path = "C:\\very\\long\\path\\that\\will\\exceed\\windows\\limits\\"
        long_component = "a" * 400  # Make it very long
        long_path = base_path + long_component + "\\file.txt"
        
        # Ensure the path is longer than Windows standard limit
        self.assertGreater(len(long_path), 260, "Test path should be longer than 260 characters")
        
        issues = self.resolver.validate_path_constraints(long_path)
        
        # Should have path length issues (either warning or error)
        length_issues = [issue for issue in issues if "too long" in issue.message.lower() or "exceeds" in issue.message.lower()]
        if platform.system() == "Windows":
            self.assertGreater(len(length_issues), 0, f"Expected path length issues for path of length {len(long_path)}")
    
    def test_validate_path_constraints_invalid_characters(self):
        """Test path validation with invalid characters."""
        if platform.system() == "Windows":
            invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        else:
            invalid_chars = ['\0']
        
        for char in invalid_chars:
            if char == ':' and platform.system() != "Windows":
                continue  # Colon is valid on Unix
            
            invalid_path = os.path.join(self.models_root, f"invalid{char}name")
            issues = self.resolver.validate_path_constraints(invalid_path)
            
            char_issues = [issue for issue in issues if issue.issue_type == "INVALID_CHARACTER"]
            self.assertGreater(len(char_issues), 0, f"Should detect invalid character: {char}")
    
    def test_ensure_directory_exists_new_directory(self):
        """Test creating a new directory."""
        new_dir = os.path.join(self.temp_dir, "new_directory")
        self.assertFalse(os.path.exists(new_dir))
        
        self.resolver.ensure_directory_exists(new_dir)
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))
    
    def test_ensure_directory_exists_existing_directory(self):
        """Test with existing directory."""
        existing_dir = os.path.join(self.temp_dir, "existing")
        os.makedirs(existing_dir)
        
        # Should not raise an error
        self.resolver.ensure_directory_exists(existing_dir)
        self.assertTrue(os.path.exists(existing_dir))
    
    def test_ensure_directory_exists_nested_directories(self):
        """Test creating nested directories."""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        self.assertFalse(os.path.exists(nested_dir))
        
        self.resolver.ensure_directory_exists(nested_dir)
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.isdir(nested_dir))
    
    @patch('pathlib.Path.mkdir')
    def test_ensure_directory_exists_permission_error(self, mock_mkdir):
        """Test handling permission errors when creating directories."""
        mock_mkdir.side_effect = PermissionError("Access denied")
        
        with self.assertRaises(ModelOrchestratorError) as cm:
            self.resolver.ensure_directory_exists("/some/path")
        
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_CONFIG)
        self.assertIn("Access denied", str(cm.exception))
    
    @patch('pathlib.Path.mkdir')
    def test_ensure_directory_exists_os_error(self, mock_mkdir):
        """Test handling OS errors when creating directories."""
        mock_mkdir.side_effect = OSError("Disk full")
        
        with self.assertRaises(ModelOrchestratorError) as cm:
            self.resolver.ensure_directory_exists("/some/path")
        
        self.assertEqual(cm.exception.error_code, ErrorCode.INVALID_CONFIG)
        self.assertIn("Disk full", str(cm.exception))
    
    def test_is_same_filesystem_same_root(self):
        """Test filesystem detection for paths under same root."""
        path1 = os.path.join(self.models_root, "path1")
        path2 = os.path.join(self.models_root, "path2")
        
        # Create parent directories
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        
        result = self.resolver.is_same_filesystem(path1, path2)
        self.assertTrue(result)
    
    def test_get_models_root(self):
        """Test getting models root directory."""
        result = self.resolver.get_models_root()
        self.assertEqual(result, str(Path(self.models_root).resolve()))
    
    @patch('platform.system')
    def test_windows_detection(self, mock_system):
        """Test Windows platform detection."""
        mock_system.return_value = "Windows"
        resolver = ModelResolver(self.models_root)
        self.assertTrue(resolver._is_windows)
    
    @patch('platform.system')
    @patch('builtins.open', mock_open(read_data="Linux version 4.4.0-Microsoft"))
    def test_wsl_detection(self, mock_system):
        """Test WSL detection."""
        mock_system.return_value = "Linux"
        resolver = ModelResolver(self.models_root)
        self.assertTrue(resolver._is_wsl)
    
    @patch('platform.system')
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_wsl_detection_no_proc_version(self, mock_open, mock_system):
        """Test WSL detection when /proc/version doesn't exist."""
        mock_system.return_value = "Linux"
        resolver = ModelResolver(self.models_root)
        self.assertFalse(resolver._is_wsl)
    
    def test_path_on_windows_drive_wsl(self):
        """Test Windows drive detection in WSL."""
        # Mock WSL environment
        with patch.object(self.resolver, '_is_wsl', True):
            # Test Windows drive path
            self.assertTrue(self.resolver._path_on_windows_drive("/mnt/c/Users/test"))
            self.assertTrue(self.resolver._path_on_windows_drive("/mnt/d/data"))
            
            # Test non-Windows drive paths
            self.assertFalse(self.resolver._path_on_windows_drive("/home/user"))
            self.assertFalse(self.resolver._path_on_windows_drive("/tmp/test"))
    
    def test_path_on_windows_drive_non_wsl(self):
        """Test Windows drive detection on non-WSL systems."""
        with patch.object(self.resolver, '_is_wsl', False):
            # Should always return False on non-WSL
            self.assertFalse(self.resolver._path_on_windows_drive("/mnt/c/Users/test"))
            self.assertFalse(self.resolver._path_on_windows_drive("C:\\Users\\test"))
    
    def test_get_invalid_path_chars_windows(self):
        """Test invalid path characters on Windows."""
        with patch.object(self.resolver, '_is_windows', True):
            invalid_chars = self.resolver._get_invalid_path_chars()
            expected_chars = {'<', '>', ':', '"', '|', '?', '*'}
            self.assertEqual(invalid_chars, expected_chars)
    
    def test_get_invalid_path_chars_unix(self):
        """Test invalid path characters on Unix."""
        with patch.object(self.resolver, '_is_windows', False):
            with patch.object(self.resolver, '_is_wsl', False):
                invalid_chars = self.resolver._get_invalid_path_chars()
                expected_chars = {'\0'}
                self.assertEqual(invalid_chars, expected_chars)
    
    def test_atomic_operations_same_filesystem(self):
        """Test that temp and final paths are on same filesystem."""
        model_id = "t2v-A14B@2.2.0"
        
        local_path = self.resolver.local_dir(model_id)
        temp_path = self.resolver.temp_dir(model_id)
        
        # Both should be under models_root
        self.assertTrue(Path(local_path).is_relative_to(self.resolver.get_models_root()))
        self.assertTrue(Path(temp_path).is_relative_to(self.resolver.get_models_root()))
        
        # Create parent directories for filesystem check
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Should be on same filesystem
        self.assertTrue(self.resolver.is_same_filesystem(local_path, temp_path))


class TestPathIssue(unittest.TestCase):
    """Test cases for PathIssue class."""
    
    def test_path_issue_creation(self):
        """Test PathIssue creation."""
        issue = PathIssue("TEST_TYPE", "Test message", "Test suggestion")
        self.assertEqual(issue.issue_type, "TEST_TYPE")
        self.assertEqual(issue.message, "Test message")
        self.assertEqual(issue.suggestion, "Test suggestion")
    
    def test_path_issue_repr(self):
        """Test PathIssue string representation."""
        issue = PathIssue("TEST_TYPE", "Test message")
        expected = "PathIssue(TEST_TYPE: Test message)"
        self.assertEqual(repr(issue), expected)


if __name__ == '__main__':
    unittest.main()