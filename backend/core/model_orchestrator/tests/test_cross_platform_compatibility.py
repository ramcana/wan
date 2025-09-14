"""
Cross-platform compatibility tests for Model Orchestrator.

Tests Windows, WSL, and Unix-specific behaviors including:
- Path handling and long path support
- File system operations and atomic moves
- Lock mechanisms and process synchronization
- Case sensitivity and reserved names
"""

import os
import platform
import stat
import tempfile
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from unittest.mock import Mock, patch

import pytest

from backend.core.model_orchestrator.model_resolver import ModelResolver
from backend.core.model_orchestrator.lock_manager import LockManager
from backend.core.model_orchestrator.exceptions import (
    PathTooLongError,
    FileSystemError,
    LockTimeoutError
)


class TestWindowsCompatibility:
    """Test Windows-specific behaviors and limitations."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_long_path_handling(self):
        """Test handling of Windows long paths (>260 characters)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Create a very long model ID that would exceed 260 chars
            long_model_id = "very-long-model-name-" + "x" * 200 + "@1.0.0"
            
            try:
                model_path = resolver.local_dir(long_model_id)
                # Should either work (if long paths enabled) or raise appropriate error
                Path(model_path).mkdir(parents=True, exist_ok=True)
            except PathTooLongError as e:
                # Should provide helpful guidance
                assert "long path support" in str(e).lower()
                assert "gpedit.msc" in str(e) or "registry" in str(e).lower()

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_reserved_filename_handling(self):
        """Test handling of Windows reserved filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Test reserved names
            reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
            
            for name in reserved_names:
                model_id = f"{name.lower()}-model@1.0.0"
                model_path = resolver.local_dir(model_id)
                
                # Should handle reserved names gracefully
                try:
                    Path(model_path).mkdir(parents=True, exist_ok=True)
                    # If it succeeds, the name was properly escaped/modified
                    assert Path(model_path).exists()
                except FileSystemError:
                    # If it fails, should provide clear error message
                    pass

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_case_insensitive_filesystem(self):
        """Test behavior on case-insensitive Windows filesystem."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Create model with lowercase name
            model_id_lower = "test-model@1.0.0"
            path_lower = resolver.local_dir(model_id_lower)
            Path(path_lower).mkdir(parents=True, exist_ok=True)
            
            # Try to access with different case
            model_id_upper = "TEST-MODEL@1.0.0"
            path_upper = resolver.local_dir(model_id_upper)
            
            # On Windows, these should resolve to same path
            assert Path(path_lower).resolve() == Path(path_upper).resolve()

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_unc_path_support(self):
        """Test UNC path support for network drives."""
        # Mock UNC path scenario
        with patch.object(Path, 'resolve') as mock_resolve:
            mock_resolve.return_value = Path(r"\\server\share\models")
            
            resolver = ModelResolver(r"\\server\share\models")
            model_path = resolver.local_dir("test-model@1.0.0")
            
            # Should handle UNC paths without errors
            assert "\\\\server\\share" in str(model_path)

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_file_locking(self):
        """Test Windows-specific file locking behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_manager = LockManager(temp_dir)
            
            # Test exclusive locking behavior
            with lock_manager.acquire_model_lock("test-model", timeout=1.0):
                # Second lock attempt should timeout on Windows
                with pytest.raises(LockTimeoutError):
                    with lock_manager.acquire_model_lock("test-model", timeout=0.1):
                        pass

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_junction_vs_symlink_preference(self):
        """Test preference for junctions over symlinks on Windows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "source.txt"
            source_file.write_text("test data")
            
            junction_target = Path(temp_dir) / "junction_link"
            symlink_target = Path(temp_dir) / "symlink_link"
            
            # Test junction creation (Windows-specific)
            try:
                # Simulate junction creation (would use mklink /J in real implementation)
                import subprocess
                subprocess.run([
                    "cmd", "/c", "mklink", "/J", 
                    str(junction_target), str(source_file.parent)
                ], check=True, capture_output=True)
                
                # Verify junction works
                assert junction_target.exists()
                junction_created = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                junction_created = False
            
            # Test symlink fallback
            try:
                symlink_target.symlink_to(source_file)
                symlink_created = True
            except OSError:
                symlink_created = False
            
            # On Windows, should prefer junctions when available
            # This test verifies the capability exists
            if platform.system() == "Windows":
                # At least one method should work
                assert junction_created or symlink_created


class TestWSLCompatibility:
    """Test WSL (Windows Subsystem for Linux) specific behaviors."""

    @pytest.mark.skipif(not self._is_wsl(), reason="WSL-specific test")
    def test_wsl_path_translation(self):
        """Test path translation between WSL and Windows paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Test that paths work correctly in WSL environment
            model_path = resolver.local_dir("test-model@1.0.0")
            
            # Should be able to create and access the path
            Path(model_path).mkdir(parents=True, exist_ok=True)
            assert Path(model_path).exists()

    @pytest.mark.skipif(not self._is_wsl(), reason="WSL-specific test")
    def test_wsl_case_sensitivity(self):
        """Test case sensitivity behavior in WSL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # WSL can be configured for case sensitivity
            test_path = Path(temp_dir) / "Test"
            test_path.mkdir()
            
            # Check if filesystem is case sensitive
            case_sensitive = not (Path(temp_dir) / "test").exists()
            
            resolver = ModelResolver(temp_dir)
            
            if case_sensitive:
                # Different case should create different paths
                path1 = resolver.local_dir("test-model@1.0.0")
                path2 = resolver.local_dir("TEST-MODEL@1.0.0")
                assert path1 != path2
            else:
                # Same as Windows behavior
                path1 = resolver.local_dir("test-model@1.0.0")
                path2 = resolver.local_dir("TEST-MODEL@1.0.0")
                assert Path(path1).resolve() == Path(path2).resolve()

    @staticmethod
    def _is_wsl():
        """Detect if running in WSL environment."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except FileNotFoundError:
            return False


class TestUnixCompatibility:
    """Test Unix/Linux specific behaviors."""

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_file_permissions(self):
        """Test Unix file permission handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            model_path = resolver.local_dir("test-model@1.0.0")
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            # Test that directories have correct permissions
            dir_stat = Path(model_path).stat()
            # Should be readable/writable by owner, readable by group/others
            expected_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            assert dir_stat.st_mode & 0o777 == expected_mode & 0o777

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_symlink_support(self):
        """Test symlink creation and handling on Unix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create source file
            source_file = Path(temp_dir) / "source.txt"
            source_file.write_text("test data")
            
            # Create symlink
            link_file = Path(temp_dir) / "link.txt"
            link_file.symlink_to(source_file)
            
            # Verify symlink works
            assert link_file.is_symlink()
            assert link_file.read_text() == "test data"

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_case_sensitivity(self):
        """Test case-sensitive filesystem behavior on Unix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Different case should create different paths
            path1 = resolver.local_dir("test-model@1.0.0")
            path2 = resolver.local_dir("TEST-MODEL@1.0.0")
            
            Path(path1).mkdir(parents=True, exist_ok=True)
            Path(path2).mkdir(parents=True, exist_ok=True)
            
            # Both should exist as separate directories
            assert Path(path1).exists()
            assert Path(path2).exists()
            assert path1 != path2

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_file_locking(self):
        """Test Unix fcntl-based file locking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_manager = LockManager(temp_dir)
            
            # Test that locks work correctly with fcntl
            with lock_manager.acquire_model_lock("test-model", timeout=1.0):
                # Should be able to detect lock from same process
                assert lock_manager.is_locked("test-model")


class TestCrossPlatformPathHandling:
    """Test path handling that works across all platforms."""

    def test_path_normalization(self):
        """Test path normalization across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Test various path separators and formats
            model_id = "test/model@1.0.0"  # Contains slash
            model_path = resolver.local_dir(model_id)
            
            # Should normalize to platform-appropriate separators
            if platform.system() == "Windows":
                assert "\\" in model_path or "/" in model_path  # Either is acceptable
            else:
                assert "/" in model_path

    def test_special_character_handling(self):
        """Test handling of special characters in model names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Test various special characters
            special_chars = ["@", "-", "_", ".", "~"]
            
            for char in special_chars:
                model_id = f"test{char}model@1.0.0"
                try:
                    model_path = resolver.local_dir(model_id)
                    Path(model_path).mkdir(parents=True, exist_ok=True)
                    assert Path(model_path).exists()
                except Exception as e:
                    # If character is not supported, should provide clear error
                    assert "character" in str(e).lower() or "invalid" in str(e).lower()

    def test_unicode_path_support(self):
        """Test Unicode character support in paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            # Test Unicode model name
            model_id = "测试模型@1.0.0"  # Chinese characters
            
            try:
                model_path = resolver.local_dir(model_id)
                Path(model_path).mkdir(parents=True, exist_ok=True)
                assert Path(model_path).exists()
            except UnicodeError:
                # Some filesystems may not support Unicode
                pytest.skip("Filesystem does not support Unicode paths")

    def test_atomic_operations_cross_platform(self):
        """Test atomic file operations work on all platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "source.txt"
            target_file = Path(temp_dir) / "target.txt"
            
            source_file.write_text("test data")
            
            # Test atomic rename (should work on all platforms)
            source_file.rename(target_file)
            
            assert target_file.exists()
            assert not source_file.exists()
            assert target_file.read_text() == "test data"

    def test_temp_directory_same_volume(self):
        """Test that temp directories are on same volume for atomic operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(temp_dir)
            
            model_id = "test-model@1.0.0"
            temp_dir_path = resolver.temp_dir(model_id)
            final_dir_path = resolver.local_dir(model_id)
            
            # Both should be under the same root to ensure same volume
            temp_parent = Path(temp_dir_path).parent
            final_parent = Path(final_dir_path).parent
            
            # Should share common root (models_root)
            assert str(temp_parent).startswith(temp_dir)
            assert str(final_parent).startswith(temp_dir)


class TestPlatformSpecificErrorHandling:
    """Test platform-specific error handling and messages."""

    def test_windows_error_messages(self):
        """Test Windows-specific error messages and guidance."""
        if platform.system() != "Windows":
            pytest.skip("Windows-specific test")
        
        # Test long path error guidance
        error = PathTooLongError("Path too long", path="C:\\very\\long\\path")
        assert "long path support" in str(error).lower()
        assert any(keyword in str(error).lower() for keyword in ["gpedit", "registry", "group policy"])

    def test_unix_error_messages(self):
        """Test Unix-specific error messages and guidance."""
        if platform.system() == "Windows":
            pytest.skip("Unix-specific test")
        
        # Test permission error guidance
        error = FileSystemError("Permission denied", path="/restricted/path")
        assert "permission" in str(error).lower()
        assert any(keyword in str(error).lower() for keyword in ["chmod", "chown", "sudo"])

    def test_cross_platform_error_consistency(self):
        """Test that error messages are consistent across platforms."""
        from backend.core.model_orchestrator.exceptions import (
            PathTooLongError, FileSystemError, LockTimeoutError
        )
        
        # Test that error messages contain helpful information regardless of platform
        path_error = PathTooLongError("Path too long", path="/very/long/path")
        assert "path" in str(path_error).lower()
        assert len(str(path_error)) > 20  # Should have helpful details
        
        fs_error = FileSystemError("Permission denied", path="/restricted/path")
        assert "permission" in str(fs_error).lower() or "access" in str(fs_error).lower()
        
        lock_error = LockTimeoutError("Lock timeout", model_id="test-model")
        assert "timeout" in str(lock_error).lower() or "lock" in str(lock_error).lower()
        
        # All errors should provide actionable information
        for error in [path_error, fs_error, lock_error]:
            error_msg = str(error).lower()
            # Should contain either a suggestion or explanation
            has_suggestion = any(word in error_msg for word in [
                "try", "check", "ensure", "verify", "enable", "disable", 
                "configure", "set", "change", "update"
            ])
            has_explanation = len(error_msg) > 30  # Reasonable explanation length
            
            assert has_suggestion or has_explanation, f"Error lacks helpful info: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])