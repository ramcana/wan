"""
Model path resolution with cross-platform support and atomic operations.

This module provides deterministic path resolution for WAN2.2 models with support
for variants, temporary directories, and cross-platform compatibility including
Windows long paths and WSL scenarios.
"""

import os
import platform
import uuid
from pathlib import Path, PurePath
from typing import List, Optional, Union
import logging

from .exceptions import ModelOrchestratorError, ErrorCode

logger = logging.getLogger(__name__)


class PathIssue:
    """Represents a path validation issue."""
    
    def __init__(self, issue_type: str, message: str, suggestion: Optional[str] = None):
        self.issue_type = issue_type
        self.message = message
        self.suggestion = suggestion
    
    def __repr__(self) -> str:
        return f"PathIssue({self.issue_type}: {self.message})"


class ModelResolver:
    """
    Provides deterministic path resolution for WAN2.2 models with cross-platform support.
    
    Handles:
    - Deterministic path generation from MODELS_ROOT
    - Model variant support in path resolution
    - Temporary directory strategy for atomic operations
    - Windows long path scenarios
    - Cross-platform compatibility (Windows, WSL, Unix)
    """
    
    # Windows reserved names that need special handling
    WINDOWS_RESERVED_NAMES = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    # Maximum path length for different platforms
    MAX_PATH_WINDOWS = 260
    MAX_PATH_WINDOWS_EXTENDED = 32767
    MAX_PATH_UNIX = 4096
    
    def __init__(self, models_root: str):
        """
        Initialize the ModelResolver with a models root directory.
        
        Args:
            models_root: Base directory for all model storage
            
        Raises:
            ModelOrchestratorError: If models_root is invalid or inaccessible
        """
        if not models_root:
            raise ModelOrchestratorError(
                "models_root cannot be empty",
                ErrorCode.INVALID_CONFIG
            )
        
        self.models_root = Path(models_root).resolve()
        self._is_windows = platform.system() == "Windows"
        self._is_wsl = self._detect_wsl()
        
        # Validate models_root - only fail on critical issues, not warnings
        issues = self.validate_path_constraints(str(self.models_root))
        critical_issues = [issue for issue in issues if issue.issue_type in ["PATH_TOO_LONG", "INVALID_CHARACTER", "RESERVED_NAME"]]
        if critical_issues:
            error_messages = [issue.message for issue in critical_issues]
            raise ModelOrchestratorError(
                f"models_root has critical issues: {'; '.join(error_messages)}",
                ErrorCode.INVALID_CONFIG
            )
        
        # Log warnings for non-critical issues
        warning_issues = [issue for issue in issues if issue.issue_type not in ["PATH_TOO_LONG", "INVALID_CHARACTER", "RESERVED_NAME"]]
        for issue in warning_issues:
            logger.warning(f"models_root warning: {issue.message}")
            if issue.suggestion:
                logger.warning(f"Suggestion: {issue.suggestion}")
        
        logger.info(f"ModelResolver initialized with models_root: {self.models_root}")
    
    def _detect_wsl(self) -> bool:
        """Detect if running under Windows Subsystem for Linux."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except (FileNotFoundError, PermissionError):
            return False
    
    def local_dir(self, model_id: str, variant: Optional[str] = None) -> str:
        """
        Get the local directory path for a model.
        
        Args:
            model_id: Canonical model ID (e.g., "t2v-A14B@2.2.0")
            variant: Optional variant (e.g., "fp16", "bf16")
            
        Returns:
            Absolute path to the model directory
            
        Raises:
            ModelOrchestratorError: If model_id is invalid or path issues exist
        """
        if not model_id:
            raise ModelOrchestratorError(
                "model_id cannot be empty",
                ErrorCode.INVALID_MODEL_ID
            )
        
        # Normalize model_id to handle various formats
        normalized_id = self._normalize_model_id(model_id)
        
        # Build path components
        if variant:
            dir_name = f"{normalized_id}@{variant}"
        else:
            dir_name = normalized_id
        
        # Construct full path: {MODELS_ROOT}/wan22/{model_id}[@variant]
        model_path = self.models_root / "wan22" / dir_name
        
        # Validate the resulting path - only fail on critical issues
        issues = self.validate_path_constraints(str(model_path))
        critical_issues = [issue for issue in issues if issue.issue_type in ["PATH_TOO_LONG", "INVALID_CHARACTER", "RESERVED_NAME"]]
        if critical_issues:
            error_messages = [issue.message for issue in critical_issues]
            error_code = ErrorCode.INVALID_CONFIG  # Use a valid ErrorCode enum value
            raise ModelOrchestratorError(
                f"Model path has critical issues: {'; '.join(error_messages)}",
                error_code
            )
        
        return str(model_path)
    
    def temp_dir(self, model_id: str, variant: Optional[str] = None) -> str:
        """
        Get a temporary directory path for atomic downloads.
        
        Uses pattern: {MODELS_ROOT}/.tmp/{model}@{variant}.{uuid}.partial
        Ensures temp and final paths are on the same filesystem for atomic rename.
        
        Args:
            model_id: Canonical model ID
            variant: Optional variant
            
        Returns:
            Absolute path to temporary directory
            
        Raises:
            ModelOrchestratorError: If path cannot be created
        """
        if not model_id:
            raise ModelOrchestratorError(
                "model_id cannot be empty",
                ErrorCode.INVALID_MODEL_ID
            )
        
        # Normalize model_id
        normalized_id = self._normalize_model_id(model_id)
        
        # Build temp directory name with UUID for uniqueness
        unique_id = str(uuid.uuid4())[:8]  # Short UUID for path length
        if variant:
            temp_name = f"{normalized_id}@{variant}.{unique_id}.partial"
        else:
            temp_name = f"{normalized_id}.{unique_id}.partial"
        
        # Construct temp path: {MODELS_ROOT}/.tmp/{temp_name}
        temp_path = self.models_root / ".tmp" / temp_name
        
        # Validate the resulting path - only fail on critical issues
        issues = self.validate_path_constraints(str(temp_path))
        critical_issues = [issue for issue in issues if issue.issue_type in ["PATH_TOO_LONG", "INVALID_CHARACTER", "RESERVED_NAME"]]
        if critical_issues:
            error_messages = [issue.message for issue in critical_issues]
            error_code = ErrorCode.INVALID_CONFIG  # Use a valid ErrorCode enum value
            raise ModelOrchestratorError(
                f"Temp path has critical issues: {'; '.join(error_messages)}",
                error_code
            )
        
        return str(temp_path)
    
    def validate_path_constraints(self, path: str) -> List[PathIssue]:
        """
        Validate path against platform-specific constraints.
        
        Args:
            path: Path to validate
            
        Returns:
            List of PathIssue objects describing any problems
        """
        issues = []
        path_obj = Path(path)
        
        # Check path length constraints
        if self._is_windows:
            # Check if long paths are enabled
            if len(str(path)) > self.MAX_PATH_WINDOWS:
                if len(str(path)) > self.MAX_PATH_WINDOWS_EXTENDED:
                    issues.append(PathIssue(
                        "PATH_TOO_LONG",
                        f"Path length {len(str(path))} exceeds Windows maximum of {self.MAX_PATH_WINDOWS_EXTENDED}",
                        "Consider using a shorter MODELS_ROOT or model IDs"
                    ))
                else:
                    issues.append(PathIssue(
                        "LONG_PATH_WARNING",
                        f"Path length {len(str(path))} exceeds standard Windows limit of {self.MAX_PATH_WINDOWS}",
                        "Enable long paths in Windows: gpedit.msc → Computer Configuration → Administrative Templates → System → Filesystem → Enable Win32 long paths"
                    ))
        else:
            if len(str(path)) > self.MAX_PATH_UNIX:
                issues.append(PathIssue(
                    "PATH_TOO_LONG",
                    f"Path length {len(str(path))} exceeds Unix maximum of {self.MAX_PATH_UNIX}",
                    "Consider using a shorter MODELS_ROOT or model IDs"
                ))
        
        # Check for Windows reserved names
        if self._is_windows or self._is_wsl:
            for part in path_obj.parts:
                part_name = part.upper().split('.')[0]  # Remove extension for check
                if part_name in self.WINDOWS_RESERVED_NAMES:
                    issues.append(PathIssue(
                        "RESERVED_NAME",
                        f"Path contains Windows reserved name: {part}",
                        f"Rename to avoid reserved name '{part_name}'"
                    ))
        
        # Check for case sensitivity issues on case-insensitive filesystems
        if self._is_windows or (self._is_wsl and self._path_on_windows_drive(path)):
            # Check for potential case collisions in the same directory
            parent_parts = list(path_obj.parts[:-1])
            if parent_parts:
                final_part = path_obj.parts[-1]
                # This is a simplified check - in practice, you'd check existing files
                if final_part.lower() != final_part and final_part.upper() != final_part:
                    issues.append(PathIssue(
                        "CASE_SENSITIVITY_WARNING",
                        f"Mixed case in filename '{final_part}' may cause issues on case-insensitive filesystems",
                        "Use consistent casing (preferably lowercase) for better compatibility"
                    ))
        
        # Check for invalid characters
        invalid_chars = self._get_invalid_path_chars()
        path_str = str(path)
        
        # On Windows or when path looks like Windows path, skip drive letter colon (e.g., C:)
        if len(path_str) >= 2 and path_str[1] == ':' and path_str[0].isalpha():
            # Skip the drive letter and colon for validation
            path_to_check = path_str[2:]
        else:
            path_to_check = path_str
            
        for char in path_to_check:
            if char in invalid_chars:
                issues.append(PathIssue(
                    "INVALID_CHARACTER",
                    f"Path contains invalid character: '{char}'",
                    f"Remove or replace invalid character '{char}'"
                ))
        
        return issues
    
    def ensure_directory_exists(self, path: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path to create
            
        Raises:
            ModelOrchestratorError: If directory cannot be created
        """
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
        except PermissionError as e:
            raise ModelOrchestratorError(
                f"Cannot create directory {path}: {e}",
                ErrorCode.INVALID_CONFIG
            )
        except OSError as e:
            raise ModelOrchestratorError(
                f"Filesystem error creating directory {path}: {e}",
                ErrorCode.INVALID_CONFIG
            )
    
    def _normalize_model_id(self, model_id: str) -> str:
        """
        Normalize model ID to a consistent format.
        
        Args:
            model_id: Raw model ID
            
        Returns:
            Normalized model ID safe for filesystem use
        """
        # Remove any path separators and normalize
        normalized = model_id.replace('/', '_').replace('\\', '_')
        
        # Handle special characters that might cause issues
        normalized = normalized.replace(':', '_')
        
        return normalized
    
    def _path_on_windows_drive(self, path: str) -> bool:
        """
        Check if a WSL path is on a Windows drive (e.g., /mnt/c/).
        
        Args:
            path: Path to check
            
        Returns:
            True if path is on Windows drive in WSL
        """
        if not self._is_wsl:
            return False
        
        # Normalize path separators for consistent parsing
        normalized_path = path.replace('\\', '/')
        
        # Check for /mnt/[drive_letter]/ pattern
        if normalized_path.startswith('/mnt/'):
            parts = normalized_path.split('/')
            if len(parts) >= 3:
                drive_part = parts[2]
                return len(drive_part) == 1 and drive_part.isalpha()
        
        return False
    
    def _get_invalid_path_chars(self) -> set:
        """
        Get set of characters invalid in paths for current platform.
        
        Returns:
            Set of invalid characters
        """
        if self._is_windows or self._is_wsl:
            # Windows invalid characters
            return {'<', '>', ':', '"', '|', '?', '*'}
        else:
            # Unix systems - only null character is truly invalid
            return {'\0'}
    
    def get_models_root(self) -> str:
        """
        Get the configured models root directory.
        
        Returns:
            Absolute path to models root
        """
        return str(self.models_root)
    
    def is_same_filesystem(self, path1: str, path2: str) -> bool:
        """
        Check if two paths are on the same filesystem for atomic operations.
        
        Args:
            path1: First path
            path2: Second path
            
        Returns:
            True if paths are on same filesystem
        """
        try:
            stat1 = os.stat(Path(path1).parent)
            stat2 = os.stat(Path(path2).parent)
            
            if hasattr(stat1, 'st_dev') and hasattr(stat2, 'st_dev'):
                return stat1.st_dev == stat2.st_dev
            
            # Fallback: assume same filesystem if both under models_root
            return (Path(path1).is_relative_to(self.models_root) and 
                    Path(path2).is_relative_to(self.models_root))
        except (OSError, ValueError):
            # If we can't determine, assume they're on the same filesystem
            # if both are under models_root
            try:
                return (Path(path1).is_relative_to(self.models_root) and 
                        Path(path2).is_relative_to(self.models_root))
            except ValueError:
                return False