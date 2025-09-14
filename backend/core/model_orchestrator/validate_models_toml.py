#!/usr/bin/env python3
"""
Models.toml Validator

Validates the models.toml manifest file for:
- Schema version compatibility
- No duplicate model IDs or file paths
- No path traversal vulnerabilities
- Windows case sensitivity compatibility
- Proper TOML structure and required fields
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import re

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions

# Add the backend directory to the path for imports
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, backend_dir)

try:
    from core.model_orchestrator.model_registry import ModelRegistry
    from core.model_orchestrator.exceptions import ManifestValidationError, SchemaVersionError
except ImportError:
    # Fallback for direct execution
    ModelRegistry = None
    ManifestValidationError = Exception
    SchemaVersionError = Exception


class ModelsTomlValidator:
    """Comprehensive validator for models.toml manifest files."""
    
    SUPPORTED_SCHEMA_VERSIONS = ["1"]
    WINDOWS_RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    
    def __init__(self, manifest_path: str):
        """Initialize validator with manifest path."""
        self.manifest_path = Path(manifest_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Perform comprehensive validation of the models.toml file.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Check if file exists
        if not self.manifest_path.exists():
            self.errors.append(f"Manifest file not found: {self.manifest_path}")
            return False, self.errors, self.warnings
        
        # Load and parse TOML
        try:
            with open(self.manifest_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            self.errors.append(f"Failed to parse TOML: {e}")
            return False, self.errors, self.warnings
        
        # Validate schema version
        self._validate_schema_version(data)
        
        # Validate models structure
        self._validate_models_structure(data)
        
        # Validate for duplicates
        self._validate_no_duplicates(data)
        
        # Validate path safety
        self._validate_path_safety(data)
        
        # Validate Windows compatibility
        self._validate_windows_compatibility(data)
        
        # Test with ModelRegistry for additional validation
        self._validate_with_model_registry()
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_schema_version(self, data: dict) -> None:
        """Validate schema version is present and supported."""
        schema_version = data.get("schema_version")
        
        if schema_version is None:
            self.errors.append("Missing required 'schema_version' field")
            return
        
        schema_version_str = str(schema_version)
        if schema_version_str not in self.SUPPORTED_SCHEMA_VERSIONS:
            self.errors.append(
                f"Unsupported schema version '{schema_version_str}'. "
                f"Supported versions: {', '.join(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )
    
    def _validate_models_structure(self, data: dict) -> None:
        """Validate the models section structure."""
        models = data.get("models", {})
        
        if not models:
            self.errors.append("No models defined in manifest")
            return
        
        if not isinstance(models, dict):
            self.errors.append("'models' must be a dictionary")
            return
        
        # Validate each model
        for model_id, model_data in models.items():
            self._validate_model_entry(model_id, model_data)
    
    def _validate_model_entry(self, model_id: str, model_data: dict) -> None:
        """Validate a single model entry."""
        # Validate model ID format
        model_id_pattern = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*@[0-9]+\.[0-9]+\.[0-9]+$")
        if not model_id_pattern.match(model_id):
            self.errors.append(f"Invalid model ID format: {model_id}")
        
        if not isinstance(model_data, dict):
            self.errors.append(f"Model '{model_id}' data must be a dictionary")
            return
        
        # Check required fields
        required_fields = ["version", "variants", "default_variant", "files", "sources"]
        for field in required_fields:
            if field not in model_data:
                self.errors.append(f"Model '{model_id}' missing required field: {field}")
        
        # Validate variants
        variants = model_data.get("variants", [])
        if not isinstance(variants, list) or not variants:
            self.errors.append(f"Model '{model_id}' variants must be a non-empty list")
        
        default_variant = model_data.get("default_variant")
        if default_variant and default_variant not in variants:
            self.errors.append(
                f"Model '{model_id}' default_variant '{default_variant}' not in variants list"
            )
        
        # Validate files
        files = model_data.get("files", [])
        if not isinstance(files, list) or not files:
            self.errors.append(f"Model '{model_id}' files must be a non-empty list")
        else:
            self._validate_files_section(model_id, files)
        
        # Validate sources
        sources = model_data.get("sources", {})
        if not isinstance(sources, dict):
            self.errors.append(f"Model '{model_id}' sources must be a dictionary")
        else:
            priority = sources.get("priority", [])
            if not isinstance(priority, list) or not priority:
                self.errors.append(f"Model '{model_id}' sources.priority must be a non-empty list")
    
    def _validate_files_section(self, model_id: str, files: List[dict]) -> None:
        """Validate the files section of a model."""
        for i, file_entry in enumerate(files):
            if not isinstance(file_entry, dict):
                self.errors.append(f"Model '{model_id}' file entry {i} must be a dictionary")
                continue
            
            # Check required file fields
            required_file_fields = ["path", "size", "sha256"]
            for field in required_file_fields:
                if field not in file_entry:
                    self.errors.append(
                        f"Model '{model_id}' file entry {i} missing required field: {field}"
                    )
            
            # Validate file path
            path = file_entry.get("path")
            if path and not isinstance(path, str):
                self.errors.append(f"Model '{model_id}' file entry {i} path must be a string")
            
            # Validate size
            size = file_entry.get("size")
            if size is not None and (not isinstance(size, int) or size < 0):
                self.errors.append(f"Model '{model_id}' file entry {i} size must be a non-negative integer")
            
            # Validate SHA256
            sha256 = file_entry.get("sha256")
            if sha256 and (not isinstance(sha256, str) or len(sha256) != 64):
                self.errors.append(f"Model '{model_id}' file entry {i} sha256 must be a 64-character string")
    
    def _validate_no_duplicates(self, data: dict) -> None:
        """Validate there are no duplicate model IDs or file paths within models."""
        models = data.get("models", {})
        
        # Check for duplicate model IDs (should be impossible with dict structure, but check anyway)
        model_ids = list(models.keys())
        if len(model_ids) != len(set(model_ids)):
            self.errors.append("Duplicate model IDs found")
        
        # Check for duplicate file paths within each model
        for model_id, model_data in models.items():
            files = model_data.get("files", [])
            file_paths = []
            
            for file_entry in files:
                if isinstance(file_entry, dict) and "path" in file_entry:
                    file_paths.append(file_entry["path"])
            
            # Check for exact duplicates
            if len(file_paths) != len(set(file_paths)):
                duplicates = [path for path in file_paths if file_paths.count(path) > 1]
                unique_duplicates = list(set(duplicates))
                self.errors.append(
                    f"Model '{model_id}' has duplicate file paths: {', '.join(unique_duplicates)}"
                )
    
    def _validate_path_safety(self, data: dict) -> None:
        """Validate paths for safety (no directory traversal)."""
        models = data.get("models", {})
        
        for model_id, model_data in models.items():
            files = model_data.get("files", [])
            
            for file_entry in files:
                if not isinstance(file_entry, dict) or "path" not in file_entry:
                    continue
                
                path = file_entry["path"]
                if not isinstance(path, str):
                    continue
                
                # Check for directory traversal
                path_obj = Path(path)
                if ".." in path_obj.parts:
                    self.errors.append(f"Model '{model_id}' has path traversal in: {path}")
                
                # Check for absolute paths
                if path_obj.is_absolute() or path.startswith('/') or (len(path) > 1 and path[1] == ':'):
                    self.errors.append(f"Model '{model_id}' has absolute path (not allowed): {path}")
                
                # Check for Windows reserved names
                for part in path_obj.parts:
                    # Remove extension for reserved name check
                    name_without_ext = part.split('.')[0].upper()
                    if name_without_ext in self.WINDOWS_RESERVED_NAMES:
                        self.errors.append(f"Model '{model_id}' uses Windows reserved name in path: {path}")
    
    def _validate_windows_compatibility(self, data: dict) -> None:
        """Validate Windows case sensitivity compatibility."""
        models = data.get("models", {})
        
        for model_id, model_data in models.items():
            files = model_data.get("files", [])
            file_paths = []
            
            for file_entry in files:
                if isinstance(file_entry, dict) and "path" in file_entry:
                    path = file_entry["path"]
                    if isinstance(path, str):
                        file_paths.append(path)
            
            # Check for case collisions (paths that differ only in case)
            lower_paths = {}
            for path in file_paths:
                lower_path = path.lower()
                if lower_path in lower_paths:
                    self.errors.append(
                        f"Model '{model_id}' has case collision between: '{lower_paths[lower_path]}' and '{path}'"
                    )
                else:
                    lower_paths[lower_path] = path
            
            # Check for paths that might cause issues on Windows
            for path in file_paths:
                # Check for trailing spaces or dots (problematic on Windows)
                if path.endswith(' ') or path.endswith('.'):
                    self.warnings.append(
                        f"Model '{model_id}' path ends with space or dot (may cause Windows issues): {path}"
                    )
                
                # Check for very long paths (Windows has 260 character limit by default)
                if len(path) > 200:  # Leave some margin for the base path
                    self.warnings.append(
                        f"Model '{model_id}' has very long path (may exceed Windows limits): {path}"
                    )
    
    def _validate_with_model_registry(self) -> None:
        """Validate using the ModelRegistry class for additional checks."""
        if ModelRegistry is None:
            self.warnings.append("ModelRegistry not available for additional validation")
            return
            
        try:
            registry = ModelRegistry(str(self.manifest_path))
            
            # If we get here, the registry loaded successfully
            # Run additional validation
            validation_errors = registry.validate_manifest()
            for error in validation_errors:
                self.errors.append(f"ModelRegistry validation: {error}")
                
        except SchemaVersionError as e:
            self.errors.append(f"Schema version error: {e}")
        except ManifestValidationError as e:
            if hasattr(e, 'errors'):
                for error_msg in e.errors:
                    self.errors.append(f"Manifest validation error: {error_msg}")
            else:
                self.errors.append(f"Manifest validation error: {e}")
        except Exception as e:
            self.errors.append(f"Unexpected error during ModelRegistry validation: {e}")


def main():
    """Main function to run validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate models.toml manifest file")
    parser.add_argument(
        "manifest_path", 
        nargs="?", 
        default="config/models.toml",
        help="Path to models.toml file (default: config/models.toml)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Show warnings in addition to errors"
    )
    
    args = parser.parse_args()
    
    # Resolve path relative to project root
    if not os.path.isabs(args.manifest_path):
        # Find project root (look for pyproject.toml or setup.py)
        current_dir = Path(__file__).parent
        while current_dir != current_dir.parent:
            if (current_dir / "pyproject.toml").exists() or (current_dir / "setup.py").exists():
                break
            current_dir = current_dir.parent
        
        manifest_path = current_dir / args.manifest_path
    else:
        manifest_path = Path(args.manifest_path)
    
    print(f"Validating models.toml: {manifest_path}")
    print("=" * 60)
    
    validator = ModelsTomlValidator(str(manifest_path))
    is_valid, errors, warnings = validator.validate()
    
    # Print results
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  • {error}")
        print()
    
    if warnings and args.verbose:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
        print()
    
    if is_valid:
        print("✅ models.toml validation PASSED")
        if warnings:
            print(f"   ({len(warnings)} warnings - use --verbose to see them)")
    else:
        print("❌ models.toml validation FAILED")
        print(f"   Found {len(errors)} errors")
        if warnings:
            print(f"   Found {len(warnings)} warnings")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())