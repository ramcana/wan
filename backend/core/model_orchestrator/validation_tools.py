"""
Validation Tools - Comprehensive validation for manifests and configurations.

This module provides advanced validation tools for model manifests,
configuration files, and system compatibility checks.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import logging

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions

from .model_registry import ModelRegistry, ModelSpec, FileSpec
from .exceptions import ValidationError, ManifestValidationError


logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue with severity and context."""
    
    severity: str  # "error", "warning", "info"
    category: str  # "schema", "security", "performance", "compatibility"
    message: str
    context: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of validation issue."""
        parts = [f"[{self.severity.upper()}]"]
        if self.context:
            parts.append(f"({self.context})")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, int]
    
    def __post_init__(self):
        """Calculate summary statistics."""
        self.summary = {
            "total": len(self.issues),
            "errors": len([i for i in self.issues if i.severity == "error"]),
            "warnings": len([i for i in self.issues if i.severity == "warning"]),
            "info": len([i for i in self.issues if i.severity == "info"])
        }
        
        # Report is valid if no errors
        self.valid = self.summary["errors"] == 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the report."""
        self.issues.append(issue)
        # Recalculate summary
        self.__post_init__()
    
    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [i for i in self.issues if i.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get issues filtered by category."""
        return [i for i in self.issues if i.category == category]


class ManifestSchemaValidator:
    """Validator for manifest schema and structure."""
    
    SUPPORTED_SCHEMA_VERSIONS = ["1"]
    REQUIRED_MODEL_FIELDS = [
        "description", "version", "variants", "default_variant", 
        "files", "sources"
    ]
    REQUIRED_FILE_FIELDS = ["path", "size", "sha256"]
    
    def __init__(self):
        """Initialize the schema validator."""
        self.logger = logging.getLogger(__name__ + ".ManifestSchemaValidator")
    
    def validate_manifest_schema(self, manifest_path: str) -> ValidationReport:
        """
        Validate manifest schema and structure.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            ValidationReport with schema validation results
        """
        report = ValidationReport(valid=True, issues=[], summary={})
        
        try:
            # Load and parse manifest
            with open(manifest_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Validate top-level structure
            self._validate_top_level_structure(data, report)
            
            # Validate schema version
            self._validate_schema_version(data, report)
            
            # Validate models section
            if "models" in data:
                self._validate_models_section(data["models"], report)
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Failed to parse manifest: {e}",
                context="manifest_parsing"
            ))
        
        return report
    
    def _validate_top_level_structure(self, data: Dict[str, Any], report: ValidationReport) -> None:
        """Validate top-level manifest structure."""
        required_fields = ["schema_version", "models"]
        
        for field in required_fields:
            if field not in data:
                report.add_issue(ValidationIssue(
                    severity="error",
                    category="schema",
                    message=f"Missing required top-level field: {field}",
                    context="top_level_structure"
                ))
    
    def _validate_schema_version(self, data: Dict[str, Any], report: ValidationReport) -> None:
        """Validate schema version."""
        schema_version = str(data.get("schema_version", ""))
        
        if not schema_version:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message="Schema version is required",
                context="schema_version"
            ))
            return
        
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Unsupported schema version: {schema_version}",
                context="schema_version",
                suggestion=f"Supported versions: {', '.join(self.SUPPORTED_SCHEMA_VERSIONS)}"
            ))
    
    def _validate_models_section(self, models: Dict[str, Any], report: ValidationReport) -> None:
        """Validate models section."""
        if not models:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message="Models section cannot be empty",
                context="models_section"
            ))
            return
        
        for model_id, model_data in models.items():
            self._validate_model_entry(model_id, model_data, report)
    
    def _validate_model_entry(self, model_id: str, model_data: Dict[str, Any], report: ValidationReport) -> None:
        """Validate individual model entry."""
        context = f"model:{model_id}"
        
        # Validate model ID format
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*@[0-9]+\.[0-9]+\.[0-9]+$", model_id):
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Invalid model ID format: {model_id}",
                context=context,
                suggestion="Use format: model-name@version (e.g., t2v-A14B@2.2.0)"
            ))
        
        # Validate required fields
        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in model_data:
                report.add_issue(ValidationIssue(
                    severity="error",
                    category="schema",
                    message=f"Missing required field: {field}",
                    context=context
                ))
        
        # Validate variants
        if "variants" in model_data and "default_variant" in model_data:
            variants = model_data["variants"]
            default_variant = model_data["default_variant"]
            
            if not isinstance(variants, list) or not variants:
                report.add_issue(ValidationIssue(
                    severity="error",
                    category="schema",
                    message="Variants must be a non-empty list",
                    context=context
                ))
            elif default_variant not in variants:
                report.add_issue(ValidationIssue(
                    severity="error",
                    category="schema",
                    message=f"Default variant '{default_variant}' not in variants list",
                    context=context
                ))
        
        # Validate files
        if "files" in model_data:
            self._validate_files_section(model_data["files"], report, context)
        
        # Validate sources
        if "sources" in model_data:
            self._validate_sources_section(model_data["sources"], report, context)
    
    def _validate_files_section(self, files: List[Dict[str, Any]], report: ValidationReport, context: str) -> None:
        """Validate files section."""
        if not files:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message="Files section cannot be empty",
                context=context
            ))
            return
        
        for i, file_data in enumerate(files):
            file_context = f"{context}:file[{i}]"
            
            # Validate required fields
            for field in self.REQUIRED_FILE_FIELDS:
                if field not in file_data:
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="schema",
                        message=f"Missing required file field: {field}",
                        context=file_context
                    ))
            
            # Validate file path
            if "path" in file_data:
                path = file_data["path"]
                if not isinstance(path, str) or not path:
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="schema",
                        message="File path must be a non-empty string",
                        context=file_context
                    ))
                elif ".." in Path(path).parts:
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="security",
                        message=f"Path traversal detected in file path: {path}",
                        context=file_context
                    ))
            
            # Validate file size
            if "size" in file_data:
                size = file_data["size"]
                if not isinstance(size, int) or size < 0:
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="schema",
                        message="File size must be a non-negative integer",
                        context=file_context
                    ))
            
            # Validate SHA256
            if "sha256" in file_data:
                sha256 = file_data["sha256"]
                if not isinstance(sha256, str) or len(sha256) != 64:
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="schema",
                        message="SHA256 must be a 64-character hex string",
                        context=file_context
                    ))
                elif not re.match(r"^[a-fA-F0-9]{64}$", sha256):
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="schema",
                        message="SHA256 contains invalid characters",
                        context=file_context
                    ))
    
    def _validate_sources_section(self, sources: Dict[str, Any], report: ValidationReport, context: str) -> None:
        """Validate sources section."""
        if "priority" not in sources:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message="Sources must contain 'priority' field",
                context=context
            ))
            return
        
        priority = sources["priority"]
        if not isinstance(priority, list) or not priority:
            report.add_issue(ValidationIssue(
                severity="error",
                category="schema",
                message="Sources priority must be a non-empty list",
                context=context
            ))


class SecurityValidator:
    """Validator for security-related issues in manifests."""
    
    WINDOWS_RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    
    def __init__(self):
        """Initialize the security validator."""
        self.logger = logging.getLogger(__name__ + ".SecurityValidator")
    
    def validate_security(self, manifest_path: str) -> ValidationReport:
        """
        Validate security aspects of manifest.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            ValidationReport with security validation results
        """
        report = ValidationReport(valid=True, issues=[], summary={})
        
        try:
            registry = ModelRegistry(manifest_path)
            
            for model_id in registry.list_models():
                model_spec = registry.spec(model_id)
                self._validate_model_security(model_spec, report)
                
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity="error",
                category="security",
                message=f"Failed to load manifest for security validation: {e}",
                context="security_validation"
            ))
        
        return report
    
    def _validate_model_security(self, model_spec: ModelSpec, report: ValidationReport) -> None:
        """Validate security aspects of a model specification."""
        context = f"model:{model_spec.model_id}"
        
        # Check for path traversal in file paths
        for file_spec in model_spec.files:
            self._validate_file_path_security(file_spec, report, context)
        
        # Check for suspicious file patterns
        self._validate_file_patterns(model_spec, report, context)
        
        # Check source URLs for security issues
        self._validate_source_urls(model_spec, report, context)
    
    def _validate_file_path_security(self, file_spec: FileSpec, report: ValidationReport, context: str) -> None:
        """Validate file path for security issues."""
        path = Path(file_spec.path)
        
        # Check for directory traversal
        if ".." in path.parts:
            report.add_issue(ValidationIssue(
                severity="error",
                category="security",
                message=f"Directory traversal detected: {file_spec.path}",
                context=context,
                suggestion="Use relative paths without '..' components"
            ))
        
        # Check for absolute paths
        if path.is_absolute():
            report.add_issue(ValidationIssue(
                severity="error",
                category="security",
                message=f"Absolute path not allowed: {file_spec.path}",
                context=context,
                suggestion="Use relative paths only"
            ))
        
        # Check for Windows reserved names
        for part in path.parts:
            name_without_ext = part.split('.')[0].upper()
            if name_without_ext in self.WINDOWS_RESERVED_NAMES:
                report.add_issue(ValidationIssue(
                    severity="warning",
                    category="security",
                    message=f"Windows reserved name in path: {file_spec.path}",
                    context=context,
                    suggestion="Avoid Windows reserved names for cross-platform compatibility"
                ))
    
    def _validate_file_patterns(self, model_spec: ModelSpec, report: ValidationReport, context: str) -> None:
        """Validate file patterns for suspicious content."""
        suspicious_extensions = {".exe", ".bat", ".cmd", ".ps1", ".sh", ".scr", ".com"}
        
        for file_spec in model_spec.files:
            file_path = Path(file_spec.path)
            
            # Check for executable file extensions
            if file_path.suffix.lower() in suspicious_extensions:
                report.add_issue(ValidationIssue(
                    severity="warning",
                    category="security",
                    message=f"Executable file detected: {file_spec.path}",
                    context=context,
                    suggestion="Verify that executable files are necessary and safe"
                ))
            
            # Check for hidden files (starting with .)
            if file_path.name.startswith('.') and file_path.name not in {'.gitkeep', '.gitignore'}:
                report.add_issue(ValidationIssue(
                    severity="info",
                    category="security",
                    message=f"Hidden file detected: {file_spec.path}",
                    context=context
                ))
    
    def _validate_source_urls(self, model_spec: ModelSpec, report: ValidationReport, context: str) -> None:
        """Validate source URLs for security issues."""
        for source_url in model_spec.sources:
            # Check for HTTP URLs (should prefer HTTPS)
            if source_url.startswith("http://"):
                report.add_issue(ValidationIssue(
                    severity="warning",
                    category="security",
                    message=f"Insecure HTTP URL: {source_url}",
                    context=context,
                    suggestion="Use HTTPS URLs for secure downloads"
                ))
            
            # Check for suspicious domains
            suspicious_patterns = [
                r"\.tk$", r"\.ml$", r"\.ga$", r"\.cf$",  # Free TLD domains
                r"bit\.ly", r"tinyurl\.com",  # URL shorteners
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, source_url, re.IGNORECASE):
                    report.add_issue(ValidationIssue(
                        severity="warning",
                        category="security",
                        message=f"Potentially suspicious URL: {source_url}",
                        context=context,
                        suggestion="Verify URL authenticity and use trusted domains"
                    ))


class PerformanceValidator:
    """Validator for performance-related issues in manifests."""
    
    def __init__(self):
        """Initialize the performance validator."""
        self.logger = logging.getLogger(__name__ + ".PerformanceValidator")
    
    def validate_performance(self, manifest_path: str) -> ValidationReport:
        """
        Validate performance aspects of manifest.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            ValidationReport with performance validation results
        """
        report = ValidationReport(valid=True, issues=[], summary={})
        
        try:
            registry = ModelRegistry(manifest_path)
            
            for model_id in registry.list_models():
                model_spec = registry.spec(model_id)
                self._validate_model_performance(model_spec, report)
                
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity="error",
                category="performance",
                message=f"Failed to load manifest for performance validation: {e}",
                context="performance_validation"
            ))
        
        return report
    
    def _validate_model_performance(self, model_spec: ModelSpec, report: ValidationReport) -> None:
        """Validate performance aspects of a model specification."""
        context = f"model:{model_spec.model_id}"
        
        # Check for large file sizes
        total_size = sum(file_spec.size for file_spec in model_spec.files)
        
        if total_size > 100 * 1024 * 1024 * 1024:  # 100GB
            report.add_issue(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Very large model size: {total_size / (1024**3):.1f}GB",
                context=context,
                suggestion="Consider model compression or sharding for better download performance"
            ))
        elif total_size > 50 * 1024 * 1024 * 1024:  # 50GB
            report.add_issue(ValidationIssue(
                severity="info",
                category="performance",
                message=f"Large model size: {total_size / (1024**3):.1f}GB",
                context=context
            ))
        
        # Check for too many small files
        small_files = [f for f in model_spec.files if f.size < 1024 * 1024]  # < 1MB
        if len(small_files) > 100:
            report.add_issue(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Many small files detected: {len(small_files)} files < 1MB",
                context=context,
                suggestion="Consider bundling small files for better download performance"
            ))
        
        # Check VRAM estimation
        if model_spec.vram_estimation:
            vram_gb = model_spec.vram_estimation.base_vram_gb
            if vram_gb > 24:
                report.add_issue(ValidationIssue(
                    severity="warning",
                    category="performance",
                    message=f"High VRAM requirement: {vram_gb}GB",
                    context=context,
                    suggestion="Consider providing lower precision variants"
                ))


class CompatibilityValidator:
    """Validator for compatibility issues across platforms and configurations."""
    
    def __init__(self):
        """Initialize the compatibility validator."""
        self.logger = logging.getLogger(__name__ + ".CompatibilityValidator")
    
    def validate_compatibility(self, manifest_path: str) -> ValidationReport:
        """
        Validate compatibility aspects of manifest.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            ValidationReport with compatibility validation results
        """
        report = ValidationReport(valid=True, issues=[], summary={})
        
        try:
            registry = ModelRegistry(manifest_path)
            
            # Check for case collisions
            self._validate_case_sensitivity(registry, report)
            
            # Check path length limits
            self._validate_path_lengths(registry, report)
            
            # Check for platform-specific issues
            self._validate_platform_compatibility(registry, report)
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity="error",
                category="compatibility",
                message=f"Failed to load manifest for compatibility validation: {e}",
                context="compatibility_validation"
            ))
        
        return report
    
    def _validate_case_sensitivity(self, registry: ModelRegistry, report: ValidationReport) -> None:
        """Validate for case sensitivity issues."""
        all_paths = []
        
        for model_id in registry.list_models():
            model_spec = registry.spec(model_id)
            model_paths = [f.path for f in model_spec.files]
            
            # Check for case collisions within model
            seen_lower = set()
            for path in model_paths:
                lower_path = path.lower()
                if lower_path in seen_lower:
                    report.add_issue(ValidationIssue(
                        severity="error",
                        category="compatibility",
                        message=f"Case collision detected: {path}",
                        context=f"model:{model_id}",
                        suggestion="Ensure unique file names on case-insensitive filesystems"
                    ))
                seen_lower.add(lower_path)
            
            all_paths.extend(model_paths)
    
    def _validate_path_lengths(self, registry: ModelRegistry, report: ValidationReport) -> None:
        """Validate path lengths for Windows compatibility."""
        for model_id in registry.list_models():
            model_spec = registry.spec(model_id)
            
            for file_spec in model_spec.files:
                # Estimate full path length (assuming reasonable base path)
                estimated_full_path = f"C:\\Models\\wan22\\{model_id}\\{file_spec.path}"
                
                if len(estimated_full_path) > 260:
                    report.add_issue(ValidationIssue(
                        severity="warning",
                        category="compatibility",
                        message=f"Path may exceed Windows limit: {file_spec.path}",
                        context=f"model:{model_id}",
                        suggestion="Enable long path support on Windows or use shorter paths"
                    ))
    
    def _validate_platform_compatibility(self, registry: ModelRegistry, report: ValidationReport) -> None:
        """Validate platform-specific compatibility issues."""
        for model_id in registry.list_models():
            model_spec = registry.spec(model_id)
            
            for file_spec in model_spec.files:
                path = file_spec.path
                
                # Check for characters problematic on Windows
                problematic_chars = '<>:"|?*'
                for char in problematic_chars:
                    if char in path:
                        report.add_issue(ValidationIssue(
                            severity="error",
                            category="compatibility",
                            message=f"Invalid character '{char}' in path: {path}",
                            context=f"model:{model_id}",
                            suggestion="Remove invalid characters for Windows compatibility"
                        ))


class ComprehensiveValidator:
    """Comprehensive validator that runs all validation checks."""
    
    def __init__(self):
        """Initialize the comprehensive validator."""
        self.schema_validator = ManifestSchemaValidator()
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
        self.compatibility_validator = CompatibilityValidator()
        self.logger = logging.getLogger(__name__ + ".ComprehensiveValidator")
    
    def validate_manifest(
        self, 
        manifest_path: str,
        include_schema: bool = True,
        include_security: bool = True,
        include_performance: bool = True,
        include_compatibility: bool = True
    ) -> ValidationReport:
        """
        Run comprehensive validation on a manifest file.
        
        Args:
            manifest_path: Path to manifest file
            include_schema: Whether to include schema validation
            include_security: Whether to include security validation
            include_performance: Whether to include performance validation
            include_compatibility: Whether to include compatibility validation
            
        Returns:
            Combined ValidationReport with all validation results
        """
        combined_report = ValidationReport(valid=True, issues=[], summary={})
        
        # Run schema validation
        if include_schema:
            schema_report = self.schema_validator.validate_manifest_schema(manifest_path)
            combined_report.issues.extend(schema_report.issues)
        
        # Only run other validations if schema is valid
        if combined_report.summary.get("errors", 0) == 0:
            
            # Run security validation
            if include_security:
                security_report = self.security_validator.validate_security(manifest_path)
                combined_report.issues.extend(security_report.issues)
            
            # Run performance validation
            if include_performance:
                performance_report = self.performance_validator.validate_performance(manifest_path)
                combined_report.issues.extend(performance_report.issues)
            
            # Run compatibility validation
            if include_compatibility:
                compatibility_report = self.compatibility_validator.validate_compatibility(manifest_path)
                combined_report.issues.extend(compatibility_report.issues)
        
        # Recalculate summary
        combined_report.__post_init__()
        
        return combined_report
    
    def validate_file_integrity(self, file_path: str, expected_sha256: str, expected_size: int) -> ValidationReport:
        """
        Validate file integrity against expected values.
        
        Args:
            file_path: Path to file to validate
            expected_sha256: Expected SHA256 hash
            expected_size: Expected file size
            
        Returns:
            ValidationReport with integrity validation results
        """
        report = ValidationReport(valid=True, issues=[], summary={})
        
        file_obj = Path(file_path)
        if not file_obj.exists():
            report.add_issue(ValidationIssue(
                severity="error",
                category="integrity",
                message=f"File not found: {file_path}",
                context="file_integrity"
            ))
            return report
        
        # Check file size
        actual_size = file_obj.stat().st_size
        if actual_size != expected_size:
            report.add_issue(ValidationIssue(
                severity="error",
                category="integrity",
                message=f"Size mismatch: expected {expected_size}, got {actual_size}",
                context=f"file:{file_path}"
            ))
        
        # Check SHA256 hash
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_sha256 = sha256_hash.hexdigest()
            if actual_sha256 != expected_sha256:
                report.add_issue(ValidationIssue(
                    severity="error",
                    category="integrity",
                    message=f"SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}",
                    context=f"file:{file_path}"
                ))
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity="error",
                category="integrity",
                message=f"Failed to compute SHA256: {e}",
                context=f"file:{file_path}"
            ))
        
        # Recalculate summary
        report.__post_init__()
        
        return report