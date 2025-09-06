import pytest
"""
Configuration health checker
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

try:
    from ..health_models import ComponentHealth, HealthIssue, HealthCategory, Severity, HealthConfig
except ImportError:
    # Fallback imports - these will be defined in health_checker.py
    pass


class ConfigurationHealthChecker:
    """Checks the health of project configuration"""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_health(self) -> ComponentHealth:
        """Check configuration health"""
        issues = []
        metrics = {}
        
        # Discover configuration files
        config_files = self._discover_configuration_files()
        metrics["total_config_files"] = len(config_files)
        
        # Check for scattered configuration
        scattered_configs = self._check_scattered_configuration(config_files)
        metrics["scattered_configs"] = len(scattered_configs)
        
        if scattered_configs:
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.CONFIGURATION,
                title="Scattered Configuration Files",
                description=f"Found {len(scattered_configs)} configuration files outside config directory",
                affected_components=["configuration"],
                remediation_steps=[
                    "Move configuration files to config directory",
                    "Create unified configuration system",
                    "Update references to moved files"
                ],
                metadata={"scattered_files": scattered_configs[:10]}
            ))
        
        # Check for duplicate configurations
        duplicate_configs = self._check_duplicate_configurations(config_files)
        metrics["duplicate_configs"] = len(duplicate_configs)
        
        if duplicate_configs:
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.CONFIGURATION,
                title="Duplicate Configuration Settings",
                description=f"Found {len(duplicate_configs)} duplicate configuration keys",
                affected_components=["configuration"],
                remediation_steps=[
                    "Consolidate duplicate configuration settings",
                    "Create configuration hierarchy",
                    "Remove redundant config files"
                ],
                metadata={"duplicates": duplicate_configs[:5]}
            ))
        
        # Check configuration validation
        validation_issues = self._validate_configurations(config_files)
        metrics["validation_errors"] = len(validation_issues)
        
        if validation_issues:
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.CONFIGURATION,
                title="Configuration Validation Errors",
                description=f"Found {len(validation_issues)} configuration validation errors",
                affected_components=["configuration"],
                remediation_steps=[
                    "Fix configuration syntax errors",
                    "Validate configuration values",
                    "Add configuration schema validation"
                ],
                metadata={"validation_errors": validation_issues[:5]}
            ))
        
        # Check for missing essential configurations
        missing_configs = self._check_missing_configurations()
        metrics["missing_configs"] = len(missing_configs)
        
        if missing_configs:
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.CONFIGURATION,
                title="Missing Essential Configuration",
                description=f"Missing: {', '.join(missing_configs)}",
                affected_components=["configuration"],
                remediation_steps=[
                    f"Create {config}" for config in missing_configs
                ]
            ))
        
        # Check for security issues in configuration
        security_issues = self._check_configuration_security(config_files)
        metrics["security_issues"] = len(security_issues)
        
        if security_issues:
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.CONFIGURATION,
                title="Configuration Security Issues",
                description=f"Found {len(security_issues)} potential security issues",
                affected_components=["configuration"],
                remediation_steps=[
                    "Remove hardcoded secrets from configuration",
                    "Use environment variables for sensitive data",
                    "Add .env files to .gitignore"
                ],
                metadata={"security_issues": security_issues[:3]}
            ))
        
        # Check unified configuration system
        unified_config_status = self._check_unified_configuration()
        metrics.update(unified_config_status)
        
        if not unified_config_status.get("has_unified_config", False):
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.CONFIGURATION,
                title="No Unified Configuration System",
                description="Project lacks a unified configuration management system",
                affected_components=["configuration"],
                remediation_steps=[
                    "Implement unified configuration schema",
                    "Create configuration API",
                    "Migrate existing configurations"
                ]
            ))
        
        # Calculate score
        score = self._calculate_configuration_score(metrics, issues)
        status = self._determine_status(score)
        
        return ComponentHealth(
            component_name="configuration",
            category=HealthCategory.CONFIGURATION,
            score=score,
            status=status,
            issues=issues,
            metrics=metrics
        )
    
    def _discover_configuration_files(self) -> List[Path]:
        """Discover configuration files throughout the project"""
        config_files = []
        
        try:
            # Common configuration file patterns
            patterns = [
                "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg",
                ".env*", "config.*", "*config*", "settings.*"
            ]
            
            for pattern in patterns:
                config_files.extend(list(self.config.project_root.rglob(pattern)))
            
            # Filter out common non-config files
            filtered_files = []
            exclude_patterns = [
                "node_modules", "__pycache__", ".git", "venv", ".pytest_cache",
                "package.json", "package-lock.json", "tsconfig.json"
            ]
            
            for file_path in config_files:
                if not any(exclude in str(file_path) for exclude in exclude_patterns):
                    filtered_files.append(file_path)
            
            return filtered_files
            
        except Exception as e:
            self.logger.warning(f"Failed to discover configuration files: {e}")
            return []
    
    def _check_scattered_configuration(self, config_files: List[Path]) -> List[str]:
        """Check for configuration files outside the config directory"""
        scattered = []
        
        for config_file in config_files:
            relative_path = config_file.relative_to(self.config.project_root)
            
            # Skip files that are legitimately outside config directory
            if (not str(relative_path).startswith(str(self.config.config_directory)) and
                not str(relative_path).startswith('.') and  # Hidden files like .env
                config_file.name not in ['package.json', 'tsconfig.json']):  # Legitimate project files
                
                scattered.append(str(relative_path))
        
        return scattered
    
    def _check_duplicate_configurations(self, config_files: List[Path]) -> List[Dict[str, Any]]:
        """Check for duplicate configuration keys across files"""
        duplicates = []
        all_keys = {}
        
        for config_file in config_files:
            try:
                keys = self._extract_config_keys(config_file)
                
                for key in keys:
                    if key in all_keys:
                        duplicates.append({
                            "key": key,
                            "files": [all_keys[key], str(config_file.relative_to(self.config.project_root))]
                        })
                    else:
                        all_keys[key] = str(config_file.relative_to(self.config.project_root))
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract keys from {config_file}: {e}")
        
        return duplicates
    
    def _extract_config_keys(self, config_file: Path) -> Set[str]:
        """Extract configuration keys from a file"""
        keys = set()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as JSON
            if config_file.suffix.lower() == '.json':
                data = json.loads(content)
                keys.update(self._flatten_dict_keys(data))
            
            # Try to parse as YAML
            elif config_file.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    keys.update(self._flatten_dict_keys(data))
            
            # For other files, extract key-like patterns
            else:
                import re
                # Look for key=value or key: value patterns
                key_patterns = [
                    r'^([A-Z_][A-Z0-9_]*)\s*=',  # Environment variables
                    r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]',  # General key patterns
                ]
                
                for line in content.split('\n'):
                    for pattern in key_patterns:
                        match = re.match(pattern, line.strip())
                        if match:
                            keys.add(match.group(1))
        
        except Exception as e:
            self.logger.warning(f"Failed to parse {config_file}: {e}")
        
        return keys
    
    def _flatten_dict_keys(self, data: Dict[str, Any], prefix: str = "") -> Set[str]:
        """Flatten nested dictionary keys"""
        keys = set()
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            
            if isinstance(value, dict):
                keys.update(self._flatten_dict_keys(value, full_key))
        
        return keys
    
    def _validate_configurations(self, config_files: List[Path]) -> List[Dict[str, str]]:
        """Validate configuration files for syntax errors"""
        validation_errors = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Validate JSON files
                if config_file.suffix.lower() == '.json':
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        validation_errors.append({
                            "file": str(config_file.relative_to(self.config.project_root)),
                            "error": f"JSON syntax error: {e.msg}"
                        })
                
                # Validate YAML files
                elif config_file.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        yaml.safe_load(content)
                    except yaml.YAMLError as e:
                        validation_errors.append({
                            "file": str(config_file.relative_to(self.config.project_root)),
                            "error": f"YAML syntax error: {str(e)}"
                        })
                        
            except Exception as e:
                validation_errors.append({
                    "file": str(config_file.relative_to(self.config.project_root)),
                    "error": f"Failed to read file: {str(e)}"
                })
        
        return validation_errors
    
    def _check_missing_configurations(self) -> List[str]:
        """Check for missing essential configuration files"""
        essential_configs = [
            "config/base.yaml",
            "config/environments/development.yaml",
            ".env.example"
        ]
        
        missing = []
        
        for config_path in essential_configs:
            full_path = self.config.project_root / config_path
            if not full_path.exists():
                missing.append(config_path)
        
        return missing
    
    def _check_configuration_security(self, config_files: List[Path]) -> List[Dict[str, str]]:
        """Check for security issues in configuration files"""
        security_issues = []
        
        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            r'password\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            r'secret\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            r'key\s*[:=]\s*["\']?[A-Za-z0-9+/]{20,}["\']?',
            r'token\s*[:=]\s*["\']?[A-Za-z0-9+/]{20,}["\']?',
        ]
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        security_issues.append({
                            "file": str(config_file.relative_to(self.config.project_root)),
                            "issue": "Potential hardcoded secret detected",
                            "pattern": pattern
                        })
                        break  # One issue per file is enough
                        
            except Exception as e:
                self.logger.warning(f"Failed to check security for {config_file}: {e}")
        
        return security_issues
    
    def _check_unified_configuration(self) -> Dict[str, Any]:
        """Check if unified configuration system exists"""
        unified_config_file = self.config.project_root / "config" / "unified-config.yaml"
        config_api_file = self.config.project_root / "tools" / "config_manager" / "config_api.py"
        
        return {
            "has_unified_config": unified_config_file.exists(),
            "has_config_api": config_api_file.exists(),
            "unified_config_path": str(unified_config_file) if unified_config_file.exists() else None
        }
    
    def _calculate_configuration_score(self, metrics: Dict[str, Any], issues: List[HealthIssue]) -> float:
        """Calculate configuration health score"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                base_score -= 25
            elif issue.severity == Severity.HIGH:
                base_score -= 20
            elif issue.severity == Severity.MEDIUM:
                base_score -= 15
            elif issue.severity == Severity.LOW:
                base_score -= 5
        
        # Bonus points for good practices
        if metrics.get("has_unified_config", False):
            base_score += 15
        if metrics.get("has_config_api", False):
            base_score += 10
        
        # Penalty for scattered configurations
        scattered_count = metrics.get("scattered_configs", 0)
        if scattered_count > 10:
            base_score -= 20
        elif scattered_count > 5:
            base_score -= 10
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_status(self, score: float) -> str:
        """Determine health status from score"""
        if score >= self.config.warning_threshold:
            return "healthy"
        elif score >= self.config.critical_threshold:
            return "warning"
        else:
            return "critical"