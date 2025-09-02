#!/usr/bin/env python3
"""
Pre-commit hook for configuration validation.
Validates configuration files against schemas and checks for consistency.
"""

import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_yaml_file(file_path: Path) -> Optional[Dict[Any, Any]]:
    """Load a YAML file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def load_json_file(file_path: Path) -> Optional[Dict[Any, Any]]:
    """Load a JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def validate_yaml_syntax(file_path: Path) -> List[str]:
    """Validate YAML syntax."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        issues.append(f"YAML syntax error in {file_path}: {e}")
    except Exception as e:
        issues.append(f"Error reading {file_path}: {e}")
    
    return issues


def validate_json_syntax(file_path: Path) -> List[str]:
    """Validate JSON syntax."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"JSON syntax error in {file_path}: {e}")
    except Exception as e:
        issues.append(f"Error reading {file_path}: {e}")
    
    return issues


def check_required_fields(config: Dict[Any, Any], file_path: Path) -> List[str]:
    """Check for required configuration fields."""
    issues = []
    
    # Define required fields for different config types
    if "system" in config:
        # Main configuration file
        required_fields = ["system.name", "system.version", "backend"]
        
        for field in required_fields:
            keys = field.split(".")
            current = config
            
            try:
                for key in keys:
                    current = current[key]
            except (KeyError, TypeError):
                issues.append(f"Missing required field '{field}' in {file_path}")
    
    return issues


def check_security_settings(config: Dict[Any, Any], file_path: Path) -> List[str]:
    """Check for security-related configuration issues."""
    issues = []
    
    # Check for insecure settings
    if "security" in config:
        security = config["security"]
        
        # Check authentication settings
        if "authentication" in security:
            auth = security["authentication"]
            if auth.get("enabled") is False and "production" in str(file_path):
                issues.append(f"Authentication disabled in production config: {file_path}")
        
        # Check CORS settings
        if "cors" in security:
            cors = security["cors"]
            if cors.get("allow_credentials") is True and "production" in str(file_path):
                origins = config.get("backend", {}).get("api", {}).get("cors_origins", [])
                if "*" in origins:
                    issues.append(f"Insecure CORS configuration in production: {file_path}")
    
    # Check for hardcoded secrets
    config_str = json.dumps(config, default=str).lower()
    secret_indicators = ["password", "secret", "key", "token"]
    
    for indicator in secret_indicators:
        if f'"{indicator}": "' in config_str and len(config_str.split(f'"{indicator}": "')[1].split('"')[0]) > 10:
            issues.append(f"Potential hardcoded secret ({indicator}) in {file_path}")
    
    return issues


def validate_environment_consistency(file_paths: List[Path]) -> List[str]:
    """Validate consistency between environment configurations."""
    issues = []
    
    # Load base config
    base_config_path = Path("config/base.yaml")
    if not base_config_path.exists():
        return ["Base configuration file not found: config/base.yaml"]
    
    base_config = load_yaml_file(base_config_path)
    if not base_config:
        return ["Could not load base configuration"]
    
    # Check environment configs
    env_configs = {}
    for file_path in file_paths:
        if "environments/" in str(file_path):
            env_name = file_path.stem
            env_config = load_yaml_file(file_path)
            if env_config:
                env_configs[env_name] = env_config
    
    # Validate environment-specific requirements
    for env_name, env_config in env_configs.items():
        if env_name == "production":
            # Production-specific checks
            if env_config.get("system", {}).get("debug") is True:
                issues.append(f"Debug mode enabled in production config")
            
            security = env_config.get("security", {})
            if security.get("authentication", {}).get("enabled") is False:
                issues.append(f"Authentication disabled in production config")
    
    return issues


def main(file_paths: List[str]) -> int:
    """Main pre-commit hook function."""
    print("🔍 Running configuration validation...")
    
    all_issues = []
    paths = [Path(p) for p in file_paths]
    
    for file_path in paths:
        print(f"  Checking {file_path}...")
        
        # Syntax validation
        if file_path.suffix in ['.yaml', '.yml']:
            all_issues.extend(validate_yaml_syntax(file_path))
            
            # Load and validate content
            config = load_yaml_file(file_path)
            if config:
                all_issues.extend(check_required_fields(config, file_path))
                all_issues.extend(check_security_settings(config, file_path))
        
        elif file_path.suffix == '.json':
            all_issues.extend(validate_json_syntax(file_path))
            
            # Load and validate content
            config = load_json_file(file_path)
            if config:
                all_issues.extend(check_required_fields(config, file_path))
                all_issues.extend(check_security_settings(config, file_path))
    
    # Cross-file validation
    all_issues.extend(validate_environment_consistency(paths))
    
    if all_issues:
        print("❌ Configuration validation failed:")
        for issue in all_issues:
            print(f"  • {issue}")
        
        print("\n💡 Recommendations:")
        print("  • Fix syntax errors in configuration files")
        print("  • Add missing required fields")
        print("  • Review security settings, especially for production")
        print("  • Remove hardcoded secrets and use environment variables")
        
        return 1
    
    print("✅ Configuration validation passed!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pre_commit_config_validation.py <file1> [file2] ...")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1:]))