import pytest
#!/usr/bin/env python3
"""
Configuration Landscape Analyzer

Scans the project for all configuration files and analyzes their relationships,
dependencies, and potential conflicts.
"""

import json
import yaml
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import configparser


@dataclass
class ConfigFile:
    """Represents a configuration file in the project."""
    path: Path
    file_type: str
    size: int
    content: Optional[Dict[str, Any]] = None
    dependencies: List[str] = None
    referenced_by: List[str] = None
    settings: Set[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.referenced_by is None:
            self.referenced_by = []
        if self.settings is None:
            self.settings = set()


@dataclass
class ConfigConflict:
    """Represents a conflict between configuration settings."""
    setting_name: str
    files: List[str]
    values: List[Any]
    severity: str  # 'high', 'medium', 'low'


@dataclass
class ConfigDependency:
    """Represents a dependency relationship between configs."""
    source_file: str
    target_file: str
    dependency_type: str  # 'import', 'reference', 'override'


@dataclass
class ConfigAnalysisReport:
    """Complete analysis report of the configuration landscape."""
    total_files: int
    config_files: List[ConfigFile]
    conflicts: List[ConfigConflict]
    dependencies: List[ConfigDependency]
    duplicate_settings: Dict[str, List[str]]
    recommendations: List[str]
    migration_plan: Dict[str, Any]


class ConfigLandscapeAnalyzer:
    """Analyzes the configuration landscape of a project."""
    
    CONFIG_EXTENSIONS = {
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.env': 'env',
        '.toml': 'toml'
    }
    
    # Patterns to identify configuration files by name
    CONFIG_PATTERNS = [
        r'config.*\.(json|yaml|yml|ini|cfg|conf|env|toml)$',
        r'.*config\.(json|yaml|yml|ini|cfg|conf|env|toml)$',
        r'settings.*\.(json|yaml|yml|ini|cfg|conf|env|toml)$',
        r'.*\.env$',
        r'docker-compose\.ya?ml$',
        r'pytest\.ini$',
        r'setup\.cfg$',
        r'pyproject\.toml$'
    ]
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.config_files: List[ConfigFile] = []
        self.all_settings: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        
    def scan_project(self) -> List[ConfigFile]:
        """Scan the project for all configuration files."""
        config_files = []
        
        # Directories to skip
        skip_dirs = {
            'node_modules', 'venv', '__pycache__', '.git',
            '.pytest_cache', 'dist', 'build', 'reports',
            'logs'
        }
        
        # Walk through all directories
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-config directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and (not d.startswith('.') or d in ['.github', '.kiro'])]
            
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                relative_path = file_path.relative_to(self.project_root)
                
                # Skip files in excluded directories that might have been missed
                if any(skip_dir in str(relative_path) for skip_dir in skip_dirs):
                    continue
                
                # Check if it's a configuration file
                if self._is_config_file(file_path):
                    config_file = self._analyze_config_file(file_path, relative_path)
                    if config_file:
                        config_files.append(config_file)
        
        self.config_files = config_files
        return config_files
    
    def _is_config_file(self, file_path: Path) -> bool:
        """Determine if a file is a configuration file."""
        # Check by extension
        if file_path.suffix.lower() in self.CONFIG_EXTENSIONS:
            return True
        
        # Check by pattern
        filename = file_path.name.lower()
        for pattern in self.CONFIG_PATTERNS:
            if re.match(pattern, filename):
                return True
        
        return False
    
    def _analyze_config_file(self, file_path: Path, relative_path: Path) -> Optional[ConfigFile]:
        """Analyze a single configuration file."""
        try:
            file_type = self._determine_file_type(file_path)
            size = file_path.stat().st_size
            
            config_file = ConfigFile(
                path=relative_path,
                file_type=file_type,
                size=size
            )
            
            # Load and parse content
            content = self._load_config_content(file_path, file_type)
            if content is not None:
                config_file.content = content
                config_file.settings = self._extract_settings(content)
                
                # Track all settings for conflict detection
                for setting in config_file.settings:
                    self.all_settings[setting].append((str(relative_path), content.get(setting)))
            
            return config_file
            
        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return None
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the configuration file type."""
        suffix = file_path.suffix.lower()
        if suffix in self.CONFIG_EXTENSIONS:
            return self.CONFIG_EXTENSIONS[suffix]
        
        # Special cases
        filename = file_path.name.lower()
        if filename.endswith('.env'):
            return 'env'
        elif 'docker-compose' in filename:
            return 'yaml'
        elif filename in ['pytest.ini', 'setup.cfg']:
            return 'ini'
        elif filename == 'pyproject.toml':
            return 'toml'
        
        return 'unknown'
    
    def _load_config_content(self, file_path: Path, file_type: str) -> Optional[Dict[str, Any]]:
        """Load configuration file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_type == 'json':
                return json.loads(content)
            elif file_type == 'yaml':
                return yaml.safe_load(content) or {}
            elif file_type == 'ini':
                parser = configparser.ConfigParser()
                parser.read_string(content)
                return {section: dict(parser[section]) for section in parser.sections()}
            elif file_type == 'env':
                return self._parse_env_file(content)
            elif file_type == 'toml':
                try:
                    import tomllib
                    return tomllib.loads(content)
                except ImportError:
                    print(f"Warning: TOML support not available for {file_path}")
                    return None
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return None
    
    def _parse_env_file(self, content: str) -> Dict[str, str]:
        """Parse environment file content."""
        env_vars = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')
        return env_vars
    
    def _extract_settings(self, content: Dict[str, Any], prefix: str = '') -> Set[str]:
        """Extract all setting keys from configuration content."""
        settings = set()
        
        if not isinstance(content, dict):
            return settings
        
        for key, value in content.items():
            full_key = f"{prefix}.{key}" if prefix else key
            settings.add(full_key)
            
            if isinstance(value, dict):
                settings.update(self._extract_settings(value, full_key))
        
        return settings
    
    def analyze_dependencies(self) -> List[ConfigDependency]:
        """Analyze dependencies between configuration files."""
        dependencies = []
        
        # Look for file references in configuration content
        for config_file in self.config_files:
            if not config_file.content:
                continue
            
            # Search for references to other config files
            content_str = json.dumps(config_file.content, default=str)
            
            for other_config in self.config_files:
                if other_config.path == config_file.path:
                    continue
                
                # Check if this config references another
                other_name = other_config.path.name
                if other_name in content_str:
                    dependencies.append(ConfigDependency(
                        source_file=str(config_file.path),
                        target_file=str(other_config.path),
                        dependency_type='reference'
                    ))
        
        return dependencies
    
    def detect_conflicts(self) -> List[ConfigConflict]:
        """Detect conflicts between configuration settings."""
        conflicts = []
        
        for setting_name, file_values in self.all_settings.items():
            if len(file_values) > 1:
                # Check if values are different
                values = [value for _, value in file_values]
                # Convert all values to strings for comparison, handling None and complex types
                str_values = []
                for v in values:
                    if v is None:
                        str_values.append('null')
                    elif isinstance(v, (dict, list)):
                        str_values.append(json.dumps(v, sort_keys=True))
                    else:
                        str_values.append(str(v))
                
                unique_values = list(set(str_values))
                
                if len(unique_values) > 1:
                    severity = self._determine_conflict_severity(setting_name, values)
                    conflicts.append(ConfigConflict(
                        setting_name=setting_name,
                        files=[file_path for file_path, _ in file_values],
                        values=values,
                        severity=severity
                    ))
        
        return conflicts
    
    def _determine_conflict_severity(self, setting_name: str, values: List[Any]) -> str:
        """Determine the severity of a configuration conflict."""
        # Convert setting_name to string if it's not already
        setting_name_str = str(setting_name).lower()
        
        # High severity for critical settings
        critical_settings = ['port', 'host', 'database_url', 'secret_key', 'api_key']
        if any(critical in setting_name_str for critical in critical_settings):
            return 'high'
        
        # Medium severity for environment-specific settings
        env_settings = ['debug', 'log_level', 'environment', 'mode']
        if any(env in setting_name_str for env in env_settings):
            return 'medium'
        
        return 'low'
    
    def find_duplicate_settings(self) -> Dict[str, List[str]]:
        """Find duplicate settings across configuration files."""
        duplicates = {}
        
        for setting_name, file_values in self.all_settings.items():
            if len(file_values) > 1:
                files = [file_path for file_path, _ in file_values]
                duplicates[setting_name] = files
        
        return duplicates
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for configuration consolidation."""
        recommendations = []
        
        # Analyze file count and types
        file_types = defaultdict(int)
        for config_file in self.config_files:
            file_types[config_file.file_type] += 1
        
        if len(self.config_files) > 10:
            recommendations.append(
                f"Consider consolidating {len(self.config_files)} configuration files "
                "into a unified configuration system"
            )
        
        if file_types['json'] > 3 and file_types['yaml'] > 3:
            recommendations.append(
                "Standardize on a single configuration format (YAML recommended) "
                "instead of mixing JSON and YAML files"
            )
        
        # Check for environment-specific configs
        env_configs = [cf for cf in self.config_files if 'env' in str(cf.path).lower()]
        if len(env_configs) > 3:
            recommendations.append(
                "Implement environment-specific configuration overrides "
                "instead of separate environment files"
            )
        
        # Check for scattered configs
        config_dirs = set()
        for config_file in self.config_files:
            config_dirs.add(config_file.path.parent)
        
        if len(config_dirs) > 5:
            recommendations.append(
                f"Configuration files are scattered across {len(config_dirs)} directories. "
                "Consider centralizing them in a single config/ directory"
            )
        
        return recommendations
    
    def create_migration_plan(self) -> Dict[str, Any]:
        """Create a migration plan for configuration consolidation."""
        plan = {
            "phases": [],
            "unified_schema": {},
            "file_mapping": {},
            "backup_strategy": {},
            "validation_steps": []
        }
        
        # Phase 1: Backup and analysis
        plan["phases"].append({
            "phase": 1,
            "name": "Backup and Analysis",
            "description": "Create backups of all configuration files and analyze dependencies",
            "tasks": [
                "Create backup directory: config/legacy/",
                "Copy all existing configuration files to backup",
                "Document current configuration usage patterns",
                "Identify critical vs non-critical settings"
            ]
        })
        
        # Phase 2: Schema design
        plan["phases"].append({
            "phase": 2,
            "name": "Unified Schema Design",
            "description": "Design comprehensive configuration schema",
            "tasks": [
                "Create unified-config-schema.yaml",
                "Define environment override structure",
                "Design validation rules",
                "Create migration mapping rules"
            ]
        })
        
        # Phase 3: Implementation
        plan["phases"].append({
            "phase": 3,
            "name": "Implementation",
            "description": "Implement unified configuration system",
            "tasks": [
                "Create config/unified-config.yaml",
                "Implement configuration loader",
                "Add validation system",
                "Create migration tools"
            ]
        })
        
        # Phase 4: Migration
        plan["phases"].append({
            "phase": 4,
            "name": "Migration",
            "description": "Migrate from scattered to unified configuration",
            "tasks": [
                "Run migration tools",
                "Update application code to use unified config",
                "Test all components with new configuration",
                "Validate configuration consistency"
            ]
        })
        
        # File mapping
        for config_file in self.config_files:
            plan["file_mapping"][str(config_file.path)] = {
                "target_section": self._suggest_target_section(config_file),
                "priority": self._determine_migration_priority(config_file),
                "requires_manual_review": self._requires_manual_review(config_file)
            }
        
        return plan
    
    def _suggest_target_section(self, config_file: ConfigFile) -> str:
        """Suggest which section of unified config this file should map to."""
        path_str = str(config_file.path).lower()
        
        if 'backend' in path_str or 'api' in path_str:
            return 'backend'
        elif 'frontend' in path_str or 'ui' in path_str:
            return 'frontend'
        elif 'test' in path_str:
            return 'testing'
        elif 'docker' in path_str or 'deploy' in path_str:
            return 'deployment'
        elif 'monitor' in path_str or 'health' in path_str:
            return 'monitoring'
        else:
            return 'system'
    
    def _determine_migration_priority(self, config_file: ConfigFile) -> str:
        """Determine migration priority for a configuration file."""
        path_str = str(config_file.path).lower()
        
        # High priority for core application configs
        if any(term in path_str for term in ['config.json', 'main', 'app', 'core']):
            return 'high'
        
        # Medium priority for component configs
        if any(term in path_str for term in ['backend', 'frontend', 'api']):
            return 'medium'
        
        # Low priority for development/testing configs
        if any(term in path_str for term in ['test', 'dev', 'example', 'demo']):
            return 'low'
        
        return 'medium'
    
    def _requires_manual_review(self, config_file: ConfigFile) -> bool:
        """Determine if a config file requires manual review during migration."""
        # Files with complex nested structures need manual review
        if config_file.content and self._has_complex_structure(config_file.content):
            return True
        
        # Large files need manual review
        if config_file.size > 10000:  # 10KB
            return True
        
        # Environment-specific files need manual review
        path_str = str(config_file.path).lower()
        if any(term in path_str for term in ['prod', 'production', 'staging', 'env']):
            return True
        
        return False
    
    def _has_complex_structure(self, content: Any, depth: int = 0) -> bool:
        """Check if configuration has complex nested structure."""
        if depth > 3:  # More than 3 levels deep
            return True
        
        if isinstance(content, dict):
            if len(content) > 10:
                return True
            for value in content.values():
                if isinstance(value, (dict, list)):
                    if self._has_complex_structure(value, depth + 1):
                        return True
        elif isinstance(content, list):
            if len(content) > 20:
                return True
            for item in content:
                if isinstance(item, (dict, list)):
                    if self._has_complex_structure(item, depth + 1):
                        return True
        
        return False
    
    def generate_report(self) -> ConfigAnalysisReport:
        """Generate comprehensive configuration analysis report."""
        # Scan for configuration files
        config_files = self.scan_project()
        
        # Analyze dependencies and conflicts
        dependencies = self.analyze_dependencies()
        conflicts = self.detect_conflicts()
        duplicate_settings = self.find_duplicate_settings()
        recommendations = self.generate_recommendations()
        migration_plan = self.create_migration_plan()
        
        return ConfigAnalysisReport(
            total_files=len(config_files),
            config_files=config_files,
            conflicts=conflicts,
            dependencies=dependencies,
            duplicate_settings=duplicate_settings,
            recommendations=recommendations,
            migration_plan=migration_plan
        )


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze configuration landscape')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output', help='Output file for analysis report')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    analyzer = ConfigLandscapeAnalyzer(args.project_root)
    report = analyzer.generate_report()
    
    # Convert to serializable format
    report_dict = asdict(report)
    
    # Convert sets to lists for JSON serialization
    for config_file in report_dict['config_files']:
        if 'settings' in config_file and config_file['settings']:
            config_file['settings'] = list(config_file['settings'])
    
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'yaml':
                yaml.dump(report_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(report_dict, f, indent=2, default=str)
        print(f"Analysis report saved to {args.output}")
    else:
        print(json.dumps(report_dict, indent=2, default=str))


if __name__ == '__main__':
    main()