#!/usr/bin/env python3
"""
Configuration Dependency Mapper

Creates a focused analysis of configuration dependencies and relationships
for the WAN22 project.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass, asdict
import re


@dataclass
class ConfigMapping:
    """Maps configuration files to their purpose and relationships."""
    file_path: str
    purpose: str
    component: str
    priority: str  # high, medium, low
    settings_count: int
    key_settings: List[str]
    depends_on: List[str]
    used_by: List[str]


class ConfigDependencyMapper:
    """Maps configuration dependencies for consolidation planning."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.config_mappings: List[ConfigMapping] = []
        
    def analyze_core_configs(self) -> List[ConfigMapping]:
        """Analyze core configuration files and their relationships."""
        core_configs = [
            # Main application configs
            ("backend/config.json", "Backend application configuration", "backend", "high"),
            ("config.json", "Root application configuration", "system", "high"),
            ("startup_config.json", "Startup configuration", "system", "high"),
            ("startup_config_windows_safe.json", "Windows-safe startup config", "system", "medium"),
            
            # Unified config system
            ("config/unified-config.yaml", "Unified configuration system", "system", "high"),
            ("config/base.yaml", "Base configuration", "system", "high"),
            ("config/alerting-config.yaml", "Alerting configuration", "monitoring", "medium"),
            ("config/production-health.yaml", "Production health monitoring", "monitoring", "medium"),
            ("config/ci-health-config.yaml", "CI health configuration", "ci", "medium"),
            
            # Environment configs
            (".env", "Root environment variables", "system", "high"),
            ("backend/.env.example", "Backend environment template", "backend", "medium"),
            ("frontend/.env", "Frontend environment variables", "frontend", "high"),
            ("frontend/.env.example", "Frontend environment template", "frontend", "medium"),
            
            # Test configs
            ("pytest.ini", "Pytest configuration", "testing", "medium"),
            ("tests/config/test-config.yaml", "Test suite configuration", "testing", "high"),
            ("tests/config/execution_config.yaml", "Test execution configuration", "testing", "medium"),
            
            # Build and deployment
            ("frontend/package.json", "Frontend package configuration", "frontend", "high"),
            ("backend/requirements.txt", "Backend dependencies", "backend", "high"),
            ("backend/docker-compose.yml", "Docker composition", "deployment", "medium"),
            ("backend/Dockerfile", "Docker build configuration", "deployment", "medium"),
            
            # Hardware and optimization
            ("rtx4080_memory_config.json", "RTX 4080 memory configuration", "hardware", "medium"),
            ("backend/hardware_profile_final.json", "Hardware profile", "hardware", "medium"),
            ("backend/frontend_vram_config.json", "VRAM configuration", "hardware", "medium"),
        ]
        
        mappings = []
        for file_path, purpose, component, priority in core_configs:
            full_path = self.project_root / file_path
            if full_path.exists():
                mapping = self._analyze_config_file(file_path, purpose, component, priority)
                if mapping:
                    mappings.append(mapping)
        
        self.config_mappings = mappings
        return mappings
    
    def _analyze_config_file(self, file_path: str, purpose: str, component: str, priority: str) -> ConfigMapping:
        """Analyze a single configuration file."""
        full_path = self.project_root / file_path
        
        try:
            # Load and analyze content
            content = self._load_file_content(full_path)
            settings_count = self._count_settings(content) if content else 0
            key_settings = self._extract_key_settings(content) if content else []
            
            # Analyze dependencies
            depends_on = self._find_dependencies(file_path, content)
            used_by = self._find_usage(file_path)
            
            return ConfigMapping(
                file_path=file_path,
                purpose=purpose,
                component=component,
                priority=priority,
                settings_count=settings_count,
                key_settings=key_settings,
                depends_on=depends_on,
                used_by=used_by
            )
        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return ConfigMapping(
                file_path=file_path,
                purpose=purpose,
                component=component,
                priority=priority,
                settings_count=0,
                key_settings=[],
                depends_on=[],
                used_by=[]
            )
    
    def _load_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.suffix.lower() == '.json':
                return json.loads(content)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif file_path.name.endswith('.env'):
                return self._parse_env_file(content)
            elif file_path.suffix.lower() == '.ini':
                import configparser
                parser = configparser.ConfigParser()
                parser.read_string(content)
                return {section: dict(parser[section]) for section in parser.sections()}
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}
    
    def _parse_env_file(self, content: str) -> Dict[str, str]:
        """Parse environment file content."""
        env_vars = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')
        return env_vars
    
    def _count_settings(self, content: Dict[str, Any]) -> int:
        """Count total settings in configuration."""
        if not isinstance(content, dict):
            return 0
        
        count = 0
        for key, value in content.items():
            count += 1
            if isinstance(value, dict):
                count += self._count_settings(value)
        return count
    
    def _extract_key_settings(self, content: Dict[str, Any]) -> List[str]:
        """Extract key configuration settings."""
        if not isinstance(content, dict):
            return []
        
        key_settings = []
        important_keys = [
            'host', 'port', 'database', 'api_key', 'secret_key', 'debug', 
            'environment', 'log_level', 'timeout', 'max_workers', 'memory',
            'gpu', 'model_path', 'output_path', 'batch_size'
        ]
        
        for key in content.keys():
            if any(important in key.lower() for important in important_keys):
                key_settings.append(key)
        
        return key_settings[:10]  # Limit to top 10
    
    def _find_dependencies(self, file_path: str, content: Dict[str, Any]) -> List[str]:
        """Find files that this configuration depends on."""
        dependencies = []
        
        if not content:
            return dependencies
        
        # Convert content to string for searching
        content_str = json.dumps(content, default=str).lower()
        
        # Look for file references
        config_patterns = [
            r'config[/\\][\w\-\.]+\.(json|yaml|yml)',
            r'[\w\-\.]+\.env',
            r'[\w\-\.]+config\.(json|yaml|yml)',
        ]
        
        for pattern in config_patterns:
            matches = re.findall(pattern, content_str)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match != Path(file_path).name:
                    dependencies.append(match)
        
        return list(set(dependencies))
    
    def _find_usage(self, file_path: str) -> List[str]:
        """Find components that use this configuration file."""
        used_by = []
        file_name = Path(file_path).name
        
        # Search for references in Python files
        python_files = list(self.project_root.rglob('*.py'))
        
        for py_file in python_files[:50]:  # Limit search for performance
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if file_name in content or file_path in content:
                        relative_path = py_file.relative_to(self.project_root)
                        used_by.append(str(relative_path))
            except:
                continue
        
        return used_by[:5]  # Limit to top 5
    
    def generate_consolidation_plan(self) -> Dict[str, Any]:
        """Generate a consolidation plan based on the analysis."""
        plan = {
            "summary": {
                "total_configs": len(self.config_mappings),
                "high_priority": len([m for m in self.config_mappings if m.priority == "high"]),
                "components": list(set(m.component for m in self.config_mappings))
            },
            "consolidation_groups": {},
            "migration_order": [],
            "unified_structure": {}
        }
        
        # Group by component
        by_component = {}
        for mapping in self.config_mappings:
            if mapping.component not in by_component:
                by_component[mapping.component] = []
            by_component[mapping.component].append(mapping)
        
        plan["consolidation_groups"] = {
            component: [
                {
                    "file": m.file_path,
                    "purpose": m.purpose,
                    "priority": m.priority,
                    "settings_count": m.settings_count,
                    "key_settings": m.key_settings
                }
                for m in mappings
            ]
            for component, mappings in by_component.items()
        }
        
        # Define migration order (high priority first)
        high_priority = [m for m in self.config_mappings if m.priority == "high"]
        medium_priority = [m for m in self.config_mappings if m.priority == "medium"]
        low_priority = [m for m in self.config_mappings if m.priority == "low"]
        
        plan["migration_order"] = [
            {"phase": "Phase 1 - Critical Configs", "files": [m.file_path for m in high_priority]},
            {"phase": "Phase 2 - Component Configs", "files": [m.file_path for m in medium_priority]},
            {"phase": "Phase 3 - Optional Configs", "files": [m.file_path for m in low_priority]}
        ]
        
        # Define unified structure
        plan["unified_structure"] = {
            "config/unified-config.yaml": {
                "system": "Core system settings (ports, hosts, debug)",
                "backend": "Backend API and service configuration",
                "frontend": "Frontend application settings",
                "database": "Database connection and settings",
                "monitoring": "Health checks and alerting",
                "hardware": "GPU and performance optimization",
                "testing": "Test execution and coverage settings"
            },
            "config/environments/": {
                "development.yaml": "Development environment overrides",
                "staging.yaml": "Staging environment overrides", 
                "production.yaml": "Production environment overrides",
                "testing.yaml": "Testing environment overrides"
            },
            "config/schemas/": {
                "unified-config-schema.yaml": "Configuration validation schema",
                "migration-rules.yaml": "Rules for migrating existing configs"
            }
        }
        
        return plan
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency mapping report."""
        mappings = self.analyze_core_configs()
        consolidation_plan = self.generate_consolidation_plan()
        
        return {
            "analysis_summary": {
                "total_configs_analyzed": len(mappings),
                "components": list(set(m.component for m in mappings)),
                "priority_breakdown": {
                    "high": len([m for m in mappings if m.priority == "high"]),
                    "medium": len([m for m in mappings if m.priority == "medium"]),
                    "low": len([m for m in mappings if m.priority == "low"])
                }
            },
            "config_mappings": [asdict(m) for m in mappings],
            "consolidation_plan": consolidation_plan,
            "recommendations": [
                "Implement unified configuration system using config/unified-config.yaml",
                "Create environment-specific override files in config/environments/",
                "Migrate high-priority configurations first (system, backend, frontend)",
                "Establish configuration validation using schemas",
                "Create backup and rollback procedures before migration",
                "Update application code to use unified configuration loader"
            ]
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze configuration dependencies')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output', help='Output file for dependency report')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    mapper = ConfigDependencyMapper(args.project_root)
    report = mapper.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'yaml':
                yaml.dump(report, f, default_flow_style=False, indent=2)
            else:
                json.dump(report, f, indent=2, default=str)
        print(f"Dependency report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()