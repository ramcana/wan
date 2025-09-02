"""
Configuration Unifier and Migration System

This module provides tools for migrating scattered configuration files
to the unified configuration system, including discovery, parsing, and backup.
"""

import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

from .unified_config import UnifiedConfig


@dataclass
class ConfigSource:
    """Represents a discovered configuration source"""
    path: Path
    format: str  # 'json', 'yaml', 'py', 'env'
    category: str  # 'system', 'api', 'models', etc.
    content: Dict[str, Any]
    confidence: float  # 0.0 to 1.0 - how confident we are about the category


@dataclass
class MigrationReport:
    """Report of configuration migration process"""
    timestamp: datetime
    sources_found: List[ConfigSource]
    sources_migrated: List[ConfigSource]
    sources_skipped: List[ConfigSource]
    backup_path: Optional[Path]
    unified_config_path: Path
    errors: List[str]
    warnings: List[str]
    success: bool


class ConfigurationUnifier:
    """
    Handles migration of scattered configuration files to unified system
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Configuration file patterns to search for
        self.config_patterns = [
            "config.json",
            "*.config.json",
            "config.yaml",
            "config.yml",
            "*.config.yaml",
            "*.config.yml",
            "settings.json",
            "settings.yaml",
            "startup_config.json",
            ".env",
            "*.env"
        ]
        
        # Mapping of file patterns to configuration categories
        self.category_mapping = {
            "startup_config": "startup",
            "backend/config": "backend",
            "frontend/config": "frontend",
            "config": "system",
            "settings": "system",
            ".env": "environment"
        }
    
    def discover_config_files(self) -> List[ConfigSource]:
        """
        Discover all configuration files in the project
        
        Returns:
            List of discovered configuration sources
        """
        sources = []
        
        # Search for configuration files
        for pattern in self.config_patterns:
            for config_file in self.project_root.rglob(pattern):
                if self._should_skip_file(config_file):
                    continue
                
                try:
                    source = self._analyze_config_file(config_file)
                    if source:
                        sources.append(source)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {config_file}: {e}")
        
        return sources
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during discovery"""
        skip_dirs = {
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'venv', '.venv', 'env', '.env', 'dist', 'build',
            '.kiro', 'logs', 'outputs', 'models', 'temp'
        }
        
        # Skip files in excluded directories
        for part in file_path.parts:
            if part in skip_dirs:
                return True
        
        # Skip backup files
        if file_path.name.endswith(('.bak', '.backup', '.old')):
            return True
        
        # Skip our own unified config files
        if 'unified-config' in file_path.name:
            return True
        
        return False
    
    def _analyze_config_file(self, file_path: Path) -> Optional[ConfigSource]:
        """
        Analyze a configuration file and extract its content
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            ConfigSource object or None if file cannot be parsed
        """
        try:
            content = {}
            format_type = self._detect_format(file_path)
            
            if format_type == 'json':
                content = json.loads(file_path.read_text(encoding='utf-8'))
            elif format_type in ['yaml', 'yml']:
                content = yaml.safe_load(file_path.read_text(encoding='utf-8'))
            elif format_type == 'env':
                content = self._parse_env_file(file_path)
            else:
                return None
            
            category, confidence = self._categorize_config(file_path, content)
            
            return ConfigSource(
                path=file_path,
                format=format_type,
                category=category,
                content=content,
                confidence=confidence
            )
        
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect the format of a configuration file"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return 'json'
        elif suffix in ['.yaml', '.yml']:
            return 'yaml'
        elif suffix == '.env' or file_path.name == '.env':
            return 'env'
        else:
            # Try to detect by content
            try:
                content = file_path.read_text(encoding='utf-8').strip()
                if content.startswith('{') and content.endswith('}'):
                    return 'json'
                elif '=' in content and not content.startswith('{'):
                    return 'env'
            except:
                pass
        
        return 'unknown'
    
    def _parse_env_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse environment file into dictionary"""
        content = {}
        
        try:
            for line in file_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    content[key] = value
        except Exception as e:
            self.logger.error(f"Failed to parse env file {file_path}: {e}")
        
        return content
    
    def _categorize_config(self, file_path: Path, content: Dict[str, Any]) -> Tuple[str, float]:
        """
        Categorize a configuration file based on path and content
        
        Returns:
            Tuple of (category, confidence_score)
        """
        path_str = str(file_path).lower()
        
        # Check path-based categorization
        for pattern, category in self.category_mapping.items():
            if pattern in path_str:
                return category, 0.8
        
        # Check content-based categorization
        if isinstance(content, dict):
            keys = set(content.keys())
            
            # API/Backend configuration
            if keys & {'api', 'host', 'port', 'cors_origins', 'database'}:
                return 'backend', 0.7
            
            # Frontend configuration
            if keys & {'frontend', 'build', 'dev', 'vite', 'webpack'}:
                return 'frontend', 0.7
            
            # Model configuration
            if keys & {'models', 't2v_model', 'i2v_model', 'quantization'}:
                return 'models', 0.7
            
            # Hardware configuration
            if keys & {'hardware', 'vram', 'gpu', 'cuda', 'optimization'}:
                return 'hardware', 0.7
            
            # Generation configuration
            if keys & {'generation', 'resolution', 'steps', 'fps', 'duration'}:
                return 'generation', 0.7
            
            # System configuration
            if keys & {'system', 'logging', 'debug', 'environment'}:
                return 'system', 0.6
        
        return 'unknown', 0.3
    
    def migrate_to_unified_config(
        self, 
        sources: List[ConfigSource] = None,
        output_path: Path = None,
        create_backup: bool = True
    ) -> MigrationReport:
        """
        Migrate discovered configuration sources to unified configuration
        
        Args:
            sources: List of configuration sources to migrate (auto-discover if None)
            output_path: Path for the unified configuration file
            create_backup: Whether to create backups of original files
            
        Returns:
            Migration report with results and any errors
        """
        if sources is None:
            sources = self.discover_config_files()
        
        if output_path is None:
            output_path = self.project_root / "config" / "unified-config.yaml"
        
        report = MigrationReport(
            timestamp=datetime.now(),
            sources_found=sources,
            sources_migrated=[],
            sources_skipped=[],
            backup_path=None,
            unified_config_path=output_path,
            errors=[],
            warnings=[],
            success=False
        )
        
        try:
            # Create backup if requested
            if create_backup:
                report.backup_path = self._create_backup(sources)
            
            # Create unified configuration
            unified_config = self._merge_configurations(sources, report)
            
            # Save unified configuration
            output_path.parent.mkdir(parents=True, exist_ok=True)
            unified_config.save_to_file(output_path)
            
            report.success = True
            self.logger.info(f"Successfully migrated configuration to {output_path}")
            
        except Exception as e:
            error_msg = f"Migration failed: {e}"
            report.errors.append(error_msg)
            self.logger.error(error_msg)
        
        return report
    
    def _create_backup(self, sources: List[ConfigSource]) -> Path:
        """Create backup of original configuration files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.project_root / "config" / "backups" / f"migration_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for source in sources:
            if source.path.exists():
                # Create relative path structure in backup
                rel_path = source.path.relative_to(self.project_root)
                backup_file = backup_dir / rel_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(source.path, backup_file)
        
        # Create backup manifest
        manifest = {
            'timestamp': timestamp,
            'sources': [
                {
                    'original_path': str(source.path),
                    'backup_path': str(backup_dir / source.path.relative_to(self.project_root)),
                    'category': source.category,
                    'format': source.format
                }
                for source in sources
            ]
        }
        
        manifest_path = backup_dir / "migration_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        return backup_dir
    
    def _merge_configurations(
        self, 
        sources: List[ConfigSource], 
        report: MigrationReport
    ) -> UnifiedConfig:
        """
        Merge multiple configuration sources into a unified configuration
        
        Args:
            sources: List of configuration sources to merge
            report: Migration report to update with results
            
        Returns:
            Unified configuration object
        """
        # Start with default configuration
        unified_config = UnifiedConfig()
        
        # Group sources by category
        categorized_sources = {}
        for source in sources:
            if source.category not in categorized_sources:
                categorized_sources[source.category] = []
            categorized_sources[source.category].append(source)
        
        # Merge each category
        for category, category_sources in categorized_sources.items():
            try:
                self._merge_category(unified_config, category, category_sources, report)
            except Exception as e:
                error_msg = f"Failed to merge {category} configuration: {e}"
                report.errors.append(error_msg)
                self.logger.error(error_msg)
        
        return unified_config
    
    def _merge_category(
        self, 
        unified_config: UnifiedConfig, 
        category: str, 
        sources: List[ConfigSource],
        report: MigrationReport
    ):
        """Merge sources for a specific category into the unified configuration"""
        
        # Define mapping from categories to unified config sections
        category_mapping = {
            'system': 'system',
            'backend': ['api', 'database', 'logging', 'security'],
            'frontend': 'frontend',
            'models': 'models',
            'hardware': 'hardware',
            'generation': 'generation',
            'startup': ['api', 'frontend', 'recovery'],
            'environment': 'environment_validation'
        }
        
        for source in sources:
            try:
                if category == 'backend':
                    self._merge_backend_config(unified_config, source)
                elif category == 'startup':
                    self._merge_startup_config(unified_config, source)
                elif category == 'system':
                    self._merge_system_config(unified_config, source)
                elif category == 'models':
                    self._merge_models_config(unified_config, source)
                elif category == 'hardware':
                    self._merge_hardware_config(unified_config, source)
                elif category == 'generation':
                    self._merge_generation_config(unified_config, source)
                elif category == 'frontend':
                    self._merge_frontend_config(unified_config, source)
                else:
                    # Generic merge for unknown categories
                    self._merge_generic_config(unified_config, source)
                
                report.sources_migrated.append(source)
                
            except Exception as e:
                error_msg = f"Failed to merge {source.path}: {e}"
                report.errors.append(error_msg)
                report.sources_skipped.append(source)
                self.logger.error(error_msg)
    
    def _merge_backend_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge backend configuration source"""
        content = source.content
        
        # API configuration
        if 'api' in content:
            api_config = content['api']
            for key, value in api_config.items():
                if hasattr(unified_config.api, key):
                    setattr(unified_config.api, key, value)
        
        # Direct API fields
        for key in ['host', 'port', 'debug', 'cors_origins', 'workers']:
            if key in content:
                if hasattr(unified_config.api, key):
                    setattr(unified_config.api, key, content[key])
        
        # Database configuration
        if 'database' in content:
            db_config = content['database']
            for key, value in db_config.items():
                if hasattr(unified_config.database, key):
                    setattr(unified_config.database, key, value)
        
        # Logging configuration
        if 'logging' in content:
            log_config = content['logging']
            for key, value in log_config.items():
                if hasattr(unified_config.logging, key):
                    setattr(unified_config.logging, key, value)
        
        # Security configuration
        if 'security' in content:
            sec_config = content['security']
            for key, value in sec_config.items():
                if hasattr(unified_config.security, key):
                    setattr(unified_config.security, key, value)
    
    def _merge_startup_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge startup configuration source"""
        content = source.content
        
        # Backend startup settings
        if 'backend' in content:
            backend_config = content['backend']
            for key, value in backend_config.items():
                if hasattr(unified_config.api, key):
                    setattr(unified_config.api, key, value)
        
        # Frontend startup settings
        if 'frontend' in content:
            frontend_config = content['frontend']
            for key, value in frontend_config.items():
                if hasattr(unified_config.frontend, key):
                    setattr(unified_config.frontend, key, value)
        
        # Recovery settings
        if 'recovery' in content:
            recovery_config = content['recovery']
            for key, value in recovery_config.items():
                if hasattr(unified_config.recovery, key):
                    setattr(unified_config.recovery, key, value)
        
        # Environment validation
        if 'environment' in content:
            env_config = content['environment']
            for key, value in env_config.items():
                if hasattr(unified_config.environment_validation, key):
                    setattr(unified_config.environment_validation, key, value)
    
    def _merge_system_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge system configuration source"""
        content = source.content
        
        # System settings
        if 'system' in content:
            sys_config = content['system']
            for key, value in sys_config.items():
                if hasattr(unified_config.system, key):
                    setattr(unified_config.system, key, value)
        
        # Direct system fields
        for key in ['debug', 'log_level', 'max_queue_size', 'stats_refresh_interval']:
            if key in content:
                if hasattr(unified_config.system, key):
                    setattr(unified_config.system, key, content[key])
        
        # Directories
        if 'directories' in content:
            dir_config = content['directories']
            for key, value in dir_config.items():
                if hasattr(unified_config.system, key):
                    setattr(unified_config.system, key, value)
    
    def _merge_models_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge models configuration source"""
        content = source.content
        
        # Models settings
        if 'models' in content:
            models_config = content['models']
            for key, value in models_config.items():
                if hasattr(unified_config.models, key):
                    setattr(unified_config.models, key, value)
        
        # Direct model fields
        for key in ['t2v_model', 'i2v_model', 'ti2v_model', 'default_model']:
            if key in content:
                if hasattr(unified_config.models, key):
                    setattr(unified_config.models, key, content[key])
    
    def _merge_hardware_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge hardware configuration source"""
        content = source.content
        
        # Hardware settings
        if 'hardware' in content:
            hw_config = content['hardware']
            for key, value in hw_config.items():
                if hasattr(unified_config.hardware, key):
                    setattr(unified_config.hardware, key, value)
        
        # Optimization settings
        if 'optimization' in content:
            opt_config = content['optimization']
            for key, value in opt_config.items():
                if hasattr(unified_config.hardware, key):
                    setattr(unified_config.hardware, key, value)
    
    def _merge_generation_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge generation configuration source"""
        content = source.content
        
        # Generation settings
        if 'generation' in content:
            gen_config = content['generation']
            for key, value in gen_config.items():
                if hasattr(unified_config.generation, key):
                    setattr(unified_config.generation, key, value)
        
        # Direct generation fields
        for key in ['default_resolution', 'default_steps', 'default_duration', 'default_fps']:
            if key in content:
                if hasattr(unified_config.generation, key):
                    setattr(unified_config.generation, key, content[key])
    
    def _merge_frontend_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Merge frontend configuration source"""
        content = source.content
        
        # Frontend settings
        if 'frontend' in content:
            frontend_config = content['frontend']
            for key, value in frontend_config.items():
                if hasattr(unified_config.frontend, key):
                    setattr(unified_config.frontend, key, value)
        
        # UI settings
        if 'ui' in content:
            ui_config = content['ui']
            for key, value in ui_config.items():
                if hasattr(unified_config.ui, key):
                    setattr(unified_config.ui, key, value)
    
    def _merge_generic_config(self, unified_config: UnifiedConfig, source: ConfigSource):
        """Generic merge for unknown configuration categories"""
        # This is a fallback that tries to match keys to any section
        content = source.content
        
        for key, value in content.items():
            # Try to find a matching section in unified config
            for section_name in ['system', 'api', 'models', 'hardware', 'generation']:
                section = getattr(unified_config, section_name)
                if hasattr(section, key):
                    setattr(section, key, value)
                    break
    
    def rollback_migration(self, backup_path: Path) -> bool:
        """
        Rollback a configuration migration using backup
        
        Args:
            backup_path: Path to the backup directory
            
        Returns:
            True if rollback was successful, False otherwise
        """
        try:
            manifest_path = backup_path / "migration_manifest.json"
            if not manifest_path.exists():
                self.logger.error(f"Backup manifest not found: {manifest_path}")
                return False
            
            manifest = json.loads(manifest_path.read_text())
            
            for source_info in manifest['sources']:
                original_path = Path(source_info['original_path'])
                backup_file = Path(source_info['backup_path'])
                
                if backup_file.exists():
                    # Restore original file
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, original_path)
                    self.logger.info(f"Restored {original_path}")
            
            self.logger.info(f"Successfully rolled back migration from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def generate_migration_preview(self, sources: List[ConfigSource] = None) -> Dict[str, Any]:
        """
        Generate a preview of what would be migrated without actually doing it
        
        Args:
            sources: List of configuration sources (auto-discover if None)
            
        Returns:
            Dictionary with migration preview information
        """
        if sources is None:
            sources = self.discover_config_files()
        
        preview = {
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(sources),
            'sources_by_category': {},
            'sources_by_format': {},
            'potential_conflicts': [],
            'recommendations': []
        }
        
        # Categorize sources
        for source in sources:
            # By category
            if source.category not in preview['sources_by_category']:
                preview['sources_by_category'][source.category] = []
            preview['sources_by_category'][source.category].append({
                'path': str(source.path),
                'confidence': source.confidence
            })
            
            # By format
            if source.format not in preview['sources_by_format']:
                preview['sources_by_format'][source.format] = 0
            preview['sources_by_format'][source.format] += 1
        
        # Detect potential conflicts
        self._detect_conflicts(sources, preview)
        
        # Generate recommendations
        self._generate_recommendations(sources, preview)
        
        return preview
    
    def _detect_conflicts(self, sources: List[ConfigSource], preview: Dict[str, Any]):
        """Detect potential conflicts in configuration migration"""
        # Group sources by category
        by_category = {}
        for source in sources:
            if source.category not in by_category:
                by_category[source.category] = []
            by_category[source.category].append(source)
        
        # Check for conflicts within categories
        for category, category_sources in by_category.items():
            if len(category_sources) > 1:
                # Multiple sources for same category - potential conflict
                conflict = {
                    'type': 'multiple_sources',
                    'category': category,
                    'sources': [str(s.path) for s in category_sources],
                    'severity': 'medium'
                }
                preview['potential_conflicts'].append(conflict)
    
    def _generate_recommendations(self, sources: List[ConfigSource], preview: Dict[str, Any]):
        """Generate recommendations for configuration migration"""
        recommendations = []
        
        # Check for low confidence sources
        low_confidence = [s for s in sources if s.confidence < 0.5]
        if low_confidence:
            recommendations.append({
                'type': 'review_categorization',
                'message': f"Review categorization for {len(low_confidence)} sources with low confidence",
                'sources': [str(s.path) for s in low_confidence]
            })
        
        # Check for unknown categories
        unknown_sources = [s for s in sources if s.category == 'unknown']
        if unknown_sources:
            recommendations.append({
                'type': 'manual_review',
                'message': f"Manually review {len(unknown_sources)} sources with unknown category",
                'sources': [str(s.path) for s in unknown_sources]
            })
        
        # Recommend backup
        recommendations.append({
            'type': 'create_backup',
            'message': "Create backup before migration to enable rollback if needed"
        })
        
        preview['recommendations'] = recommendations