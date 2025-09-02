#!/usr/bin/env python3
"""
Configuration Migration Script
Consolidates scattered configuration files into the unified config system.
"""

import json
import yaml
import os
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

class ConfigMigrator:
    def __init__(self):
        self.root_dir = Path(".")
        self.config_dir = self.root_dir / "config"
        self.backup_dir = self.root_dir / "config_backups" / f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.unified_config_path = self.config_dir / "unified-config.yaml"
        
    def backup_existing_configs(self):
        """Create backups of all existing configuration files."""
        print("Creating configuration backups...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        config_files = [
            "config.json",
            "startup_config.json", 
            "startup_config_windows_safe.json",
            "backend/config.json",
            "recovery_config.json",
            "rtx4080_memory_config.json",
            "quality-config.yaml"
        ]
        
        for config_file in config_files:
            source = self.root_dir / config_file
            if source.exists():
                dest = self.backup_dir / config_file.replace("/", "_")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                print(f"  Backed up: {config_file}")
    
    def load_existing_configs(self) -> Dict[str, Any]:
        """Load all existing configuration files."""
        configs = {}
        
        # Load JSON configs
        json_configs = [
            ("main", "config.json"),
            ("startup", "startup_config.json"),
            ("startup_safe", "startup_config_windows_safe.json"),
            ("backend", "backend/config.json"),
            ("recovery", "recovery_config.json"),
            ("rtx4080", "rtx4080_memory_config.json")
        ]
        
        for name, path in json_configs:
            file_path = self.root_dir / path
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        configs[name] = json.load(f)
                    print(f"  Loaded: {path}")
                except Exception as e:
                    print(f"  Error loading {path}: {e}")
        
        # Load YAML configs
        yaml_configs = [
            ("quality", "quality-config.yaml")
        ]
        
        for name, path in yaml_configs:
            file_path = self.root_dir / path
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        configs[name] = yaml.safe_load(f)
                    print(f"  Loaded: {path}")
                except Exception as e:
                    print(f"  Error loading {path}: {e}")
        
        return configs
    
    def merge_configurations(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all configurations into unified structure."""
        print("Merging configurations...")
        
        # Load current unified config as base
        with open(self.unified_config_path, 'r') as f:
            unified = yaml.safe_load(f)
        
        # Merge main config
        if 'main' in configs:
            main = configs['main']
            if 'system' in main:
                unified['system'].update(main['system'])
            if 'directories' in main:
                unified['system']['output_directory'] = main['directories'].get('output_directory', unified['system']['output_directory'])
                unified['system']['models_directory'] = main['directories'].get('models_directory', unified['system']['models_directory'])
                unified['system']['loras_directory'] = main['directories'].get('loras_directory', unified['system']['loras_directory'])
            if 'generation' in main:
                unified['generation'].update(main['generation'])
            if 'models' in main:
                unified['models'].update(main['models'])
            if 'optimization' in main:
                unified['hardware'].update(main['optimization'])
            if 'ui' in main:
                unified['ui'].update(main['ui'])
            if 'performance' in main:
                unified['performance'].update(main['performance'])
            if 'prompt_enhancement' in main:
                unified['prompt_enhancement'].update(main['prompt_enhancement'])
        
        # Merge startup config
        if 'startup' in configs:
            startup = configs['startup']
            if 'backend' in startup:
                unified['api'].update(startup['backend'])
            if 'frontend' in startup:
                unified['frontend'].update(startup['frontend'])
            if 'logging' in startup:
                unified['logging'].update(startup['logging'])
            if 'recovery' in startup:
                unified['recovery'].update(startup['recovery'])
            if 'environment' in startup:
                unified['environment_validation'].update(startup['environment'])
            if 'security' in startup:
                unified['security'].update(startup['security'])
        
        # Merge backend config
        if 'backend' in configs:
            backend = configs['backend']
            if 'generation' in backend:
                unified['generation'].update(backend['generation'])
            if 'models' in backend:
                unified['models'].update(backend['models'])
            if 'hardware' in backend:
                unified['hardware'].update(backend['hardware'])
            if 'optimization' in backend:
                unified['hardware'].update(backend['optimization'])
            if 'websocket' in backend:
                unified['websocket'].update(backend['websocket'])
            if 'api' in backend:
                unified['api'].update(backend['api'])
            if 'database' in backend:
                unified['database'].update(backend['database'])
            if 'logging' in backend:
                unified['logging'].update(backend['logging'])
            if 'security' in backend:
                unified['security'].update(backend['security'])
            if 'performance' in backend:
                unified['performance'].update(backend['performance'])
        
        # Update metadata
        unified['last_updated'] = datetime.now().isoformat()
        
        return unified
    
    def save_unified_config(self, config: Dict[str, Any]):
        """Save the merged configuration."""
        print("Saving unified configuration...")
        
        # Create backup of current unified config
        if self.unified_config_path.exists():
            backup_path = self.backup_dir / "unified-config.yaml.backup"
            shutil.copy2(self.unified_config_path, backup_path)
        
        # Save merged config
        with open(self.unified_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"  Saved unified config to: {self.unified_config_path}")
    
    def create_migration_report(self, configs: Dict[str, Any]):
        """Create a report of the migration process."""
        report_path = self.backup_dir / "migration_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Configuration Migration Report\n\n")
            f.write(f"**Migration Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Migrated Configuration Files\n\n")
            
            for name, config in configs.items():
                f.write(f"- **{name}**: {len(config)} settings migrated\n")
            
            f.write(f"\n## Backup Location\n\n")
            f.write(f"All original configuration files have been backed up to:\n")
            f.write(f"`{self.backup_dir}`\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review the unified configuration file\n")
            f.write("2. Test the application with the new configuration\n")
            f.write("3. Remove old configuration files if everything works correctly\n")
            f.write("4. Update any scripts or documentation that reference old config files\n")
        
        print(f"Migration report saved to: {report_path}")
    
    def migrate(self):
        """Run the complete migration process."""
        print("Starting configuration migration...")
        
        # Create backups
        self.backup_existing_configs()
        
        # Load existing configs
        configs = self.load_existing_configs()
        
        if not configs:
            print("No configuration files found to migrate.")
            return
        
        # Merge configurations
        unified_config = self.merge_configurations(configs)
        
        # Save unified config
        self.save_unified_config(unified_config)
        
        # Create migration report
        self.create_migration_report(configs)
        
        print("\nConfiguration migration completed successfully!")
        print(f"Backups saved to: {self.backup_dir}")
        print(f"Unified config: {self.unified_config_path}")

if __name__ == "__main__":
    migrator = ConfigMigrator()
    migrator.migrate()