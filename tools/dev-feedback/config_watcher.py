#!/usr/bin/env python3
"""
Configuration Watcher with Hot-Reloading

This module provides hot-reloading for configuration changes during development.
"""

import os
import sys
import time
import subprocess
import threading
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
import logging
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog for file watching...")
    subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"], check=True)
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

@dataclass
class ConfigChange:
    """Configuration change event"""
    file_path: Path
    change_type: str  # 'modified', 'created', 'deleted'
    timestamp: datetime
    old_content: Optional[str] = None
    new_content: Optional[str] = None

@dataclass
class ServiceConfig:
    """Service configuration for reloading"""
    name: str
    command: List[str]
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None
    restart_on_config_change: bool = True
    config_files: List[str] = None

class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration watching"""
    
    def __init__(self, watcher: 'ConfigWatcher'):
        self.watcher = watcher
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self.watcher.is_config_file(file_path):
            self.logger.info(f"Configuration file changed: {file_path}")
            self.watcher.handle_config_change(file_path, 'modified')
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self.watcher.is_config_file(file_path):
            self.logger.info(f"Configuration file created: {file_path}")
            self.watcher.handle_config_change(file_path, 'created')
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self.watcher.is_config_file(file_path):
            self.logger.info(f"Configuration file deleted: {file_path}")
            self.watcher.handle_config_change(file_path, 'deleted')

class ConfigWatcher:
    """Watch configuration files and provide hot-reloading"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config_patterns = [
            "*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.env"
        ]
        self.watch_dirs = [
            self.project_root / "config",
            self.project_root / "backend",
            self.project_root / "frontend",
            self.project_root / ".kiro"
        ]
        
        # State management
        self.observer = Observer()
        self.running = False
        self.config_cache = {}
        self.change_handlers = []
        self.services = {}
        
        # Debouncing
        self.pending_changes = {}
        self.debounce_delay = 0.5  # 500ms
        
        # Load initial configuration
        self._load_initial_configs()
    
    def _load_initial_configs(self):
        """Load initial configuration files into cache"""
        for watch_dir in self.watch_dirs:
            if watch_dir.exists():
                for pattern in self.config_patterns:
                    for config_file in watch_dir.rglob(pattern):
                        if config_file.is_file():
                            try:
                                content = self._read_config_file(config_file)
                                self.config_cache[str(config_file)] = content
                            except Exception as e:
                                self.logger.warning(f"Failed to load config {config_file}: {e}")
    
    def _read_config_file(self, file_path: Path) -> Any:
        """Read and parse configuration file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse based on file extension
        if file_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(content)
        elif file_path.suffix == '.json':
            return json.loads(content)
        elif file_path.suffix == '.env':
            # Simple .env parsing
            env_vars = {}
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
            return env_vars
        else:
            return content
    
    def is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file"""
        # Check file extension
        for pattern in self.config_patterns:
            if file_path.match(pattern):
                return True
        
        # Check specific config file names
        config_names = [
            'config.json', 'config.yaml', 'config.yml',
            'settings.json', 'settings.yaml', 'settings.yml',
            '.env', '.env.local', '.env.development'
        ]
        
        return file_path.name in config_names
    
    def handle_config_change(self, file_path: Path, change_type: str):
        """Handle configuration file change with debouncing"""
        file_str = str(file_path)
        
        # Cancel previous pending change for this file
        if file_str in self.pending_changes:
            self.pending_changes[file_str].cancel()
        
        # Schedule new change handling
        timer = threading.Timer(
            self.debounce_delay,
            self._process_config_change,
            args=[file_path, change_type]
        )
        self.pending_changes[file_str] = timer
        timer.start()
    
    def _process_config_change(self, file_path: Path, change_type: str):
        """Process configuration change after debouncing"""
        file_str = str(file_path)
        
        # Remove from pending changes
        self.pending_changes.pop(file_str, None)
        
        try:
            # Get old content
            old_content = self.config_cache.get(file_str)
            
            # Get new content
            new_content = None
            if change_type != 'deleted' and file_path.exists():
                new_content = self._read_config_file(file_path)
                self.config_cache[file_str] = new_content
            elif change_type == 'deleted':
                self.config_cache.pop(file_str, None)
            
            # Create change event
            change = ConfigChange(
                file_path=file_path,
                change_type=change_type,
                timestamp=datetime.now(),
                old_content=old_content,
                new_content=new_content
            )
            
            # Validate configuration
            if new_content is not None:
                validation_result = self._validate_config(file_path, new_content)
                if not validation_result['valid']:
                    self.logger.error(f"Invalid configuration in {file_path}: {validation_result['error']}")
                    return
            
            # Notify handlers
            self._notify_change_handlers(change)
            
            # Handle service reloading
            self._handle_service_reloading(change)
            
            self.logger.info(f"✅ Configuration change processed: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing config change for {file_path}: {e}")
    
    def _validate_config(self, file_path: Path, content: Any) -> Dict[str, Any]:
        """Validate configuration content"""
        try:
            # Basic validation - check if it's parseable
            if isinstance(content, dict):
                # Additional validation for specific config files
                if file_path.name == 'unified-config.yaml':
                    return self._validate_unified_config(content)
                elif file_path.name == 'package.json':
                    return self._validate_package_json(content)
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _validate_unified_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate unified configuration"""
        required_sections = ['system', 'backend', 'frontend']
        
        for section in required_sections:
            if section not in config:
                return {'valid': False, 'error': f'Missing required section: {section}'}
        
        return {'valid': True}
    
    def _validate_package_json(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate package.json"""
        required_fields = ['name', 'version']
        
        for field in required_fields:
            if field not in config:
                return {'valid': False, 'error': f'Missing required field: {field}'}
        
        return {'valid': True}
    
    def _notify_change_handlers(self, change: ConfigChange):
        """Notify registered change handlers"""
        for handler in self.change_handlers:
            try:
                handler(change)
            except Exception as e:
                self.logger.error(f"Error in change handler: {e}")
    
    def _handle_service_reloading(self, change: ConfigChange):
        """Handle service reloading based on configuration changes"""
        file_path = change.file_path
        
        # Check which services need to be reloaded
        services_to_reload = []
        
        for service_name, service_config in self.services.items():
            if self._should_reload_service(service_config, file_path):
                services_to_reload.append(service_name)
        
        # Reload services
        for service_name in services_to_reload:
            self._reload_service(service_name)
    
    def _should_reload_service(self, service_config: ServiceConfig, changed_file: Path) -> bool:
        """Check if service should be reloaded for the changed file"""
        if not service_config.restart_on_config_change:
            return False
        
        # Check if file matches service's config files
        if service_config.config_files:
            file_str = str(changed_file)
            for config_pattern in service_config.config_files:
                if config_pattern in file_str:
                    return True
        
        # Default behavior: reload for any config change in relevant directories
        file_str = str(changed_file)
        service_dirs = [
            f"/{service_config.name}/",
            "/config/",
            "/.env"
        ]
        
        return any(service_dir in file_str for service_dir in service_dirs)
    
    def _reload_service(self, service_name: str):
        """Reload a service"""
        service_config = self.services.get(service_name)
        if not service_config:
            return
        
        self.logger.info(f"🔄 Reloading service: {service_name}")
        
        try:
            # For now, just log the reload action
            # In a real implementation, you might restart the service process
            self.logger.info(f"Service {service_name} would be reloaded with command: {' '.join(service_config.command)}")
            
            # You could implement actual service reloading here:
            # subprocess.run(service_config.command, cwd=service_config.cwd, env=service_config.env)
            
        except Exception as e:
            self.logger.error(f"Failed to reload service {service_name}: {e}")
    
    def register_change_handler(self, handler: Callable[[ConfigChange], None]):
        """Register a configuration change handler"""
        self.change_handlers.append(handler)
    
    def register_service(self, service_config: ServiceConfig):
        """Register a service for automatic reloading"""
        self.services[service_config.name] = service_config
        self.logger.info(f"Registered service for config watching: {service_config.name}")
    
    def get_config(self, file_path: str) -> Optional[Any]:
        """Get cached configuration content"""
        return self.config_cache.get(file_path)
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all cached configurations"""
        return self.config_cache.copy()
    
    def start_watching(self):
        """Start watching for configuration changes"""
        self.logger.info("🔍 Starting configuration watcher...")
        
        # Setup file system observers
        handler = ConfigFileHandler(self)
        
        for watch_dir in self.watch_dirs:
            if watch_dir.exists():
                self.observer.schedule(handler, str(watch_dir), recursive=True)
                self.logger.info(f"Watching config directory: {watch_dir}")
        
        # Log watched config files
        config_files = list(self.config_cache.keys())
        self.logger.info(f"Watching {len(config_files)} configuration files")
        
        self.observer.start()
        self.running = True
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_watching()
    
    def stop_watching(self):
        """Stop watching for configuration changes"""
        self.logger.info("🛑 Stopping configuration watcher...")
        self.running = False
        self.observer.stop()
        self.observer.join()
        
        # Cancel pending changes
        for timer in self.pending_changes.values():
            timer.cancel()
        self.pending_changes.clear()

def create_example_services(project_root: Path) -> List[ServiceConfig]:
    """Create example service configurations"""
    return [
        ServiceConfig(
            name="backend",
            command=["python", "start_server.py"],
            cwd=project_root / "backend",
            config_files=["backend/config.json", "config/unified-config.yaml", ".env"]
        ),
        ServiceConfig(
            name="frontend",
            command=["npm", "run", "dev"],
            cwd=project_root / "frontend",
            config_files=["frontend/package.json", "frontend/.env", "config/unified-config.yaml"]
        )
    ]

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration watcher with hot-reloading")
    parser.add_argument('--files', action='append', help='Specific config files to watch')
    parser.add_argument('--reload-services', action='store_true', help='Enable service reloading')
    parser.add_argument('--debounce', type=float, default=0.5, help='Debounce delay in seconds')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create watcher
    project_root = Path.cwd()
    watcher = ConfigWatcher(project_root)
    watcher.debounce_delay = args.debounce
    
    # Register example change handler
    def log_change_handler(change: ConfigChange):
        logging.info(f"📝 Config change: {change.file_path.name} ({change.change_type})")
    
    watcher.register_change_handler(log_change_handler)
    
    # Register services if requested
    if args.reload_services:
        services = create_example_services(project_root)
        for service in services:
            watcher.register_service(service)
    
    # Override watch files if specified
    if args.files:
        specific_files = [Path(f) for f in args.files]
        watcher.watch_dirs = list(set(f.parent for f in specific_files))
        watcher.config_patterns = [f.name for f in specific_files]
        watcher._load_initial_configs()
    
    try:
        watcher.start_watching()
    except KeyboardInterrupt:
        print("\nConfiguration watcher stopped.")

if __name__ == "__main__":
    main()