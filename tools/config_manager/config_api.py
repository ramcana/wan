"""
Configuration API and Management System

This module provides a comprehensive API for configuration management,
including get/set operations, hot-reloading, and change notifications.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .unified_config import UnifiedConfig
from .config_validator import ConfigurationValidator, ValidationResult


@dataclass
class ConfigChangeEvent:
    """Represents a configuration change event"""
    timestamp: datetime
    field_path: str
    old_value: Any
    new_value: Any
    source: str  # 'api', 'file', 'reload'


class ConfigurationChangeHandler:
    """Handles configuration change notifications"""
    
    def __init__(self):
        self.callbacks: List[Callable[[ConfigChangeEvent], None]] = []
    
    def register_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """Register a callback for configuration changes"""
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """Unregister a configuration change callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def notify_change(self, event: ConfigChangeEvent):
        """Notify all registered callbacks of a configuration change"""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logging.getLogger(__name__).error(f"Configuration change callback failed: {e}")


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes"""
    
    def __init__(self, config_api: 'ConfigurationAPI'):
        self.config_api = config_api
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if this is a configuration file we're watching
        if file_path == self.config_api.config_file_path:
            self.logger.info(f"Configuration file changed: {file_path}")
            self.config_api._reload_from_file()


class ConfigurationAPI:
    """
    Comprehensive configuration API with hot-reloading and change notifications
    """
    
    def __init__(
        self, 
        config_file_path: Optional[Union[str, Path]] = None,
        auto_reload: bool = True,
        validate_changes: bool = True
    ):
        self.config_file_path = Path(config_file_path) if config_file_path else Path("config/unified-config.yaml")
        self.auto_reload = auto_reload
        self.validate_changes = validate_changes
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Load initial configuration
        self._config = self._load_config()
        
        # Initialize validator
        self.validator = ConfigurationValidator() if validate_changes else None
        
        # Initialize change handler
        self.change_handler = ConfigurationChangeHandler()
        
        # Initialize file watcher
        self.file_observer = None
        if auto_reload:
            self._setup_file_watcher()
    
    def _load_config(self) -> UnifiedConfig:
        """Load configuration from file or create default"""
        if self.config_file_path.exists():
            try:
                return UnifiedConfig.from_file(self.config_file_path)
            except Exception as e:
                self.logger.error(f"Failed to load configuration from {self.config_file_path}: {e}")
        
        # Return default configuration
        return UnifiedConfig()
    
    def _setup_file_watcher(self):
        """Setup file system watcher for configuration changes"""
        if not self.config_file_path.exists():
            return
        
        try:
            self.file_observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            # Watch the directory containing the config file
            watch_dir = self.config_file_path.parent
            self.file_observer.schedule(event_handler, str(watch_dir), recursive=False)
            self.file_observer.start()
            
            self.logger.info(f"Started watching configuration file: {self.config_file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to setup file watcher: {e}")
    
    def _reload_from_file(self):
        """Reload configuration from file"""
        with self._lock:
            try:
                new_config = UnifiedConfig.from_file(self.config_file_path)
                
                # Compare with current config and generate change events
                self._detect_and_notify_changes(self._config, new_config, "file")
                
                self._config = new_config
                self.logger.info("Configuration reloaded from file")
                
            except Exception as e:
                self.logger.error(f"Failed to reload configuration: {e}")
    
    def _detect_and_notify_changes(
        self, 
        old_config: UnifiedConfig, 
        new_config: UnifiedConfig, 
        source: str
    ):
        """Detect changes between configurations and notify callbacks"""
        old_dict = old_config.to_dict()
        new_dict = new_config.to_dict()
        
        self._compare_dicts(old_dict, new_dict, "", source)
    
    def _compare_dicts(self, old_dict: Dict, new_dict: Dict, path_prefix: str, source: str):
        """Recursively compare dictionaries and generate change events"""
        all_keys = set(old_dict.keys()) | set(new_dict.keys())
        
        for key in all_keys:
            field_path = f"{path_prefix}.{key}" if path_prefix else key
            
            old_value = old_dict.get(key)
            new_value = new_dict.get(key)
            
            if old_value != new_value:
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    # Recursively compare nested dictionaries
                    self._compare_dicts(old_value, new_value, field_path, source)
                else:
                    # Value changed
                    event = ConfigChangeEvent(
                        timestamp=datetime.now(),
                        field_path=field_path,
                        old_value=old_value,
                        new_value=new_value,
                        source=source
                    )
                    self.change_handler.notify_change(event)
    
    def get_config(self, path: Optional[str] = None) -> Any:
        """
        Get configuration value by path
        
        Args:
            path: Dot-separated path to configuration value (e.g., 'api.port')
                 If None, returns entire configuration
        
        Returns:
            Configuration value or entire configuration
        """
        with self._lock:
            if path is None:
                return self._config
            
            return self._config.get_config_path(path)
    
    def set_config(self, path: str, value: Any, validate: bool = None) -> bool:
        """
        Set configuration value by path
        
        Args:
            path: Dot-separated path to configuration value
            value: New value to set
            validate: Whether to validate the change (uses instance default if None)
        
        Returns:
            True if successful, False otherwise
        """
        validate = validate if validate is not None else self.validate_changes
        
        with self._lock:
            try:
                # Get old value for change notification
                old_value = self._config.get_config_path(path)
                
                # Create a copy of the config for validation
                if validate and self.validator:
                    test_config = UnifiedConfig.from_dict(self._config.to_dict())
                    test_config.set_config_path(path, value)
                    
                    # Validate the change
                    result = self.validator.validate_config(test_config)
                    if result.has_errors():
                        self.logger.error(f"Configuration validation failed for {path}: {result.errors_count} errors")
                        return False
                
                # Apply the change
                self._config.set_config_path(path, value)
                
                # Notify change
                event = ConfigChangeEvent(
                    timestamp=datetime.now(),
                    field_path=path,
                    old_value=old_value,
                    new_value=value,
                    source="api"
                )
                self.change_handler.notify_change(event)
                
                self.logger.info(f"Configuration updated: {path} = {value}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set configuration {path}: {e}")
                return False
    
    def update_config(self, updates: Dict[str, Any], validate: bool = None) -> Dict[str, bool]:
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of path -> value updates
            validate: Whether to validate changes
        
        Returns:
            Dictionary of path -> success status
        """
        results = {}
        
        for path, value in updates.items():
            results[path] = self.set_config(path, value, validate)
        
        return results
    
    def reload_config(self) -> bool:
        """
        Manually reload configuration from file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._reload_from_file()
            return True
        except Exception as e:
            self.logger.error(f"Manual reload failed: {e}")
            return False
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save current configuration to file
        
        Args:
            file_path: Path to save to (uses default if None)
        
        Returns:
            True if successful, False otherwise
        """
        save_path = Path(file_path) if file_path else self.config_file_path
        
        with self._lock:
            try:
                self._config.save_to_file(save_path)
                self.logger.info(f"Configuration saved to {save_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")
                return False
    
    def validate_current_config(self) -> ValidationResult:
        """
        Validate the current configuration
        
        Returns:
            Validation result
        """
        if not self.validator:
            raise RuntimeError("Validator not available")
        
        with self._lock:
            return self.validator.validate_config(self._config)
    
    def register_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """Register a callback for configuration changes"""
        self.change_handler.register_callback(callback)
    
    def unregister_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """Unregister a configuration change callback"""
        self.change_handler.unregister_callback(callback)
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration
        
        Returns:
            Dictionary with configuration metadata
        """
        with self._lock:
            return {
                'config_file_path': str(self.config_file_path),
                'config_exists': self.config_file_path.exists(),
                'auto_reload': self.auto_reload,
                'validate_changes': self.validate_changes,
                'file_watcher_active': self.file_observer is not None and self.file_observer.is_alive(),
                'config_version': self._config.config_version,
                'last_updated': self._config.last_updated,
                'system_name': self._config.system.name,
                'system_version': self._config.system.version,
                'environment': self._config.system.environment.value if hasattr(self._config.system.environment, 'value') else str(self._config.system.environment)
            }
    
    def apply_environment_overrides(self, environment: str) -> bool:
        """
        Apply environment-specific configuration overrides
        
        Args:
            environment: Environment name (development, staging, production, testing)
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                old_config = UnifiedConfig.from_dict(self._config.to_dict())
                self._config = self._config.apply_environment_overrides(environment)
                
                # Detect and notify changes
                self._detect_and_notify_changes(old_config, self._config, f"environment:{environment}")
                
                self.logger.info(f"Applied environment overrides for: {environment}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to apply environment overrides: {e}")
                return False
    
    def export_config(self, format: str = 'yaml') -> str:
        """
        Export current configuration as string
        
        Args:
            format: Export format ('yaml', 'json')
        
        Returns:
            Configuration as formatted string
        """
        with self._lock:
            if format.lower() == 'json':
                return self._config.to_json()
            elif format.lower() in ['yaml', 'yml']:
                return self._config.to_yaml()
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def import_config(self, config_str: str, format: str = 'auto') -> bool:
        """
        Import configuration from string
        
        Args:
            config_str: Configuration as string
            format: Format of the string ('yaml', 'json', 'auto')
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                old_config = UnifiedConfig.from_dict(self._config.to_dict())
                
                if format == 'auto':
                    # Try to detect format
                    config_str_stripped = config_str.strip()
                    if config_str_stripped.startswith('{'):
                        format = 'json'
                    else:
                        format = 'yaml'
                
                if format.lower() == 'json':
                    new_config = UnifiedConfig.from_json(config_str)
                elif format.lower() in ['yaml', 'yml']:
                    new_config = UnifiedConfig.from_yaml(config_str)
                else:
                    raise ValueError(f"Unsupported import format: {format}")
                
                # Validate if enabled
                if self.validate_changes and self.validator:
                    result = self.validator.validate_config(new_config)
                    if result.has_errors():
                        self.logger.error(f"Imported configuration validation failed: {result.errors_count} errors")
                        return False
                
                # Apply the new configuration
                self._config = new_config
                
                # Detect and notify changes
                self._detect_and_notify_changes(old_config, new_config, "import")
                
                self.logger.info("Configuration imported successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to import configuration: {e}")
                return False
    
    def shutdown(self):
        """Shutdown the configuration API and cleanup resources"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.logger.info("Configuration file watcher stopped")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()