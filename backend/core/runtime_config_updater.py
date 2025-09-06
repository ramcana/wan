"""
Runtime Configuration Update System

Provides hot-reload capabilities for configuration changes without requiring
application restart, with proper validation and rollback mechanisms.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from .enhanced_model_config import (
    ConfigurationManager, get_config_manager,
    UserPreferences, AdminPolicies, FeatureFlag
)

logger = logging.getLogger(__name__)


if WATCHDOG_AVAILABLE:
    class ConfigurationChangeHandler(FileSystemEventHandler):
        """File system event handler for configuration file changes"""
        
        def __init__(self, updater: 'RuntimeConfigurationUpdater'):
            self.updater = updater
            self.last_modified = {}
            self.debounce_delay = 1.0  # seconds
        
        def on_modified(self, event):
            """Handle file modification events"""
            if event.is_directory:
                return
            
            file_path = Path(event.src_path)
            
            # Check if it's a configuration file
            if file_path.suffix == '.json' and 'config' in file_path.name.lower():
                # Debounce rapid file changes
                now = datetime.now()
                last_mod = self.last_modified.get(file_path, datetime.min)
                
                if (now - last_mod).total_seconds() > self.debounce_delay:
                    self.last_modified[file_path] = now
                    asyncio.create_task(self.updater.handle_config_file_change(str(file_path)))
else:
    ConfigurationChangeHandler = None


class RuntimeConfigurationUpdater:
    """Manages runtime configuration updates without application restart"""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or get_config_manager()
        self.observers: List[Observer] = []
        self.update_callbacks: Dict[str, List[Callable]] = {
            'user_preferences': [],
            'admin_policies': [],
            'feature_flags': [],
            'any': []  # Called for any configuration change
        }
        self.rollback_stack: List[Dict[str, Any]] = []
        self.max_rollback_entries = 10
        self.is_monitoring = False
        
        # Add configuration change observer
        self.config_manager.add_observer(self._on_config_change)
    
    def start_monitoring(self, watch_paths: Optional[List[str]] = None) -> None:
        """
        Start monitoring configuration files for changes
        
        Args:
            watch_paths: List of paths to monitor, defaults to config file directory
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("File monitoring not available - watchdog module not installed")
            return
            
        if self.is_monitoring:
            logger.warning("Configuration monitoring is already active")
            return
        
        if watch_paths is None:
            # Monitor the directory containing the config file
            config_dir = self.config_manager.config_path.parent
            watch_paths = [str(config_dir)]
        
        for path in watch_paths:
            if Path(path).exists():
                observer = Observer()
                event_handler = ConfigurationChangeHandler(self)
                observer.schedule(event_handler, path, recursive=False)
                observer.start()
                self.observers.append(observer)
                logger.info(f"Started monitoring configuration changes in: {path}")
            else:
                logger.warning(f"Watch path does not exist: {path}")
        
        self.is_monitoring = True
    
    def stop_monitoring(self) -> None:
        """Stop monitoring configuration files"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        self.observers.clear()
        self.is_monitoring = False
        logger.info("Stopped monitoring configuration changes")
    
    def add_update_callback(self, section: str, callback: Callable) -> None:
        """
        Add callback for configuration updates
        
        Args:
            section: Configuration section ('user_preferences', 'admin_policies', 'feature_flags', 'any')
            callback: Async function to call on updates
        """
        if section not in self.update_callbacks:
            raise ValueError(f"Invalid section: {section}")
        
        self.update_callbacks[section].append(callback)
        logger.debug(f"Added update callback for section: {section}")
    
    def remove_update_callback(self, section: str, callback: Callable) -> None:
        """Remove configuration update callback"""
        if section in self.update_callbacks and callback in self.update_callbacks[section]:
            self.update_callbacks[section].remove(callback)
            logger.debug(f"Removed update callback for section: {section}")
    
    async def update_user_preferences_runtime(self, preferences: UserPreferences, validate: bool = True) -> bool:
        """
        Update user preferences at runtime with validation
        
        Args:
            preferences: New user preferences
            validate: Whether to validate before applying
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Create rollback point
            self._create_rollback_point()
            
            # Validate if requested
            if validate:
                validation_result = self.config_manager.validate_user_preferences(preferences)
                if not validation_result.is_valid:
                    logger.error("Runtime preferences update failed validation:")
                    for error in validation_result.errors:
                        logger.error(f"  - {error.field}: {error.message}")
                    return False
            
            # Apply update
            success = await self.config_manager.update_user_preferences(preferences)
            
            if success:
                logger.info("User preferences updated at runtime")
                await self._notify_callbacks('user_preferences', preferences)
                return True
            else:
                logger.error("Failed to update user preferences at runtime")
                await self.rollback_last_change()
                return False
                
        except Exception as e:
            logger.error(f"Runtime preferences update failed: {e}")
            await self.rollback_last_change()
            return False
    
    async def update_admin_policies_runtime(self, policies: AdminPolicies, validate: bool = True) -> bool:
        """
        Update admin policies at runtime with validation
        
        Args:
            policies: New admin policies
            validate: Whether to validate before applying
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Create rollback point
            self._create_rollback_point()
            
            # Validate if requested
            if validate:
                validation_result = self.config_manager.validate_admin_policies(policies)
                if not validation_result.is_valid:
                    logger.error("Runtime policies update failed validation:")
                    for error in validation_result.errors:
                        logger.error(f"  - {error.field}: {error.message}")
                    return False
            
            # Apply update
            success = await self.config_manager.update_admin_policies(policies)
            
            if success:
                logger.info("Admin policies updated at runtime")
                await self._notify_callbacks('admin_policies', policies)
                return True
            else:
                logger.error("Failed to update admin policies at runtime")
                await self.rollback_last_change()
                return False
                
        except Exception as e:
            logger.error(f"Runtime policies update failed: {e}")
            await self.rollback_last_change()
            return False
    
    async def update_feature_flag_runtime(self, flag: FeatureFlag, enabled: bool, user_id: Optional[str] = None) -> bool:
        """
        Update feature flag at runtime
        
        Args:
            flag: Feature flag to update
            enabled: New enabled state
            user_id: Optional user ID for user-specific override
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Create rollback point
            self._create_rollback_point()
            
            # Apply update
            success = await self.config_manager.update_feature_flag(flag, enabled, user_id)
            
            if success:
                logger.info(f"Feature flag {flag.value} updated at runtime: {enabled}")
                await self._notify_callbacks('feature_flags', {'flag': flag, 'enabled': enabled, 'user_id': user_id})
                return True
            else:
                logger.error(f"Failed to update feature flag {flag.value} at runtime")
                await self.rollback_last_change()
                return False
                
        except Exception as e:
            logger.error(f"Runtime feature flag update failed: {e}")
            await self.rollback_last_change()
            return False
    
    async def handle_config_file_change(self, file_path: str) -> None:
        """
        Handle external configuration file changes
        
        Args:
            file_path: Path to changed configuration file
        """
        try:
            logger.info(f"Detected configuration file change: {file_path}")
            
            # Check if it's our main config file
            if Path(file_path) == self.config_manager.config_path:
                # Reload configuration
                old_config = self.config_manager.config
                
                # Create rollback point
                self._create_rollback_point()
                
                # Reload from file
                success = self.config_manager.load_configuration()
                
                if success:
                    logger.info("Configuration reloaded from file")
                    
                    # Notify callbacks about the change
                    await self._notify_callbacks('any', {
                        'type': 'file_reload',
                        'file_path': file_path,
                        'old_config': old_config,
                        'new_config': self.config_manager.config
                    })
                else:
                    logger.error("Failed to reload configuration from file")
                    await self.rollback_last_change()
            
        except Exception as e:
            logger.error(f"Error handling config file change: {e}")
            await self.rollback_last_change()
    
    async def rollback_last_change(self) -> bool:
        """
        Rollback the last configuration change
        
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            if not self.rollback_stack:
                logger.warning("No configuration changes to rollback")
                return False
            
            # Get last rollback point
            rollback_data = self.rollback_stack.pop()
            
            # Restore configuration
            self.config_manager.config = self.config_manager._deserialize_config(rollback_data)
            
            # Save restored configuration
            success = self.config_manager.save_configuration()
            
            if success:
                logger.info("Configuration rolled back successfully")
                await self._notify_callbacks('any', {
                    'type': 'rollback',
                    'timestamp': datetime.now()
                })
                return True
            else:
                logger.error("Failed to save rolled back configuration")
                return False
                
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False
    
    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """Get list of available rollback points"""
        return [
            {
                'index': i,
                'timestamp': entry.get('timestamp', 'unknown'),
                'version': entry.get('version', 'unknown')
            }
            for i, entry in enumerate(self.rollback_stack)
        ]
    
    def _create_rollback_point(self) -> None:
        """Create a rollback point with current configuration"""
        try:
            # Serialize current configuration
            rollback_data = self.config_manager._serialize_config(self.config_manager.config)
            rollback_data['rollback_timestamp'] = datetime.now().isoformat()
            
            # Add to rollback stack
            self.rollback_stack.append(rollback_data)
            
            # Limit rollback stack size
            if len(self.rollback_stack) > self.max_rollback_entries:
                self.rollback_stack.pop(0)
            
            logger.debug("Created configuration rollback point")
            
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
    
    async def _on_config_change(self, change_type: str, section: str, old_value: Any, new_value: Any) -> None:
        """Handle configuration changes from the configuration manager"""
        logger.debug(f"Configuration change detected: {change_type} in {section}")
        
        # Notify section-specific callbacks
        if section in self.update_callbacks:
            await self._notify_callbacks(section, new_value)
        
        # Notify general callbacks
        await self._notify_callbacks('any', {
            'type': change_type,
            'section': section,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.now()
        })
    
    async def _notify_callbacks(self, section: str, data: Any) -> None:
        """Notify callbacks about configuration changes"""
        callbacks = self.update_callbacks.get(section, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in configuration update callback: {e}")
    
    def __del__(self):
        """Cleanup when updater is destroyed"""
        if self.is_monitoring:
            self.stop_monitoring()


# Global runtime updater instance
_runtime_updater: Optional[RuntimeConfigurationUpdater] = None


def get_runtime_updater() -> RuntimeConfigurationUpdater:
    """Get global runtime configuration updater instance"""
    global _runtime_updater
    if _runtime_updater is None:
        _runtime_updater = RuntimeConfigurationUpdater()
    return _runtime_updater


def reset_runtime_updater() -> None:
    """Reset global runtime updater (for testing)"""
    global _runtime_updater
    if _runtime_updater is not None:
        _runtime_updater.stop_monitoring()
    _runtime_updater = None