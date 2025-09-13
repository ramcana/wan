"""
User preference management system for the startup manager.
Handles persistent user preferences, configuration migration, and backup/restore.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from packaging import version

from .config import StartupConfig, ConfigLoader


class UserPreferences(BaseModel):
    """User preferences for startup manager behavior."""
    
    # UI Preferences
    auto_open_browser: bool = Field(default=True, description="Automatically open browser after successful startup")
    show_progress_bars: bool = Field(default=True, description="Show progress bars during operations")
    verbose_output: bool = Field(default=False, description="Show verbose output by default")
    confirm_destructive_actions: bool = Field(default=True, description="Ask for confirmation before destructive actions")
    
    # Recovery Preferences
    preferred_recovery_strategy: str = Field(default="auto", pattern="^(auto|manual|aggressive)$", description="Preferred recovery strategy")
    auto_retry_failed_operations: bool = Field(default=True, description="Automatically retry failed operations")
    max_auto_retries: int = Field(default=3, ge=1, le=10, description="Maximum automatic retry attempts")
    
    # Port Management Preferences
    preferred_backend_port: Optional[int] = Field(default=None, ge=1024, le=65535, description="Preferred backend port")
    preferred_frontend_port: Optional[int] = Field(default=None, ge=1024, le=65535, description="Preferred frontend port")
    allow_port_auto_increment: bool = Field(default=True, description="Allow automatic port increment when preferred ports are taken")
    
    # Security Preferences
    allow_admin_elevation: bool = Field(default=True, description="Allow automatic admin elevation when needed")
    trust_local_processes: bool = Field(default=False, description="Trust local processes when resolving port conflicts")
    
    # Logging Preferences
    keep_detailed_logs: bool = Field(default=True, description="Keep detailed logs of all operations")
    log_retention_days: int = Field(default=30, ge=1, le=365, description="Number of days to keep log files")
    
    # Advanced Preferences
    enable_experimental_features: bool = Field(default=False, description="Enable experimental features")
    startup_timeout_multiplier: float = Field(default=1.0, ge=0.5, le=5.0, description="Multiplier for startup timeouts")


class ConfigurationVersion(BaseModel):
    """Configuration version information for migration tracking."""
    version: str = Field(description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now)
    startup_manager_version: str = Field(default="2.0.0", description="Startup manager version")
    migration_notes: List[str] = Field(default_factory=list, description="Migration notes")


class PreferenceManager:
    """Manages user preferences, configuration migration, and backup/restore."""
    
    def __init__(self, preferences_dir: Optional[Path] = None):
        self.preferences_dir = preferences_dir or Path.home() / ".wan22" / "startup_manager"
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        
        self.preferences_file = self.preferences_dir / "preferences.json"
        self.version_file = self.preferences_dir / "version.json"
        self.backup_dir = self.preferences_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self._preferences: Optional[UserPreferences] = None
        self._version_info: Optional[ConfigurationVersion] = None
    
    def load_preferences(self) -> UserPreferences:
        """Load user preferences from file or create defaults."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    preferences_data = json.load(f)
                self._preferences = UserPreferences(**preferences_data)
            except (json.JSONDecodeError, ValueError) as e:
                # If preferences are corrupted, create backup and use defaults
                self._backup_corrupted_file(self.preferences_file, f"corrupted_preferences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                self._preferences = UserPreferences()
                self.save_preferences()
        else:
            self._preferences = UserPreferences()
            self.save_preferences()
        
        return self._preferences
    
    def save_preferences(self) -> None:
        """Save current preferences to file."""
        if self._preferences is None:
            raise ValueError("No preferences loaded to save")
        
        preferences_dict = self._preferences.model_dump()
        with open(self.preferences_file, 'w', encoding='utf-8') as f:
            json.dump(preferences_dict, f, indent=2, ensure_ascii=False)
    
    def load_version_info(self) -> ConfigurationVersion:
        """Load configuration version information."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
                    # Handle datetime deserialization
                    if 'created_at' in version_data and isinstance(version_data['created_at'], str):
                        version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                self._version_info = ConfigurationVersion(**version_data)
            except (json.JSONDecodeError, ValueError) as e:
                # Create new version info if corrupted
                self._version_info = ConfigurationVersion(version="2.0.0")
                self.save_version_info()
        else:
            self._version_info = ConfigurationVersion(version="2.0.0")
            self.save_version_info()
        
        return self._version_info
    
    def save_version_info(self) -> None:
        """Save version information to file."""
        if self._version_info is None:
            raise ValueError("No version info loaded to save")
        
        version_dict = self._version_info.model_dump()
        # Handle datetime serialization
        if isinstance(version_dict['created_at'], datetime):
            version_dict['created_at'] = version_dict['created_at'].isoformat()
        
        with open(self.version_file, 'w', encoding='utf-8') as f:
            json.dump(version_dict, f, indent=2, ensure_ascii=False)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """Create a backup of current configuration and preferences."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Backup preferences
        if self.preferences_file.exists():
            shutil.copy2(self.preferences_file, backup_path / "preferences.json")
        
        # Backup version info
        if self.version_file.exists():
            shutil.copy2(self.version_file, backup_path / "version.json")
        
        # Backup main configuration if it exists
        config_file = Path("startup_config.json")
        if config_file.exists():
            shutil.copy2(config_file, backup_path / "startup_config.json")
        
        # Create backup manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "files": [
                "preferences.json",
                "version.json",
                "startup_config.json"
            ],
            "description": f"Automatic backup created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        with open(backup_path / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return backup_path
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore configuration from a backup."""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            raise ValueError(f"Backup '{backup_name}' not found")
        
        # Create current backup before restoring
        self.create_backup("pre_restore_backup")
        
        try:
            # Restore preferences
            backup_preferences = backup_path / "preferences.json"
            if backup_preferences.exists():
                shutil.copy2(backup_preferences, self.preferences_file)
            
            # Restore version info
            backup_version = backup_path / "version.json"
            if backup_version.exists():
                shutil.copy2(backup_version, self.version_file)
            
            # Restore main configuration
            backup_config = backup_path / "startup_config.json"
            config_file = Path("startup_config.json")
            if backup_config.exists():
                shutil.copy2(backup_config, config_file)
            
            # Reload preferences and version info
            self._preferences = None
            self._version_info = None
            self.load_preferences()
            self.load_version_info()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to restore backup: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups with their information."""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                manifest_file = backup_dir / "manifest.json"
                backup_info = {
                    "name": backup_dir.name,
                    "path": str(backup_dir),
                    "created_at": None,
                    "description": "No description available",
                    "files": []
                }
                
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        backup_info.update(manifest)
                    except (json.JSONDecodeError, ValueError):
                        pass
                else:
                    # Fallback to directory modification time
                    backup_info["created_at"] = datetime.fromtimestamp(backup_dir.stat().st_mtime).isoformat()
                
                backups.append(backup_info)
        
        # Sort by creation time, newest first
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        removed_count = 0
        for backup in backups[keep_count:]:
            backup_path = Path(backup["path"])
            if backup_path.exists():
                shutil.rmtree(backup_path)
                removed_count += 1
        
        return removed_count
    
    def migrate_configuration(self, target_version: str = "2.0.0") -> bool:
        """Migrate configuration to target version."""
        current_version_info = self.load_version_info()
        current_version = current_version_info.version
        
        if version.parse(current_version) >= version.parse(target_version):
            return False  # No migration needed
        
        # Create backup before migration
        backup_path = self.create_backup(f"pre_migration_{current_version}_to_{target_version}")
        
        migration_notes = []
        
        try:
            # Migration from 1.x to 2.0.0
            if version.parse(current_version) < version.parse("2.0.0"):
                migration_notes.extend(self._migrate_to_2_0_0())
            
            # Update version info
            self._version_info = ConfigurationVersion(
                version=target_version,
                created_at=datetime.now(),
                startup_manager_version=target_version,
                migration_notes=migration_notes
            )
            self.save_version_info()
            
            return True
            
        except Exception as e:
            # Restore backup on migration failure
            try:
                self.restore_backup(backup_path.name)
            except Exception:
                pass  # Best effort restore
            
            raise RuntimeError(f"Configuration migration failed: {e}")
    
    def _migrate_to_2_0_0(self) -> List[str]:
        """Migrate configuration from 1.x to 2.0.0."""
        notes = []
        
        # Load current preferences and configuration
        preferences = self.load_preferences()
        
        # Try to load old configuration format
        old_config_file = Path("startup_config.json")
        if old_config_file.exists():
            try:
                with open(old_config_file, 'r', encoding='utf-8') as f:
                    old_config = json.load(f)
                
                # Migrate legacy settings to new structure
                if "verbose_logging" in old_config and old_config["verbose_logging"]:
                    preferences.verbose_output = True
                    notes.append("Migrated verbose_logging setting to preferences")
                
                if "auto_fix_issues" in old_config:
                    preferences.auto_retry_failed_operations = old_config["auto_fix_issues"]
                    notes.append("Migrated auto_fix_issues setting to preferences")
                
                # Create new structured configuration
                loader = ConfigLoader(old_config_file)
                config = loader.load_config(apply_env_overrides=False)
                loader.save_config()
                
                notes.append("Migrated configuration to new structured format")
                
            except Exception as e:
                notes.append(f"Warning: Could not fully migrate old configuration: {e}")
        
        # Save updated preferences
        self._preferences = preferences
        self.save_preferences()
        
        return notes
    
    def _backup_corrupted_file(self, file_path: Path, backup_name: str) -> None:
        """Backup a corrupted file for debugging."""
        corrupted_backup_path = self.backup_dir / backup_name
        try:
            shutil.copy2(file_path, corrupted_backup_path)
        except Exception:
            pass  # Best effort backup
    
    def apply_preferences_to_config(self, config: StartupConfig) -> StartupConfig:
        """Apply user preferences to a startup configuration."""
        if self._preferences is None:
            self.load_preferences()
        
        # Apply port preferences
        if self._preferences.preferred_backend_port:
            config.backend.port = self._preferences.preferred_backend_port
        
        if self._preferences.preferred_frontend_port:
            config.frontend.port = self._preferences.preferred_frontend_port
        
        # Apply auto port setting
        config.backend.auto_port = self._preferences.allow_port_auto_increment
        config.frontend.auto_port = self._preferences.allow_port_auto_increment
        
        # Apply browser setting
        config.frontend.open_browser = self._preferences.auto_open_browser
        
        # Apply recovery settings
        config.recovery.max_retry_attempts = self._preferences.max_auto_retries
        config.recovery.enabled = self._preferences.auto_retry_failed_operations
        
        # Apply security settings
        config.security.allow_admin_elevation = self._preferences.allow_admin_elevation
        
        # Apply timeout multiplier
        config.backend.timeout = int(config.backend.timeout * self._preferences.startup_timeout_multiplier)
        config.frontend.timeout = int(config.frontend.timeout * self._preferences.startup_timeout_multiplier)
        
        # Apply logging preferences
        if self._preferences.verbose_output:
            config.logging.level = "debug"
        
        return config
    
    @property
    def preferences(self) -> Optional[UserPreferences]:
        """Get current preferences."""
        return self._preferences
    
    @property
    def version_info(self) -> Optional[ConfigurationVersion]:
        """Get current version information."""
        return self._version_info


def load_preferences(preferences_dir: Optional[Path] = None) -> UserPreferences:
    """Convenience function to load user preferences."""
    manager = PreferenceManager(preferences_dir)
    return manager.load_preferences()
