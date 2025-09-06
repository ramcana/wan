"""
Version Manager
Handles version checking, updates, and migration scripts for the WAN2.2 installation system.
"""

import os
import json
import requests
import logging
import shutil
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from packaging import version

from interfaces import InstallationError, ErrorCategory
from rollback_manager import RollbackManager


@dataclass
class UpdateInfo:
    """Information about available updates."""
    current_version: str
    latest_version: str
    update_available: bool
    release_url: str
    release_notes: str
    download_url: str
    published_at: str
    prerelease: bool
    size_bytes: Optional[int] = None


@dataclass
class MigrationInfo:
    """Information about a migration script."""
    from_version: str
    to_version: str
    script_path: str
    description: str
    required: bool = True


class VersionManager:
    """
    Manages version checking, updates, and migrations for the WAN2.2 installation system.
    """
    
    def __init__(self, installation_path: str, dry_run: bool = False):
        self.installation_path = Path(installation_path)
        self.dry_run = dry_run
        self.version_file = self.installation_path / "version.json"
        self.migrations_dir = self.installation_path / "migrations"
        self.backup_dir = self.installation_path / ".wan22_backup"
        
        self.logger = logging.getLogger(__name__)
        
        # GitHub repository information
        self.github_repo = "wan22/wan22-installer"  # Replace with actual repo
        self.github_api_base = "https://api.github.com/repos"
        
        # Initialize rollback manager for backup operations
        self.rollback_manager = RollbackManager(installation_path, dry_run)
        
        # Initialize version tracking
        self._initialize_version_tracking()
    
    def _initialize_version_tracking(self) -> None:
        """Initialize version tracking system."""
        if not self.version_file.exists():
            # Create initial version file
            initial_version = {
                "version": "1.0.0",
                "installed_at": datetime.now().isoformat(),
                "last_update_check": None,
                "update_channel": "stable",  # stable, beta, alpha
                "auto_update": False
            }
            
            if not self.dry_run:
                with open(self.version_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_version, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("Initialized version tracking")
        
        # Ensure migrations directory exists
        if not self.dry_run:
            self.migrations_dir.mkdir(exist_ok=True)
    
    def get_current_version(self) -> str:
        """Get the current installed version."""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
                return version_data.get("version", "1.0.0")
            else:
                return "1.0.0"
        except Exception as e:
            self.logger.error(f"Failed to read version file: {e}")
            return "1.0.0"
    
    def check_for_updates(self, include_prerelease: bool = False) -> UpdateInfo:
        """
        Check for available updates via GitHub releases API.
        
        Args:
            include_prerelease: Whether to include pre-release versions
            
        Returns:
            UpdateInfo object with update details
        """
        current_ver = self.get_current_version()
        
        try:
            # Get latest release from GitHub API
            url = f"{self.github_api_base}/{self.github_repo}/releases"
            if not include_prerelease:
                url += "/latest"
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'WAN22-Installer-VersionManager'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            if include_prerelease:
                releases = response.json()
                # Find the latest release (including pre-releases)
                latest_release = releases[0] if releases else None
            else:
                latest_release = response.json()
            
            if not latest_release:
                return UpdateInfo(
                    current_version=current_ver,
                    latest_version=current_ver,
                    update_available=False,
                    release_url="",
                    release_notes="No releases found",
                    download_url="",
                    published_at="",
                    prerelease=False
                )
            
            latest_ver = latest_release['tag_name'].lstrip('v')
            update_available = version.parse(latest_ver) > version.parse(current_ver)
            
            # Find download URL for the installer
            download_url = ""
            size_bytes = None
            for asset in latest_release.get('assets', []):
                if asset['name'].endswith('.zip') or 'installer' in asset['name'].lower():
                    download_url = asset['download_url']
                    size_bytes = asset.get('size')
                    break
            
            # Update last check time
            self._update_last_check_time()
            
            return UpdateInfo(
                current_version=current_ver,
                latest_version=latest_ver,
                update_available=update_available,
                release_url=latest_release['html_url'],
                release_notes=latest_release.get('body', ''),
                download_url=download_url,
                published_at=latest_release['published_at'],
                prerelease=latest_release.get('prerelease', False),
                size_bytes=size_bytes
            )
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to check for updates: {e}")
            return UpdateInfo(
                current_version=current_ver,
                latest_version=current_ver,
                update_available=False,
                release_url="",
                release_notes=f"Update check failed: {str(e)}",
                download_url="",
                published_at="",
                prerelease=False
            )
        except Exception as e:
            self.logger.error(f"Unexpected error checking for updates: {e}")
            return UpdateInfo(
                current_version=current_ver,
                latest_version=current_ver,
                update_available=False,
                release_url="",
                release_notes=f"Update check error: {str(e)}",
                download_url="",
                published_at="",
                prerelease=False
            )
    
    def download_update(self, update_info: UpdateInfo, 
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """
        Download an update package.
        
        Args:
            update_info: Update information from check_for_updates
            progress_callback: Optional callback for download progress (bytes_downloaded, total_bytes)
            
        Returns:
            Path to downloaded update package
        """
        if not update_info.download_url:
            raise InstallationError(
                "No download URL available for update",
                ErrorCategory.NETWORK,
                ["Check internet connection", "Try again later"]
            )
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would download update from: {update_info.download_url}")
            return str(self.backup_dir / f"update_{update_info.latest_version}.zip")
        
        try:
            # Create downloads directory
            downloads_dir = self.backup_dir / "downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            filename = f"wan22_installer_v{update_info.latest_version}.zip"
            download_path = downloads_dir / filename
            
            self.logger.info(f"Downloading update v{update_info.latest_version}...")
            
            response = requests.get(update_info.download_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded_size, total_size)
            
            self.logger.info(f"Downloaded update to: {download_path}")
            return str(download_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download update: {e}")
            raise InstallationError(
                f"Failed to download update: {str(e)}",
                ErrorCategory.NETWORK,
                ["Check internet connection", "Check available disk space", "Try again later"]
            )
    
    def backup_current_installation(self) -> str:
        """
        Create a backup of the current installation before update.
        
        Returns:
            Snapshot ID of the backup
        """
        current_version = self.get_current_version()
        
        # Files to backup before update
        files_to_backup = [
            "config.json",
            "version.json",
            "main.py",
            "ui.py",
            "utils.py"
        ]
        
        # Directories to backup
        dirs_to_backup = [
            "scripts",
            "resources",
            "logs"
        ]
        
        # Filter existing files/directories
        existing_files = [f for f in files_to_backup 
                         if (self.installation_path / f).exists()]
        existing_dirs = [d for d in dirs_to_backup 
                        if (self.installation_path / d).exists()]
        
        snapshot_id = self.rollback_manager.create_snapshot(
            description=f"Pre-update backup (v{current_version})",
            phase="pre-update",
            files_to_backup=existing_files,
            dirs_to_backup=existing_dirs
        )
        
        self.logger.info(f"Created pre-update backup: {snapshot_id}")
        return snapshot_id
    
    def run_migration(self, from_version: str, to_version: str) -> bool:
        """
        Run migration scripts between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            True if migration was successful
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would run migration from v{from_version} to v{to_version}")
            return True
        
        try:
            # Find applicable migration scripts
            migrations = self._find_migration_scripts(from_version, to_version)
            
            if not migrations:
                self.logger.info(f"No migrations needed from v{from_version} to v{to_version}")
                return True
            
            self.logger.info(f"Running {len(migrations)} migration(s)...")
            
            # Create pre-migration backup
            pre_migration_snapshot = self.rollback_manager.create_snapshot(
                description=f"Pre-migration backup (v{from_version} -> v{to_version})",
                phase="pre-migration"
            )
            
            # Run migrations in order
            for migration in migrations:
                self.logger.info(f"Running migration: {migration.description}")
                
                if not self._execute_migration_script(migration):
                    self.logger.error(f"Migration failed: {migration.script_path}")
                    
                    # Restore from backup on failure
                    self.logger.info("Restoring from pre-migration backup...")
                    self.rollback_manager.restore_snapshot(pre_migration_snapshot)
                    
                    return False
            
            self.logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False
    
    def _find_migration_scripts(self, from_version: str, to_version: str) -> List[MigrationInfo]:
        """Find applicable migration scripts between versions."""
        migrations = []
        
        if not self.migrations_dir.exists():
            return migrations
        
        # Look for migration scripts
        for script_file in self.migrations_dir.glob("migrate_*.py"):
            try:
                # Parse filename: migrate_v1.0.0_to_v1.1.0.py
                filename = script_file.stem
                parts = filename.split('_')
                if len(parts) >= 4 and parts[0] == 'migrate' and parts[2] == 'to':
                    script_from = parts[1].lstrip('v')
                    script_to = parts[3].lstrip('v')
                    
                    # Check if this migration is applicable
                    if (version.parse(from_version) <= version.parse(script_from) and
                        version.parse(script_to) <= version.parse(to_version)):
                        
                        migrations.append(MigrationInfo(
                            from_version=script_from,
                            to_version=script_to,
                            script_path=str(script_file),
                            description=f"Migration from v{script_from} to v{script_to}"
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Failed to parse migration script {script_file}: {e}")
        
        # Sort migrations by version order
        migrations.sort(key=lambda m: version.parse(m.from_version))
        return migrations
    
    def _execute_migration_script(self, migration: MigrationInfo) -> bool:
        """Execute a migration script."""
        try:
            # Load migration module
            spec = importlib.util.spec_from_file_location("migration", migration.script_path)
            if not spec or not spec.loader:
                self.logger.error(f"Failed to load migration script: {migration.script_path}")
                return False
            
            migration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration_module)
            
            # Check for required functions
            if not hasattr(migration_module, 'migrate'):
                self.logger.error(f"Migration script missing 'migrate' function: {migration.script_path}")
                return False
            
            # Execute migration
            context = {
                'installation_path': str(self.installation_path),
                'from_version': migration.from_version,
                'to_version': migration.to_version,
                'logger': self.logger
            }
            
            result = migration_module.migrate(context)
            
            if result is False:
                self.logger.error(f"Migration script returned failure: {migration.script_path}")
                return False
            
            # Validate migration if validation function exists
            if hasattr(migration_module, 'validate_migration'):
                validation_result = migration_module.validate_migration(context)
                if validation_result is False:
                    self.logger.error(f"Migration validation failed: {migration.script_path}")
                    return False
            
            self.logger.info(f"Migration completed successfully: {migration.description}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute migration script {migration.script_path}: {e}")
            return False
    
    def update_version_info(self, new_version: str) -> None:
        """Update version information after successful update."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would update version to: {new_version}")
            return
        
        try:
            version_data = {}
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
            
            version_data.update({
                'version': new_version,
                'updated_at': datetime.now().isoformat(),
                'previous_version': self.get_current_version()
            })
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Updated version info to v{new_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to update version info: {e}")
    
    def _update_last_check_time(self) -> None:
        """Update the last update check timestamp."""
        if self.dry_run:
            return
        
        try:
            version_data = {}
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
            
            version_data['last_update_check'] = datetime.now().isoformat()
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to update last check time: {e}")
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get version history from snapshots and version file."""
        history = []
        
        try:
            # Get current version info
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
                    history.append({
                        'version': version_data.get('version', '1.0.0'),
                        'installed_at': version_data.get('installed_at'),
                        'updated_at': version_data.get('updated_at'),
                        'previous_version': version_data.get('previous_version'),
                        'current': True
                    })
            
            # Get version info from snapshots
            snapshots = self.rollback_manager.list_snapshots()
            for snapshot in snapshots:
                if 'update' in snapshot.description.lower() or 'migration' in snapshot.description.lower():
                    history.append({
                        'version': 'unknown',
                        'snapshot_id': snapshot.id,
                        'timestamp': snapshot.timestamp,
                        'description': snapshot.description,
                        'current': False
                    })
            
            # Sort by timestamp
            history.sort(key=lambda x: x.get('updated_at') or x.get('timestamp') or '', reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to get version history: {e}")
        
        return history
    
    def create_migration_template(self, from_version: str, to_version: str) -> str:
        """Create a template migration script."""
        script_name = f"migrate_v{from_version}_to_v{to_version}.py"
        script_path = self.migrations_dir / script_name
        
        template = f'''"""
Migration script from v{from_version} to v{to_version}
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import json
import logging
from pathlib import Path


def migrate(context):
    """
    Perform migration from v{from_version} to v{to_version}.
    
    Args:
        context: Dictionary containing:
            - installation_path: Path to installation directory
            - from_version: Source version
            - to_version: Target version
            - logger: Logger instance
    
    Returns:
        True if migration successful, False otherwise
    """
    installation_path = Path(context['installation_path'])
    logger = context['logger']
    
    logger.info(f"Starting migration from v{from_version} to v{to_version}")
    
    try:
        # Example: Update configuration file
        config_file = installation_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Add migration logic here
            # Example: config['new_setting'] = 'default_value'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info("Updated configuration file")
        
        # Add more migration steps here
        
        logger.info(f"Migration from v{from_version} to v{to_version} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {{e}}")
        return False


def validate_migration(context):
    """
    Validate that the migration was successful.
    
    Args:
        context: Migration context dictionary
    
    Returns:
        True if validation passes, False otherwise
    """
    installation_path = Path(context['installation_path'])
    logger = context['logger']
    
    try:
        # Add validation logic here
        # Example: Check that new configuration settings exist
        
        logger.info("Migration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Migration validation failed: {{e}}")
        return False
'''
        
        if not self.dry_run:
            self.migrations_dir.mkdir(exist_ok=True)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(template)
        
        self.logger.info(f"Created migration template: {script_path}")
        return str(script_path)