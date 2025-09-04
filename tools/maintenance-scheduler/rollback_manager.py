import pytest
"""
Rollback management system for safe automated maintenance operations.
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import tempfile
import hashlib
from threading import Lock
from dataclasses import dataclass, field
from datetime import timedelta
import uuid

from models import MaintenanceTask, MaintenanceResult, MaintenanceHistory


@dataclass
class RollbackPoint:
    """Represents a rollback point for maintenance operations."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # Backup data
    file_backups: Dict[str, str] = field(default_factory=dict)  # original_path -> backup_path
    git_commit: Optional[str] = None
    database_backup: Optional[str] = None
    config_backup: Optional[str] = None
    
    # Metadata
    description: str = ""
    size_bytes: int = 0
    checksum: str = ""
    
    # Status
    valid: bool = True
    used: bool = False
    cleanup_after: Optional[datetime] = None


class RollbackManager:
    """
    Manages rollback points and operations for safe automated maintenance.
    
    Provides capabilities to:
    - Create rollback points before maintenance operations
    - Store file backups, git states, and configuration snapshots
    - Execute rollbacks when operations fail or need to be undone
    - Clean up old rollback points
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Storage configuration
        self.backup_root = Path(self.config.get('backup_root', 'data/maintenance/rollbacks'))
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Rollback points storage
        self.rollback_points: Dict[str, RollbackPoint] = {}
        self.rollback_lock = Lock()
        
        # Configuration
        self.max_rollback_points = self.config.get('max_rollback_points', 50)
        self.cleanup_after_days = self.config.get('cleanup_after_days', 30)
        self.max_backup_size_mb = self.config.get('max_backup_size_mb', 1000)
        
        # Load existing rollback points
        self._load_rollback_points()
        
        self.logger.info(f"RollbackManager initialized with {len(self.rollback_points)} rollback points")
    
    async def create_rollback_point(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Create a rollback point before executing a maintenance task."""
        if not task.rollback_enabled:
            return {}
        
        rollback_point = RollbackPoint(
            task_id=task.id,
            description=f"Rollback point for {task.name}"
        )
        
        try:
            self.logger.info(f"Creating rollback point for task: {task.name}")
            
            # Create backup directory
            backup_dir = self.backup_root / rollback_point.id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup files that might be modified
            await self._backup_files(task, rollback_point, backup_dir)
            
            # Backup git state
            await self._backup_git_state(rollback_point, backup_dir)
            
            # Backup configuration files
            await self._backup_configurations(task, rollback_point, backup_dir)
            
            # Calculate checksum and size
            rollback_point.checksum = self._calculate_backup_checksum(backup_dir)
            rollback_point.size_bytes = self._calculate_backup_size(backup_dir)
            
            # Set cleanup time
            rollback_point.cleanup_after = datetime.now() + timedelta(days=self.cleanup_after_days)
            
            # Store rollback point
            with self.rollback_lock:
                self.rollback_points[rollback_point.id] = rollback_point
                self._save_rollback_points()
            
            self.logger.info(
                f"Created rollback point {rollback_point.id} "
                f"({rollback_point.size_bytes / 1024 / 1024:.2f} MB)"
            )
            
            return {
                'rollback_id': rollback_point.id,
                'created_at': rollback_point.created_at.isoformat(),
                'size_bytes': rollback_point.size_bytes,
                'file_count': len(rollback_point.file_backups)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create rollback point: {e}", exc_info=True)
            # Clean up partial backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            return {}
    
    async def execute_rollback(self, rollback_id: str, reason: str = "") -> bool:
        """Execute a rollback operation."""
        rollback_point = self.rollback_points.get(rollback_id)
        if not rollback_point:
            self.logger.error(f"Rollback point not found: {rollback_id}")
            return False
        
        if rollback_point.used:
            self.logger.warning(f"Rollback point already used: {rollback_id}")
            return False
        
        if not rollback_point.valid:
            self.logger.error(f"Rollback point is invalid: {rollback_id}")
            return False
        
        try:
            self.logger.info(f"Executing rollback {rollback_id}: {reason}")
            
            backup_dir = self.backup_root / rollback_id
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # Verify backup integrity
            if not self._verify_backup_integrity(rollback_point, backup_dir):
                self.logger.error(f"Backup integrity check failed: {rollback_id}")
                return False
            
            # Restore files
            await self._restore_files(rollback_point, backup_dir)
            
            # Restore git state if available
            await self._restore_git_state(rollback_point, backup_dir)
            
            # Restore configurations
            await self._restore_configurations(rollback_point, backup_dir)
            
            # Mark rollback point as used
            rollback_point.used = True
            with self.rollback_lock:
                self._save_rollback_points()
            
            self.logger.info(f"Successfully executed rollback {rollback_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute rollback {rollback_id}: {e}", exc_info=True)
            return False
    
    def get_rollback_points(self, task_id: Optional[str] = None) -> List[RollbackPoint]:
        """Get available rollback points."""
        points = list(self.rollback_points.values())
        
        if task_id:
            points = [p for p in points if p.task_id == task_id]
        
        # Sort by creation time (most recent first)
        points.sort(key=lambda p: p.created_at, reverse=True)
        
        return points
    
    def cleanup_old_rollback_points(self) -> int:
        """Clean up old and invalid rollback points."""
        now = datetime.now()
        cleaned_count = 0
        
        with self.rollback_lock:
            points_to_remove = []
            
            for rollback_id, point in self.rollback_points.items():
                should_remove = False
                
                # Remove if past cleanup time
                if point.cleanup_after and now > point.cleanup_after:
                    should_remove = True
                    self.logger.debug(f"Removing expired rollback point: {rollback_id}")
                
                # Remove if used and old
                elif point.used and (now - point.created_at).days > 7:
                    should_remove = True
                    self.logger.debug(f"Removing used rollback point: {rollback_id}")
                
                # Remove if invalid
                elif not point.valid:
                    should_remove = True
                    self.logger.debug(f"Removing invalid rollback point: {rollback_id}")
                
                if should_remove:
                    points_to_remove.append(rollback_id)
                    
                    # Remove backup directory
                    backup_dir = self.backup_root / rollback_id
                    if backup_dir.exists():
                        try:
                            shutil.rmtree(backup_dir)
                        except Exception as e:
                            self.logger.warning(f"Failed to remove backup directory {backup_dir}: {e}")
            
            # Remove from memory
            for rollback_id in points_to_remove:
                del self.rollback_points[rollback_id]
                cleaned_count += 1
            
            if cleaned_count > 0:
                self._save_rollback_points()
        
        # Enforce maximum rollback points limit
        if len(self.rollback_points) > self.max_rollback_points:
            excess_count = len(self.rollback_points) - self.max_rollback_points
            oldest_points = sorted(
                self.rollback_points.items(),
                key=lambda x: x[1].created_at
            )[:excess_count]
            
            with self.rollback_lock:
                for rollback_id, point in oldest_points:
                    del self.rollback_points[rollback_id]
                    cleaned_count += 1
                    
                    # Remove backup directory
                    backup_dir = self.backup_root / rollback_id
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir, ignore_errors=True)
                
                if excess_count > 0:
                    self._save_rollback_points()
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} rollback points")
        
        return cleaned_count
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get statistics about rollback points."""
        points = list(self.rollback_points.values())
        
        total_size = sum(p.size_bytes for p in points)
        valid_points = [p for p in points if p.valid]
        used_points = [p for p in points if p.used]
        
        return {
            'total_rollback_points': len(points),
            'valid_rollback_points': len(valid_points),
            'used_rollback_points': len(used_points),
            'total_size_mb': total_size / 1024 / 1024,
            'average_size_mb': (total_size / len(points) / 1024 / 1024) if points else 0,
            'oldest_point': min(p.created_at for p in points) if points else None,
            'newest_point': max(p.created_at for p in points) if points else None
        }
    
    async def _backup_files(self, task: MaintenanceTask, rollback_point: RollbackPoint, backup_dir: Path) -> None:
        """Backup files that might be modified by the task."""
        # Determine files to backup based on task category and config
        files_to_backup = self._get_files_to_backup(task)
        
        files_backup_dir = backup_dir / "files"
        files_backup_dir.mkdir(exist_ok=True)
        
        for file_path in files_to_backup:
            if not file_path.exists():
                continue
            
            try:
                # Create relative backup path
                relative_path = file_path.relative_to(Path.cwd())
                backup_path = files_backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, backup_path)
                rollback_point.file_backups[str(file_path)] = str(backup_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to backup file {file_path}: {e}")
    
    async def _backup_git_state(self, rollback_point: RollbackPoint, backup_dir: Path) -> None:
        """Backup current git state."""
        try:
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                rollback_point.git_commit = result.stdout.strip()
                
                # Save git state info
                git_info = {
                    'commit': rollback_point.git_commit,
                    'branch': self._get_current_branch(),
                    'status': self._get_git_status(),
                    'timestamp': datetime.now().isoformat()
                }
                
                git_backup_file = backup_dir / "git_state.json"
                with open(git_backup_file, 'w') as f:
                    json.dump(git_info, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to backup git state: {e}")
    
    async def _backup_configurations(self, task: MaintenanceTask, rollback_point: RollbackPoint, backup_dir: Path) -> None:
        """Backup configuration files."""
        config_files = self._get_config_files_to_backup(task)
        
        if not config_files:
            return
        
        config_backup_dir = backup_dir / "configs"
        config_backup_dir.mkdir(exist_ok=True)
        
        for config_file in config_files:
            if not config_file.exists():
                continue
            
            try:
                backup_path = config_backup_dir / config_file.name
                shutil.copy2(config_file, backup_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to backup config file {config_file}: {e}")
    
    async def _restore_files(self, rollback_point: RollbackPoint, backup_dir: Path) -> None:
        """Restore files from backup."""
        for original_path, backup_path in rollback_point.file_backups.items():
            try:
                original_file = Path(original_path)
                backup_file = Path(backup_path)
                
                if not backup_file.exists():
                    self.logger.warning(f"Backup file not found: {backup_file}")
                    continue
                
                # Ensure parent directory exists
                original_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Restore file
                shutil.copy2(backup_file, original_file)
                self.logger.debug(f"Restored file: {original_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to restore file {original_path}: {e}")
    
    async def _restore_git_state(self, rollback_point: RollbackPoint, backup_dir: Path) -> None:
        """Restore git state if available."""
        if not rollback_point.git_commit:
            return
        
        try:
            # Reset to the backed up commit
            result = subprocess.run(
                ['git', 'reset', '--hard', rollback_point.git_commit],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info(f"Restored git state to commit: {rollback_point.git_commit}")
            else:
                self.logger.warning(f"Failed to restore git state: {result.stderr}")
                
        except Exception as e:
            self.logger.warning(f"Failed to restore git state: {e}")
    
    async def _restore_configurations(self, rollback_point: RollbackPoint, backup_dir: Path) -> None:
        """Restore configuration files."""
        config_backup_dir = backup_dir / "configs"
        if not config_backup_dir.exists():
            return
        
        for backup_file in config_backup_dir.iterdir():
            if not backup_file.is_file():
                continue
            
            try:
                # Restore to original location (assuming same name)
                original_file = Path.cwd() / backup_file.name
                shutil.copy2(backup_file, original_file)
                self.logger.debug(f"Restored config file: {original_file}")
                
            except Exception as e:
                self.logger.warning(f"Failed to restore config file {backup_file}: {e}")
    
    def _get_files_to_backup(self, task: MaintenanceTask) -> List[Path]:
        """Determine which files to backup based on task type."""
        files_to_backup = []
        
        # Common files that might be modified
        common_files = [
            Path("pyproject.toml"),
            Path("setup.py"),
            Path("requirements.txt"),
            Path("pytest.ini"),
            Path(".pre-commit-config.yaml")
        ]
        
        # Category-specific files
        if task.category == TaskCategory.CODE_QUALITY:
            files_to_backup.extend([
                Path("**/*.py"),  # This would need glob expansion
                Path(".flake8"),
                Path(".pylintrc"),
                Path("mypy.ini")
            ])
        elif task.category == TaskCategory.CONFIG_CLEANUP:
            files_to_backup.extend([
                Path("config/**/*.yaml"),
                Path("config/**/*.json"),
                Path("**/*.env")
            ])
        elif task.category == TaskCategory.TEST_MAINTENANCE:
            files_to_backup.extend([
                Path("tests/**/*.py"),
                Path("conftest.py")
            ])
        
        # Add common files
        files_to_backup.extend(common_files)
        
        # Expand globs and filter existing files
        expanded_files = []
        for pattern in files_to_backup:
            if "*" in str(pattern):
                # Handle glob patterns
                try:
                    expanded_files.extend(Path.cwd().glob(str(pattern)))
                except Exception:
                    pass
            else:
                expanded_files.append(pattern)
        
        # Filter to existing files and limit size
        existing_files = []
        total_size = 0
        max_size = self.max_backup_size_mb * 1024 * 1024
        
        for file_path in expanded_files:
            if file_path.exists() and file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    if total_size + file_size <= max_size:
                        existing_files.append(file_path)
                        total_size += file_size
                    else:
                        self.logger.warning(f"Skipping file due to size limit: {file_path}")
                except Exception:
                    pass
        
        return existing_files
    
    def _get_config_files_to_backup(self, task: MaintenanceTask) -> List[Path]:
        """Get configuration files to backup."""
        config_files = []
        
        # Common config files
        common_configs = [
            Path("config.json"),
            Path("config.yaml"),
            Path(".env"),
            Path("docker-compose.yml")
        ]
        
        for config_file in common_configs:
            if config_file.exists():
                config_files.append(config_file)
        
        return config_files
    
    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for backup directory."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(backup_dir.rglob("*")):
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                except Exception:
                    pass
        
        return hasher.hexdigest()
    
    def _calculate_backup_size(self, backup_dir: Path) -> int:
        """Calculate total size of backup directory."""
        total_size = 0
        
        for file_path in backup_dir.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except Exception:
                    pass
        
        return total_size
    
    def _verify_backup_integrity(self, rollback_point: RollbackPoint, backup_dir: Path) -> bool:
        """Verify backup integrity using checksum."""
        current_checksum = self._calculate_backup_checksum(backup_dir)
        return current_checksum == rollback_point.checksum
    
    def _get_current_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_git_status(self) -> Optional[str]:
        """Get git status."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _load_rollback_points(self) -> None:
        """Load rollback points from storage."""
        storage_file = self.backup_root / "rollback_points.json"
        
        if not storage_file.exists():
            return
        
        try:
            with open(storage_file, 'r') as f:
                data = json.load(f)
            
            for point_data in data.get('rollback_points', []):
                point = self._deserialize_rollback_point(point_data)
                if point:
                    self.rollback_points[point.id] = point
            
            self.logger.info(f"Loaded {len(self.rollback_points)} rollback points")
            
        except Exception as e:
            self.logger.error(f"Error loading rollback points: {e}", exc_info=True)
    
    def _save_rollback_points(self) -> None:
        """Save rollback points to storage."""
        storage_file = self.backup_root / "rollback_points.json"
        
        try:
            data = {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'rollback_points': [
                    self._serialize_rollback_point(point)
                    for point in self.rollback_points.values()
                ]
            }
            
            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving rollback points: {e}", exc_info=True)
    
    def _serialize_rollback_point(self, point: RollbackPoint) -> Dict:
        """Serialize rollback point to dictionary."""
        return {
            'id': point.id,
            'task_id': point.task_id,
            'created_at': point.created_at.isoformat(),
            'file_backups': point.file_backups,
            'git_commit': point.git_commit,
            'database_backup': point.database_backup,
            'config_backup': point.config_backup,
            'description': point.description,
            'size_bytes': point.size_bytes,
            'checksum': point.checksum,
            'valid': point.valid,
            'used': point.used,
            'cleanup_after': point.cleanup_after.isoformat() if point.cleanup_after else None
        }
    
    def _deserialize_rollback_point(self, data: Dict) -> Optional[RollbackPoint]:
        """Deserialize rollback point from dictionary."""
        try:
            return RollbackPoint(
                id=data['id'],
                task_id=data['task_id'],
                created_at=datetime.fromisoformat(data['created_at']),
                file_backups=data.get('file_backups', {}),
                git_commit=data.get('git_commit'),
                database_backup=data.get('database_backup'),
                config_backup=data.get('config_backup'),
                description=data.get('description', ''),
                size_bytes=data.get('size_bytes', 0),
                checksum=data.get('checksum', ''),
                valid=data.get('valid', True),
                used=data.get('used', False),
                cleanup_after=datetime.fromisoformat(data['cleanup_after']) if data.get('cleanup_after') else None
            )
        except Exception as e:
            self.logger.error(f"Error deserializing rollback point: {e}", exc_info=True)
            return None