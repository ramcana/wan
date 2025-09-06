"""
Rollback Manager
Provides rollback capabilities with snapshot creation and restoration.
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from interfaces import InstallationError, ErrorCategory


@dataclass
class RollbackSnapshot:
    """Represents a rollback snapshot."""
    id: str
    timestamp: str
    description: str
    phase: str
    files: Dict[str, str]  # original_path -> backup_path
    directories: Dict[str, str]  # original_path -> backup_path
    config_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class RollbackManager:
    """
    Manages rollback snapshots and restoration capabilities.
    """
    
    def __init__(self, installation_path: str, dry_run: bool = False):
        self.installation_path = Path(installation_path)
        self.dry_run = dry_run
        self.backup_root = self.installation_path / ".wan22_backup"
        self.snapshots_dir = self.backup_root / "snapshots"
        self.index_file = self.backup_root / "rollback_index.json"
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize backup directories
        self._initialize_backup_structure()
    
    def _initialize_backup_structure(self) -> None:
        """Initialize the backup directory structure."""
        if not self.dry_run:
            self.backup_root.mkdir(exist_ok=True)
            self.snapshots_dir.mkdir(exist_ok=True)
        
        self.logger.debug(f"Backup structure initialized at {self.backup_root}")
    
    def create_snapshot(self, description: str, phase: str = "unknown", 
                       files_to_backup: Optional[List[str]] = None,
                       dirs_to_backup: Optional[List[str]] = None) -> str:
        """
        Create a rollback snapshot.
        
        Args:
            description: Description of the snapshot
            phase: Installation phase when snapshot was created
            files_to_backup: List of specific files to backup
            dirs_to_backup: List of specific directories to backup
            
        Returns:
            Snapshot ID
        """
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create snapshot: {snapshot_id}")
            return snapshot_id
        
        try:
            snapshot_dir = self.snapshots_dir / snapshot_id
            snapshot_dir.mkdir(exist_ok=True)
            
            # Backup files
            files_backup = {}
            if files_to_backup:
                files_backup = self._backup_files(files_to_backup, snapshot_dir / "files")
            
            # Backup directories
            dirs_backup = {}
            if dirs_to_backup:
                dirs_backup = self._backup_directories(dirs_to_backup, snapshot_dir / "directories")
            
            # Backup current configuration
            config_data = self._backup_configuration()
            
            # Create snapshot metadata
            snapshot = RollbackSnapshot(
                id=snapshot_id,
                timestamp=datetime.now().isoformat(),
                description=description,
                phase=phase,
                files=files_backup,
                directories=dirs_backup,
                config_data=config_data,
                metadata={
                    "installation_path": str(self.installation_path),
                    "created_by": "RollbackManager",
                    "version": "1.0"
                }
            )
            
            # Save snapshot metadata
            snapshot_file = snapshot_dir / "snapshot.json"
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False)
            
            # Update index
            self._update_index(snapshot)
            
            self.logger.info(f"Created snapshot: {snapshot_id} - {description}")
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            raise InstallationError(
                f"Failed to create rollback snapshot: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check disk space", "Verify write permissions", "Close other applications"]
            )
    
    def _backup_files(self, files: List[str], backup_dir: Path) -> Dict[str, str]:
        """Backup specified files."""
        backup_dir.mkdir(parents=True, exist_ok=True)
        files_backup = {}
        
        for file_path in files:
            source_path = Path(file_path)
            if not source_path.is_absolute():
                source_path = self.installation_path / source_path
            
            if source_path.exists() and source_path.is_file():
                backup_path = backup_dir / source_path.name
                
                # Handle name conflicts
                counter = 1
                while backup_path.exists():
                    name_parts = source_path.name.rsplit('.', 1)
                    if len(name_parts) == 2:
                        backup_path = backup_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        backup_path = backup_dir / f"{source_path.name}_{counter}"
                    counter += 1
                
                shutil.copy2(source_path, backup_path)
                files_backup[str(source_path)] = str(backup_path)
                self.logger.debug(f"Backed up file: {source_path} -> {backup_path}")
        
        return files_backup
    
    def _backup_directories(self, directories: List[str], backup_dir: Path) -> Dict[str, str]:
        """Backup specified directories."""
        backup_dir.mkdir(parents=True, exist_ok=True)
        dirs_backup = {}
        
        for dir_path in directories:
            source_path = Path(dir_path)
            if not source_path.is_absolute():
                source_path = self.installation_path / source_path
            
            if source_path.exists() and source_path.is_dir():
                backup_path = backup_dir / source_path.name
                
                # Handle name conflicts
                counter = 1
                while backup_path.exists():
                    backup_path = backup_dir / f"{source_path.name}_{counter}"
                    counter += 1
                
                shutil.copytree(source_path, backup_path)
                dirs_backup[str(source_path)] = str(backup_path)
                self.logger.debug(f"Backed up directory: {source_path} -> {backup_path}")
        
        return dirs_backup
    
    def _backup_configuration(self) -> Optional[Dict[str, Any]]:
        """Backup current configuration."""
        config_file = self.installation_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to backup configuration: {e}")
        return None
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to restore
            
        Returns:
            True if restoration was successful
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would restore snapshot: {snapshot_id}")
            return True
        
        try:
            snapshot = self._load_snapshot(snapshot_id)
            if not snapshot:
                raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
            
            self.logger.info(f"Starting restoration of snapshot: {snapshot_id}")
            
            # Create a backup of current state before restoration
            pre_restore_id = self.create_snapshot(
                f"Pre-restore backup before {snapshot_id}",
                "pre-restore"
            )
            
            # Restore files
            for original_path, backup_path in snapshot.files.items():
                if Path(backup_path).exists():
                    # Ensure target directory exists
                    Path(original_path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, original_path)
                    self.logger.debug(f"Restored file: {backup_path} -> {original_path}")
            
            # Restore directories
            for original_path, backup_path in snapshot.directories.items():
                if Path(backup_path).exists():
                    # Remove existing directory if it exists
                    if Path(original_path).exists():
                        shutil.rmtree(original_path)
                    
                    # Ensure parent directory exists
                    Path(original_path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(backup_path, original_path)
                    self.logger.debug(f"Restored directory: {backup_path} -> {original_path}")
            
            # Restore configuration
            if snapshot.config_data:
                config_file = self.installation_path / "config.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(snapshot.config_data, f, indent=2, ensure_ascii=False)
                self.logger.debug("Restored configuration")
            
            self.logger.info(f"Successfully restored snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
            return False
    
    def _load_snapshot(self, snapshot_id: str) -> Optional[RollbackSnapshot]:
        """Load snapshot metadata."""
        snapshot_file = self.snapshots_dir / snapshot_id / "snapshot.json"
        
        if not snapshot_file.exists():
            return None
        
        try:
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)
            
            return RollbackSnapshot(**snapshot_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def list_snapshots(self) -> List[RollbackSnapshot]:
        """List all available snapshots."""
        snapshots = []
        
        if not self.snapshots_dir.exists():
            return snapshots
        
        for snapshot_dir in self.snapshots_dir.iterdir():
            if snapshot_dir.is_dir():
                snapshot = self._load_snapshot(snapshot_dir.name)
                if snapshot:
                    snapshots.append(snapshot)
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x.timestamp, reverse=True)
        return snapshots
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would delete snapshot: {snapshot_id}")
            return True
        
        try:
            snapshot_dir = self.snapshots_dir / snapshot_id
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
                self._remove_from_index(snapshot_id)
                self.logger.info(f"Deleted snapshot: {snapshot_id}")
                return True
            else:
                self.logger.warning(f"Snapshot not found: {snapshot_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
            return False
    
    def cleanup_old_snapshots(self, keep_count: int = 5, 
                            keep_days: int = 30) -> int:
        """
        Clean up old snapshots based on count and age.
        
        Args:
            keep_count: Number of recent snapshots to keep
            keep_days: Number of days to keep snapshots
            
        Returns:
            Number of snapshots deleted
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would cleanup snapshots (keep {keep_count}, {keep_days} days)")
            return 0
        
        snapshots = self.list_snapshots()
        deleted_count = 0
        
        # Keep recent snapshots
        snapshots_to_keep = snapshots[:keep_count]
        snapshots_to_check = snapshots[keep_count:]
        
        # Check age for remaining snapshots
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        
        for snapshot in snapshots_to_check:
            snapshot_date = datetime.fromisoformat(snapshot.timestamp).timestamp()
            
            if snapshot_date < cutoff_date:
                if self.delete_snapshot(snapshot.id):
                    deleted_count += 1
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old snapshots")
        
        return deleted_count
    
    def _update_index(self, snapshot: RollbackSnapshot) -> None:
        """Update the snapshots index."""
        try:
            index = {}
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            
            index[snapshot.id] = {
                'id': snapshot.id,
                'timestamp': snapshot.timestamp,
                'description': snapshot.description,
                'phase': snapshot.phase
            }
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to update index: {e}")
    
    def _remove_from_index(self, snapshot_id: str) -> None:
        """Remove snapshot from index."""
        try:
            if not self.index_file.exists():
                return
            
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            if snapshot_id in index:
                del index[snapshot_id]
                
                with open(self.index_file, 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            self.logger.error(f"Failed to remove from index: {e}")
    
    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a snapshot."""
        snapshot = self._load_snapshot(snapshot_id)
        if not snapshot:
            return None
        
        snapshot_dir = self.snapshots_dir / snapshot_id
        
        # Calculate snapshot size
        total_size = 0
        if snapshot_dir.exists():
            for file_path in snapshot_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return {
            'id': snapshot.id,
            'timestamp': snapshot.timestamp,
            'description': snapshot.description,
            'phase': snapshot.phase,
            'files_count': len(snapshot.files),
            'directories_count': len(snapshot.directories),
            'has_config': snapshot.config_data is not None,
            'size_bytes': total_size,
            'size_mb': round(total_size / (1024 * 1024), 2),
            'metadata': snapshot.metadata
        }
    
    def validate_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Validate a snapshot's integrity."""
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        snapshot = self._load_snapshot(snapshot_id)
        if not snapshot:
            result['valid'] = False
            result['issues'].append("Snapshot metadata not found")
            return result
        
        snapshot_dir = self.snapshots_dir / snapshot_id
        if not snapshot_dir.exists():
            result['valid'] = False
            result['issues'].append("Snapshot directory not found")
            return result
        
        # Check backed up files
        for original_path, backup_path in snapshot.files.items():
            if not Path(backup_path).exists():
                result['valid'] = False
                result['issues'].append(f"Backup file missing: {backup_path}")
        
        # Check backed up directories
        for original_path, backup_path in snapshot.directories.items():
            if not Path(backup_path).exists():
                result['valid'] = False
                result['issues'].append(f"Backup directory missing: {backup_path}")
        
        return result
    
    def create_recovery_point(self, description: str = "Recovery point") -> str:
        """
        Create a comprehensive recovery point for the entire installation.
        
        Args:
            description: Description of the recovery point
            
        Returns:
            Snapshot ID of the recovery point
        """
        # Define critical files and directories for recovery
        critical_files = [
            "config.json",
            "version.json",
            "main.py",
            "ui.py",
            "utils.py",
            "error_handler.py"
        ]
        
        critical_dirs = [
            "scripts",
            "resources",
            "models",
            "logs"
        ]
        
        # Filter existing files/directories
        existing_files = [f for f in critical_files 
                         if (self.installation_path / f).exists()]
        existing_dirs = [d for d in critical_dirs 
                        if (self.installation_path / d).exists()]
        
        return self.create_snapshot(
            description=f"Recovery point: {description}",
            phase="recovery",
            files_to_backup=existing_files,
            dirs_to_backup=existing_dirs
        )
    
    def recover_from_failed_installation(self, failure_phase: str = "unknown") -> bool:
        """
        Attempt to recover from a failed installation by restoring the most recent valid snapshot.
        
        Args:
            failure_phase: The phase where the installation failed
            
        Returns:
            True if recovery was successful
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would recover from failed installation in phase: {failure_phase}")
            return True
        
        try:
            self.logger.info(f"Starting recovery from failed installation (phase: {failure_phase})")
            
            # Find the most recent valid snapshot
            snapshots = self.list_snapshots()
            
            if not snapshots:
                self.logger.error("No snapshots available for recovery")
                return False
            
            # Look for pre-installation or pre-update snapshots first
            recovery_candidates = []
            
            # First priority: pre-update snapshots
            for snapshot in snapshots:
                if 'pre-update' in snapshot.description.lower():
                    recovery_candidates.append(snapshot)
            
            # Second priority: other pre- snapshots (pre-install, etc.)
            if not recovery_candidates:
                for snapshot in snapshots:
                    if snapshot.description.lower().startswith('pre-') and 'emergency' not in snapshot.description.lower():
                        recovery_candidates.append(snapshot)
            
            # Third priority: recovery points
            if not recovery_candidates:
                for snapshot in snapshots:
                    if 'recovery point' in snapshot.description.lower():
                        recovery_candidates.append(snapshot)
            
            # If no specific recovery snapshots, use the most recent one
            if not recovery_candidates:
                recovery_candidates = snapshots[:1]  # Most recent
            
            # Try to restore from each candidate until one succeeds
            for snapshot in recovery_candidates:
                self.logger.info(f"Attempting recovery from snapshot: {snapshot.id}")
                
                # Validate snapshot before attempting restore
                validation = self.validate_snapshot(snapshot.id)
                if not validation['valid']:
                    self.logger.warning(f"Snapshot {snapshot.id} is invalid: {validation['issues']}")
                    continue
                
                # Attempt restoration
                if self.restore_snapshot(snapshot.id):
                    self.logger.info(f"Successfully recovered from snapshot: {snapshot.id}")
                    
                    # Create a post-recovery snapshot
                    self.create_snapshot(
                        f"Post-recovery state (recovered from {snapshot.id})",
                        "post-recovery"
                    )
                    
                    return True
                else:
                    self.logger.warning(f"Failed to restore from snapshot: {snapshot.id}")
            
            self.logger.error("All recovery attempts failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Recovery procedure failed: {e}")
            return False
    
    def create_emergency_backup(self) -> Optional[str]:
        """
        Create an emergency backup of the current state before attempting recovery.
        
        Returns:
            Snapshot ID if successful, None if failed
        """
        try:
            return self.create_snapshot(
                f"Emergency backup before recovery - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "emergency"
            )
        except Exception as e:
            self.logger.error(f"Failed to create emergency backup: {e}")
            return None
    
    def get_recovery_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recovery recommendations based on available snapshots and system state.
        
        Returns:
            List of recovery recommendations
        """
        recommendations = []
        
        try:
            snapshots = self.list_snapshots()
            
            if not snapshots:
                recommendations.append({
                    'type': 'no_snapshots',
                    'priority': 'high',
                    'message': 'No snapshots available for recovery',
                    'action': 'Consider reinstalling from scratch'
                })
                return recommendations
            
            # Check for recent pre-installation snapshots
            pre_install_snapshots = [s for s in snapshots 
                                   if 'pre-' in s.description.lower()]
            
            if pre_install_snapshots:
                recommendations.append({
                    'type': 'pre_install_available',
                    'priority': 'high',
                    'message': f'Found {len(pre_install_snapshots)} pre-installation snapshot(s)',
                    'action': f'Restore from: {pre_install_snapshots[0].description}',
                    'snapshot_id': pre_install_snapshots[0].id
                })
            
            # Check for old snapshots that might need cleanup
            if len(snapshots) > 10:
                recommendations.append({
                    'type': 'cleanup_needed',
                    'priority': 'low',
                    'message': f'Found {len(snapshots)} snapshots, cleanup recommended',
                    'action': 'Run cleanup_old_snapshots() to free disk space'
                })
            
            # Check snapshot validity
            invalid_snapshots = []
            for snapshot in snapshots[:5]:  # Check recent snapshots
                validation = self.validate_snapshot(snapshot.id)
                if not validation['valid']:
                    invalid_snapshots.append(snapshot.id)
            
            if invalid_snapshots:
                recommendations.append({
                    'type': 'invalid_snapshots',
                    'priority': 'medium',
                    'message': f'Found {len(invalid_snapshots)} invalid snapshot(s)',
                    'action': 'Consider deleting invalid snapshots',
                    'snapshot_ids': invalid_snapshots
                })
            
            # Check disk space
            try:
                backup_size = sum(f.stat().st_size for f in self.backup_root.rglob('*') if f.is_file())
                backup_size_mb = backup_size / (1024 * 1024)
                
                if backup_size_mb > 1000:  # More than 1GB
                    recommendations.append({
                        'type': 'large_backup_size',
                        'priority': 'medium',
                        'message': f'Backup directory is {backup_size_mb:.1f}MB',
                        'action': 'Consider cleaning up old snapshots to save space'
                    })
            except Exception:
                pass  # Ignore disk space check errors
            
        except Exception as e:
            self.logger.error(f"Failed to generate recovery recommendations: {e}")
            recommendations.append({
                'type': 'error',
                'priority': 'high',
                'message': f'Error generating recommendations: {str(e)}',
                'action': 'Check system logs for more details'
            })
        
        return recommendations
    
    def cleanup_failed_installation_artifacts(self) -> bool:
        """
        Clean up artifacts from a failed installation.
        
        Returns:
            True if cleanup was successful
        """
        if self.dry_run:
            self.logger.info("[DRY RUN] Would clean up failed installation artifacts")
            return True
        
        try:
            self.logger.info("Cleaning up failed installation artifacts")
            
            # List of temporary files/directories that might be left from failed installation
            temp_artifacts = [
                "temp_download",
                "temp_extract",
                "installation.tmp",
                "models_temp",
                "*.tmp",
                "*.partial"
            ]
            
            cleaned_count = 0
            
            for artifact_pattern in temp_artifacts:
                if '*' in artifact_pattern:
                    # Handle glob patterns
                    for artifact_path in self.installation_path.glob(artifact_pattern):
                        try:
                            if artifact_path.is_file():
                                artifact_path.unlink()
                            elif artifact_path.is_dir():
                                shutil.rmtree(artifact_path)
                            cleaned_count += 1
                            self.logger.debug(f"Cleaned up: {artifact_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to clean up {artifact_path}: {e}")
                else:
                    # Handle specific files/directories
                    artifact_path = self.installation_path / artifact_pattern
                    if artifact_path.exists():
                        try:
                            if artifact_path.is_file():
                                artifact_path.unlink()
                            elif artifact_path.is_dir():
                                shutil.rmtree(artifact_path)
                            cleaned_count += 1
                            self.logger.debug(f"Cleaned up: {artifact_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to clean up {artifact_path}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} failed installation artifacts")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clean up installation artifacts: {e}")
            return False