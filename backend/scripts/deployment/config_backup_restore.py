#!/usr/bin/env python3
"""
Configuration Backup and Restore Tools for Enhanced Model Availability System

This script provides comprehensive backup and restore capabilities for all
configuration files and settings related to the enhanced model availability system.
"""

import os
import sys
import json
import shutil
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import zipfile
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of configuration backups"""
    FULL = "full"
    CONFIGURATION_ONLY = "configuration_only"
    USER_PREFERENCES = "user_preferences"
    SYSTEM_SETTINGS = "system_settings"
    CUSTOM = "custom"

class BackupStatus(Enum):
    """Status of backup operations"""
    CREATED = "created"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"
    RESTORED = "restored"
    FAILED = "failed"

@dataclass
class ConfigFile:
    """Represents a configuration file"""
    path: str
    relative_path: str
    size_bytes: int
    checksum: str
    last_modified: str
    backup_priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class BackupManifest:
    """Manifest of a configuration backup"""
    backup_id: str
    timestamp: str
    backup_type: BackupType
    description: str
    files: List[ConfigFile]
    total_size_bytes: int
    checksum: str
    created_by: str = "system"
    tags: List[str] = None

@dataclass
class RestoreResult:
    """Result of a restore operation"""
    success: bool
    backup_id: str
    files_restored: int
    files_failed: int
    warnings: List[str]
    errors: List[str]
    duration_seconds: float

class ConfigurationBackupManager:
    """Manages configuration backups and restores"""
    
    def __init__(self, backup_dir: str = "backups/configuration"):
        self.backup_dir = Path(backup_dir)
        self.manifests_file = self.backup_dir / "backup_manifests.json"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing manifests
        self.manifests = self._load_manifests()
        
        # Define configuration file patterns
        self.config_patterns = self._get_config_patterns()
    
    async def create_backup(self, backup_type: BackupType = BackupType.FULL, 
                          description: str = "", tags: List[str] = None,
                          custom_files: List[str] = None) -> str:
        """Create a configuration backup"""
        logger.info(f"Creating {backup_type.value} configuration backup")
        
        backup_id = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Discover configuration files
            config_files = await self._discover_config_files(backup_type, custom_files)
            
            if not config_files:
                raise ValueError(f"No configuration files found for backup type: {backup_type.value}")
            
            # Create backup archive
            archive_path = backup_path / f"{backup_id}.zip"
            total_size = await self._create_backup_archive(config_files, archive_path)
            
            # Calculate backup checksum
            backup_checksum = await self._calculate_file_checksum(archive_path)
            
            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                backup_type=backup_type,
                description=description or f"Automatic {backup_type.value} backup",
                files=config_files,
                total_size_bytes=total_size,
                checksum=backup_checksum,
                tags=tags or []
            )
            
            # Save manifest
            self.manifests[backup_id] = manifest
            await self._save_manifests()
            
            # Create backup info file
            info_file = backup_path / "backup_info.json"
            with open(info_file, 'w') as f:
                json.dump(asdict(manifest), f, indent=2, default=str)
            
            logger.info(f"Configuration backup created: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            # Cleanup partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    async def restore_backup(self, backup_id: str, 
                           target_dir: str = None,
                           dry_run: bool = False,
                           force: bool = False) -> RestoreResult:
        """Restore a configuration backup"""
        logger.info(f"Restoring configuration backup: {backup_id}")
        
        start_time = datetime.now()
        warnings = []
        errors = []
        files_restored = 0
        files_failed = 0
        
        try:
            if backup_id not in self.manifests:
                raise ValueError(f"Backup not found: {backup_id}")
            
            manifest = self.manifests[backup_id]
            backup_path = self.backup_dir / backup_id
            archive_path = backup_path / f"{backup_id}.zip"
            
            # Verify backup integrity
            if not await self._verify_backup_integrity(manifest, archive_path):
                raise ValueError(f"Backup integrity check failed: {backup_id}")
            
            # Create pre-restore backup if not dry run
            pre_restore_backup_id = None
            if not dry_run and not force:
                pre_restore_backup_id = await self.create_backup(
                    BackupType.FULL,
                    f"Pre-restore backup for {backup_id}",
                    ["pre-restore", "automatic"]
                )
                warnings.append(f"Created pre-restore backup: {pre_restore_backup_id}")
            
            # Extract and restore files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract backup archive
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    zip_file.extractall(temp_path)
                
                # Restore each file
                for config_file in manifest.files:
                    try:
                        source_file = temp_path / config_file.relative_path
                        target_file = Path(target_dir) / config_file.relative_path if target_dir else Path(config_file.path)
                        
                        if not source_file.exists():
                            errors.append(f"Source file not found in backup: {config_file.relative_path}")
                            files_failed += 1
                            continue
                        
                        if dry_run:
                            logger.info(f"[DRY RUN] Would restore: {config_file.path}")
                            files_restored += 1
                            continue
                        
                        # Create target directory if needed
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Check if target file exists and is different
                        if target_file.exists() and not force:
                            existing_checksum = await self._calculate_file_checksum(target_file)
                            if existing_checksum != config_file.checksum:
                                warnings.append(f"Target file differs from backup: {config_file.path}")
                        
                        # Copy file
                        shutil.copy2(source_file, target_file)
                        
                        # Verify restored file
                        restored_checksum = await self._calculate_file_checksum(target_file)
                        if restored_checksum != config_file.checksum:
                            errors.append(f"Checksum mismatch after restore: {config_file.path}")
                            files_failed += 1
                        else:
                            files_restored += 1
                            logger.info(f"Restored: {config_file.path}")
                        
                    except Exception as e:
                        errors.append(f"Failed to restore {config_file.path}: {str(e)}")
                        files_failed += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            success = files_failed == 0
            
            result = RestoreResult(
                success=success,
                backup_id=backup_id,
                files_restored=files_restored,
                files_failed=files_failed,
                warnings=warnings,
                errors=errors,
                duration_seconds=duration
            )
            
            logger.info(f"Restore {'completed' if success else 'completed with errors'}: {backup_id}")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            errors.append(f"Restore failed: {str(e)}")
            
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                files_restored=files_restored,
                files_failed=files_failed,
                warnings=warnings,
                errors=errors,
                duration_seconds=duration
            )
    
    async def list_backups(self, backup_type: BackupType = None, 
                          tags: List[str] = None) -> List[BackupManifest]:
        """List available backups with optional filtering"""
        backups = list(self.manifests.values())
        
        # Filter by backup type
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        # Filter by tags
        if tags:
            backups = [b for b in backups if any(tag in (b.tags or []) for tag in tags)]
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        logger.info(f"Deleting backup: {backup_id}")
        
        try:
            if backup_id not in self.manifests:
                logger.warning(f"Backup not found: {backup_id}")
                return False
            
            # Remove backup directory
            backup_path = self.backup_dir / backup_id
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            # Remove from manifests
            del self.manifests[backup_id]
            await self._save_manifests()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity"""
        logger.info(f"Verifying backup: {backup_id}")
        
        try:
            if backup_id not in self.manifests:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            manifest = self.manifests[backup_id]
            backup_path = self.backup_dir / backup_id
            archive_path = backup_path / f"{backup_id}.zip"
            
            return await self._verify_backup_integrity(manifest, archive_path)
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    async def cleanup_old_backups(self, keep_count: int = 20, 
                                keep_days: int = 30) -> int:
        """Clean up old backups"""
        logger.info(f"Cleaning up old backups (keep {keep_count} recent, {keep_days} days)")
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        backups = sorted(self.manifests.values(), key=lambda x: x.timestamp, reverse=True)
        
        # Determine backups to delete
        backups_to_delete = []
        
        # Keep recent backups
        recent_backups = backups[:keep_count]
        old_backups = backups[keep_count:]
        
        # Delete old backups beyond date cutoff
        for backup in old_backups:
            backup_date = datetime.fromisoformat(backup.timestamp)
            if backup_date < cutoff_date:
                backups_to_delete.append(backup.backup_id)
        
        # Delete backups
        deleted_count = 0
        for backup_id in backups_to_delete:
            if await self.delete_backup(backup_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count
    
    async def export_backup(self, backup_id: str, export_path: str) -> bool:
        """Export a backup to external location"""
        logger.info(f"Exporting backup {backup_id} to {export_path}")
        
        try:
            if backup_id not in self.manifests:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            backup_path = self.backup_dir / backup_id
            export_path = Path(export_path)
            
            # Create export directory
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Copy backup directory
            export_backup_path = export_path / backup_id
            shutil.copytree(backup_path, export_backup_path, dirs_exist_ok=True)
            
            logger.info(f"Backup exported to: {export_backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export backup: {e}")
            return False
    
    async def import_backup(self, import_path: str) -> Optional[str]:
        """Import a backup from external location"""
        logger.info(f"Importing backup from {import_path}")
        
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                logger.error(f"Import path does not exist: {import_path}")
                return None
            
            # Look for backup info file
            info_file = import_path / "backup_info.json"
            if not info_file.exists():
                logger.error(f"Backup info file not found: {info_file}")
                return None
            
            # Load backup manifest
            with open(info_file, 'r') as f:
                manifest_data = json.load(f)
            
            backup_id = manifest_data["backup_id"]
            
            # Check if backup already exists
            if backup_id in self.manifests:
                logger.warning(f"Backup already exists: {backup_id}")
                return backup_id
            
            # Copy backup to backup directory
            target_path = self.backup_dir / backup_id
            shutil.copytree(import_path, target_path, dirs_exist_ok=True)
            
            # Add to manifests
            manifest = BackupManifest(
                backup_id=manifest_data["backup_id"],
                timestamp=manifest_data["timestamp"],
                backup_type=BackupType(manifest_data["backup_type"]),
                description=manifest_data["description"],
                files=[ConfigFile(**f) for f in manifest_data["files"]],
                total_size_bytes=manifest_data["total_size_bytes"],
                checksum=manifest_data["checksum"],
                created_by=manifest_data.get("created_by", "imported"),
                tags=manifest_data.get("tags", [])
            )
            
            self.manifests[backup_id] = manifest
            await self._save_manifests()
            
            logger.info(f"Backup imported: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to import backup: {e}")
            return None
    
    def _get_config_patterns(self) -> Dict[BackupType, List[str]]:
        """Get configuration file patterns for different backup types"""
        return {
            BackupType.FULL: [
                "config.json",
                "backend/config.json",
                "startup_config.json",
                "recovery_config.json",
                "monitoring_config.json",
                "backend/core/enhanced_model_config.py",
                "backend/api/enhanced_model_management.py",
                "backend/services/generation_service.py",
                "backend/websocket/model_notifications.py",
                ".env",
                "backend/.env",
                "frontend/.env"
            ],
            BackupType.CONFIGURATION_ONLY: [
                "config.json",
                "backend/config.json",
                "startup_config.json",
                "recovery_config.json",
                "monitoring_config.json"
            ],
            BackupType.USER_PREFERENCES: [
                "user_preferences.json",
                "ui_settings.json",
                ".env",
                "backend/.env",
                "frontend/.env"
            ],
            BackupType.SYSTEM_SETTINGS: [
                "config.json",
                "backend/config.json",
                "startup_config.json",
                "recovery_config.json"
            ]
        }
    
    async def _discover_config_files(self, backup_type: BackupType, 
                                   custom_files: List[str] = None) -> List[ConfigFile]:
        """Discover configuration files for backup"""
        config_files = []
        
        if backup_type == BackupType.CUSTOM and custom_files:
            file_patterns = custom_files
        else:
            file_patterns = self.config_patterns.get(backup_type, [])
        
        for pattern in file_patterns:
            # Handle glob patterns
            if "*" in pattern:
                from glob import glob
                matching_files = glob(pattern, recursive=True)
            else:
                matching_files = [pattern] if Path(pattern).exists() else []
            
            for file_path in matching_files:
                file_path = Path(file_path)
                if file_path.exists() and file_path.is_file():
                    try:
                        stat = file_path.stat()
                        checksum = await self._calculate_file_checksum(file_path)
                        
                        config_file = ConfigFile(
                            path=str(file_path),
                            relative_path=str(file_path),
                            size_bytes=stat.st_size,
                            checksum=checksum,
                            last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            backup_priority=1 if "config" in str(file_path).lower() else 2
                        )
                        
                        config_files.append(config_file)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process config file {file_path}: {e}")
        
        return config_files
    
    async def _create_backup_archive(self, config_files: List[ConfigFile], 
                                   archive_path: Path) -> int:
        """Create backup archive from configuration files"""
        total_size = 0
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for config_file in config_files:
                source_path = Path(config_file.path)
                if source_path.exists():
                    zip_file.write(source_path, config_file.relative_path)
                    total_size += config_file.size_bytes
        
        return total_size
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _verify_backup_integrity(self, manifest: BackupManifest, 
                                     archive_path: Path) -> bool:
        """Verify backup integrity"""
        try:
            # Check if archive exists
            if not archive_path.exists():
                logger.error(f"Backup archive not found: {archive_path}")
                return False
            
            # Verify archive checksum
            archive_checksum = await self._calculate_file_checksum(archive_path)
            if archive_checksum != manifest.checksum:
                logger.error(f"Backup checksum mismatch: expected {manifest.checksum}, got {archive_checksum}")
                return False
            
            # Verify archive can be opened
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    # Test archive integrity
                    bad_files = zip_file.testzip()
                    if bad_files:
                        logger.error(f"Corrupted files in backup: {bad_files}")
                        return False
            except zipfile.BadZipFile:
                logger.error(f"Backup archive is corrupted: {archive_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Backup integrity verification failed: {e}")
            return False
    
    def _load_manifests(self) -> Dict[str, BackupManifest]:
        """Load backup manifests from storage"""
        if self.manifests_file.exists():
            try:
                with open(self.manifests_file, 'r') as f:
                    data = json.load(f)
                
                manifests = {}
                for backup_id, manifest_data in data.items():
                    manifests[backup_id] = BackupManifest(
                        backup_id=manifest_data["backup_id"],
                        timestamp=manifest_data["timestamp"],
                        backup_type=BackupType(manifest_data["backup_type"]),
                        description=manifest_data["description"],
                        files=[ConfigFile(**f) for f in manifest_data["files"]],
                        total_size_bytes=manifest_data["total_size_bytes"],
                        checksum=manifest_data["checksum"],
                        created_by=manifest_data.get("created_by", "system"),
                        tags=manifest_data.get("tags", [])
                    )
                
                return manifests
                
            except Exception as e:
                logger.error(f"Failed to load backup manifests: {e}")
        
        return {}
    
    async def _save_manifests(self):
        """Save backup manifests to storage"""
        try:
            data = {}
            for backup_id, manifest in self.manifests.items():
                data[backup_id] = asdict(manifest)
            
            with open(self.manifests_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save backup manifests: {e}")

async def main():
    """Main configuration backup/restore function"""
    import argparse
    from datetime import timedelta
    
    parser = argparse.ArgumentParser(description="Configuration Backup and Restore Tool")
    parser.add_argument("action", choices=["backup", "restore", "list", "verify", "delete", "cleanup", "export", "import"],
                       help="Action to perform")
    parser.add_argument("--type", choices=["full", "configuration_only", "user_preferences", "system_settings", "custom"],
                       default="full", help="Backup type")
    parser.add_argument("--backup-id", help="Backup ID for restore/verify/delete operations")
    parser.add_argument("--description", help="Backup description")
    parser.add_argument("--tags", nargs="*", help="Backup tags")
    parser.add_argument("--custom-files", nargs="*", help="Custom files for custom backup type")
    parser.add_argument("--target-dir", help="Target directory for restore")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run (restore only)")
    parser.add_argument("--force", action="store_true", help="Force restore without pre-backup")
    parser.add_argument("--keep-count", type=int, default=20, help="Number of backups to keep during cleanup")
    parser.add_argument("--keep-days", type=int, default=30, help="Number of days to keep backups during cleanup")
    parser.add_argument("--export-path", help="Path for export operation")
    parser.add_argument("--import-path", help="Path for import operation")
    
    args = parser.parse_args()
    
    backup_manager = ConfigurationBackupManager()
    
    if args.action == "backup":
        backup_type = BackupType(args.type)
        backup_id = await backup_manager.create_backup(
            backup_type=backup_type,
            description=args.description or "",
            tags=args.tags,
            custom_files=args.custom_files
        )
        print(f"Backup created: {backup_id}")
        
    elif args.action == "restore":
        if not args.backup_id:
            print("Backup ID required for restore operation")
            return 1
        
        result = await backup_manager.restore_backup(
            backup_id=args.backup_id,
            target_dir=args.target_dir,
            dry_run=args.dry_run,
            force=args.force
        )
        
        print(f"Restore {'succeeded' if result.success else 'failed'}")
        print(f"Files restored: {result.files_restored}")
        print(f"Files failed: {result.files_failed}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        return 0 if result.success else 1
        
    elif args.action == "list":
        backup_type = BackupType(args.type) if args.type != "full" else None
        backups = await backup_manager.list_backups(backup_type=backup_type, tags=args.tags)
        
        print(f"Available backups ({len(backups)}):")
        for backup in backups:
            size_mb = backup.total_size_bytes / (1024 * 1024)
            print(f"  {backup.backup_id}: {backup.description}")
            print(f"    Type: {backup.backup_type.value}, Size: {size_mb:.1f} MB")
            print(f"    Created: {backup.timestamp}, Files: {len(backup.files)}")
            if backup.tags:
                print(f"    Tags: {', '.join(backup.tags)}")
        
    elif args.action == "verify":
        if not args.backup_id:
            print("Backup ID required for verify operation")
            return 1
        
        is_valid = await backup_manager.verify_backup(args.backup_id)
        print(f"Backup {args.backup_id} is {'valid' if is_valid else 'invalid'}")
        return 0 if is_valid else 1
        
    elif args.action == "delete":
        if not args.backup_id:
            print("Backup ID required for delete operation")
            return 1
        
        success = await backup_manager.delete_backup(args.backup_id)
        print(f"Backup {'deleted' if success else 'deletion failed'}")
        return 0 if success else 1
        
    elif args.action == "cleanup":
        removed = await backup_manager.cleanup_old_backups(args.keep_count, args.keep_days)
        print(f"Cleaned up {removed} old backups")
        
    elif args.action == "export":
        if not args.backup_id or not args.export_path:
            print("Backup ID and export path required for export operation")
            return 1
        
        success = await backup_manager.export_backup(args.backup_id, args.export_path)
        print(f"Backup {'exported' if success else 'export failed'}")
        return 0 if success else 1
        
    elif args.action == "import":
        if not args.import_path:
            print("Import path required for import operation")
            return 1
        
        backup_id = await backup_manager.import_backup(args.import_path)
        if backup_id:
            print(f"Backup imported: {backup_id}")
            return 0
        else:
            print("Backup import failed")
            return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)