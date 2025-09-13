"""
WAN Model Rollback Service

Provides comprehensive rollback capabilities for WAN model deployments,
including backup creation, restoration, and rollback verification.
"""

import asyncio
import json
import logging
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiofiles


@dataclass
class BackupInfo:
    """Information about a model backup"""
    backup_id: str
    deployment_id: str
    model_names: List[str]
    backup_path: str
    created_at: datetime
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]


@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    deployment_id: str
    backup_id: str
    models_restored: List[str]
    rollback_time: float
    error: Optional[str] = None


class RollbackService:
    """
    Service for managing model deployment rollbacks
    
    Provides:
    - Backup creation before deployments
    - Rollback to previous model versions
    - Backup management and cleanup
    - Rollback verification and validation
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backup_registry: Dict[str, BackupInfo] = {}
        self.rollback_history: List[RollbackResult] = []
        
        # Ensure backup directory exists
        Path(self.config.backup_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing backup registry
        asyncio.create_task(self._load_backup_registry())
    
    async def _load_backup_registry(self):
        """Load backup registry from disk"""
        registry_file = Path(self.config.backup_path) / "backup_registry.json"
        
        if registry_file.exists():
            try:
                async with aiofiles.open(registry_file, 'r') as f:
                    registry_data = json.loads(await f.read())
                    
                for backup_id, backup_data in registry_data.items():
                    # Convert datetime strings back to datetime objects
                    backup_data['created_at'] = datetime.fromisoformat(backup_data['created_at'])
                    self.backup_registry[backup_id] = BackupInfo(**backup_data)
                    
                self.logger.info(f"Loaded {len(self.backup_registry)} backups from registry")
                
            except Exception as e:
                self.logger.error(f"Failed to load backup registry: {str(e)}")
    
    async def _save_backup_registry(self):
        """Save backup registry to disk"""
        registry_file = Path(self.config.backup_path) / "backup_registry.json"
        
        try:
            registry_data = {}
            for backup_id, backup_info in self.backup_registry.items():
                backup_data = asdict(backup_info)
                backup_data['created_at'] = backup_info.created_at.isoformat()
                registry_data[backup_id] = backup_data
            
            async with aiofiles.open(registry_file, 'w') as f:
                await f.write(json.dumps(registry_data, indent=2))
                
        except Exception as e:
            self.logger.error(f"Failed to save backup registry: {str(e)}")
    
    async def create_backup(self, deployment_id: str, models: Optional[List[str]] = None) -> RollbackResult:
        """
        Create a backup of current models before deployment
        
        Args:
            deployment_id: ID of the deployment this backup is for
            models: List of model names to backup (if None, backup all)
            
        Returns:
            RollbackResult indicating backup success/failure
        """
        start_time = datetime.now()
        backup_id = f"backup_{deployment_id}_{int(start_time.timestamp())}"
        
        self.logger.info(f"Creating backup {backup_id} for deployment {deployment_id}")
        
        try:
            # Determine which models to backup
            if models is None:
                models = await self._discover_existing_models()
            
            if not models:
                self.logger.warning("No models found to backup")
                return RollbackResult(
                    success=True,
                    deployment_id=deployment_id,
                    backup_id=backup_id,
                    models_restored=[],
                    rollback_time=0.0
                )
            
            # Create backup directory
            backup_dir = Path(self.config.backup_path) / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup each model
            backed_up_models = []
            total_size = 0
            
            for model_name in models:
                model_backup_result = await self._backup_single_model(
                    model_name, backup_dir, deployment_id
                )
                
                if model_backup_result['success']:
                    backed_up_models.append(model_name)
                    total_size += model_backup_result['size']
                else:
                    self.logger.warning(f"Failed to backup model {model_name}: {model_backup_result['error']}")
            
            # Create backup metadata
            metadata = {
                "deployment_id": deployment_id,
                "models": backed_up_models,
                "created_at": start_time.isoformat(),
                "backup_type": "pre_deployment",
                "system_info": await self._get_system_info()
            }
            
            metadata_file = backup_dir / "backup_metadata.json"
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            # Calculate checksum for backup integrity
            checksum = await self._calculate_backup_checksum(backup_dir)
            
            # Register backup
            backup_info = BackupInfo(
                backup_id=backup_id,
                deployment_id=deployment_id,
                model_names=backed_up_models,
                backup_path=str(backup_dir),
                created_at=start_time,
                size_bytes=total_size,
                checksum=checksum,
                metadata=metadata
            )
            
            self.backup_registry[backup_id] = backup_info
            await self._save_backup_registry()
            
            rollback_time = (datetime.now() - start_time).total_seconds()
            
            result = RollbackResult(
                success=True,
                deployment_id=deployment_id,
                backup_id=backup_id,
                models_restored=backed_up_models,
                rollback_time=rollback_time
            )
            
            self.logger.info(f"Successfully created backup {backup_id} in {rollback_time:.2f}s")
            return result
            
        except Exception as e:
            rollback_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Backup creation failed: {str(e)}"
            self.logger.error(error_msg)
            
            return RollbackResult(
                success=False,
                deployment_id=deployment_id,
                backup_id=backup_id,
                models_restored=[],
                rollback_time=rollback_time,
                error=error_msg
            )
    
    async def rollback_deployment(self, deployment_id: str, reason: str = "Manual rollback") -> RollbackResult:
        """
        Rollback a deployment to its previous state
        
        Args:
            deployment_id: ID of the deployment to rollback
            reason: Reason for the rollback
            
        Returns:
            RollbackResult indicating rollback success/failure
        """
        start_time = datetime.now()
        self.logger.info(f"Starting rollback for deployment {deployment_id}: {reason}")
        
        try:
            # Find the backup for this deployment
            backup_info = None
            for backup_id, info in self.backup_registry.items():
                if info.deployment_id == deployment_id:
                    backup_info = info
                    break
            
            if not backup_info:
                raise ValueError(f"No backup found for deployment {deployment_id}")
            
            # Verify backup integrity
            if not await self._verify_backup_integrity(backup_info):
                raise Exception(f"Backup integrity check failed for {backup_info.backup_id}")
            
            # Perform rollback
            restored_models = []
            
            for model_name in backup_info.model_names:
                restore_result = await self._restore_single_model(
                    model_name, backup_info.backup_path
                )
                
                if restore_result['success']:
                    restored_models.append(model_name)
                else:
                    self.logger.error(f"Failed to restore model {model_name}: {restore_result['error']}")
            
            # Verify rollback
            await self._verify_rollback(restored_models)
            
            rollback_time = (datetime.now() - start_time).total_seconds()
            
            result = RollbackResult(
                success=len(restored_models) > 0,
                deployment_id=deployment_id,
                backup_id=backup_info.backup_id,
                models_restored=restored_models,
                rollback_time=rollback_time
            )
            
            self.rollback_history.append(result)
            
            self.logger.info(f"Rollback completed for deployment {deployment_id} in {rollback_time:.2f}s")
            return result
            
        except Exception as e:
            rollback_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Rollback failed for deployment {deployment_id}: {str(e)}"
            self.logger.error(error_msg)
            
            result = RollbackResult(
                success=False,
                deployment_id=deployment_id,
                backup_id="",
                models_restored=[],
                rollback_time=rollback_time,
                error=error_msg
            )
            
            self.rollback_history.append(result)
            return result
    
    async def _discover_existing_models(self) -> List[str]:
        """Discover existing models in the target directory"""
        models_dir = Path(self.config.target_models_path)
        
        if not models_dir.exists():
            return []
        
        models = []
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                models.append(item.name)
        
        return models
    
    async def _backup_single_model(self, model_name: str, backup_dir: Path, deployment_id: str) -> Dict[str, Any]:
        """Backup a single model"""
        try:
            source_path = Path(self.config.target_models_path) / model_name
            
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f"Model {model_name} not found at {source_path}",
                    'size': 0
                }
            
            # Create compressed backup
            backup_file = backup_dir / f"{model_name}.tar.gz"
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(source_path, arcname=model_name)
            
            size = backup_file.stat().st_size
            
            self.logger.info(f"Backed up model {model_name} ({size / (1024**2):.1f}MB)")
            
            return {
                'success': True,
                'error': None,
                'size': size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'size': 0
            }
    
    async def _restore_single_model(self, model_name: str, backup_path: str) -> Dict[str, Any]:
        """Restore a single model from backup"""
        try:
            backup_file = Path(backup_path) / f"{model_name}.tar.gz"
            
            if not backup_file.exists():
                return {
                    'success': False,
                    'error': f"Backup file not found: {backup_file}"
                }
            
            target_path = Path(self.config.target_models_path) / model_name
            
            # Remove existing model if present
            if target_path.exists():
                shutil.rmtree(target_path)
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(path=self.config.target_models_path)
            
            self.logger.info(f"Restored model {model_name} from backup")
            
            return {
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for backup integrity verification"""
        import hashlib
        
        hasher = hashlib.sha256()
        
        # Sort files for consistent checksum
        files = sorted(backup_dir.rglob('*'))
        
        for file_path in files:
            if file_path.is_file():
                hasher.update(str(file_path.relative_to(backup_dir)).encode())
                
                async with aiofiles.open(file_path, 'rb') as f:
                    while chunk := await f.read(8192):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def _verify_backup_integrity(self, backup_info: BackupInfo) -> bool:
        """Verify backup integrity using checksum"""
        try:
            backup_dir = Path(backup_info.backup_path)
            
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            current_checksum = await self._calculate_backup_checksum(backup_dir)
            
            if current_checksum != backup_info.checksum:
                self.logger.error(f"Backup integrity check failed for {backup_info.backup_id}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup integrity verification failed: {str(e)}")
            return False
    
    async def _verify_rollback(self, restored_models: List[str]):
        """Verify that rollback was successful"""
        for model_name in restored_models:
            model_path = Path(self.config.target_models_path) / model_name
            
            if not model_path.exists():
                raise Exception(f"Rollback verification failed: {model_name} not found after restore")
            
            # Additional verification could include loading the model, checking configs, etc.
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for backup metadata"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage": {
                "total_gb": psutil.disk_usage('.').total / (1024**3),
                "free_gb": psutil.disk_usage('.').free / (1024**3)
            }
        }
    
    async def list_backups(self, deployment_id: Optional[str] = None) -> List[BackupInfo]:
        """List available backups, optionally filtered by deployment ID"""
        backups = list(self.backup_registry.values())
        
        if deployment_id:
            backups = [b for b in backups if b.deployment_id == deployment_id]
        
        # Sort by creation time (most recent first)
        backups.sort(key=lambda x: x.created_at, reverse=True)
        
        return backups
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a specific backup"""
        if backup_id not in self.backup_registry:
            self.logger.error(f"Backup {backup_id} not found")
            return False
        
        try:
            backup_info = self.backup_registry[backup_id]
            backup_path = Path(backup_info.backup_path)
            
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            del self.backup_registry[backup_id]
            await self._save_backup_registry()
            
            self.logger.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup {backup_id}: {str(e)}")
            return False
    
    async def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """Clean up backups older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        backups_to_delete = []
        for backup_id, backup_info in self.backup_registry.items():
            if backup_info.created_at < cutoff_date:
                backups_to_delete.append(backup_id)
        
        deleted_count = 0
        for backup_id in backups_to_delete:
            if await self.delete_backup(backup_id):
                deleted_count += 1
        
        self.logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count
    
    async def get_rollback_history(self) -> List[RollbackResult]:
        """Get the history of all rollback operations"""
        return self.rollback_history.copy()
    
    async def export_backup_report(self, output_path: str):
        """Export backup information to a report file"""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "total_backups": len(self.backup_registry),
            "total_rollbacks": len(self.rollback_history),
            "backups": [asdict(backup) for backup in self.backup_registry.values()],
            "rollbacks": [asdict(rollback) for rollback in self.rollback_history]
        }
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(report_data, indent=2, default=str))
        
        self.logger.info(f"Backup report exported to {output_path}")
