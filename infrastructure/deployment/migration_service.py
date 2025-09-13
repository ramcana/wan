"""
WAN Model Migration Service

Handles the actual migration of models from placeholder to production,
including file operations, configuration updates, and dependency management.
"""

import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiofiles
import aiofiles.os


@dataclass
class MigrationResult:
    """Result of a model migration operation"""
    model_name: str
    success: bool
    source_path: str
    target_path: str
    file_size: int
    checksum: str
    migration_time: float
    error: Optional[str] = None


@dataclass
class ModelMetadata:
    """Metadata for a WAN model"""
    name: str
    version: str
    size_bytes: int
    checksum: str
    dependencies: List[str]
    config_requirements: Dict[str, Any]
    hardware_requirements: Dict[str, Any]


class MigrationService:
    """
    Service for migrating WAN models from development to production
    
    Handles:
    - File copying with integrity verification
    - Configuration updates
    - Dependency management
    - Atomic operations with rollback capability
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.migration_log: List[MigrationResult] = []
    
    async def migrate_model(self, model_name: str, deployment_id: str) -> MigrationResult:
        """
        Migrate a single model from source to target location
        
        Args:
            model_name: Name of the model to migrate
            deployment_id: ID of the deployment operation
            
        Returns:
            MigrationResult with migration details
        """
        start_time = datetime.now()
        self.logger.info(f"Starting migration of model {model_name}")
        
        try:
            # Determine source and target paths
            source_path = await self._find_model_source(model_name)
            target_path = Path(self.config.target_models_path) / model_name
            
            # Load model metadata
            metadata = await self._load_model_metadata(source_path)
            
            # Validate source model
            await self._validate_source_model(source_path, metadata)
            
            # Prepare target directory
            await self._prepare_target_directory(target_path)
            
            # Copy model files with integrity checking
            checksum = await self._copy_model_files(source_path, target_path)
            
            # Update model configuration
            await self._update_model_configuration(model_name, target_path, metadata)
            
            # Verify migration integrity
            await self._verify_migration_integrity(target_path, metadata)
            
            # Calculate migration time and file size
            migration_time = (datetime.now() - start_time).total_seconds()
            file_size = await self._calculate_directory_size(target_path)
            
            result = MigrationResult(
                model_name=model_name,
                success=True,
                source_path=str(source_path),
                target_path=str(target_path),
                file_size=file_size,
                checksum=checksum,
                migration_time=migration_time
            )
            
            self.migration_log.append(result)
            self.logger.info(f"Successfully migrated model {model_name} in {migration_time:.2f}s")
            
            return result
            
        except Exception as e:
            migration_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Migration failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            
            result = MigrationResult(
                model_name=model_name,
                success=False,
                source_path=str(source_path) if 'source_path' in locals() else "",
                target_path=str(target_path) if 'target_path' in locals() else "",
                file_size=0,
                checksum="",
                migration_time=migration_time,
                error=error_msg
            )
            
            self.migration_log.append(result)
            return result
    
    async def _find_model_source(self, model_name: str) -> Path:
        """Find the source path for a model"""
        # Check multiple possible source locations
        possible_sources = [
            Path(self.config.source_models_path) / model_name,
            Path("models") / model_name,
            Path("models") / "models" / model_name,
            Path("infrastructure") / "storage" / "models" / model_name
        ]
        
        for source_path in possible_sources:
            if source_path.exists():
                return source_path
        
        raise FileNotFoundError(f"Model {model_name} not found in any source location")
    
    async def _load_model_metadata(self, model_path: Path) -> ModelMetadata:
        """Load metadata for a model"""
        metadata_file = model_path / "metadata.json"
        
        if metadata_file.exists():
            async with aiofiles.open(metadata_file, 'r') as f:
                metadata_dict = json.loads(await f.read())
                return ModelMetadata(**metadata_dict)
        else:
            # Generate basic metadata if not available
            size = await self._calculate_directory_size(model_path)
            checksum = await self._calculate_directory_checksum(model_path)
            
            return ModelMetadata(
                name=model_path.name,
                version="1.0.0",
                size_bytes=size,
                checksum=checksum,
                dependencies=[],
                config_requirements={},
                hardware_requirements={}
            )
    
    async def _validate_source_model(self, source_path: Path, metadata: ModelMetadata):
        """Validate the source model before migration"""
        if not source_path.exists():
            raise FileNotFoundError(f"Source model path does not exist: {source_path}")
        
        # Check if it's a directory or file
        if not source_path.is_dir():
            raise ValueError(f"Source model must be a directory: {source_path}")
        
        # Verify model files exist
        required_files = ["config.json"]  # Add more as needed
        for required_file in required_files:
            file_path = source_path / required_file
            if not file_path.exists():
                self.logger.warning(f"Required file {required_file} not found in {source_path}")
        
        # Verify checksum if available
        if metadata.checksum:
            current_checksum = await self._calculate_directory_checksum(source_path)
            if current_checksum != metadata.checksum:
                self.logger.warning(f"Checksum mismatch for {source_path}")
    
    async def _prepare_target_directory(self, target_path: Path):
        """Prepare the target directory for migration"""
        if target_path.exists():
            # Create backup of existing model
            backup_path = target_path.parent / f"{target_path.name}_backup_{int(datetime.now().timestamp())}"
            shutil.move(str(target_path), str(backup_path))
            self.logger.info(f"Backed up existing model to {backup_path}")
        
        target_path.mkdir(parents=True, exist_ok=True)
    
    async def _copy_model_files(self, source_path: Path, target_path: Path) -> str:
        """Copy model files from source to target with integrity checking"""
        self.logger.info(f"Copying files from {source_path} to {target_path}")
        
        # Use shutil.copytree for efficient directory copying
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.copytree(str(source_path), str(target_path))
        
        # Calculate checksum of copied files
        checksum = await self._calculate_directory_checksum(target_path)
        
        return checksum
    
    async def _update_model_configuration(self, model_name: str, target_path: Path, metadata: ModelMetadata):
        """Update model configuration after migration"""
        config_file = target_path / "config.json"
        
        # Load existing config or create new one
        if config_file.exists():
            async with aiofiles.open(config_file, 'r') as f:
                config = json.loads(await f.read())
        else:
            config = {}
        
        # Update configuration with deployment information
        config.update({
            "deployment_info": {
                "deployed_at": datetime.now().isoformat(),
                "model_name": model_name,
                "version": metadata.version,
                "checksum": metadata.checksum
            }
        })
        
        # Write updated configuration
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(json.dumps(config, indent=2))
    
    async def _verify_migration_integrity(self, target_path: Path, metadata: ModelMetadata):
        """Verify the integrity of the migrated model"""
        # Check that all expected files exist
        if not target_path.exists():
            raise Exception(f"Target path does not exist after migration: {target_path}")
        
        # Verify file count and size
        target_size = await self._calculate_directory_size(target_path)
        if abs(target_size - metadata.size_bytes) > (metadata.size_bytes * 0.01):  # 1% tolerance
            self.logger.warning(f"Size mismatch: expected {metadata.size_bytes}, got {target_size}")
        
        # Verify checksum
        target_checksum = await self._calculate_directory_checksum(target_path)
        if metadata.checksum and target_checksum != metadata.checksum:
            raise Exception(f"Checksum verification failed for {target_path}")
    
    async def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    async def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate a checksum for all files in a directory"""
        hasher = hashlib.sha256()
        
        # Sort files for consistent checksum
        files = sorted(directory.rglob('*'))
        
        for file_path in files:
            if file_path.is_file():
                # Include file path in hash for structure verification
                hasher.update(str(file_path.relative_to(directory)).encode())
                
                # Include file content
                async with aiofiles.open(file_path, 'rb') as f:
                    while chunk := await f.read(8192):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def get_migration_history(self) -> List[MigrationResult]:
        """Get the history of all migrations"""
        return self.migration_log.copy()
    
    async def rollback_migration(self, model_name: str) -> bool:
        """Rollback a model migration"""
        target_path = Path(self.config.target_models_path) / model_name
        
        # Look for backup
        backup_pattern = f"{model_name}_backup_*"
        backup_paths = list(target_path.parent.glob(backup_pattern))
        
        if not backup_paths:
            self.logger.error(f"No backup found for model {model_name}")
            return False
        
        # Use the most recent backup
        latest_backup = max(backup_paths, key=lambda p: p.stat().st_mtime)
        
        try:
            # Remove current version
            if target_path.exists():
                shutil.rmtree(target_path)
            
            # Restore from backup
            shutil.move(str(latest_backup), str(target_path))
            
            self.logger.info(f"Successfully rolled back model {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback model {model_name}: {str(e)}")
            return False
