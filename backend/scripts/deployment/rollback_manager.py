#!/usr/bin/env python3
"""
Rollback Manager for Enhanced Model Availability System

This script provides rollback capabilities for failed deployments of the
enhanced model availability system, ensuring system stability and data integrity.
"""

import os
import sys
import json
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RollbackType(Enum):
    """Types of rollback operations"""
    FULL_SYSTEM = "full_system"
    CONFIGURATION = "configuration"
    MODELS_ONLY = "models_only"
    DATABASE = "database"
    CODE_ONLY = "code_only"

class RollbackStatus(Enum):
    """Status of rollback operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class RollbackPoint:
    """Represents a rollback point in the system"""
    id: str
    timestamp: str
    description: str
    rollback_type: RollbackType
    backup_paths: Dict[str, str]
    metadata: Dict[str, Any]
    created_by: str = "system"

@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    rollback_point_id: str
    actions_taken: List[str]
    warnings: List[str]
    errors: List[str]
    duration_seconds: float
    status: RollbackStatus

class RollbackManager:
    """Manages rollback operations for enhanced model availability system"""
    
    def __init__(self, backup_dir: str = "backups/rollback_points"):
        self.backup_dir = Path(backup_dir)
        self.rollback_log_path = self.backup_dir / "rollback_log.json"
        self.rollback_points_file = self.backup_dir / "rollback_points.json"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize rollback points tracking
        self.rollback_points = self._load_rollback_points()
    
    async def create_rollback_point(self, description: str, 
                                  rollback_type: RollbackType = RollbackType.FULL_SYSTEM) -> str:
        """Create a rollback point before deployment"""
        logger.info(f"Creating rollback point: {description}")
        
        rollback_id = f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rollback_path = self.backup_dir / rollback_id
        rollback_path.mkdir(exist_ok=True)
        
        backup_paths = {}
        
        try:
            if rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.CONFIGURATION]:
                # Backup configuration files
                config_backup = await self._backup_configuration(rollback_path)
                backup_paths.update(config_backup)
            
            if rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.MODELS_ONLY]:
                # Backup models directory
                models_backup = await self._backup_models(rollback_path)
                backup_paths.update(models_backup)
            
            if rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.DATABASE]:
                # Backup database
                db_backup = await self._backup_database(rollback_path)
                backup_paths.update(db_backup)
            
            if rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.CODE_ONLY]:
                # Backup critical code files
                code_backup = await self._backup_code_files(rollback_path)
                backup_paths.update(code_backup)
            
            # Create rollback point metadata
            rollback_point = RollbackPoint(
                id=rollback_id,
                timestamp=datetime.now().isoformat(),
                description=description,
                rollback_type=rollback_type,
                backup_paths=backup_paths,
                metadata={
                    "system_info": await self._gather_system_info(),
                    "deployment_state": "pre_deployment"
                }
            )
            
            # Save rollback point
            self.rollback_points[rollback_id] = rollback_point
            await self._save_rollback_points()
            
            logger.info(f"Rollback point created: {rollback_id}")
            return rollback_id
            
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            # Cleanup partial backup
            if rollback_path.exists():
                shutil.rmtree(rollback_path)
            raise
    
    async def execute_rollback(self, rollback_point_id: str) -> RollbackResult:
        """Execute rollback to a specific point"""
        logger.info(f"Executing rollback to: {rollback_point_id}")
        
        start_time = datetime.now()
        actions_taken = []
        warnings = []
        errors = []
        
        try:
            if rollback_point_id not in self.rollback_points:
                raise ValueError(f"Rollback point not found: {rollback_point_id}")
            
            rollback_point = self.rollback_points[rollback_point_id]
            
            # Verify rollback point integrity
            if not await self._verify_rollback_point(rollback_point):
                raise ValueError(f"Rollback point integrity check failed: {rollback_point_id}")
            
            # Create pre-rollback backup
            pre_rollback_id = await self.create_rollback_point(
                f"Pre-rollback backup for {rollback_point_id}",
                rollback_point.rollback_type
            )
            actions_taken.append(f"Created pre-rollback backup: {pre_rollback_id}")
            
            # Execute rollback based on type
            if rollback_point.rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.CONFIGURATION]:
                config_result = await self._restore_configuration(rollback_point)
                actions_taken.extend(config_result["actions"])
                warnings.extend(config_result["warnings"])
                errors.extend(config_result["errors"])
            
            if rollback_point.rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.MODELS_ONLY]:
                models_result = await self._restore_models(rollback_point)
                actions_taken.extend(models_result["actions"])
                warnings.extend(models_result["warnings"])
                errors.extend(models_result["errors"])
            
            if rollback_point.rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.DATABASE]:
                db_result = await self._restore_database(rollback_point)
                actions_taken.extend(db_result["actions"])
                warnings.extend(db_result["warnings"])
                errors.extend(db_result["errors"])
            
            if rollback_point.rollback_type in [RollbackType.FULL_SYSTEM, RollbackType.CODE_ONLY]:
                code_result = await self._restore_code_files(rollback_point)
                actions_taken.extend(code_result["actions"])
                warnings.extend(code_result["warnings"])
                errors.extend(code_result["errors"])
            
            # Validate rollback success
            validation_result = await self._validate_rollback(rollback_point)
            actions_taken.extend(validation_result["actions"])
            warnings.extend(validation_result["warnings"])
            errors.extend(validation_result["errors"])
            
            # Determine final status
            success = len(errors) == 0
            status = RollbackStatus.COMPLETED if success else (
                RollbackStatus.PARTIAL if len(actions_taken) > 0 else RollbackStatus.FAILED
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = RollbackResult(
                success=success,
                rollback_point_id=rollback_point_id,
                actions_taken=actions_taken,
                warnings=warnings,
                errors=errors,
                duration_seconds=duration,
                status=status
            )
            
            # Log rollback operation
            await self._log_rollback_operation(result)
            
            logger.info(f"Rollback {'completed' if success else 'failed'}: {rollback_point_id}")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            errors.append(f"Rollback execution failed: {str(e)}")
            
            result = RollbackResult(
                success=False,
                rollback_point_id=rollback_point_id,
                actions_taken=actions_taken,
                warnings=warnings,
                errors=errors,
                duration_seconds=duration,
                status=RollbackStatus.FAILED
            )
            
            await self._log_rollback_operation(result)
            logger.error(f"Rollback failed: {e}")
            return result
    
    async def list_rollback_points(self) -> List[RollbackPoint]:
        """List available rollback points"""
        return list(self.rollback_points.values())
    
    async def cleanup_old_rollback_points(self, keep_count: int = 10) -> int:
        """Clean up old rollback points, keeping the most recent ones"""
        logger.info(f"Cleaning up old rollback points, keeping {keep_count} most recent")
        
        # Sort rollback points by timestamp
        sorted_points = sorted(
            self.rollback_points.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        # Identify points to remove
        points_to_remove = sorted_points[keep_count:]
        removed_count = 0
        
        for point in points_to_remove:
            try:
                # Remove backup directory
                rollback_path = self.backup_dir / point.id
                if rollback_path.exists():
                    shutil.rmtree(rollback_path)
                
                # Remove from tracking
                del self.rollback_points[point.id]
                removed_count += 1
                
                logger.info(f"Removed rollback point: {point.id}")
                
            except Exception as e:
                logger.error(f"Failed to remove rollback point {point.id}: {e}")
        
        # Save updated rollback points
        await self._save_rollback_points()
        
        logger.info(f"Cleaned up {removed_count} rollback points")
        return removed_count
    
    async def _backup_configuration(self, rollback_path: Path) -> Dict[str, str]:
        """Backup configuration files"""
        config_backup_path = rollback_path / "configuration"
        config_backup_path.mkdir(exist_ok=True)
        
        backup_paths = {}
        
        # Configuration files to backup
        config_files = [
            "config.json",
            "backend/config.json",
            "startup_config.json",
            "recovery_config.json"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                backup_file = config_backup_path / config_path.name
                shutil.copy2(config_path, backup_file)
                backup_paths[f"config_{config_path.name}"] = str(backup_file)
        
        return backup_paths
    
    async def _backup_models(self, rollback_path: Path) -> Dict[str, str]:
        """Backup models directory"""
        models_backup_path = rollback_path / "models"
        backup_paths = {}
        
        models_dir = Path("models")
        if models_dir.exists():
            shutil.copytree(models_dir, models_backup_path, dirs_exist_ok=True)
            backup_paths["models_directory"] = str(models_backup_path)
        
        return backup_paths
    
    async def _backup_database(self, rollback_path: Path) -> Dict[str, str]:
        """Backup database files"""
        db_backup_path = rollback_path / "database"
        db_backup_path.mkdir(exist_ok=True)
        
        backup_paths = {}
        
        # Database files to backup
        db_files = [
            "backend/wan22_tasks.db",
            "wan22_tasks.db"
        ]
        
        for db_file in db_files:
            db_path = Path(db_file)
            if db_path.exists():
                backup_file = db_backup_path / db_path.name
                shutil.copy2(db_path, backup_file)
                backup_paths[f"database_{db_path.name}"] = str(backup_file)
        
        return backup_paths
    
    async def _backup_code_files(self, rollback_path: Path) -> Dict[str, str]:
        """Backup critical code files"""
        code_backup_path = rollback_path / "code"
        code_backup_path.mkdir(exist_ok=True)
        
        backup_paths = {}
        
        # Critical directories to backup
        critical_dirs = [
            "backend/core",
            "backend/api",
            "backend/services",
            "backend/websocket"
        ]
        
        for dir_path in critical_dirs:
            source_dir = Path(dir_path)
            if source_dir.exists():
                backup_dir = code_backup_path / source_dir.name
                shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
                backup_paths[f"code_{source_dir.name}"] = str(backup_dir)
        
        return backup_paths
    
    async def _restore_configuration(self, rollback_point: RollbackPoint) -> Dict[str, List[str]]:
        """Restore configuration files"""
        actions = []
        warnings = []
        errors = []
        
        try:
            rollback_path = self.backup_dir / rollback_point.id
            config_backup_path = rollback_path / "configuration"
            
            if config_backup_path.exists():
                for backup_file in config_backup_path.iterdir():
                    if backup_file.is_file():
                        # Determine original location
                        if backup_file.name == "config.json":
                            target_path = Path("config.json")
                        elif backup_file.name.startswith("backend_"):
                            target_path = Path("backend") / backup_file.name.replace("backend_", "")
                        else:
                            target_path = Path(backup_file.name)
                        
                        # Restore file
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_file, target_path)
                        actions.append(f"Restored configuration: {target_path}")
            else:
                warnings.append("No configuration backup found")
                
        except Exception as e:
            errors.append(f"Configuration restore failed: {str(e)}")
        
        return {"actions": actions, "warnings": warnings, "errors": errors}
    
    async def _restore_models(self, rollback_point: RollbackPoint) -> Dict[str, List[str]]:
        """Restore models directory"""
        actions = []
        warnings = []
        errors = []
        
        try:
            rollback_path = self.backup_dir / rollback_point.id
            models_backup_path = rollback_path / "models"
            
            if models_backup_path.exists():
                models_dir = Path("models")
                
                # Remove current models directory
                if models_dir.exists():
                    shutil.rmtree(models_dir)
                    actions.append("Removed current models directory")
                
                # Restore from backup
                shutil.copytree(models_backup_path, models_dir)
                actions.append("Restored models directory")
            else:
                warnings.append("No models backup found")
                
        except Exception as e:
            errors.append(f"Models restore failed: {str(e)}")
        
        return {"actions": actions, "warnings": warnings, "errors": errors}
    
    async def _restore_database(self, rollback_point: RollbackPoint) -> Dict[str, List[str]]:
        """Restore database files"""
        actions = []
        warnings = []
        errors = []
        
        try:
            rollback_path = self.backup_dir / rollback_point.id
            db_backup_path = rollback_path / "database"
            
            if db_backup_path.exists():
                for backup_file in db_backup_path.iterdir():
                    if backup_file.is_file():
                        # Determine original location
                        if backup_file.name == "wan22_tasks.db":
                            target_paths = [Path("backend/wan22_tasks.db"), Path("wan22_tasks.db")]
                        else:
                            target_paths = [Path(backup_file.name)]
                        
                        # Restore to all possible locations
                        for target_path in target_paths:
                            if target_path.parent != Path("."):
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_file, target_path)
                            actions.append(f"Restored database: {target_path}")
            else:
                warnings.append("No database backup found")
                
        except Exception as e:
            errors.append(f"Database restore failed: {str(e)}")
        
        return {"actions": actions, "warnings": warnings, "errors": errors}
    
    async def _restore_code_files(self, rollback_point: RollbackPoint) -> Dict[str, List[str]]:
        """Restore critical code files"""
        actions = []
        warnings = []
        errors = []
        
        try:
            rollback_path = self.backup_dir / rollback_point.id
            code_backup_path = rollback_path / "code"
            
            if code_backup_path.exists():
                for backup_dir in code_backup_path.iterdir():
                    if backup_dir.is_dir():
                        target_dir = Path("backend") / backup_dir.name
                        
                        # Remove current directory
                        if target_dir.exists():
                            shutil.rmtree(target_dir)
                            actions.append(f"Removed current directory: {target_dir}")
                        
                        # Restore from backup
                        shutil.copytree(backup_dir, target_dir)
                        actions.append(f"Restored code directory: {target_dir}")
            else:
                warnings.append("No code backup found")
                
        except Exception as e:
            errors.append(f"Code restore failed: {str(e)}")
        
        return {"actions": actions, "warnings": warnings, "errors": errors}
    
    async def _verify_rollback_point(self, rollback_point: RollbackPoint) -> bool:
        """Verify rollback point integrity"""
        try:
            rollback_path = self.backup_dir / rollback_point.id
            
            if not rollback_path.exists():
                return False
            
            # Check that all backup paths exist
            for backup_type, backup_path in rollback_point.backup_paths.items():
                if not Path(backup_path).exists():
                    logger.warning(f"Backup path missing: {backup_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback point verification failed: {e}")
            return False
    
    async def _validate_rollback(self, rollback_point: RollbackPoint) -> Dict[str, List[str]]:
        """Validate rollback success"""
        actions = []
        warnings = []
        errors = []
        
        try:
            # Basic validation - check that key files exist
            key_files = ["config.json", "backend/config.json"]
            
            for key_file in key_files:
                if Path(key_file).exists():
                    actions.append(f"Validated file exists: {key_file}")
                else:
                    warnings.append(f"Key file missing after rollback: {key_file}")
            
            # Check models directory
            if Path("models").exists():
                actions.append("Validated models directory exists")
            else:
                warnings.append("Models directory missing after rollback")
            
        except Exception as e:
            errors.append(f"Rollback validation failed: {str(e)}")
        
        return {"actions": actions, "warnings": warnings, "errors": errors}
    
    async def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for rollback point"""
        try:
            import platform
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return {"timestamp": datetime.now().isoformat()}
    
    def _load_rollback_points(self) -> Dict[str, RollbackPoint]:
        """Load rollback points from storage"""
        if self.rollback_points_file.exists():
            try:
                with open(self.rollback_points_file, 'r') as f:
                    data = json.load(f)
                
                rollback_points = {}
                for point_id, point_data in data.items():
                    rollback_points[point_id] = RollbackPoint(
                        id=point_data["id"],
                        timestamp=point_data["timestamp"],
                        description=point_data["description"],
                        rollback_type=RollbackType(point_data["rollback_type"]),
                        backup_paths=point_data["backup_paths"],
                        metadata=point_data["metadata"],
                        created_by=point_data.get("created_by", "system")
                    )
                
                return rollback_points
                
            except Exception as e:
                logger.error(f"Failed to load rollback points: {e}")
        
        return {}
    
    async def _save_rollback_points(self):
        """Save rollback points to storage"""
        try:
            data = {}
            for point_id, point in self.rollback_points.items():
                data[point_id] = {
                    "id": point.id,
                    "timestamp": point.timestamp,
                    "description": point.description,
                    "rollback_type": point.rollback_type.value,
                    "backup_paths": point.backup_paths,
                    "metadata": point.metadata,
                    "created_by": point.created_by
                }
            
            with open(self.rollback_points_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save rollback points: {e}")
    
    async def _log_rollback_operation(self, result: RollbackResult):
        """Log rollback operation"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": "rollback",
                "result": asdict(result)
            }
            
            # Load existing log
            log_data = []
            if self.rollback_log_path.exists():
                with open(self.rollback_log_path, 'r') as f:
                    log_data = json.load(f)
            
            # Add new entry
            log_data.append(log_entry)
            
            # Save log
            with open(self.rollback_log_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to log rollback operation: {e}")

async def main():
    """Main rollback function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Model Availability Rollback Manager")
    parser.add_argument("action", choices=["create", "rollback", "list", "cleanup"],
                       help="Action to perform")
    parser.add_argument("--description", help="Description for rollback point")
    parser.add_argument("--rollback-id", help="Rollback point ID")
    parser.add_argument("--type", choices=["full_system", "configuration", "models_only", "database", "code_only"],
                       default="full_system", help="Type of rollback")
    parser.add_argument("--keep", type=int, default=10, help="Number of rollback points to keep during cleanup")
    
    args = parser.parse_args()
    
    rollback_manager = RollbackManager()
    
    if args.action == "create":
        if not args.description:
            print("Description required for creating rollback point")
            return 1
        
        rollback_type = RollbackType(args.type)
        rollback_id = await rollback_manager.create_rollback_point(args.description, rollback_type)
        print(f"Created rollback point: {rollback_id}")
        
    elif args.action == "rollback":
        if not args.rollback_id:
            print("Rollback ID required for rollback operation")
            return 1
        
        result = await rollback_manager.execute_rollback(args.rollback_id)
        print(f"Rollback {'succeeded' if result.success else 'failed'}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        return 0 if result.success else 1
        
    elif args.action == "list":
        points = await rollback_manager.list_rollback_points()
        print(f"Available rollback points ({len(points)}):")
        for point in sorted(points, key=lambda x: x.timestamp, reverse=True):
            print(f"  {point.id}: {point.description} ({point.timestamp})")
        
    elif args.action == "cleanup":
        removed = await rollback_manager.cleanup_old_rollback_points(args.keep)
        print(f"Cleaned up {removed} old rollback points")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
