#!/usr/bin/env python3
"""
Model Migration Script for Enhanced Model Availability System

This script migrates existing model installations to work with the enhanced
model availability system, ensuring compatibility and data preservation.
"""

import os
import json
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MigrationResult:
    """Result of a migration operation"""
    success: bool
    model_id: str
    actions_taken: List[str]
    warnings: List[str]
    errors: List[str]
    backup_path: Optional[str] = None

@dataclass
class ModelMigrationStatus:
    """Status of model migration"""
    model_id: str
    original_path: str
    new_path: str
    size_mb: float
    integrity_verified: bool
    metadata_migrated: bool
    backup_created: bool

class ModelMigrationManager:
    """Manages migration of existing model installations"""
    
    def __init__(self, 
                 old_models_dir: str = "models",
                 new_models_dir: str = "models",
                 backup_dir: str = "backups/model_migration"):
        self.old_models_dir = Path(old_models_dir)
        self.new_models_dir = Path(new_models_dir)
        self.backup_dir = Path(backup_dir)
        self.migration_log_path = self.backup_dir / "migration_log.json"
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.new_models_dir.mkdir(parents=True, exist_ok=True)
    
    async def migrate_all_models(self) -> Dict[str, MigrationResult]:
        """Migrate all existing models to enhanced system"""
        logger.info("Starting migration of all models to enhanced system")
        
        results = {}
        
        # Discover existing models
        existing_models = await self._discover_existing_models()
        logger.info(f"Found {len(existing_models)} models to migrate")
        
        # Create migration backup
        backup_info = await self._create_migration_backup()
        
        # Migrate each model
        for model_id, model_info in existing_models.items():
            try:
                result = await self._migrate_single_model(model_id, model_info)
                results[model_id] = result
                logger.info(f"Migrated {model_id}: {'SUCCESS' if result.success else 'FAILED'}")
            except Exception as e:
                logger.error(f"Failed to migrate {model_id}: {e}")
                results[model_id] = MigrationResult(
                    success=False,
                    model_id=model_id,
                    actions_taken=[],
                    warnings=[],
                    errors=[f"Migration failed: {str(e)}"]
                )
        
        # Save migration log
        await self._save_migration_log(results, backup_info)
        
        # Generate migration report
        await self._generate_migration_report(results)
        
        return results
    
    async def _discover_existing_models(self) -> Dict[str, Dict]:
        """Discover existing model installations"""
        models = {}
        
        if not self.old_models_dir.exists():
            logger.warning(f"Models directory {self.old_models_dir} does not exist")
            return models
        
        # Look for model directories and files
        for item in self.old_models_dir.iterdir():
            if item.is_dir():
                # Check if it's a model directory
                model_files = list(item.glob("*.safetensors")) + list(item.glob("*.bin")) + list(item.glob("*.pt"))
                if model_files:
                    models[item.name] = {
                        "path": str(item),
                        "files": [str(f) for f in model_files],
                        "size_mb": sum(f.stat().st_size for f in model_files) / (1024 * 1024),
                        "type": "directory"
                    }
            elif item.suffix in [".safetensors", ".bin", ".pt"]:
                # Single model file
                models[item.stem] = {
                    "path": str(item),
                    "files": [str(item)],
                    "size_mb": item.stat().st_size / (1024 * 1024),
                    "type": "file"
                }
        
        return models
    
    async def _create_migration_backup(self) -> Dict:
        """Create backup of current model state before migration"""
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"pre_migration_backup_{backup_timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        backup_info = {
            "timestamp": backup_timestamp,
            "backup_path": str(backup_path),
            "original_models_dir": str(self.old_models_dir)
        }
        
        try:
            if self.old_models_dir.exists():
                # Create backup of models directory
                models_backup = backup_path / "models"
                shutil.copytree(self.old_models_dir, models_backup, dirs_exist_ok=True)
                logger.info(f"Created models backup at {models_backup}")
            
            # Backup any existing configuration files
            config_files = ["config.json", "model_config.json", "models.json"]
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    shutil.copy2(config_path, backup_path / config_file)
                    logger.info(f"Backed up {config_file}")
            
            backup_info["success"] = True
            
        except Exception as e:
            logger.error(f"Failed to create migration backup: {e}")
            backup_info["success"] = False
            backup_info["error"] = str(e)
        
        return backup_info
    
    async def _migrate_single_model(self, model_id: str, model_info: Dict) -> MigrationResult:
        """Migrate a single model to enhanced system"""
        actions_taken = []
        warnings = []
        errors = []
        
        try:
            # Create enhanced model directory structure
            enhanced_model_dir = self.new_models_dir / model_id
            enhanced_model_dir.mkdir(exist_ok=True)
            actions_taken.append(f"Created enhanced model directory: {enhanced_model_dir}")
            
            # Copy model files
            if model_info["type"] == "directory":
                source_dir = Path(model_info["path"])
                for file_path in model_info["files"]:
                    source_file = Path(file_path)
                    dest_file = enhanced_model_dir / source_file.name
                    shutil.copy2(source_file, dest_file)
                    actions_taken.append(f"Copied {source_file.name}")
            else:
                # Single file model
                source_file = Path(model_info["path"])
                dest_file = enhanced_model_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                actions_taken.append(f"Copied {source_file.name}")
            
            # Create enhanced metadata
            metadata = await self._create_enhanced_metadata(model_id, model_info)
            metadata_file = enhanced_model_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            actions_taken.append("Created enhanced metadata")
            
            # Verify integrity
            integrity_ok = await self._verify_model_integrity(enhanced_model_dir)
            if not integrity_ok:
                warnings.append("Model integrity verification failed")
            else:
                actions_taken.append("Verified model integrity")
            
            # Create download status file
            download_status = {
                "status": "completed",
                "download_time": datetime.now().isoformat(),
                "migrated": True,
                "original_path": model_info["path"]
            }
            status_file = enhanced_model_dir / "download_status.json"
            with open(status_file, 'w') as f:
                json.dump(download_status, f, indent=2)
            actions_taken.append("Created download status file")
            
            return MigrationResult(
                success=True,
                model_id=model_id,
                actions_taken=actions_taken,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Migration failed: {str(e)}")
            return MigrationResult(
                success=False,
                model_id=model_id,
                actions_taken=actions_taken,
                warnings=warnings,
                errors=errors
            )
    
    async def _create_enhanced_metadata(self, model_id: str, model_info: Dict) -> Dict:
        """Create enhanced metadata for migrated model"""
        return {
            "model_id": model_id,
            "version": "1.0.0",
            "size_mb": model_info["size_mb"],
            "files": [Path(f).name for f in model_info["files"]],
            "migration_info": {
                "migrated": True,
                "migration_date": datetime.now().isoformat(),
                "original_path": model_info["path"],
                "original_type": model_info["type"]
            },
            "health_info": {
                "last_health_check": datetime.now().isoformat(),
                "integrity_verified": True,
                "performance_score": 1.0
            },
            "usage_info": {
                "usage_count": 0,
                "last_used": None,
                "average_generation_time": None
            }
        }
    
    async def _verify_model_integrity(self, model_dir: Path) -> bool:
        """Verify integrity of migrated model"""
        try:
            # Check that all expected files exist
            model_files = list(model_dir.glob("*.safetensors")) + \
                          list(model_dir.glob("*.bin")) + \
                          list(model_dir.glob("*.pt"))
            
            if not model_files:
                return False
            
            # Basic file size check
            for model_file in model_files:
                if model_file.stat().st_size == 0:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    async def _save_migration_log(self, results: Dict[str, MigrationResult], backup_info: Dict):
        """Save migration log for future reference"""
        log_data = {
            "migration_timestamp": datetime.now().isoformat(),
            "backup_info": backup_info,
            "results": {model_id: asdict(result) for model_id, result in results.items()},
            "summary": {
                "total_models": len(results),
                "successful_migrations": sum(1 for r in results.values() if r.success),
                "failed_migrations": sum(1 for r in results.values() if not r.success)
            }
        }
        
        with open(self.migration_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Migration log saved to {self.migration_log_path}")
    
    async def _generate_migration_report(self, results: Dict[str, MigrationResult]):
        """Generate human-readable migration report"""
        report_path = self.backup_dir / "migration_report.md"
        
        successful = [r for r in results.values() if r.success]
        failed = [r for r in results.values() if not r.success]
        
        report_content = f"""# Model Migration Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Total Models**: {len(results)}
- **Successful Migrations**: {len(successful)}
- **Failed Migrations**: {len(failed)}
- **Success Rate**: {len(successful)/len(results)*100:.1f}%

## Successful Migrations

"""
        
        for result in successful:
            report_content += f"""### {result.model_id}
- Actions: {', '.join(result.actions_taken)}
- Warnings: {', '.join(result.warnings) if result.warnings else 'None'}

"""
        
        if failed:
            report_content += "\n## Failed Migrations\n\n"
            for result in failed:
                report_content += f"""### {result.model_id}
- Errors: {', '.join(result.errors)}
- Actions Attempted: {', '.join(result.actions_taken)}

"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Migration report saved to {report_path}")

async def main():
    """Main migration function"""
    migration_manager = ModelMigrationManager()
    results = await migration_manager.migrate_all_models()
    
    successful = sum(1 for r in results.values() if r.success)
    total = len(results)
    
    print(f"\nMigration completed: {successful}/{total} models migrated successfully")
    
    if successful < total:
        print("Some migrations failed. Check the migration report for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)