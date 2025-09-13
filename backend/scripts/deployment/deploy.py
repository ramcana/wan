#!/usr/bin/env python3
"""
Deployment Automation Script for Enhanced Model Availability System

This script automates the deployment of the enhanced model availability system,
including validation, migration, monitoring setup, and rollback capabilities.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentPhase(Enum):
    """Deployment phases"""
    PRE_VALIDATION = "pre_validation"
    BACKUP_CREATION = "backup_creation"
    MIGRATION = "migration"
    DEPLOYMENT = "deployment"
    POST_VALIDATION = "post_validation"
    MONITORING_SETUP = "monitoring_setup"
    HEALTH_CHECK = "health_check"
    CLEANUP = "cleanup"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    success: bool
    deployment_id: str
    phases_completed: List[DeploymentPhase]
    phases_failed: List[DeploymentPhase]
    warnings: List[str]
    errors: List[str]
    rollback_point_id: Optional[str] = None
    duration_seconds: float = 0.0

class EnhancedModelAvailabilityDeployer:
    """Main deployment orchestrator"""
    
    def __init__(self, config_file: str = "deployment_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.deployment_log_path = Path("logs/deployment.log")
        
        # Ensure logs directory exists
        self.deployment_log_path.parent.mkdir(exist_ok=True)
        
        # Setup file logging
        file_handler = logging.FileHandler(self.deployment_log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    async def deploy(self, dry_run: bool = False, skip_validation: bool = False,
                    skip_backup: bool = False, force: bool = False) -> DeploymentResult:
        """Execute full deployment"""
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting deployment: {deployment_id}")
        
        start_time = datetime.now()
        phases_completed = []
        phases_failed = []
        warnings = []
        errors = []
        rollback_point_id = None
        
        try:
            # Phase 1: Pre-deployment validation
            if not skip_validation:
                logger.info("Phase 1: Pre-deployment validation")
                try:
                    await self._pre_deployment_validation()
                    phases_completed.append(DeploymentPhase.PRE_VALIDATION)
                    logger.info("✓ Pre-deployment validation completed")
                except Exception as e:
                    phases_failed.append(DeploymentPhase.PRE_VALIDATION)
                    error_msg = f"Pre-deployment validation failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    if not force:
                        raise
            
            # Phase 2: Create rollback point
            if not skip_backup and not dry_run:
                logger.info("Phase 2: Creating rollback point")
                try:
                    rollback_point_id = await self._create_rollback_point(deployment_id)
                    phases_completed.append(DeploymentPhase.BACKUP_CREATION)
                    logger.info(f"✓ Rollback point created: {rollback_point_id}")
                except Exception as e:
                    phases_failed.append(DeploymentPhase.BACKUP_CREATION)
                    error_msg = f"Rollback point creation failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    if not force:
                        raise
            
            # Phase 3: Migration
            logger.info("Phase 3: Model migration")
            try:
                migration_result = await self._execute_migration(dry_run)
                phases_completed.append(DeploymentPhase.MIGRATION)
                if migration_result.get("warnings"):
                    warnings.extend(migration_result["warnings"])
                logger.info("✓ Model migration completed")
            except Exception as e:
                phases_failed.append(DeploymentPhase.MIGRATION)
                error_msg = f"Migration failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                if not force:
                    raise
            
            # Phase 4: Core deployment
            logger.info("Phase 4: Core system deployment")
            try:
                await self._deploy_core_system(dry_run)
                phases_completed.append(DeploymentPhase.DEPLOYMENT)
                logger.info("✓ Core system deployment completed")
            except Exception as e:
                phases_failed.append(DeploymentPhase.DEPLOYMENT)
                error_msg = f"Core deployment failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                if not force:
                    raise
            
            # Phase 5: Post-deployment validation
            logger.info("Phase 5: Post-deployment validation")
            try:
                validation_result = await self._post_deployment_validation()
                phases_completed.append(DeploymentPhase.POST_VALIDATION)
                if validation_result.get("warnings"):
                    warnings.extend(validation_result["warnings"])
                logger.info("✓ Post-deployment validation completed")
            except Exception as e:
                phases_failed.append(DeploymentPhase.POST_VALIDATION)
                error_msg = f"Post-deployment validation failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                if not force:
                    raise
            
            # Phase 6: Monitoring setup
            if self.config.get("setup_monitoring", True) and not dry_run:
                logger.info("Phase 6: Setting up monitoring")
                try:
                    await self._setup_monitoring()
                    phases_completed.append(DeploymentPhase.MONITORING_SETUP)
                    logger.info("✓ Monitoring setup completed")
                except Exception as e:
                    phases_failed.append(DeploymentPhase.MONITORING_SETUP)
                    warning_msg = f"Monitoring setup failed: {str(e)}"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
            
            # Phase 7: Health check
            logger.info("Phase 7: Final health check")
            try:
                health_result = await self._final_health_check()
                phases_completed.append(DeploymentPhase.HEALTH_CHECK)
                if not health_result["healthy"]:
                    warnings.append("System health check shows degraded status")
                logger.info("✓ Final health check completed")
            except Exception as e:
                phases_failed.append(DeploymentPhase.HEALTH_CHECK)
                warning_msg = f"Health check failed: {str(e)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
            
            # Phase 8: Cleanup
            if self.config.get("cleanup_after_deployment", True) and not dry_run:
                logger.info("Phase 8: Cleanup")
                try:
                    await self._cleanup_deployment()
                    phases_completed.append(DeploymentPhase.CLEANUP)
                    logger.info("✓ Cleanup completed")
                except Exception as e:
                    phases_failed.append(DeploymentPhase.CLEANUP)
                    warning_msg = f"Cleanup failed: {str(e)}"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
            
            duration = (datetime.now() - start_time).total_seconds()
            success = len(phases_failed) == 0
            
            result = DeploymentResult(
                success=success,
                deployment_id=deployment_id,
                phases_completed=phases_completed,
                phases_failed=phases_failed,
                warnings=warnings,
                errors=errors,
                rollback_point_id=rollback_point_id,
                duration_seconds=duration
            )
            
            # Log deployment result
            await self._log_deployment_result(result)
            
            if success:
                logger.info(f"🎉 Deployment completed successfully: {deployment_id}")
            else:
                logger.error(f"❌ Deployment completed with errors: {deployment_id}")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            errors.append(f"Deployment failed: {str(e)}")
            
            result = DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                phases_completed=phases_completed,
                phases_failed=phases_failed,
                warnings=warnings,
                errors=errors,
                rollback_point_id=rollback_point_id,
                duration_seconds=duration
            )
            
            await self._log_deployment_result(result)
            logger.error(f"💥 Deployment failed: {deployment_id}")
            
            # Attempt rollback if rollback point exists
            if rollback_point_id and not dry_run:
                logger.info("Attempting automatic rollback...")
                try:
                    await self._execute_rollback(rollback_point_id)
                    result.success = False  # Still failed, but rolled back
                    logger.info("✓ Automatic rollback completed")
                except Exception as rollback_error:
                    logger.error(f"Rollback also failed: {rollback_error}")
                    errors.append(f"Rollback failed: {str(rollback_error)}")
            
            return result
    
    async def _pre_deployment_validation(self):
        """Execute pre-deployment validation"""
        try:
            from deployment_validator import EnhancedModelAvailabilityValidator
            
            validator = EnhancedModelAvailabilityValidator()
            report = await validator.validate_deployment()
            
            if not report.overall_success:
                critical_issues = [r for r in report.results if not r.success and r.level.value == "critical"]
                if critical_issues:
                    raise Exception(f"Critical validation issues found: {len(critical_issues)}")
            
        except ImportError:
            logger.warning("Deployment validator not available, skipping detailed validation")
            
            # Basic validation
            required_dirs = ["backend/core", "backend/api", "backend/services"]
            missing_dirs = [d for d in required_dirs if not Path(d).exists()]
            
            if missing_dirs:
                raise Exception(f"Required directories missing: {', '.join(missing_dirs)}")
    
    async def _create_rollback_point(self, deployment_id: str) -> str:
        """Create rollback point"""
        try:
            from rollback_manager import RollbackManager, RollbackType
            
            rollback_manager = RollbackManager()
            rollback_point_id = await rollback_manager.create_rollback_point(
                description=f"Pre-deployment backup for {deployment_id}",
                rollback_type=RollbackType.FULL_SYSTEM
            )
            
            return rollback_point_id
            
        except ImportError:
            logger.warning("Rollback manager not available")
            return None
    
    async def _execute_migration(self, dry_run: bool) -> Dict[str, Any]:
        """Execute model migration"""
        try:
            from model_migration import ModelMigrationManager
            
            migration_manager = ModelMigrationManager()
            
            if dry_run:
                logger.info("[DRY RUN] Would execute model migration")
                return {"warnings": ["Dry run - migration not executed"]}
            
            results = await migration_manager.migrate_all_models()
            
            failed_migrations = [r for r in results.values() if not r.success]
            if failed_migrations:
                warnings = [f"Migration failed for {len(failed_migrations)} models"]
                return {"warnings": warnings}
            
            return {}
            
        except ImportError:
            logger.warning("Model migration manager not available")
            return {"warnings": ["Model migration skipped - manager not available"]}
    
    async def _deploy_core_system(self, dry_run: bool):
        """Deploy core system components"""
        if dry_run:
            logger.info("[DRY RUN] Would deploy core system components")
            return
        
        # Verify core components are available
        core_components = [
            "backend.core.enhanced_model_downloader",
            "backend.core.model_health_monitor",
            "backend.core.model_availability_manager",
            "backend.core.intelligent_fallback_manager",
            "backend.core.model_usage_analytics",
            "backend.core.enhanced_error_recovery",
            "backend.core.model_update_manager"
        ]
        
        missing_components = []
        for component in core_components:
            try:
                __import__(component)
            except ImportError:
                missing_components.append(component)
        
        if missing_components:
            raise Exception(f"Core components missing: {', '.join(missing_components)}")
        
        # Initialize database if needed
        await self._initialize_database()
        
        # Update configuration
        await self._update_configuration()
    
    async def _post_deployment_validation(self) -> Dict[str, Any]:
        """Execute post-deployment validation"""
        try:
            from deployment_validator import EnhancedModelAvailabilityValidator
            
            validator = EnhancedModelAvailabilityValidator()
            report = await validator.validate_deployment()
            
            warnings = []
            if not report.overall_success:
                warning_issues = [r for r in report.results if not r.success and r.level.value == "warning"]
                if warning_issues:
                    warnings = [f"Post-deployment warnings: {len(warning_issues)}"]
                
                critical_issues = [r for r in report.results if not r.success and r.level.value == "critical"]
                if critical_issues:
                    raise Exception(f"Critical post-deployment issues: {len(critical_issues)}")
            
            return {"warnings": warnings}
            
        except ImportError:
            logger.warning("Post-deployment validator not available")
            return {"warnings": ["Post-deployment validation skipped"]}
    
    async def _setup_monitoring(self):
        """Setup monitoring system"""
        try:
            from monitoring_setup import EnhancedModelAvailabilityMonitor, setup_monitoring_config
            
            # Create monitoring configuration if it doesn't exist
            config_path = Path("monitoring_config.json")
            if not config_path.exists():
                setup_monitoring_config()
            
            # Initialize monitoring
            monitor = EnhancedModelAvailabilityMonitor()
            monitor.start_monitoring()
            
            # Let it run for a few seconds to collect initial metrics
            await asyncio.sleep(5)
            
            status = monitor.get_monitoring_status()
            if not status["running"]:
                raise Exception("Monitoring system failed to start")
            
            monitor.stop_monitoring()
            
        except ImportError:
            logger.warning("Monitoring setup not available")
    
    async def _final_health_check(self) -> Dict[str, Any]:
        """Execute final health check"""
        try:
            # Import health check functions
            sys.path.append(str(Path(__file__).parent.parent / "api"))
            from deployment_health import (
                check_database_health,
                check_file_system_health,
                check_enhanced_downloader_health
            )
            
            # Run critical health checks
            await check_database_health()
            await check_file_system_health()
            await check_enhanced_downloader_health()
            
            return {"healthy": True}
            
        except Exception as e:
            logger.warning(f"Health check issues detected: {e}")
            return {"healthy": False, "issues": str(e)}
    
    async def _cleanup_deployment(self):
        """Cleanup after deployment"""
        # Clean up temporary files
        temp_dirs = ["temp", "tmp", ".tmp"]
        for temp_dir in temp_dirs:
            temp_path = Path(temp_dir)
            if temp_path.exists() and temp_path.is_dir():
                import shutil
                try:
                    shutil.rmtree(temp_path)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_dir}: {e}")
        
        # Clean up old log files (keep last 10)
        logs_dir = Path("logs")
        if logs_dir.exists():
            log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_log in log_files[10:]:
                try:
                    old_log.unlink()
                    logger.info(f"Cleaned up old log file: {old_log}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {old_log}: {e}")
    
    async def _initialize_database(self):
        """Initialize database if needed"""
        try:
            import sqlite3
            
            db_paths = ["backend/wan22_tasks.db", "wan22_tasks.db"]
            db_path = None
            
            for path in db_paths:
                if Path(path).exists():
                    db_path = path
                    break
            
            if not db_path:
                # Create database
                db_path = "backend/wan22_tasks.db"
                Path(db_path).parent.mkdir(exist_ok=True)
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create basic tables for enhanced features
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_usage_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        last_used TIMESTAMP,
                        average_generation_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_health_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        health_score REAL NOT NULL,
                        integrity_score REAL NOT NULL,
                        performance_score REAL NOT NULL,
                        check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS download_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        download_status TEXT NOT NULL,
                        download_size_mb REAL,
                        download_time_seconds REAL,
                        retry_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                conn.close()
                
                logger.info(f"Database initialized: {db_path}")
            
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    
    async def _update_configuration(self):
        """Update configuration for enhanced features"""
        config_path = Path("backend/config.json")
        
        try:
            # Load existing config
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Add enhanced model availability configuration
            if "enhanced_model_availability" not in config:
                config["enhanced_model_availability"] = {
                    "download_retry_attempts": 3,
                    "health_check_interval": 3600,
                    "analytics_enabled": True,
                    "fallback_enabled": True,
                    "auto_cleanup_enabled": True,
                    "monitoring_enabled": True
                }
                
                # Ensure parent directory exists
                config_path.parent.mkdir(exist_ok=True)
                
                # Save updated config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info("Configuration updated with enhanced model availability settings")
            
        except Exception as e:
            logger.warning(f"Configuration update failed: {e}")
    
    async def _execute_rollback(self, rollback_point_id: str):
        """Execute rollback"""
        try:
            from rollback_manager import RollbackManager
            
            rollback_manager = RollbackManager()
            result = await rollback_manager.execute_rollback(rollback_point_id)
            
            if not result.success:
                raise Exception(f"Rollback failed: {', '.join(result.errors)}")
            
        except ImportError:
            raise Exception("Rollback manager not available")
    
    async def _log_deployment_result(self, result: DeploymentResult):
        """Log deployment result"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "deployment_id": result.deployment_id,
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "phases_completed": [p.value for p in result.phases_completed],
            "phases_failed": [p.value for p in result.phases_failed],
            "warnings": result.warnings,
            "errors": result.errors,
            "rollback_point_id": result.rollback_point_id
        }
        
        # Save to deployment log file
        deployment_log_file = Path("logs/deployment_history.json")
        deployment_log_file.parent.mkdir(exist_ok=True)
        
        # Load existing log
        log_data = []
        if deployment_log_file.exists():
            try:
                with open(deployment_log_file, 'r') as f:
                    log_data = json.load(f)
            except Exception:
                log_data = []
        
        # Add new entry
        log_data.append(log_entry)
        
        # Keep only last 100 entries
        log_data = log_data[-100:]
        
        # Save log
        with open(deployment_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "setup_monitoring": True,
            "cleanup_after_deployment": True,
            "auto_rollback_on_failure": True,
            "validation_timeout_seconds": 300,
            "migration_timeout_seconds": 600,
            "deployment_timeout_seconds": 1800
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults
                default_config.update(config)
                
            except Exception as e:
                logger.error(f"Error loading deployment config: {e}")
        
        return default_config

async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Enhanced Model Availability Deployment Tool")
    parser.add_argument("--config", help="Deployment configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")
    parser.add_argument("--skip-validation", action="store_true", help="Skip pre-deployment validation")
    parser.add_argument("--skip-backup", action="store_true", help="Skip rollback point creation")
    parser.add_argument("--force", action="store_true", help="Continue deployment even if non-critical phases fail")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize deployer
    config_file = args.config or "deployment_config.json"
    deployer = EnhancedModelAvailabilityDeployer(config_file)
    
    # Execute deployment
    try:
        result = await deployer.deploy(
            dry_run=args.dry_run,
            skip_validation=args.skip_validation,
            skip_backup=args.skip_backup,
            force=args.force
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Deployment Summary: {result.deployment_id}")
        print(f"{'='*60}")
        print(f"Success: {'✓' if result.success else '✗'}")
        print(f"Duration: {result.duration_seconds:.1f} seconds")
        print(f"Phases Completed: {len(result.phases_completed)}")
        print(f"Phases Failed: {len(result.phases_failed)}")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.rollback_point_id:
            print(f"Rollback Point: {result.rollback_point_id}")
        
        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  ❌ {error}")
        
        print(f"\nDetailed logs: {deployer.deployment_log_path}")
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Deployment failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
