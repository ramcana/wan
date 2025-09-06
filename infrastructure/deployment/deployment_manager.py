"""
WAN Model Deployment Manager

Orchestrates the deployment process from placeholder to real WAN models
with comprehensive validation, rollback, and monitoring capabilities.
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .migration_service import MigrationService
from .validation_service import ValidationService
from .rollback_service import RollbackService
from .monitoring_service import MonitoringService


class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Configuration for WAN model deployment"""
    source_models_path: str
    target_models_path: str
    backup_path: str
    validation_enabled: bool = True
    rollback_enabled: bool = True
    monitoring_enabled: bool = True
    health_check_interval: int = 300  # 5 minutes
    max_deployment_time: int = 3600   # 1 hour
    
    
@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    models_deployed: List[str]
    validation_results: Dict[str, Any]
    rollback_available: bool
    error_message: Optional[str] = None


class DeploymentManager:
    """
    Main orchestrator for WAN model deployments
    
    Handles the complete deployment lifecycle:
    - Pre-deployment validation
    - Model migration
    - Post-deployment validation
    - Rollback capabilities
    - Health monitoring
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.migration_service = MigrationService(config)
        self.validation_service = ValidationService(config)
        self.rollback_service = RollbackService(config)
        self.monitoring_service = MonitoringService(config)
        
        # Deployment state
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for path in [self.config.target_models_path, self.config.backup_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    async def deploy_models(
        self, 
        models: List[str], 
        deployment_id: Optional[str] = None
    ) -> DeploymentResult:
        """
        Deploy WAN models from placeholder to production
        
        Args:
            models: List of model names to deploy
            deployment_id: Optional custom deployment ID
            
        Returns:
            DeploymentResult with deployment status and details
        """
        if not deployment_id:
            deployment_id = f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting deployment {deployment_id} for models: {models}")
        
        # Initialize deployment result
        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            models_deployed=[],
            validation_results={},
            rollback_available=False
        )
        
        self.active_deployments[deployment_id] = result
        
        try:
            # Phase 1: Pre-deployment validation
            result.status = DeploymentStatus.IN_PROGRESS
            self.logger.info(f"Phase 1: Pre-deployment validation for {deployment_id}")
            
            pre_validation = await self.validation_service.validate_pre_deployment(models)
            if not pre_validation.is_valid:
                raise Exception(f"Pre-deployment validation failed: {pre_validation.errors}")
            
            result.validation_results['pre_deployment'] = asdict(pre_validation)
            
            # Phase 2: Create backup
            self.logger.info(f"Phase 2: Creating backup for {deployment_id}")
            backup_result = await self.rollback_service.create_backup(deployment_id)
            result.rollback_available = backup_result.success
            
            # Phase 3: Migrate models
            self.logger.info(f"Phase 3: Migrating models for {deployment_id}")
            migration_results = []
            
            for model in models:
                migration_result = await self.migration_service.migrate_model(
                    model, deployment_id
                )
                migration_results.append(migration_result)
                
                if migration_result.success:
                    result.models_deployed.append(model)
                else:
                    raise Exception(f"Migration failed for model {model}: {migration_result.error}")
            
            # Phase 4: Post-deployment validation
            result.status = DeploymentStatus.VALIDATING
            self.logger.info(f"Phase 4: Post-deployment validation for {deployment_id}")
            
            post_validation = await self.validation_service.validate_post_deployment(
                result.models_deployed
            )
            
            if not post_validation.is_valid:
                # Attempt rollback
                self.logger.warning(f"Post-deployment validation failed, attempting rollback")
                await self._perform_rollback(deployment_id, "Post-deployment validation failed")
                result.status = DeploymentStatus.ROLLED_BACK
                result.error_message = f"Validation failed: {post_validation.errors}"
            else:
                result.validation_results['post_deployment'] = asdict(post_validation)
                result.status = DeploymentStatus.COMPLETED
                
                # Start monitoring if enabled
                if self.config.monitoring_enabled:
                    await self.monitoring_service.start_monitoring(
                        deployment_id, result.models_deployed
                    )
            
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {str(e)}")
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            
            # Attempt rollback if backup is available
            if result.rollback_available:
                await self._perform_rollback(deployment_id, str(e))
                result.status = DeploymentStatus.ROLLED_BACK
        
        finally:
            result.end_time = datetime.now()
            self.deployment_history.append(result)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
        
        self.logger.info(f"Deployment {deployment_id} completed with status: {result.status}")
        return result
    
    async def _perform_rollback(self, deployment_id: str, reason: str):
        """Perform rollback for a failed deployment"""
        try:
            rollback_result = await self.rollback_service.rollback_deployment(
                deployment_id, reason
            )
            if rollback_result.success:
                self.logger.info(f"Successfully rolled back deployment {deployment_id}")
            else:
                self.logger.error(f"Rollback failed for {deployment_id}: {rollback_result.error}")
        except Exception as e:
            self.logger.error(f"Exception during rollback of {deployment_id}: {str(e)}")
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment"""
        # Check active deployments first
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
        
        # Check deployment history
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        
        return None
    
    async def list_deployments(
        self, 
        status_filter: Optional[DeploymentStatus] = None,
        limit: int = 50
    ) -> List[DeploymentResult]:
        """List deployments with optional status filtering"""
        deployments = list(self.active_deployments.values()) + self.deployment_history
        
        if status_filter:
            deployments = [d for d in deployments if d.status == status_filter]
        
        # Sort by start time (most recent first)
        deployments.sort(key=lambda x: x.start_time, reverse=True)
        
        return deployments[:limit]
    
    async def rollback_deployment(self, deployment_id: str, reason: str = "Manual rollback"):
        """Manually rollback a deployment"""
        deployment = await self.get_deployment_status(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if not deployment.rollback_available:
            raise ValueError(f"Rollback not available for deployment {deployment_id}")
        
        return await self.rollback_service.rollback_deployment(deployment_id, reason)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of deployed models"""
        return await self.monitoring_service.get_health_status()
    
    async def cleanup_old_deployments(self, days_to_keep: int = 30):
        """Clean up old deployment artifacts and logs"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        # Clean up deployment history
        self.deployment_history = [
            d for d in self.deployment_history 
            if d.start_time.timestamp() > cutoff_date
        ]
        
        # Clean up backup files
        await self.rollback_service.cleanup_old_backups(days_to_keep)
        
        self.logger.info(f"Cleaned up deployments older than {days_to_keep} days")