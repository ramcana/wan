"""
Deployment Health Check API Endpoints

This module provides health check endpoints for validating the deployment
of the enhanced model availability system.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from enum import Enum

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.enhanced_model_downloader import EnhancedModelDownloader
    from core.model_health_monitor import ModelHealthMonitor
    from core.model_availability_manager import ModelAvailabilityManager
    from core.intelligent_fallback_manager import IntelligentFallbackManager
    from core.model_usage_analytics import ModelUsageAnalytics
    from core.enhanced_error_recovery import EnhancedErrorRecovery
    from core.model_update_manager import ModelUpdateManager
except ImportError as e:
    logging.warning(f"Could not import enhanced model availability components: {e}")

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/deployment", tags=["deployment-health"])

class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentStatus(BaseModel):
    """Status of a system component"""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    last_check: str
    response_time_ms: Optional[float] = None

class SystemHealthResponse(BaseModel):
    """System health check response"""
    overall_status: HealthStatus
    timestamp: str
    components: List[ComponentStatus]
    summary: Dict[str, int]
    uptime_seconds: Optional[float] = None
    version: Optional[str] = None

class DeploymentValidationResponse(BaseModel):
    """Deployment validation response"""
    deployment_valid: bool
    timestamp: str
    validation_results: List[Dict[str, Any]]
    critical_issues: int
    warnings: int
    recommendations: List[str]

class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    timestamp: str
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    alerts: List[str]

# Global variables for tracking
_system_start_time = datetime.now()
_health_cache = {}
_cache_ttl = 30  # seconds

async def get_cached_health_check(component_name: str, check_function):
    """Get cached health check result or perform new check"""
    cache_key = f"health_{component_name}"
    now = datetime.now()
    
    # Check cache
    if cache_key in _health_cache:
        cached_result, cached_time = _health_cache[cache_key]
        if (now - cached_time).total_seconds() < _cache_ttl:
            return cached_result
    
    # Perform new check
    start_time = datetime.now()
    try:
        result = await check_function()
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        status = ComponentStatus(
            name=component_name,
            status=HealthStatus.HEALTHY,
            message="Component is functioning normally",
            details=result if isinstance(result, dict) else None,
            last_check=now.isoformat(),
            response_time_ms=response_time
        )
    except Exception as e:
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        status = ComponentStatus(
            name=component_name,
            status=HealthStatus.UNHEALTHY,
            message=f"Component check failed: {str(e)}",
            last_check=now.isoformat(),
            response_time_ms=response_time
        )
    
    # Cache result
    _health_cache[cache_key] = (status, now)
    return status

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """
    Get overall system health status
    
    Returns comprehensive health information for all enhanced model availability components.
    """
    try:
        components = []
        
        # Check core components
        components.append(await get_cached_health_check("database", check_database_health))
        components.append(await get_cached_health_check("file_system", check_file_system_health))
        components.append(await get_cached_health_check("enhanced_downloader", check_enhanced_downloader_health))
        components.append(await get_cached_health_check("health_monitor", check_health_monitor_health))
        components.append(await get_cached_health_check("availability_manager", check_availability_manager_health))
        components.append(await get_cached_health_check("fallback_manager", check_fallback_manager_health))
        components.append(await get_cached_health_check("usage_analytics", check_usage_analytics_health))
        components.append(await get_cached_health_check("error_recovery", check_error_recovery_health))
        components.append(await get_cached_health_check("update_manager", check_update_manager_health))
        
        # Calculate overall status
        healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate uptime
        uptime = (datetime.now() - _system_start_time).total_seconds()
        
        return SystemHealthResponse(
            overall_status=overall_status,
            timestamp=datetime.now().isoformat(),
            components=components,
            summary={
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "total": len(components)
            },
            uptime_seconds=uptime,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/health/{component}", response_model=ComponentStatus)
async def get_component_health(component: str):
    """
    Get health status for a specific component
    
    Args:
        component: Name of the component to check
    """
    component_checks = {
        "database": check_database_health,
        "file_system": check_file_system_health,
        "enhanced_downloader": check_enhanced_downloader_health,
        "health_monitor": check_health_monitor_health,
        "availability_manager": check_availability_manager_health,
        "fallback_manager": check_fallback_manager_health,
        "usage_analytics": check_usage_analytics_health,
        "error_recovery": check_error_recovery_health,
        "update_manager": check_update_manager_health
    }
    
    if component not in component_checks:
        raise HTTPException(status_code=404, detail=f"Component not found: {component}")
    
    try:
        status = await get_cached_health_check(component, component_checks[component])
        return status
    except Exception as e:
        logger.error(f"Component health check failed for {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Component health check failed: {str(e)}")

@router.get("/validate", response_model=DeploymentValidationResponse)
async def validate_deployment():
    """
    Validate deployment of enhanced model availability system
    
    Performs comprehensive validation of the deployment including:
    - Component availability
    - Configuration validation
    - Database schema validation
    - File system permissions
    - Integration tests
    """
    try:
        validation_results = []
        critical_issues = 0
        warnings = 0
        recommendations = []
        
        # Import deployment validator
        try:
            sys.path.append(str(Path(__file__).parent.parent / "scripts" / "deployment"))
            from deployment_validator import EnhancedModelAvailabilityValidator
            
            validator = EnhancedModelAvailabilityValidator()
            report = await validator.validate_deployment()
            
            # Convert validation results
            for result in report.results:
                validation_results.append({
                    "check_name": result.check_name,
                    "success": result.success,
                    "level": result.level.value,
                    "message": result.message,
                    "fix_suggestion": result.fix_suggestion
                })
                
                if not result.success:
                    if result.level.value == "critical":
                        critical_issues += 1
                    elif result.level.value == "warning":
                        warnings += 1
                
                if result.fix_suggestion:
                    recommendations.append(result.fix_suggestion)
            
            deployment_valid = report.overall_success
            
        except ImportError:
            # Fallback validation
            validation_results.append({
                "check_name": "deployment_validator_import",
                "success": False,
                "level": "warning",
                "message": "Deployment validator not available, performing basic validation",
                "fix_suggestion": "Ensure deployment validator is properly installed"
            })
            
            # Basic validation checks
            basic_checks = await perform_basic_validation()
            validation_results.extend(basic_checks["results"])
            critical_issues += basic_checks["critical_issues"]
            warnings += basic_checks["warnings"]
            recommendations.extend(basic_checks["recommendations"])
            
            deployment_valid = critical_issues == 0
        
        return DeploymentValidationResponse(
            deployment_valid=deployment_valid,
            timestamp=datetime.now().isoformat(),
            validation_results=validation_results,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=list(set(recommendations))  # Remove duplicates
        )
        
    except Exception as e:
        logger.error(f"Deployment validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment validation failed: {str(e)}")

@router.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """
    Get performance metrics for deployment monitoring
    
    Returns key performance indicators and alerts for the enhanced model availability system.
    """
    try:
        metrics = {}
        alerts = []
        
        # System metrics
        try:
            import psutil
            
            metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=1)
            metrics["memory_usage_percent"] = psutil.virtual_memory().percent
            
            disk_usage = psutil.disk_usage('.')
            metrics["disk_usage_percent"] = (disk_usage.used / disk_usage.total) * 100
            
        except ImportError:
            metrics["cpu_usage_percent"] = 0
            metrics["memory_usage_percent"] = 0
            metrics["disk_usage_percent"] = 0
            alerts.append("psutil not available for system metrics")
        
        # Model metrics
        models_dir = Path("models")
        if models_dir.exists():
            model_count = len([d for d in models_dir.iterdir() if d.is_dir()])
            metrics["total_models"] = model_count
            
            # Calculate total model size
            total_size = 0
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    for file in model_dir.rglob("*"):
                        if file.is_file():
                            total_size += file.stat().st_size
            
            metrics["total_model_size_gb"] = total_size / (1024**3)
        else:
            metrics["total_models"] = 0
            metrics["total_model_size_gb"] = 0
        
        # Performance thresholds
        thresholds = {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "total_models": 100,
            "total_model_size_gb": 500.0
        }
        
        # Check for alerts
        for metric, value in metrics.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                if metric.endswith("_percent") and value > threshold:
                    alerts.append(f"High {metric.replace('_', ' ')}: {value:.1f}% (threshold: {threshold}%)")
                elif metric == "total_model_size_gb" and value > threshold:
                    alerts.append(f"High model storage usage: {value:.1f} GB (threshold: {threshold} GB)")
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            thresholds=thresholds,
            alerts=alerts
        )
        
    except Exception as e:
        logger.error(f"Performance metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics collection failed: {str(e)}")

@router.get("/readiness")
async def readiness_check():
    """
    Kubernetes-style readiness check
    
    Returns 200 if the system is ready to serve requests, 503 otherwise.
    """
    try:
        # Check critical components
        critical_checks = [
            check_database_health,
            check_file_system_health
        ]
        
        for check in critical_checks:
            try:
                await check()
            except Exception as e:
                logger.error(f"Readiness check failed: {e}")
                raise HTTPException(status_code=503, detail="System not ready")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        raise HTTPException(status_code=503, detail="System not ready")

@router.get("/liveness")
async def liveness_check():
    """
    Kubernetes-style liveness check
    
    Returns 200 if the system is alive, 503 otherwise.
    """
    try:
        # Basic liveness check - just verify we can respond
        return {"status": "alive", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Liveness check error: {e}")
        raise HTTPException(status_code=503, detail="System not alive")

# Health check functions for individual components

async def check_database_health():
    """Check database health"""
    try:
        import sqlite3
        
        db_paths = ["backend/wan22_tasks.db", "wan22_tasks.db"]
        db_path = None
        
        for path in db_paths:
            if Path(path).exists():
                db_path = path
                break
        
        if not db_path:
            raise Exception("Database file not found")
        
        # Test database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        conn.close()
        
        if result[0] != 1:
            raise Exception("Database query failed")
        
        return {"database_path": db_path, "connection": "ok"}
        
    except Exception as e:
        raise Exception(f"Database health check failed: {str(e)}")

async def check_file_system_health():
    """Check file system health"""
    try:
        # Check required directories
        required_dirs = [
            "backend/core",
            "backend/api",
            "backend/services",
            "models"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            raise Exception(f"Missing directories: {', '.join(missing_dirs)}")
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free / (1024**3)
        
        if free_gb < 1:
            raise Exception(f"Low disk space: {free_gb:.1f} GB available")
        
        return {
            "required_directories": "ok",
            "free_space_gb": free_gb
        }
        
    except Exception as e:
        raise Exception(f"File system health check failed: {str(e)}")

async def check_enhanced_downloader_health():
    """Check enhanced model downloader health"""
    try:
        # Try to import and instantiate
        from core.enhanced_model_downloader import EnhancedModelDownloader
        
        # Basic instantiation test
        downloader = EnhancedModelDownloader(None)  # Would normally pass base downloader
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("EnhancedModelDownloader not available")
    except Exception as e:
        raise Exception(f"EnhancedModelDownloader health check failed: {str(e)}")

async def check_health_monitor_health():
    """Check model health monitor health"""
    try:
        from core.model_health_monitor import ModelHealthMonitor
        
        monitor = ModelHealthMonitor()
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("ModelHealthMonitor not available")
    except Exception as e:
        raise Exception(f"ModelHealthMonitor health check failed: {str(e)}")

async def check_availability_manager_health():
    """Check model availability manager health"""
    try:
        from core.model_availability_manager import ModelAvailabilityManager
        
        # Basic instantiation test
        manager = ModelAvailabilityManager(None, None)  # Would normally pass dependencies
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("ModelAvailabilityManager not available")
    except Exception as e:
        raise Exception(f"ModelAvailabilityManager health check failed: {str(e)}")

async def check_fallback_manager_health():
    """Check intelligent fallback manager health"""
    try:
        from core.intelligent_fallback_manager import IntelligentFallbackManager
        
        manager = IntelligentFallbackManager(None)  # Would normally pass availability manager
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("IntelligentFallbackManager not available")
    except Exception as e:
        raise Exception(f"IntelligentFallbackManager health check failed: {str(e)}")

async def check_usage_analytics_health():
    """Check model usage analytics health"""
    try:
        from core.model_usage_analytics import ModelUsageAnalytics
        
        analytics = ModelUsageAnalytics()
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("ModelUsageAnalytics not available")
    except Exception as e:
        raise Exception(f"ModelUsageAnalytics health check failed: {str(e)}")

async def check_error_recovery_health():
    """Check enhanced error recovery health"""
    try:
        from core.enhanced_error_recovery import EnhancedErrorRecovery
        
        recovery = EnhancedErrorRecovery(None, None)  # Would normally pass dependencies
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("EnhancedErrorRecovery not available")
    except Exception as e:
        raise Exception(f"EnhancedErrorRecovery health check failed: {str(e)}")

async def check_update_manager_health():
    """Check model update manager health"""
    try:
        from core.model_update_manager import ModelUpdateManager
        
        manager = ModelUpdateManager()
        
        return {"component": "available", "instantiation": "ok"}
        
    except ImportError:
        raise Exception("ModelUpdateManager not available")
    except Exception as e:
        raise Exception(f"ModelUpdateManager health check failed: {str(e)}")

async def perform_basic_validation():
    """Perform basic validation when full validator is not available"""
    results = []
    critical_issues = 0
    warnings = 0
    recommendations = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        results.append({
            "check_name": "python_version",
            "success": True,
            "level": "info",
            "message": f"Python version {python_version.major}.{python_version.minor} is supported",
            "fix_suggestion": None
        })
    else:
        results.append({
            "check_name": "python_version",
            "success": False,
            "level": "critical",
            "message": f"Python version {python_version.major}.{python_version.minor} is not supported",
            "fix_suggestion": "Upgrade to Python 3.8 or higher"
        })
        critical_issues += 1
        recommendations.append("Upgrade to Python 3.8 or higher")
    
    # Check required directories
    required_dirs = ["backend/core", "backend/api", "backend/services", "models"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            results.append({
                "check_name": f"directory_{dir_path.replace('/', '_')}",
                "success": True,
                "level": "info",
                "message": f"Required directory exists: {dir_path}",
                "fix_suggestion": None
            })
        else:
            results.append({
                "check_name": f"directory_{dir_path.replace('/', '_')}",
                "success": False,
                "level": "critical",
                "message": f"Required directory missing: {dir_path}",
                "fix_suggestion": f"Create directory: {dir_path}"
            })
            critical_issues += 1
            recommendations.append(f"Create directory: {dir_path}")
    
    return {
        "results": results,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "recommendations": recommendations
    }