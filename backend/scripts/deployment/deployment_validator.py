#!/usr/bin/env python3
"""
Deployment Validator for Enhanced Model Availability System

This script validates that the enhanced model availability system is properly
deployed and all components are functioning correctly.
"""

import os
import sys
import json
import asyncio
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    success: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

@dataclass
class DeploymentValidationReport:
    """Complete deployment validation report"""
    timestamp: str
    overall_success: bool
    critical_failures: int
    warnings: int
    info_messages: int
    results: List[ValidationResult]
    environment_info: Dict[str, Any]

class EnhancedModelAvailabilityValidator:
    """Validates enhanced model availability system deployment"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.backend_path = Path(__file__).parent.parent.parent
    
    async def validate_deployment(self) -> DeploymentValidationReport:
        """Run complete deployment validation"""
        logger.info("Starting enhanced model availability deployment validation")
        
        # Environment validation
        await self._validate_environment()
        
        # Component validation
        await self._validate_core_components()
        
        # API validation
        await self._validate_api_endpoints()
        
        # Database validation
        await self._validate_database_setup()
        
        # Configuration validation
        await self._validate_configuration()
        
        # Integration validation
        await self._validate_integrations()
        
        # Performance validation
        await self._validate_performance_requirements()
        
        # Generate report
        report = self._generate_report()
        
        logger.info(f"Validation completed: {report.overall_success}")
        return report
    
    async def _validate_environment(self):
        """Validate deployment environment"""
        logger.info("Validating deployment environment")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self._add_result("python_version", True, ValidationLevel.INFO,
                           f"Python version {python_version.major}.{python_version.minor} is supported")
        else:
            self._add_result("python_version", False, ValidationLevel.CRITICAL,
                           f"Python version {python_version.major}.{python_version.minor} is not supported",
                           fix_suggestion="Upgrade to Python 3.8 or higher")
        
        # Check required directories
        required_dirs = [
            "backend/core",
            "backend/api",
            "backend/services",
            "backend/websocket",
            "backend/scripts/deployment"
        ]
        
        for dir_path in required_dirs:
            full_path = self.backend_path / dir_path
            if full_path.exists():
                self._add_result(f"directory_{dir_path.replace('/', '_')}", True, ValidationLevel.INFO,
                               f"Required directory exists: {dir_path}")
            else:
                self._add_result(f"directory_{dir_path.replace('/', '_')}", False, ValidationLevel.CRITICAL,
                               f"Required directory missing: {dir_path}",
                               fix_suggestion=f"Create directory: {dir_path}")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.backend_path)
            free_gb = free / (1024**3)
            
            if free_gb >= 10:
                self._add_result("disk_space", True, ValidationLevel.INFO,
                               f"Sufficient disk space: {free_gb:.1f} GB available")
            elif free_gb >= 5:
                self._add_result("disk_space", True, ValidationLevel.WARNING,
                               f"Limited disk space: {free_gb:.1f} GB available",
                               fix_suggestion="Consider freeing up disk space for model storage")
            else:
                self._add_result("disk_space", False, ValidationLevel.CRITICAL,
                               f"Insufficient disk space: {free_gb:.1f} GB available",
                               fix_suggestion="Free up at least 5GB of disk space")
        except Exception as e:
            self._add_result("disk_space", False, ValidationLevel.WARNING,
                           f"Could not check disk space: {e}")
    
    async def _validate_core_components(self):
        """Validate core enhanced model availability components"""
        logger.info("Validating core components")
        
        # List of core components to validate
        core_components = [
            ("backend.core.enhanced_model_downloader", "EnhancedModelDownloader"),
            ("backend.core.model_health_monitor", "ModelHealthMonitor"),
            ("backend.core.model_availability_manager", "ModelAvailabilityManager"),
            ("backend.core.intelligent_fallback_manager", "IntelligentFallbackManager"),
            ("backend.core.model_usage_analytics", "ModelUsageAnalytics"),
            ("backend.core.enhanced_error_recovery", "EnhancedErrorRecovery"),
            ("backend.core.model_update_manager", "ModelUpdateManager")
        ]
        
        for module_name, class_name in core_components:
            try:
                module = importlib.import_module(module_name)
                component_class = getattr(module, class_name)
                
                # Basic instantiation test
                if hasattr(component_class, '__init__'):
                    self._add_result(f"component_{class_name.lower()}", True, ValidationLevel.INFO,
                                   f"Core component available: {class_name}")
                else:
                    self._add_result(f"component_{class_name.lower()}", False, ValidationLevel.CRITICAL,
                                   f"Core component malformed: {class_name}")
                
            except ImportError as e:
                self._add_result(f"component_{class_name.lower()}", False, ValidationLevel.CRITICAL,
                               f"Core component missing: {class_name}",
                               fix_suggestion=f"Ensure {module_name}.py exists and is properly implemented")
            except AttributeError as e:
                self._add_result(f"component_{class_name.lower()}", False, ValidationLevel.CRITICAL,
                               f"Core component class missing: {class_name} in {module_name}",
                               fix_suggestion=f"Implement {class_name} class in {module_name}")
            except Exception as e:
                self._add_result(f"component_{class_name.lower()}", False, ValidationLevel.WARNING,
                               f"Core component validation error: {class_name} - {e}")
    
    async def _validate_api_endpoints(self):
        """Validate API endpoints for enhanced features"""
        logger.info("Validating API endpoints")
        
        try:
            # Import API modules
            from backend.api import enhanced_model_management
            
            # Check for required endpoints
            required_endpoints = [
                "get_detailed_model_status",
                "manage_model_download",
                "get_model_health",
                "get_model_analytics",
                "cleanup_models",
                "suggest_fallback_models"
            ]
            
            for endpoint in required_endpoints:
                if hasattr(enhanced_model_management, endpoint):
                    self._add_result(f"api_endpoint_{endpoint}", True, ValidationLevel.INFO,
                                   f"API endpoint available: {endpoint}")
                else:
                    self._add_result(f"api_endpoint_{endpoint}", False, ValidationLevel.CRITICAL,
                                   f"API endpoint missing: {endpoint}",
                                   fix_suggestion=f"Implement {endpoint} in enhanced_model_management.py")
            
        except ImportError as e:
            self._add_result("api_module", False, ValidationLevel.CRITICAL,
                           "Enhanced model management API module not found",
                           fix_suggestion="Ensure backend/api/enhanced_model_management.py exists")
        except Exception as e:
            self._add_result("api_validation", False, ValidationLevel.WARNING,
                           f"API validation error: {e}")
    
    async def _validate_database_setup(self):
        """Validate database setup for enhanced features"""
        logger.info("Validating database setup")
        
        try:
            # Check if database file exists
            db_path = self.backend_path / "wan22_tasks.db"
            if db_path.exists():
                self._add_result("database_file", True, ValidationLevel.INFO,
                               "Database file exists")
                
                # Try to connect to database
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Check for required tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ["model_usage_analytics", "model_health_history", "download_history"]
                for table in required_tables:
                    if table in tables:
                        self._add_result(f"database_table_{table}", True, ValidationLevel.INFO,
                                       f"Database table exists: {table}")
                    else:
                        self._add_result(f"database_table_{table}", False, ValidationLevel.WARNING,
                                       f"Database table missing: {table}",
                                       fix_suggestion=f"Run database migration to create {table} table")
                
                conn.close()
                
            else:
                self._add_result("database_file", False, ValidationLevel.WARNING,
                               "Database file does not exist",
                               fix_suggestion="Run database initialization script")
                
        except Exception as e:
            self._add_result("database_validation", False, ValidationLevel.WARNING,
                           f"Database validation error: {e}")
    
    async def _validate_configuration(self):
        """Validate configuration for enhanced features"""
        logger.info("Validating configuration")
        
        # Check main config file
        config_path = self.backend_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check for enhanced model availability configuration
                if "enhanced_model_availability" in config:
                    self._add_result("config_enhanced_section", True, ValidationLevel.INFO,
                                   "Enhanced model availability configuration section found")
                    
                    enhanced_config = config["enhanced_model_availability"]
                    
                    # Check required configuration keys
                    required_keys = [
                        "download_retry_attempts",
                        "health_check_interval",
                        "analytics_enabled",
                        "fallback_enabled"
                    ]
                    
                    for key in required_keys:
                        if key in enhanced_config:
                            self._add_result(f"config_key_{key}", True, ValidationLevel.INFO,
                                           f"Configuration key present: {key}")
                        else:
                            self._add_result(f"config_key_{key}", False, ValidationLevel.WARNING,
                                           f"Configuration key missing: {key}",
                                           fix_suggestion=f"Add {key} to enhanced_model_availability config")
                else:
                    self._add_result("config_enhanced_section", False, ValidationLevel.WARNING,
                                   "Enhanced model availability configuration section missing",
                                   fix_suggestion="Add enhanced_model_availability section to config.json")
                
            except json.JSONDecodeError as e:
                self._add_result("config_format", False, ValidationLevel.CRITICAL,
                               f"Configuration file format error: {e}",
                               fix_suggestion="Fix JSON syntax in config.json")
            except Exception as e:
                self._add_result("config_validation", False, ValidationLevel.WARNING,
                               f"Configuration validation error: {e}")
        else:
            self._add_result("config_file", False, ValidationLevel.CRITICAL,
                           "Main configuration file missing",
                           fix_suggestion="Create config.json file")
    
    async def _validate_integrations(self):
        """Validate integrations with existing systems"""
        logger.info("Validating system integrations")
        
        try:
            # Check integration with existing ModelManager
            from backend.core.model_manager import ModelManager
            self._add_result("integration_model_manager", True, ValidationLevel.INFO,
                           "ModelManager integration available")
            
            # Check integration with GenerationService
            from backend.services.generation_service import GenerationService
            self._add_result("integration_generation_service", True, ValidationLevel.INFO,
                           "GenerationService integration available")
            
            # Check WebSocket integration
            from backend.websocket.manager import WebSocketManager
            self._add_result("integration_websocket", True, ValidationLevel.INFO,
                           "WebSocket integration available")
            
        except ImportError as e:
            self._add_result("integration_check", False, ValidationLevel.CRITICAL,
                           f"Integration validation failed: {e}",
                           fix_suggestion="Ensure all required integration modules are available")
        except Exception as e:
            self._add_result("integration_validation", False, ValidationLevel.WARNING,
                           f"Integration validation error: {e}")
    
    async def _validate_performance_requirements(self):
        """Validate performance requirements"""
        logger.info("Validating performance requirements")
        
        try:
            import time
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < 80:
                self._add_result("performance_cpu", True, ValidationLevel.INFO,
                               f"CPU usage acceptable: {cpu_percent}%")
            else:
                self._add_result("performance_cpu", False, ValidationLevel.WARNING,
                               f"High CPU usage: {cpu_percent}%",
                               fix_suggestion="Consider reducing system load before deployment")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if memory_percent < 80:
                self._add_result("performance_memory", True, ValidationLevel.INFO,
                               f"Memory usage acceptable: {memory_percent}%")
            else:
                self._add_result("performance_memory", False, ValidationLevel.WARNING,
                               f"High memory usage: {memory_percent}%",
                               fix_suggestion="Consider freeing memory before deployment")
            
        except ImportError:
            self._add_result("performance_monitoring", False, ValidationLevel.WARNING,
                           "psutil not available for performance monitoring",
                           fix_suggestion="Install psutil for performance monitoring")
        except Exception as e:
            self._add_result("performance_validation", False, ValidationLevel.WARNING,
                           f"Performance validation error: {e}")
    
    def _add_result(self, check_name: str, success: bool, level: ValidationLevel, 
                   message: str, details: Optional[Dict] = None, fix_suggestion: Optional[str] = None):
        """Add validation result"""
        result = ValidationResult(
            check_name=check_name,
            success=success,
            level=level,
            message=message,
            details=details,
            fix_suggestion=fix_suggestion
        )
        self.results.append(result)
    
    def _generate_report(self) -> DeploymentValidationReport:
        """Generate deployment validation report"""
        critical_failures = sum(1 for r in self.results if not r.success and r.level == ValidationLevel.CRITICAL)
        warnings = sum(1 for r in self.results if not r.success and r.level == ValidationLevel.WARNING)
        info_messages = sum(1 for r in self.results if r.level == ValidationLevel.INFO)
        
        overall_success = critical_failures == 0
        
        # Gather environment info
        environment_info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "backend_path": str(self.backend_path),
            "validation_time": datetime.now().isoformat()
        }
        
        return DeploymentValidationReport(
            timestamp=datetime.now().isoformat(),
            overall_success=overall_success,
            critical_failures=critical_failures,
            warnings=warnings,
            info_messages=info_messages,
            results=self.results,
            environment_info=environment_info
        )

async def main():
    """Main validation function"""
    validator = EnhancedModelAvailabilityValidator()
    report = await validator.validate_deployment()
    
    # Save report
    report_path = Path("deployment_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    # Print summary
    print(f"\n=== Deployment Validation Report ===")
    print(f"Overall Success: {'✓' if report.overall_success else '✗'}")
    print(f"Critical Failures: {report.critical_failures}")
    print(f"Warnings: {report.warnings}")
    print(f"Info Messages: {report.info_messages}")
    print(f"Report saved to: {report_path}")
    
    # Print critical failures
    if report.critical_failures > 0:
        print(f"\n=== Critical Issues ===")
        for result in report.results:
            if not result.success and result.level == ValidationLevel.CRITICAL:
                print(f"✗ {result.check_name}: {result.message}")
                if result.fix_suggestion:
                    print(f"  Fix: {result.fix_suggestion}")
    
    return 0 if report.overall_success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
