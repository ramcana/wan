#!/usr/bin/env python3
"""
Deployment validation script to verify all components are working correctly
after real AI model integration deployment.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.system_integration import SystemIntegration
from core.configuration_bridge import ConfigurationBridge
from core.model_integration_bridge import ModelIntegrationBridge
from services.generation_service import GenerationService
from database.database import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Validates deployment of real AI model integration."""
    
    def __init__(self):
        self.system_integration = None
        self.config_bridge = None
        self.generation_service = None
        self.validation_results = {}
        
    async def initialize(self) -> bool:
        """Initialize validation components."""
        try:
            self.system_integration = SystemIntegration()
            await self.system_integration.initialize()
            
            self.config_bridge = ConfigurationBridge()
            await self.config_bridge.initialize()
            
            self.generation_service = GenerationService()
            await self.generation_service.initialize()
            
            logger.info("Validation components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize validation components: {e}")
            return False
    
    async def validate_configuration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate system configuration."""
        logger.info("Validating system configuration...")
        results = {}
        
        try:
            # Check configuration file exists
            config_path = Path("config.json")
            results["config_file_exists"] = config_path.exists()
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Validate required configuration sections
                required_sections = ["generation", "models", "hardware", "websocket"]
                for section in required_sections:
                    results[f"config_section_{section}"] = section in config
                
                # Validate generation mode
                generation_config = config.get("generation", {})
                results["real_generation_enabled"] = generation_config.get("mode") == "real"
                results["auto_download_enabled"] = generation_config.get("auto_download_models", False)
                
                # Validate model settings
                model_config = config.get("models", {})
                results["auto_optimize_enabled"] = model_config.get("auto_optimize", False)
                results["vram_management_enabled"] = model_config.get("vram_management", False)
            
            success = all(results.values())
            return success, results
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    async def validate_database(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate database connectivity and schema."""
        logger.info("Validating database...")
        results = {}
        
        try:
            db = await get_database()
            
            # Test basic connectivity
            test_result = await db.fetch_one("SELECT 1 as test")
            results["connectivity"] = test_result is not None
            
            # Check required tables exist
            tables_query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('generation_tasks', 'users');
            """
            tables = await db.fetch_all(tables_query)
            table_names = [row["name"] for row in tables]
            
            results["generation_tasks_table"] = "generation_tasks" in table_names
            results["users_table"] = "users" in table_names
            
            # Check for new columns in generation_tasks
            if "generation_tasks" in table_names:
                columns_query = "PRAGMA table_info(generation_tasks);"
                columns = await db.fetch_all(columns_query)
                column_names = [col["name"] for col in columns]
                
                new_columns = [
                    "model_used", "generation_time_seconds", "peak_vram_usage_mb",
                    "optimizations_applied", "error_category", "recovery_suggestions"
                ]
                
                for col in new_columns:
                    results[f"column_{col}"] = col in column_names
            
            success = all(results.values())
            return success, results
            
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    async def validate_system_integration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate system integration components."""
        logger.info("Validating system integration...")
        results = {}
        
        try:
            # Test system status
            status = await self.system_integration.get_system_status()
            results["system_initialized"] = status.get("initialized", False)
            results["wan22_infrastructure"] = status.get("wan22_infrastructure_loaded", False)
            
            # Test model bridge
            model_bridge = await self.system_integration.get_model_bridge()
            results["model_bridge_available"] = model_bridge is not None
            
            if model_bridge:
                model_status = model_bridge.get_system_model_status()
                results["model_status_accessible"] = isinstance(model_status, dict)
            
            # Test optimizer
            optimizer = await self.system_integration.get_system_optimizer()
            results["optimizer_available"] = optimizer is not None
            
            success = all(results.values())
            return success, results
            
        except Exception as e:
            logger.error(f"System integration validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    async def validate_model_management(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate model management functionality."""
        logger.info("Validating model management...")
        results = {}
        
        try:
            model_bridge = await self.system_integration.get_model_bridge()
            
            if model_bridge:
                # Test model status checking
                model_status = model_bridge.get_system_model_status()
                results["model_status_check"] = isinstance(model_status, dict)
                
                # Test model availability checking for each type
                model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
                for model_type in model_types:
                    try:
                        # Just check if we can query availability (don't actually download)
                        status = model_bridge.check_model_availability(model_type)
                        results[f"can_check_{model_type}"] = True
                    except Exception as e:
                        logger.warning(f"Cannot check {model_type}: {e}")
                        results[f"can_check_{model_type}"] = False
                
                # Test hardware optimization
                try:
                    hardware_profile = model_bridge.get_hardware_profile()
                    results["hardware_profile_available"] = hardware_profile is not None
                except Exception:
                    results["hardware_profile_available"] = False
            else:
                results["model_bridge_missing"] = True
            
            success = all(v for k, v in results.items() if not k.endswith("_missing"))
            return success, results
            
        except Exception as e:
            logger.error(f"Model management validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    async def validate_generation_service(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate generation service functionality."""
        logger.info("Validating generation service...")
        results = {}
        
        try:
            # Test service initialization
            results["service_initialized"] = self.generation_service is not None
            
            if self.generation_service:
                # Test queue functionality
                queue_status = await self.generation_service.get_queue_status()
                results["queue_accessible"] = isinstance(queue_status, dict)
                
                # Test real generation mode
                is_real_mode = getattr(self.generation_service, 'use_real_generation', False)
                results["real_generation_mode"] = is_real_mode
                
                # Test error handler
                error_handler = getattr(self.generation_service, 'error_handler', None)
                results["error_handler_available"] = error_handler is not None
            
            success = all(results.values())
            return success, results
            
        except Exception as e:
            logger.error(f"Generation service validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    async def validate_api_endpoints(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate API endpoints are accessible."""
        logger.info("Validating API endpoints...")
        results = {}
        
        try:
            # This is a basic check - in a real deployment you'd use HTTP client
            # For now, we'll check if the modules can be imported
            
            # Check main app
            try:
                from app import app
                results["main_app_importable"] = True
            except Exception:
                results["main_app_importable"] = False
            
            # Check API modules
            api_modules = [
                "api.generation",
                "api.model_management", 
                "api.fallback_recovery"
            ]
            
            for module in api_modules:
                try:
                    __import__(module)
                    results[f"{module}_importable"] = True
                except Exception:
                    results[f"{module}_importable"] = False
            
            # Check WebSocket manager
            try:
                from websocket.manager import ConnectionManager
                results["websocket_manager_importable"] = True
            except Exception:
                results["websocket_manager_importable"] = False
            
            success = all(results.values())
            return success, results
            
        except Exception as e:
            logger.error(f"API endpoints validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    async def validate_performance_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate performance requirements are met."""
        logger.info("Validating performance requirements...")
        results = {}
        
        try:
            # Test system resource availability
            import psutil
            
            # Check available RAM (should have at least 8GB for AI models)
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            results["sufficient_ram"] = available_ram_gb >= 8.0
            results["available_ram_gb"] = round(available_ram_gb, 2)
            
            # Check available disk space (should have at least 50GB for models)
            disk_usage = psutil.disk_usage('.')
            available_disk_gb = disk_usage.free / (1024**3)
            results["sufficient_disk"] = available_disk_gb >= 50.0
            results["available_disk_gb"] = round(available_disk_gb, 2)
            
            # Check CPU cores (should have at least 4 cores)
            cpu_count = psutil.cpu_count()
            results["sufficient_cpu"] = cpu_count >= 4
            results["cpu_count"] = cpu_count
            
            # Try to detect GPU (optional but recommended)
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                results["gpu_available"] = gpu_available
                if gpu_available:
                    results["gpu_count"] = torch.cuda.device_count()
                    results["gpu_memory_gb"] = round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                    )
            except Exception:
                results["gpu_available"] = False
            
            # Performance requirements are met if we have sufficient resources
            success = (results["sufficient_ram"] and 
                      results["sufficient_disk"] and 
                      results["sufficient_cpu"])
            
            return success, results
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            results["error"] = str(e)
            return False, results
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("=" * 60)
        report.append("DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        overall_success = True
        
        for category, (success, details) in self.validation_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            report.append(f"{category.upper()}: {status}")
            
            if not success:
                overall_success = False
            
            # Add details
            for key, value in details.items():
                if key != "error":
                    status_icon = "✅" if value else "❌"
                    report.append(f"  {status_icon} {key}: {value}")
                else:
                    report.append(f"  ❌ Error: {value}")
            
            report.append("")
        
        report.append("=" * 60)
        if overall_success:
            report.append("🎉 DEPLOYMENT VALIDATION SUCCESSFUL!")
            report.append("Your real AI model integration is ready for use.")
        else:
            report.append("⚠️ DEPLOYMENT VALIDATION FAILED!")
            report.append("Please resolve the issues above before proceeding.")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    async def run_full_validation(self) -> bool:
        """Run complete deployment validation."""
        logger.info("Starting deployment validation...")
        
        # Initialize components
        if not await self.initialize():
            logger.error("Failed to initialize validation components")
            return False
        
        # Run all validation checks
        validation_checks = [
            ("configuration", self.validate_configuration),
            ("database", self.validate_database),
            ("system_integration", self.validate_system_integration),
            ("model_management", self.validate_model_management),
            ("generation_service", self.validate_generation_service),
            ("api_endpoints", self.validate_api_endpoints),
            ("performance", self.validate_performance_requirements)
        ]
        
        for check_name, check_func in validation_checks:
            try:
                success, results = await check_func()
                self.validation_results[check_name] = (success, results)
            except Exception as e:
                logger.error(f"Validation check {check_name} failed: {e}")
                self.validation_results[check_name] = (False, {"error": str(e)})
        
        # Generate and display report
        report = self.generate_validation_report()
        print(report)
        
        # Return overall success
        return all(success for success, _ in self.validation_results.values())

async def main():
    """Main validation function."""
    validator = DeploymentValidator()
    
    try:
        success = await validator.run_full_validation()
        
        if success:
            print("\n🎉 All validation checks passed!")
            print("Your deployment is ready for production use.")
            sys.exit(0)
        else:
            print("\n⚠️ Some validation checks failed.")
            print("Please resolve the issues before deploying to production.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())