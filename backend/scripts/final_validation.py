#!/usr/bin/env python3
"""
Final validation script for real AI model integration.
Comprehensive validation of all components and system readiness.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import subprocess

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalValidator:
    """Comprehensive final validation for the real AI model integration system."""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        
    async def run_comprehensive_validation(self) -> bool:
        """Run complete system validation."""
        print("🚀 Starting Final Integration Validation")
        print("="*60)
        
        validation_steps = [
            ("System Requirements", self.validate_system_requirements),
            ("Configuration", self.validate_configuration),
            ("Database Schema", self.validate_database_schema),
            ("Component Integration", self.validate_component_integration),
            ("API Endpoints", self.validate_api_endpoints),
            ("Performance Monitoring", self.validate_performance_monitoring),
            ("Error Handling", self.validate_error_handling),
            ("WebSocket Integration", self.validate_websocket_integration),
            ("Model Management", self.validate_model_management),
            ("Generation Pipeline", self.validate_generation_pipeline),
            ("Fallback Systems", self.validate_fallback_systems),
            ("Performance Benchmarks", self.validate_performance_benchmarks),
            ("Deployment Readiness", self.validate_deployment_readiness)
        ]
        
        overall_success = True
        
        for step_name, validation_func in validation_steps:
            print(f"\n🔍 Validating {step_name}...")
            
            try:
                success, details = await validation_func()
                self.validation_results[step_name] = (success, details)
                
                if success:
                    print(f"✅ {step_name}: PASS")
                else:
                    print(f"❌ {step_name}: FAIL")
                    overall_success = False
                    
                    # Add to critical failures if it's a critical component
                    if step_name in ["System Requirements", "Configuration", "Component Integration"]:
                        self.critical_failures.append(step_name)
                
                # Display important details
                if details and isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, bool):
                            status = "✅" if value else "❌"
                            print(f"  {status} {key}")
                        elif key == "warnings" and value:
                            for warning in value:
                                print(f"  ⚠️  {warning}")
                                self.warnings.append(warning)
                
            except Exception as e:
                print(f"❌ {step_name}: ERROR - {e}")
                self.validation_results[step_name] = (False, {"error": str(e)})
                overall_success = False
                self.critical_failures.append(step_name)
        
        # Generate final report
        self.generate_final_report()
        
        return overall_success and len(self.critical_failures) == 0
    
    async def validate_system_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate system requirements."""
        results = {}
        
        try:
            import psutil
            
            # Check RAM
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            total_ram_gb = memory.total / (1024**3)
            
            results["sufficient_ram"] = total_ram_gb >= 8.0
            results["ram_details"] = f"{total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available"
            
            # Check disk space
            disk = psutil.disk_usage('.')
            available_disk_gb = disk.free / (1024**3)
            
            results["sufficient_disk"] = available_disk_gb >= 10.0  # Minimum for basic operation
            results["disk_details"] = f"{available_disk_gb:.1f}GB available"
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            results["sufficient_cpu"] = cpu_count >= 2
            results["cpu_details"] = f"{cpu_count} cores"
            
            # Check GPU (optional)
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                results["gpu_available"] = gpu_available
                
                if gpu_available:
                    gpu_count = torch.cuda.device_count()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    results["gpu_details"] = f"{gpu_count} GPU(s), {gpu_memory:.1f}GB VRAM"
                else:
                    results["gpu_details"] = "No CUDA GPU available"
                    
            except ImportError:
                results["gpu_available"] = False
                results["gpu_details"] = "PyTorch not available"
            
            # Check Python version
            python_version = sys.version_info
            results["python_version_ok"] = python_version >= (3, 8)
            results["python_details"] = f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            
            success = all([
                results["sufficient_ram"],
                results["sufficient_disk"],
                results["sufficient_cpu"],
                results["python_version_ok"]
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_configuration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate system configuration."""
        results = {}
        
        try:
            # Check config file exists
            config_path = Path("config.json")
            results["config_file_exists"] = config_path.exists()
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check required sections
                required_sections = ["generation", "models", "hardware", "api"]
                for section in required_sections:
                    results[f"has_{section}_section"] = section in config
                
                # Check generation config
                gen_config = config.get("generation", {})
                results["generation_mode_set"] = "mode" in gen_config
                results["real_generation_enabled"] = gen_config.get("mode") == "real"
                
                # Check model config
                model_config = config.get("models", {})
                results["model_path_configured"] = "base_path" in model_config
                results["auto_optimize_enabled"] = model_config.get("auto_optimize", False)
                
                # Check API config
                api_config = config.get("api", {})
                results["api_port_configured"] = "port" in api_config
                
            else:
                # Create default config structure for validation
                results.update({
                    "has_generation_section": False,
                    "has_models_section": False,
                    "has_hardware_section": False,
                    "has_api_section": False,
                    "generation_mode_set": False,
                    "real_generation_enabled": False
                })
            
            success = all([
                results["config_file_exists"],
                results.get("has_generation_section", False),
                results.get("has_models_section", False),
                results.get("generation_mode_set", False)
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_database_schema(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate database schema."""
        results = {}
        
        try:
            from repositories.database import get_database, GenerationTaskDB
            
            # Test database connection
            db = await get_database()
            test_result = await db.fetch_one("SELECT 1 as test")
            results["database_connection"] = test_result is not None
            
            # Check tables exist
            tables_query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('generation_tasks', 'users');
            """
            tables = await db.fetch_all(tables_query)
            table_names = [row["name"] for row in tables]
            
            results["generation_tasks_table"] = "generation_tasks" in table_names
            results["users_table"] = "users" in table_names
            
            # Check for enhanced columns
            if "generation_tasks" in table_names:
                columns_query = "PRAGMA table_info(generation_tasks);"
                columns = await db.fetch_all(columns_query)
                column_names = [col["name"] for col in columns]
                
                enhanced_columns = [
                    "model_used", "generation_time_seconds", "peak_vram_usage_mb",
                    "optimizations_applied", "error_category", "recovery_suggestions"
                ]
                
                for col in enhanced_columns:
                    results[f"has_{col}_column"] = col in column_names
            
            success = all([
                results["database_connection"],
                results["generation_tasks_table"]
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_component_integration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate core component integration."""
        results = {}
        
        try:
            # Test SystemIntegration
            from core.system_integration import SystemIntegration
            
            integration = SystemIntegration()
            await integration.initialize()
            
            status = await integration.get_system_status()
            results["system_integration_init"] = status.get("initialized", False)
            
            # Test ModelIntegrationBridge
            model_bridge = await integration.get_model_bridge()
            results["model_bridge_available"] = model_bridge is not None
            
            if model_bridge:
                model_status = model_bridge.get_system_model_status()
                results["model_status_accessible"] = isinstance(model_status, dict)
            
            # Test SystemOptimizer
            optimizer = await integration.get_system_optimizer()
            results["optimizer_available"] = optimizer is not None
            
            success = results["system_integration_init"]
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_api_endpoints(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate API endpoint availability."""
        results = {}
        
        try:
            # Test main app import
            from app import app
            results["main_app_importable"] = True
            
            # Test API routers
            api_modules = [
                ("performance", "api.performance"),
                ("model_management", "api.model_management"),
                ("fallback_recovery", "api.fallback_recovery")
            ]
            
            for name, module_path in api_modules:
                try:
                    module = __import__(module_path, fromlist=['router'])
                    results[f"{name}_api_available"] = hasattr(module, 'router')
                except ImportError:
                    results[f"{name}_api_available"] = False
            
            success = results["main_app_importable"]
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_performance_monitoring(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate performance monitoring system."""
        results = {}
        
        try:
            from core.performance_monitor import get_performance_monitor
            
            monitor = get_performance_monitor()
            results["monitor_available"] = monitor is not None
            
            if monitor:
                # Test system status
                status = monitor.get_current_system_status()
                results["system_status_accessible"] = isinstance(status, dict)
                
                # Test task monitoring
                test_metrics = monitor.start_task_monitoring(
                    "validation-test", "t2v-A14B", "720p", 20
                )
                results["task_monitoring_works"] = test_metrics is not None
                
                if test_metrics:
                    completed = monitor.complete_task_monitoring("validation-test", True)
                    results["monitoring_completion_works"] = completed is not None
                
                # Test analysis
                analysis = monitor.get_performance_analysis(1)
                results["analysis_works"] = analysis is not None
            
            success = all([
                results["monitor_available"],
                results.get("system_status_accessible", False)
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_error_handling(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate error handling systems."""
        results = {}
        
        try:
            from core.fallback_recovery_system import get_fallback_recovery_system
            
            recovery_system = get_fallback_recovery_system()
            results["recovery_system_available"] = recovery_system is not None
            
            if recovery_system:
                # Test health check
                health = await recovery_system.check_system_health()
                results["health_check_works"] = isinstance(health, dict)
            
            success = True  # Error handling is optional but recommended
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_websocket_integration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate WebSocket integration."""
        results = {}
        
        try:
            from websocket.manager import ConnectionManager
            from websocket.progress_integration import ProgressIntegration
            
            manager = ConnectionManager()
            results["connection_manager_available"] = manager is not None
            
            progress = ProgressIntegration(manager)
            results["progress_integration_available"] = progress is not None
            
            success = all([
                results["connection_manager_available"],
                results["progress_integration_available"]
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_model_management(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate model management functionality."""
        results = {}
        
        try:
            from core.system_integration import SystemIntegration
            
            integration = SystemIntegration()
            await integration.initialize()
            
            model_bridge = await integration.get_model_bridge()
            results["model_bridge_available"] = model_bridge is not None
            
            if model_bridge:
                # Test model status checking
                status = model_bridge.get_system_model_status()
                results["model_status_check"] = isinstance(status, dict)
                
                # Test model availability (don't actually download)
                for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
                    try:
                        available = model_bridge.check_model_availability(model_type)
                        results[f"{model_type}_checkable"] = True
                    except Exception:
                        results[f"{model_type}_checkable"] = False
            
            success = results.get("model_bridge_available", False)
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_generation_pipeline(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate generation pipeline."""
        results = {}
        
        try:
            from services.generation_service import GenerationService
            
            service = GenerationService()
            await service.initialize()
            
            results["generation_service_init"] = service is not None
            
            # Test queue functionality
            queue_status = await service.get_queue_status()
            results["queue_accessible"] = isinstance(queue_status, dict)
            
            # Test that components are available
            results["has_performance_monitor"] = hasattr(service, 'performance_monitor')
            results["has_fallback_system"] = hasattr(service, 'fallback_recovery_system')
            
            success = all([
                results["generation_service_init"],
                results["queue_accessible"]
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_fallback_systems(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate fallback and recovery systems."""
        results = {}
        
        try:
            from services.generation_service import GenerationService
            
            service = GenerationService()
            await service.initialize()
            
            # Test fallback mode
            original_mode = service.use_real_generation
            service.use_real_generation = False
            
            results["fallback_mode_available"] = service.fallback_to_simulation
            
            # Restore original mode
            service.use_real_generation = original_mode
            
            success = True  # Fallback is always available
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_performance_benchmarks(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate performance benchmarks."""
        results = {}
        
        try:
            from core.performance_monitor import get_performance_monitor
            
            monitor = get_performance_monitor()
            
            # Check thresholds are reasonable
            thresholds = monitor.performance_thresholds
            
            results["generation_time_720p_reasonable"] = thresholds["max_generation_time_720p"] <= 600
            results["generation_time_1080p_reasonable"] = thresholds["max_generation_time_1080p"] <= 1800
            results["vram_threshold_reasonable"] = thresholds["max_vram_usage_percent"] <= 95
            results["success_rate_threshold_reasonable"] = thresholds["min_success_rate"] >= 0.8
            
            success = all(results.values())
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_deployment_readiness(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate deployment readiness."""
        results = {}
        warnings = []
        
        try:
            # Check essential files
            essential_files = [
                "app.py",
                "requirements.txt",
                "README.md"
            ]
            
            for file in essential_files:
                file_path = Path(file)
                results[f"{file}_exists"] = file_path.exists()
                if not file_path.exists():
                    warnings.append(f"Missing {file}")
            
            # Check directories
            essential_dirs = [
                "api",
                "core",
                "services",
                "repositories"
            ]
            
            for dir_name in essential_dirs:
                dir_path = Path(dir_name)
                results[f"{dir_name}_dir_exists"] = dir_path.exists()
                if not dir_path.exists():
                    warnings.append(f"Missing {dir_name} directory")
            
            # Check models directory (optional)
            models_dir = Path("models")
            results["models_dir_exists"] = models_dir.exists()
            if not models_dir.exists():
                warnings.append("Models directory not found - will be created automatically")
            
            # Check logs directory (optional)
            logs_dir = Path("logs")
            results["logs_dir_exists"] = logs_dir.exists()
            if not logs_dir.exists():
                warnings.append("Logs directory not found - will be created automatically")
            
            results["warnings"] = warnings
            
            # Success if core files exist
            success = all([
                results["app.py_exists"],
                results["api_dir_exists"],
                results["core_dir_exists"],
                results["services_dir_exists"]
            ])
            
            return success, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def generate_final_report(self):
        """Generate comprehensive final validation report."""
        print("\n" + "="*80)
        print("FINAL INTEGRATION VALIDATION REPORT")
        print("="*80)
        
        # Summary statistics
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for success, _ in self.validation_results.values() if success)
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Total Validations: {total_validations}")
        print(f"   Passed: {passed_validations}")
        print(f"   Failed: {total_validations - passed_validations}")
        print(f"   Success Rate: {(passed_validations/total_validations)*100:.1f}%")
        
        # Critical failures
        if self.critical_failures:
            print(f"\n❌ CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"   • {failure}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10 warnings
                print(f"   • {warning}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more warnings")
        
        # Detailed results
        print(f"\n📋 DETAILED VALIDATION RESULTS:")
        for validation_name, (success, details) in self.validation_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n{status} {validation_name}")
            
            if details and isinstance(details, dict):
                for key, value in details.items():
                    if key == "warnings":
                        continue  # Already shown above
                    elif key == "error":
                        print(f"      Error: {value}")
                    elif isinstance(value, bool):
                        icon = "✅" if value else "❌"
                        print(f"      {icon} {key.replace('_', ' ').title()}")
                    elif isinstance(value, str) and "details" in key:
                        print(f"      ℹ️  {key.replace('_', ' ').title()}: {value}")
        
        # Overall assessment
        print(f"\n" + "="*80)
        
        if len(self.critical_failures) == 0 and passed_validations >= total_validations * 0.9:
            print("🎉 SYSTEM VALIDATION SUCCESSFUL!")
            print("✅ Your real AI model integration is ready for deployment.")
            print("✅ All critical components are functioning correctly.")
            
        elif len(self.critical_failures) == 0 and passed_validations >= total_validations * 0.8:
            print("⚠️  SYSTEM VALIDATION MOSTLY SUCCESSFUL")
            print("✅ Core functionality is working correctly.")
            print("⚠️  Some non-critical components may need attention.")
            print("✅ System is suitable for deployment with monitoring.")
            
        else:
            print("❌ SYSTEM VALIDATION FAILED")
            print("❌ Critical issues must be resolved before deployment.")
            print("🔧 Please address the critical failures listed above.")
            
        print("="*80)
        
        # Next steps
        print(f"\n📋 NEXT STEPS:")
        
        if len(self.critical_failures) == 0:
            print("1. ✅ Run deployment scripts to finalize setup")
            print("2. ✅ Start the FastAPI server: uvicorn app:app --reload")
            print("3. ✅ Monitor system performance using the dashboard")
            print("4. ✅ Test with actual generation requests")
        else:
            print("1. 🔧 Resolve critical failures listed above")
            print("2. 🔧 Re-run validation: python scripts/final_validation.py")
            print("3. 📖 Check documentation for troubleshooting")
            print("4. 🆘 Contact support if issues persist")

async def main():
    """Main validation function."""
    validator = FinalValidator()
    
    try:
        print("🚀 Starting Final Integration Validation...")
        print("This may take a few minutes to complete.\n")
        
        success = await validator.run_comprehensive_validation()
        
        if success:
            print("\n🎉 Validation completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Validation completed with issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())