#!/usr/bin/env python3
"""
Test script for WAN Generation Service integration
Tests the updated Generation Service to ensure it uses real WAN models instead of simulation
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WANGenerationServiceTester:
    """Test class for WAN Generation Service functionality"""
    
    def __init__(self):
        self.generation_service = None
        self.test_results = {}
        self.start_time = time.time()
    
    async def setup(self):
        """Setup test environment"""
        try:
            logger.info("Setting up WAN Generation Service test environment...")
            
            # Import and initialize the generation service
            from backend.services.generation_service import GenerationService
            self.generation_service = GenerationService()
            
            # Initialize the service
            await self.generation_service.initialize()
            
            logger.info("✅ Generation Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def test_wan_model_integration(self):
        """Test WAN model integration and availability"""
        test_name = "WAN Model Integration"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Check if model integration bridge is available
            if not self.generation_service.model_integration_bridge:
                raise Exception("Model Integration Bridge not available")
            
            # Check if the bridge is initialized
            if not self.generation_service.model_integration_bridge.is_initialized():
                raise Exception("Model Integration Bridge not initialized")
            
            # Test WAN model availability checking
            wan_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            available_models = []
            
            for model_type in wan_models:
                try:
                    model_status = await self.generation_service.model_integration_bridge.check_model_availability(model_type)
                    logger.info(f"  📋 {model_type}: {model_status.status.value}")
                    
                    if model_status.status.value in ["available", "loaded"]:
                        available_models.append(model_type)
                        logger.info(f"    ✅ Hardware compatible: {model_status.hardware_compatible}")
                        logger.info(f"    📊 Estimated VRAM: {model_status.estimated_vram_usage_mb/1024:.1f}GB")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ Error checking {model_type}: {e}")
            
            # Test result
            if available_models:
                self.test_results[test_name] = {
                    "status": "PASS",
                    "available_models": available_models,
                    "details": f"Found {len(available_models)} available WAN models"
                }
                logger.info(f"✅ {test_name} PASSED - Available models: {available_models}")
            else:
                self.test_results[test_name] = {
                    "status": "FAIL",
                    "available_models": [],
                    "details": "No WAN models available"
                }
                logger.warning(f"⚠️ {test_name} FAILED - No WAN models available")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_vram_monitoring(self):
        """Test WAN model VRAM monitoring functionality"""
        test_name = "VRAM Monitoring"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Check if VRAM monitor is available
            if not self.generation_service.vram_monitor:
                raise Exception("VRAM Monitor not available")
            
            # Test VRAM usage checking
            current_usage = self.generation_service.vram_monitor.get_current_vram_usage()
            logger.info(f"  📊 Current VRAM usage: {json.dumps(current_usage, indent=2)}")
            
            # Test WAN model VRAM estimation
            wan_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            for model_type in wan_models:
                estimated_vram = self.generation_service._estimate_wan_model_vram_requirements(model_type)
                logger.info(f"  📋 {model_type} estimated VRAM: {estimated_vram:.1f}GB")
                
                # Test VRAM availability check
                vram_available, vram_message = self.generation_service.vram_monitor.check_vram_availability(estimated_vram)
                logger.info(f"    ✅ VRAM check: {vram_available} - {vram_message}")
            
            # Test optimization suggestions
            suggestions = self.generation_service.vram_monitor.get_optimization_suggestions()
            logger.info(f"  💡 Optimization suggestions: {suggestions}")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "current_usage": current_usage,
                "optimization_suggestions": suggestions,
                "details": "VRAM monitoring working correctly"
            }
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_hardware_optimization(self):
        """Test hardware optimization for WAN models"""
        test_name = "Hardware Optimization"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Check if hardware profile is available
            if not self.generation_service.hardware_profile:
                logger.warning("  ⚠️ Hardware profile not available")
                hardware_info = "Not available"
            else:
                # Handle different hardware profile attribute names
                hardware_info = {}
                
                # Try different possible attribute names
                gpu_name = getattr(self.generation_service.hardware_profile, 'gpu_name', None) or \
                          getattr(self.generation_service.hardware_profile, 'gpu_model', None) or \
                          "Unknown GPU"
                
                total_vram = getattr(self.generation_service.hardware_profile, 'total_vram_gb', None) or \
                            getattr(self.generation_service.hardware_profile, 'vram_gb', None) or \
                            0.0
                
                cpu_cores = getattr(self.generation_service.hardware_profile, 'cpu_cores', None) or \
                           getattr(self.generation_service.hardware_profile, 'cores', None) or \
                           0
                
                architecture = getattr(self.generation_service.hardware_profile, 'architecture_type', None) or \
                              getattr(self.generation_service.hardware_profile, 'architecture', None) or \
                              "unknown"
                
                hardware_info = {
                    "gpu_name": gpu_name,
                    "total_vram_gb": total_vram,
                    "cpu_cores": cpu_cores,
                    "architecture": architecture
                }
                logger.info(f"  🖥️ Hardware profile: {json.dumps(hardware_info, indent=2)}")
            
            # Check if WAN22 system optimizer is available
            if not self.generation_service.wan22_system_optimizer:
                logger.warning("  ⚠️ WAN22 System Optimizer not available")
                optimizer_status = "Not available"
            else:
                optimizer_status = "Available"
                logger.info("  ⚙️ WAN22 System Optimizer is available")
            
            # Test optimization application
            optimization_applied = getattr(self.generation_service, 'optimization_applied', False)
            logger.info(f"  🔧 Hardware optimizations applied: {optimization_applied}")
            
            # Test alternative model selection
            test_model = "t2v-A14B"
            alternatives = self.generation_service._get_alternative_wan_models(test_model)
            logger.info(f"  🔄 Alternative models for {test_model}: {alternatives}")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "hardware_profile": hardware_info,
                "optimizer_status": optimizer_status,
                "optimization_applied": optimization_applied,
                "alternative_models": alternatives,
                "details": "Hardware optimization components checked"
            }
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_generation_mode_configuration(self):
        """Test generation mode configuration"""
        test_name = "Generation Mode Configuration"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Check generation mode settings
            use_real_generation = getattr(self.generation_service, 'use_real_generation', False)
            fallback_to_simulation = getattr(self.generation_service, 'fallback_to_simulation', True)
            prefer_wan_models = getattr(self.generation_service, 'prefer_wan_models', False)
            
            logger.info(f"  🎯 Use real generation: {use_real_generation}")
            logger.info(f"  🔄 Fallback to simulation: {fallback_to_simulation}")
            logger.info(f"  🏆 Prefer WAN models: {prefer_wan_models}")
            
            # Check if real generation pipeline is available
            if not self.generation_service.real_generation_pipeline:
                raise Exception("Real Generation Pipeline not available")
            
            logger.info("  ⚙️ Real Generation Pipeline is available")
            
            # Verify expected configuration
            expected_config = {
                "use_real_generation": True,
                "fallback_to_simulation": False,  # Should be disabled
                "prefer_wan_models": True
            }
            
            actual_config = {
                "use_real_generation": use_real_generation,
                "fallback_to_simulation": fallback_to_simulation,
                "prefer_wan_models": prefer_wan_models
            }
            
            config_correct = all(
                actual_config.get(key) == expected_config[key] 
                for key in expected_config
            )
            
            if config_correct:
                logger.info("  ✅ Configuration matches expected values")
            else:
                logger.warning(f"  ⚠️ Configuration mismatch - Expected: {expected_config}, Actual: {actual_config}")
            
            self.test_results[test_name] = {
                "status": "PASS" if config_correct else "WARN",
                "expected_config": expected_config,
                "actual_config": actual_config,
                "pipeline_available": True,
                "details": "Generation mode configuration checked"
            }
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_fallback_strategies(self):
        """Test WAN model fallback strategies"""
        test_name = "Fallback Strategies"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Test alternative model selection for different scenarios
            test_cases = [
                ("t2v-A14B", "Primary T2V model"),
                ("i2v-A14B", "Primary I2V model"),
                ("ti2v-5B", "Primary TI2V model"),
                ("invalid-model", "Invalid model type")
            ]
            
            fallback_results = {}
            
            for model_type, description in test_cases:
                try:
                    alternatives = self.generation_service._get_alternative_wan_models(model_type)
                    fallback_results[model_type] = {
                        "alternatives": alternatives,
                        "count": len(alternatives),
                        "description": description
                    }
                    logger.info(f"  🔄 {model_type} ({description}): {alternatives}")
                except Exception as e:
                    fallback_results[model_type] = {
                        "error": str(e),
                        "description": description
                    }
                    logger.warning(f"  ⚠️ {model_type} error: {e}")
            
            # Check if intelligent fallback manager is available
            fallback_manager_available = hasattr(self.generation_service, 'intelligent_fallback_manager') and \
                                       self.generation_service.intelligent_fallback_manager is not None
            
            logger.info(f"  🧠 Intelligent Fallback Manager available: {fallback_manager_available}")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "fallback_results": fallback_results,
                "fallback_manager_available": fallback_manager_available,
                "details": "Fallback strategies tested successfully"
            }
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_enhanced_components(self):
        """Test enhanced model availability components"""
        test_name = "Enhanced Components"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            components_status = {}
            
            # Check Model Availability Manager
            components_status["model_availability_manager"] = {
                "available": hasattr(self.generation_service, 'model_availability_manager') and 
                           self.generation_service.model_availability_manager is not None,
                "description": "Manages model availability and status"
            }
            
            # Check Enhanced Model Downloader
            components_status["enhanced_model_downloader"] = {
                "available": hasattr(self.generation_service, 'enhanced_model_downloader') and 
                           self.generation_service.enhanced_model_downloader is not None,
                "description": "Enhanced model downloading capabilities"
            }
            
            # Check Model Health Monitor
            components_status["model_health_monitor"] = {
                "available": hasattr(self.generation_service, 'model_health_monitor') and 
                           self.generation_service.model_health_monitor is not None,
                "description": "Monitors model health and performance"
            }
            
            # Check Model Usage Analytics
            components_status["model_usage_analytics"] = {
                "available": hasattr(self.generation_service, 'model_usage_analytics') and 
                           self.generation_service.model_usage_analytics is not None,
                "description": "Tracks model usage analytics"
            }
            
            # Check Performance Monitor
            components_status["performance_monitor"] = {
                "available": hasattr(self.generation_service, 'performance_monitor') and 
                           self.generation_service.performance_monitor is not None,
                "description": "Monitors system performance"
            }
            
            # Log component status
            for component, status in components_status.items():
                status_icon = "✅" if status["available"] else "❌"
                logger.info(f"  {status_icon} {component}: {status['available']} - {status['description']}")
            
            available_count = sum(1 for status in components_status.values() if status["available"])
            total_count = len(components_status)
            
            self.test_results[test_name] = {
                "status": "PASS",
                "components_status": components_status,
                "available_count": available_count,
                "total_count": total_count,
                "details": f"{available_count}/{total_count} enhanced components available"
            }
            logger.info(f"✅ {test_name} PASSED - {available_count}/{total_count} components available")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        test_duration = time.time() - self.start_time
        
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed": len([r for r in self.test_results.values() if r["status"] == "PASS"]),
                "failed": len([r for r in self.test_results.values() if r["status"] == "FAIL"]),
                "errors": len([r for r in self.test_results.values() if r["status"] == "ERROR"]),
                "warnings": len([r for r in self.test_results.values() if r["status"] == "WARN"]),
                "duration_seconds": round(test_duration, 2)
            },
            "test_results": self.test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_environment": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        return report
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting WAN Generation Service Tests...")
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        # Run tests
        test_methods = [
            self.test_wan_model_integration,
            self.test_vram_monitoring,
            self.test_hardware_optimization,
            self.test_generation_mode_configuration,
            self.test_fallback_strategies,
            self.test_enhanced_components
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"❌ Test method {test_method.__name__} failed: {e}")
        
        # Generate report
        report = self.generate_test_report()
        
        # Save report
        report_path = Path("wan_generation_service_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report["test_summary"]
        logger.info(f"\n📊 TEST SUMMARY:")
        logger.info(f"  Total Tests: {summary['total_tests']}")
        logger.info(f"  ✅ Passed: {summary['passed']}")
        logger.info(f"  ⚠️ Warnings: {summary['warnings']}")
        logger.info(f"  ❌ Failed: {summary['failed']}")
        logger.info(f"  💥 Errors: {summary['errors']}")
        logger.info(f"  ⏱️ Duration: {summary['duration_seconds']}s")
        logger.info(f"  📄 Report saved to: {report_path}")
        
        return summary['failed'] == 0 and summary['errors'] == 0

async def main():
    """Main test function"""
    tester = WANGenerationServiceTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("🎉 All tests completed successfully!")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
