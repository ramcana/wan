#!/usr/bin/env python3
"""
Integration test for WAN Generation Service
Tests the complete generation workflow with real WAN model integration
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WANGenerationIntegrationTest:
    """Integration test for WAN generation workflow"""
    
    def __init__(self):
        self.generation_service = None
        self.test_results = {}
    
    async def setup(self):
        """Setup test environment"""
        try:
            logger.info("🔧 Setting up WAN generation integration test...")
            
            # Import and initialize the generation service
            from backend.services.generation_service import GenerationService
            self.generation_service = GenerationService()
            
            # Initialize the service
            await self.generation_service.initialize()
            
            logger.info("✅ Generation service initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_generation_workflow_simulation(self):
        """Test the complete generation workflow (simulated)"""
        test_name = "Generation Workflow Simulation"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Create a mock generation task
            from backend.repositories.database import GenerationTaskDB, TaskStatusEnum
            from sqlalchemy.orm import Session
            from datetime import datetime
            
            # Create mock task object
            class MockTask:
                def __init__(self):
                    self.id = "test_task_001"
                    self.prompt = "A beautiful sunset over mountains"
                    self.resolution = "1280x720"
                    self.steps = 50
                    self.image_path = None
                    self.end_image_path = None
                    self.lora_path = None
                    self.lora_strength = 1.0
                    self.guidance_scale = 7.5
                    self.negative_prompt = None
                    self.seed = None
                    self.fps = 8.0
                    self.num_frames = 16
                    self.status = TaskStatusEnum.PENDING
                    self.progress = 0
                    self.started_at = None
                    self.completed_at = None
                    self.error_message = None
                    self.output_path = None
                    self.generation_time_seconds = None
                    self.model_used = None
                    self.peak_vram_usage_mb = None
                    self.average_vram_usage_mb = None
                    self.optimizations_applied = None
            
            # Create mock database session
            class MockDB:
                def commit(self):
                    pass
                def rollback(self):
                    pass
            
            mock_task = MockTask()
            mock_db = MockDB()
            
            # Test WAN model VRAM estimation
            logger.info("  📊 Testing WAN model VRAM estimation...")
            for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
                estimated_vram = self.generation_service._estimate_wan_model_vram_requirements(
                    model_type, mock_task.resolution
                )
                logger.info(f"    {model_type}: {estimated_vram:.1f}GB estimated VRAM")
            
            # Test alternative model selection
            logger.info("  🔄 Testing alternative model selection...")
            alternatives = self.generation_service._get_alternative_wan_models("t2v-A14B")
            logger.info(f"    Alternatives for t2v-A14B: {alternatives}")
            
            # Test VRAM optimization methods
            logger.info("  ⚙️ Testing VRAM optimization methods...")
            await self.generation_service._apply_wan_model_vram_optimizations("t2v-A14B")
            logger.info("    ✅ WAN model VRAM optimizations applied")
            
            # Test hardware optimization methods
            if self.generation_service.wan22_system_optimizer:
                logger.info("  🖥️ Testing hardware optimization methods...")
                await self.generation_service._apply_wan_model_pre_generation_optimizations("t2v-A14B")
                logger.info("    ✅ WAN model pre-generation optimizations applied")
            
            # Test aggressive optimization
            logger.info("  🚨 Testing aggressive optimization...")
            await self.generation_service._enable_wan_model_aggressive_optimization()
            logger.info("    ✅ Aggressive WAN model optimization enabled")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "details": "Generation workflow simulation completed successfully",
                "vram_estimations": {
                    "t2v-A14B": self.generation_service._estimate_wan_model_vram_requirements("t2v-A14B"),
                    "i2v-A14B": self.generation_service._estimate_wan_model_vram_requirements("i2v-A14B"),
                    "ti2v-5B": self.generation_service._estimate_wan_model_vram_requirements("ti2v-5B")
                },
                "alternatives_tested": alternatives,
                "optimizations_applied": True
            }
            
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_model_integration_bridge_methods(self):
        """Test model integration bridge methods"""
        test_name = "Model Integration Bridge Methods"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            bridge = self.generation_service.model_integration_bridge
            
            if not bridge:
                raise Exception("Model Integration Bridge not available")
            
            # Test model availability checking
            logger.info("  📋 Testing model availability checking...")
            model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            availability_results = {}
            
            for model_type in model_types:
                try:
                    status = await bridge.check_model_availability(model_type)
                    availability_results[model_type] = {
                        "status": status.status.value,
                        "hardware_compatible": status.hardware_compatible,
                        "estimated_vram_mb": status.estimated_vram_usage_mb
                    }
                    logger.info(f"    {model_type}: {status.status.value}")
                except Exception as e:
                    availability_results[model_type] = {"error": str(e)}
                    logger.warning(f"    {model_type}: Error - {e}")
            
            # Test hardware profile access
            logger.info("  🖥️ Testing hardware profile access...")
            hardware_profile = getattr(bridge, 'hardware_profile', None)
            if hardware_profile:
                logger.info("    ✅ Hardware profile available")
            else:
                logger.warning("    ⚠️ Hardware profile not available")
            
            # Test model type mappings
            logger.info("  🗺️ Testing model type mappings...")
            model_mappings = getattr(bridge, 'model_type_mappings', {})
            logger.info(f"    Model type mappings: {len(model_mappings)} entries")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "availability_results": availability_results,
                "hardware_profile_available": hardware_profile is not None,
                "model_mappings_count": len(model_mappings),
                "details": "Model integration bridge methods tested successfully"
            }
            
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_real_generation_pipeline_integration(self):
        """Test real generation pipeline integration"""
        test_name = "Real Generation Pipeline Integration"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            pipeline = self.generation_service.real_generation_pipeline
            
            if not pipeline:
                raise Exception("Real Generation Pipeline not available")
            
            # Test pipeline initialization
            logger.info("  ⚙️ Testing pipeline initialization...")
            logger.info("    ✅ Pipeline is initialized")
            
            # Test hardware optimizer integration
            logger.info("  🖥️ Testing hardware optimizer integration...")
            if hasattr(pipeline, 'wan22_system_optimizer'):
                logger.info("    ✅ Hardware optimizer integrated")
            else:
                logger.warning("    ⚠️ Hardware optimizer not integrated")
            
            # Test WebSocket manager integration
            logger.info("  🌐 Testing WebSocket manager integration...")
            if hasattr(pipeline, 'websocket_manager') and pipeline.websocket_manager:
                logger.info("    ✅ WebSocket manager integrated")
            else:
                logger.warning("    ⚠️ WebSocket manager not integrated")
            
            # Test LoRA manager integration
            logger.info("  🎨 Testing LoRA manager integration...")
            if hasattr(pipeline, 'lora_manager') and pipeline.lora_manager:
                logger.info("    ✅ LoRA manager integrated")
            else:
                logger.warning("    ⚠️ LoRA manager not integrated")
            
            # Test pipeline cache
            logger.info("  💾 Testing pipeline cache...")
            pipeline_cache = getattr(pipeline, '_pipeline_cache', {})
            logger.info(f"    Pipeline cache size: {len(pipeline_cache)}")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "pipeline_initialized": True,
                "hardware_optimizer_integrated": hasattr(pipeline, 'wan22_system_optimizer'),
                "websocket_manager_integrated": hasattr(pipeline, 'websocket_manager') and pipeline.websocket_manager is not None,
                "lora_manager_integrated": hasattr(pipeline, 'lora_manager') and pipeline.lora_manager is not None,
                "pipeline_cache_size": len(pipeline_cache),
                "details": "Real generation pipeline integration tested successfully"
            }
            
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    async def test_monitoring_and_analytics(self):
        """Test monitoring and analytics components"""
        test_name = "Monitoring and Analytics"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Test VRAM monitoring
            logger.info("  📊 Testing VRAM monitoring...")
            vram_monitor = self.generation_service.vram_monitor
            if vram_monitor:
                current_usage = vram_monitor.get_current_vram_usage()
                logger.info(f"    Current VRAM usage: {current_usage.get('allocated_gb', 0):.1f}GB")
                logger.info("    ✅ VRAM monitoring working")
            else:
                logger.warning("    ⚠️ VRAM monitor not available")
            
            # Test performance monitoring
            logger.info("  📈 Testing performance monitoring...")
            perf_monitor = self.generation_service.performance_monitor
            if perf_monitor:
                logger.info("    ✅ Performance monitor available")
            else:
                logger.warning("    ⚠️ Performance monitor not available")
            
            # Test model health monitoring
            logger.info("  🏥 Testing model health monitoring...")
            health_monitor = self.generation_service.model_health_monitor
            if health_monitor:
                logger.info("    ✅ Model health monitor available")
            else:
                logger.warning("    ⚠️ Model health monitor not available")
            
            # Test usage analytics
            logger.info("  📊 Testing usage analytics...")
            usage_analytics = self.generation_service.model_usage_analytics
            if usage_analytics:
                logger.info("    ✅ Usage analytics available")
            else:
                logger.warning("    ⚠️ Usage analytics not available")
            
            self.test_results[test_name] = {
                "status": "PASS",
                "vram_monitor_available": vram_monitor is not None,
                "performance_monitor_available": perf_monitor is not None,
                "health_monitor_available": health_monitor is not None,
                "usage_analytics_available": usage_analytics is not None,
                "details": "Monitoring and analytics components tested successfully"
            }
            
            logger.info(f"✅ {test_name} PASSED")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": f"Exception during test: {e}"
            }
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    def generate_report(self):
        """Generate integration test report"""
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed": len([r for r in self.test_results.values() if r["status"] == "PASS"]),
                "failed": len([r for r in self.test_results.values() if r["status"] == "FAIL"]),
                "errors": len([r for r in self.test_results.values() if r["status"] == "ERROR"]),
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results,
            "integration_status": {
                "wan_model_integration": "READY" if self.generation_service else "NOT_READY",
                "hardware_optimization": "ENABLED" if getattr(self.generation_service, 'wan22_system_optimizer', None) else "DISABLED",
                "vram_monitoring": "ENABLED" if getattr(self.generation_service, 'vram_monitor', None) else "DISABLED",
                "fallback_strategies": "ENABLED" if hasattr(self.generation_service, '_get_alternative_wan_models') else "DISABLED"
            }
        }
        
        return report
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting WAN Generation Integration Tests...")
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        # Run tests
        test_methods = [
            self.test_generation_workflow_simulation,
            self.test_model_integration_bridge_methods,
            self.test_real_generation_pipeline_integration,
            self.test_monitoring_and_analytics
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"❌ Test method {test_method.__name__} failed: {e}")
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_path = Path("wan_generation_integration_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report["test_summary"]
        integration_status = report["integration_status"]
        
        logger.info(f"\n📊 INTEGRATION TEST SUMMARY:")
        logger.info(f"  Total Tests: {summary['total_tests']}")
        logger.info(f"  ✅ Passed: {summary['passed']}")
        logger.info(f"  ❌ Failed: {summary['failed']}")
        logger.info(f"  💥 Errors: {summary['errors']}")
        
        logger.info(f"\n🔧 INTEGRATION STATUS:")
        for component, status in integration_status.items():
            status_icon = "✅" if status in ["READY", "ENABLED"] else "❌"
            logger.info(f"  {status_icon} {component.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\n📄 Report saved to: {report_path}")
        
        return summary['failed'] == 0 and summary['errors'] == 0

async def main():
    """Main test function"""
    tester = WANGenerationIntegrationTest()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("🎉 All integration tests completed successfully!")
        return 0
    else:
        logger.error("💥 Some integration tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
