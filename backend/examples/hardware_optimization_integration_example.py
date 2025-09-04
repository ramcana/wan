from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Hardware Optimization Integration Example

This example demonstrates how the enhanced generation service integrates
with the WAN22SystemOptimizer for hardware-specific optimizations.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from services.generation_service import GenerationService, VRAMMonitor
from core.system_integration import SystemIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_hardware_optimization_integration():
    """Demonstrate hardware optimization integration with generation service"""
    
    logger.info("=== Hardware Optimization Integration Demo ===")
    
    try:
        # Initialize system integration
        logger.info("1. Initializing system integration...")
        system_integration = SystemIntegration()
        init_success = await system_integration.initialize()
        
        if not init_success:
            logger.warning("System integration initialization had issues, continuing with available components")
        
        # Get system information
        system_info = system_integration.get_system_info()
        logger.info(f"System components initialized: {system_info['components']}")
        
        # Check if WAN22SystemOptimizer is available
        wan22_optimizer = system_integration.get_wan22_system_optimizer()
        if wan22_optimizer:
            logger.info("✅ WAN22SystemOptimizer is available")
            
            # Get hardware profile
            hardware_profile = wan22_optimizer.get_hardware_profile()
            if hardware_profile:
                logger.info(f"Hardware Profile:")
                logger.info(f"  CPU: {hardware_profile.cpu_model} ({hardware_profile.cpu_cores} cores)")
                logger.info(f"  Memory: {hardware_profile.total_memory_gb}GB")
                logger.info(f"  GPU: {hardware_profile.gpu_model} ({hardware_profile.vram_gb}GB VRAM)")
                logger.info(f"  CUDA: {hardware_profile.cuda_version}")
            
            # Get system health metrics
            health_metrics = wan22_optimizer.monitor_system_health()
            logger.info(f"System Health:")
            logger.info(f"  CPU Usage: {health_metrics.cpu_usage_percent}%")
            logger.info(f"  Memory Usage: {health_metrics.memory_usage_gb}GB")
            logger.info(f"  VRAM Usage: {health_metrics.vram_usage_mb}MB / {health_metrics.vram_total_mb}MB")
            
        else:
            logger.warning("❌ WAN22SystemOptimizer is not available")
        
        # Initialize enhanced generation service
        logger.info("\n2. Initializing enhanced generation service...")
        generation_service = GenerationService()
        await generation_service.initialize()
        
        # Check hardware optimization integration
        if generation_service.wan22_system_optimizer:
            logger.info("✅ Hardware optimization integrated with generation service")
            
            if generation_service.hardware_profile:
                logger.info(f"Hardware profile loaded: {generation_service.hardware_profile.gpu_model}")
                
                # Demonstrate VRAM monitoring
                if generation_service.vram_monitor:
                    logger.info("✅ VRAM monitoring enabled")
                    
                    # Test VRAM availability check
                    vram_available, vram_message = generation_service.vram_monitor.check_vram_availability(8.0)
                    logger.info(f"VRAM availability check (8GB): {vram_message}")
                    
                    # Get optimization suggestions
                    suggestions = generation_service.vram_monitor.get_optimization_suggestions()
                    if suggestions:
                        logger.info(f"VRAM optimization suggestions: {', '.join(suggestions[:3])}")
                    else:
                        logger.info("No VRAM optimization suggestions needed")
                
                # Demonstrate VRAM requirements estimation
                logger.info("\n3. VRAM Requirements Estimation:")
                for model_type in ["t2v", "i2v", "ti2v"]:
                    for resolution in ["1280x720", "1920x1080"]:
                        vram_req = generation_service._estimate_vram_requirements(model_type, resolution)
                        logger.info(f"  {model_type.upper()} @ {resolution}: {vram_req:.1f}GB")
                
                # Show applied optimizations
                if generation_service.optimization_applied:
                    logger.info("\n4. Applied Hardware Optimizations:")
                    if hasattr(generation_service, 'optimal_vram_usage_gb'):
                        logger.info(f"  Optimal VRAM usage: {generation_service.optimal_vram_usage_gb:.1f}GB")
                    if hasattr(generation_service, 'enable_tensor_cores'):
                        logger.info(f"  Tensor cores enabled: {generation_service.enable_tensor_cores}")
                    if hasattr(generation_service, 'enable_cpu_multithreading'):
                        logger.info(f"  CPU multithreading: {generation_service.enable_cpu_multithreading}")
                    if hasattr(generation_service, 'cpu_worker_threads'):
                        logger.info(f"  CPU worker threads: {generation_service.cpu_worker_threads}")
                    if hasattr(generation_service, 'enable_model_caching'):
                        logger.info(f"  Model caching enabled: {generation_service.enable_model_caching}")
                else:
                    logger.info("No specific hardware optimizations applied")
            
        else:
            logger.warning("❌ Hardware optimization not integrated")
        
        # Get enhanced queue status
        logger.info("\n5. Enhanced Queue Status:")
        queue_status = generation_service.get_queue_status()
        
        if "hardware_optimization" in queue_status:
            hw_opt = queue_status["hardware_optimization"]
            logger.info(f"  Hardware optimizer available: {hw_opt['optimizer_available']}")
            logger.info(f"  Hardware profile loaded: {hw_opt['hardware_profile_loaded']}")
            logger.info(f"  Optimization applied: {hw_opt['optimization_applied']}")
            logger.info(f"  VRAM monitoring enabled: {hw_opt['vram_monitoring_enabled']}")
            
            if "gpu_model" in hw_opt:
                logger.info(f"  GPU: {hw_opt['gpu_model']} ({hw_opt['vram_gb']}GB)")
            
            if "current_vram_usage_gb" in hw_opt:
                logger.info(f"  Current VRAM usage: {hw_opt['current_vram_usage_gb']:.1f}GB ({hw_opt['current_vram_usage_percent']:.1f}%)")
        
        # Get generation statistics
        logger.info("\n6. Generation Statistics:")
        gen_stats = generation_service.get_generation_stats()
        
        components = gen_stats["components_status"]
        logger.info(f"  Hardware optimizer: {components['hardware_optimizer']}")
        logger.info(f"  VRAM monitor: {components['vram_monitor']}")
        
        if "system_health" in gen_stats:
            health = gen_stats["system_health"]
            logger.info(f"  System health - CPU: {health['cpu_usage_percent']}%, Memory: {health['memory_usage_gb']}GB")
        
        if "vram_monitoring" in gen_stats:
            vram = gen_stats["vram_monitoring"]
            logger.info(f"  VRAM monitoring - Current: {vram['current_usage_gb']:.1f}GB, Optimal: {vram['optimal_usage_gb']:.1f}GB")
        
        logger.info("\n✅ Hardware optimization integration demonstration completed successfully!")
        
        # Clean up
        generation_service.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def demonstrate_vram_monitor():
    """Demonstrate VRAM monitor functionality"""
    
    logger.info("\n=== VRAM Monitor Demo ===")
    
    try:
        # Create VRAM monitor for RTX 4080 (16GB)
        vram_monitor = VRAMMonitor(
            total_vram_gb=16.0,
            optimal_usage_gb=13.6,  # 85% of 16GB
            system_optimizer=None
        )
        
        logger.info(f"VRAM Monitor created:")
        logger.info(f"  Total VRAM: {vram_monitor.total_vram_gb}GB")
        logger.info(f"  Optimal usage: {vram_monitor.optimal_usage_gb}GB")
        logger.info(f"  Warning threshold: {vram_monitor.warning_threshold * 100}%")
        logger.info(f"  Critical threshold: {vram_monitor.critical_threshold * 100}%")
        
        # Test VRAM availability checks
        logger.info("\nVRAM Availability Tests:")
        test_requirements = [4.0, 8.0, 12.0, 16.0]
        
        for req_gb in test_requirements:
            # Mock current usage for demonstration
            import unittest.mock
            with unittest.mock.patch.object(vram_monitor, 'get_current_vram_usage') as mock_usage:
                mock_usage.return_value = {
                    "allocated_gb": 6.0,  # Simulate 6GB currently used
                    "usage_percent": 37.5
                }
                
                available, message = vram_monitor.check_vram_availability(req_gb)
                status = "✅ Available" if available else "❌ Insufficient"
                logger.info(f"  {req_gb}GB required: {status} - {message}")
        
        # Test optimization suggestions
        logger.info("\nOptimization Suggestions Tests:")
        usage_scenarios = [
            {"allocated_gb": 8.0, "optimal_usage_percent": 58.8, "scenario": "Normal usage"},
            {"allocated_gb": 12.5, "optimal_usage_percent": 91.9, "scenario": "High usage"},
            {"allocated_gb": 13.5, "optimal_usage_percent": 99.3, "scenario": "Critical usage"}
        ]
        
        for scenario in usage_scenarios:
            import unittest.mock
            with unittest.mock.patch.object(vram_monitor, 'get_current_vram_usage') as mock_usage:
                mock_usage.return_value = scenario
                
                suggestions = vram_monitor.get_optimization_suggestions()
                logger.info(f"  {scenario['scenario']} ({scenario['allocated_gb']}GB):")
                if suggestions:
                    for suggestion in suggestions[:3]:  # Show first 3 suggestions
                        logger.info(f"    - {suggestion}")
                else:
                    logger.info(f"    - No optimizations needed")
        
        logger.info("\n✅ VRAM monitor demonstration completed!")
        
    except Exception as e:
        logger.error(f"❌ VRAM monitor demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def main():
    """Main demonstration function"""
    logger.info("Starting Hardware Optimization Integration Demonstration")
    
    # Run hardware optimization integration demo
    await demonstrate_hardware_optimization_integration()
    
    # Run VRAM monitor demo
    await demonstrate_vram_monitor()
    
    logger.info("\n🎉 All demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(main())