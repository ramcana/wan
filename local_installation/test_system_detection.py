"""
Comprehensive test script for system and environment detection.
Tests both hardware detection and environment analysis.
"""

import sys
import os
import json
import logging

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Import the detection modules
from scripts.detect_system import SystemDetector
from scripts.environment_detector import EnvironmentDetector

def main():
    """Test comprehensive system detection functionality."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=== WAN2.2 Comprehensive System Detection Test ===\n")
    
    try:
        # Create detectors
        hardware_detector = SystemDetector(".")
        env_detector = EnvironmentDetector(".")
        
        # Phase 1: Hardware Detection
        print("🔍 Phase 1: Hardware Detection")
        print("=" * 50)
        hardware_profile = hardware_detector.detect_hardware()
        
        # Phase 2: Environment Detection
        print("\n🔍 Phase 2: Environment Detection")
        print("=" * 50)
        env_info = env_detector.detect_environment()
        
        # Phase 3: Generate Optimal Settings
        print("\n⚙️  Phase 3: Optimal Settings Generation")
        print("=" * 50)
        settings = hardware_detector.get_optimal_settings(hardware_profile)
        print("Generated optimal settings based on hardware profile")
        
        # Phase 4: Hardware Validation
        print("\n✅ Phase 4: Hardware Validation")
        print("=" * 50)
        hardware_validation = hardware_detector.validate_requirements(hardware_profile)
        print(f"Hardware Status: {'PASSED' if hardware_validation.success else 'FAILED'}")
        print(f"Message: {hardware_validation.message}")
        
        if hardware_validation.warnings:
            print("\nHardware Warnings:")
            for warning in hardware_validation.warnings:
                print(f"  ⚠️  {warning}")
        
        # Phase 5: System Capabilities Validation
        print("\n✅ Phase 5: System Capabilities Validation")
        print("=" * 50)
        system_validation = env_detector.validate_system_capabilities(hardware_profile, env_info)
        print(f"System Status: {'PASSED' if system_validation.success else 'FAILED'}")
        print(f"Message: {system_validation.message}")
        
        if system_validation.warnings:
            print("\nSystem Warnings:")
            for warning in system_validation.warnings:
                print(f"  ⚠️  {warning}")
        
        if not system_validation.success and system_validation.details:
            print("\nSystem Issues:")
            for issue in system_validation.details.get("issues", []):
                print(f"  ❌ {issue}")
            
            print("\nRecommendations:")
            for rec in system_validation.details.get("recommendations", []):
                print(f"  💡 {rec}")
        
        # Phase 6: Performance Tier Classification
        print("\n🏆 Phase 6: Performance Classification")
        print("=" * 50)
        tier = env_detector.classify_system_performance_tier(hardware_profile, env_info)
        print(f"Performance Tier: {tier.upper()}")
        
        # Phase 7: Summary Report
        print("\n📊 Phase 7: Summary Report")
        print("=" * 50)
        
        overall_status = hardware_validation.success and system_validation.success
        print(f"Overall System Status: {'✅ READY FOR INSTALLATION' if overall_status else '❌ ISSUES DETECTED'}")
        
        print(f"\nSystem Summary:")
        print(f"  • CPU: {hardware_profile.cpu.model} ({hardware_profile.cpu.cores}C/{hardware_profile.cpu.threads}T)")
        print(f"  • Memory: {hardware_profile.memory.total_gb}GB ({hardware_profile.memory.available_gb}GB available)")
        if hardware_profile.gpu:
            print(f"  • GPU: {hardware_profile.gpu.model} ({hardware_profile.gpu.vram_gb}GB VRAM)")
        else:
            print(f"  • GPU: None detected")
        print(f"  • Storage: {hardware_profile.storage.available_gb}GB available ({hardware_profile.storage.type})")
        print(f"  • OS: {env_info.windows_version} ({env_info.architecture})")
        print(f"  • Python Installations: {len(env_info.python_installations)}")
        print(f"  • Performance Tier: {tier.upper()}")
        
        # Show optimal settings summary
        print(f"\nOptimal Settings Summary:")
        print(f"  • CPU Threads: {settings['cpu_threads']}")
        print(f"  • Memory Allocation: {settings['memory_allocation_gb']}GB")
        print(f"  • GPU Acceleration: {'Enabled' if settings['enable_gpu_acceleration'] else 'Disabled'}")
        if settings.get('max_vram_usage_gb'):
            print(f"  • Max VRAM Usage: {settings['max_vram_usage_gb']}GB")
        print(f"  • Quantization: {settings.get('quantization', 'N/A')}")
        print(f"  • Model Offload: {'Enabled' if settings.get('enable_model_offload', False) else 'Disabled'}")
        
        # Export detailed report
        print(f"\n💾 Exporting detailed report...")
        report = {
            "timestamp": str(os.popen('date /t').read().strip()),
            "hardware_profile": {
                "cpu": {
                    "model": hardware_profile.cpu.model,
                    "cores": hardware_profile.cpu.cores,
                    "threads": hardware_profile.cpu.threads,
                    "base_clock": hardware_profile.cpu.base_clock,
                    "boost_clock": hardware_profile.cpu.boost_clock,
                    "architecture": hardware_profile.cpu.architecture
                },
                "memory": {
                    "total_gb": hardware_profile.memory.total_gb,
                    "available_gb": hardware_profile.memory.available_gb,
                    "type": hardware_profile.memory.type,
                    "speed": hardware_profile.memory.speed
                },
                "gpu": {
                    "model": hardware_profile.gpu.model if hardware_profile.gpu else None,
                    "vram_gb": hardware_profile.gpu.vram_gb if hardware_profile.gpu else 0,
                    "cuda_version": hardware_profile.gpu.cuda_version if hardware_profile.gpu else None,
                    "driver_version": hardware_profile.gpu.driver_version if hardware_profile.gpu else None
                } if hardware_profile.gpu else None,
                "storage": {
                    "available_gb": hardware_profile.storage.available_gb,
                    "type": hardware_profile.storage.type
                },
                "os": {
                    "name": hardware_profile.os.name,
                    "version": hardware_profile.os.version,
                    "architecture": hardware_profile.os.architecture
                }
            },
            "environment_info": {
                "windows_version": env_info.windows_version,
                "windows_build": env_info.windows_build,
                "windows_edition": env_info.windows_edition,
                "architecture": env_info.architecture,
                "user_privileges": env_info.user_privileges,
                "python_installations": env_info.python_installations,
                "installed_software": env_info.installed_software,
                "system_capabilities": env_info.system_capabilities
            },
            "validation_results": {
                "hardware_validation": {
                    "success": hardware_validation.success,
                    "message": hardware_validation.message,
                    "warnings": hardware_validation.warnings
                },
                "system_validation": {
                    "success": system_validation.success,
                    "message": system_validation.message,
                    "warnings": system_validation.warnings,
                    "issues": system_validation.details.get("issues", []) if system_validation.details else [],
                    "recommendations": system_validation.details.get("recommendations", []) if system_validation.details else []
                }
            },
            "performance_tier": tier,
            "optimal_settings": settings,
            "overall_status": overall_status
        }
        
        # Save report to file
        report_file = "system_detection_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed report saved to: {report_file}")
        
        if overall_status:
            print(f"\n🎉 System is ready for WAN2.2 installation!")
        else:
            print(f"\n⚠️  Please address the issues above before proceeding with installation.")
        
    except Exception as e:
        print(f"❌ System detection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()