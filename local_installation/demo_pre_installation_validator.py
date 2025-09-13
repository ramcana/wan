"""
Demo script for the pre-installation validation system.
Demonstrates comprehensive validation capabilities before WAN2.2 installation.
"""

import sys
import json
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from pre_installation_validator import PreInstallationValidator
from interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


def create_demo_hardware_profile():
    """Create a demo hardware profile for testing."""
    return HardwareProfile(
        cpu=CPUInfo(
            model="Intel Core i7-12700K",
            cores=12,
            threads=20,
            base_clock=3.6,
            boost_clock=5.0,
            architecture="x86_64"
        ),
        memory=MemoryInfo(
            total_gb=32,
            available_gb=24,
            type="DDR4",
            speed=3200
        ),
        gpu=GPUInfo(
            model="NVIDIA RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.0",
            compute_capability="8.9"
        ),
        storage=StorageInfo(
            available_gb=1000,
            type="NVMe SSD"
        ),
        os=OSInfo(
            name="Windows 11",
            version="22H2",
            architecture="x86_64"
        )
    )


def demo_individual_validations():
    """Demonstrate individual validation components."""
    print("=== WAN2.2 Pre-Installation Validation Demo ===\n")
    
    # Create validator
    installation_path = "./demo_wan22_installation"
    hardware_profile = create_demo_hardware_profile()
    validator = PreInstallationValidator(installation_path, hardware_profile)
    
    print(f"Installation Path: {installation_path}")
    print(f"Hardware Profile: {hardware_profile.cpu.model}, {hardware_profile.memory.total_gb}GB RAM, {hardware_profile.gpu.model}\n")
    
    # Test each validation component
    validations = [
        ("System Requirements", validator.validate_system_requirements),
        ("Network Connectivity", validator.validate_network_connectivity),
        ("File Permissions", validator.validate_permissions),
        ("Installation Conflicts", validator.validate_existing_installation)
    ]
    
    results = {}
    
    for name, validation_func in validations:
        print(f"--- {name} ---")
        try:
            result = validation_func()
            results[name] = result
            
            print(f"✓ Success: {result.success}")
            print(f"  Message: {result.message}")
            
            if result.warnings:
                print(f"  Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    print(f"    - {warning}")
                if len(result.warnings) > 3:
                    print(f"    ... and {len(result.warnings) - 3} more")
            
            # Show some key details
            if result.details:
                if name == "System Requirements" and "requirements" in result.details:
                    reqs = result.details["requirements"]
                    print(f"  Requirements checked: {len(reqs)}")
                    for req in reqs:
                        status = "✓" if req["met"] else "✗"
                        print(f"    {status} {req['name']}: {req['current_value']}{req['unit']} (min: {req['minimum_value']}{req['unit']})")
                
                elif name == "Network Connectivity" and "network_tests" in result.details:
                    tests = result.details["network_tests"]
                    print(f"  Network tests: {len(tests)}")
                    for test in tests:
                        status = "✓" if test["success"] else "✗"
                        latency = f" ({test['latency_ms']:.1f}ms)" if test.get("latency_ms") else ""
                        bandwidth = f" ({test['bandwidth_mbps']:.1f} Mbps)" if test.get("bandwidth_mbps") else ""
                        print(f"    {status} {test['test_name']}{latency}{bandwidth}")
                
                elif name == "Installation Conflicts" and "conflicts" in result.details:
                    conflicts = result.details["conflicts"]
                    if conflicts:
                        print(f"  Conflicts found: {len(conflicts)}")
                        for conflict in conflicts[:3]:  # Show first 3 conflicts
                            print(f"    - {conflict['conflict_type']}: {conflict['description']}")
                    else:
                        print("  No conflicts detected")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results[name] = None
        
        print()
    
    return results


def demo_comprehensive_report():
    """Demonstrate comprehensive validation report generation."""
    print("=== Comprehensive Validation Report ===\n")
    
    installation_path = "./demo_wan22_installation"
    hardware_profile = create_demo_hardware_profile()
    validator = PreInstallationValidator(installation_path, hardware_profile)
    
    print("Generating comprehensive pre-installation validation report...")
    print("This may take a moment as it performs all validation checks...\n")
    
    try:
        report = validator.generate_validation_report()
        
        print(f"Report Generated: {report.timestamp}")
        print(f"Overall Success: {'✓' if report.overall_success else '✗'}")
        print(f"Total Errors: {len(report.errors)}")
        print(f"Total Warnings: {len(report.warnings)}")
        
        if report.estimated_install_time_minutes:
            print(f"Estimated Installation Time: {report.estimated_install_time_minutes} minutes")
        
        print(f"\nValidation Summary:")
        print(f"  System Requirements: {len(report.system_requirements)} checked")
        print(f"  Network Tests: {len(report.network_tests)} performed")
        print(f"  Permission Tests: {len(report.permission_tests)} performed")
        print(f"  Conflicts Detected: {len(report.conflicts)}")
        
        # Show critical issues
        if report.errors:
            print(f"\n❌ Critical Issues:")
            for error in report.errors:
                print(f"  - {error}")
        
        if report.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in report.warnings[:5]:  # Show first 5 warnings
                print(f"  - {warning}")
            if len(report.warnings) > 5:
                print(f"  ... and {len(report.warnings) - 5} more warnings")
        
        # Show readiness assessment
        print(f"\n🎯 Installation Readiness Assessment:")
        if report.overall_success:
            print("  ✅ System is ready for WAN2.2 installation")
            print("  🚀 You can proceed with confidence")
        else:
            print("  ❌ System has issues that should be resolved first")
            print("  🔧 Please address the critical issues above before installing")
        
        # Show report location
        report_path = Path(installation_path) / "logs" / "pre_validation_report.json"
        if report_path.exists():
            print(f"\n📄 Detailed report saved to: {report_path}")
            print(f"   Report size: {report_path.stat().st_size / 1024:.1f} KB")
        
        return report
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return None


def demo_timeout_management():
    """Demonstrate timeout management capabilities."""
    print("=== Timeout Management Demo ===\n")
    
    from pre_installation_validator import TimeoutManager
    import time
    
    print("Testing timeout manager with normal operation...")
    cleanup_called = False
    
    def cleanup_func():
        nonlocal cleanup_called
        cleanup_called = True
        print("  Cleanup function called!")
    
    # Normal operation (no timeout)
    with TimeoutManager(2, cleanup_func) as tm:
        time.sleep(0.5)
        print(f"  Operation completed in {tm.elapsed_time():.2f} seconds")
        print(f"  Timed out: {tm.is_timed_out()}")
    
    print(f"  Cleanup called: {cleanup_called}")
    print()
    
    # Timeout scenario (simulated)
    print("Testing timeout manager with timeout scenario...")
    cleanup_called = False
    
    with TimeoutManager(0.2, cleanup_func) as tm:
        time.sleep(0.3)  # This should trigger timeout
        print(f"  Operation time: {tm.elapsed_time():.2f} seconds")
    
    time.sleep(0.1)  # Give timeout a moment to trigger
    print(f"  Timed out: {tm.is_timed_out()}")
    print(f"  Cleanup called: {cleanup_called}")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-Installation Validator Demo")
    parser.add_argument("--individual", action="store_true",
                       help="Demo individual validation components")
    parser.add_argument("--report", action="store_true",
                       help="Demo comprehensive report generation")
    parser.add_argument("--timeout", action="store_true",
                       help="Demo timeout management")
    parser.add_argument("--all", action="store_true",
                       help="Run all demos")
    
    args = parser.parse_args()
    
    if args.all or not any([args.individual, args.report, args.timeout]):
        # Run all demos by default
        demo_individual_validations()
        print("\n" + "="*60 + "\n")
        demo_comprehensive_report()
        print("\n" + "="*60 + "\n")
        demo_timeout_management()
    else:
        if args.individual:
            demo_individual_validations()
        
        if args.report:
            demo_comprehensive_report()
        
        if args.timeout:
            demo_timeout_management()
    
    print("\n🎉 Demo completed! The pre-installation validator is ready for use.")


if __name__ == "__main__":
    main()
