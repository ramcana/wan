#!/usr/bin/env python3
"""
WAN Model Configuration Validation Script

This script validates all WAN model configurations, checks hardware compatibility,
and provides comprehensive reporting on model status and requirements.

Usage:
    python validate_wan_model_configurations.py [--detailed] [--hardware-check]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from backend.core.services.model_manager import (
        get_model_manager,
        validate_all_wan_configurations,
        get_all_wan_model_status,
        assess_hardware_compatibility,
        get_performance_profile,
        get_wan_model_recommendations
    )
    from backend.core.models.wan_models.wan_model_config import (
        get_available_wan_models,
        get_wan_model_info
    )
except ImportError as e:
    print(f"Error importing WAN model configuration system: {e}")
    print("Please ensure the WAN model configuration system is properly installed.")
    sys.exit(1)


def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection_header(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def validate_configurations(detailed: bool = False) -> Dict[str, Any]:
    """Validate all WAN model configurations"""
    print_section_header("WAN Model Configuration Validation")
    
    # Get validation results
    validation_results = validate_all_wan_configurations()
    
    if "error" in validation_results:
        print(f"❌ Error: {validation_results['error']}")
        return validation_results
    
    # Print summary
    summary = validation_results["summary"]
    print(f"Total Models: {summary['total_models']}")
    print(f"Valid Models: {summary['valid_models']}")
    print(f"Invalid Models: {summary['invalid_models']}")
    
    if validation_results["overall_valid"]:
        print("✅ All WAN model configurations are valid!")
    else:
        print("❌ Some WAN model configurations have issues:")
        for error in summary["errors"]:
            print(f"  - {error}")
    
    # Print detailed results if requested
    if detailed:
        print_subsection_header("Detailed Validation Results")
        for model_type, result in validation_results["model_validations"].items():
            status_icon = "✅" if result["is_valid"] else "❌"
            print(f"{status_icon} {model_type}: {'Valid' if result['is_valid'] else 'Invalid'}")
            
            if result["errors"]:
                for error in result["errors"]:
                    print(f"    - {error}")
    
    return validation_results


def check_model_status(detailed: bool = False) -> Dict[str, Any]:
    """Check status of all WAN models"""
    print_section_header("WAN Model Status Check")
    
    # Get model status
    status_results = get_all_wan_model_status()
    
    for model_type, status in status_results.items():
        print_subsection_header(f"Model: {model_type}")
        
        # Basic status
        print(f"Model ID: {status['model_id']}")
        print(f"Is WAN Model: {'✅' if status['is_wan_model'] else '❌'}")
        print(f"Is Cached: {'✅' if status['is_cached'] else '❌'}")
        print(f"Is Loaded: {'✅' if status['is_loaded'] else '❌'}")
        print(f"Is Valid: {'✅' if status['is_valid'] else '❌'}")
        
        if status['size_mb'] > 0:
            print(f"Size: {status['size_mb']:.1f} MB")
        
        # WAN-specific information
        if status['is_wan_model'] and status['wan_capabilities']:
            wan_caps = status['wan_capabilities']
            if 'capabilities' in wan_caps:
                caps = wan_caps['capabilities']
                print(f"Display Name: {caps.get('display_name', 'N/A')}")
                print(f"Architecture: {caps.get('architecture_type', 'N/A')}")
                print(f"Max Resolution: {caps.get('max_resolution', 'N/A')}")
                print(f"Max Frames: {caps.get('max_frames', 'N/A')}")
                
                if 'parameter_count' in caps:
                    param_count = caps['parameter_count']
                    if param_count >= 1_000_000_000:
                        print(f"Parameters: {param_count / 1_000_000_000:.1f}B")
                    elif param_count >= 1_000_000:
                        print(f"Parameters: {param_count / 1_000_000:.1f}M")
                    else:
                        print(f"Parameters: {param_count:,}")
            
            if 'requirements' in wan_caps:
                reqs = wan_caps['requirements']
                print(f"Min VRAM: {reqs.get('min_vram_gb', 'N/A')} GB")
                print(f"Estimated VRAM: {reqs.get('estimated_vram_gb', 'N/A')} GB")
        
        # Validation results
        if status.get('wan_validation'):
            validation = status['wan_validation']
            if validation['is_valid']:
                print("Validation: ✅ Passed")
            else:
                print("Validation: ❌ Failed")
                for error in validation['errors']:
                    print(f"  - {error}")
        
        # Hardware compatibility
        if detailed and status.get('hardware_compatibility'):
            hw_compat = status['hardware_compatibility']
            print(f"Hardware Compatible: {'✅' if hw_compat.get('is_compatible', False) else '❌'}")
            
            if 'vram_utilization' in hw_compat:
                vram = hw_compat['vram_utilization']
                print(f"VRAM Utilization: {vram.get('utilization_percent', 0):.1f}%")
    
    return status_results


def check_hardware_compatibility() -> Dict[str, Any]:
    """Check hardware compatibility for all WAN models"""
    print_section_header("Hardware Compatibility Assessment")
    
    # Try to detect available VRAM
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU: {gpu_name}")
            print(f"Available VRAM: {available_vram:.1f} GB")
        else:
            print("No CUDA GPU detected - CPU only mode")
            available_vram = 0.0
    except ImportError:
        print("PyTorch not available - cannot detect hardware")
        available_vram = 0.0
    
    # Get recommendations
    if available_vram > 0:
        recommendations = get_wan_model_recommendations(available_vram)
        
        if "error" not in recommendations:
            print_subsection_header("Model Recommendations")
            
            if recommendations.get("recommended_models"):
                print("✅ Recommended Models (should run optimally):")
                for model in recommendations["recommended_models"]:
                    print(f"  - {model['display_name']} ({model['model_type']})")
                    print(f"    VRAM: {model['estimated_vram_gb']} GB")
            
            if recommendations.get("compatible_models"):
                print("\n⚠️  Compatible Models (may need optimization):")
                for model in recommendations["compatible_models"]:
                    print(f"  - {model['display_name']} ({model['model_type']})")
                    print(f"    VRAM: {model['estimated_vram_gb']} GB")
            
            if recommendations.get("incompatible_models"):
                print("\n❌ Incompatible Models:")
                for model in recommendations["incompatible_models"]:
                    print(f"  - {model['display_name']} ({model['model_type']})")
                    print(f"    VRAM Deficit: {model['vram_deficit_gb']:.1f} GB")
            
            if recommendations.get("optimization_suggestions"):
                print("\n💡 Optimization Suggestions:")
                for suggestion in recommendations["optimization_suggestions"]:
                    print(f"  - {suggestion}")
    
    # Check individual model compatibility
    compatibility_results = {}
    wan_models = get_available_wan_models()
    
    for model_type in wan_models:
        print_subsection_header(f"Compatibility: {model_type}")
        
        compat_result = assess_hardware_compatibility(model_type)
        compatibility_results[model_type] = compat_result
        
        if "error" in compat_result:
            print(f"❌ Error: {compat_result['error']}")
            continue
        
        is_compatible = compat_result.get("is_compatible", False)
        print(f"Compatible: {'✅' if is_compatible else '❌'}")
        
        if not is_compatible and compat_result.get("compatibility_errors"):
            print("Issues:")
            for error in compat_result["compatibility_errors"]:
                print(f"  - {error}")
        
        if compat_result.get("optimal_profile"):
            print(f"Optimal Profile: {compat_result['optimal_profile']}")
        
        if compat_result.get("recommendations"):
            print("Recommendations:")
            for rec in compat_result["recommendations"]:
                print(f"  - {rec}")
    
    return compatibility_results


def generate_report(output_file: str = None) -> Dict[str, Any]:
    """Generate comprehensive validation report"""
    print_section_header("Generating Comprehensive Report")
    
    report = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "validation_results": validate_all_wan_configurations(),
        "model_status": get_all_wan_model_status(),
        "available_models": []
    }
    
    # Add model information
    wan_models = get_available_wan_models()
    for model_type in wan_models:
        model_info = get_wan_model_info(model_type)
        if model_info:
            report["available_models"].append(model_info)
    
    # Add hardware compatibility if possible
    try:
        import torch
        if torch.cuda.is_available():
            available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            report["hardware_recommendations"] = get_wan_model_recommendations(available_vram)
    except ImportError:
        pass
    
    # Save report if output file specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Report saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
    
    return report


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate WAN model configurations and check hardware compatibility"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true", 
        help="Show detailed validation results"
    )
    parser.add_argument(
        "--hardware-check", 
        action="store_true", 
        help="Perform hardware compatibility check"
    )
    parser.add_argument(
        "--report", 
        type=str, 
        help="Generate JSON report and save to file"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Suppress output (useful with --report)"
    )
    
    args = parser.parse_args()
    
    if args.quiet and not args.report:
        print("Warning: --quiet specified without --report, no output will be generated")
        return
    
    try:
        if not args.quiet:
            # Run validation
            validation_results = validate_configurations(detailed=args.detailed)
            
            # Check model status
            status_results = check_model_status(detailed=args.detailed)
            
            # Hardware compatibility check
            if args.hardware_check:
                compatibility_results = check_hardware_compatibility()
        
        # Generate report if requested
        if args.report:
            report = generate_report(args.report)
            
            if args.quiet:
                # Just print summary when quiet
                summary = report["validation_results"]["summary"]
                if report["validation_results"]["overall_valid"]:
                    print(f"✅ All {summary['total_models']} WAN model configurations are valid")
                else:
                    print(f"❌ {summary['invalid_models']}/{summary['total_models']} WAN model configurations have issues")
        
        if not args.quiet:
            print_section_header("Validation Complete")
            print("✅ WAN model configuration validation completed successfully!")
    
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
