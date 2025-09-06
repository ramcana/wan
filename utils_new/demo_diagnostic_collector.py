#!/usr/bin/env python3
"""
Demo script for the Wan model compatibility diagnostic collector.

This script demonstrates how to use the DiagnosticCollector to analyze
model compatibility and generate comprehensive diagnostic reports.
"""

import sys
import tempfile
import json
from pathlib import Path

from wan_diagnostic_collector import DiagnosticCollector


def create_mock_wan_model(model_dir: Path):
    """Create a mock Wan model directory for testing."""
    model_dir.mkdir(exist_ok=True)
    
    # Create model_index.json
    model_index = {
        "_class_name": "WanPipeline",
        "_diffusers_version": "0.21.0",
        "transformer": ["diffusers", "Transformer2DModel"],
        "transformer_2": ["diffusers", "Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "DDIMScheduler"],
        "boundary_ratio": 0.5
    }
    
    with open(model_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    
    # Create component directories
    for component in ["transformer", "transformer_2", "vae", "scheduler"]:
        comp_dir = model_dir / component
        comp_dir.mkdir(exist_ok=True)
        
        # Create config.json
        config = {
            "_class_name": model_index[component][1],
            "in_channels": 4 if component == "vae" else None,
            "out_channels": 4 if component == "vae" else None
        }
        config = {k: v for k, v in config.items() if v is not None}
        
        with open(comp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)


def demo_diagnostic_collection():
    """Demonstrate diagnostic collection and reporting."""
    print("=" * 60)
    print("WAN MODEL COMPATIBILITY DIAGNOSTIC DEMO")
    print("=" * 60)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock model
        model_path = temp_path / "wan_model_demo"
        create_mock_wan_model(model_path)
        
        # Create diagnostic collector
        diagnostics_dir = temp_path / "diagnostics"
        collector = DiagnosticCollector(diagnostics_dir=str(diagnostics_dir))
        
        print(f"Created mock Wan model at: {model_path}")
        print(f"Diagnostics will be saved to: {diagnostics_dir}")
        print()
        
        # Simulate a pipeline loading error
        pipeline_error = ImportError("WanPipeline class not found. Please install the required pipeline code.")
        
        # Collect diagnostics
        print("Collecting model diagnostics...")
        diagnostics = collector.collect_model_diagnostics(
            str(model_path), 
            load_attempt_result=pipeline_error
        )
        
        # Write compatibility report
        print("Writing compatibility report...")
        report_path = collector.write_compatibility_report("wan_model_demo", diagnostics)
        print(f"Report written to: {report_path}")
        print()
        
        # Generate and display summary
        print("Generating diagnostic summary...")
        summary = collector.generate_diagnostic_summary(diagnostics)
        print(summary)
        
        # Show JSON report structure
        print("\n" + "=" * 60)
        print("JSON REPORT STRUCTURE")
        print("=" * 60)
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        print("Report sections:")
        for key in report_data.keys():
            print(f"  - {key}")
        
        print(f"\nErrors found: {len(report_data['errors'])}")
        print(f"Warnings found: {len(report_data['warnings'])}")
        print(f"Recommendations provided: {len(report_data['recommendations'])}")
        
        if report_data['errors']:
            print("\nSample errors:")
            for error in report_data['errors'][:3]:
                print(f"  • {error}")
        
        if report_data['recommendations']:
            print("\nSample recommendations:")
            for rec in report_data['recommendations'][:3]:
                print(f"  • {rec}")


def demo_system_analysis():
    """Demonstrate system information collection."""
    print("\n" + "=" * 60)
    print("SYSTEM ANALYSIS DEMO")
    print("=" * 60)
    
    from wan_diagnostic_collector import SystemInfo
    
    print("Collecting system information...")
    system_info = SystemInfo.collect()
    
    print(f"GPU: {system_info.gpu_name or 'Not detected'}")
    print(f"VRAM: {system_info.vram_available or 'N/A'}MB / {system_info.vram_total or 'N/A'}MB")
    print(f"CUDA Available: {system_info.cuda_available}")
    print(f"CUDA Version: {system_info.cuda_version or 'N/A'}")
    print(f"Python: {system_info.python_version}")
    print(f"PyTorch: {system_info.torch_version}")
    print(f"Diffusers: {system_info.diffusers_version or 'Not installed'}")
    print(f"Platform: {system_info.platform_info.get('system', 'Unknown')}")


def main():
    """Main demo function."""
    try:
        demo_system_analysis()
        demo_diagnostic_collection()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The diagnostic collector can help identify and resolve")
        print("Wan model compatibility issues by providing:")
        print("  • Comprehensive system analysis")
        print("  • Model architecture detection")
        print("  • Pipeline compatibility validation")
        print("  • Detailed error reporting")
        print("  • Actionable recommendations")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())