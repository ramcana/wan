"""
Demonstration of VRAM optimization and resource management functionality
Shows proactive VRAM checking, parameter optimization, and memory cleanup
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from resource_manager import (
    VRAMOptimizer, ResourceStatus, get_resource_manager,
    check_vram_availability, estimate_resource_requirements,
    optimize_parameters_for_resources, cleanup_memory, get_system_resource_info
)

def demo_basic_functionality():
    """Demonstrate basic resource management functionality"""
    print("=== Basic Resource Management Demo ===\n")
    
    # Get system resource information
    print("1. System Resource Information:")
    resource_info = get_system_resource_info()
    print(f"   VRAM: {resource_info.vram.total_mb:.0f}MB total, {resource_info.vram.free_mb:.0f}MB free ({resource_info.vram.utilization_percent:.1f}% used)")
    print(f"   RAM: {resource_info.ram_total_gb:.1f}GB total, {resource_info.ram_available_gb:.1f}GB available ({resource_info.ram_usage_percent:.1f}% used)")
    print(f"   CPU: {resource_info.cpu_usage_percent:.1f}% usage")
    print(f"   Disk: {resource_info.disk_free_gb:.1f}GB free")
    print()
    
    # Check VRAM availability for different requirements
    print("2. VRAM Availability Checks:")
    test_requirements = [2000, 4000, 6000, 8000, 12000]  # MB
    
    for req_mb in test_requirements:
        available, message = check_vram_availability(req_mb)
        status = "✓ Available" if available else "✗ Insufficient"
        print(f"   {req_mb}MB: {status} - {message}")
    print()

def demo_resource_estimation():
    """Demonstrate resource requirement estimation"""
    print("=== Resource Requirement Estimation Demo ===\n")
    
    # Test different model types and configurations
    test_configs = [
        {"model": "t2v-A14B", "resolution": "720p", "steps": 50, "duration": 4, "loras": 0},
        {"model": "t2v-A14B", "resolution": "1080p", "steps": 50, "duration": 4, "loras": 0},
        {"model": "i2v-A14B", "resolution": "720p", "steps": 40, "duration": 6, "loras": 1},
        {"model": "ti2v-5B", "resolution": "720p", "steps": 30, "duration": 4, "loras": 2},
    ]
    
    print("Resource estimates for different configurations:")
    print(f"{'Model':<10} {'Resolution':<10} {'Steps':<6} {'Duration':<8} {'LoRAs':<6} {'VRAM (MB)':<10} {'RAM (MB)':<9} {'Time (s)':<8}")
    print("-" * 80)
    
    for config in test_configs:
        requirement = estimate_resource_requirements(
            model_type=config["model"],
            resolution=config["resolution"],
            steps=config["steps"],
            duration=config["duration"],
            lora_count=config["loras"]
        )
        
        print(f"{config['model']:<10} {config['resolution']:<10} {config['steps']:<6} "
              f"{config['duration']:<8} {config['loras']:<6} {requirement.vram_mb:<10.0f} "
              f"{requirement.ram_mb:<9.0f} {requirement.estimated_time_seconds:<8.0f}")
    print()

def demo_parameter_optimization():
    """Demonstrate automatic parameter optimization"""
    print("=== Parameter Optimization Demo ===\n")
    
    # Test scenarios with different resource constraints
    test_scenarios = [
        {
            "name": "High-end configuration",
            "params": {
                "model_type": "t2v-A14B",
                "resolution": "1080p",
                "steps": 60,
                "duration": 8,
                "guidance_scale": 12.0,
                "lora_config": {"lora1": 0.8, "lora2": 0.6}
            }
        },
        {
            "name": "Medium configuration",
            "params": {
                "model_type": "i2v-A14B",
                "resolution": "720p",
                "steps": 50,
                "duration": 4,
                "guidance_scale": 8.0,
                "lora_config": {"lora1": 1.0}
            }
        },
        {
            "name": "Conservative configuration",
            "params": {
                "model_type": "ti2v-5B",
                "resolution": "480p",
                "steps": 25,
                "duration": 2,
                "guidance_scale": 6.0,
                "lora_config": {}
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"Scenario: {scenario['name']}")
        print("Original parameters:")
        for key, value in scenario['params'].items():
            if key != 'lora_config':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {len(value)} LoRAs")
        
        # Get optimized parameters
        optimized_params, suggestions = optimize_parameters_for_resources(scenario['params'])
        
        if suggestions:
            print("Optimization suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion.parameter}: {suggestion.current_value} → {suggestion.suggested_value}")
                print(f"    Reason: {suggestion.reason}")
                if suggestion.vram_savings_mb > 0:
                    print(f"    VRAM savings: {suggestion.vram_savings_mb:.0f}MB")
        else:
            print("No optimizations needed - configuration is optimal for current resources")
        
        print("Optimized parameters:")
        for key, value in optimized_params.items():
            if key != 'lora_config' and key != 'optimization_settings':
                print(f"  {key}: {value}")
            elif key == 'lora_config':
                print(f"  {key}: {len(value)} LoRAs")
            elif key == 'optimization_settings' and value:
                print(f"  optimization_settings: {list(value.keys())}")
        print()

def demo_memory_cleanup():
    """Demonstrate memory cleanup functionality"""
    print("=== Memory Cleanup Demo ===\n")
    
    # Get initial resource state
    initial_info = get_system_resource_info()
    print("Initial resource state:")
    print(f"  VRAM: {initial_info.vram.allocated_mb:.0f}MB allocated, {initial_info.vram.free_mb:.0f}MB free")
    print(f"  RAM: {initial_info.ram_usage_percent:.1f}% used")
    print()
    
    # Perform basic cleanup
    print("Performing basic memory cleanup...")
    cleanup_result = cleanup_memory(aggressive=False)
    
    print("Cleanup results:")
    print(f"  VRAM freed: {cleanup_result['vram_freed_mb']:.1f}MB")
    print(f"  RAM freed: {cleanup_result['ram_freed_mb']:.1f}MB")
    print(f"  Actions taken: {', '.join(cleanup_result['actions_taken'])}")
    
    # Get final resource state
    final_info = get_system_resource_info()
    print("\nFinal resource state:")
    print(f"  VRAM: {final_info.vram.allocated_mb:.0f}MB allocated, {final_info.vram.free_mb:.0f}MB free")
    print(f"  RAM: {final_info.ram_usage_percent:.1f}% used")
    print()

def demo_resource_monitoring():
    """Demonstrate resource monitoring capabilities"""
    print("=== Resource Monitoring Demo ===\n")
    
    # Get resource manager
    resource_manager = get_resource_manager()
    
    # Show current resource status
    status = resource_manager.get_resource_status()
    print(f"Current resource status: {status.value}")
    
    # Show resource history (if available)
    history = resource_manager.get_resource_history(last_n=5)
    if history:
        print(f"\nRecent resource history ({len(history)} entries):")
        for i, entry in enumerate(history):
            vram_usage = entry['vram']['utilization_percent']
            ram_usage = entry['ram_usage_percent']
            print(f"  Entry {i+1}: VRAM {vram_usage:.1f}%, RAM {ram_usage:.1f}%")
    else:
        print("\nNo resource history available yet")
    print()

def demo_integration_with_generation():
    """Demonstrate integration with generation pipeline"""
    print("=== Generation Pipeline Integration Demo ===\n")
    
    # Simulate a generation request
    generation_params = {
        "model_type": "t2v-A14B",
        "prompt": "A beautiful sunset over mountains",
        "resolution": "1080p",
        "steps": 50,
        "duration": 6,
        "guidance_scale": 9.0,
        "lora_config": {"style_lora": 0.8, "quality_lora": 0.6}
    }
    
    print("Simulated generation request:")
    for key, value in generation_params.items():
        if key != 'lora_config':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {len(value)} LoRAs")
    print()
    
    # Check if generation is feasible
    requirement = estimate_resource_requirements(
        model_type=generation_params["model_type"],
        resolution=generation_params["resolution"],
        steps=generation_params["steps"],
        duration=generation_params["duration"],
        lora_count=len(generation_params["lora_config"])
    )
    
    print("Resource requirements:")
    print(f"  VRAM needed: {requirement.vram_mb:.0f}MB")
    print(f"  RAM needed: {requirement.ram_mb:.0f}MB")
    print(f"  Estimated time: {requirement.estimated_time_seconds:.0f}s")
    print()
    
    # Check availability
    vram_available, vram_message = check_vram_availability(requirement.vram_mb)
    print(f"VRAM availability: {vram_message}")
    
    if not vram_available:
        print("\nApplying resource optimizations...")
        optimized_params, suggestions = optimize_parameters_for_resources(generation_params)
        
        if suggestions:
            print("Applied optimizations:")
            for suggestion in suggestions:
                print(f"  • {suggestion.parameter}: {suggestion.current_value} → {suggestion.suggested_value}")
                print(f"    Impact: {suggestion.impact}")
            
            # Re-check with optimized parameters
            optimized_requirement = estimate_resource_requirements(
                model_type=optimized_params["model_type"],
                resolution=optimized_params["resolution"],
                steps=optimized_params["steps"],
                duration=optimized_params["duration"],
                lora_count=len(optimized_params["lora_config"])
            )
            
            print(f"\nOptimized resource requirements:")
            print(f"  VRAM needed: {optimized_requirement.vram_mb:.0f}MB (saved {requirement.vram_mb - optimized_requirement.vram_mb:.0f}MB)")
            print(f"  RAM needed: {optimized_requirement.ram_mb:.0f}MB")
            print(f"  Estimated time: {optimized_requirement.estimated_time_seconds:.0f}s")
            
            vram_available_opt, vram_message_opt = check_vram_availability(optimized_requirement.vram_mb)
            print(f"  VRAM availability: {vram_message_opt}")
        else:
            print("No further optimizations possible")
    else:
        print("Generation can proceed with current parameters")
    print()

def main():
    """Run all demonstrations"""
    print("VRAM Optimization and Resource Management Demo")
    print("=" * 50)
    print()
    
    try:
        demo_basic_functionality()
        demo_resource_estimation()
        demo_parameter_optimization()
        demo_memory_cleanup()
        demo_resource_monitoring()
        demo_integration_with_generation()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()