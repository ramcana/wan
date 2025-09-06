#!/usr/bin/env python3
"""
Quick test script to verify WAN model configuration system functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.services.model_manager import (
    get_model_manager,
    get_wan_model_capabilities,
    validate_wan_model_configuration,
    assess_hardware_compatibility,
    get_performance_profile,
    is_wan_model
)

def test_model_manager():
    """Test basic model manager functionality"""
    print("Testing Model Manager...")
    
    manager = get_model_manager()
    print(f"✅ Model Manager initialized")
    
    # Test model ID mapping
    for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        model_id = manager.get_model_id(model_type)
        print(f"✅ {model_type} -> {model_id}")
        
        # Test WAN model detection
        is_wan = is_wan_model(model_type)
        print(f"✅ {model_type} is WAN model: {is_wan}")
        
        # Test model status
        status = manager.get_model_status(model_type)
        print(f"✅ {model_type} status retrieved: {status['is_wan_model']}")

def test_wan_capabilities():
    """Test WAN model capabilities"""
    print("\nTesting WAN Model Capabilities...")
    
    for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        capabilities = get_wan_model_capabilities(model_type)
        
        if "error" not in capabilities:
            caps = capabilities["capabilities"]
            print(f"✅ {model_type}:")
            print(f"   Display Name: {caps['display_name']}")
            print(f"   Architecture: {caps['architecture_type']}")
            print(f"   Parameters: {caps.get('parameter_count', 'N/A')}")
            print(f"   Max Resolution: {caps['max_resolution']}")
            print(f"   Max Frames: {caps['max_frames']}")
        else:
            print(f"❌ {model_type}: {capabilities['error']}")

def test_validation():
    """Test model validation"""
    print("\nTesting Model Validation...")
    
    for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        is_valid, errors = validate_wan_model_configuration(model_type)
        
        if is_valid:
            print(f"✅ {model_type}: Valid")
        else:
            print(f"❌ {model_type}: Invalid - {errors}")

def test_hardware_compatibility():
    """Test hardware compatibility assessment"""
    print("\nTesting Hardware Compatibility...")
    
    for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        compat = assess_hardware_compatibility(model_type)
        
        if "error" not in compat:
            is_compatible = compat.get("is_compatible", False)
            vram_util = compat.get("vram_utilization", {})
            optimal_profile = compat.get("optimal_profile", "None")
            
            print(f"{'✅' if is_compatible else '❌'} {model_type}:")
            print(f"   Compatible: {is_compatible}")
            print(f"   VRAM Utilization: {vram_util.get('utilization_percent', 0):.1f}%")
            print(f"   Optimal Profile: {optimal_profile}")
        else:
            print(f"❌ {model_type}: {compat['error']}")

def test_performance_profile():
    """Test performance profiling"""
    print("\nTesting Performance Profiling...")
    
    for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        profile = get_performance_profile(model_type)
        
        if "error" not in profile:
            metrics = profile.get("estimated_metrics", {})
            settings = profile.get("optimization_settings", {})
            
            print(f"✅ {model_type}:")
            print(f"   Estimated Inference Time: {metrics.get('inference_time_seconds', 0):.1f}s")
            print(f"   VRAM Usage: {metrics.get('vram_usage_gb', 0):.1f} GB")
            print(f"   Recommended Precision: {settings.get('precision', 'N/A')}")
            print(f"   Batch Size: {settings.get('batch_size', 'N/A')}")
        else:
            print(f"❌ {model_type}: {profile['error']}")

def main():
    """Run all tests"""
    print("WAN Model Configuration System Test")
    print("=" * 50)
    
    try:
        test_model_manager()
        test_wan_capabilities()
        test_validation()
        test_hardware_compatibility()
        test_performance_profile()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()