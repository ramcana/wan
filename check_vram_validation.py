#!/usr/bin/env python3
"""
Quick VRAM Validation Check
Tests if the VRAM validation error is resolved
"""

import torch
import json
from pathlib import Path

def check_cuda_status():
    """Check CUDA status and VRAM"""
    print("🔍 CUDA Status Check")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("   Suggestions:")
        print("   • Reinstall PyTorch with CUDA support")
        print("   • Check NVIDIA drivers")
        print("   • Verify GPU is properly connected")
        return False
    
    print("✅ CUDA Available")
    print(f"   Version: {torch.version.cuda}")
    print(f"   PyTorch: {torch.__version__}")
    
    gpu_props = torch.cuda.get_device_properties(0)
    vram_gb = gpu_props.total_memory / (1024**3)
    
    print(f"   GPU: {gpu_props.name}")
    print(f"   VRAM: {vram_gb:.1f}GB")
    print(f"   Compute: {gpu_props.major}.{gpu_props.minor}")
    
    return True

def check_vram_usage():
    """Check current VRAM usage"""
    print("\n💾 VRAM Usage Check")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("❌ Cannot check VRAM - CUDA not available")
        return False
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        
        total_gb = total_memory / (1024**3)
        used_gb = allocated_memory / (1024**3)
        free_gb = total_gb - used_gb
        
        print(f"✅ Total VRAM: {total_gb:.1f}GB")
        print(f"   Used: {used_gb:.2f}GB")
        print(f"   Free: {free_gb:.1f}GB")
        print(f"   Usage: {(used_gb/total_gb)*100:.1f}%")
        
        if free_gb >= 10:
            print("✅ Sufficient VRAM for generation")
        elif free_gb >= 6:
            print("⚠️ Moderate VRAM available - use lower resolutions")
        else:
            print("❌ Low VRAM available - may need optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ VRAM check failed: {e}")
        return False

def check_backend_config():
    """Check backend configuration"""
    print("\n⚙️ Backend Configuration Check")
    print("-" * 30)
    
    config_path = Path("backend/config.json")
    if not config_path.exists():
        print("❌ Backend config not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check generation mode
        gen_mode = config.get("generation", {}).get("mode", "unknown")
        if gen_mode == "real":
            print("✅ Generation mode: Real AI")
        else:
            print(f"⚠️ Generation mode: {gen_mode}")
        
        # Check VRAM settings
        vram_limit = config.get("hardware", {}).get("vram_limit_gb", 0)
        print(f"✅ VRAM limit: {vram_limit}GB")
        
        # Check offloading
        enable_offload = config.get("optimization", {}).get("enable_offload", True)
        if not enable_offload:
            print("✅ CPU offload: Disabled (good for high VRAM)")
        else:
            print("⚠️ CPU offload: Enabled (may not be needed)")
        
        # Check quantization
        quantization = config.get("optimization", {}).get("quantization_level", "none")
        print(f"✅ Quantization: {quantization}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False

def check_frontend_config():
    """Check frontend VRAM configuration"""
    print("\n🎨 Frontend Configuration Check")
    print("-" * 30)
    
    override_path = Path("frontend/gpu_override.json")
    if override_path.exists():
        try:
            with open(override_path, 'r') as f:
                config = json.load(f)
            
            gpu_name = config.get("gpu_name", "Unknown")
            total_vram = config.get("total_vram_gb", 0)
            usable_vram = config.get("usable_vram_gb", 0)
            
            print(f"✅ GPU Override: {gpu_name}")
            print(f"✅ Total VRAM: {total_vram:.1f}GB")
            print(f"✅ Usable VRAM: {usable_vram:.1f}GB")
            
            # Check VRAM estimates
            estimates = config.get("vram_estimates", {})
            if estimates:
                print("✅ VRAM Estimates:")
                for resolution, estimate in estimates.items():
                    if isinstance(estimate, dict):
                        base = estimate.get("base", 0)
                        print(f"   {resolution}: {base}GB base")
            
            return True
            
        except Exception as e:
            print(f"❌ Frontend config check failed: {e}")
            return False
    else:
        print("⚠️ No GPU override found - using default detection")
        return False

def simulate_vram_validation():
    """Simulate VRAM validation for common settings"""
    print("\n🧪 VRAM Validation Simulation")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("❌ Cannot simulate - CUDA not available")
        return False
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Test scenarios
    scenarios = [
        ("1280x720, 4s", 6.0),
        ("1920x1080, 4s", 10.0),
        ("1920x1080, 8s", 12.0),
        ("2560x1440, 4s", 14.0),
        ("2560x1440, 8s", 16.0)
    ]
    
    print(f"Available VRAM: {total_vram:.1f}GB")
    print("Validation Results:")
    
    for scenario, required_vram in scenarios:
        if required_vram <= total_vram - 2:  # 2GB buffer
            status = "✅ PASS"
        elif required_vram <= total_vram:
            status = "⚠️ TIGHT"
        else:
            status = "❌ FAIL"
        
        print(f"   {scenario}: {required_vram:.1f}GB - {status}")
    
    return True

def main():
    """Main check function"""
    print("🔍 VRAM VALIDATION CHECK")
    print("=" * 50)
    
    checks = [
        check_cuda_status,
        check_vram_usage,
        check_backend_config,
        check_frontend_config,
        simulate_vram_validation
    ]
    
    passed = 0
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"❌ Check failed: {e}")
    
    print(f"\n📋 SUMMARY: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("🎉 All checks passed! VRAM validation should work correctly.")
        print("\n💡 Next steps:")
        print("1. Start servers: start_optimized_rtx4080.bat")
        print("2. Test generation with 1920x1080, 4 seconds")
        print("3. Monitor VRAM usage during generation")
    elif passed >= 3:
        print("✅ Most checks passed. System should work with minor issues.")
    else:
        print("⚠️ Several issues found. Check configuration and dependencies.")
    
    print(f"\n🔧 If you still see 'Insufficient VRAM' errors:")
    print("• Restart both frontend and backend servers")
    print("• Clear browser cache and refresh")
    print("• Check that gpu_override.json is being loaded")
    print("• Verify PyTorch CUDA installation")

if __name__ == "__main__":
    main()