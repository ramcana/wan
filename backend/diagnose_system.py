from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Quick system diagnostic to identify what's preventing real AI generation
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def diagnose_system():
    print("🔍 DIAGNOSING SYSTEM FOR REAL AI GENERATION")
    print("=" * 50)
    
    issues = []
    
    # 1. Check System Integration
    print("\n1. 🔧 Checking System Integration...")
    try:
        from core.system_integration import SystemIntegration
        integration = SystemIntegration()
        await integration.initialize()
        
        status = await integration.get_system_status()
        if status.get("initialized"):
            print("   ✅ System Integration: OK")
        else:
            print("   ❌ System Integration: Not initialized")
            issues.append("System integration not initialized")
            
    except Exception as e:
        print(f"   ❌ System Integration: ERROR - {e}")
        issues.append(f"System integration error: {e}")
    
    # 2. Check Model Bridge
    print("\n2. 🌉 Checking Model Integration Bridge...")
    try:
        model_bridge = await integration.get_model_bridge()
        if model_bridge:
            print("   ✅ Model Bridge: Available")
            
            # Check model status
            model_status = model_bridge.get_system_model_status()
            print(f"   ℹ️  Model Status: {len(model_status)} models tracked")
            
        else:
            print("   ❌ Model Bridge: Not available")
            issues.append("Model integration bridge not available")
            
    except Exception as e:
        print(f"   ❌ Model Bridge: ERROR - {e}")
        issues.append(f"Model bridge error: {e}")
    
    # 3. Check WAN22 Infrastructure
    print("\n3. 🏗️ Checking WAN22 Infrastructure...")
    try:
        optimizer = await integration.get_system_optimizer()
        if optimizer:
            print("   ✅ WAN22 System Optimizer: Available")
        else:
            print("   ❌ WAN22 System Optimizer: Not available")
            issues.append("WAN22 system optimizer not available")
            
    except Exception as e:
        print(f"   ❌ WAN22 Infrastructure: ERROR - {e}")
        issues.append(f"WAN22 infrastructure error: {e}")
    
    # 4. Check Model Availability
    print("\n4. 🤖 Checking Model Availability...")
    try:
        if model_bridge:
            model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            available_models = 0
            
            for model_type in model_types:
                try:
                    available = model_bridge.check_model_availability(model_type)
                    if available:
                        print(f"   ✅ {model_type}: Available")
                        available_models += 1
                    else:
                        print(f"   ⚠️  {model_type}: Not available (will auto-download)")
                except Exception as e:
                    print(f"   ❌ {model_type}: Error - {e}")
            
            if available_models == 0:
                issues.append("No AI models are currently available")
                print("   ℹ️  Note: Models will be downloaded automatically on first use")
        else:
            print("   ❌ Cannot check models - Model bridge not available")
            
    except Exception as e:
        print(f"   ❌ Model Availability: ERROR - {e}")
        issues.append(f"Model availability check error: {e}")
    
    # 5. Check Hardware Requirements
    print("\n5. 💻 Checking Hardware Requirements...")
    try:
        import psutil
        
        # RAM check
        memory = psutil.virtual_memory()
        ram_gb = memory.total / (1024**3)
        available_ram_gb = memory.available / (1024**3)
        
        if ram_gb >= 8:
            print(f"   ✅ RAM: {ram_gb:.1f}GB total ({available_ram_gb:.1f}GB available)")
        else:
            print(f"   ⚠️  RAM: {ram_gb:.1f}GB total (8GB+ recommended)")
            issues.append(f"Low RAM: {ram_gb:.1f}GB (8GB+ recommended)")
        
        # Disk space check
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        
        if disk_gb >= 50:
            print(f"   ✅ Disk Space: {disk_gb:.1f}GB available")
        else:
            print(f"   ⚠️  Disk Space: {disk_gb:.1f}GB available (50GB+ recommended for models)")
            issues.append(f"Low disk space: {disk_gb:.1f}GB")
        
        # GPU check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   ✅ GPU: {gpu_count} CUDA GPU(s), {gpu_memory:.1f}GB VRAM")
            else:
                print("   ⚠️  GPU: No CUDA GPU available (CPU generation will be very slow)")
                issues.append("No CUDA GPU available")
        except ImportError:
            print("   ⚠️  GPU: PyTorch not available for GPU detection")
            issues.append("PyTorch not available")
            
    except Exception as e:
        print(f"   ❌ Hardware Check: ERROR - {e}")
        issues.append(f"Hardware check error: {e}")
    
    # 6. Check Configuration
    print("\n6. ⚙️ Checking Configuration...")
    try:
        config_path = Path("config.json")
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            gen_config = config.get("generation", {})
            mode = gen_config.get("mode", "unknown")
            
            if mode == "real":
                print("   ✅ Generation Mode: Real AI generation enabled")
            elif mode == "mock":
                print("   ⚠️  Generation Mode: Mock generation (change to 'real' for AI generation)")
                issues.append("Generation mode set to 'mock'")
            else:
                print(f"   ❌ Generation Mode: Unknown mode '{mode}'")
                issues.append(f"Invalid generation mode: {mode}")
                
        else:
            print("   ⚠️  Configuration: config.json not found (using defaults)")
            issues.append("No configuration file found")
            
    except Exception as e:
        print(f"   ❌ Configuration: ERROR - {e}")
        issues.append(f"Configuration error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if not issues:
        print("🎉 NO ISSUES FOUND!")
        print("✅ System should be ready for real AI generation")
        print("\n💡 If generation is still using mock mode:")
        print("   1. Try submitting a generation request")
        print("   2. Check that models download successfully")
        print("   3. Monitor logs for specific errors")
    else:
        print(f"⚠️  FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        
        if "No AI models are currently available" in issues:
            print("   • Models will download automatically on first generation request")
            print("   • Ensure good internet connection for model downloads")
        
        if "No CUDA GPU available" in issues:
            print("   • Install CUDA-compatible PyTorch for GPU acceleration")
            print("   • CPU generation will work but be much slower")
        
        if any("RAM" in issue for issue in issues):
            print("   • Close other applications to free up RAM")
            print("   • Consider upgrading system RAM")
        
        if "Generation mode set to 'mock'" in issues:
            print("   • Run: python scripts/migrate_to_real_generation.py")
            print("   • Or manually set generation.mode to 'real' in config.json")
        
        if any("error" in issue.lower() for issue in issues):
            print("   • Check logs for detailed error messages")
            print("   • Ensure all dependencies are installed")
    
    print(f"\n🚀 NEXT STEPS:")
    print("   1. Address any issues listed above")
    print("   2. Try submitting a generation request")
    print("   3. Monitor logs during generation")
    print("   4. Check performance dashboard: /api/v1/performance/status")

if __name__ == "__main__":
    asyncio.run(diagnose_system())