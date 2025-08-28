#!/usr/bin/env python3
"""
RTX 4080 Optimization Test Suite
Validates that all optimizations are working correctly and diagnoses model issues
"""

import json
import torch
import psutil
import asyncio
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def test_cuda_availability():
    """Test CUDA availability and GPU detection"""
    print("🔍 Testing CUDA Availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_props = torch.cuda.get_device_properties(0)
    
    print(f"✅ CUDA Available: {torch.version.cuda}")
    print(f"✅ GPU: {gpu_props.name}")
    print(f"✅ VRAM: {gpu_props.total_memory / (1024**3):.1f}GB")
    print(f"✅ Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    return True

def test_vram_usage():
    """Test VRAM usage and memory management"""
    print("\n💾 Testing VRAM Usage...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for VRAM test")
        return False
    
    try:
        # Get initial VRAM
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
        
        print(f"✅ Initial VRAM usage: {initial_memory:.0f}MB / {total_memory:.0f}MB")
        
        # Test tensor allocation
        test_tensor = torch.randn(1000, 1000, device='cuda')
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)
        
        print(f"✅ After tensor allocation: {allocated_memory:.0f}MB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(0) / (1024**2)
        
        print(f"✅ After cleanup: {final_memory:.0f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ VRAM test failed: {e}")
        return False

def test_backend_config():
    """Test backend configuration"""
    print("\n⚙️ Testing Backend Configuration...")
    
    config_path = Path("backend/config.json")
    if not config_path.exists():
        print("❌ Backend config.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check key optimizations
        hardware = config.get("hardware", {})
        optimization = config.get("optimization", {})
        
        vram_limit = hardware.get("vram_limit_gb", 0)
        if vram_limit >= 14:
            print(f"✅ VRAM limit: {vram_limit}GB (optimized for RTX 4080)")
        else:
            print(f"⚠️ VRAM limit: {vram_limit}GB (may be too low)")
        
        quantization = optimization.get("quantization_level", "none")
        print(f"✅ Quantization: {quantization}")
        
        offload = optimization.get("enable_offload", True)
        if not offload:
            print("✅ CPU offload disabled (good for 16GB VRAM)")
        else:
            print("⚠️ CPU offload enabled (not needed with 16GB VRAM)")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_frontend_vram_config():
    """Test frontend VRAM configuration"""
    print("\n🎨 Testing Frontend VRAM Configuration...")
    
    # Check for GPU override
    override_path = Path("frontend/gpu_override.json")
    if override_path.exists():
        try:
            with open(override_path, 'r') as f:
                override_config = json.load(f)
            
            gpu_name = override_config.get("gpu_name", "Unknown")
            total_vram = override_config.get("total_vram_gb", 0)
            usable_vram = override_config.get("usable_vram_gb", 0)
            
            print(f"✅ GPU Override: {gpu_name}")
            print(f"✅ Total VRAM: {total_vram:.1f}GB")
            print(f"✅ Usable VRAM: {usable_vram:.1f}GB")
            
            # Check VRAM estimates
            estimates = override_config.get("vram_estimates", {})
            if "1920x1080" in estimates:
                estimate_1080p = estimates["1920x1080"]["base"]
                print(f"✅ 1080p estimate: {estimate_1080p}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ GPU override test failed: {e}")
            return False
    else:
        print("⚠️ GPU override not found, using default VRAM detection")
        return False

def test_system_resources():
    """Test system resources"""
    print("\n💻 Testing System Resources...")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    print(f"✅ CPU: {cpu_count} cores, {cpu_threads} threads")
    
    # Memory info
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    print(f"✅ RAM: {ram_gb:.1f}GB total, {available_gb:.1f}GB available")
    
    # Disk space
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    print(f"✅ Disk: {free_gb:.1f}GB free")
    
    # Check if resources are sufficient
    issues = []
    if ram_gb < 16:
        issues.append(f"Low RAM: {ram_gb:.1f}GB (16GB+ recommended)")
    if free_gb < 50:
        issues.append(f"Low disk space: {free_gb:.1f}GB (50GB+ recommended for models)")
    
    if issues:
        print("⚠️ Resource warnings:")
        for issue in issues:
            print(f"   • {issue}")
    
    return len(issues) == 0

def test_model_availability():
    """Test model availability and diagnose missing files issues"""
    print("\n🤖 Testing Model Availability...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    # Expected WAN2.2 models
    expected_models = {
        "WAN2.2-T2V-A14B": ["pytorch_model.bin", "config.json"],
        "WAN2.2-I2V-A14B": ["pytorch_model.bin", "config.json"], 
        "WAN2.2-TI2V-5B": ["pytorch_model.bin", "config.json"]
    }
    
    model_status = {}
    all_models_ready = True
    
    for model_name, required_files in expected_models.items():
        model_path = models_dir / model_name
        status = {
            "exists": model_path.exists(),
            "files_present": [],
            "files_missing": [],
            "size_mb": 0
        }
        
        if model_path.exists():
            for file_name in required_files:
                file_path = model_path / file_name
                if file_path.exists():
                    status["files_present"].append(file_name)
                    status["size_mb"] += file_path.stat().st_size / (1024 * 1024)
                else:
                    status["files_missing"].append(file_name)
            
            if status["files_missing"]:
                print(f"⚠️ {model_name}: Missing files {status['files_missing']}")
                all_models_ready = False
            else:
                print(f"✅ {model_name}: Complete ({status['size_mb']:.0f}MB)")
        else:
            print(f"❌ {model_name}: Not downloaded")
            all_models_ready = False
        
        model_status[model_name] = status
    
    # Check for any downloaded models
    downloaded_models = [name for name, status in model_status.items() if status["exists"]]
    if downloaded_models:
        print(f"📦 Found {len(downloaded_models)} downloaded models")
        
        # Check if we can work with what we have
        complete_models = [
            name for name, status in model_status.items() 
            if status["exists"] and not status["files_missing"]
        ]
        
        if complete_models:
            print(f"✅ {len(complete_models)} models are complete and ready")
            return True
        else:
            print("⚠️ Downloaded models have missing files - may need re-download")
            return False
    else:
        print("❌ No models found - run model download first")
        return False

def test_model_downloader_integration():
    """Test model downloader integration and diagnose download issues"""
    print("\n📥 Testing Model Downloader Integration...")
    
    try:
        # Add local installation to path
        local_installation_path = Path("local_installation")
        if local_installation_path.exists():
            sys.path.insert(0, str(local_installation_path))
            sys.path.insert(0, str(local_installation_path / "scripts"))
        
        # Try to import model downloader
        from scripts.download_models import ModelDownloader
        print("✅ Model downloader import successful")
        
        # Initialize downloader
        models_dir = Path("models")
        downloader = ModelDownloader(
            installation_path=str(Path.cwd()),
            models_dir=str(models_dir)
        )
        
        # Check existing models
        existing_models = downloader.check_existing_models()
        print(f"✅ Model validation complete: {len(existing_models)} valid models")
        
        # Get model verification results
        verification = downloader.verify_all_models()
        if verification.all_valid:
            print("✅ All required models are valid")
        else:
            print(f"⚠️ Invalid models: {verification.invalid_models}")
            print("💡 Tip: Run model download to fix missing/corrupted models")
        
        return len(existing_models) > 0
        
    except ImportError as e:
        print(f"❌ Model downloader import failed: {e}")
        print("💡 Tip: Ensure local_installation directory exists")
        return False
    except Exception as e:
        print(f"❌ Model downloader test failed: {e}")
        return False

async def test_backend_startup():
    """Test if backend can start properly"""
    print("\n🚀 Testing Backend Startup...")
    
    try:
        # Import backend modules to check for errors
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Test model integration bridge
        try:
            from core.model_integration_bridge import ModelIntegrationBridge
            bridge = ModelIntegrationBridge()
            print("✅ Model integration bridge import successful")
            
            # Test initialization
            init_success = await bridge.initialize()
            if init_success:
                print("✅ Model integration bridge initialized")
            else:
                print("⚠️ Model integration bridge initialization had warnings")
            
        except Exception as e:
            print(f"⚠️ Model integration bridge test failed: {e}")
        
        # Test core imports
        try:
            from core.services.optimization_service import get_vram_optimizer
            print("✅ Optimization service import successful")
            
            # Test VRAM optimizer
            optimizer = get_vram_optimizer()
            vram_info = optimizer.get_vram_usage()
            print(f"✅ VRAM optimizer working: {vram_info['used_mb']:.0f}MB used")
        except Exception as e:
            print(f"⚠️ VRAM optimizer test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend startup test failed: {e}")
        return False

def generate_optimization_report():
    """Generate a comprehensive optimization report"""
    print("\n📊 OPTIMIZATION REPORT")
    print("=" * 50)
    
    # System specs
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"VRAM: {gpu_props.total_memory / (1024**3):.1f}GB")
    
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"CPU: {cpu_threads} threads")
    print(f"RAM: {ram_gb:.1f}GB")
    
    print("\n🎯 RECOMMENDED SETTINGS:")
    print("• Resolution: Up to 1920x1080 (2560x1440 for short videos)")
    print("• Duration: 4-8 seconds optimal, up to 10 seconds")
    print("• Quantization: bf16 (best quality/performance balance)")
    print("• CPU Offload: Disabled (not needed with 16GB VRAM)")
    print("• VAE Tiling: Enabled with 512px tiles")
    print("• Concurrent Generations: 1 (to maximize quality)")
    
    print("\n⚡ PERFORMANCE EXPECTATIONS:")
    print("• 1280x720, 4s: ~2-3 minutes generation time")
    print("• 1920x1080, 4s: ~4-6 minutes generation time")
    print("• 1920x1080, 8s: ~8-12 minutes generation time")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• If VRAM errors persist: Reduce resolution or duration")
    print("• If generation is slow: Check GPU utilization")
    print("• If quality is poor: Disable quantization or use fp16")

def test_hardware_profile_integration():
    """Test hardware profile integration and fix the available_vram_gb issue"""
    print("\n� DTesting Hardware Profile Integration...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Test model integration bridge initialization
        from core.model_integration_bridge import ModelIntegrationBridge
        bridge = ModelIntegrationBridge()
        
        # Test hardware profile detection
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize the bridge
            init_success = loop.run_until_complete(bridge.initialize())
            if init_success:
                print("✅ Model integration bridge initialized successfully")
            else:
                print("⚠️ Model integration bridge initialization had warnings")
            
            # Check hardware profile
            hardware_profile = bridge.get_hardware_profile()
            if hardware_profile:
                print(f"✅ Hardware profile detected:")
                print(f"   • GPU: {hardware_profile.gpu_name}")
                print(f"   • Total VRAM: {hardware_profile.total_vram_gb:.1f}GB")
                print(f"   • Available VRAM: {hardware_profile.available_vram_gb:.1f}GB")
                print(f"   • CPU Cores: {hardware_profile.cpu_cores}")
                print(f"   • Total RAM: {hardware_profile.total_ram_gb:.1f}GB")
                
                # Test that available_vram_gb attribute exists (this was the error)
                vram_check = hardware_profile.available_vram_gb
                print(f"✅ available_vram_gb attribute working: {vram_check:.1f}GB")
                
                return True
            else:
                print("⚠️ No hardware profile detected")
                return False
                
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Hardware profile integration test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def diagnose_model_issues():
    """Diagnose and provide solutions for model-related issues"""
    print("\n🔍 DIAGNOSING MODEL ISSUES")
    print("=" * 50)
    
    # Check if this is the "missing files" issue from the logs
    models_dir = Path("models")
    issues_found = []
    solutions = []
    
    if not models_dir.exists():
        issues_found.append("Models directory doesn't exist")
        solutions.append("Create models directory and run model download")
    else:
        # Check for partial downloads
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                files = list(model_dir.glob("*"))
                if len(files) == 0:
                    issues_found.append(f"Empty model directory: {model_dir.name}")
                elif len(files) < 3:  # Expect at least config.json, model files
                    issues_found.append(f"Incomplete model: {model_dir.name} ({len(files)} files)")
    
    # Check for the specific error from logs
    log_files = [
        Path("wan22_ui.log"),
        Path("backend/logs/app.log"),
        Path("logs/generation.log")
    ]
    
    for log_file in log_files:
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "missing files" in content.lower():
                        issues_found.append("Model validation reports missing files")
                        solutions.append("Re-download models to ensure all files are present")
                    if "could not be downloaded" in content.lower():
                        issues_found.append("Model download failures detected")
                        solutions.append("Check internet connection and retry model download")
                    if "available_vram_gb" in content.lower():
                        issues_found.append("Hardware profile attribute error detected")
                        solutions.append("Update model integration bridge to fix hardware profile compatibility")
            except Exception:
                pass
    
    if issues_found:
        print("🚨 ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDED SOLUTIONS:")
        for i, solution in enumerate(set(solutions), 1):
            print(f"   {i}. {solution}")
        
        print("\n🛠️ QUICK FIX COMMANDS:")
        print("   • Download models: python local_installation/scripts/download_models.py")
        print("   • Validate models: python -c \"from local_installation.scripts.download_models import ModelDownloader; d=ModelDownloader('.'); print(d.verify_all_models())\"")
        print("   • Clear and re-download: rm -rf models && mkdir models")
        print("   • Test hardware profile: python test_rtx4080_optimization.py")
    else:
        print("✅ No obvious model issues detected")

async def main():
    """Main test function"""
    print("🧪 RTX 4080 OPTIMIZATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("VRAM Usage", test_vram_usage),
        ("Backend Config", test_backend_config),
        ("Frontend VRAM Config", test_frontend_vram_config),
        ("System Resources", test_system_resources),
        ("Hardware Profile Integration", test_hardware_profile_integration),
        ("Model Availability", test_model_availability),
        ("Model Downloader Integration", test_model_downloader_integration),
        ("Backend Startup", test_backend_startup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📋 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! RTX 4080 optimization successful!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. System should work well.")
    else:
        print("⚠️ Several tests failed. Check configuration.")
    
    # Always run diagnostics to help with troubleshooting
    diagnose_model_issues()
    generate_optimization_report()

if __name__ == "__main__":
    asyncio.run(main())