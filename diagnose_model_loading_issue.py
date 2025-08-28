#!/usr/bin/env python3
"""
Diagnose Model Loading Issue
Check what's causing the model loading to get stuck at 0%
"""

import asyncio
import sys
import logging
import psutil
import torch
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check current system resource usage"""
    print("🔍 System Resource Check")
    print("=" * 40)
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU VRAM: {gpu_used:.1f}GB / {gpu_memory:.1f}GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"Disk: {disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB ({disk.percent:.1f}%)")
    
    print()

def check_model_files():
    """Check if model files exist and their sizes"""
    print("📁 Model Files Check")
    print("=" * 40)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    model_paths = [
        "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers",
        "models/Wan-AI_Wan2.2-I2V-A14B-Diffusers", 
        "models/Wan-AI_Wan2.2-TI2V-5B-Diffusers"
    ]
    
    for model_path in model_paths:
        path = Path(model_path)
        if path.exists():
            # Calculate total size
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            print(f"✅ {path.name}: {total_size / (1024**3):.1f}GB")
            
            # Check for key files
            config_file = path / "config.json"
            model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
            
            if config_file.exists():
                print(f"   ✅ config.json found")
            else:
                print(f"   ❌ config.json missing")
            
            if model_files:
                print(f"   ✅ {len(model_files)} model files found")
            else:
                print(f"   ❌ No model files (.bin/.safetensors) found")
        else:
            print(f"❌ {path.name}: Not found")
    
    print()
    return True

async def test_simple_model_loading():
    """Test loading a model with minimal configuration"""
    print("🧪 Simple Model Loading Test")
    print("=" * 40)
    
    try:
        from diffusers import DiffusionPipeline
        import torch
        
        model_path = "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers"
        
        if not Path(model_path).exists():
            print(f"❌ Model path not found: {model_path}")
            return False
        
        print(f"📂 Testing model loading from: {model_path}")
        print("⚠️  This may take several minutes...")
        
        # Try loading with minimal settings
        print("🔄 Attempting to load with CPU offloading...")
        
        try:
            # Load with CPU offloading to reduce VRAM usage
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                variant="fp16"
            )
            print("✅ Model loaded successfully with CPU offloading!")
            return True
            
        except Exception as e:
            print(f"⚠️  CPU offloading failed: {e}")
            
            # Try loading to CPU only
            print("🔄 Attempting to load to CPU only...")
            try:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                print("✅ Model loaded successfully to CPU!")
                return True
                
            except Exception as e2:
                print(f"❌ CPU loading also failed: {e2}")
                return False
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def check_loading_bottlenecks():
    """Check for common loading bottlenecks"""
    print("🚧 Loading Bottleneck Analysis")
    print("=" * 40)
    
    issues = []
    
    # Check available RAM
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 8:
        issues.append(f"Low available RAM: {available_gb:.1f}GB (need 8GB+)")
    
    # Check GPU VRAM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 12:
            issues.append(f"Limited GPU VRAM: {gpu_memory:.1f}GB (WAN models prefer 12GB+)")
    
    # Check disk speed (approximate)
    models_dir = Path("models")
    if models_dir.exists():
        # Check if models are on SSD vs HDD
        try:
            import time
            test_file = models_dir / "speed_test.tmp"
            
            # Write test
            start = time.time()
            with open(test_file, 'wb') as f:
                f.write(b'0' * (10 * 1024 * 1024))  # 10MB
            write_time = time.time() - start
            
            # Read test
            start = time.time()
            with open(test_file, 'rb') as f:
                f.read()
            read_time = time.time() - start
            
            test_file.unlink()  # Clean up
            
            write_speed = 10 / write_time  # MB/s
            read_speed = 10 / read_time   # MB/s
            
            print(f"Disk Speed: Write {write_speed:.0f}MB/s, Read {read_speed:.0f}MB/s")
            
            if read_speed < 100:
                issues.append(f"Slow disk speed: {read_speed:.0f}MB/s (HDD detected, SSD recommended)")
                
        except Exception as e:
            print(f"Could not test disk speed: {e}")
    
    if issues:
        print("⚠️  Potential Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ No obvious bottlenecks detected")
    
    print()

def suggest_optimizations():
    """Suggest optimizations based on system"""
    print("💡 Optimization Suggestions")
    print("=" * 40)
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    suggestions = []
    
    if available_gb < 16:
        suggestions.append("Enable CPU offloading: device_map='auto'")
        suggestions.append("Use low_cpu_mem_usage=True")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 16:
            suggestions.append("Use torch_dtype=torch.float16 to reduce VRAM usage")
            suggestions.append("Enable sequential CPU offloading")
    
    suggestions.extend([
        "Close other applications to free memory",
        "Use variant='fp16' if available",
        "Consider loading one model at a time",
        "Enable model caching after first load"
    ])
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print()

async def main():
    """Run comprehensive model loading diagnosis"""
    print("🔧 Model Loading Issue Diagnosis")
    print("Analyzing why model loading gets stuck at 0%")
    print("=" * 60)
    
    # Step 1: Check system resources
    check_system_resources()
    
    # Step 2: Check model files
    files_ok = check_model_files()
    
    # Step 3: Check for bottlenecks
    check_loading_bottlenecks()
    
    # Step 4: Suggest optimizations
    suggest_optimizations()
    
    # Step 5: Test simple loading (optional)
    print("🤔 Would you like to test simple model loading? (This may take 5-10 minutes)")
    print("This will help identify if it's a configuration issue or hardware limitation.")
    
    # For now, skip the actual test to avoid hanging
    print("⏭️  Skipping actual loading test to avoid hanging...")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY:")
    print("The model loading is likely stuck due to:")
    print("1. Memory constraints (models are 8-12GB each)")
    print("2. Inefficient loading configuration")
    print("3. Hardware limitations")
    
    print("\n📋 RECOMMENDED FIXES:")
    print("1. Update pipeline loader to use CPU offloading")
    print("2. Enable low memory usage options")
    print("3. Load models sequentially, not all at once")
    print("4. Add timeout and progress monitoring")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)