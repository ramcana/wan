#!/usr/bin/env python3
"""
Test CUDA detection to debug the issue
"""

def test_cuda_detection():
    print("🔍 Testing CUDA Detection")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully: {torch.__version__}")
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"🎮 CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"📊 GPU count: {device_count}")
            
            if device_count > 0:
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                print(f"🖥️  Current GPU: {gpu_name}")
                
                # Test memory
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                total_memory_gb = total_memory / (1024**3)
                print(f"💾 Total VRAM: {total_memory_gb:.1f}GB")
                
                # Test tensor creation
                try:
                    test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                    print(f"✅ GPU tensor creation successful: {test_tensor.device}")
                except Exception as e:
                    print(f"❌ GPU tensor creation failed: {e}")
            else:
                print("❌ No GPU devices found")
        else:
            print("❌ CUDA not available")
            
            # Check CUDA version
            try:
                cuda_version = torch.version.cuda
                print(f"🔧 PyTorch CUDA version: {cuda_version}")
            except:
                print("❌ Could not get CUDA version")
    
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    
    print("\n🔍 Testing Environment Variables")
    print("=" * 50)
    
    import os
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    print("\n🔍 Testing System GPU Detection")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi available")
            lines = result.stdout.split('\n')[:10]  # First 10 lines
            for line in lines:
                if 'RTX' in line or 'GeForce' in line:
                    print(f"🎮 Found: {line.strip()}")
        else:
            print("❌ nvidia-smi failed")
    except Exception as e:
        print(f"❌ nvidia-smi test failed: {e}")

if __name__ == "__main__":
    test_cuda_detection()