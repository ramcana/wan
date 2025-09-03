#!/usr/bin/env python3
"""
Quick test script to verify PyTorch is working after the fix
"""

def test_pytorch_import():
    """Test PyTorch import and basic functionality"""
    try:
        print("Testing PyTorch import...")
        import torch
print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✅ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test basic tensor operations
        print("Testing tensor operations...")
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print(f"✅ CPU tensor operations working")
        
        if cuda_available:
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print(f"✅ GPU tensor operations working")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_other_dependencies():
    """Test other critical dependencies"""
    dependencies = [
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'),
        ('gradio', 'gradio'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('accelerate', 'accelerate'),
        ('huggingface_hub', 'huggingface-hub')
    ]
    
    print("\nTesting other dependencies...")
    failed_deps = []
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name} imported successfully")
        except ImportError:
            print(f"❌ {package_name} import failed")
            failed_deps.append(package_name)
        except Exception as e:
            print(f"⚠️  {package_name} import warning: {e}")
    
    return failed_deps

if __name__ == "__main__":
    print("PyTorch Fix Verification")
    print("=" * 40)
    
    if test_pytorch_import():
        print("\n✅ PyTorch is working correctly!")
        failed_deps = test_other_dependencies()
        
        if failed_deps:
            print(f"\n⚠️  Some dependencies failed: {', '.join(failed_deps)}")
            print("Installing missing dependencies...")
            import subprocess
import sys
            for dep in failed_deps:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
                    print(f"✅ Installed {dep}")
                except subprocess.CalledProcessError:
                    print(f"❌ Failed to install {dep}")
        
        print("\n🎉 Setup complete! You can now run the UI.")
    else:
        print("\n❌ PyTorch is still not working.")
        print("Please run: python local_installation/fix_pytorch_dll.py")
    
    input("\nPress Enter to exit...")