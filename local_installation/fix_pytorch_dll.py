#!/usr/bin/env python3
"""
PyTorch DLL Fix Script for Windows
Addresses the common "DLL load failed while importing _C" error
"""

import os
import sys
import subprocess
import platform
import importlib.util
import shutil
import glob
import site

def check_system_requirements():
    """Check if system meets basic requirements"""
    print("=== System Requirements Check ===")
    
    # Check Windows version
    if platform.system() != "Windows":
        print("❌ This script is designed for Windows only")
        return False
    
    print(f"✅ Windows {platform.release()} detected")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Python 3.8+ required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
    
    return True

def check_pytorch_import():
    """Test if PyTorch can be imported"""
    print("\n=== PyTorch Import Test ===")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def fix_pytorch_installation():
    """Attempt to fix PyTorch installation"""
    print("\n=== PyTorch Installation Fix ===")
    
    try:
        # Force uninstall all PyTorch related packages
        print("Force uninstalling all PyTorch packages...")
        pytorch_packages = [
            "torch", "torchvision", "torchaudio", "torchtext", 
            "functorch", "torchdata", "torchserve", "torchx"
        ]
        
        for package in pytorch_packages:
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", "-y", package
            ], check=False, capture_output=True)
        
        # Clean up corrupted directories manually
        print("Cleaning up corrupted PyTorch directories...")
        import site
        site_packages = site.getsitepackages()[0]
        
        import shutil
        import glob
        
        # Remove any torch-related directories
        torch_dirs = glob.glob(os.path.join(site_packages, "*torch*"))
        torch_dirs.extend(glob.glob(os.path.join(site_packages, "~*torch*")))
        
        for torch_dir in torch_dirs:
            try:
                if os.path.exists(torch_dir):
                    shutil.rmtree(torch_dir, ignore_errors=True)
                    print(f"   Removed: {os.path.basename(torch_dir)}")
            except Exception as e:
                print(f"   Warning: Could not remove {torch_dir}: {e}")
        
        # Clear pip cache
        print("Clearing pip cache...")
        subprocess.run([
            sys.executable, "-m", "pip", "cache", "purge"
        ], check=False, capture_output=True)
        
        # Install PyTorch with CUDA 12.1
        print("Installing fresh PyTorch with CUDA 12.1 support...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1+cu121",
            "torchvision==0.20.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--force-reinstall", "--no-cache-dir"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ PyTorch installation failed: {result.stderr}")
            return False
        
        # Install other critical dependencies
        print("Installing other critical dependencies...")
        critical_deps = [
            "diffusers>=0.20.0,<0.30.0",
            "transformers>=4.30.0,<5.0.0",
            "accelerate>=0.20.0,<1.0.0",
            "huggingface-hub>=0.16.0,<1.0.0"
        ]
        
        for dep in critical_deps:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠️  Warning: Could not install {dep}")
            else:
                print(f"✅ Installed {dep.split('>=')[0]}")
        
        print("✅ PyTorch and dependencies installation completed")
        return True
        
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def check_visual_cpp_redist():
    """Check if Visual C++ Redistributable is installed"""
    print("\n=== Visual C++ Redistributable Check ===")
    
    # Common registry paths for VC++ Redist
    import winreg
    
    redist_keys = [
        r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        r"SOFTWARE\Classes\Installer\Dependencies\Microsoft.VS.VC_RuntimeMinimumVSU_amd64,v14"
    ]
    
    for key_path in redist_keys:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                print("✅ Visual C++ Redistributable found")
                return True
        except FileNotFoundError:
            continue
    
    print("❌ Visual C++ Redistributable not found")
    print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    return False

def main():
    """Main fix routine"""
    print("PyTorch DLL Fix Script")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        return False
    
    # Check VC++ Redistributable
    check_visual_cpp_redist()
    
    # Test current PyTorch
    if check_pytorch_import():
        print("\n✅ PyTorch is working correctly!")
        return True
    
    # Attempt fix
    print("\n🔧 Attempting to fix PyTorch installation...")
    if fix_pytorch_installation():
        # Test again
        if check_pytorch_import():
            print("\n✅ PyTorch fix successful!")
            return True
        else:
            print("\n❌ PyTorch fix failed - manual intervention required")
            return False
    else:
        print("\n❌ Could not fix PyTorch installation")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n" + "=" * 50)
        print("MANUAL FIX REQUIRED")
        print("=" * 50)
        print("1. Install Visual C++ Redistributable 2019-2022:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. Install CUDA 12.1 Runtime:")
        print("   https://developer.nvidia.com/cuda-12-1-0-download-archive")
        print("3. Reinstall PyTorch:")
        print("   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
        
    input("\nPress Enter to exit...")