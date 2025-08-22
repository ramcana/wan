#!/usr/bin/env python3
"""
Quick check for all missing modules in the WAN2.2 Gradio UI
"""

import sys
import subprocess
import importlib.util

def check_module(module_name, import_name=None):
    """Check if a module is available"""
    if import_name is None:
        import_name = module_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True, "✓ Available"
        else:
            return False, "✗ Missing"
    except (ImportError, ModuleNotFoundError, ValueError):
        return False, "✗ Missing (ImportError)"

def install_module(module_name):
    """Install a module using pip"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", module_name], 
                      check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Check and install missing modules"""
    
    # List of modules to check (module_name, import_name, pip_name)
    modules_to_check = [
        ("torch", "torch", "torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121"),
        ("torchvision", "torchvision", "torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121"),
        ("transformers", "transformers", "transformers==4.37.2"),
        ("diffusers", "diffusers", "diffusers==0.25.0"),
        ("accelerate", "accelerate", "accelerate"),
        ("huggingface_hub", "huggingface_hub", "huggingface_hub==0.25.0"),
        ("gradio", "gradio", "gradio==4.44.0"),
        ("PIL", "PIL", "Pillow"),
        ("cv2", "cv2", "opencv-python"),
        ("imageio", "imageio", "imageio"),
        ("imageio_ffmpeg", "imageio_ffmpeg", "imageio-ffmpeg"),
        ("psutil", "psutil", "psutil"),
        ("GPUtil", "GPUtil", "GPUtil"),
        ("pynvml", "pynvml", "pynvml"),
        ("numpy", "numpy", "numpy"),
        ("pandas", "pandas", "pandas"),
        ("yaml", "yaml", "pyyaml"),
        ("tqdm", "tqdm", "tqdm"),
        ("bitsandbytes", "bitsandbytes", "bitsandbytes"),
        ("safetensors", "safetensors", "safetensors"),
    ]
    
    print("🔍 Checking WAN2.2 Gradio UI Dependencies")
    print("=" * 50)
    
    missing_modules = []
    available_modules = []
    
    for module_name, import_name, pip_name in modules_to_check:
        available, status = check_module(module_name, import_name)
        print(f"{status} {module_name} ({import_name})")
        
        if available:
            available_modules.append((module_name, import_name, pip_name))
        else:
            missing_modules.append((module_name, import_name, pip_name))
    
    print("\n📊 Summary:")
    print(f"Available: {len(available_modules)}")
    print(f"Missing: {len(missing_modules)}")
    
    if missing_modules:
        print(f"\n❌ Missing modules:")
        for module_name, import_name, pip_name in missing_modules:
            print(f"  - {module_name} (pip install {pip_name})")
        
        print(f"\n🔧 Installing missing modules...")
        
        for module_name, import_name, pip_name in missing_modules:
            print(f"Installing {module_name}...")
            cmd = f"pip install {pip_name}"
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print(f"  ✓ {module_name} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to install {module_name}: {e.stderr}")
    
    else:
        print("\n✅ All modules are available!")
    
    print(f"\n🧪 Testing critical imports...")
    
    # Test critical imports
    critical_imports = [
        ("torch", "import torch; print(f'PyTorch: {torch.__version__}')"),
        ("diffusers", "from diffusers import DiffusionPipeline; print('Diffusers: OK')"),
        ("gradio", "import gradio as gr; print(f'Gradio: {gr.__version__}')"),
        ("transformers", "import transformers; print(f'Transformers: {transformers.__version__}')"),
    ]
    
    for name, test_code in critical_imports:
        try:
            exec(test_code)
        except Exception as e:
            print(f"✗ {name} test failed: {e}")

if __name__ == "__main__":
    main()