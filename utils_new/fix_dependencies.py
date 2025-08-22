#!/usr/bin/env python3
"""
Fix dependencies for CogVideoX models
"""

import subprocess
import sys
import importlib

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run {command}: {e}")
        return False

def check_package_version(package_name):
    """Check if package is installed and get version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True, version
    except ImportError:
        print(f"❌ {package_name}: not installed")
        return False, None

def main():
    """Main dependency fixing function"""
    print("CogVideoX Dependency Fixer")
    print("=" * 40)
    
    # Check current versions
    print("\nCurrent package versions:")
    packages = ['torch', 'transformers', 'diffusers', 'accelerate', 'safetensors']
    
    for package in packages:
        check_package_version(package)
    
    print("\nUpdating packages to compatible versions...")
    
    # Update commands
    update_commands = [
        "pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install --upgrade transformers>=4.44.0",
        "pip install --upgrade diffusers>=0.30.0",
        "pip install --upgrade accelerate>=0.33.0",
        "pip install --upgrade safetensors",
        "pip install --upgrade pillow numpy"
    ]
    
    for command in update_commands:
        print(f"\nRunning: {command}")
        run_command(command)
    
    print("\n" + "=" * 40)
    print("Verifying updated versions:")
    
    for package in packages:
        check_package_version(package)
    
    print("\n" + "=" * 40)
    print("Testing CogVideoX import...")
    
    try:
        from diffusers import CogVideoXPipeline
        print("✅ CogVideoXPipeline import successful")
        
        from transformers import T5Tokenizer, T5EncoderModel
        print("✅ T5 components import successful")
        
        print("\n🎉 All dependencies are now properly configured!")
        print("You can now run: python test_ti2v_5b_fixed.py")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Restart your Python environment")
        print("2. Clear pip cache: pip cache purge")
        print("3. Try installing in a fresh virtual environment")

if __name__ == "__main__":
    main()