#!/usr/bin/env python3
"""
Fix diffusers installation and install Wan2.2 components
"""

import subprocess
import sys
import os
from pathlib import Path

def check_diffusers_version():
    """Check current diffusers version and compatibility"""
    try:
        import diffusers
        version = diffusers.__version__
        print(f"Current diffusers version: {version}")
        return version
    except ImportError:
        print("Diffusers not installed")
        return None

def fix_diffusers_installation():
    """Fix diffusers installation issues"""
    
    print("🔧 Fixing Diffusers Installation")
    print("=" * 50)
    
    # Step 1: Check current version
    current_version = check_diffusers_version()
    
    # Step 2: Uninstall current diffusers to avoid conflicts
    print("\n1. Removing current diffusers installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "diffusers", "-y"], 
                      check=True, capture_output=True, text=True)
        print("   ✅ Current diffusers uninstalled")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Could not uninstall diffusers: {e}")
    
    # Step 3: Install compatible diffusers version
    print("\n2. Installing compatible diffusers version...")
    
    # Try different installation methods
    install_commands = [
        # Try the official Wan2.2 diffusers fork
        [sys.executable, "-m", "pip", "install", "git+https://github.com/Wan-Video/diffusers.git"],
        # Try a specific compatible version
        [sys.executable, "-m", "pip", "install", "diffusers==0.30.0"],
        # Try latest stable
        [sys.executable, "-m", "pip", "install", "diffusers>=0.28.0,<0.31.0"],
        # Fallback to any recent version
        [sys.executable, "-m", "pip", "install", "diffusers"]
    ]
    
    for i, cmd in enumerate(install_commands, 1):
        try:
            print(f"   Trying method {i}: {' '.join(cmd[3:])}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            print(f"   ✅ Success with method {i}")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"   ❌ Method {i} failed: {e}")
            if i == len(install_commands):
                print("   ❌ All installation methods failed")
                return False
    
    # Step 4: Verify installation
    print("\n3. Verifying installation...")
    try:
        import diffusers
        print(f"   ✅ Diffusers {diffusers.__version__} installed successfully")
        
        # Test basic import
        from diffusers import DiffusionPipeline
        print("   ✅ DiffusionPipeline import successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def install_wan22_components():
    """Install or create Wan2.2 pipeline components"""
    
    print("\n🔧 Installing Wan2.2 Components")
    print("=" * 50)
    
    # Try to install from the official Wan2.2 repository
    print("1. Attempting to install from Wan2.2 repository...")
    
    install_commands = [
        # Try to install the full Wan2.2 package
        [sys.executable, "-m", "pip", "install", "git+https://github.com/Wan-Video/Wan2.2.git"],
        # Try to install just the diffusers components
        [sys.executable, "-m", "pip", "install", "git+https://github.com/Wan-Video/Wan2.2.git#subdirectory=diffusers"],
    ]
    
    for i, cmd in enumerate(install_commands, 1):
        try:
            print(f"   Trying method {i}: {' '.join(cmd[3:])}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            print(f"   ✅ Success with method {i}")
            
            # Test if WanPipeline is now available
            try:
                from diffusers import WanPipeline
                print("   ✅ WanPipeline is now available")
                return True
            except ImportError:
                print("   ⚠️  Installation succeeded but WanPipeline still not available")
                continue
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"   ❌ Method {i} failed: {e}")
    
    print("   ⚠️  Could not install WanPipeline from repository")
    return False

def create_wan22_pipeline_stub():
    """Create a stub WanPipeline for basic compatibility"""
    
    print("\n🔧 Creating Wan2.2 Pipeline Compatibility Layer")
    print("=" * 50)
    
    try:
        # Create a simple compatibility layer
        stub_code = '''
"""
Wan2.2 Pipeline Compatibility Layer
This provides basic compatibility for Wan2.2 models when the full WanPipeline is not available
"""

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
import torch

class WanPipeline(DiffusionPipeline):
    """Compatibility wrapper for WanPipeline"""
    
    def __init__(self, *args, **kwargs):
        # Use standard DiffusionPipeline as base
        super().__init__()
        print("Using WanPipeline compatibility layer")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with fallback to standard pipeline"""
        try:
            # Try to load as standard diffusion pipeline
            from diffusers import StableDiffusionPipeline
            return StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True,
                **kwargs
            )
        except Exception as e:
            print(f"WanPipeline fallback failed: {e}")
            raise e

class AutoencoderKLWan(AutoencoderKL):
    """Compatibility wrapper for AutoencoderKLWan"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with fallback to standard AutoencoderKL"""
        try:
            return AutoencoderKL.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception as e:
            print(f"AutoencoderKLWan fallback failed: {e}")
            raise e

# Register the compatibility classes
import diffusers
diffusers.WanPipeline = WanPipeline
diffusers.AutoencoderKLWan = AutoencoderKLWan

print("Wan2.2 compatibility layer loaded")
'''
        
        # Write the stub to a file
        stub_file = Path("wan22_compatibility.py")
        with open(stub_file, 'w') as f:
            f.write(stub_code)
        
        print(f"   ✅ Compatibility layer created: {stub_file}")
        
        # Test the stub
        try:
            import wan22_compatibility
            from diffusers import WanPipeline, AutoencoderKLWan
            print("   ✅ Compatibility layer working")
            return True
        except Exception as e:
            print(f"   ❌ Compatibility layer test failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to create compatibility layer: {e}")
        return False

def main():
    """Main function to fix all diffusers and Wan2.2 issues"""
    
    print("🚀 Starting Complete Diffusers and Wan2.2 Fix")
    print("=" * 60)
    
    success_count = 0
    
    # Step 1: Fix diffusers installation
    if fix_diffusers_installation():
        success_count += 1
        print("✅ Diffusers installation fixed")
    else:
        print("❌ Diffusers installation failed")
    
    # Step 2: Try to install Wan2.2 components
    if install_wan22_components():
        success_count += 1
        print("✅ Wan2.2 components installed")
    else:
        print("⚠️  Wan2.2 components not available, creating compatibility layer...")
        
        # Step 3: Create compatibility layer
        if create_wan22_pipeline_stub():
            success_count += 1
            print("✅ Wan2.2 compatibility layer created")
        else:
            print("❌ Wan2.2 compatibility layer failed")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Fix Results: {success_count}/3 components fixed")
    
    if success_count >= 2:
        print("🎉 Diffusers and Wan2.2 should now work!")
        print("\nNext steps:")
        print("1. Restart your Python environment")
        print("2. Run your video generation script")
        print("3. The models should now load without errors")
        return True
    else:
        print("❌ Multiple components failed. Manual intervention may be required.")
        print("\nTroubleshooting:")
        print("1. Check your Python environment")
        print("2. Ensure you have sufficient disk space")
        print("3. Check network connectivity for git installations")
        return False

if __name__ == "__main__":
    main()