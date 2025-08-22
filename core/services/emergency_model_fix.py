#!/usr/bin/env python3
"""
Emergency fix for model loading - force use of local models
"""

import os
import sys
from pathlib import Path

def create_model_override():
    """Create a simple override to force local model usage"""
    
    print("Creating Emergency Model Loading Override")
    print("=" * 45)
    
    # Create a simple model path override
    override_code = '''#!/usr/bin/env python3
"""
Model path override - forces use of local models
"""

from pathlib import Path

def get_local_model_path(model_id):
    """Get the path to a local model, bypassing download"""
    
    # Known local model paths
    local_paths = {
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "models/Wan-AI_Wan2.2-I2V-A14B-Diffusers", 
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "models/Wan-AI_Wan2.2-TI2V-5B-Diffusers"
    }
    
    if model_id in local_paths:
        local_path = Path(local_paths[model_id])
        if local_path.exists() and (local_path / "model_index.json").exists():
            print(f"Using local model: {local_path}")
            return str(local_path)
    
    # Fallback - try common patterns
    fallback_paths = [
        Path("models") / model_id.replace("/", "_"),
        Path("models") / f"models--{model_id.replace('/', '--')}",
        Path("models") / model_id.split("/")[-1]
    ]
    
    for path in fallback_paths:
        if path.exists() and (path / "model_index.json").exists():
            print(f"Using fallback local model: {path}")
            return str(path)
    
    raise FileNotFoundError(f"Local model not found: {model_id}")

# Monkey patch the model loading
def patch_model_loading():
    """Patch the model loading to use local models"""
    from core.services from core.services import utils
    
    # Store original method
    if hasattr(utils.ModelManager, '_original_download_model'):
        return  # Already patched
    
    original_download = utils.ModelManager.download_model
    utils.ModelManager._original_download_model = original_download
    
    def patched_download_model(self, model_id, force_download=False):
        """Patched download method that prioritizes local models"""
        full_model_id = self.get_model_id(model_id)
        
        try:
            # Try to get local model first
            local_path = get_local_model_path(full_model_id)
            print(f"SUCCESS: Using local model at {local_path}")
            return local_path
        except FileNotFoundError:
            print(f"WARNING: Local model not found for {full_model_id}")
            # Fall back to original method
            return original_download(self, model_id, force_download)
    
    # Apply the patch
    utils.ModelManager.download_model = patched_download_model
    print("Model loading patched to use local models")

if __name__ == "__main__":
    patch_model_loading()
'''
    
    # Write the override file
    Path("model_override.py").write_text(override_code)
    print("✅ Created model_override.py")
    
    # Create a simple test
    test_code = '''#!/usr/bin/env python3
"""
Test the model override
"""

import sys
sys.path.insert(0, ".")

from model_override import get_local_model_path, patch_model_loading

def test_override():
    """Test the model override functionality"""
    
    print("Testing Model Override")
    print("=" * 25)
    
    # Test direct path lookup
    try:
        model_path = get_local_model_path("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        print(f"✅ Found T2V model: {model_path}")
        return True
    except FileNotFoundError as e:
        print(f"❌ T2V model not found: {e}")
        return False

if __name__ == "__main__":
    success = test_override()
    if success:
        print("\\n🎉 Model override is working!")
        print("To use: import model_override; model_override.patch_model_loading()")
    else:
        print("\\n❌ Model override failed")
'''
    
    Path("test_model_override.py").write_text(test_code)
    print("✅ Created test_model_override.py")
    
    # Create startup script that applies the patch
    startup_code = '''#!/usr/bin/env python3
"""
Startup script that applies model loading patches
Import this at the beginning of main.py or ui.py
"""

print("Applying emergency model loading fixes...")

try:
    from model_override import patch_model_loading
    patch_model_loading()
    print("✅ Model loading patches applied successfully")
except Exception as e:
    print(f"⚠️  Failed to apply model patches: {e}")
'''
    
    Path("apply_model_fixes.py").write_text(startup_code)
    print("✅ Created apply_model_fixes.py")
    
    print("\n" + "=" * 45)
    print("EMERGENCY FIX CREATED")
    print("=" * 45)
    print("Files created:")
    print("1. model_override.py - Core override functionality")
    print("2. test_model_override.py - Test the override")
    print("3. apply_model_fixes.py - Startup patch script")
    
    print("\nTo apply the fix:")
    print("1. Test: python test_model_override.py")
    print("2. Add to main.py: import apply_model_fixes")
    print("3. Or manually: from model_override import patch_model_loading; patch_model_loading()")
    
    return True

if __name__ == "__main__":
    create_model_override()