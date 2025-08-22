#!/usr/bin/env python3
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
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "models/Wan-AI_Wan2.2-TI2V-5B-Diffusers",
        # Additional fallback mappings for local directories
        "WAN2.2-T2V-A14B": "models/WAN2.2-T2V-A14B",
        "WAN2.2-I2V-A14B": "models/WAN2.2-I2V-A14B",
        "WAN2.2-TI2V-5B": "models/WAN2.2-TI2V-5B"
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
        Path("models") / model_id.split("/")[-1],
        # Additional patterns for WAN2.2 models
        Path("models") / f"WAN2.2-{model_id.split('/')[-1].replace('-Diffusers', '')}",
        Path("models") / model_id.split("/")[-1].replace("-Diffusers", ""),
        # Direct mapping for Wan2.2 models to WAN2.2 format
        Path("models") / model_id.replace("Wan2.2/", "WAN2.2-"),
        Path("models") / model_id.replace("Wan2.2/", "WAN2.2-").replace("-Diffusers", "")
    ]
    
    for path in fallback_paths:
        if path.exists():
            # Check for model_index.json or other model files
            if (path / "model_index.json").exists() or (path / "config.json").exists() or any(path.glob("*.safetensors")):
                print(f"Using fallback local model: {path}")
                return str(path)
    
    raise FileNotFoundError(f"Local model not found: {model_id}")

# Monkey patch the model loading
def patch_model_loading():
    """Patch the model loading to use local models"""
    try:
        from core.services from core.services import utils
        
        # Check if ModelManager exists
        if not hasattr(utils, 'ModelManager'):
            print("WARNING: ModelManager not found in utils module")
            return False
        
        # Store original method
        if hasattr(utils.ModelManager, '_original_download_model'):
            print("Model loading already patched")
            return True  # Already patched
        
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
        return True
        
    except Exception as e:
        print(f"Failed to patch model loading: {e}")
        return False

if __name__ == "__main__":
    patch_model_loading()