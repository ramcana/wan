#!/usr/bin/env python3
"""
Model Validation and Setup for WAN2.2 UI
Ensures models are in the correct location and properly configured
"""

import os
import json
import shutil
from pathlib import Path
import logging

def setup_logging():
    """Setup logging for model validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """Load the main configuration file"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load config.json: {e}")
        return None

def check_model_directory_structure():
    """Check and validate model directory structure"""
    logger = logging.getLogger(__name__)
    
    # Expected model locations
    main_models_dir = Path("models")
    local_models_dir = Path("local_installation/models")
    
    print("🔍 Checking model directory structure...")
    
    # Check main models directory
    if main_models_dir.exists():
        models_in_main = list(main_models_dir.glob("*"))
        print(f"✅ Main models directory exists with {len(models_in_main)} items")
        for model in models_in_main:
            if model.is_dir():
                print(f"   📁 {model.name}")
    else:
        print("❌ Main models directory not found")
        main_models_dir.mkdir(exist_ok=True)
        print("✅ Created main models directory")
    
    # Check local installation models directory
    if local_models_dir.exists():
        models_in_local = list(local_models_dir.glob("*"))
        print(f"✅ Local installation models directory exists with {len(models_in_local)} items")
        for model in models_in_local:
            if model.is_dir():
                print(f"   📁 {model.name}")
    else:
        print("❌ Local installation models directory not found")
    
    return main_models_dir, local_models_dir

def validate_required_models(config):
    """Validate that required models are available"""
    if not config:
        return False
    
    required_models = config.get("models", {})
    models_dir = Path(config.get("directories", {}).get("models_directory", "models"))
    
    print(f"\n🔍 Validating required models in {models_dir}...")
    
    missing_models = []
    found_models = []
    
    for model_type, model_name in required_models.items():
        model_path = models_dir / model_name
        if model_path.exists() and model_path.is_dir():
            # Check if model has essential files
            essential_files = ["config.json", "model_index.json"]
            has_essential = any((model_path / f).exists() for f in essential_files)
            
            if has_essential:
                print(f"✅ {model_type}: {model_name} - Found and valid")
                found_models.append((model_type, model_name))
            else:
                print(f"⚠️  {model_type}: {model_name} - Directory exists but missing essential files")
                missing_models.append((model_type, model_name))
        else:
            print(f"❌ {model_type}: {model_name} - Not found")
            missing_models.append((model_type, model_name))
    
    return len(missing_models) == 0, missing_models, found_models

def copy_models_from_local_installation():
    """Copy models from local_installation/models to main models directory"""
    local_models_dir = Path("local_installation/models")
    main_models_dir = Path("models")
    
    if not local_models_dir.exists():
        print("❌ No local installation models to copy")
        return False
    
    print("\n🔄 Copying models from local installation...")
    
    copied_count = 0
    for model_dir in local_models_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith('.'):
            target_dir = main_models_dir / model_dir.name
            
            if target_dir.exists():
                print(f"⚠️  {model_dir.name} already exists in main models directory")
                continue
            
            try:
                print(f"📋 Copying {model_dir.name}...")
                shutil.copytree(model_dir, target_dir)
                print(f"✅ Successfully copied {model_dir.name}")
                copied_count += 1
            except Exception as e:
                print(f"❌ Failed to copy {model_dir.name}: {e}")
    
    print(f"\n✅ Copied {copied_count} models to main directory")
    return copied_count > 0

def create_model_symlinks():
    """Create symbolic links for models if copying fails"""
    local_models_dir = Path("local_installation/models")
    main_models_dir = Path("models")
    
    if not local_models_dir.exists():
        return False
    
    print("\n🔗 Creating symbolic links for models...")
    
    linked_count = 0
    for model_dir in local_models_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith('.'):
            target_link = main_models_dir / model_dir.name
            
            if target_link.exists():
                continue
            
            try:
                # Create relative symlink
                relative_path = os.path.relpath(model_dir, main_models_dir)
                target_link.symlink_to(relative_path, target_is_directory=True)
                print(f"✅ Created symlink for {model_dir.name}")
                linked_count += 1
            except Exception as e:
                print(f"❌ Failed to create symlink for {model_dir.name}: {e}")
    
    print(f"\n✅ Created {linked_count} model symlinks")
    return linked_count > 0

def fix_model_paths():
    """Fix model paths by copying or linking from local installation"""
    print("\n🔧 Attempting to fix model paths...")
    
    # Try copying first
    if copy_models_from_local_installation():
        return True
    
    # If copying fails, try symlinks
    if create_model_symlinks():
        return True
    
    print("❌ Could not fix model paths automatically")
    return False

def validate_model_files(model_path):
    """Validate that a model directory contains necessary files"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        return False, "Model directory does not exist"
    
    # Check for essential files
    essential_files = [
        "config.json",
        "model_index.json"
    ]
    
    missing_files = []
    for file in essential_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        return False, f"Missing essential files: {', '.join(missing_files)}"
    
    # Check for model weights (at least one should exist)
    weight_patterns = [
        "*.safetensors",
        "*.bin",
        "pytorch_model.bin",
        "diffusion_pytorch_model.safetensors"
    ]
    
    has_weights = False
    for pattern in weight_patterns:
        if list(model_path.glob(pattern)) or list(model_path.rglob(pattern)):
            has_weights = True
            break
    
    if not has_weights:
        return False, "No model weight files found"
    
    return True, "Model validation passed"

def main():
    """Main model validation and setup function"""
    logger = setup_logging()
    
    print("WAN2.2 Model Validator")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    if not config:
        print("❌ Cannot proceed without valid configuration")
        return False
    
    # Check directory structure
    main_models_dir, local_models_dir = check_model_directory_structure()
    
    # Validate required models
    models_valid, missing_models, found_models = validate_required_models(config)
    
    if models_valid:
        print("\n🎉 All required models are available and valid!")
        
        # Validate each found model
        print("\n🔍 Detailed model validation...")
        for model_type, model_name in found_models:
            model_path = Path(config["directories"]["models_directory"]) / model_name
            is_valid, message = validate_model_files(model_path)
            status = "✅" if is_valid else "⚠️ "
            print(f"   {status} {model_name}: {message}")
        
        return True
    
    else:
        print(f"\n❌ Missing {len(missing_models)} required models:")
        for model_type, model_name in missing_models:
            print(f"   - {model_type}: {model_name}")
        
        # Attempt to fix by copying/linking from local installation
        if local_models_dir.exists():
            if fix_model_paths():
                print("\n🔄 Re-validating after fix...")
                models_valid, missing_models, found_models = validate_required_models(config)
                
                if models_valid:
                    print("🎉 Model paths fixed successfully!")
                    return True
                else:
                    print("❌ Some models are still missing after fix attempt")
        
        print("\n📋 Manual steps required:")
        print("1. Download the required models:")
        for model_type, model_name in missing_models:
            print(f"   - {model_name}")
        print("2. Place them in the 'models' directory")
        print("3. Run this validator again")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Model validation passed - ready to launch UI!")
    else:
        print("\n❌ Model validation failed - please fix issues before launching")
    
    input("\nPress Enter to exit...")
