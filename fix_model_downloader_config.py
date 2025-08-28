#!/usr/bin/env python3
"""
Fix Model Downloader Configuration
Updates the model downloader to match actual DialoGPT file structure
"""

import sys
from pathlib import Path

def fix_model_downloader_config():
    """Fix the model downloader configuration to match actual files"""
    print("🔧 Fixing Model Downloader Configuration...")
    
    # Add local installation to path
    local_installation_path = Path("local_installation")
    if local_installation_path.exists():
        sys.path.insert(0, str(local_installation_path))
        sys.path.insert(0, str(local_installation_path / "scripts"))
    
    try:
        from scripts.download_models import ModelDownloader
        
        # Get the actual files in downloaded models
        models_dir = Path("models")
        actual_files = {}
        
        for model_name in ["WAN2.2-T2V-A14B", "WAN2.2-I2V-A14B", "WAN2.2-TI2V-5B"]:
            model_path = models_dir / model_name
            if model_path.exists():
                files = [f.name for f in model_path.iterdir() if f.is_file()]
                actual_files[model_name] = files
                print(f"📁 {model_name}: {len(files)} files")
                for file in sorted(files):
                    print(f"   • {file}")
        
        # Update the MODEL_CONFIG to match actual files
        if actual_files:
            print("\n🔄 Updating model configuration...")
            
            # Common files found in DialoGPT models
            common_files = [
                "pytorch_model.bin",
                "config.json", 
                "tokenizer_config.json",
                "vocab.json",  # Instead of tokenizer.json
                "merges.txt"   # Instead of special_tokens_map.json
            ]
            
            # Patch the MODEL_CONFIG
            for model_name in ModelDownloader.MODEL_CONFIG:
                ModelDownloader.MODEL_CONFIG[model_name].files = common_files
            
            print("✅ Model configuration updated to match actual files")
            
            # Test validation again
            downloader = ModelDownloader(
                installation_path=str(Path.cwd()),
                models_dir=str(models_dir)
            )
            
            existing_models = downloader.check_existing_models()
            print(f"✅ Validation after fix: {len(existing_models)} valid models")
            
            verification = downloader.verify_all_models()
            if verification.all_valid:
                print("🎉 All models now validate successfully!")
                return True
            else:
                print(f"⚠️ Still invalid: {verification.invalid_models}")
                return False
        else:
            print("❌ No downloaded models found to analyze")
            return False
            
    except Exception as e:
        print(f"❌ Failed to fix configuration: {e}")
        return False

def create_permanent_fix():
    """Create a permanent fix by patching the model downloader file"""
    print("\n🛠️ Creating permanent fix...")
    
    downloader_file = Path("local_installation/scripts/download_models.py")
    if not downloader_file.exists():
        print("❌ Model downloader file not found")
        return False
    
    try:
        # Read the current file
        with open(downloader_file, 'r') as f:
            content = f.read()
        
        # Replace the expected files with actual DialoGPT files
        old_files = '''files=[
                "pytorch_model.bin",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ]'''
        
        new_files = '''files=[
                "pytorch_model.bin",
                "config.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt"
            ]'''
        
        if old_files in content:
            # Create backup
            backup_file = downloader_file.with_suffix('.py.backup')
            with open(backup_file, 'w') as f:
                f.write(content)
            print(f"✅ Created backup: {backup_file}")
            
            # Apply fix
            fixed_content = content.replace(old_files, new_files)
            
            with open(downloader_file, 'w') as f:
                f.write(fixed_content)
            
            print("✅ Applied permanent fix to model downloader")
            return True
        else:
            print("⚠️ Expected pattern not found in file - manual fix needed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create permanent fix: {e}")
        return False

def main():
    """Main function"""
    print("🔧 MODEL DOWNLOADER CONFIGURATION FIX")
    print("=" * 50)
    
    # Try runtime fix first
    if fix_model_downloader_config():
        print("\n✅ Runtime fix successful!")
        
        # Offer permanent fix
        response = input("\n🤔 Apply permanent fix to model downloader? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if create_permanent_fix():
                print("🎉 Permanent fix applied successfully!")
            else:
                print("⚠️ Permanent fix failed - runtime fix still active")
        else:
            print("ℹ️ Runtime fix applied - restart may reset this")
    else:
        print("❌ Fix failed - check model files and try again")

if __name__ == "__main__":
    main()