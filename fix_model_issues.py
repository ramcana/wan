#!/usr/bin/env python3
"""
Model Issues Fix Script
Diagnoses and fixes common WAN2.2 model issues including missing files
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_directory():
    """Check the models directory structure"""
    print("🔍 Checking models directory...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory doesn't exist")
        models_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Created models directory")
        return False
    
    # List all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("❌ No model directories found")
        return False
    
    print(f"📁 Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        files = list(model_dir.glob("*"))
        print(f"   • {model_dir.name}: {len(files)} files")
        
        # Check for common model files
        expected_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        found_files = [f.name for f in files]
        
        for expected in expected_files:
            if expected in found_files:
                file_size = (model_dir / expected).stat().st_size / (1024 * 1024)
                print(f"     ✅ {expected} ({file_size:.1f}MB)")
            else:
                print(f"     ❌ {expected} (missing)")
    
    return len(model_dirs) > 0

def validate_with_model_downloader():
    """Use the existing model downloader to validate models"""
    print("\n🔧 Validating with model downloader...")
    
    try:
        # Add local installation to path
        local_installation_path = Path("local_installation")
        if local_installation_path.exists():
            sys.path.insert(0, str(local_installation_path))
            sys.path.insert(0, str(local_installation_path / "scripts"))
        
        from scripts.download_models import ModelDownloader
        
        # Initialize downloader
        downloader = ModelDownloader(
            installation_path=str(Path.cwd()),
            models_dir=str(Path("models"))
        )
        
        print("✅ Model downloader initialized")
        
        # Check existing models
        existing_models = downloader.check_existing_models()
        print(f"📦 Valid models found: {existing_models}")
        
        # Get verification results
        verification = downloader.verify_all_models()
        
        if verification.all_valid:
            print("✅ All required models are valid")
            return True
        else:
            print(f"❌ Invalid models: {verification.invalid_models}")
            
            # Offer to download missing models
            response = input("\n🤔 Would you like to download missing models? (y/n): ")
            if response.lower() in ['y', 'yes']:
                print("📥 Starting model download...")
                success = downloader.download_wan22_models()
                if success:
                    print("✅ Model download completed successfully")
                    return True
                else:
                    print("❌ Model download failed")
                    return False
            else:
                print("⏭️ Skipping model download")
                return False
    
    except ImportError as e:
        print(f"❌ Could not import model downloader: {e}")
        print("💡 Make sure local_installation directory exists")
        return False
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def check_huggingface_hub():
    """Check if Hugging Face Hub is available and configured"""
    print("\n🤗 Checking Hugging Face Hub integration...")
    
    try:
        from huggingface_hub import hf_hub_download, login
        print("✅ Hugging Face Hub library available")
        
        # Check if user is logged in (optional for public models)
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            print(f"✅ Logged in as: {user_info['name']}")
        except Exception:
            print("ℹ️ Not logged in to Hugging Face Hub (OK for public models)")
        
        return True
        
    except ImportError:
        print("❌ Hugging Face Hub library not available")
        print("💡 Install with: pip install huggingface_hub")
        return False

def fix_model_metadata():
    """Fix or create model metadata file"""
    print("\n📝 Checking model metadata...")
    
    models_dir = Path("models")
    metadata_file = models_dir / "model_metadata.json"
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Model metadata exists with {len(metadata)} entries")
            return True
        except Exception as e:
            print(f"❌ Model metadata corrupted: {e}")
    
    # Create new metadata
    print("🔧 Creating new model metadata...")
    
    metadata = {}
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            files = list(model_dir.glob("*"))
            metadata[model_dir.name] = {
                "downloaded": True,
                "files": [f.name for f in files],
                "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
                "last_verified": None
            }
    
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Model metadata created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create metadata: {e}")
        return False

def suggest_solutions():
    """Suggest solutions based on common issues"""
    print("\n💡 COMMON SOLUTIONS:")
    print("=" * 50)
    
    print("1. 🔄 Re-download models:")
    print("   python local_installation/scripts/download_models.py")
    
    print("\n2. 🧹 Clean and fresh download:")
    print("   rm -rf models")
    print("   mkdir models")
    print("   python local_installation/scripts/download_models.py")
    
    print("\n3. 🔍 Check model configuration:")
    print("   python -c \"from local_installation.scripts.download_models import ModelDownloader; print(ModelDownloader.MODEL_CONFIG)\"")
    
    print("\n4. 🌐 Check internet connection and try again")
    
    print("\n5. 🔑 If using private models, login to Hugging Face:")
    print("   huggingface-cli login")
    
    print("\n6. 📊 Check available disk space (models need ~70GB)")

def main():
    """Main diagnostic and fix function"""
    print("🛠️ WAN2.2 MODEL ISSUES FIX SCRIPT")
    print("=" * 50)
    
    # Run diagnostics
    issues = []
    
    if not check_model_directory():
        issues.append("Models directory issues")
    
    if not check_huggingface_hub():
        issues.append("Hugging Face Hub not available")
    
    if not validate_with_model_downloader():
        issues.append("Model validation failed")
    
    if not fix_model_metadata():
        issues.append("Metadata issues")
    
    # Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    
    if not issues:
        print("✅ No issues detected - models should be working")
    else:
        print(f"❌ Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        suggest_solutions()

if __name__ == "__main__":
    main()