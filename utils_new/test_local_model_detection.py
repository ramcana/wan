#!/usr/bin/env python3
"""
Test script to verify local model detection works correctly
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_local_model_paths():
    """Test that we can find the local models"""
    
    print("Testing Local Model Detection")
    print("=" * 40)
    
    # Model we're looking for
    full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    
    # Possible local paths (same logic as in utils.py)
    local_paths_to_check = [
        # Alternative format in models directory
        Path("models") / full_model_id.replace("/", "_"),
        # Hugging Face cache format
        Path("models") / f"models--{full_model_id.replace('/', '--')}",
        # Direct models directory
        Path("models") / full_model_id.split("/")[-1]
    ]
    
    print(f"Looking for model: {full_model_id}")
    print()
    
    found_models = []
    
    for i, local_path in enumerate(local_paths_to_check, 1):
        print(f"Checking path {i}: {local_path}")
        
        if local_path.exists():
            print(f"  ✅ Directory exists")
            
            # Check for model_index.json
            model_index = local_path / "model_index.json"
            if model_index.exists():
                print(f"  ✅ model_index.json found")
                found_models.append(str(local_path))
                
                # List contents
                try:
                    contents = list(local_path.iterdir())
                    print(f"  📁 Contains {len(contents)} items:")
                    for item in contents[:5]:  # Show first 5 items
                        print(f"    - {item.name}")
                    if len(contents) > 5:
                        print(f"    ... and {len(contents) - 5} more")
                except Exception as e:
                    print(f"  ⚠️  Error listing contents: {e}")
            else:
                print(f"  ❌ model_index.json not found")
        else:
            print(f"  ❌ Directory does not exist")
        
        print()
    
    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    if found_models:
        print(f"✅ Found {len(found_models)} valid local model(s):")
        for model_path in found_models:
            print(f"  - {model_path}")
        print()
        print("🎉 Local model detection should work!")
        return True
    else:
        print("❌ No valid local models found")
        print("The system will try to download from Hugging Face")
        return False

def test_model_manager_integration():
    """Test the model manager's local detection"""
    
    print("\n" + "=" * 40)
    print("Testing Model Manager Integration")
    print("=" * 40)
    
    try:
        from core.services.model_manager import ModelManager
        
        # Create model manager
        manager = ModelManager()
        
        # Test model ID resolution
        model_id = manager.get_model_id("t2v-A14B")
        print(f"Model ID resolution: t2v-A14B -> {model_id}")
        
        # Test download (should find local)
        try:
            model_path = manager.download_model("t2v-A14B")
            print(f"✅ Model found at: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Model manager failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import ModelManager: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing model manager: {e}")
        return False

def main():
    """Run all tests"""
    
    print("Local Model Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Direct path checking
    test1_passed = test_local_model_paths()
    
    # Test 2: Model manager integration
    test2_passed = test_model_manager_integration()
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed!")
        print("Local model detection is working correctly.")
        print("The system should use existing models instead of downloading.")
    elif test1_passed:
        print("⚠️  Local models found but model manager has issues.")
        print("Check the model manager implementation.")
    else:
        print("❌ Local model detection failed.")
        print("Models may need to be re-downloaded or paths fixed.")

if __name__ == "__main__":
    main()