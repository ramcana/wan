#!/usr/bin/env python3
"""
Verify that the model loading fix will work
"""

from pathlib import Path
import json

def verify_local_model_exists():
    """Verify the local model exists and is valid"""
    
    print("Verifying Local Model Fix")
    print("=" * 30)
    
    # Model we're looking for
    full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    
    # Check the path that should work
    model_path = Path("models") / full_model_id.replace("/", "_")
    
    print(f"Checking model path: {model_path}")
    
    if not model_path.exists():
        print("❌ Model directory not found")
        return False
    
    print("✅ Model directory exists")
    
    # Check for model_index.json
    model_index_path = model_path / "model_index.json"
    if not model_index_path.exists():
        print("❌ model_index.json not found")
        return False
    
    print("✅ model_index.json exists")
    
    # Try to read model_index.json
    try:
        with open(model_index_path, 'r') as f:
            model_config = json.load(f)
        
        print("✅ model_index.json is valid JSON")
        print(f"Model components: {list(model_config.keys())}")
        
        # Check for essential components
        essential_components = ['scheduler', 'text_encoder', 'tokenizer', 'transformer']
        missing_components = []
        
        for component in essential_components:
            component_path = model_path / component
            if component_path.exists():
                print(f"✅ {component} directory exists")
            else:
                print(f"❌ {component} directory missing")
                missing_components.append(component)
        
        if missing_components:
            print(f"⚠️  Missing components: {missing_components}")
            return False
        
        print("✅ All essential components present")
        return True
        
    except Exception as e:
        print(f"❌ Error reading model_index.json: {e}")
        return False

def simulate_model_loading_logic():
    """Simulate the model loading logic from the fix"""
    
    print("\n" + "=" * 30)
    print("Simulating Model Loading Logic")
    print("=" * 30)
    
    full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    
    # Simulate the paths that will be checked
    local_paths_to_check = [
        # Alternative format in models directory
        Path("models") / full_model_id.replace("/", "_"),
        # Hugging Face cache format
        Path("models") / f"models--{full_model_id.replace('/', '--')}",
        # Direct models directory
        Path("models") / full_model_id.split("/")[-1]
    ]
    
    print("Checking paths in order:")
    
    for i, local_path in enumerate(local_paths_to_check, 1):
        print(f"\n{i}. {local_path}")
        
        if local_path.exists() and (local_path / "model_index.json").exists():
            print(f"   ✅ FOUND! This path will be used")
            print(f"   📁 Model will load from: {local_path}")
            return str(local_path)
        else:
            print(f"   ❌ Not found or invalid")
    
    print("\n❌ No valid local model found - would attempt download")
    return None

def main():
    """Main verification"""
    
    print("Model Loading Fix Verification")
    print("=" * 40)
    
    # Test 1: Verify local model exists
    model_exists = verify_local_model_exists()
    
    # Test 2: Simulate loading logic
    found_path = simulate_model_loading_logic()
    
    print("\n" + "=" * 40)
    print("VERIFICATION RESULTS")
    print("=" * 40)
    
    if model_exists and found_path:
        print("🎉 SUCCESS!")
        print("✅ Local model exists and is valid")
        print("✅ Model loading logic will find it")
        print("✅ No download should be attempted")
        print(f"✅ Model will load from: {found_path}")
        print("\nThe fix should resolve the download issue!")
        return True
    elif model_exists:
        print("⚠️  PARTIAL SUCCESS")
        print("✅ Local model exists but loading logic may have issues")
        return False
    else:
        print("❌ FAILED")
        print("❌ Local model not found or invalid")
        print("❌ Download will still be attempted")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)