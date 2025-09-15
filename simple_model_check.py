#!/usr/bin/env python3
"""
Simple model detection test - check what files exist vs what's expected
"""

import os
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.core.model_orchestrator.model_registry import ModelRegistry

def check_models():
    """Check what models exist vs what's expected"""
    
    print("🔍 Simple Model Check")
    print("=" * 50)
    
    models_root = os.getenv("WAN_MODELS_ROOT", "D:/AI/models")
    config_path = "config/models.toml"
    
    print(f"Models root: {models_root}")
    print(f"Config path: {config_path}")
    
    try:
        # Load registry
        registry = ModelRegistry(config_path)
        model_ids = registry.list_models()
        
        print(f"\n📋 Found {len(model_ids)} models in config:")
        
        for model_id in model_ids:
            print(f"\n🤖 {model_id}")
            
            # Get model spec
            spec = registry.spec(model_id)
            print(f"  Description: {spec.description}")
            print(f"  Version: {spec.version}")
            
            # Expected path
            expected_path = Path(models_root) / "wan22" / model_id
            print(f"  Expected path: {expected_path}")
            print(f"  Path exists: {expected_path.exists()}")
            
            if expected_path.exists():
                # List actual files
                actual_files = list(expected_path.rglob("*"))
                print(f"  Actual files found: {len(actual_files)}")
                
                # Show expected files
                print(f"  Expected files ({len(spec.files)}):")
                for file_spec in spec.files:
                    file_path = expected_path / file_spec.path
                    exists = file_path.exists()
                    size_match = False
                    if exists:
                        actual_size = file_path.stat().st_size
                        size_match = actual_size == file_spec.size
                    
                    status = "✅" if exists and size_match else "❌"
                    print(f"    {status} {file_spec.path} ({file_spec.component})")
                    if exists and not size_match:
                        print(f"        Size mismatch: expected {file_spec.size}, got {actual_size}")
                
                # Show some actual files for comparison
                print(f"  Sample actual files:")
                for f in actual_files[:10]:
                    if f.is_file():
                        print(f"    📄 {f.relative_to(expected_path)} ({f.stat().st_size} bytes)")
            else:
                print(f"  ❌ Directory not found")
        
        print(f"\n💡 Summary:")
        print(f"The model orchestrator expects specific files as defined in models.toml")
        print(f"Your models may have different file structures that need mapping.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_models()