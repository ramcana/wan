#!/usr/bin/env python3
"""
Test script to check what the model orchestrator actually detects
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.core.model_orchestrator.model_registry import ModelRegistry
from backend.core.model_orchestrator.model_resolver import ModelResolver
from backend.core.model_orchestrator.model_ensurer import ModelEnsurer

def test_model_detection():
    """Test what models the orchestrator can detect"""
    
    print("🔍 Testing Model Detection")
    print("=" * 50)
    
    # Initialize components
    models_root = os.getenv("WAN_MODELS_ROOT", "D:/AI/models")
    config_path = "config/models.toml"
    
    print(f"Models root: {models_root}")
    print(f"Config path: {config_path}")
    
    try:
        # Initialize registry
        registry = ModelRegistry(config_path)
        model_ids = registry.list_models()
        print(f"✅ Registry loaded with {len(model_ids)} models")
        
        # List available models
        for model_id in model_ids:
            print(f"  📋 {model_id}")
        
        # Initialize resolver
        resolver = ModelResolver(models_root)
        print(f"✅ Resolver initialized")
        
        # Initialize ensurer
        ensurer = ModelEnsurer(registry, resolver)
        print(f"✅ Ensurer initialized")
        
        # Check each model status
        print("\n📊 Model Status Check:")
        for model_id in model_ids:
            try:
                status_info = ensurer.get_model_status(model_id)
                print(f"  🤖 {model_id}:")
                print(f"    Status: {status_info.status.value}")
                print(f"    Path: {status_info.local_path}")
                
                if status_info.missing_files:
                    print(f"    Missing files: {status_info.missing_files}")
                if status_info.error_message:
                    print(f"    Error: {status_info.error_message}")
                    
                # Check what files actually exist
                if status_info.local_path:
                    local_path = Path(status_info.local_path)
                    if local_path.exists():
                        actual_files = list(local_path.rglob("*"))
                        print(f"    Actual files found: {len(actual_files)}")
                        for f in actual_files[:5]:  # Show first 5
                            print(f"      - {f.name}")
                        if len(actual_files) > 5:
                            print(f"      ... and {len(actual_files) - 5} more")
                    else:
                        print(f"    Directory doesn't exist: {local_path}")
                        
            except Exception as e:
                print(f"  ❌ Error checking {model_id}: {e}")
        
        print("\n🎯 Summary:")
        print("The model orchestrator is looking for specific files defined in models.toml")
        print("but your actual model files may have different names/structure.")
        
    except Exception as e:
        print(f"❌ Failed to initialize model orchestrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_detection()