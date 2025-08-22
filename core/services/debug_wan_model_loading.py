"""
Debug script for WAN model loading issues.

This script helps diagnose why WAN models are falling back to StableDiffusionPipeline
instead of loading with the proper WanPipeline.
"""

import os
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_model_structure(model_path: str):
    """Debug the structure of a model directory"""
    print(f"\n=== Debugging Model Structure: {model_path} ===")
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return
    
    print(f"✅ Model path exists: {model_path}")
    
    # Check for model_index.json
    model_index_path = model_path / "model_index.json"
    if model_index_path.exists():
        print(f"✅ Found model_index.json")
        try:
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            print("📋 Model Index Contents:")
            for key, value in model_index.items():
                print(f"   {key}: {value}")
            
            # Check for WAN-specific indicators
            wan_indicators = []
            if "transformer" in model_index:
                wan_indicators.append("transformer")
            if "transformer_2" in model_index:
                wan_indicators.append("transformer_2")
            if "boundary_ratio" in model_index:
                wan_indicators.append("boundary_ratio")
            if model_index.get("_class_name") and "Wan" in model_index["_class_name"]:
                wan_indicators.append("WAN pipeline class")
            
            if wan_indicators:
                print(f"✅ WAN indicators found: {', '.join(wan_indicators)}")
            else:
                print("❌ No WAN indicators found in model_index.json")
                
        except Exception as e:
            print(f"❌ Error reading model_index.json: {e}")
    else:
        print(f"❌ No model_index.json found")
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    for item in sorted(model_path.iterdir()):
        if item.is_dir():
            print(f"   📁 {item.name}/")
            # Check for config.json in subdirectories
            config_path = item / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    class_name = config.get("_class_name", "Unknown")
                    print(f"      └── config.json (_class_name: {class_name})")
                except:
                    print(f"      └── config.json (unreadable)")
        else:
            print(f"   📄 {item.name}")

def test_architecture_detection(model_path: str):
    """Test architecture detection for a model"""
    print(f"\n=== Testing Architecture Detection ===")
    
    try:
        from infrastructure.hardware.architecture_detector import ArchitectureDetector
        
        detector = ArchitectureDetector()
        architecture = detector.detect_model_architecture(model_path)
        
        print(f"✅ Architecture detected: {architecture.architecture_type.value}")
        print(f"📋 Architecture Details:")
        print(f"   Type: {architecture.architecture_type.value}")
        print(f"   Components: {list(architecture.components.keys())}")
        print(f"   Capabilities: {architecture.capabilities}")
        
        if architecture.signature:
            print(f"   Signature:")
            print(f"     - has_transformer: {architecture.signature.has_transformer}")
            print(f"     - has_transformer_2: {architecture.signature.has_transformer_2}")
            print(f"     - has_boundary_ratio: {architecture.signature.has_boundary_ratio}")
            print(f"     - vae_dimensions: {architecture.signature.vae_dimensions}")
            print(f"     - pipeline_class: {architecture.signature.pipeline_class}")
            print(f"     - is_wan_architecture: {architecture.signature.is_wan_architecture()}")
        
        return architecture
        
    except Exception as e:
        print(f"❌ Architecture detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pipeline_selection(architecture):
    """Test pipeline selection for detected architecture"""
    print(f"\n=== Testing Pipeline Selection ===")
    
    if not architecture:
        print("❌ No architecture to test")
        return None
    
    try:
        from pipeline_manager import PipelineManager
        
        manager = PipelineManager()
        pipeline_class = manager.select_pipeline_class(architecture.signature)
        
        print(f"✅ Selected pipeline class: {pipeline_class}")
        
        # Check pipeline requirements
        requirements = manager.PIPELINE_REQUIREMENTS.get(pipeline_class)
        if requirements:
            print(f"📋 Pipeline Requirements:")
            print(f"   Required args: {requirements.required_args}")
            print(f"   Optional args: {requirements.optional_args}")
            print(f"   Trust remote code: {requirements.requires_trust_remote_code}")
            print(f"   Min VRAM: {requirements.min_vram_mb}MB")
        
        return pipeline_class
        
    except Exception as e:
        print(f"❌ Pipeline selection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pipeline_loading(model_path: str, pipeline_class: str):
    """Test actual pipeline loading"""
    print(f"\n=== Testing Pipeline Loading ===")
    
    try:
        from pipeline_manager import PipelineManager
        
        manager = PipelineManager()
        
        print(f"Attempting to load {pipeline_class} from {model_path}")
        
        load_result = manager.load_custom_pipeline(
            model_path=model_path,
            pipeline_class=pipeline_class,
            trust_remote_code=True
        )
        
        print(f"Load status: {load_result.status.value}")
        
        if load_result.status.name == "SUCCESS":
            print(f"✅ Pipeline loaded successfully!")
            print(f"   Pipeline type: {type(load_result.pipeline).__name__}")
            print(f"   Pipeline class: {load_result.pipeline_class}")
        else:
            print(f"❌ Pipeline loading failed:")
            print(f"   Error: {load_result.error_message}")
            print(f"   Warnings: {load_result.warnings}")
        
        return load_result
        
    except Exception as e:
        print(f"❌ Pipeline loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_diffusers_direct_loading(model_path: str):
    """Test direct loading with diffusers"""
    print(f"\n=== Testing Direct Diffusers Loading ===")
    
    try:
        from diffusers import DiffusionPipeline
        
        print("Attempting direct DiffusionPipeline.from_pretrained with trust_remote_code=True")
        
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"✅ Direct loading successful!")
        print(f"   Pipeline type: {type(pipeline).__name__}")
        print(f"   Pipeline components: {list(pipeline.components.keys()) if hasattr(pipeline, 'components') else 'N/A'}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Direct loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function"""
    print("WAN Model Loading Debug Script")
    print("=" * 50)
    
    # Get model path from user or use default
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find a WAN model in common locations
        possible_paths = [
            "models/WAN2.2-T2V-A14B",
            "local_installation/models/WAN2.2-T2V-A14B",
            "models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        ]
        
        model_path = None
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if not model_path:
            print("No model path provided and no WAN models found in common locations.")
            print("Usage: python debug_wan_model_loading.py <model_path>")
            return
    
    print(f"Using model path: {model_path}")
    
    # Run debug steps
    debug_model_structure(model_path)
    architecture = test_architecture_detection(model_path)
    pipeline_class = test_pipeline_selection(architecture)
    
    if pipeline_class:
        load_result = test_pipeline_loading(model_path, pipeline_class)
    
    # Also test direct loading
    direct_pipeline = test_diffusers_direct_loading(model_path)
    
    print(f"\n=== Debug Summary ===")
    print(f"Model path: {model_path}")
    print(f"Architecture detected: {architecture.architecture_type.value if architecture else 'Failed'}")
    print(f"Pipeline class selected: {pipeline_class if pipeline_class else 'Failed'}")
    print(f"Pipeline loading: {'Success' if load_result and load_result.status.name == 'SUCCESS' else 'Failed'}")
    print(f"Direct loading: {'Success' if direct_pipeline else 'Failed'}")

if __name__ == "__main__":
    main()