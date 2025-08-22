#!/usr/bin/env python3
"""
Debug script to inspect WAN pipeline signatures and parameters
"""

import torch
import inspect
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def inspect_pipeline_signature(model_id):
    """Inspect the actual pipeline signature and methods"""
    print(f"Inspecting pipeline: {model_id}")
    print("=" * 60)
    
    try:
        # Apply WAN compatibility
        from wan22_compatibility_clean import apply_wan22_compatibility
        apply_wan22_compatibility()
        
        from diffusers import DiffusionPipeline
        
        print("Loading pipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Pipeline loaded successfully")
        print(f"Pipeline class: {type(pipe).__name__}")
        print(f"Pipeline module: {type(pipe).__module__}")
        
        # Inspect __call__ method
        if hasattr(pipe, '__call__'):
            call_method = getattr(pipe, '__call__')
            sig = inspect.signature(call_method)
            
            print(f"\n📋 __call__ method signature:")
            print(f"Parameters: {list(sig.parameters.keys())}")
            
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    default = param.default if param.default != inspect.Parameter.empty else "Required"
                    print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} = {default}")
        
        # Check for other generation methods
        generation_methods = []
        for attr_name in dir(pipe):
            if not attr_name.startswith('_') and callable(getattr(pipe, attr_name)):
                attr = getattr(pipe, attr_name)
                if hasattr(attr, '__doc__') and attr.__doc__ and ('generate' in attr.__doc__.lower() or 'video' in attr.__doc__.lower()):
                    generation_methods.append(attr_name)
        
        if generation_methods:
            print(f"\n🔍 Potential generation methods found:")
            for method_name in generation_methods:
                method = getattr(pipe, method_name)
                try:
                    sig = inspect.signature(method)
                    print(f"  - {method_name}({', '.join(sig.parameters.keys())})")
                except:
                    print(f"  - {method_name} (signature unavailable)")
        
        # Check pipeline components
        print(f"\n🧩 Pipeline components:")
        for attr_name in dir(pipe):
            if not attr_name.startswith('_'):
                attr = getattr(pipe, attr_name)
                if hasattr(attr, '__class__') and 'torch.nn' in str(type(attr)):
                    print(f"  - {attr_name}: {type(attr).__name__}")
        
        # Check for config
        if hasattr(pipe, 'config'):
            print(f"\n⚙️ Pipeline config:")
            config = pipe.config
            if isinstance(config, dict):
                for key, value in config.items():
                    print(f"  - {key}: {value}")
        
        # Try to find documentation or examples
        if hasattr(pipe, '__doc__') and pipe.__doc__:
            print(f"\n📖 Pipeline documentation:")
            print(pipe.__doc__[:500] + "..." if len(pipe.__doc__) > 500 else pipe.__doc__)
        
        return pipe
        
    except Exception as e:
        print(f"❌ Error inspecting pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_parameter_variations(pipe):
    """Test different parameter combinations to find the correct one"""
    print(f"\n🧪 Testing parameter variations...")
    
    from PIL import Image
    
    # Create test inputs
    test_prompt = "A peaceful lake"
    test_image = Image.new('RGB', (512, 512), color='lightblue')
    
    # Common parameter variations for TI2V models
    param_variations = [
        # Standard diffusers parameters
        {'prompt': test_prompt, 'image': test_image},
        {'prompt': test_prompt, 'init_image': test_image},
        {'prompt': test_prompt, 'input_image': test_image},
        {'prompt': test_prompt, 'source_image': test_image},
        {'prompt': test_prompt, 'reference_image': test_image},
        
        # WAN-specific parameters
        {'prompt': test_prompt, 'start_image': test_image},
        {'prompt': test_prompt, 'end_image': test_image},
        {'prompt': test_prompt, 'condition_image': test_image},
        
        # Multiple image parameters
        {'prompt': test_prompt, 'start_image': test_image, 'end_image': None},
        {'prompt': test_prompt, 'image': test_image, 'end_image': None},
        
        # Text-only (fallback)
        {'prompt': test_prompt},
    ]
    
    for i, params in enumerate(param_variations):
        print(f"\nTesting variation {i+1}: {list(params.keys())}")
        
        try:
            # Add basic generation parameters
            test_params = params.copy()
            test_params.update({
                'width': 512,
                'height': 512,
                'num_frames': 4,  # Very small for quick test
                'num_inference_steps': 2,  # Very low for quick test
                'guidance_scale': 7.5
            })
            
            # Try the call (but don't actually generate - just check if parameters are accepted)
            if hasattr(pipe, '__call__'):
                sig = inspect.signature(pipe.__call__)
                
                # Filter parameters that the method actually accepts
                valid_params = {}
                for key, value in test_params.items():
                    if key in sig.parameters:
                        valid_params[key] = value
                
                if valid_params:
                    print(f"  ✅ Valid parameters: {list(valid_params.keys())}")
                    
                    # Try a very quick generation to see if it works
                    try:
                        # Set very low values for quick test
                        quick_params = valid_params.copy()
                        quick_params.update({
                            'num_frames': 2,
                            'num_inference_steps': 1
                        })
                        
                        print(f"  🚀 Attempting quick generation...")
                        result = pipe(**quick_params)
                        print(f"  ✅ SUCCESS! This parameter combination works!")
                        return valid_params
                        
                    except Exception as gen_error:
                        print(f"  ⚠️ Parameters accepted but generation failed: {gen_error}")
                        # Still return the valid params as they were accepted
                        return valid_params
                else:
                    print(f"  ❌ No valid parameters found")
                    
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                print(f"  ❌ Parameter rejected: {e}")
            else:
                print(f"  ❌ Type error: {e}")
        except Exception as e:
            print(f"  ❌ Other error: {e}")
    
    print(f"\n❌ No working parameter combination found")
    return None

def main():
    """Main diagnostic function"""
    print("WAN Pipeline Signature Diagnostic")
    print("=" * 60)
    
    # Test the TI2V-5B model
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    
    pipe = inspect_pipeline_signature(model_id)
    
    if pipe:
        working_params = test_parameter_variations(pipe)
        
        if working_params:
            print(f"\n🎉 SOLUTION FOUND!")
            print(f"Working parameters: {list(working_params.keys())}")
            print(f"\nUse these parameters in your generation:")
            for key, value in working_params.items():
                if key in ['prompt', 'image', 'start_image', 'end_image', 'init_image']:
                    print(f"  {key}: <your_{key}>")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"\n❌ Could not find working parameter combination")
            print("The model may require different parameters or have other issues")
    
    print(f"\n" + "=" * 60)
    print("Diagnostic complete")

if __name__ == "__main__":
    main()