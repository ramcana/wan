#!/usr/bin/env python3
"""
Debug Pipeline Loading Issue
Isolate and identify which component is causing the 75% hang
"""

import torch
import logging
import time
from diffusers import DiffusionPipeline
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_loading():
    """Test pipeline loading with detailed logging"""
    print("="*60)
    print("DEBUGGING PIPELINE LOADING ISSUE")
    print("="*60)
    
    # Test with a simple model first
    model_paths = [
        "local_installation/models/WAN2.2-T2V-A14B",
        "local_installation/models/WAN2.2-I2V-A14B", 
        "local_installation/models/WAN2.2-TI2V-5B"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing model: {model_path}")
            try:
                print("  Starting pipeline loading...")
                start_time = time.time()
                
                # Try loading with minimal parameters
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                load_time = time.time() - start_time
                print(f"  ✅ SUCCESS: Loaded in {load_time:.2f}s")
                print(f"  Pipeline type: {type(pipeline)}")
                
                # Clean up
                del pipeline
                torch.cuda.empty_cache()
                
                return True
                
            except Exception as e:
                load_time = time.time() - start_time
                print(f"  ❌ FAILED after {load_time:.2f}s: {e}")
                print(f"  Error type: {type(e)}")
                
                # Try to get more details
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                
        else:
            print(f"  ⚠️ Model path not found: {model_path}")
    
    return False

def test_component_loading():
    """Test loading individual components"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENT LOADING")
    print("="*60)
    
    model_path = "local_installation/models/WAN2.2-T2V-A14B"
    if not os.path.exists(model_path):
        print("Model path not found, skipping component test")
        return
        
    components = ["unet", "vae", "text_encoder", "scheduler", "tokenizer"]
    
    for component in components:
        print(f"\n🔍 Testing component: {component}")
        try:
            start_time = time.time()
            
            if component == "unet":
                from diffusers import UNet2DConditionModel
                model = UNet2DConditionModel.from_pretrained(
                    model_path, subfolder="unet", torch_dtype=torch.float16
                )
            elif component == "vae":
                from diffusers import AutoencoderKL
                model = AutoencoderKL.from_pretrained(
                    model_path, subfolder="vae", torch_dtype=torch.float16
                )
            elif component == "text_encoder":
                from transformers import CLIPTextModel
                model = CLIPTextModel.from_pretrained(
                    model_path, subfolder="text_encoder", torch_dtype=torch.float16
                )
            elif component == "scheduler":
                from diffusers import DDIMScheduler
                model = DDIMScheduler.from_pretrained(
                    model_path, subfolder="scheduler"
                )
            elif component == "tokenizer":
                from transformers import CLIPTokenizer
                model = CLIPTokenizer.from_pretrained(
                    model_path, subfolder="tokenizer"
                )
            
            load_time = time.time() - start_time
            print(f"  ✅ SUCCESS: {component} loaded in {load_time:.2f}s")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            load_time = time.time() - start_time
            print(f"  ❌ FAILED: {component} failed after {load_time:.2f}s")
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("Starting pipeline loading debug...")
    
    # Test 1: Full pipeline loading
    success = test_pipeline_loading()
    
    # Test 2: Individual components (if full pipeline fails)
    if not success:
        test_component_loading()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)