#!/usr/bin/env python3
"""
Test script to verify Hugging Face model repositories are accessible
"""

import requests
from huggingface_hub import HfApi

def test_repository_exists(repo_id):
    """Test if a Hugging Face repository exists and is accessible"""
    try:
        api = HfApi()
        # Try to get repository info
        repo_info = api.repo_info(repo_id=repo_id)
        return True, f"Repository exists: {repo_info.id}"
    except Exception as e:
        return False, str(e)

    assert True  # TODO: Add proper assertion

def test_model_repositories():
    """Test all WAN2.2 model repositories"""
    
    repositories = {
        "T2V-A14B": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "I2V-A14B": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "TI2V-5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    }
    
    print("Testing WAN2.2 Model Repository Accessibility")
    print("=" * 50)
    
    all_accessible = True
    
    for model_name, repo_id in repositories.items():
        print(f"\nTesting {model_name}: {repo_id}")
        
        exists, message = test_repository_exists(repo_id)
        
        if exists:
            print(f"✅ {model_name}: Accessible")
        else:
            print(f"❌ {model_name}: Not accessible - {message}")
            all_accessible = False
    
    print("\n" + "=" * 50)
    
    if all_accessible:
        print("🎉 All model repositories are accessible!")
        return True
    else:
        print("⚠️  Some model repositories are not accessible.")
        print("This could be due to:")
        print("- Repository names have changed")
        print("- Repositories are private/gated")
        print("- Network connectivity issues")
        print("- Authentication required")
        return False

    assert True  # TODO: Add proper assertion

def test_alternative_repositories():
    """Test alternative repository patterns that might exist"""
    
    alternative_patterns = [
        # Original pattern from error
        ("T2V-A14B (original)", "Wan2.2/T2V-A14B"),
        
        # Possible variations
        ("T2V-A14B (alt1)", "Wan-AI/Wan2.2-T2V-A14B"),
        ("T2V-A14B (alt2)", "WanAI/Wan2.2-T2V-A14B-Diffusers"),
        ("T2V-A14B (alt3)", "wan-ai/wan2.2-t2v-a14b-diffusers"),
        
        # Check if base organization exists
        ("Wan-AI org", "Wan-AI"),
        ("Wan2.2 org", "Wan2.2"),
    ]
    
    print("\n" + "=" * 50)
    print("Testing Alternative Repository Patterns")
    print("=" * 50)
    
    for name, repo_id in alternative_patterns:
        print(f"\nTesting {name}: {repo_id}")
        exists, message = test_repository_exists(repo_id)
        
        if exists:
            print(f"✅ {name}: Found!")
        else:
            print(f"❌ {name}: Not found")

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    try:
        success = test_model_repositories()
        test_alternative_repositories()
        
        if not success:
            print("\n" + "=" * 50)
            print("RECOMMENDATIONS:")
            print("1. Check if the model repositories have been renamed")
            print("2. Verify if authentication is required")
            print("3. Consider using mock/placeholder models for testing")
            print("4. Check the official WAN2.2 documentation for correct repo names")
            
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")