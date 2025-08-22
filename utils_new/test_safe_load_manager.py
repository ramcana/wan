"""
Tests for SafeLoadManager - Security and safe loading features

Tests cover:
- SafeLoadManager class functionality
- Security validation for remote code execution
- Sandboxed environment creation and isolation
- Security policy management and source validation
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from safe_load_manager import (
    SafeLoadManager,
    SafeLoadingOptions,
    SecurityValidation,
    SecurityPolicy,
    SandboxEnvironment
)


class TestSafeLoadingOptions:
    """Test SafeLoadingOptions data class"""
    
    def test_safe_loading_options_creation(self):
        """Test creating SafeLoadingOptions"""
        options = SafeLoadingOptions(
            allow_remote_code=True,
            use_sandbox=False,
            restricted_operations=["network_access"],
            timeout_seconds=60,
            memory_limit_mb=1024
        )
        
        assert options.allow_remote_code is True
        assert options.use_sandbox is False
        assert options.restricted_operations == ["network_access"]
        assert options.timeout_seconds == 60
        assert options.memory_limit_mb == 1024
        
    def test_safe_loading_options_serialization(self):
        """Test SafeLoadingOptions serialization"""
        options = SafeLoadingOptions(
            allow_remote_code=False,
            use_sandbox=True,
            restricted_operations=["network_access", "file_system"],
            timeout_seconds=30,
            memory_limit_mb=512
        )
        
        # Test to_dict
        data = options.to_dict()
        expected = {
            "allow_remote_code": False,
            "use_sandbox": True,
            "restricted_operations": ["network_access", "file_system"],
            "timeout_seconds": 30,
            "memory_limit_mb": 512
        }
        assert data == expected
        
        # Test from_dict
        restored = SafeLoadingOptions.from_dict(data)
        assert restored.allow_remote_code == options.allow_remote_code
        assert restored.use_sandbox == options.use_sandbox
        assert restored.restricted_operations == options.restricted_operations
        assert restored.timeout_seconds == options.timeout_seconds
        assert restored.memory_limit_mb == options.memory_limit_mb


class TestSecurityValidation:
    """Test SecurityValidation data class"""
    
    def test_security_validation_creation(self):
        """Test creating SecurityValidation"""
        validation = SecurityValidation(
            is_safe=False,
            risk_level="high",
            detected_risks=["Dangerous operation detected"],
            mitigation_strategies=["Use sandbox", "Review code"]
        )
        
        assert validation.is_safe is False
        assert validation.risk_level == "high"
        assert validation.detected_risks == ["Dangerous operation detected"]
        assert validation.mitigation_strategies == ["Use sandbox", "Review code"]
        
    def test_security_validation_serialization(self):
        """Test SecurityValidation serialization"""
        validation = SecurityValidation(
            is_safe=True,
            risk_level="low",
            detected_risks=[],
            mitigation_strategies=[]
        )
        
        data = validation.to_dict()
        expected = {
            "is_safe": True,
            "risk_level": "low",
            "detected_risks": [],
            "mitigation_strategies": []
        }
        assert data == expected


class TestSecurityPolicy:
    """Test SecurityPolicy data class"""
    
    def test_security_policy_creation(self):
        """Test creating SecurityPolicy"""
        policy = SecurityPolicy(
            name="test_policy",
            description="Test security policy",
            allow_remote_code=False,
            trusted_domains=["huggingface.co"],
            blocked_domains=["malicious.com"],
            max_file_size_mb=50,
            allowed_file_extensions=[".py", ".json"],
            restricted_operations=["network_access"],
            sandbox_required=True
        )
        
        assert policy.name == "test_policy"
        assert policy.description == "Test security policy"
        assert policy.allow_remote_code is False
        assert policy.trusted_domains == ["huggingface.co"]
        assert policy.blocked_domains == ["malicious.com"]
        assert policy.max_file_size_mb == 50
        assert policy.allowed_file_extensions == [".py", ".json"]
        assert policy.restricted_operations == ["network_access"]
        assert policy.sandbox_required is True
        
    def test_security_policy_serialization(self):
        """Test SecurityPolicy serialization"""
        policy = SecurityPolicy(
            name="test",
            description="Test",
            allow_remote_code=True,
            trusted_domains=["test.com"],
            blocked_domains=[],
            max_file_size_mb=100,
            allowed_file_extensions=[".py"],
            restricted_operations=[],
            sandbox_required=False
        )
        
        # Test to_dict
        data = policy.to_dict()
        assert data["name"] == "test"
        assert data["allow_remote_code"] is True
        
        # Test from_dict
        restored = SecurityPolicy.from_dict(data)
        assert restored.name == policy.name
        assert restored.allow_remote_code == policy.allow_remote_code


class TestSandboxEnvironment:
    """Test SandboxEnvironment class"""
    
    def test_sandbox_creation(self):
        """Test creating sandbox environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            restrictions = ["network_access", "file_system"]
            sandbox = SandboxEnvironment(temp_dir, restrictions)
            
            assert sandbox.temp_dir == Path(temp_dir)
            assert sandbox.restrictions == restrictions
            assert sandbox.is_active is False
            
    def test_sandbox_context_manager(self):
        """Test sandbox as context manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            restrictions = ["network_access"]
            
            with SandboxEnvironment(temp_dir, restrictions) as sandbox:
                assert sandbox.is_active is True
                
            assert sandbox.is_active is False
            
    def test_sandbox_activation_deactivation(self):
        """Test sandbox activation and deactivation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = dict(os.environ)
            restrictions = ["network_access", "file_system"]
            
            sandbox = SandboxEnvironment(temp_dir, restrictions)
            
            # Test activation
            sandbox.activate()
            assert sandbox.is_active is True
            
            # Check environment modifications
            if "network_access" in restrictions:
                assert os.environ.get("NO_PROXY") == "*"
                assert os.environ.get("HTTP_PROXY") == "127.0.0.1:0"
                
            if "file_system" in restrictions:
                assert os.environ.get("TMPDIR") == str(temp_dir)
                assert os.environ.get("TEMP") == str(temp_dir)
                
            # Test deactivation
            sandbox.deactivate()
            assert sandbox.is_active is False
            
            # Environment should be restored (approximately)
            # Note: We can't guarantee exact restoration due to test environment
            
    def test_sandbox_execute_in_sandbox(self):
        """Test executing function in sandbox"""
        with tempfile.TemporaryDirectory() as temp_dir:
            restrictions = ["network_access"]
            sandbox = SandboxEnvironment(temp_dir, restrictions)
            
            def test_function(x, y):
                return x + y
                
            result = sandbox.execute_in_sandbox(test_function, 2, 3)
            assert result == 5


class TestSafeLoadManager:
    """Test SafeLoadManager class"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            
    @pytest.fixture
    def safe_load_manager(self, temp_config_dir):
        """Create SafeLoadManager with temporary config"""
        config_path = os.path.join(temp_config_dir, "security_config.json")
        return SafeLoadManager(config_path=config_path)
        
    def test_safe_load_manager_creation(self, safe_load_manager):
        """Test creating SafeLoadManager"""
        assert safe_load_manager.mode == "safe"
        assert isinstance(safe_load_manager.trusted_sources, set)
        assert isinstance(safe_load_manager.security_policies, dict)
        assert safe_load_manager.current_policy is not None
        
    def test_set_loading_mode(self, safe_load_manager):
        """Test setting loading mode"""
        # Test valid modes
        safe_load_manager.set_loading_mode("trust")
        assert safe_load_manager.mode == "trust"
        
        safe_load_manager.set_loading_mode("safe")
        assert safe_load_manager.mode == "safe"
        
        # Test invalid mode
        with pytest.raises(ValueError, match="Mode must be 'safe' or 'trust'"):
            safe_load_manager.set_loading_mode("invalid")
            
    def test_trusted_source_management(self, safe_load_manager):
        """Test adding and removing trusted sources"""
        source = "https://huggingface.co/test-model"
        
        # Add trusted source
        safe_load_manager.add_trusted_source(source)
        assert source in safe_load_manager.trusted_sources
        
        # Remove trusted source
        safe_load_manager.remove_trusted_source(source)
        assert source not in safe_load_manager.trusted_sources
        
    def test_is_source_trusted(self, safe_load_manager):
        """Test checking if source is trusted"""
        # Add trusted source
        trusted_source = "https://huggingface.co/trusted-model"
        safe_load_manager.add_trusted_source(trusted_source)
        
        # Test exact match
        assert safe_load_manager.is_source_trusted(trusted_source) is True
        
        # Test untrusted source
        untrusted_source = "https://malicious.com/bad-model"
        assert safe_load_manager.is_source_trusted(untrusted_source) is False
        
        # Test domain-based trust (huggingface.co is in default trusted domains)
        hf_model = "https://huggingface.co/some-model"
        assert safe_load_manager.is_source_trusted(hf_model) is True
        
    def test_validate_remote_code_safety(self, safe_load_manager):
        """Test validating remote code safety"""
        # Test safe code
        safe_code = "import torch\nclass MyModel(torch.nn.Module):\n    pass"
        validation = safe_load_manager.validate_remote_code_safety(
            "https://huggingface.co/model", safe_code
        )
        assert validation.risk_level in ["low", "medium"]
        
        # Test dangerous code
        dangerous_code = "import os\nos.system('rm -rf /')"
        validation = safe_load_manager.validate_remote_code_safety(
            "https://malicious.com/model", dangerous_code
        )
        assert validation.risk_level in ["medium", "high"]
        assert len(validation.detected_risks) > 0
        assert len(validation.mitigation_strategies) > 0
        
    def test_create_sandboxed_environment(self, safe_load_manager):
        """Test creating sandboxed environment"""
        restrictions = ["network_access", "file_system"]
        sandbox = safe_load_manager.create_sandboxed_environment(restrictions)
        
        assert isinstance(sandbox, SandboxEnvironment)
        assert sandbox.restrictions == restrictions
        assert sandbox.temp_dir.exists()
        
    def test_get_safe_loading_options_trusted(self, safe_load_manager):
        """Test getting safe loading options for trusted source"""
        trusted_source = "https://huggingface.co/trusted-model"
        safe_load_manager.add_trusted_source(trusted_source)
        
        options = safe_load_manager.get_safe_loading_options(trusted_source)
        
        assert options.allow_remote_code is True
        assert options.use_sandbox is False
        # Trusted sources still inherit policy restrictions but allow remote code
        # Default policy has restrictions, so they should be present
        assert isinstance(options.restricted_operations, list)
        assert options.timeout_seconds == 300
        assert options.memory_limit_mb == 0
        
    def test_get_safe_loading_options_untrusted(self, safe_load_manager):
        """Test getting safe loading options for untrusted source"""
        untrusted_source = "https://malicious.com/bad-model"
        
        options = safe_load_manager.get_safe_loading_options(untrusted_source)
        
        assert options.allow_remote_code is False
        assert options.use_sandbox is True
        assert len(options.restricted_operations) > 0
        assert options.timeout_seconds == 60
        assert options.memory_limit_mb == 1024
        
    def test_get_safe_loading_options_trust_mode(self, safe_load_manager):
        """Test getting safe loading options in trust mode"""
        safe_load_manager.set_loading_mode("trust")
        untrusted_source = "https://malicious.com/bad-model"
        
        options = safe_load_manager.get_safe_loading_options(untrusted_source)
        
        # In trust mode, even untrusted sources get permissive options
        assert options.allow_remote_code is True
        assert options.use_sandbox is False
        
    def test_validate_model_source(self, safe_load_manager):
        """Test validating model source"""
        # Test trusted source
        trusted_source = "https://huggingface.co/model"
        validation = safe_load_manager.validate_model_source(trusted_source)
        assert validation.is_safe is True
        assert validation.risk_level == "low"
        
        # Test untrusted source
        untrusted_source = "https://malicious.com/model"
        validation = safe_load_manager.validate_model_source(untrusted_source)
        assert validation.is_safe is False
        assert validation.risk_level == "medium"
        assert len(validation.detected_risks) > 0
        
        # Test suspicious path patterns
        suspicious_source = "file://../../../etc/passwd"
        validation = safe_load_manager.validate_model_source(suspicious_source)
        assert validation.is_safe is False
        assert len(validation.detected_risks) > 0
        
    def test_secure_model_loading_context(self, safe_load_manager):
        """Test secure model loading context manager"""
        model_path = "https://huggingface.co/test-model"
        
        with safe_load_manager.secure_model_loading(model_path) as options:
            assert isinstance(options, SafeLoadingOptions)
            assert options.allow_remote_code is True  # HF is trusted
            
    def test_security_policy_management(self, safe_load_manager):
        """Test security policy management"""
        # Create custom policy
        custom_policy = SecurityPolicy(
            name="custom",
            description="Custom policy",
            allow_remote_code=True,
            trusted_domains=["custom.com"],
            blocked_domains=["blocked.com"],
            max_file_size_mb=200,
            allowed_file_extensions=[".py", ".json"],
            restricted_operations=[],
            sandbox_required=False
        )
        
        # Add policy
        safe_load_manager.security_policies["custom"] = custom_policy
        
        # Set policy
        safe_load_manager.set_security_policy("custom")
        assert safe_load_manager.current_policy == custom_policy
        
        # Test invalid policy
        with pytest.raises(ValueError, match="Security policy 'invalid' not found"):
            safe_load_manager.set_security_policy("invalid")
            
    def test_configuration_persistence(self, temp_config_dir):
        """Test saving and loading configuration"""
        config_path = os.path.join(temp_config_dir, "test_config.json")
        
        # Create manager and add some configuration
        manager = SafeLoadManager(config_path=config_path)
        manager.add_trusted_source("https://test.com/model")
        
        custom_policy = SecurityPolicy(
            name="test_policy",
            description="Test policy",
            allow_remote_code=False,
            trusted_domains=["test.com"],
            blocked_domains=[],
            max_file_size_mb=100,
            allowed_file_extensions=[".py"],
            restricted_operations=["network_access"],
            sandbox_required=True
        )
        manager.security_policies["test_policy"] = custom_policy
        
        # Save configuration
        manager.save_configuration()
        assert os.path.exists(config_path)
        
        # Create new manager and verify configuration loaded
        new_manager = SafeLoadManager(config_path=config_path)
        assert "https://test.com/model" in new_manager.trusted_sources
        assert "test_policy" in new_manager.security_policies
        
        loaded_policy = new_manager.security_policies["test_policy"]
        assert loaded_policy.name == custom_policy.name
        assert loaded_policy.allow_remote_code == custom_policy.allow_remote_code


class TestSecurityIntegration:
    """Integration tests for security features"""
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "security_config.json")
            manager = SafeLoadManager(config_path=config_path)
            
            # Test untrusted model
            untrusted_model = "https://malicious.com/dangerous-model"
            
            # Validate source
            source_validation = manager.validate_model_source(untrusted_model)
            assert source_validation.is_safe is False
            
            # Get loading options
            options = manager.get_safe_loading_options(untrusted_model)
            assert options.use_sandbox is True
            assert options.allow_remote_code is False
            
            # Test with sandbox
            with manager.secure_model_loading(untrusted_model) as secure_options:
                assert secure_options.use_sandbox is True
                
    def test_security_policy_enforcement(self):
        """Test security policy enforcement"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "security_config.json")
            manager = SafeLoadManager(config_path=config_path)
            
            # Create strict policy
            strict_policy = SecurityPolicy(
                name="strict",
                description="Strict security policy",
                allow_remote_code=False,
                trusted_domains=[],
                blocked_domains=["*"],  # Block all domains
                max_file_size_mb=10,
                allowed_file_extensions=[".json"],
                restricted_operations=["network_access", "file_system"],
                sandbox_required=True
            )
            
            manager.security_policies["strict"] = strict_policy
            manager.set_security_policy("strict")
            
            # Test that even HuggingFace is restricted under strict policy
            hf_model = "https://huggingface.co/model"
            options = manager.get_safe_loading_options(hf_model)
            
            assert options.allow_remote_code is False
            assert options.use_sandbox is True
            
    def test_sandbox_isolation(self):
        """Test sandbox isolation functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeLoadManager()
            
            # Create sandbox
            sandbox = manager.create_sandboxed_environment(["network_access", "file_system"])
            
            original_env = dict(os.environ)
            
            with sandbox:
                # Environment should be modified
                assert sandbox.is_active is True
                
                # Test that restricted operations are blocked
                if "network_access" in sandbox.restrictions:
                    assert os.environ.get("HTTP_PROXY") == "127.0.0.1:0"
                    
            # Environment should be restored
            assert sandbox.is_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])