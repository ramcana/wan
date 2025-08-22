"""
Integration tests for SafeLoadManager with other system components

Tests the integration of SafeLoadManager with:
- DependencyManager for remote code handling
- ArchitectureDetector for model validation
- PipelineManager for secure pipeline loading
- WanPipelineLoader for safe model loading
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from safe_load_manager import SafeLoadManager, SecurityPolicy, SafeLoadingOptions


class TestSafeLoadManagerIntegration:
    """Integration tests for SafeLoadManager"""
    
    @pytest.fixture
    def safe_load_manager(self):
        """Create SafeLoadManager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            manager = SafeLoadManager(config_path=config_path)
            yield manager
            
    def test_integration_with_dependency_manager(self, safe_load_manager):
        """Test SafeLoadManager integration with dependency management"""
        # Mock dependency manager
        mock_dependency_manager = Mock()
        mock_dependency_manager.check_remote_code_availability.return_value = Mock(
            is_available=True,
            source_url="https://huggingface.co/model/pipeline.py",
            version="1.0.0"
        )
        
        # Test secure dependency checking
        model_path = "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Validate source first
        source_validation = safe_load_manager.validate_model_source(model_path)
        assert source_validation.is_safe is True
        
        # Get safe loading options
        options = safe_load_manager.get_safe_loading_options(model_path)
        assert options.allow_remote_code is True
        
        # Simulate dependency manager using safe loading options
        if options.allow_remote_code:
            remote_code_status = mock_dependency_manager.check_remote_code_availability(model_path)
            assert remote_code_status.is_available is True
            
    def test_integration_with_architecture_detector(self, safe_load_manager):
        """Test SafeLoadManager integration with architecture detection"""
        # Mock architecture detector
        mock_detector = Mock()
        mock_detector.detect_model_architecture.return_value = Mock(
            architecture_type="wan_t2v",
            requires_custom_pipeline=True,
            security_requirements=["trust_remote_code"]
        )
        
        model_path = "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Use secure loading context
        with safe_load_manager.secure_model_loading(model_path) as options:
            # Simulate architecture detection within secure context
            if options.allow_remote_code:
                architecture = mock_detector.detect_model_architecture(model_path)
                assert architecture.architecture_type == "wan_t2v"
                assert architecture.requires_custom_pipeline is True
                
    def test_integration_with_pipeline_manager(self, safe_load_manager):
        """Test SafeLoadManager integration with pipeline management"""
        # Mock pipeline manager
        mock_pipeline_manager = Mock()
        mock_pipeline_manager.select_pipeline_class.return_value = "WanPipeline"
        mock_pipeline_manager.validate_pipeline_args.return_value = Mock(
            is_valid=True,
            missing_args=[],
            security_warnings=[]
        )
        
        model_path = "https://huggingface.co/test-model"
        safe_load_manager.add_trusted_source(model_path)
        
        # Get safe loading options
        options = safe_load_manager.get_safe_loading_options(model_path)
        
        # Simulate pipeline manager using security context
        if options.allow_remote_code:
            pipeline_class = mock_pipeline_manager.select_pipeline_class("wan_t2v")
            validation = mock_pipeline_manager.validate_pipeline_args(
                pipeline_class, {"trust_remote_code": True}
            )
            assert validation.is_valid is True
            
    def test_integration_with_wan_pipeline_loader(self, safe_load_manager):
        """Test SafeLoadManager integration with WanPipelineLoader"""
        # Mock WanPipelineLoader
        mock_loader = Mock()
        mock_loader.load_wan_pipeline.return_value = Mock(
            pipeline=Mock(),
            security_context="trusted",
            optimizations_applied=["mixed_precision"]
        )
        
        model_path = "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Use secure loading context
        with safe_load_manager.secure_model_loading(model_path) as options:
            # Simulate WanPipelineLoader using security options
            loader_kwargs = {
                "trust_remote_code": options.allow_remote_code,
                "use_sandbox": options.use_sandbox,
                "timeout": options.timeout_seconds
            }
            
            if options.allow_remote_code:
                result = mock_loader.load_wan_pipeline(model_path, **loader_kwargs)
                assert result.security_context == "trusted"
                
    def test_security_policy_enforcement_integration(self, safe_load_manager):
        """Test security policy enforcement across components"""
        # Create strict security policy
        strict_policy = SecurityPolicy(
            name="strict_integration",
            description="Strict policy for integration testing",
            allow_remote_code=False,
            trusted_domains=[],
            blocked_domains=["*"],
            max_file_size_mb=10,
            allowed_file_extensions=[".json"],
            restricted_operations=["network_access", "file_system"],
            sandbox_required=True
        )
        
        safe_load_manager.security_policies["strict_integration"] = strict_policy
        safe_load_manager.set_security_policy("strict_integration")
        
        # Test that strict policy affects all components
        untrusted_model = "https://untrusted.com/model"
        
        # Source validation should fail
        source_validation = safe_load_manager.validate_model_source(untrusted_model)
        assert source_validation.is_safe is False
        
        # Loading options should be restrictive
        options = safe_load_manager.get_safe_loading_options(untrusted_model)
        assert options.allow_remote_code is False
        assert options.use_sandbox is True
        assert "network_access" in options.restricted_operations
        
        # Mock components should respect security restrictions
        mock_components = {
            "dependency_manager": Mock(),
            "pipeline_manager": Mock(),
            "wan_loader": Mock()
        }
        
        # Simulate component behavior under strict policy
        for component_name, component in mock_components.items():
            if not options.allow_remote_code:
                # Components should not attempt remote operations
                component.fetch_remote_code = Mock(side_effect=SecurityError("Remote code disabled"))
                component.trust_remote_code = False
                
    def test_sandbox_isolation_integration(self, safe_load_manager):
        """Test sandbox isolation with integrated components"""
        untrusted_model = "https://malicious.com/dangerous-model"
        
        # Create sandbox for untrusted model
        with safe_load_manager.secure_model_loading(untrusted_model) as options:
            assert options.use_sandbox is True
            
            # Create actual sandbox
            sandbox = safe_load_manager.create_sandboxed_environment(
                options.restricted_operations
            )
            
            with sandbox:
                # Verify sandbox environment modifications
                assert sandbox.is_active is True
                
                # Check that network restrictions are in place
                if "network_access" in sandbox.restrictions:
                    assert os.environ.get("HTTP_PROXY") == "127.0.0.1:0"
                    assert os.environ.get("NO_PROXY") == "*"
                
                # Test that sandbox directory is set up
                if "file_system" in sandbox.restrictions:
                    assert os.environ.get("TMPDIR") == str(sandbox.temp_dir)
                    
                # Mock a function that would be affected by sandbox
                def mock_restricted_operation():
                    # Check if we're in a restricted environment
                    proxy_set = os.environ.get("HTTP_PROXY") == "127.0.0.1:0"
                    return "restricted" if proxy_set else "unrestricted"
                
                # Execute in sandbox
                result = sandbox.execute_in_sandbox(mock_restricted_operation)
                if "network_access" in options.restricted_operations:
                    assert result == "restricted"
                
    def test_error_handling_integration(self, safe_load_manager):
        """Test error handling across integrated components"""
        problematic_model = "https://broken.com/corrupted-model"
        
        # Test error propagation through security layers
        source_validation = safe_load_manager.validate_model_source(problematic_model)
        assert source_validation.is_safe is False
        
        # Mock component errors
        mock_errors = {
            "network_error": ConnectionError("Network unreachable"),
            "security_error": SecurityError("Untrusted source"),
            "validation_error": ValueError("Invalid model format")
        }
        
        for error_type, error in mock_errors.items():
            # Test that SafeLoadManager handles component errors gracefully
            try:
                with safe_load_manager.secure_model_loading(problematic_model) as options:
                    # Simulate component raising error
                    if error_type == "security_error" and not options.allow_remote_code:
                        raise error
                    elif error_type == "network_error" and options.use_sandbox:
                        # Network errors expected in sandbox
                        pass
                    elif error_type == "validation_error":
                        # Validation errors should be caught and handled
                        raise error
                        
            except SecurityError:
                # Security errors should be properly handled
                assert error_type == "security_error"
            except ValueError:
                # Validation errors should be properly handled
                assert error_type == "validation_error"
                
    def test_configuration_integration(self, safe_load_manager):
        """Test configuration integration across components"""
        # Create comprehensive configuration
        config = {
            "security_policies": {
                "production": {
                    "name": "production",
                    "description": "Production security policy",
                    "allow_remote_code": True,
                    "trusted_domains": ["huggingface.co", "hf.co"],
                    "blocked_domains": ["malicious.com"],
                    "max_file_size_mb": 1000,
                    "allowed_file_extensions": [".py", ".json", ".txt"],
                    "restricted_operations": [],
                    "sandbox_required": False
                },
                "development": {
                    "name": "development", 
                    "description": "Development security policy",
                    "allow_remote_code": True,
                    "trusted_domains": ["huggingface.co", "localhost", "127.0.0.1"],
                    "blocked_domains": [],
                    "max_file_size_mb": 500,
                    "allowed_file_extensions": [".py", ".json", ".txt", ".yaml"],
                    "restricted_operations": ["network_access"],
                    "sandbox_required": False
                }
            },
            "trusted_sources": [
                "https://huggingface.co/Wan-AI/",
                "https://huggingface.co/microsoft/",
                "/local/models/"
            ]
        }
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
            
        try:
            # Create manager with configuration
            manager = SafeLoadManager(config_path=config_path)
            
            # Verify configuration loaded
            assert "production" in manager.security_policies
            assert "development" in manager.security_policies
            assert len(manager.trusted_sources) == 3
            
            # Test policy switching
            manager.set_security_policy("production")
            prod_options = manager.get_safe_loading_options("https://huggingface.co/model")
            assert prod_options.allow_remote_code is True
            assert prod_options.use_sandbox is False
            
            manager.set_security_policy("development")
            # localhost should be trusted in development policy, so restrictions should apply but be lenient
            dev_options = manager.get_safe_loading_options("https://localhost/model")
            assert dev_options.allow_remote_code is True
            # For trusted sources, we still apply policy restrictions but allow remote code
            assert dev_options.restricted_operations == ["network_access"]
            
        finally:
            os.unlink(config_path)
            
    def test_performance_integration(self, safe_load_manager):
        """Test performance impact of security features"""
        import time
        
        models_to_test = [
            "https://huggingface.co/trusted-model",
            "https://untrusted.com/suspicious-model"
        ]
        
        performance_results = {}
        
        for model in models_to_test:
            start_time = time.time()
            
            # Measure security validation time
            source_validation = safe_load_manager.validate_model_source(model)
            validation_time = time.time() - start_time
            
            # Measure loading options time
            start_time = time.time()
            options = safe_load_manager.get_safe_loading_options(model)
            options_time = time.time() - start_time
            
            # Measure sandbox creation time (if needed)
            sandbox_time = 0
            if options.use_sandbox:
                start_time = time.time()
                sandbox = safe_load_manager.create_sandboxed_environment()
                sandbox_time = time.time() - start_time
                
            performance_results[model] = {
                "validation_time": validation_time,
                "options_time": options_time,
                "sandbox_time": sandbox_time,
                "total_time": validation_time + options_time + sandbox_time
            }
            
        # Verify performance is reasonable (< 1 second for security operations)
        for model, results in performance_results.items():
            assert results["total_time"] < 1.0, f"Security operations too slow for {model}"
            assert results["validation_time"] < 0.1, f"Validation too slow for {model}"
            assert results["options_time"] < 0.1, f"Options generation too slow for {model}"


class SecurityError(Exception):
    """Custom security error for testing"""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])