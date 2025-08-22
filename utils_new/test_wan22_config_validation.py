"""
Tests for WAN22 Configuration Validation System

This module tests the comprehensive validation system for WAN22 configurations.
"""

import pytest
from wan22_config_validation import (
    ConfigurationValidator,
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    validate_config,
    get_validation_summary
)
from wan22_config_manager import (
    WAN22Config,
    OptimizationConfig,
    PipelineConfig,
    SecurityConfig,
    CompatibilityConfig,
    UserPreferences,
    OptimizationStrategy,
    PipelineSelectionMode,
    SecurityLevel
)


class TestValidationMessage:
    """Test validation message functionality"""
    
    def test_validation_message_creation(self):
        """Test creating validation messages"""
        msg = ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TEST_CODE",
            message="Test message",
            field_path="test.field",
            current_value="current",
            suggested_value="suggested",
            help_text="Help text"
        )
        
        assert msg.severity == ValidationSeverity.WARNING
        assert msg.code == "TEST_CODE"
        assert msg.message == "Test message"
        assert msg.field_path == "test.field"
        assert msg.current_value == "current"
        assert msg.suggested_value == "suggested"
        assert msg.help_text == "Help text"


class TestValidationResult:
    """Test validation result functionality"""
    
    def test_validation_result_creation(self):
        """Test creating validation results"""
        messages = [
            ValidationMessage(ValidationSeverity.INFO, "INFO_CODE", "Info", "field1"),
            ValidationMessage(ValidationSeverity.WARNING, "WARN_CODE", "Warning", "field2"),
            ValidationMessage(ValidationSeverity.ERROR, "ERROR_CODE", "Error", "field3")
        ]
        
        result = ValidationResult(is_valid=False, messages=messages, score=0.7)
        
        assert result.is_valid is False
        assert len(result.messages) == 3
        assert result.score == 0.7
    
    def test_get_messages_by_severity(self):
        """Test filtering messages by severity"""
        messages = [
            ValidationMessage(ValidationSeverity.INFO, "INFO1", "Info 1", "field1"),
            ValidationMessage(ValidationSeverity.WARNING, "WARN1", "Warning 1", "field2"),
            ValidationMessage(ValidationSeverity.WARNING, "WARN2", "Warning 2", "field3"),
            ValidationMessage(ValidationSeverity.ERROR, "ERROR1", "Error 1", "field4")
        ]
        
        result = ValidationResult(is_valid=False, messages=messages, score=0.5)
        
        info_messages = result.get_messages_by_severity(ValidationSeverity.INFO)
        assert len(info_messages) == 1
        assert info_messages[0].code == "INFO1"
        
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert len(warning_messages) == 2
        
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert len(error_messages) == 1
    
    def test_has_errors(self):
        """Test checking for errors"""
        # No errors
        result = ValidationResult(
            is_valid=True,
            messages=[ValidationMessage(ValidationSeverity.INFO, "INFO", "Info", "field")],
            score=1.0
        )
        assert result.has_errors() is False
        
        # Has error
        result = ValidationResult(
            is_valid=False,
            messages=[ValidationMessage(ValidationSeverity.ERROR, "ERROR", "Error", "field")],
            score=0.5
        )
        assert result.has_errors() is True
        
        # Has critical error
        result = ValidationResult(
            is_valid=False,
            messages=[ValidationMessage(ValidationSeverity.CRITICAL, "CRITICAL", "Critical", "field")],
            score=0.0
        )
        assert result.has_errors() is True
    
    def test_has_warnings(self):
        """Test checking for warnings"""
        # No warnings
        result = ValidationResult(
            is_valid=True,
            messages=[ValidationMessage(ValidationSeverity.INFO, "INFO", "Info", "field")],
            score=1.0
        )
        assert result.has_warnings() is False
        
        # Has warnings
        result = ValidationResult(
            is_valid=True,
            messages=[ValidationMessage(ValidationSeverity.WARNING, "WARNING", "Warning", "field")],
            score=0.9
        )
        assert result.has_warnings() is True


class TestConfigurationValidator:
    """Test configuration validator"""
    
    def setup_method(self):
        """Set up test environment"""
        self.validator = ConfigurationValidator()
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration"""
        config = WAN22Config()
        result = self.validator.validate_config(config)
        
        assert result.is_valid is True
        assert result.score > 0.8
        assert not result.has_errors()
    
    def test_validate_structure_missing_version(self):
        """Test validation with missing version"""
        config = WAN22Config()
        config.version = None
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("version" in msg.message.lower() for msg in error_messages)
    
    def test_validate_structure_old_version(self):
        """Test validation with old version"""
        config = WAN22Config()
        config.version = "0.5.0"
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("outdated" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_optimization_invalid_strategy(self):
        """Test validation with invalid optimization strategy"""
        config = WAN22Config()
        config.optimization.strategy = "invalid_strategy"
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("strategy" in msg.message.lower() for msg in error_messages)
    
    def test_validate_optimization_low_vram_threshold(self):
        """Test validation with low VRAM threshold"""
        config = WAN22Config()
        config.optimization.vram_threshold_mb = 512
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("vram" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_optimization_invalid_chunk_size(self):
        """Test validation with invalid chunk size"""
        config = WAN22Config()
        config.optimization.max_chunk_size = -1
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("chunk" in msg.message.lower() for msg in error_messages)
    
    def test_validate_optimization_performance_with_offload(self):
        """Test validation of performance strategy with CPU offload"""
        config = WAN22Config()
        config.optimization.strategy = OptimizationStrategy.PERFORMANCE
        config.optimization.enable_cpu_offload = True
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("offload" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_pipeline_invalid_timeout(self):
        """Test validation with invalid pipeline timeout"""
        config = WAN22Config()
        config.pipeline.pipeline_timeout_seconds = -10
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("timeout" in msg.message.lower() for msg in error_messages)
    
    def test_validate_pipeline_excessive_retries(self):
        """Test validation with excessive retry attempts"""
        config = WAN22Config()
        config.pipeline.max_retry_attempts = 20
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("retry" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_pipeline_manual_no_preferred(self):
        """Test validation of manual mode without preferred pipeline"""
        config = WAN22Config()
        config.pipeline.selection_mode = PipelineSelectionMode.MANUAL
        config.pipeline.preferred_pipeline_class = None
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("manual" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_security_invalid_sandbox_timeout(self):
        """Test validation with invalid sandbox timeout"""
        config = WAN22Config()
        config.security.sandbox_timeout_seconds = 0
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("sandbox" in msg.message.lower() for msg in error_messages)
    
    def test_validate_security_strict_with_remote_code(self):
        """Test validation of strict security with remote code trust"""
        config = WAN22Config()
        config.security.security_level = SecurityLevel.STRICT
        config.security.trust_remote_code = True
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("strict" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_compatibility_invalid_cache_ttl(self):
        """Test validation with invalid cache TTL"""
        config = WAN22Config()
        config.compatibility.detection_cache_ttl_hours = -1
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("cache" in msg.message.lower() for msg in error_messages)
    
    def test_validate_user_preferences_invalid_fps(self):
        """Test validation with invalid FPS"""
        config = WAN22Config()
        config.user_preferences.default_fps = -5.0
        
        result = self.validator.validate_config(config)
        
        assert result.has_errors()
        error_messages = result.get_messages_by_severity(ValidationSeverity.ERROR)
        assert any("fps" in msg.message.lower() for msg in error_messages)
    
    def test_validate_user_preferences_high_concurrent(self):
        """Test validation with high concurrent generations"""
        config = WAN22Config()
        config.user_preferences.max_concurrent_generations = 10
        
        result = self.validator.validate_config(config)
        
        assert result.has_warnings()
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("concurrent" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_experimental_features_unknown(self):
        """Test validation with unknown experimental features"""
        config = WAN22Config()
        config.experimental_features = {
            "unknown_feature": True,
            "another_unknown": False
        }
        
        result = self.validator.validate_config(config)
        
        info_messages = result.get_messages_by_severity(ValidationSeverity.INFO)
        assert any("unknown" in msg.message.lower() for msg in info_messages)
    
    def test_validate_experimental_features_invalid_value(self):
        """Test validation with invalid experimental feature values"""
        config = WAN22Config()
        config.experimental_features = {
            "debug_mode": "yes"  # Should be boolean
        }
        
        result = self.validator.validate_config(config)
        
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("boolean" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_custom_settings_private_name(self):
        """Test validation with private custom setting names"""
        config = WAN22Config()
        config.custom_settings = {
            "_private_setting": "value"
        }
        
        result = self.validator.validate_config(config)
        
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("private" in msg.message.lower() for msg in warning_messages)
    
    def test_validate_cross_section_memory_concurrent(self):
        """Test cross-section validation for memory strategy with concurrent generations"""
        config = WAN22Config()
        config.optimization.strategy = OptimizationStrategy.MEMORY
        config.user_preferences.max_concurrent_generations = 3
        
        result = self.validator.validate_config(config)
        
        warning_messages = result.get_messages_by_severity(ValidationSeverity.WARNING)
        assert any("memory" in msg.message.lower() and "concurrent" in msg.message.lower() 
                  for msg in warning_messages)
    
    def test_calculate_score_no_messages(self):
        """Test score calculation with no messages"""
        score = self.validator._calculate_score([])
        assert score == 1.0
    
    def test_calculate_score_with_messages(self):
        """Test score calculation with various message severities"""
        messages = [
            ValidationMessage(ValidationSeverity.INFO, "INFO", "Info", "field1"),
            ValidationMessage(ValidationSeverity.WARNING, "WARN", "Warning", "field2"),
            ValidationMessage(ValidationSeverity.ERROR, "ERROR", "Error", "field3")
        ]
        
        score = self.validator._calculate_score(messages)
        assert 0.0 <= score <= 1.0
        assert score < 1.0  # Should be penalized for warnings and errors
    
    def test_is_valid_url_pattern(self):
        """Test URL pattern validation"""
        # Valid patterns
        assert self.validator._is_valid_url_pattern("huggingface.co") is True
        assert self.validator._is_valid_url_pattern("hf.co") is True
        assert self.validator._is_valid_url_pattern("example.com") is True
        assert self.validator._is_valid_url_pattern("sub.example.com") is True
        
        # Invalid patterns
        assert self.validator._is_valid_url_pattern("") is False
        assert self.validator._is_valid_url_pattern("invalid..domain") is False
        assert self.validator._is_valid_url_pattern(".invalid") is False
        assert self.validator._is_valid_url_pattern("invalid.") is False


class TestValidationFunctions:
    """Test validation utility functions"""
    
    def test_validate_config_function(self):
        """Test convenience validation function"""
        config = WAN22Config()
        result = validate_config(config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_get_validation_summary_valid(self):
        """Test validation summary for valid configuration"""
        result = ValidationResult(is_valid=True, messages=[], score=1.0)
        summary = get_validation_summary(result)
        
        assert "valid" in summary.lower()
        assert "✅" in summary
        assert "1.00" in summary
    
    def test_get_validation_summary_with_issues(self):
        """Test validation summary with various issues"""
        messages = [
            ValidationMessage(ValidationSeverity.INFO, "INFO", "Info", "field1"),
            ValidationMessage(ValidationSeverity.WARNING, "WARN", "Warning", "field2"),
            ValidationMessage(ValidationSeverity.ERROR, "ERROR", "Error", "field3"),
            ValidationMessage(ValidationSeverity.CRITICAL, "CRITICAL", "Critical", "field4")
        ]
        
        result = ValidationResult(is_valid=False, messages=messages, score=0.3)
        summary = get_validation_summary(result)
        
        assert "0.30" in summary
        assert "❌" in summary
        assert "🔴 1 critical" in summary
        assert "🟠 1 error" in summary
        assert "🟡 1 warning" in summary
        assert "ℹ️ 1 info" in summary
    
    def test_get_validation_summary_valid_with_warnings(self):
        """Test validation summary for valid config with warnings"""
        messages = [
            ValidationMessage(ValidationSeverity.WARNING, "WARN", "Warning", "field1"),
            ValidationMessage(ValidationSeverity.INFO, "INFO", "Info", "field2")
        ]
        
        result = ValidationResult(is_valid=True, messages=messages, score=0.9)
        summary = get_validation_summary(result)
        
        assert "0.90" in summary
        assert "✅" in summary
        assert "🟡 1 warning" in summary
        assert "ℹ️ 1 info" in summary


class TestValidationIntegration:
    """Integration tests for validation system"""
    
    def test_comprehensive_validation_workflow(self):
        """Test complete validation workflow"""
        # Create configuration with various issues
        config = WAN22Config()
        config.version = "0.5.0"  # Old version - warning
        config.optimization.max_chunk_size = -1  # Invalid - error
        config.optimization.vram_threshold_mb = 512  # Low - warning
        config.pipeline.max_retry_attempts = 15  # High - warning
        config.security.sandbox_timeout_seconds = 0  # Invalid - error
        config.user_preferences.default_fps = -10  # Invalid - error
        config.experimental_features = {"unknown_feature": True}  # Unknown - info
        
        # Validate
        result = validate_config(config)
        
        # Should not be valid due to errors
        assert result.is_valid is False
        assert result.has_errors() is True
        assert result.has_warnings() is True
        
        # Check specific issue counts
        errors = result.get_messages_by_severity(ValidationSeverity.ERROR)
        warnings = result.get_messages_by_severity(ValidationSeverity.WARNING)
        infos = result.get_messages_by_severity(ValidationSeverity.INFO)
        
        assert len(errors) >= 3  # chunk_size, sandbox_timeout, fps
        assert len(warnings) >= 3  # version, vram_threshold, retry_attempts
        assert len(infos) >= 1  # unknown_feature
        
        # Score should be reduced due to multiple issues
        assert result.score < 0.8
        
        # Get summary
        summary = get_validation_summary(result)
        assert "❌" in summary
        assert "error" in summary.lower()
        assert "warning" in summary.lower()
    
    def test_validation_with_preset_configs(self):
        """Test validation with different preset-style configurations"""
        from wan22_config_presets import ConfigurationPresets
        
        # Test all presets are valid
        preset_names = ConfigurationPresets.get_preset_names()
        
        for preset_name in preset_names:
            config = ConfigurationPresets.get_preset(preset_name)
            result = validate_config(config)
            
            # All presets should be valid (may have warnings but no errors)
            assert result.is_valid is True, f"Preset {preset_name} failed validation"
            assert not result.has_errors(), f"Preset {preset_name} has errors: {result.get_messages_by_severity(ValidationSeverity.ERROR)}"
            
            # Score should be reasonable
            assert result.score >= 0.7, f"Preset {preset_name} has low score: {result.score}"


if __name__ == "__main__":
    pytest.main([__file__])