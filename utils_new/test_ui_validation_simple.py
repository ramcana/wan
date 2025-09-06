"""
Simple UI Validation Tests

Tests the UI validation system without heavy dependencies.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ui_validation_manager_creation():
    """Test that UIValidationManager can be created"""
    from ui_validation import UIValidationManager
    
    config = {
        "max_prompt_length": 500,
        "enable_realtime_validation": True
    }
    
    manager = UIValidationManager(config)
    assert manager is not None
    assert manager.config == config
    assert manager.enable_realtime == True

def test_prompt_validation_basic():
    """Test basic prompt validation"""
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager()
    
    # Test empty prompt
    validation_html, is_valid, char_count = manager.validate_prompt_realtime("", "t2v-A14B")
    assert char_count == "0/500"
    assert is_valid
    assert validation_html == ""
    
    # Test normal prompt
    prompt = "A beautiful sunset"
    validation_html, is_valid, char_count = manager.validate_prompt_realtime(prompt, "t2v-A14B")
    assert char_count == f"{len(prompt)}/500"
    assert is_valid

def test_validation_state_creation():
    """Test UIValidationState creation"""
    from ui_validation import UIValidationState
    
    state = UIValidationState()
    assert state.is_valid == True
    assert state.errors == []
    assert state.warnings == []
    assert state.suggestions == []

def test_progress_indicator_creation():
    """Test progress indicator HTML creation"""
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager()
    
    progress_html = manager.create_progress_indicator("validation", 0.5, "Testing...")
    assert "validation" in progress_html.lower()
    assert "50.0%" in progress_html
    assert "testing" in progress_html.lower()
    assert "progress-container" in progress_html

def test_error_display_creation():
    """Test error display creation"""
    from ui_validation import UIValidationManager
    from error_handler import UserFriendlyError, ErrorCategory, ErrorSeverity
    
    manager = UIValidationManager()
    
    # Create a test error
    error = UserFriendlyError(
        category=ErrorCategory.INPUT_VALIDATION,
        severity=ErrorSeverity.MEDIUM,
        title="Test Error",
        message="This is a test error",
        recovery_suggestions=["Try again", "Check inputs"],
        recovery_actions=[]
    )
    
    error_html, show_display = manager.create_error_display_with_recovery(error, "test")
    assert show_display
    assert "test error" in error_html.lower()
    assert "try again" in error_html.lower()

def test_validation_summary_creation():
    """Test validation summary creation"""
    from ui_validation import UIValidationManager, UIValidationState
    
    manager = UIValidationManager()
    
    # Test with no validation states (should show success)
    summary_html, all_valid = manager.create_comprehensive_validation_summary()
    assert all_valid
    assert "all validations passed" in summary_html.lower()
    
    # Test with error states
    manager.validation_states = {
        'prompt': UIValidationState(
            is_valid=False,
            errors=["Test error"],
            warnings=[],
            suggestions=[]
        )
    }
    
    summary_html, all_valid = manager.create_comprehensive_validation_summary()
    assert not all_valid
    assert "test error" in summary_html.lower()

def test_component_registration():
    """Test UI component registration"""
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager()
    
    mock_components = {
        'prompt_input': Mock(),
        'image_input': Mock(),
        'resolution': Mock()
    }
    
    manager.register_ui_components(mock_components)
    assert manager.ui_components == mock_components

if __name__ == "__main__":
    pytest.main([__file__, "-v"])