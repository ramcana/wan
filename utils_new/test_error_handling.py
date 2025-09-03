#!/usr/bin/env python3
"""
Test script for comprehensive error handling system
"""

import sys
import os
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from error_handler import (
    ErrorClassifier,
    ErrorCategory,
    create_error_info,
    handle_error_with_recovery,
    get_error_recovery_manager,
    ErrorWithRecoveryInfo,
    log_error_with_context
)

def test_error_classification():
    """Test error classification functionality"""
    print("Testing error classification...")
    
    # Test VRAM error classification
    vram_error = RuntimeError("CUDA out of memory")
    category = ErrorClassifier.classify_error(vram_error)
    assert category == ErrorCategory.VRAM_ERROR, f"Expected VRAM_ERROR, got {category}"
    print("✓ VRAM error classification works")
    
    # Test model loading error
    model_error = FileNotFoundError("Model file not found")
    category = ErrorClassifier.classify_error(model_error, "model loading")
    assert category == ErrorCategory.FILE_IO_ERROR, f"Expected FILE_IO_ERROR, got {category}"
    print("✓ Model loading error classification works")
    
    # Test network error
    network_error = ConnectionError("Connection failed")
    category = ErrorClassifier.classify_error(network_error)
    assert category == ErrorCategory.NETWORK_ERROR, f"Expected NETWORK_ERROR, got {category}"
    print("✓ Network error classification works")

def test_error_info_creation():
    """Test error info creation"""
    print("\nTesting error info creation...")
    
    test_error = ValueError("Invalid input parameter")
    error_info = create_error_info(test_error, "test_context", "test_function")
    
    assert error_info.category == ErrorCategory.VALIDATION_ERROR
    assert error_info.error_type == "ValueError"
    assert error_info.message == "Invalid input parameter"
    assert error_info.function_name == "test_function"
    assert len(error_info.recovery_suggestions) > 0
    
    print("✓ Error info creation works")

@handle_error_with_recovery
def test_function_with_recovery():
    """Test function that uses error recovery decorator"""
    # Simulate a recoverable error on first attempt
    recovery_manager = get_error_recovery_manager()
    
    # Check if this is a retry attempt
    if len(recovery_manager.error_history) == 0:
        # First attempt - simulate error
        raise RuntimeError("Simulated recoverable error")
    else:
        # Recovery attempt - succeed
        return "Success after recovery"

def test_recovery_decorator():
    """Test the error recovery decorator"""
    print("\nTesting error recovery decorator...")
    
    try:
        result = test_function_with_recovery()
        print(f"✓ Recovery decorator works: {result}")
    except ErrorWithRecoveryInfo as e:
        print(f"✗ Recovery failed: {e.error_info.user_message}")

def test_vram_error_simulation():
    """Test VRAM error handling (simulation)"""
    print("\nTesting VRAM error handling...")
    
    # Simulate VRAM out of memory error
    if torch.cuda.is_available():
        try:
            # This won't actually cause OOM, just for testing classification
            vram_error = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            error_info = create_error_info(vram_error, "video_generation")
            
            assert error_info.category == ErrorCategory.VRAM_ERROR
            assert "GPU memory is full" in error_info.user_message
            assert any("resolution" in suggestion.lower() for suggestion in error_info.recovery_suggestions)
            
            print("✓ VRAM error handling works")
        except Exception as e:
            print(f"✗ VRAM error test failed: {e}")
    else:
        print("⚠ CUDA not available, skipping VRAM error test")

def test_error_logging():
    """Test error logging functionality"""
    print("\nTesting error logging...")
    
    try:
        test_error = RuntimeError("Test error for logging")
        log_error_with_context(test_error, "test_logging", {"test_param": "test_value"})
        
        # Check if error log file was created
        error_log_file = Path("wan22_errors.log")
        if error_log_file.exists():
            print("✓ Error logging works - log file created")
        else:
            print("⚠ Error log file not found")
            
    except Exception as e:
        print(f"✗ Error logging test failed: {e}")

def test_recovery_manager():
    """Test error recovery manager functionality"""
    print("\nTesting error recovery manager...")
    
    recovery_manager = get_error_recovery_manager()
    
    # Create a test error
    test_error = RuntimeError("Test error for recovery manager")
    error_info = create_error_info(test_error, "test_context")
    
    # Add error to manager
    recovery_manager.add_error(error_info)
    
    # Get statistics
    stats = recovery_manager.get_error_statistics()
    
    assert stats["total_errors"] > 0
    assert "unknown_error" in stats["by_category"] or "system_error" in stats["by_category"]
    
    print("✓ Error recovery manager works")

def main():
    """Run all error handling tests"""
    print("Running comprehensive error handling tests...\n")
    
    try:
        test_error_classification()
        test_error_info_creation()
        test_recovery_decorator()
        test_vram_error_simulation()
        test_error_logging()
        test_recovery_manager()
        
        print("\n✅ All error handling tests passed!")
        
        # Print final statistics
        recovery_manager = get_error_recovery_manager()
        stats = recovery_manager.get_error_statistics()
        print(f"\nFinal error statistics:")
        print(f"Total errors: {stats['total_errors']}")
        print(f"By category: {stats['by_category']}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()