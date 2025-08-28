"""
Test script for the comprehensive error handling and recovery system.
"""

import sys
import logging
from pathlib import Path

# Add scripts directory to path
scripts_path = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_path))

# Import with absolute imports
import interfaces
from interfaces import InstallationError, ErrorCategory

# Import the modules we need to test
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import our modules
error_handler_module = import_module_from_path("error_handler", scripts_path / "error_handler.py")
user_guidance_module = import_module_from_path("user_guidance", scripts_path / "user_guidance.py")
diagnostic_tool_module = import_module_from_path("diagnostic_tool", scripts_path / "diagnostic_tool.py")

# Extract classes we need
ComprehensiveErrorHandler = error_handler_module.ComprehensiveErrorHandler
ErrorClassifier = error_handler_module.ErrorClassifier
RetryManager = error_handler_module.RetryManager
FallbackManager = error_handler_module.FallbackManager
ErrorContext = error_handler_module.ErrorContext
RecoveryAction = error_handler_module.RecoveryAction
ErrorSeverity = error_handler_module.ErrorSeverity

UserGuidanceSystem = user_guidance_module.UserGuidanceSystem
InstallationDiagnosticTool = diagnostic_tool_module.InstallationDiagnosticTool


def test_error_classification():
    """Test error classification functionality."""
    print("Testing Error Classification...")
    
    classifier = ErrorClassifier()
    context = ErrorContext()
    
    # Test different error types
    test_cases = [
        (ConnectionError("Connection refused"), ErrorCategory.NETWORK),
        (PermissionError("Access denied"), ErrorCategory.PERMISSION),
        (ValueError("Invalid configuration"), ErrorCategory.CONFIGURATION),
        (OSError("System error"), ErrorCategory.SYSTEM),
        (Exception("Download timeout occurred"), ErrorCategory.NETWORK),
        (Exception("Permission denied to write file"), ErrorCategory.PERMISSION),
    ]
    
    for error, expected_category in test_cases:
        classified_category = classifier.classify_error(error, context)
        status = "✅" if classified_category == expected_category else "❌"
        print(f"  {status} {type(error).__name__}: {error} -> {classified_category.value}")
    
    print()


def test_error_severity():
    """Test error severity determination."""
    print("Testing Error Severity...")
    
    classifier = ErrorClassifier()
    context = ErrorContext()
    
    test_cases = [
        ("Insufficient memory to continue", ErrorSeverity.CRITICAL),
        ("Permission denied", ErrorSeverity.HIGH),
        ("Connection timeout", ErrorSeverity.MEDIUM),
        ("Minor configuration warning", ErrorSeverity.LOW),
    ]
    
    for message, expected_severity in test_cases:
        error = InstallationError(message, ErrorCategory.SYSTEM)
        severity = classifier.determine_severity(error, context)
        status = "✅" if severity == expected_severity else "❌"
        print(f"  {status} '{message}' -> {severity.value}")
    
    print()


def test_retry_logic():
    """Test retry logic functionality."""
    print("Testing Retry Logic...")
    
    retry_manager = RetryManager()
    context = ErrorContext()
    
    # Test retry decision
    network_error = InstallationError("Connection timeout", ErrorCategory.NETWORK)
    permission_error = InstallationError("Access denied", ErrorCategory.PERMISSION)
    
    from error_handler import RetryConfig
    config = RetryConfig(max_attempts=3)
    
    # Network error should be retryable
    should_retry_network = retry_manager.should_retry(network_error, context, config)
    print(f"  ✅ Network error retryable: {should_retry_network}")
    
    # Permission error should not be retryable
    should_retry_permission = retry_manager.should_retry(permission_error, context, config)
    print(f"  ✅ Permission error not retryable: {not should_retry_permission}")
    
    # Test delay calculation
    delay = retry_manager.calculate_delay(0, config)
    print(f"  ✅ First retry delay: {delay:.2f}s")
    
    delay = retry_manager.calculate_delay(2, config)
    print(f"  ✅ Third retry delay: {delay:.2f}s")
    
    print()


def test_comprehensive_error_handler():
    """Test the comprehensive error handler."""
    print("Testing Comprehensive Error Handler...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    handler = ComprehensiveErrorHandler("./test_installation")
    
    # Test error handling
    test_error = InstallationError(
        "Failed to download model due to network timeout",
        ErrorCategory.NETWORK,
        ["Check internet connection", "Try again later"]
    )
    
    context = ErrorContext(
        phase="model_download",
        task="downloading WAN2.2 models",
        retry_count=1
    )
    
    recovery_action = handler.handle_error(test_error, context)
    print(f"  ✅ Recovery action for network error: {recovery_action.value}")
    
    # Test error statistics
    stats = handler.get_error_statistics()
    print(f"  ✅ Error statistics: {stats['total_errors']} errors recorded")
    
    print()


def test_user_guidance():
    """Test user guidance system."""
    print("Testing User Guidance System...")
    
    guidance = UserGuidanceSystem("./test_installation")
    
    # Test error formatting
    test_error = InstallationError(
        "Python not found in system PATH",
        ErrorCategory.SYSTEM,
        ["Install Python 3.9+", "Add Python to PATH"]
    )
    
    formatted_message = guidance.format_user_friendly_error(test_error)
    print("  ✅ Formatted error message:")
    print("     " + formatted_message.replace("\n", "\n     ")[:200] + "...")
    
    # Test troubleshooting guide lookup
    guide = guidance.find_relevant_troubleshooting_guide(test_error)
    if guide:
        print(f"  ✅ Found relevant guide: '{guide.title}'")
    else:
        print("  ❌ No relevant guide found")
    
    # Test available guides
    guides = guidance.list_available_guides()
    print(f"  ✅ Available guides: {len(guides)} guides loaded")
    
    print()


def test_diagnostic_tool():
    """Test diagnostic tool functionality."""
    print("Testing Diagnostic Tool...")
    
    diagnostic = InstallationDiagnosticTool("./test_installation")
    
    # Test quick health check
    health_check = diagnostic.get_quick_health_check()
    print(f"  ✅ Quick health check completed:")
    print(f"     Python OK: {health_check['python_ok']}")
    print(f"     Memory OK: {health_check['memory_ok']}")
    print(f"     Disk OK: {health_check['disk_ok']}")
    print(f"     Network OK: {health_check['network_ok']}")
    print(f"     Overall OK: {health_check['overall_ok']}")
    
    print()


def test_fallback_manager():
    """Test fallback manager functionality."""
    print("Testing Fallback Manager...")
    
    fallback_manager = FallbackManager()
    context = ErrorContext()
    
    # Test getting fallback options
    python_fallbacks = fallback_manager.get_fallback_options("python_download", context)
    print(f"  ✅ Python download fallbacks: {len(python_fallbacks)} options")
    
    model_fallbacks = fallback_manager.get_fallback_options("model_download", context)
    print(f"  ✅ Model download fallbacks: {len(model_fallbacks)} options")
    
    package_fallbacks = fallback_manager.get_fallback_options("package_install", context)
    print(f"  ✅ Package install fallbacks: {len(package_fallbacks)} options")
    
    print()


def main():
    """Run all error handling tests."""
    print("🧪 WAN2.2 Error Handling System Tests")
    print("=" * 50)
    print()
    
    try:
        test_error_classification()
        test_error_severity()
        test_retry_logic()
        test_comprehensive_error_handler()
        test_user_guidance()
        test_diagnostic_tool()
        test_fallback_manager()
        
        print("✅ All tests completed successfully!")
        print("\nTo test interactively:")
        print("  python scripts/user_guidance.py")
        print("  python scripts/diagnostic_tool.py")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())