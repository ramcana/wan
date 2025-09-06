"""
Integration test for the comprehensive error handling and recovery system.

This test simulates various error scenarios and validates that the error
handling system responds appropriately with user guidance and recovery actions.
"""

import sys
import logging
from pathlib import Path
import tempfile
import shutil

# Add scripts directory to path
scripts_path = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_path))

from interfaces import InstallationError, ErrorCategory
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

ComprehensiveErrorHandler = error_handler_module.ComprehensiveErrorHandler
ErrorContext = error_handler_module.ErrorContext
UserGuidanceSystem = user_guidance_module.UserGuidanceSystem
InstallationDiagnosticTool = diagnostic_tool_module.InstallationDiagnosticTool


class ErrorHandlingIntegrationTest:
    """Integration test suite for error handling system."""
    
    def __init__(self):
        # Create temporary installation directory
        self.temp_dir = tempfile.mkdtemp(prefix="wan22_test_")
        self.installation_path = self.temp_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.error_handler = ComprehensiveErrorHandler(self.installation_path)
        self.user_guidance = UserGuidanceSystem(self.installation_path)
        self.diagnostic_tool = InstallationDiagnosticTool(self.installation_path)
        
        self.test_results = []
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results."""
        print(f"\n🧪 Running test: {test_name}")
        print("-" * 50)
        
        try:
            result = test_func()
            status = "✅ PASS" if result else "❌ FAIL"
            self.test_results.append((test_name, result))
            print(f"{status}: {test_name}")
            return result
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            self.test_results.append((test_name, False))
            import traceback
            traceback.print_exc()
            return False
    
    def test_network_error_handling(self):
        """Test handling of network-related errors."""
        error = InstallationError(
            "Failed to download model: Connection timeout after 30 seconds",
            ErrorCategory.NETWORK,
            ["Check internet connection", "Try again later"]
        )
        
        context = ErrorContext(
            phase="model_download",
            task="downloading WAN2.2-T2V-A14B model",
            retry_count=1
        )
        
        # Test error handling
        recovery_action = self.error_handler.handle_error(error, context)
        
        # Should recommend retry for network errors
        expected_actions = ["retry", "retry_with_fallback"]
        if recovery_action.value not in expected_actions:
            print(f"Unexpected recovery action: {recovery_action.value}")
            return False
        
        # Test user guidance
        friendly_message = self.user_guidance.format_user_friendly_error(error)
        if "Network Connection Issue" not in friendly_message:
            print("User-friendly message doesn't contain expected network guidance")
            return False
        
        # Test troubleshooting guide lookup
        guide = self.user_guidance.find_relevant_troubleshooting_guide(error)
        if not guide or "network" not in guide.title.lower():
            print("Failed to find relevant network troubleshooting guide")
            return False
        
        print(f"✅ Network error handled with recovery action: {recovery_action.value}")
        print(f"✅ Found relevant guide: {guide.title}")
        return True
    
    def test_permission_error_handling(self):
        """Test handling of permission-related errors."""
        error = InstallationError(
            "Permission denied: Cannot write to installation directory",
            ErrorCategory.PERMISSION,
            ["Run as administrator", "Check folder permissions"]
        )
        
        context = ErrorContext(
            phase="dependency_installation",
            task="creating virtual environment",
            retry_count=0
        )
        
        # Test error handling
        recovery_action = self.error_handler.handle_error(error, context)
        
        # Should recommend elevation for permission errors
        if recovery_action.value != "elevate":
            print(f"Expected 'elevate' action, got: {recovery_action.value}")
            return False
        
        # Test user guidance
        friendly_message = self.user_guidance.format_user_friendly_error(error)
        if "Permission Problem" not in friendly_message:
            print("User-friendly message doesn't contain expected permission guidance")
            return False
        
        print(f"✅ Permission error handled with recovery action: {recovery_action.value}")
        return True
    
    def test_system_error_handling(self):
        """Test handling of system-related errors."""
        error = InstallationError(
            "Insufficient disk space: Only 2GB available, 50GB required",
            ErrorCategory.SYSTEM,
            ["Free up disk space", "Choose different installation directory"]
        )
        
        context = ErrorContext(
            phase="system_detection",
            task="checking system requirements",
            retry_count=0
        )
        
        # Test error handling
        recovery_action = self.error_handler.handle_error(error, context)
        
        # Should recommend manual intervention for critical system errors
        expected_actions = ["manual", "abort"]
        if recovery_action.value not in expected_actions:
            print(f"Unexpected recovery action for system error: {recovery_action.value}")
            return False
        
        # Test user guidance
        guide = self.user_guidance.find_relevant_troubleshooting_guide(error)
        if not guide or "resource" not in guide.title.lower():
            print("Failed to find relevant system resource troubleshooting guide")
            return False
        
        print(f"✅ System error handled with recovery action: {recovery_action.value}")
        print(f"✅ Found relevant guide: {guide.title}")
        return True
    
    def test_error_statistics_tracking(self):
        """Test error statistics and tracking functionality."""
        # Reset error history for clean test
        self.error_handler.error_history = []
        
        # Generate several errors
        errors = [
            InstallationError("Network timeout 1", ErrorCategory.NETWORK),
            InstallationError("Network timeout 2", ErrorCategory.NETWORK),
            InstallationError("Permission denied", ErrorCategory.PERMISSION),
            InstallationError("Invalid config", ErrorCategory.CONFIGURATION),
        ]
        
        context = ErrorContext(phase="test", task="testing")
        
        for error in errors:
            self.error_handler.handle_error(error, context)
        
        # Check statistics
        stats = self.error_handler.get_error_statistics()
        
        if stats["total_errors"] != len(errors):
            print(f"Expected {len(errors)} errors, got {stats['total_errors']}")
            return False
        
        if stats["by_category"].get("network", 0) != 2:
            print(f"Expected 2 network errors, got {stats['by_category'].get('network', 0)}")
            return False
        
        print(f"✅ Error statistics correctly tracked: {stats['total_errors']} total errors")
        print(f"✅ Category breakdown: {stats['by_category']}")
        return True
    
    def test_diagnostic_integration(self):
        """Test integration with diagnostic tool."""
        # Run quick health check
        health_check = self.diagnostic_tool.get_quick_health_check()
        
        if not isinstance(health_check, dict):
            print("Health check didn't return expected dictionary")
            return False
        
        required_keys = ["python_ok", "memory_ok", "disk_ok", "network_ok", "overall_ok"]
        for key in required_keys:
            if key not in health_check:
                print(f"Missing key in health check: {key}")
                return False
        
        print(f"✅ Health check completed: Overall OK = {health_check['overall_ok']}")
        return True
    
    def test_troubleshooting_guide_coverage(self):
        """Test that troubleshooting guides cover common error scenarios."""
        guides = self.user_guidance.list_available_guides()
        
        if len(guides) < 5:
            print(f"Expected at least 5 troubleshooting guides, got {len(guides)}")
            return False
        
        # Check for essential guides
        guide_titles = [title.lower() for _, title, _ in guides]
        essential_topics = ["python", "network", "permission", "gpu", "model"]
        
        missing_topics = []
        for topic in essential_topics:
            if not any(topic in title for title in guide_titles):
                missing_topics.append(topic)
        
        if missing_topics:
            print(f"Missing essential troubleshooting topics: {missing_topics}")
            return False
        
        print(f"✅ Found {len(guides)} troubleshooting guides covering essential topics")
        return True
    
    def test_retry_and_fallback_integration(self):
        """Test retry and fallback mechanism integration."""
        # Create a function that fails initially then succeeds
        attempt_count = 0
        
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Simulated network failure")
            return "success"
        
        context = ErrorContext(phase="test", task="testing retry")
        
        try:
            result = self.error_handler.execute_with_retry_and_fallback(
                failing_function, "test_scenario", context
            )
            
            if result != "success":
                print(f"Expected 'success', got: {result}")
                return False
            
            if attempt_count != 3:
                print(f"Expected 3 attempts, got: {attempt_count}")
                return False
            
            print(f"✅ Retry mechanism worked: succeeded after {attempt_count} attempts")
            return True
            
        except Exception as e:
            print(f"Retry mechanism failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Error Handling Integration Tests")
        print("=" * 60)
        
        tests = [
            ("Network Error Handling", self.test_network_error_handling),
            ("Permission Error Handling", self.test_permission_error_handling),
            ("System Error Handling", self.test_system_error_handling),
            ("Error Statistics Tracking", self.test_error_statistics_tracking),
            ("Diagnostic Integration", self.test_diagnostic_integration),
            ("Troubleshooting Guide Coverage", self.test_troubleshooting_guide_coverage),
            ("Retry and Fallback Integration", self.test_retry_and_fallback_integration),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All integration tests passed!")
            return True
        else:
            print(f"⚠️  {total - passed} tests failed")
            return False


def main():
    """Run the integration tests."""
    test_suite = ErrorHandlingIntegrationTest()
    
    try:
        success = test_suite.run_all_tests()
        return 0 if success else 1
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    sys.exit(main())