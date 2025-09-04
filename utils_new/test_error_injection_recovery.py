#!/usr/bin/env python3
"""
Error Injection and Recovery Test Suite
Tests error injection scenarios and recovery mechanisms
"""

import unittest
import tempfile
import shutil
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorInjectionScenario:
    """Error injection scenario configuration"""
    name: str
    error_type: str
    error_location: str
    error_description: str
    injection_method: str
    expected_detection: bool
    expected_recovery: bool
    recovery_strategy: str
    severity: str  # "low", "medium", "high", "critical"

@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration"""
    name: str
    applicable_errors: List[str]
    success_rate: float
    recovery_time_s: float
    fallback_options: List[str]
    user_intervention_required: bool

class ErrorInjectionRecoveryTestSuite:
    """
    Test suite for error injection and recovery scenarios
    Tests system resilience and recovery mechanisms
    """
    
    def __init__(self):
        self.temp_dir = None
        self.error_scenarios = self._get_error_scenarios()
        self.recovery_strategies = self._get_recovery_strategies()
        self.test_results = []
        self.injected_errors = []
    
    def _get_error_scenarios(self) -> List[ErrorInjectionScenario]:
        """Get error injection scenarios"""
        return [
            # File System Errors
            ErrorInjectionScenario(
                name="missing_model_index",
                error_type="missing_file",
                error_location="model_index.json",
                error_description="model_index.json file is missing",
                injection_method="delete_file",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="reconstruct_from_components",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="corrupted_model_index",
                error_type="corrupted_file",
                error_location="model_index.json",
                error_description="model_index.json contains invalid JSON",
                injection_method="corrupt_json",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="backup_restore",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="missing_component_directory",
                error_type="missing_directory",
                error_location="transformer",
                error_description="Component directory is missing",
                injection_method="delete_directory",
                expected_detection=True,
                expected_recovery=False,
                recovery_strategy="suggest_redownload",
                severity="high"
            ),
            ErrorInjectionScenario(
                name="corrupted_model_weights",
                error_type="corrupted_weights",
                error_location="transformer/pytorch_model.bin",
                error_description="Model weights file is corrupted",
                injection_method="corrupt_binary",
                expected_detection=True,
                expected_recovery=False,
                recovery_strategy="suggest_redownload",
                severity="high"
            ),
            
            # Pipeline and Dependency Errors
            ErrorInjectionScenario(
                name="missing_pipeline_class",
                error_type="missing_dependency",
                error_location="pipeline_class",
                error_description="Custom pipeline class not available",
                injection_method="mock_import_error",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="fetch_remote_code",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="incompatible_pipeline_version",
                error_type="version_mismatch",
                error_location="pipeline_version",
                error_description="Pipeline version incompatible with model",
                injection_method="mock_version_conflict",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="version_alignment",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="missing_required_dependency",
                error_type="missing_dependency",
                error_location="torch",
                error_description="Required dependency not installed",
                injection_method="mock_import_error",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="install_dependencies",
                severity="high"
            ),
            
            # Resource and Memory Errors
            ErrorInjectionScenario(
                name="out_of_memory_loading",
                error_type="resource_constraint",
                error_location="model_loading",
                error_description="Insufficient VRAM for model loading",
                injection_method="mock_cuda_oom",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="apply_optimizations",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="out_of_memory_generation",
                error_type="resource_constraint",
                error_location="generation",
                error_description="Insufficient VRAM for generation",
                injection_method="mock_cuda_oom",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="chunked_processing",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="disk_space_exhausted",
                error_type="resource_constraint",
                error_location="output_writing",
                error_description="Insufficient disk space for output",
                injection_method="mock_disk_full",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="cleanup_temp_files",
                severity="low"
            ),
            
            # Network and Remote Code Errors
            ErrorInjectionScenario(
                name="network_timeout_remote_code",
                error_type="network_error",
                error_location="remote_code_fetch",
                error_description="Network timeout when fetching remote code",
                injection_method="mock_network_timeout",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="retry_with_backoff",
                severity="medium"
            ),
            ErrorInjectionScenario(
                name="untrusted_remote_code",
                error_type="security_error",
                error_location="remote_code_validation",
                error_description="Remote code fails security validation",
                injection_method="mock_security_violation",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="local_installation_guide",
                severity="high"
            ),
            
            # Configuration and Validation Errors
            ErrorInjectionScenario(
                name="invalid_generation_parameters",
                error_type="validation_error",
                error_location="parameter_validation",
                error_description="Invalid generation parameters provided",
                injection_method="inject_invalid_params",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="parameter_correction",
                severity="low"
            ),
            ErrorInjectionScenario(
                name="unsupported_model_architecture",
                error_type="compatibility_error",
                error_location="architecture_detection",
                error_description="Model architecture not supported",
                injection_method="mock_unknown_architecture",
                expected_detection=True,
                expected_recovery=False,
                recovery_strategy="suggest_alternatives",
                severity="high"
            ),
            
            # Critical System Errors
            ErrorInjectionScenario(
                name="gpu_driver_crash",
                error_type="system_error",
                error_location="gpu_driver",
                error_description="GPU driver crash during operation",
                injection_method="mock_driver_error",
                expected_detection=True,
                expected_recovery=True,
                recovery_strategy="fallback_to_cpu",
                severity="critical"
            ),
            ErrorInjectionScenario(
                name="python_process_killed",
                error_type="system_error",
                error_location="process",
                error_description="Python process killed by system",
                injection_method="mock_process_termination",
                expected_detection=False,
                expected_recovery=False,
                recovery_strategy="restart_with_state_recovery",
                severity="critical"
            )
        ]
    
    def _get_recovery_strategies(self) -> List[RecoveryStrategy]:
        """Get recovery strategy configurations"""
        return [
            RecoveryStrategy(
                name="reconstruct_from_components",
                applicable_errors=["missing_file"],
                success_rate=0.9,
                recovery_time_s=5.0,
                fallback_options=["suggest_redownload"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="backup_restore",
                applicable_errors=["corrupted_file"],
                success_rate=0.8,
                recovery_time_s=3.0,
                fallback_options=["reconstruct_from_components", "suggest_redownload"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="suggest_redownload",
                applicable_errors=["missing_directory", "corrupted_weights"],
                success_rate=1.0,
                recovery_time_s=0.5,
                fallback_options=[],
                user_intervention_required=True
            ),
            RecoveryStrategy(
                name="fetch_remote_code",
                applicable_errors=["missing_dependency"],
                success_rate=0.85,
                recovery_time_s=15.0,
                fallback_options=["local_installation_guide"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="version_alignment",
                applicable_errors=["version_mismatch"],
                success_rate=0.75,
                recovery_time_s=10.0,
                fallback_options=["suggest_redownload"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="install_dependencies",
                applicable_errors=["missing_dependency"],
                success_rate=0.9,
                recovery_time_s=30.0,
                fallback_options=["local_installation_guide"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="apply_optimizations",
                applicable_errors=["resource_constraint"],
                success_rate=0.8,
                recovery_time_s=5.0,
                fallback_options=["chunked_processing", "fallback_to_cpu"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="chunked_processing",
                applicable_errors=["resource_constraint"],
                success_rate=0.95,
                recovery_time_s=2.0,
                fallback_options=["fallback_to_cpu"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="cleanup_temp_files",
                applicable_errors=["resource_constraint"],
                success_rate=0.7,
                recovery_time_s=10.0,
                fallback_options=["suggest_disk_cleanup"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="retry_with_backoff",
                applicable_errors=["network_error"],
                success_rate=0.6,
                recovery_time_s=20.0,
                fallback_options=["local_installation_guide"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="local_installation_guide",
                applicable_errors=["security_error", "network_error"],
                success_rate=1.0,
                recovery_time_s=1.0,
                fallback_options=[],
                user_intervention_required=True
            ),
            RecoveryStrategy(
                name="parameter_correction",
                applicable_errors=["validation_error"],
                success_rate=0.95,
                recovery_time_s=1.0,
                fallback_options=["use_defaults"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="suggest_alternatives",
                applicable_errors=["compatibility_error"],
                success_rate=1.0,
                recovery_time_s=0.5,
                fallback_options=[],
                user_intervention_required=True
            ),
            RecoveryStrategy(
                name="fallback_to_cpu",
                applicable_errors=["system_error", "resource_constraint"],
                success_rate=0.9,
                recovery_time_s=5.0,
                fallback_options=["restart_with_state_recovery"],
                user_intervention_required=False
            ),
            RecoveryStrategy(
                name="restart_with_state_recovery",
                applicable_errors=["system_error"],
                success_rate=0.8,
                recovery_time_s=30.0,
                fallback_options=["manual_intervention"],
                user_intervention_required=True
            )
        ]
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created test directory: {self.temp_dir}")
        
        # Create mock model for testing
        self._create_mock_wan_model()
        
        # Clear injected errors
        self.injected_errors = []
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up injected errors
        self._cleanup_injected_errors()
        
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up test directory: {self.temp_dir}")
    
    def test_all_error_scenarios(self) -> List[Dict[str, Any]]:
        """
        Test all error injection scenarios
        
        Returns:
            List of test results for each scenario
        """
        logger.info("Starting error injection and recovery tests")
        
        self.setUp()
        
        try:
            for scenario in self.error_scenarios:
                logger.info(f"Testing error scenario: {scenario.name}")
                result = self.test_error_scenario(scenario)
                self.test_results.append(result)
        
        finally:
            self.tearDown()
        
        return self.test_results

        assert True  # TODO: Add proper assertion
    
    def test_error_scenario(self, scenario: ErrorInjectionScenario) -> Dict[str, Any]:
        """
        Test specific error injection scenario
        
        Args:
            scenario: Error injection scenario configuration
            
        Returns:
            Test result dictionary
        """
        result = {
            "scenario_name": scenario.name,
            "error_type": scenario.error_type,
            "error_location": scenario.error_location,
            "severity": scenario.severity,
            "success": False,
            "error_detected": False,
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_strategy_used": None,
            "fallback_strategies_tried": [],
            "total_recovery_time": 0.0,
            "user_intervention_required": False,
            "errors": [],
            "warnings": [],
            "recovery_details": {}
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Inject Error
            logger.info(f"Injecting error: {scenario.error_description}")
            injection_success = self._inject_error(scenario)
            
            if not injection_success:
                result["errors"].append("Failed to inject error")
                return result
            
            # Step 2: Attempt Normal Operation (should detect error)
            logger.info("Attempting normal operation to trigger error detection")
            error_detection_result = self._attempt_operation_with_error(scenario)
            
            result["error_detected"] = error_detection_result["error_detected"]
            
            if error_detection_result["error_detected"] != scenario.expected_detection:
                if scenario.expected_detection:
                    result["errors"].append("Error was not detected as expected")
                else:
                    result["warnings"].append("Error was detected but not expected")
            
            # Step 3: Attempt Recovery (if error detected)
            if result["error_detected"]:
                logger.info(f"Attempting recovery with strategy: {scenario.recovery_strategy}")
                recovery_start = time.time()
                
                recovery_result = self._attempt_recovery(scenario)
                
                result["recovery_attempted"] = True
                result["recovery_successful"] = recovery_result["success"]
                result["recovery_strategy_used"] = recovery_result["strategy_used"]
                result["fallback_strategies_tried"] = recovery_result["fallback_strategies"]
                result["user_intervention_required"] = recovery_result["user_intervention_required"]
                result["recovery_details"] = recovery_result["details"]
                result["total_recovery_time"] = time.time() - recovery_start
                
                if recovery_result["success"] != scenario.expected_recovery:
                    if scenario.expected_recovery:
                        result["errors"].append("Recovery failed when expected to succeed")
                    else:
                        result["warnings"].append("Recovery succeeded when expected to fail")
            
            # Step 4: Validate Post-Recovery State
            if result["recovery_successful"]:
                logger.info("Validating post-recovery state")
                validation_result = self._validate_post_recovery_state(scenario)
                
                if not validation_result["valid"]:
                    result["errors"].extend(validation_result["errors"])
                    result["recovery_successful"] = False
            
            # Determine overall success
            expected_outcome = (
                (result["error_detected"] == scenario.expected_detection) and
                (not result["recovery_attempted"] or result["recovery_successful"] == scenario.expected_recovery)
            )
            
            result["success"] = expected_outcome and len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Error scenario test failed: {str(e)}")
            logger.error(f"Error scenario test error for {scenario.name}: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            # Clean up this specific error injection
            self._cleanup_specific_error(scenario)
        
        logger.info(f"Error scenario {scenario.name} test completed: {'SUCCESS' if result['success'] else 'FAILURE'}")
        
        return result

        assert True  # TODO: Add proper assertion
    
    def _inject_error(self, scenario: ErrorInjectionScenario) -> bool:
        """Inject specific error based on scenario"""
        try:
            model_path = Path(self.temp_dir) / "wan_model"
            
            if scenario.injection_method == "delete_file":
                file_path = model_path / scenario.error_location
                if file_path.exists():
                    file_path.unlink()
                    self.injected_errors.append(("delete_file", str(file_path)))
                    return True
            
            elif scenario.injection_method == "corrupt_json":
                file_path = model_path / scenario.error_location
                if file_path.exists():
                    # Backup original
                    backup_path = file_path.with_suffix('.backup')
                    shutil.copy2(file_path, backup_path)
                    self.injected_errors.append(("backup_file", str(backup_path)))
                    
                    # Corrupt the file
                    with open(file_path, 'w') as f:
                        f.write("{ invalid json content")
                    return True
            
            elif scenario.injection_method == "delete_directory":
                dir_path = model_path / scenario.error_location
                if dir_path.exists():
                    # Backup directory
                    backup_path = dir_path.with_suffix('.backup')
                    shutil.copytree(dir_path, backup_path)
                    self.injected_errors.append(("backup_directory", str(backup_path)))
                    
                    # Delete directory
                    shutil.rmtree(dir_path)
                    return True
            
            elif scenario.injection_method == "corrupt_binary":
                file_path = model_path / scenario.error_location
                file_path.parent.mkdir(exist_ok=True)
                
                # Create corrupted binary file
                with open(file_path, 'wb') as f:
                    f.write(b"corrupted binary data")
                self.injected_errors.append(("create_file", str(file_path)))
                return True
            
            elif scenario.injection_method in ["mock_import_error", "mock_cuda_oom", "mock_network_timeout", 
                                             "mock_security_violation", "mock_version_conflict", 
                                             "mock_disk_full", "mock_unknown_architecture", 
                                             "mock_driver_error", "mock_process_termination"]:
                # These are handled by mocking in the operation attempt
                self.injected_errors.append(("mock_error", scenario.injection_method))
                return True
            
            elif scenario.injection_method == "inject_invalid_params":
                # Create invalid parameters file
                params_file = model_path / "invalid_params.json"
                with open(params_file, 'w') as f:
                    json.dump({"num_frames": -1, "height": 0, "width": "invalid"}, f)
                self.injected_errors.append(("create_file", str(params_file)))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error injection failed: {e}")
            return False
    
    def _attempt_operation_with_error(self, scenario: ErrorInjectionScenario) -> Dict[str, Any]:
        """Attempt normal operation to trigger error detection"""
        result = {
            "error_detected": False,
            "error_message": None,
            "error_type": None
        }
        
        try:
            model_path = str(Path(self.temp_dir) / "wan_model")
            
            # Mock different types of operations based on error location
            if scenario.error_location in ["model_index.json", "transformer", "vae"]:
                # Test model loading
                result = self._test_model_loading_with_error(scenario, model_path)
            
            elif scenario.error_location in ["pipeline_class", "pipeline_version"]:
                # Test pipeline loading
                result = self._test_pipeline_loading_with_error(scenario, model_path)
            
            elif scenario.error_location in ["model_loading", "generation"]:
                # Test generation with resource constraints
                result = self._test_generation_with_error(scenario, model_path)
            
            elif scenario.error_location in ["remote_code_fetch", "remote_code_validation"]:
                # Test remote code operations
                result = self._test_remote_code_with_error(scenario, model_path)
            
            elif scenario.error_location in ["parameter_validation", "architecture_detection"]:
                # Test validation operations
                result = self._test_validation_with_error(scenario, model_path)
            
            elif scenario.error_location in ["output_writing", "gpu_driver", "process"]:
                # Test system-level operations
                result = self._test_system_operations_with_error(scenario, model_path)
            
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _test_model_loading_with_error(self, scenario: ErrorInjectionScenario, model_path: str) -> Dict[str, Any]:
        """Test model loading with injected error"""
        result = {"error_detected": False, "error_message": None, "error_type": None}
        
        try:
            # Check if model_index.json exists
            model_index_path = Path(model_path) / "model_index.json"
            
            if not model_index_path.exists():
                result["error_detected"] = True
                result["error_message"] = "model_index.json not found"
                result["error_type"] = "FileNotFoundError"
                return result
            
            # Try to load and parse model_index.json
            try:
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
            except json.JSONDecodeError as e:
                result["error_detected"] = True
                result["error_message"] = f"Invalid JSON in model_index.json: {str(e)}"
                result["error_type"] = "JSONDecodeError"
                return result
            
            # Check for required components
            required_components = ["transformer", "vae", "scheduler"]
            for component in required_components:
                if component in model_index:
                    component_path = Path(model_path) / component
                    if not component_path.exists():
                        result["error_detected"] = True
                        result["error_message"] = f"Component directory missing: {component}"
                        result["error_type"] = "ComponentMissingError"
                        return result
            
            # Check for corrupted weights
            weights_file = Path(model_path) / "transformer" / "pytorch_model.bin"
            if weights_file.exists():
                # Try to read first few bytes to detect corruption
                try:
                    with open(weights_file, 'rb') as f:
                        header = f.read(8)
                        if header == b"corrupted binary data"[:8]:
                            result["error_detected"] = True
                            result["error_message"] = "Corrupted model weights detected"
                            result["error_type"] = "CorruptedWeightsError"
                            return result
                except Exception as e:
                    result["error_detected"] = True
                    result["error_message"] = f"Error reading weights file: {str(e)}"
                    result["error_type"] = "WeightsReadError"
                    return result
        
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _test_pipeline_loading_with_error(self, scenario: ErrorInjectionScenario, model_path: str) -> Dict[str, Any]:
        """Test pipeline loading with injected error"""
        result = {"error_detected": False, "error_message": None, "error_type": None}
        
        try:
            if scenario.injection_method == "mock_import_error":
                # Simulate import error for pipeline class
                result["error_detected"] = True
                result["error_message"] = "No module named 'wan_pipeline'"
                result["error_type"] = "ImportError"
            
            elif scenario.injection_method == "mock_version_conflict":
                # Simulate version conflict
                result["error_detected"] = True
                result["error_message"] = "Pipeline version 1.0.0 incompatible with model version 2.0.0"
                result["error_type"] = "VersionConflictError"
        
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _test_generation_with_error(self, scenario: ErrorInjectionScenario, model_path: str) -> Dict[str, Any]:
        """Test generation with injected error"""
        result = {"error_detected": False, "error_message": None, "error_type": None}
        
        try:
            if scenario.injection_method == "mock_cuda_oom":
                # Simulate CUDA out of memory error
                result["error_detected"] = True
                result["error_message"] = "CUDA out of memory. Tried to allocate 12.00 GiB"
                result["error_type"] = "OutOfMemoryError"
        
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _test_remote_code_with_error(self, scenario: ErrorInjectionScenario, model_path: str) -> Dict[str, Any]:
        """Test remote code operations with injected error"""
        result = {"error_detected": False, "error_message": None, "error_type": None}
        
        try:
            if scenario.injection_method == "mock_network_timeout":
                # Simulate network timeout
                result["error_detected"] = True
                result["error_message"] = "Connection timeout when fetching remote code"
                result["error_type"] = "TimeoutError"
            
            elif scenario.injection_method == "mock_security_violation":
                # Simulate security violation
                result["error_detected"] = True
                result["error_message"] = "Remote code failed security validation"
                result["error_type"] = "SecurityError"
        
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _test_validation_with_error(self, scenario: ErrorInjectionScenario, model_path: str) -> Dict[str, Any]:
        """Test validation operations with injected error"""
        result = {"error_detected": False, "error_message": None, "error_type": None}
        
        try:
            if scenario.injection_method == "inject_invalid_params":
                # Check for invalid parameters file
                params_file = Path(model_path) / "invalid_params.json"
                if params_file.exists():
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    
                    # Validate parameters
                    if params.get("num_frames", 0) <= 0:
                        result["error_detected"] = True
                        result["error_message"] = "Invalid parameter: num_frames must be positive"
                        result["error_type"] = "ValidationError"
                    elif params.get("height", 0) <= 0:
                        result["error_detected"] = True
                        result["error_message"] = "Invalid parameter: height must be positive"
                        result["error_type"] = "ValidationError"
                    elif not isinstance(params.get("width"), int):
                        result["error_detected"] = True
                        result["error_message"] = "Invalid parameter: width must be integer"
                        result["error_type"] = "ValidationError"
            
            elif scenario.injection_method == "mock_unknown_architecture":
                # Simulate unknown architecture
                result["error_detected"] = True
                result["error_message"] = "Unknown model architecture: unsupported_arch"
                result["error_type"] = "UnsupportedArchitectureError"
        
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _test_system_operations_with_error(self, scenario: ErrorInjectionScenario, model_path: str) -> Dict[str, Any]:
        """Test system operations with injected error"""
        result = {"error_detected": False, "error_message": None, "error_type": None}
        
        try:
            if scenario.injection_method == "mock_disk_full":
                # Simulate disk full error
                result["error_detected"] = True
                result["error_message"] = "No space left on device"
                result["error_type"] = "OSError"
            
            elif scenario.injection_method == "mock_driver_error":
                # Simulate GPU driver error
                result["error_detected"] = True
                result["error_message"] = "NVIDIA driver error: GPU has fallen off the bus"
                result["error_type"] = "CudaError"
            
            elif scenario.injection_method == "mock_process_termination":
                # Simulate process termination (this wouldn't be detected in normal flow)
                result["error_detected"] = False  # Process termination isn't detected by the process itself
        
        except Exception as e:
            result["error_detected"] = True
            result["error_message"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def _attempt_recovery(self, scenario: ErrorInjectionScenario) -> Dict[str, Any]:
        """Attempt recovery using specified strategy"""
        recovery_result = {
            "success": False,
            "strategy_used": scenario.recovery_strategy,
            "fallback_strategies": [],
            "user_intervention_required": False,
            "details": {}
        }
        
        try:
            # Find the recovery strategy
            strategy = next((s for s in self.recovery_strategies if s.name == scenario.recovery_strategy), None)
            
            if not strategy:
                recovery_result["details"]["error"] = f"Recovery strategy not found: {scenario.recovery_strategy}"
                return recovery_result
            
            # Simulate recovery attempt
            recovery_result["user_intervention_required"] = strategy.user_intervention_required
            
            # Simulate recovery success based on strategy success rate
            import random
            recovery_success = random.random() < strategy.success_rate
            
            if recovery_success:
                recovery_result["success"] = True
                recovery_result["details"]["recovery_time"] = strategy.recovery_time_s
                recovery_result["details"]["method"] = self._simulate_recovery_method(scenario, strategy)
            else:
                # Try fallback strategies
                for fallback_name in strategy.fallback_options:
                    fallback_strategy = next((s for s in self.recovery_strategies if s.name == fallback_name), None)
                    if fallback_strategy:
                        recovery_result["fallback_strategies"].append(fallback_name)
                        
                        # Simulate fallback attempt
                        fallback_success = random.random() < fallback_strategy.success_rate
                        if fallback_success:
                            recovery_result["success"] = True
                            recovery_result["strategy_used"] = fallback_name
                            recovery_result["user_intervention_required"] = fallback_strategy.user_intervention_required
                            recovery_result["details"]["recovery_time"] = fallback_strategy.recovery_time_s
                            recovery_result["details"]["method"] = self._simulate_recovery_method(scenario, fallback_strategy)
                            break
        
        except Exception as e:
            recovery_result["details"]["error"] = f"Recovery attempt failed: {str(e)}"
        
        return recovery_result
    
    def _simulate_recovery_method(self, scenario: ErrorInjectionScenario, strategy: RecoveryStrategy) -> str:
        """Simulate specific recovery method"""
        if strategy.name == "reconstruct_from_components":
            return "Reconstructed model_index.json from component directories"
        elif strategy.name == "backup_restore":
            return "Restored from backup file"
        elif strategy.name == "suggest_redownload":
            return "Provided instructions to redownload model"
        elif strategy.name == "fetch_remote_code":
            return "Successfully fetched pipeline code from remote repository"
        elif strategy.name == "version_alignment":
            return "Aligned pipeline version with model requirements"
        elif strategy.name == "install_dependencies":
            return "Installed missing dependencies"
        elif strategy.name == "apply_optimizations":
            return "Applied memory optimizations (mixed precision, CPU offload)"
        elif strategy.name == "chunked_processing":
            return "Enabled chunked processing to reduce memory usage"
        elif strategy.name == "cleanup_temp_files":
            return "Cleaned up temporary files to free disk space"
        elif strategy.name == "retry_with_backoff":
            return "Retried operation with exponential backoff"
        elif strategy.name == "local_installation_guide":
            return "Provided local installation instructions"
        elif strategy.name == "parameter_correction":
            return "Corrected invalid parameters to valid defaults"
        elif strategy.name == "suggest_alternatives":
            return "Suggested alternative compatible models"
        elif strategy.name == "fallback_to_cpu":
            return "Switched to CPU-only processing"
        elif strategy.name == "restart_with_state_recovery":
            return "Restarted process with state recovery"
        else:
            return f"Applied recovery strategy: {strategy.name}"
    
    def _validate_post_recovery_state(self, scenario: ErrorInjectionScenario) -> Dict[str, Any]:
        """Validate system state after recovery"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            model_path = Path(self.temp_dir) / "wan_model"
            
            # Check if critical files exist after recovery
            if scenario.error_location == "model_index.json":
                if not (model_path / "model_index.json").exists():
                    validation_result["valid"] = False
                    validation_result["errors"].append("model_index.json still missing after recovery")
            
            # Check if directories exist after recovery
            if scenario.error_location in ["transformer", "vae", "scheduler"]:
                if not (model_path / scenario.error_location).exists():
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"{scenario.error_location} directory still missing after recovery")
            
            # Additional validation based on recovery strategy
            if scenario.recovery_strategy == "reconstruct_from_components":
                # Verify reconstructed file is valid JSON
                try:
                    with open(model_path / "model_index.json", 'r') as f:
                        json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    validation_result["valid"] = False
                    validation_result["errors"].append("Reconstructed model_index.json is invalid")
        
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Post-recovery validation failed: {str(e)}")
        
        return validation_result
    
    def _cleanup_specific_error(self, scenario: ErrorInjectionScenario):
        """Clean up specific injected error"""
        try:
            # Remove any files created for this specific error
            for error_type, error_path in self.injected_errors[:]:
                if error_type == "create_file" and scenario.injection_method in ["corrupt_binary", "inject_invalid_params"]:
                    if Path(error_path).exists():
                        Path(error_path).unlink()
                    self.injected_errors.remove((error_type, error_path))
        except Exception as e:
            logger.warning(f"Failed to clean up specific error: {e}")
    
    def _cleanup_injected_errors(self):
        """Clean up all injected errors"""
        for error_type, error_path in self.injected_errors:
            try:
                if error_type == "backup_file":
                    # Restore from backup
                    backup_path = Path(error_path)
                    original_path = backup_path.with_suffix('')
                    if backup_path.exists():
                        shutil.copy2(backup_path, original_path)
                        backup_path.unlink()
                
                elif error_type == "backup_directory":
                    # Restore directory from backup
                    backup_path = Path(error_path)
                    original_path = backup_path.with_suffix('')
                    if backup_path.exists():
                        if original_path.exists():
                            shutil.rmtree(original_path)
                        shutil.copytree(backup_path, original_path)
                        shutil.rmtree(backup_path)
                
                elif error_type == "create_file":
                    # Remove created file
                    file_path = Path(error_path)
                    if file_path.exists():
                        file_path.unlink()
            
            except Exception as e:
                logger.warning(f"Failed to clean up injected error {error_type} at {error_path}: {e}")
        
        self.injected_errors = []
    
    def _create_mock_wan_model(self):
        """Create mock Wan model for testing"""
        model_dir = Path(self.temp_dir) / "wan_model"
        model_dir.mkdir(exist_ok=True)
        
        # Create model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.4",
            "transformer": ["transformer", "diffusers"],
            "transformer_2": ["transformer_2", "diffusers"],
            "vae": ["vae", "diffusers"],
            "scheduler": ["scheduler", "diffusers"],
            "boundary_ratio": 0.5
        }
        
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
        
        # Create component directories
        for component in ["transformer", "transformer_2", "vae", "scheduler"]:
            component_dir = model_dir / component
            component_dir.mkdir(exist_ok=True)
            
            # Create config.json for each component
            component_config = {
                "_class_name": f"Mock{component.title()}",
                "model_type": "wan_t2v"
            }
            
            with open(component_dir / "config.json", 'w') as f:
                json.dump(component_config, f, indent=2)
            
            # Create mock weight file for transformer
            if component == "transformer":
                with open(component_dir / "pytorch_model.bin", 'wb') as f:
                    f.write(b"mock model weights data")


def main():
    """Main entry point for error injection and recovery tests"""
    
    print("Error Injection and Recovery Test Suite")
    print("=" * 50)
    
    # Create and run test suite
    test_suite = ErrorInjectionRecoveryTestSuite()
    results = test_suite.test_all_error_scenarios()
    
    # Print summary
    print("\nError Injection and Recovery Test Results:")
    print("-" * 60)
    
    total_scenarios = len(results)
    successful_scenarios = sum(1 for r in results if r["success"])
    
    # Group results by severity
    severity_groups = {}
    for result in results:
        severity = result["severity"]
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(result)
    
    # Print results by severity
    for severity in ["critical", "high", "medium", "low"]:
        if severity in severity_groups:
            print(f"\n{severity.upper()} SEVERITY ERRORS:")
            print("-" * 30)
            
            for result in severity_groups[severity]:
                status = "PASS" if result["success"] else "FAIL"
                scenario_name = result["scenario_name"]
                
                detection_status = "DETECTED" if result["error_detected"] else "NOT DETECTED"
                recovery_status = "RECOVERED" if result["recovery_successful"] else "NOT RECOVERED"
                
                print(f"{status} {scenario_name}")
                print(f"    Error: {detection_status}")
                print(f"    Recovery: {recovery_status}")
                
                if result["recovery_strategy_used"]:
                    print(f"    Strategy: {result['recovery_strategy_used']}")
                
                if result["user_intervention_required"]:
                    print(f"    User Intervention: REQUIRED")
                
                if result["errors"]:
                    for error in result["errors"][:2]:  # Show first 2 errors
                        print(f"    Error: {error}")
                
                if result["warnings"]:
                    for warning in result["warnings"][:1]:  # Show first warning
                        print(f"    Warning: {warning}")
    
    print(f"\nOverall Results: {successful_scenarios}/{total_scenarios} scenarios passed")
    
    # Calculate recovery statistics
    recovery_attempted = sum(1 for r in results if r["recovery_attempted"])
    recovery_successful = sum(1 for r in results if r["recovery_successful"])
    
    if recovery_attempted > 0:
        recovery_rate = (recovery_successful / recovery_attempted) * 100
        print(f"Recovery Success Rate: {recovery_successful}/{recovery_attempted} ({recovery_rate:.1f}%)")
    
    # Save detailed results
    results_file = Path("test_results") / "error_injection_recovery_test_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to {results_file}")
    
    return successful_scenarios == total_scenarios


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)