from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Integration Tester for Local Testing Framework
Provides comprehensive integration testing capabilities including video generation workflows,
UI testing, API validation, and resource monitoring.
"""

import os
import sys
import time
import json
import subprocess
import threading
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import requests
import psutil

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    # Define fallback exceptions when selenium is not available
    class TimeoutException(Exception):
        pass
    class WebDriverException(Exception):
        pass

try:
from .models.test_results import (
        TestResults, IntegrationTestResults, UITestResults, APITestResults,
        ValidationResult, TestStatus, ResourceMetrics, ValidationStatus
    )
from .models.configuration import LocalTestConfiguration
except ImportError:
    from models.test_results import (
        TestResults, IntegrationTestResults, UITestResults, APITestResults,
        ValidationResult, TestStatus, ResourceMetrics, ValidationStatus
    )
    from models.configuration import LocalTestConfiguration


@dataclass
class GenerationTestResult:
    """Result of a video generation test"""
    model_type: str
    resolution: str
    prompt: str
    success: bool
    generation_time_seconds: float
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    resource_usage: Optional[ResourceMetrics] = None


@dataclass
class WorkflowTestResult:
    """Result of an end-to-end workflow test"""
    workflow_name: str
    success: bool
    total_time_seconds: float
    steps_completed: int
    steps_total: int
    error_message: Optional[str] = None
    generation_results: List[GenerationTestResult] = None


class IntegrationTester:
    """
    Main integration testing orchestrator that coordinates all integration testing activities.
    Integrates with existing run_integration_tests.py and test_error_integration.py.
    """
    
    def __init__(self, config: LocalTestConfiguration):
        self.config = config
        self.test_results = IntegrationTestResults(
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        self.temp_dir = None
        self.ui_tester = None
        self.api_tester = None
        self.resource_monitor = None
        
        # Initialize components
        self._setup_temp_environment()
        if SELENIUM_AVAILABLE:
            self.ui_tester = UITester(config)
        self.api_tester = APITester(config)
        
    def _setup_temp_environment(self):
        """Set up temporary testing environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="integration_test_")
        
        # Create necessary directories
        for dir_name in ['outputs', 'models', 'loras']:
            os.makedirs(os.path.join(self.temp_dir, dir_name), exist_ok=True)
    
    def run_video_generation_tests(self) -> List[GenerationTestResult]:
        """
        Test video generation end-to-end workflows using run_integration_tests.py
        Requirements: 3.1, 3.3
        """
        print("Running video generation end-to-end workflow tests...")
        
        generation_results = []
        
        # Test configurations for different generation modes
        test_configs = [
            {
                "model_type": "t2v-A14B",
                "prompt": "A beautiful sunset over mountains, cinematic lighting",
                "resolution": "1280x720",
                "expected_time_limit": 540  # 9 minutes in seconds
            },
            {
                "model_type": "i2v-A14B", 
                "prompt": "Animate this landscape with gentle movement",
                "resolution": "1280x720",
                "expected_time_limit": 540
            },
            {
                "model_type": "ti2v-5B",
                "prompt": "Transform this scene into a dynamic cinematic sequence", 
                "resolution": "1920x1080",
                "expected_time_limit": 1020  # 17 minutes in seconds
            }
        ]
        
        for test_config in test_configs:
            print(f"Testing {test_config['model_type']} generation...")
            
            start_time = time.time()
            resource_start = self._capture_resource_snapshot()
            
            try:
                # Execute generation test using existing integration test framework
                result = self._execute_generation_test(test_config)
                
                end_time = time.time()
                generation_time = end_time - start_time
                resource_end = self._capture_resource_snapshot()
                
                # Calculate resource usage
                resource_usage = self._calculate_resource_delta(resource_start, resource_end)
                
                generation_result = GenerationTestResult(
                    model_type=test_config['model_type'],
                    resolution=test_config['resolution'],
                    prompt=test_config['prompt'],
                    success=result['success'],
                    generation_time_seconds=generation_time,
                    output_path=result.get('output_path'),
                    error_message=result.get('error_message'),
                    resource_usage=resource_usage
                )
                
                generation_results.append(generation_result)
                
                # Validate against performance targets
                if generation_time <= test_config['expected_time_limit']:
                    print(f"✓ {test_config['model_type']} completed in {generation_time:.1f}s (limit: {test_config['expected_time_limit']}s)")
                else:
                    print(f"✗ {test_config['model_type']} took {generation_time:.1f}s (limit: {test_config['expected_time_limit']}s)")
                
            except Exception as e:
                print(f"✗ {test_config['model_type']} generation failed: {e}")
                generation_result = GenerationTestResult(
                    model_type=test_config['model_type'],
                    resolution=test_config['resolution'],
                    prompt=test_config['prompt'],
                    success=False,
                    generation_time_seconds=time.time() - start_time,
                    error_message=str(e)
                )
                generation_results.append(generation_result)
        
        return generation_results
    
    def _execute_generation_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single generation test using existing integration framework"""
        try:
            # Use existing run_integration_tests.py functionality
            cmd = [
                sys.executable, "run_integration_tests.py",
                "--model", test_config['model_type'],
                "--prompt", test_config['prompt'],
                "--resolution", test_config['resolution'],
                "--output-dir", self.temp_dir
            ]
            
            # Execute with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            try:
                stdout, stderr = process.communicate(timeout=test_config.get('expected_time_limit', 600))
                
                if process.returncode == 0:
                    # Parse output for success indicators
                    output_path = self._extract_output_path(stdout)
                    return {
                        'success': True,
                        'output_path': output_path,
                        'stdout': stdout,
                        'stderr': stderr
                    }
                else:
                    return {
                        'success': False,
                        'error_message': f"Process failed with code {process.returncode}: {stderr}",
                        'stdout': stdout,
                        'stderr': stderr
                    }
                    
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    'success': False,
                    'error_message': f"Generation timed out after {test_config.get('expected_time_limit', 600)} seconds"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_message': f"Failed to execute generation test: {e}"
            }
    
    def test_error_handling_and_recovery(self) -> ValidationResult:
        """
        Test error handling and recovery mechanisms using test_error_integration.py
        Requirements: 3.1, 3.3
        """
        print("Testing error handling and recovery mechanisms...")
        
        try:
            # Execute existing error integration tests
            cmd = [sys.executable, "test_error_integration.py"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            stdout, stderr = process.communicate(timeout=120)  # 2 minute timeout
            
            # Parse results
            success = process.returncode == 0
            
            # Extract test results from output
            passed_tests = stdout.count("✓")
            failed_tests = stdout.count("✗")
            
            result = ValidationResult(
                component="error_handling",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                message=f"Error handling tests: {passed_tests} passed, {failed_tests} failed",
                details={
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": process.returncode
                },
                remediation_steps=[] if success else [
                    "Review error handling implementation",
                    "Check error recovery mechanisms",
                    "Validate error logging functionality"
                ]
            )
            
            print(f"Error handling test result: {result.status.value}")
            return result
            
        except subprocess.TimeoutExpired:
            return ValidationResult(
                component="error_handling",
                status=TestStatus.FAILED,
                message="Error handling tests timed out",
                details={"error": "timeout"},
                remediation_steps=["Investigate test performance issues"]
            )
        except Exception as e:
            return ValidationResult(
                component="error_handling", 
                status=TestStatus.FAILED,
                message=f"Failed to run error handling tests: {e}",
                details={"error": str(e)},
                remediation_steps=["Check test_error_integration.py availability"]
            )
    
    def run_comprehensive_integration_tests(self) -> IntegrationTestResults:
        """
        Run comprehensive integration tests covering all components
        Requirements: 3.1, 3.3, 3.4, 3.6
        """
        print("Running comprehensive integration tests...")
        
        start_time = datetime.now()
        
        # Run video generation tests
        generation_results = self.run_video_generation_tests()
        
        # Run error handling tests
        error_handling_result = self.test_error_handling_and_recovery()
        
        # Run UI tests if available
        ui_results = None
        if self.ui_tester:
            ui_results = self.ui_tester.run_ui_tests()
        
        # Run API tests
        api_results = self.api_tester.run_api_tests()
        
        # Run resource monitoring validation
        resource_monitoring_result = self.test_resource_monitoring_accuracy()
        
        # Compile results
        end_time = datetime.now()
        
        self.test_results = IntegrationTestResults(
            start_time=start_time,
            end_time=end_time,
            generation_results=generation_results,
            error_handling_result=error_handling_result,
            ui_results=ui_results,
            api_results=api_results,
            resource_monitoring_result=resource_monitoring_result,
            overall_status=self._determine_overall_status(
                generation_results, error_handling_result, ui_results, 
                api_results, resource_monitoring_result
            )
        )
        
        return self.test_results
    
    def test_resource_monitoring_accuracy(self) -> ValidationResult:
        """
        Test resource monitoring accuracy using test_resource_monitoring.py
        Requirements: 3.4, 3.6
        """
        print("Testing resource monitoring accuracy...")
        
        try:
            # Execute existing resource monitoring tests
            cmd = [sys.executable, "test_resource_monitoring.py"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            stdout, stderr = process.communicate(timeout=60)
            
            success = process.returncode == 0
            
            return ValidationResult(
                component="resource_monitoring",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                message=f"Resource monitoring accuracy test {'passed' if success else 'failed'}",
                details={
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": process.returncode
                },
                remediation_steps=[] if success else [
                    "Check resource monitoring implementation",
                    "Validate metrics collection accuracy",
                    "Review monitoring thresholds"
                ]
            )
            
        except Exception as e:
            return ValidationResult(
                component="resource_monitoring",
                status=TestStatus.FAILED,
                message=f"Failed to run resource monitoring tests: {e}",
                details={"error": str(e)},
                remediation_steps=["Check test_resource_monitoring.py availability"]
            )
    
    def _capture_resource_snapshot(self) -> ResourceMetrics:
        """Capture current resource usage snapshot"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Mock GPU metrics (would use actual GPU monitoring in real implementation)
            gpu_percent = 0.0
            vram_used_mb = 0
            vram_total_mb = 0
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_percent = gpu.load * 100
                    vram_used_mb = gpu.memoryUsed
                    vram_total_mb = gpu.memoryTotal
            except ImportError:
                pass
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                gpu_percent=gpu_percent,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_percent=(vram_used_mb / vram_total_mb * 100) if vram_total_mb > 0 else 0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Warning: Could not capture resource snapshot: {e}")
            return ResourceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                gpu_percent=0.0,
                vram_used_mb=0,
                vram_total_mb=0,
                vram_percent=0.0,
                timestamp=datetime.now()
            )
    
    def _calculate_resource_delta(self, start: ResourceMetrics, end: ResourceMetrics) -> ResourceMetrics:
        """Calculate resource usage delta between two snapshots"""
        return ResourceMetrics(
            cpu_percent=end.cpu_percent - start.cpu_percent,
            memory_percent=end.memory_percent - start.memory_percent,
            memory_used_gb=end.memory_used_gb - start.memory_used_gb,
            memory_total_gb=end.memory_total_gb,
            gpu_percent=end.gpu_percent - start.gpu_percent,
            vram_used_mb=end.vram_used_mb - start.vram_used_mb,
            vram_total_mb=end.vram_total_mb,
            vram_percent=end.vram_percent - start.vram_percent,
            timestamp=end.timestamp
        )
    
    def _extract_output_path(self, stdout: str) -> Optional[str]:
        """Extract output file path from test stdout"""
        lines = stdout.split('\n')
        for line in lines:
            if 'output' in line.lower() and ('.mp4' in line or '.avi' in line):
                # Simple extraction - would be more sophisticated in real implementation
                parts = line.split()
                for part in parts:
                    if part.endswith(('.mp4', '.avi')):
                        return part
        return None
    
    def _determine_overall_status(self, generation_results, error_handling_result, 
                                ui_results, api_results, resource_monitoring_result) -> TestStatus:
        """Determine overall test status based on individual results"""
        failed_components = 0
        total_components = 0
        
        # Check generation results
        if generation_results:
            total_components += 1
            if not all(result.success for result in generation_results):
                failed_components += 1
        
        # Check other results
        for result in [error_handling_result, resource_monitoring_result]:
            if result:
                total_components += 1
                if result.status == TestStatus.FAILED:
                    failed_components += 1
        
        if ui_results:
            total_components += 1
            if ui_results.overall_status == TestStatus.FAILED:
                failed_components += 1
                
        if api_results:
            total_components += 1
            if api_results.overall_status == TestStatus.FAILED:
                failed_components += 1
        
        # Determine status
        if failed_components == 0:
            return TestStatus.PASSED
        elif failed_components < total_components:
            return TestStatus.WARNING
        else:
            return TestStatus.FAILED
    
    def cleanup(self):
        """Clean up temporary resources"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        if self.ui_tester:
            self.ui_tester.cleanup()


class UITester:
    """
    UI testing capabilities using Selenium for automated browser testing
    Requirements: 3.2, 3.5
    """
    
    def __init__(self, config: LocalTestConfiguration):
        self.config = config
        self.driver = None
        self.base_url = "http://localhost:7860"
        self.app_process = None
        
    def setup_headless_browser(self):
        """Set up headless browser for UI testing"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available for UI testing")
        
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"Failed to setup Chrome driver: {e}")
            # Fallback to Firefox
            firefox_options = FirefoxOptions()
            firefox_options.add_argument("--headless")
            return webdriver.Firefox(options=firefox_options)
    
    def launch_application(self) -> bool:
        """Launch main.py --port 7860 for UI testing"""
        try:
            cmd = [sys.executable, "main.py", "--port", "7860"]
            
            self.app_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            # Wait for application to start
            time.sleep(10)
            
            # Check if process is still running
            if self.app_process.poll() is None:
                print("Application launched successfully")
                return True
            else:
                print("Application failed to start")
                return False
                
        except Exception as e:
            print(f"Failed to launch application: {e}")
            return False
    
    def test_browser_access(self) -> ValidationResult:
        """Test browser access to http://localhost:7860"""
        try:
            self.driver = self.setup_headless_browser()
            self.driver.get(self.base_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check if page loaded successfully
            title = self.driver.title
            page_source = self.driver.page_source
            
            success = "Wan2.2" in title or "gradio" in page_source.lower()
            
            return ValidationResult(
                component="browser_access",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                message=f"Browser access test {'passed' if success else 'failed'}",
                details={
                    "url": self.base_url,
                    "title": title,
                    "page_loaded": success
                },
                remediation_steps=[] if success else [
                    "Check if application is running on port 7860",
                    "Verify Gradio interface is properly initialized",
                    "Check for JavaScript errors in browser console"
                ]
            )
            
        except TimeoutException:
            return ValidationResult(
                component="browser_access",
                status=TestStatus.FAILED,
                message="Browser access timed out",
                details={"error": "timeout"},
                remediation_steps=[
                    "Increase application startup time",
                    "Check network connectivity",
                    "Verify port 7860 is not blocked"
                ]
            )
        except Exception as e:
            return ValidationResult(
                component="browser_access",
                status=TestStatus.FAILED,
                message=f"Browser access failed: {e}",
                details={"error": str(e)},
                remediation_steps=[
                    "Check browser driver installation",
                    "Verify application is accessible",
                    "Check for firewall issues"
                ]
            )
    
    def test_ui_functionality(self) -> UITestResults:
        """Test UI functionality and interactions"""
        if not self.driver:
            return UITestResults(
                overall_status=TestStatus.FAILED,
                browser_access_result=ValidationResult(
                    component="ui_functionality",
                    status=TestStatus.FAILED,
                    message="No browser driver available",
                    details={},
                    remediation_steps=["Setup browser driver first"]
                )
            )
        
        test_results = []
        
        # Test model selection
        model_selection_result = self._test_model_selection()
        test_results.append(model_selection_result)
        
        # Test prompt input
        prompt_input_result = self._test_prompt_input()
        test_results.append(prompt_input_result)
        
        # Test generation button
        generation_button_result = self._test_generation_button()
        test_results.append(generation_button_result)
        
        # Determine overall status
        failed_tests = sum(1 for result in test_results if result.status == TestStatus.FAILED)
        overall_status = TestStatus.PASSED if failed_tests == 0 else TestStatus.FAILED
        
        return UITestResults(
            overall_status=overall_status,
            browser_access_result=self.test_browser_access(),
            component_test_results=test_results
        )
    
    def _test_model_selection(self) -> ValidationResult:
        """Test model selection dropdown functionality"""
        try:
            # Look for model selection dropdown
            model_dropdown = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select, .dropdown"))
            )
            
            # Test selecting different models
            success = model_dropdown.is_displayed() and model_dropdown.is_enabled()
            
            return ValidationResult(
                component="model_selection",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                message=f"Model selection test {'passed' if success else 'failed'}",
                details={"element_found": success},
                remediation_steps=[] if success else [
                    "Check model dropdown implementation",
                    "Verify dropdown is properly rendered"
                ]
            )
            
        except TimeoutException:
            return ValidationResult(
                component="model_selection",
                status=TestStatus.FAILED,
                message="Model selection element not found",
                details={"error": "element_not_found"},
                remediation_steps=[
                    "Check UI component rendering",
                    "Verify model selection dropdown exists"
                ]
            )
    
    def _test_prompt_input(self) -> ValidationResult:
        """Test prompt input field functionality"""
        try:
            # Look for prompt input field
            prompt_input = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea, input[type='text']"))
            )
            
            # Test typing in prompt field
            test_prompt = "Test prompt for UI validation"
            prompt_input.clear()
            prompt_input.send_keys(test_prompt)
            
            # Verify text was entered
            entered_text = prompt_input.get_attribute("value") or prompt_input.text
            success = test_prompt in entered_text
            
            return ValidationResult(
                component="prompt_input",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                message=f"Prompt input test {'passed' if success else 'failed'}",
                details={
                    "test_prompt": test_prompt,
                    "entered_text": entered_text,
                    "text_match": success
                },
                remediation_steps=[] if success else [
                    "Check prompt input field implementation",
                    "Verify text input functionality"
                ]
            )
            
        except TimeoutException:
            return ValidationResult(
                component="prompt_input",
                status=TestStatus.FAILED,
                message="Prompt input element not found",
                details={"error": "element_not_found"},
                remediation_steps=[
                    "Check prompt input field rendering",
                    "Verify input element exists"
                ]
            )
    
    def _test_generation_button(self) -> ValidationResult:
        """Test generation button functionality"""
        try:
            # Look for generation button
            generate_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button"))
            )
            
            # Test button click
            initial_text = generate_button.text
            generate_button.click()
            
            # Wait a moment and check for any changes
            time.sleep(1)
            
            success = generate_button.is_displayed() and generate_button.is_enabled()
            
            return ValidationResult(
                component="generation_button",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                message=f"Generation button test {'passed' if success else 'failed'}",
                details={
                    "button_text": initial_text,
                    "clickable": success
                },
                remediation_steps=[] if success else [
                    "Check generation button implementation",
                    "Verify button click handlers"
                ]
            )
            
        except TimeoutException:
            return ValidationResult(
                component="generation_button",
                status=TestStatus.FAILED,
                message="Generation button not found or not clickable",
                details={"error": "button_not_clickable"},
                remediation_steps=[
                    "Check button rendering",
                    "Verify button is properly enabled"
                ]
            )
    
    def test_accessibility_compliance(self) -> ValidationResult:
        """Test accessibility compliance using basic checks"""
        try:
            # Basic accessibility checks
            accessibility_issues = []
            
            # Check for alt text on images
            images = self.driver.find_elements(By.TAG_NAME, "img")
            for img in images:
                if not img.get_attribute("alt"):
                    accessibility_issues.append("Image missing alt text")
            
            # Check for form labels
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            for input_elem in inputs:
                if input_elem.get_attribute("type") != "hidden":
                    label_id = input_elem.get_attribute("id")
                    if label_id:
                        labels = self.driver.find_elements(By.CSS_SELECTOR, f"label[for='{label_id}']")
                        if not labels:
                            accessibility_issues.append(f"Input {label_id} missing label")
            
            success = len(accessibility_issues) == 0
            
            return ValidationResult(
                component="accessibility",
                status=TestStatus.PASSED if success else TestStatus.WARNING,
                message=f"Accessibility check found {len(accessibility_issues)} issues",
                details={"issues": accessibility_issues},
                remediation_steps=accessibility_issues if accessibility_issues else []
            )
            
        except Exception as e:
            return ValidationResult(
                component="accessibility",
                status=TestStatus.FAILED,
                message=f"Accessibility check failed: {e}",
                details={"error": str(e)},
                remediation_steps=["Review accessibility testing implementation"]
            )
    
    def run_ui_tests(self) -> UITestResults:
        """Run comprehensive UI tests"""
        print("Running UI functionality tests...")
        
        # Launch application
        if not self.launch_application():
            return UITestResults(
                overall_status=TestStatus.FAILED,
                browser_access_result=ValidationResult(
                    component="application_launch",
                    status=TestStatus.FAILED,
                    message="Failed to launch application",
                    details={},
                    remediation_steps=["Check application startup process"]
                )
            )
        
        try:
            # Test browser access
            browser_access_result = self.test_browser_access()
            
            if browser_access_result.status == TestStatus.PASSED:
                # Run UI functionality tests
                ui_functionality_results = self.test_ui_functionality()
                
                # Test accessibility
                accessibility_result = self.test_accessibility_compliance()
                ui_functionality_results.component_test_results.append(accessibility_result)
                
                return ui_functionality_results
            else:
                return UITestResults(
                    overall_status=TestStatus.FAILED,
                    browser_access_result=browser_access_result
                )
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up UI testing resources"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        
        if self.app_process:
            try:
                self.app_process.terminate()
                self.app_process.wait(timeout=5)
            except Exception:
                try:
                    self.app_process.kill()
                except Exception:
                    pass
            self.app_process = None


class APITester:
    """
    API testing capabilities using requests library
    Requirements: 3.2, 3.5
    """
    
    def __init__(self, config: LocalTestConfiguration):
        self.config = config
        self.base_url = "http://localhost:7860"
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_health_endpoint(self) -> ValidationResult:
        """Test /health endpoint for "status": "healthy" and GPU availability"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    
                    # Check for required fields
                    has_status = "status" in health_data
                    is_healthy = health_data.get("status") == "healthy"
                    has_gpu_info = "gpu_available" in health_data
                    
                    success = has_status and is_healthy
                    
                    return ValidationResult(
                        component="health_endpoint",
                        status=TestStatus.PASSED if success else TestStatus.FAILED,
                        message=f"Health endpoint test {'passed' if success else 'failed'}",
                        details={
                            "status_code": response.status_code,
                            "response_data": health_data,
                            "has_status": has_status,
                            "is_healthy": is_healthy,
                            "has_gpu_info": has_gpu_info
                        },
                        remediation_steps=[] if success else [
                            "Implement /health endpoint",
                            "Ensure endpoint returns 'status': 'healthy'",
                            "Add GPU availability information"
                        ]
                    )
                    
                except json.JSONDecodeError:
                    return ValidationResult(
                        component="health_endpoint",
                        status=TestStatus.FAILED,
                        message="Health endpoint returned invalid JSON",
                        details={
                            "status_code": response.status_code,
                            "response_text": response.text
                        },
                        remediation_steps=[
                            "Fix health endpoint JSON response format"
                        ]
                    )
            else:
                return ValidationResult(
                    component="health_endpoint",
                    status=TestStatus.FAILED,
                    message=f"Health endpoint returned status {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "response_text": response.text
                    },
                    remediation_steps=[
                        "Check health endpoint implementation",
                        "Verify endpoint is properly configured"
                    ]
                )
                
        except requests.RequestException as e:
            return ValidationResult(
                component="health_endpoint",
                status=TestStatus.FAILED,
                message=f"Health endpoint request failed: {e}",
                details={"error": str(e)},
                remediation_steps=[
                    "Check if application is running",
                    "Verify network connectivity",
                    "Check if port 7860 is accessible"
                ]
            )
    
    def test_authentication_endpoints(self) -> ValidationResult:
        """Test authentication endpoint functionality"""
        try:
            # Test authentication endpoint (if it exists)
            auth_response = self.session.get(f"{self.base_url}/auth")
            
            # For now, just check if endpoint exists or returns appropriate error
            if auth_response.status_code in [200, 401, 403]:
                return ValidationResult(
                    component="authentication",
                    status=TestStatus.PASSED,
                    message="Authentication endpoint responds appropriately",
                    details={
                        "status_code": auth_response.status_code,
                        "response_headers": dict(auth_response.headers)
                    },
                    remediation_steps=[]
                )
            else:
                return ValidationResult(
                    component="authentication",
                    status=TestStatus.WARNING,
                    message=f"Authentication endpoint returned unexpected status {auth_response.status_code}",
                    details={
                        "status_code": auth_response.status_code,
                        "response_text": auth_response.text
                    },
                    remediation_steps=[
                        "Review authentication endpoint implementation"
                    ]
                )
                
        except requests.RequestException:
            # Authentication endpoint might not exist, which is acceptable
            return ValidationResult(
                component="authentication",
                status=TestStatus.WARNING,
                message="Authentication endpoint not available (optional)",
                details={},
                remediation_steps=[]
            )
    
    def test_api_rate_limiting(self) -> ValidationResult:
        """Test API rate limiting functionality"""
        try:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(10):
                response = self.session.get(f"{self.base_url}/health")
                responses.append(response.status_code)
                time.sleep(0.1)
            
            # Check if any requests were rate limited (429 status)
            rate_limited = any(status == 429 for status in responses)
            
            return ValidationResult(
                component="rate_limiting",
                status=TestStatus.PASSED,  # Rate limiting is optional
                message=f"Rate limiting test completed ({'active' if rate_limited else 'not active'})",
                details={
                    "responses": responses,
                    "rate_limited": rate_limited
                },
                remediation_steps=[] if not rate_limited else [
                    "Consider implementing rate limiting for production use"
                ]
            )
            
        except Exception as e:
            return ValidationResult(
                component="rate_limiting",
                status=TestStatus.WARNING,
                message=f"Rate limiting test failed: {e}",
                details={"error": str(e)},
                remediation_steps=["Review rate limiting implementation"]
            )
    
    def test_error_response_formats(self) -> ValidationResult:
        """Test API error response format validation"""
        try:
            # Test with invalid endpoint to trigger error response
            response = self.session.get(f"{self.base_url}/invalid_endpoint")
            
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    has_error_field = "error" in error_data or "message" in error_data
                    
                    return ValidationResult(
                        component="error_responses",
                        status=TestStatus.PASSED if has_error_field else TestStatus.WARNING,
                        message=f"Error response format {'valid' if has_error_field else 'could be improved'}",
                        details={
                            "status_code": response.status_code,
                            "response_data": error_data,
                            "has_error_field": has_error_field
                        },
                        remediation_steps=[] if has_error_field else [
                            "Standardize error response format",
                            "Include 'error' or 'message' field in error responses"
                        ]
                    )
                    
                except json.JSONDecodeError:
                    return ValidationResult(
                        component="error_responses",
                        status=TestStatus.WARNING,
                        message="Error responses are not in JSON format",
                        details={
                            "status_code": response.status_code,
                            "response_text": response.text
                        },
                        remediation_steps=[
                            "Consider using JSON format for error responses"
                        ]
                    )
            else:
                return ValidationResult(
                    component="error_responses",
                    status=TestStatus.WARNING,
                    message=f"Unexpected response for invalid endpoint: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "response_text": response.text
                    },
                    remediation_steps=[
                        "Review error handling for invalid endpoints"
                    ]
                )
                
        except Exception as e:
            return ValidationResult(
                component="error_responses",
                status=TestStatus.WARNING,
                message=f"Error response test failed: {e}",
                details={"error": str(e)},
                remediation_steps=["Review error response implementation"]
            )
    
    def run_api_tests(self) -> APITestResults:
        """Run comprehensive API tests"""
        print("Running API validation tests...")
        
        test_results = []
        
        # Test health endpoint
        health_result = self.test_health_endpoint()
        test_results.append(health_result)
        
        # Test authentication endpoints
        auth_result = self.test_authentication_endpoints()
        test_results.append(auth_result)
        
        # Test rate limiting
        rate_limit_result = self.test_api_rate_limiting()
        test_results.append(rate_limit_result)
        
        # Test error response formats
        error_format_result = self.test_error_response_formats()
        test_results.append(error_format_result)
        
        # Determine overall status
        failed_tests = sum(1 for result in test_results if result.status == TestStatus.FAILED)
        overall_status = TestStatus.PASSED if failed_tests == 0 else TestStatus.FAILED
        
        return APITestResults(
            overall_status=overall_status,
            endpoint_test_results=test_results
        )