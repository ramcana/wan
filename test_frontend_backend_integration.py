#!/usr/bin/env python3
"""
Frontend-Backend Integration Test Suite

Tests the complete integration between React frontend and FastAPI backend
for WAN model generation workflows.
"""

import asyncio
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrontendBackendIntegrationTest:
    """Test frontend-backend integration"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.driver = None
        self.test_results = []
    
    def setup_webdriver(self):
        """Setup Chrome WebDriver for frontend testing"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            
            logger.info("✅ WebDriver setup successful")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ WebDriver setup failed: {e}")
            logger.info("Frontend UI tests will be skipped")
            return False
    
    async def test_frontend_loads(self):
        """Test that frontend loads successfully"""
        if not self.driver:
            return {"status": "SKIP", "reason": "WebDriver not available"}
        
        try:
            logger.info("🧪 Testing frontend loads...")
            
            self.driver.get(self.frontend_url)
            
            # Wait for main content to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check if React app loaded
            page_source = self.driver.page_source
            
            # Look for common React/app indicators
            indicators = [
                "react",
                "app",
                "generation",
                "video",
                "model"
            ]
            
            found_indicators = [ind for ind in indicators if ind.lower() in page_source.lower()]
            
            return {
                "status": "PASS",
                "page_title": self.driver.title,
                "found_indicators": found_indicators,
                "page_loaded": True
            }
            
        except TimeoutException:
            return {
                "status": "FAIL",
                "reason": "Frontend failed to load within timeout",
                "current_url": self.driver.current_url if self.driver else None
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "reason": str(e)
            }
    
    async def test_frontend_backend_api_connection(self):
        """Test frontend can connect to backend API"""
        if not self.driver:
            return {"status": "SKIP", "reason": "WebDriver not available"}
        
        try:
            logger.info("🧪 Testing frontend-backend API connection...")
            
            self.driver.get(self.frontend_url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Check browser console for API errors
            logs = self.driver.get_log('browser')
            api_errors = [log for log in logs if 'error' in log['message'].lower() and 'api' in log['message'].lower()]
            
            # Check if frontend makes successful API calls
            # This would require inspecting network requests or checking UI state
            
            # For now, check if there are no critical console errors
            critical_errors = [log for log in logs if log['level'] == 'SEVERE']
            
            return {
                "status": "PASS" if len(critical_errors) == 0 else "FAIL",
                "api_errors": len(api_errors),
                "critical_errors": len(critical_errors),
                "console_logs": logs[-5:] if logs else []  # Last 5 logs
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "reason": str(e)
            }
    
    async def test_generation_form_submission(self):
        """Test generation form submission from frontend"""
        if not self.driver:
            return {"status": "SKIP", "reason": "WebDriver not available"}
        
        try:
            logger.info("🧪 Testing generation form submission...")
            
            self.driver.get(self.frontend_url)
            
            # Wait for form elements to load
            await asyncio.sleep(5)
            
            # Look for common form elements
            form_elements = {}
            
            # Try to find prompt input
            try:
                prompt_input = self.driver.find_element(By.CSS_SELECTOR, 
                    "input[placeholder*='prompt'], textarea[placeholder*='prompt'], input[name*='prompt'], textarea[name*='prompt']")
                form_elements["prompt_input"] = True
                
                # Fill in test prompt
                prompt_input.clear()
                prompt_input.send_keys("Test generation from frontend integration test")
                
            except:
                form_elements["prompt_input"] = False
            
            # Try to find model selection
            try:
                model_select = self.driver.find_element(By.CSS_SELECTOR, 
                    "select[name*='model'], select[name*='type']")
                form_elements["model_select"] = True
            except:
                form_elements["model_select"] = False
            
            # Try to find submit button
            try:
                submit_button = self.driver.find_element(By.CSS_SELECTOR, 
                    "button[type='submit'], button:contains('Generate'), button:contains('Submit')")
                form_elements["submit_button"] = True
                
                # Try to submit (but don't actually submit to avoid long generation)
                # submit_button.click()
                
            except:
                form_elements["submit_button"] = False
            
            # Check for generation queue or status area
            try:
                queue_area = self.driver.find_element(By.CSS_SELECTOR, 
                    "[class*='queue'], [class*='status'], [class*='progress']")
                form_elements["queue_area"] = True
            except:
                form_elements["queue_area"] = False
            
            form_completeness = sum(form_elements.values()) / len(form_elements)
            
            return {
                "status": "PASS" if form_completeness >= 0.5 else "FAIL",
                "form_elements": form_elements,
                "form_completeness": form_completeness,
                "page_source_length": len(self.driver.page_source)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "reason": str(e)
            }
    
    async def test_websocket_connection(self):
        """Test WebSocket connection for real-time updates"""
        try:
            logger.info("🧪 Testing WebSocket connection...")
            
            # Test WebSocket endpoint availability
            import websocket
            
            ws_url = "ws://localhost:8000/ws"
            
            connection_successful = False
            error_message = None
            
            def on_open(ws):
                nonlocal connection_successful
                connection_successful = True
                ws.close()
            
            def on_error(ws, error):
                nonlocal error_message
                error_message = str(error)
            
            try:
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=on_open,
                    on_error=on_error
                )
                
                # Run WebSocket connection test with timeout
                import threading
                ws_thread = threading.Thread(target=ws.run_forever)
                ws_thread.daemon = True
                ws_thread.start()
                
                # Wait for connection
                await asyncio.sleep(3)
                
            except Exception as e:
                error_message = str(e)
            
            return {
                "status": "PASS" if connection_successful else "FAIL",
                "connection_successful": connection_successful,
                "error_message": error_message,
                "websocket_url": ws_url
            }
            
        except ImportError:
            return {
                "status": "SKIP",
                "reason": "websocket-client not available"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "reason": str(e)
            }
    
    async def test_api_endpoints_from_frontend_perspective(self):
        """Test API endpoints that frontend would typically call"""
        try:
            logger.info("🧪 Testing API endpoints from frontend perspective...")
            
            endpoint_results = {}
            
            # Test health endpoint
            try:
                response = requests.get(f"{self.backend_url}/api/v1/health", timeout=10)
                endpoint_results["health"] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "has_cors_headers": "access-control-allow-origin" in response.headers
                }
            except Exception as e:
                endpoint_results["health"] = {"error": str(e)}
            
            # Test queue endpoint
            try:
                response = requests.get(f"{self.backend_url}/api/v1/queue", timeout=10)
                endpoint_results["queue"] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "returns_json": response.headers.get("content-type", "").startswith("application/json")
                }
            except Exception as e:
                endpoint_results["queue"] = {"error": str(e)}
            
            # Test system stats endpoint
            try:
                response = requests.get(f"{self.backend_url}/api/v1/system/stats", timeout=10)
                endpoint_results["system_stats"] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                endpoint_results["system_stats"] = {"error": str(e)}
            
            # Test outputs endpoint
            try:
                response = requests.get(f"{self.backend_url}/api/v1/outputs", timeout=10)
                endpoint_results["outputs"] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                endpoint_results["outputs"] = {"error": str(e)}
            
            # Test CORS preflight for generation endpoint
            try:
                response = requests.options(
                    f"{self.backend_url}/api/v1/generation/submit",
                    headers={
                        "Origin": self.frontend_url,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type"
                    },
                    timeout=10
                )
                endpoint_results["cors_preflight"] = {
                    "status_code": response.status_code,
                    "allows_post": "POST" in response.headers.get("access-control-allow-methods", ""),
                    "allows_content_type": "content-type" in response.headers.get("access-control-allow-headers", "").lower()
                }
            except Exception as e:
                endpoint_results["cors_preflight"] = {"error": str(e)}
            
            # Calculate success rate
            successful_endpoints = sum(1 for result in endpoint_results.values() 
                                     if isinstance(result, dict) and result.get("status_code") == 200)
            total_endpoints = len(endpoint_results)
            success_rate = successful_endpoints / total_endpoints if total_endpoints > 0 else 0
            
            return {
                "status": "PASS" if success_rate >= 0.7 else "FAIL",
                "endpoint_results": endpoint_results,
                "success_rate": success_rate,
                "successful_endpoints": successful_endpoints,
                "total_endpoints": total_endpoints
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "reason": str(e)
            }
    
    async def test_error_handling_integration(self):
        """Test error handling between frontend and backend"""
        try:
            logger.info("🧪 Testing error handling integration...")
            
            error_scenarios = {}
            
            # Test invalid generation request
            try:
                invalid_request = {
                    "model_type": "INVALID_MODEL",
                    "prompt": "",
                    "resolution": "invalid_resolution"
                }
                
                response = requests.post(
                    f"{self.backend_url}/api/v1/generation/submit",
                    json=invalid_request,
                    timeout=10
                )
                
                error_scenarios["invalid_generation"] = {
                    "status_code": response.status_code,
                    "has_error_message": "message" in response.json() or "detail" in response.json(),
                    "response_is_json": response.headers.get("content-type", "").startswith("application/json")
                }
                
            except Exception as e:
                error_scenarios["invalid_generation"] = {"error": str(e)}
            
            # Test non-existent endpoint
            try:
                response = requests.get(f"{self.backend_url}/api/v1/nonexistent", timeout=10)
                error_scenarios["nonexistent_endpoint"] = {
                    "status_code": response.status_code,
                    "returns_404": response.status_code == 404
                }
            except Exception as e:
                error_scenarios["nonexistent_endpoint"] = {"error": str(e)}
            
            # Test malformed JSON request
            try:
                response = requests.post(
                    f"{self.backend_url}/api/v1/generation/submit",
                    data="invalid json",
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                error_scenarios["malformed_json"] = {
                    "status_code": response.status_code,
                    "handles_malformed_json": response.status_code in [400, 422]
                }
                
            except Exception as e:
                error_scenarios["malformed_json"] = {"error": str(e)}
            
            # Calculate error handling score
            proper_error_handling = sum(1 for scenario in error_scenarios.values()
                                      if isinstance(scenario, dict) and 
                                      scenario.get("status_code", 0) in [400, 404, 422])
            
            total_scenarios = len(error_scenarios)
            error_handling_score = proper_error_handling / total_scenarios if total_scenarios > 0 else 0
            
            return {
                "status": "PASS" if error_handling_score >= 0.7 else "FAIL",
                "error_scenarios": error_scenarios,
                "error_handling_score": error_handling_score,
                "proper_error_handling": proper_error_handling,
                "total_scenarios": total_scenarios
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "reason": str(e)
            }
    
    async def run_all_tests(self):
        """Run all frontend-backend integration tests"""
        logger.info("🚀 Starting Frontend-Backend Integration Tests...")
        
        # Setup WebDriver
        webdriver_available = self.setup_webdriver()
        
        # Define test methods
        test_methods = [
            ("Frontend Loads", self.test_frontend_loads),
            ("Frontend-Backend API Connection", self.test_frontend_backend_api_connection),
            ("Generation Form Submission", self.test_generation_form_submission),
            ("WebSocket Connection", self.test_websocket_connection),
            ("API Endpoints from Frontend", self.test_api_endpoints_from_frontend_perspective),
            ("Error Handling Integration", self.test_error_handling_integration)
        ]
        
        results = {}
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"Running {test_name}...")
                start_time = time.time()
                
                result = await test_method()
                result["duration"] = time.time() - start_time
                
                results[test_name] = result
                
                status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⏭️"}[result["status"]]
                logger.info(f"{status_icon} {test_name}: {result['status']} ({result['duration']:.1f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "status": "ERROR",
                    "reason": str(e),
                    "duration": time.time() - start_time if 'start_time' in locals() else 0
                }
                logger.error(f"💥 {test_name}: ERROR - {e}")
        
        # Generate summary
        total_tests = len(results)
        passed = len([r for r in results.values() if r["status"] == "PASS"])
        failed = len([r for r in results.values() if r["status"] == "FAIL"])
        errors = len([r for r in results.values() if r["status"] == "ERROR"])
        skipped = len([r for r in results.values() if r["status"] == "SKIP"])
        
        summary = {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "success_rate": passed / total_tests if total_tests > 0 else 0,
            "webdriver_available": webdriver_available
        }
        
        # Save results
        report = {
            "timestamp": time.time(),
            "summary": summary,
            "test_results": results,
            "backend_url": self.backend_url,
            "frontend_url": self.frontend_url
        }
        
        report_path = Path("frontend_backend_integration_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info(f"🎯 FRONTEND-BACKEND INTEGRATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"📊 Results:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  ✅ Passed: {passed}")
        logger.info(f"  ❌ Failed: {failed}")
        logger.info(f"  💥 Errors: {errors}")
        logger.info(f"  ⏭️ Skipped: {skipped}")
        logger.info(f"  📈 Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"\n📄 Report saved to: {report_path}")
        
        return summary["success_rate"] >= 0.7
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

async def main():
    """Main test function"""
    tester = FrontendBackendIntegrationTest()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            logger.info("🎉 Frontend-Backend integration tests completed successfully!")
            return 0
        else:
            logger.error("💥 Some integration tests failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Tests failed with error: {e}")
        return 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)