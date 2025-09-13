#!/usr/bin/env python3
"""
Integration Test Report Generator
Creates comprehensive test reports with pass/fail status for integration testing components.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .models.test_results import (
    IntegrationTestResults, UITestResults, APITestResults,
    ValidationResult, TestStatus, ValidationStatus
)


class IntegrationTestReportGenerator:
    """
    Generates comprehensive test reports for integration testing results
    Requirements: 3.4, 3.6
    """
    
    def __init__(self, output_dir: str = "integration_test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, results: IntegrationTestResults) -> Dict[str, str]:
        """Generate comprehensive test report with pass/fail status"""
        
        # Generate different report formats
        report_files = {}
        
        # JSON report for programmatic access
        json_report_path = os.path.join(self.output_dir, "integration_test_report.json")
        self._generate_json_report(results, json_report_path)
        report_files["json"] = json_report_path
        
        # HTML report for human readability
        html_report_path = os.path.join(self.output_dir, "integration_test_report.html")
        self._generate_html_report(results, html_report_path)
        report_files["html"] = html_report_path
        
        # Text summary report
        text_report_path = os.path.join(self.output_dir, "integration_test_summary.txt")
        self._generate_text_summary(results, text_report_path)
        report_files["text"] = text_report_path
        
        return report_files
    
    def _generate_json_report(self, results: IntegrationTestResults, output_path: str):
        """Generate JSON format report"""
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "integration_test_results",
                "version": "1.0"
            },
            "test_execution": {
                "start_time": results.start_time.isoformat(),
                "end_time": results.end_time.isoformat(),
                "duration_seconds": (results.end_time - results.start_time).total_seconds(),
                "overall_status": results.overall_status.value
            },
            "test_results": {
                "generation_tests": self._format_generation_results(results.generation_results),
                "error_handling": self._format_validation_result(results.error_handling_result),
                "ui_tests": self._format_ui_results(results.ui_results),
                "api_tests": self._format_api_results(results.api_results),
                "resource_monitoring": self._format_validation_result(results.resource_monitoring_result)
            },
            "summary": self._generate_test_summary(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_html_report(self, results: IntegrationTestResults, output_path: str):
        """Generate HTML format report"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-passed {{ color: #28a745; font-weight: bold; }}
        .status-failed {{ color: #dc3545; font-weight: bold; }}
        .status-warning {{ color: #ffc107; font-weight: bold; }}
        .test-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .test-details {{ margin-left: 20px; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .summary-table th {{ background-color: #f2f2f2; }}
        .remediation {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Integration Test Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Test Duration:</strong> {(results.end_time - results.start_time).total_seconds():.1f} seconds</p>
        <p><strong>Overall Status:</strong> <span class="status-{results.overall_status.value.lower()}">{results.overall_status.value.upper()}</span></p>
    </div>
    
    {self._generate_html_summary_table(results)}
    
    <h2>Detailed Test Results</h2>
    
    {self._generate_html_generation_results(results.generation_results)}
    
    {self._generate_html_validation_result("Error Handling Tests", results.error_handling_result)}
    
    {self._generate_html_ui_results(results.ui_results)}
    
    {self._generate_html_api_results(results.api_results)}
    
    {self._generate_html_validation_result("Resource Monitoring Tests", results.resource_monitoring_result)}
    
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_text_summary(self, results: IntegrationTestResults, output_path: str):
        """Generate text format summary report"""
        
        summary = self._generate_test_summary(results)
        
        text_content = f"""
INTEGRATION TEST REPORT SUMMARY
===============================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test Duration: {(results.end_time - results.start_time).total_seconds():.1f} seconds
Overall Status: {results.overall_status.value.upper()}

TEST SUMMARY
------------
Total Test Categories: {summary['total_categories']}
Passed Categories: {summary['passed_categories']}
Failed Categories: {summary['failed_categories']}
Warning Categories: {summary['warning_categories']}
Success Rate: {summary['success_rate']:.1f}%

DETAILED RESULTS
----------------

Generation Tests:
  Total: {len(results.generation_results)}
  Passed: {sum(1 for r in results.generation_results if r.success)}
  Failed: {sum(1 for r in results.generation_results if not r.success)}

Error Handling: {results.error_handling_result.status.value.upper() if results.error_handling_result else 'NOT_RUN'}

UI Tests: {results.ui_results.overall_status.value.upper() if results.ui_results else 'NOT_RUN'}

API Tests: {results.api_results.overall_status.value.upper() if results.api_results else 'NOT_RUN'}

Resource Monitoring: {results.resource_monitoring_result.status.value.upper() if results.resource_monitoring_result else 'NOT_RUN'}

RECOMMENDATIONS
---------------
{self._generate_text_recommendations(results)}
        """
        
        with open(output_path, 'w') as f:
            f.write(text_content.strip())
    
    def _format_generation_results(self, generation_results: List[Any]) -> List[Dict[str, Any]]:
        """Format generation test results for JSON output"""
        formatted_results = []
        
        for result in generation_results:
            if hasattr(result, '__dict__'):
                # Convert dataclass to dict
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
                
                # Ensure resource_usage is properly formatted
                if 'resource_usage' in result_dict and result_dict['resource_usage']:
                    if hasattr(result_dict['resource_usage'], 'to_dict'):
                        result_dict['resource_usage'] = result_dict['resource_usage'].to_dict()
                
                formatted_results.append(result_dict)
            else:
                formatted_results.append(str(result))
        
        return formatted_results
    
    def _format_validation_result(self, result: Optional[ValidationResult]) -> Optional[Dict[str, Any]]:
        """Format validation result for JSON output"""
        if result:
            return result.to_dict()
        return None
    
    def _format_ui_results(self, ui_results: Optional[UITestResults]) -> Optional[Dict[str, Any]]:
        """Format UI test results for JSON output"""
        if ui_results:
            return ui_results.to_dict()
        return None
    
    def _format_api_results(self, api_results: Optional[APITestResults]) -> Optional[Dict[str, Any]]:
        """Format API test results for JSON output"""
        if api_results:
            return api_results.to_dict()
        return None
    
    def _generate_test_summary(self, results: IntegrationTestResults) -> Dict[str, Any]:
        """Generate test summary statistics"""
        
        total_categories = 0
        passed_categories = 0
        failed_categories = 0
        warning_categories = 0
        
        # Count generation tests
        if results.generation_results:
            total_categories += 1
            if all(r.success for r in results.generation_results):
                passed_categories += 1
            else:
                failed_categories += 1
        
        # Count other test categories
        test_results = [
            results.error_handling_result,
            results.resource_monitoring_result
        ]
        
        for result in test_results:
            if result:
                total_categories += 1
                if result.status == TestStatus.PASSED:
                    passed_categories += 1
                elif result.status == TestStatus.FAILED:
                    failed_categories += 1
                else:
                    warning_categories += 1
        
        # Count UI and API tests
        if results.ui_results:
            total_categories += 1
            if results.ui_results.overall_status == TestStatus.PASSED:
                passed_categories += 1
            elif results.ui_results.overall_status == TestStatus.FAILED:
                failed_categories += 1
            else:
                warning_categories += 1
        
        if results.api_results:
            total_categories += 1
            if results.api_results.overall_status == TestStatus.PASSED:
                passed_categories += 1
            elif results.api_results.overall_status == TestStatus.FAILED:
                failed_categories += 1
            else:
                warning_categories += 1
        
        success_rate = (passed_categories / total_categories * 100) if total_categories > 0 else 0
        
        return {
            "total_categories": total_categories,
            "passed_categories": passed_categories,
            "failed_categories": failed_categories,
            "warning_categories": warning_categories,
            "success_rate": success_rate
        }
    
    def _generate_html_summary_table(self, results: IntegrationTestResults) -> str:
        """Generate HTML summary table"""
        
        summary = self._generate_test_summary(results)
        
        return f"""
        <h2>Test Summary</h2>
        <table class="summary-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Test Categories</td>
                <td>{summary['total_categories']}</td>
            </tr>
            <tr>
                <td>Passed Categories</td>
                <td class="status-passed">{summary['passed_categories']}</td>
            </tr>
            <tr>
                <td>Failed Categories</td>
                <td class="status-failed">{summary['failed_categories']}</td>
            </tr>
            <tr>
                <td>Warning Categories</td>
                <td class="status-warning">{summary['warning_categories']}</td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td>{summary['success_rate']:.1f}%</td>
            </tr>
        </table>
        """
    
    def _generate_html_generation_results(self, generation_results: List[Any]) -> str:
        """Generate HTML for generation test results"""
        
        if not generation_results:
            return '<div class="test-section"><h3>Generation Tests</h3><p>No generation tests were run.</p></div>'
        
        html = '<div class="test-section"><h3>Generation Tests</h3>'
        
        for i, result in enumerate(generation_results):
            status_class = "status-passed" if result.success else "status-failed"
            status_text = "PASSED" if result.success else "FAILED"
            
            html += f"""
            <div class="test-details">
                <h4>Test {i+1}: {result.model_type} - {result.resolution}</h4>
                <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
                <p><strong>Prompt:</strong> {result.prompt}</p>
                <p><strong>Generation Time:</strong> {result.generation_time_seconds:.1f} seconds</p>
                {f'<p><strong>Output Path:</strong> {result.output_path}</p>' if result.output_path else ''}
                {f'<div class="remediation"><strong>Error:</strong> {result.error_message}</div>' if result.error_message else ''}
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_html_validation_result(self, title: str, result: Optional[ValidationResult]) -> str:
        """Generate HTML for validation result"""
        
        if not result:
            return f'<div class="test-section"><h3>{title}</h3><p>Test was not run.</p></div>'
        
        status_class = f"status-{result.status.value.lower()}"
        status_text = result.status.value.upper()
        
        html = f"""
        <div class="test-section">
            <h3>{title}</h3>
            <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
            <p><strong>Message:</strong> {result.message}</p>
        """
        
        if result.remediation_steps:
            html += '<div class="remediation"><strong>Remediation Steps:</strong><ul>'
            for step in result.remediation_steps:
                html += f'<li>{step}</li>'
            html += '</ul></div>'
        
        html += '</div>'
        return html
    
    def _generate_html_ui_results(self, ui_results: Optional[UITestResults]) -> str:
        """Generate HTML for UI test results"""
        
        if not ui_results:
            return '<div class="test-section"><h3>UI Tests</h3><p>UI tests were not run.</p></div>'
        
        status_class = f"status-{ui_results.overall_status.value.lower()}"
        status_text = ui_results.overall_status.value.upper()
        
        html = f"""
        <div class="test-section">
            <h3>UI Tests</h3>
            <p><strong>Overall Status:</strong> <span class="{status_class}">{status_text}</span></p>
            
            <h4>Browser Access Test</h4>
            {self._generate_html_validation_result("", ui_results.browser_access_result).replace('<div class="test-section"><h3></h3>', '<div class="test-details">').replace('</div>', '</div>', 1)}
        """
        
        if ui_results.component_test_results:
            html += '<h4>Component Tests</h4>'
            for result in ui_results.component_test_results:
                html += self._generate_html_validation_result("", result).replace('<div class="test-section"><h3></h3>', '<div class="test-details">').replace('</div>', '</div>', 1)
        
        html += '</div>'
        return html
    
    def _generate_html_api_results(self, api_results: Optional[APITestResults]) -> str:
        """Generate HTML for API test results"""
        
        if not api_results:
            return '<div class="test-section"><h3>API Tests</h3><p>API tests were not run.</p></div>'
        
        status_class = f"status-{api_results.overall_status.value.lower()}"
        status_text = api_results.overall_status.value.upper()
        
        html = f"""
        <div class="test-section">
            <h3>API Tests</h3>
            <p><strong>Overall Status:</strong> <span class="{status_class}">{status_text}</span></p>
        """
        
        if api_results.endpoint_test_results:
            html += '<h4>Endpoint Tests</h4>'
            for result in api_results.endpoint_test_results:
                html += self._generate_html_validation_result("", result).replace('<div class="test-section"><h3></h3>', '<div class="test-details">').replace('</div>', '</div>', 1)
        
        html += '</div>'
        return html
    
    def _generate_text_recommendations(self, results: IntegrationTestResults) -> str:
        """Generate text recommendations based on test results"""
        
        recommendations = []
        
        # Check generation test failures
        if results.generation_results:
            failed_generations = [r for r in results.generation_results if not r.success]
            if failed_generations:
                recommendations.append("- Review failed generation tests and check model configurations")
                recommendations.append("- Verify system resources meet generation requirements")
        
        # Check error handling
        if results.error_handling_result and results.error_handling_result.status == TestStatus.FAILED:
            recommendations.extend([f"- {step}" for step in results.error_handling_result.remediation_steps])
        
        # Check UI tests
        if results.ui_results and results.ui_results.overall_status == TestStatus.FAILED:
            recommendations.append("- Review UI test failures and check browser compatibility")
            recommendations.append("- Verify application is properly accessible on port 7860")
        
        # Check API tests
        if results.api_results and results.api_results.overall_status == TestStatus.FAILED:
            recommendations.append("- Review API endpoint implementations")
            recommendations.append("- Check API response formats and error handling")
        
        # Check resource monitoring
        if results.resource_monitoring_result and results.resource_monitoring_result.status == TestStatus.FAILED:
            recommendations.extend([f"- {step}" for step in results.resource_monitoring_result.remediation_steps])
        
        if not recommendations:
            recommendations.append("- All tests passed successfully. No immediate action required.")
        
        return '\n'.join(recommendations)


def generate_integration_test_report(results: IntegrationTestResults, output_dir: str = "integration_test_results") -> Dict[str, str]:
    """
    Convenience function to generate comprehensive integration test report
    
    Args:
        results: Integration test results to report on
        output_dir: Directory to save reports to
        
    Returns:
        Dictionary mapping report format to file path
    """
    generator = IntegrationTestReportGenerator(output_dir)
    return generator.generate_comprehensive_report(results)
