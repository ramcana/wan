"""
Tests for report generator module
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from ..report_generator import ReportGenerator, HTMLReport, JSONReport, PDFReport, TroubleshootingGuide
from ..models.test_results import (
    TestResults, EnvironmentValidationResults, ValidationResult,
    ValidationStatus, TestStatus, PerformanceTestResults, BenchmarkResult,
    OptimizationResult
)


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(output_dir=self.temp_dir)
        
        # Create sample test results
        self.sample_results = self._create_sample_results()
    
    def _create_sample_results(self) -> TestResults:
        """Create sample test results for testing"""
        # Create environment validation results
        env_results = EnvironmentValidationResults(
            python_version=ValidationResult(
                component="python_version",
                status=ValidationStatus.PASSED,
                message="Python 3.9.7 detected"
            ),
            cuda_availability=ValidationResult(
                component="cuda_availability",
                status=ValidationStatus.PASSED,
                message="CUDA 11.8 available"
            ),
            dependencies=ValidationResult(
                component="dependencies",
                status=ValidationStatus.PASSED,
                message="All dependencies satisfied"
            ),
            configuration=ValidationResult(
                component="configuration",
                status=ValidationStatus.FAILED,
                message="Missing HF_TOKEN in .env",
                remediation_steps=["Add HF_TOKEN to .env file"]
            ),
            environment_variables=ValidationResult(
                component="environment_variables",
                status=ValidationStatus.WARNING,
                message="Some optional variables missing"
            ),
            overall_status=ValidationStatus.WARNING,
            remediation_steps=["Fix configuration issues"]
        )
        
        # Create performance test results
        perf_results = PerformanceTestResults(
            benchmark_720p=BenchmarkResult(
                resolution="720p",
                generation_time=7.2,
                target_time=9.0,
                meets_target=True,
                vram_usage=8.4,
                cpu_usage=45.0,
                memory_usage=60.0,
                optimization_level="high"
            ),
            benchmark_1080p=BenchmarkResult(
                resolution="1080p",
                generation_time=18.5,
                target_time=17.0,
                meets_target=False,
                vram_usage=11.2,
                cpu_usage=55.0,
                memory_usage=70.0,
                optimization_level="medium"
            ),
            vram_optimization=OptimizationResult(
                baseline_vram_mb=12288,
                optimized_vram_mb=8192,
                reduction_percent=33.3,
                target_reduction_percent=80.0,
                meets_target=False,
                optimizations_applied=["attention_slicing", "vae_tiling"]
            ),
            overall_status=TestStatus.PARTIAL,
            recommendations=["Enable CPU offload", "Reduce batch size"]
        )
        
        # Create main test results
        results = TestResults(
            session_id="test_session_123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            environment_results=env_results,
            performance_results=perf_results,
            overall_status=TestStatus.PARTIAL,
            recommendations=["Enable attention slicing", "Update configuration"]
        )
        
        return results
    
    def test_html_report_generation(self):
        """Test HTML report generation"""
        report = self.generator.generate_html_report(self.sample_results)
        
        # Verify report structure
        self.assertIsInstance(report, HTMLReport)
        self.assertIn("test_session_123", report.title)
        self.assertIsInstance(report.content, str)
        self.assertGreater(len(report.content), 0)
        self.assertIsInstance(report.charts, list)
        self.assertGreater(len(report.charts), 0)
        
        # Verify HTML content contains expected elements
        self.assertIn("<!DOCTYPE html>", report.content)
        self.assertIn("Test Report - test_session_123", report.content)
        self.assertIn("Environment Validation", report.content)
        self.assertIn("Performance Visualization", report.content)
        self.assertIn("chart.js", report.content)
        
        # Verify status badges are included
        self.assertIn("status-passed", report.content)
        self.assertIn("status-failed", report.content)
        self.assertIn("status-warning", report.content)
        
        # Verify remediation steps are included
        self.assertIn("Add HF_TOKEN to .env file", report.content)
        self.assertIn("Enable attention slicing", report.content)
    
    def test_json_report_generation(self):
        """Test JSON report generation"""
        report = self.generator.generate_json_report(self.sample_results)
        
        # Verify report structure
        self.assertIsInstance(report, JSONReport)
        self.assertIsInstance(report.data, dict)
        
        # Verify report data structure
        self.assertIn("report_metadata", report.data)
        self.assertIn("test_results", report.data)
        self.assertIn("summary", report.data)
        
        # Verify metadata
        metadata = report.data["report_metadata"]
        self.assertIn("generated_at", metadata)
        self.assertEqual(metadata["session_id"], "test_session_123")
        self.assertEqual(metadata["report_version"], "1.0")
        
        # Verify test results are properly serialized
        test_results = report.data["test_results"]
        self.assertEqual(test_results["session_id"], "test_session_123")
        self.assertEqual(test_results["overall_status"], "partial")
        
        # Verify environment results are included
        self.assertIn("environment_results", test_results)
        env_results = test_results["environment_results"]
        self.assertEqual(env_results["overall_status"], "warning")
        self.assertIn("python_version", env_results)
        self.assertIn("cuda_availability", env_results)
        
        # Verify summary
        summary = report.data["summary"]
        self.assertEqual(summary["overall_status"], "partial")
        self.assertEqual(summary["recommendations_count"], 2)
        self.assertIn("environment_validation", summary)
    
    def test_chart_generation(self):
        """Test performance chart generation"""
        report = self.generator.generate_html_report(self.sample_results)
        
        # Verify charts are generated
        self.assertGreater(len(report.charts), 0)
        
        # Check environment status chart
        env_chart = report.charts[0]
        self.assertEqual(env_chart.chart_type, "bar")
        self.assertEqual(env_chart.title, "Environment Validation Status")
        self.assertIn("Passed", env_chart.labels)
        self.assertIn("Failed", env_chart.labels)
        self.assertIn("Warning", env_chart.labels)
        
        # Check resource metrics chart (should be the last chart)
        resource_chart = report.charts[-1]
        self.assertEqual(resource_chart.chart_type, "line")
        self.assertEqual(resource_chart.title, "Resource Usage Over Time")
        self.assertGreater(len(resource_chart.datasets), 0)
        
        # Verify dataset structure
        for dataset in resource_chart.datasets:
            self.assertIn("label", dataset)
            self.assertIn("data", dataset)
            self.assertIn("borderColor", dataset)
    
    def test_file_output(self):
        """Test that reports are saved to files"""
        # Generate reports
        html_report = self.generator.generate_html_report(self.sample_results)
        json_report = self.generator.generate_json_report(self.sample_results)
        
        # Check that files were created
        output_path = Path(self.temp_dir)
        html_files = list(output_path.glob("*.html"))
        json_files = list(output_path.glob("*.json"))
        
        self.assertEqual(len(html_files), 1)
        self.assertEqual(len(json_files), 1)
        
        # Verify file contents
        with open(html_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
            self.assertIn("test_session_123", html_content)
            self.assertIn("Environment Validation", html_content)
        
        with open(json_files[0], 'r', encoding='utf-8') as f:
            json_content = json.load(f)
            self.assertEqual(json_content["report_metadata"]["session_id"], "test_session_123")
    
    def test_empty_results_handling(self):
        """Test handling of empty or minimal results"""
        minimal_results = TestResults(
            session_id="minimal_test",
            start_time=datetime.now(),
            overall_status=TestStatus.ERROR
        )
        
        # Should not raise exceptions
        html_report = self.generator.generate_html_report(minimal_results)
        json_report = self.generator.generate_json_report(minimal_results)
        
        self.assertIsInstance(html_report, HTMLReport)
        self.assertIsInstance(json_report, JSONReport)
        
        # Verify minimal content is present
        self.assertIn("minimal_test", html_report.content)
        self.assertEqual(json_report.data["test_results"]["session_id"], "minimal_test")
    
    def test_benchmark_comparison_charts(self):
        """Test benchmark comparison chart generation"""
        report = self.generator.generate_html_report(self.sample_results)
        
        # Should have multiple charts including benchmark comparison
        self.assertGreaterEqual(len(report.charts), 3)
        
        # Find benchmark comparison chart
        benchmark_chart = None
        for chart in report.charts:
            if "Benchmark Performance" in chart.title:
                benchmark_chart = chart
                break
        
        self.assertIsNotNone(benchmark_chart)
        self.assertEqual(benchmark_chart.chart_type, "bar")
        self.assertIn("720p Generation", benchmark_chart.labels)
        self.assertIn("1080p Generation", benchmark_chart.labels)
        
        # Should have actual and target datasets
        self.assertEqual(len(benchmark_chart.datasets), 2)
        self.assertEqual(benchmark_chart.datasets[0]["label"], "Actual Time (minutes)")
        self.assertEqual(benchmark_chart.datasets[1]["label"], "Target Time (minutes)")
    
    def test_failure_analysis_generation(self):
        """Test failure analysis section generation"""
        report = self.generator.generate_html_report(self.sample_results)
        
        # Should contain failure analysis
        self.assertIn("Failure Analysis", report.content)
        self.assertIn("ENV_CONFIGURATION_FAILED", report.content)
        self.assertIn("PERF_1080P_TIMEOUT", report.content)
        self.assertIn("PERF_VRAM_OPTIMIZATION_INSUFFICIENT", report.content)
        
        # Should contain remediation steps
        self.assertIn("Enable attention slicing", report.content)
        self.assertIn("Add HF_TOKEN to .env file", report.content)
    
    def test_json_report_with_performance_data(self):
        """Test JSON report includes performance data"""
        report = self.generator.generate_json_report(self.sample_results)
        
        # Verify performance results are included
        test_results = report.data["test_results"]
        self.assertIn("performance_results", test_results)
        
        perf_results = test_results["performance_results"]
        self.assertIn("benchmark_720p", perf_results)
        self.assertIn("benchmark_1080p", perf_results)
        self.assertIn("vram_optimization", perf_results)
        
        # Verify benchmark data
        benchmark_720p = perf_results["benchmark_720p"]
        self.assertEqual(benchmark_720p["resolution"], "720p")
        self.assertEqual(benchmark_720p["generation_time"], 7.2)
        self.assertTrue(benchmark_720p["meets_target"])
        
        benchmark_1080p = perf_results["benchmark_1080p"]
        self.assertEqual(benchmark_1080p["resolution"], "1080p")
        self.assertEqual(benchmark_1080p["generation_time"], 18.5)
        self.assertFalse(benchmark_1080p["meets_target"])
        
        # Verify VRAM optimization data
        vram_opt = perf_results["vram_optimization"]
        self.assertEqual(vram_opt["reduction_percent"], 33.3)
        self.assertFalse(vram_opt["meets_target"])
        
        # Verify summary includes performance data
        summary = report.data["summary"]
        self.assertIn("performance_testing", summary)
        perf_summary = summary["performance_testing"]
        self.assertTrue(perf_summary["benchmark_720p_passed"])
        self.assertFalse(perf_summary["benchmark_1080p_passed"])
        self.assertFalse(perf_summary["vram_optimization_passed"])
    
    def test_pdf_export_fallback(self):
        """Test PDF export with fallback when WeasyPrint is not available"""
        html_report = self.generator.generate_html_report(self.sample_results)
        pdf_report = self.generator.export_to_pdf(html_report)
        
        # Verify PDF report structure
        self.assertIsInstance(pdf_report, PDFReport)
        self.assertTrue(os.path.exists(pdf_report.file_path))
        
        # Should create a text file as fallback (since WeasyPrint likely not installed)
        self.assertTrue(pdf_report.file_path.endswith('.txt'))
        
        # Verify content
        with open(pdf_report.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("LOCAL TESTING FRAMEWORK REPORT", content)
            self.assertIn("test_session_123", content)
    
    def test_troubleshooting_guide_generation(self):
        """Test troubleshooting guide generation"""
        guide = self.generator.generate_troubleshooting_guide(self.sample_results)
        
        # Verify guide structure
        self.assertIsInstance(guide, TroubleshootingGuide)
        self.assertGreater(guide.issues_count, 0)
        self.assertIsInstance(guide.content, str)
        self.assertGreater(len(guide.content), 0)
        
        # Verify content includes expected sections
        self.assertIn("# Troubleshooting Guide", guide.content)
        self.assertIn("test_session_123", guide.content)
        self.assertIn("Issues and Solutions", guide.content)
        self.assertIn("General Troubleshooting Steps", guide.content)
        
        # Should include specific issues from our test data
        self.assertIn("configuration", guide.content.lower())
        self.assertIn("1080p Generation Timeout", guide.content)
        self.assertIn("VRAM Reduction", guide.content)
        
        # Should include remediation steps
        self.assertIn("Enable attention slicing", guide.content)
        self.assertIn("Add HF_TOKEN to .env file", guide.content)
    
    def test_troubleshooting_guide_no_issues(self):
        """Test troubleshooting guide generation when no issues exist"""
        # Create results with no failures
        clean_env_results = EnvironmentValidationResults(
            python_version=ValidationResult("python_version", ValidationStatus.PASSED, "Python 3.9.7"),
            cuda_availability=ValidationResult("cuda_availability", ValidationStatus.PASSED, "CUDA available"),
            dependencies=ValidationResult("dependencies", ValidationStatus.PASSED, "All dependencies OK"),
            configuration=ValidationResult("configuration", ValidationStatus.PASSED, "Configuration valid"),
            environment_variables=ValidationResult("environment_variables", ValidationStatus.PASSED, "Variables OK"),
            overall_status=ValidationStatus.PASSED
        )
        
        clean_results = TestResults(
            session_id="clean_test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            environment_results=clean_env_results,
            overall_status=TestStatus.PASSED
        )
        
        guide = self.generator.generate_troubleshooting_guide(clean_results)
        
        # Should indicate no issues
        self.assertEqual(guide.issues_count, 0)
        self.assertIn("All Tests Passed", guide.content)
        self.assertIn("No issues were detected", guide.content)
        self.assertIn("Maintenance Recommendations", guide.content)
    
    def test_multi_format_consistency(self):
        """Test that all report formats contain consistent information"""
        # Generate all formats
        html_report = self.generator.generate_html_report(self.sample_results)
        json_report = self.generator.generate_json_report(self.sample_results)
        pdf_report = self.generator.export_to_pdf(html_report)
        guide = self.generator.generate_troubleshooting_guide(self.sample_results)
        
        # All should reference the same session
        self.assertIn("test_session_123", html_report.content)
        self.assertEqual(json_report.data["test_results"]["session_id"], "test_session_123")
        self.assertIn("test_session_123", guide.content)
        
        # All should be created in the same output directory
        output_path = Path(self.temp_dir)
        html_files = list(output_path.glob("*.html"))
        json_files = list(output_path.glob("*.json"))
        pdf_files = list(output_path.glob("*.txt"))  # Fallback format
        guide_files = list(output_path.glob("*.md"))
        
        self.assertEqual(len(html_files), 1)
        self.assertEqual(len(json_files), 1)
        self.assertEqual(len(pdf_files), 1)
        self.assertEqual(len(guide_files), 1)


if __name__ == '__main__':
    unittest.main()