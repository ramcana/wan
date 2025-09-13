"""
Report generation module for local testing framework
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .models.test_results import (
    TestResults, EnvironmentValidationResults, IntegrationTestResults,
    ResourceMetrics, ValidationStatus, TestStatus, PerformanceTestResults,
    BenchmarkResult, OptimizationResult
)


@dataclass
class ChartData:
    """Data structure for chart generation"""
    chart_type: str  # 'line', 'bar', 'gauge'
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Dict[str, Any]


@dataclass
class HTMLReport:
    """HTML report container"""
    title: str
    content: str
    charts: List[ChartData]
    timestamp: datetime


@dataclass
class JSONReport:
    """JSON report container"""
    data: Dict[str, Any]
    timestamp: datetime


@dataclass
class PDFReport:
    """PDF report container"""
    file_path: str
    timestamp: datetime


@dataclass
class TroubleshootingGuide:
    """Troubleshooting guide container"""
    content: str
    issues_count: int
    timestamp: datetime


class ReportGenerator:
    """Main report generation orchestrator"""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_html_report(self, results: TestResults) -> HTMLReport:
        """Generate comprehensive HTML report with charts
        
        Args:
            results: Test results to generate report from
            
        Returns:
            HTMLReport object with generated content
        """
        timestamp = datetime.now()
        
        # Generate charts for performance visualization
        charts = self._generate_performance_charts(results)
        
        # Generate HTML content
        html_content = self._generate_html_content(results, charts)
        
        report = HTMLReport(
            title=f"Test Report - {results.session_id}",
            content=html_content,
            charts=charts,
            timestamp=timestamp
        )
        
        # Save to file
        report_path = self.output_dir / f"test_report_{results.session_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report
    
    def generate_json_report(self, results: TestResults) -> JSONReport:
        """Generate JSON report for programmatic access
        
        Args:
            results: Test results to generate report from
            
        Returns:
            JSONReport object with generated data
        """
        timestamp = datetime.now()
        
        # Convert results to dictionary format
        report_data = {
            "report_metadata": {
                "generated_at": timestamp.isoformat(),
                "session_id": results.session_id,
                "report_version": "1.0"
            },
            "test_results": results.to_dict(),
            "summary": self._generate_summary(results)
        }
        
        report = JSONReport(
            data=report_data,
            timestamp=timestamp
        )
        
        # Save to file
        report_path = self.output_dir / f"test_report_{results.session_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        return report
    
    def export_to_pdf(self, html_report: HTMLReport) -> PDFReport:
        """Export HTML report to PDF using WeasyPrint
        
        Args:
            html_report: HTML report to convert to PDF
            
        Returns:
            PDFReport object with file path
        """
        try:
            # Try to import WeasyPrint
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            timestamp = datetime.now()
            pdf_filename = f"test_report_{html_report.title.split(' - ')[-1]}_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf_path = self.output_dir / pdf_filename
            
            # Create CSS for better PDF formatting
            pdf_css = CSS(string="""
                @page {
                    size: A4;
                    margin: 2cm;
                }
                body {
                    font-family: 'DejaVu Sans', sans-serif;
                    font-size: 10pt;
                    line-height: 1.4;
                }
                .chart-container {
                    page-break-inside: avoid;
                    margin: 10px 0;
                }
                .section {
                    page-break-inside: avoid;
                    margin-bottom: 20px;
                }
                .validation-item {
                    page-break-inside: avoid;
                }
                h1, h2 {
                    page-break-after: avoid;
                }
                canvas {
                    display: none; /* Hide charts in PDF as they won't render properly */
                }
            """)
            
            # Convert HTML to PDF
            font_config = FontConfiguration()
            html_doc = HTML(string=html_report.content)
            html_doc.write_pdf(str(pdf_path), stylesheets=[pdf_css], font_config=font_config)
            
            return PDFReport(
                file_path=str(pdf_path),
                timestamp=timestamp
            )
            
        except ImportError:
            # WeasyPrint not available, create a fallback text-based PDF
            return self._create_fallback_pdf(html_report)
        except Exception as e:
            # If PDF generation fails, create a fallback
            print(f"Warning: PDF generation failed: {e}")
            return self._create_fallback_pdf(html_report)
    
    def _create_fallback_pdf(self, html_report: HTMLReport) -> PDFReport:
        """Create a fallback text-based report when WeasyPrint is not available
        
        Args:
            html_report: HTML report to convert
            
        Returns:
            PDFReport object with text file path
        """
        timestamp = datetime.now()
        txt_filename = f"test_report_{html_report.title.split(' - ')[-1]}_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        txt_path = self.output_dir / txt_filename
        
        # Extract text content from HTML (simple approach)
        import re
        
        # Remove HTML tags and clean up content
        text_content = re.sub(r'<[^>]+>', '', html_report.content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        text_content = text_content.replace('&nbsp;', ' ')
        text_content = text_content.replace('&lt;', '<')
        text_content = text_content.replace('&gt;', '>')
        text_content = text_content.replace('&amp;', '&')
        
        # Format as readable text report
        formatted_content = f"""
LOCAL TESTING FRAMEWORK REPORT
{'=' * 50}

{text_content}

Note: This is a text-based fallback report. For full formatting and charts,
install WeasyPrint: pip install weasyprint

Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        return PDFReport(
            file_path=str(txt_path),
            timestamp=timestamp
        )
    
    def generate_troubleshooting_guide(self, results: TestResults) -> TroubleshootingGuide:
        """Generate troubleshooting guide from diagnostic data
        
        Args:
            results: Test results containing diagnostic information
            
        Returns:
            TroubleshootingGuide object with generated content
        """
        timestamp = datetime.now()
        issues = []
        
        # Collect issues from environment validation
        if results.environment_results:
            failed_validations = results.environment_results.get_failed_validations()
            for validation in failed_validations:
                issues.append({
                    "category": "Environment Setup",
                    "issue": validation.component,
                    "description": validation.message,
                    "symptoms": [f"Environment validation fails for {validation.component}"],
                    "solutions": validation.remediation_steps or ["Contact support for assistance"],
                    "severity": "High" if validation.status == ValidationStatus.FAILED else "Medium"
                })
        
        # Collect issues from performance testing
        if results.performance_results:
            failed_benchmarks = results.performance_results.get_failed_benchmarks()
            for benchmark in failed_benchmarks:
                issues.append({
                    "category": "Performance",
                    "issue": f"{benchmark.resolution} Generation Timeout",
                    "description": f"Video generation takes {benchmark.generation_time:.1f} minutes, exceeding target of {benchmark.target_time} minutes",
                    "symptoms": [
                        "Slow video generation",
                        f"High VRAM usage ({benchmark.vram_usage:.1f} GB)",
                        f"High CPU usage ({benchmark.cpu_usage:.1f}%)"
                    ],
                    "solutions": [
                        "Enable attention slicing in config.json",
                        "Reduce VAE tile size to 128 or lower",
                        "Enable CPU offload for models",
                        "Use lower resolution for testing",
                        "Close other GPU-intensive applications"
                    ],
                    "severity": "High"
                })
            
            # VRAM optimization issues
            if results.performance_results.vram_optimization and not results.performance_results.vram_optimization.meets_target:
                vram = results.performance_results.vram_optimization
                issues.append({
                    "category": "Memory Optimization",
                    "issue": "Insufficient VRAM Reduction",
                    "description": f"VRAM reduction of {vram.reduction_percent:.1f}% is below target of {vram.target_reduction_percent}%",
                    "symptoms": [
                        "High VRAM usage during generation",
                        "CUDA out of memory errors",
                        "System instability during long generations"
                    ],
                    "solutions": [
                        "Enable additional memory optimizations",
                        "Use smaller model variants (e.g., SD 1.5 instead of SDXL)",
                        "Increase attention slicing factor",
                        "Enable VAE tiling with smaller tile size",
                        "Enable model CPU offload",
                        "Reduce batch size to 1"
                    ],
                    "severity": "High"
                })
        
        # Generate troubleshooting guide content
        guide_content = self._format_troubleshooting_guide(issues, results)
        
        # Save to file
        guide_filename = f"troubleshooting_guide_{results.session_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        guide_path = self.output_dir / guide_filename
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        return TroubleshootingGuide(
            content=guide_content,
            issues_count=len(issues),
            timestamp=timestamp
        )
    
    def _format_troubleshooting_guide(self, issues: List[Dict[str, Any]], results: TestResults) -> str:
        """Format troubleshooting guide content
        
        Args:
            issues: List of issues to include in guide
            results: Test results for context
            
        Returns:
            Formatted troubleshooting guide content
        """
        if not issues:
            return f"""
# Troubleshooting Guide

**Session ID:** {results.session_id}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Status: All Tests Passed ✅

Congratulations! No issues were detected during testing. Your system is properly configured and meeting all performance targets.

### System Summary
- Environment validation: Passed
- Performance benchmarks: Passed
- All optimizations working correctly

### Maintenance Recommendations
- Regularly update dependencies
- Monitor system resources during production use
- Keep model cache clean to prevent disk space issues
            """.strip()
        
        # Sort issues by severity
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        guide_content = f"""
# Troubleshooting Guide

**Session ID:** {results.session_id}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Issues Found:** {len(issues)}

## Quick Summary

This guide contains solutions for {len(issues)} issue(s) identified during testing. Issues are ordered by severity (High → Medium → Low).

## Issues and Solutions

"""
        
        for i, issue in enumerate(issues, 1):
            severity_emoji = "🔴" if issue["severity"] == "High" else "🟡" if issue["severity"] == "Medium" else "🟢"
            
            guide_content += f"""
### {i}. {issue["issue"]} {severity_emoji}

**Category:** {issue["category"]}  
**Severity:** {issue["severity"]}

**Description:**  
{issue["description"]}

**Symptoms:**
"""
            for symptom in issue["symptoms"]:
                guide_content += f"- {symptom}\n"
            
            guide_content += f"""
**Solutions:**
"""
            for j, solution in enumerate(issue["solutions"], 1):
                guide_content += f"{j}. {solution}\n"
            
            guide_content += "\n---\n"
        
        # Add general troubleshooting section
        guide_content += f"""

## General Troubleshooting Steps

### Before You Start
1. Ensure you have the latest version of the application
2. Check that your system meets minimum requirements
3. Close unnecessary applications to free up resources

### Common Issues and Quick Fixes

#### CUDA Out of Memory
```bash
# Enable memory optimizations in config.json
"optimization": {{
    "enable_attention_slicing": true,
    "enable_vae_tiling": true,
    "vae_tile_size": 128,
    "enable_cpu_offload": true
}}
```

#### Slow Performance
1. Check GPU utilization: `nvidia-smi`
2. Enable optimizations in configuration
3. Use appropriate resolution for your hardware
4. Monitor system temperature

#### Environment Issues
1. Verify Python version: `python --version`
2. Check CUDA installation: `nvidia-smi`
3. Validate dependencies: `pip list`
4. Check environment variables in `.env` file

### Getting Help

If these solutions don't resolve your issues:

1. **Check Logs:** Review `wan22_errors.log` for detailed error messages
2. **System Info:** Run environment validation to get system details
3. **Community:** Search for similar issues in project documentation
4. **Support:** Contact support with your session ID: `{results.session_id}`

### Useful Commands

```bash
# Run environment validation
python -m local_testing_framework validate-env --report

# Test performance with diagnostics
python -m local_testing_framework test-performance --benchmark

# Generate diagnostic report
python -m local_testing_framework diagnose --system --cuda --memory

# Monitor system resources
python -m local_testing_framework monitor --duration 300
```

---

*This troubleshooting guide was automatically generated based on your test results. For the most up-to-date solutions, check the project documentation.*
        """.strip()
        
        return guide_content
    
    def _generate_performance_charts(self, results: TestResults) -> List[ChartData]:
        """Generate performance visualization charts
        
        Args:
            results: Test results containing performance data
            
        Returns:
            List of ChartData objects for visualization
        """
        charts = []
        
        # Environment validation status chart
        if results.environment_results:
            charts.append(self._create_environment_status_chart(results.environment_results))
        
        # Benchmark comparison chart
        if results.performance_results:
            charts.append(self._create_benchmark_comparison_chart(results.performance_results))
            
            # VRAM optimization chart
            if results.performance_results.vram_optimization:
                charts.append(self._create_vram_optimization_chart(results.performance_results.vram_optimization))
        
        # Add placeholder for resource metrics over time (would be populated with actual metrics)
        charts.append(self._create_resource_metrics_chart())
        
        return charts
    
    def _create_environment_status_chart(self, env_results: EnvironmentValidationResults) -> ChartData:
        """Create chart for environment validation status
        
        Args:
            env_results: Environment validation results
            
        Returns:
            ChartData for environment status visualization
        """
        # Count validation statuses
        status_counts = {
            "Passed": 0,
            "Failed": 0,
            "Warning": 0,
            "Skipped": 0
        }
        
        validations = [
            env_results.python_version,
            env_results.cuda_availability,
            env_results.dependencies,
            env_results.configuration,
            env_results.environment_variables
        ]
        
        for validation in validations:
            if validation.status == ValidationStatus.PASSED:
                status_counts["Passed"] += 1
            elif validation.status == ValidationStatus.FAILED:
                status_counts["Failed"] += 1
            elif validation.status == ValidationStatus.WARNING:
                status_counts["Warning"] += 1
            else:
                status_counts["Skipped"] += 1
        
        return ChartData(
            chart_type="bar",
            title="Environment Validation Status",
            labels=list(status_counts.keys()),
            datasets=[{
                "label": "Validation Count",
                "data": list(status_counts.values()),
                "backgroundColor": [
                    "rgba(75, 192, 192, 0.6)",  # Green for passed
                    "rgba(255, 99, 132, 0.6)",  # Red for failed
                    "rgba(255, 206, 86, 0.6)",  # Yellow for warning
                    "rgba(201, 203, 207, 0.6)"  # Gray for skipped
                ],
                "borderColor": [
                    "rgba(75, 192, 192, 1)",
                    "rgba(255, 99, 132, 1)",
                    "rgba(255, 206, 86, 1)",
                    "rgba(201, 203, 207, 1)"
                ],
                "borderWidth": 1
            }],
            options={
                "responsive": True,
                "plugins": {
                    "legend": {
                        "display": False
                    },
                    "title": {
                        "display": True,
                        "text": "Environment Validation Results"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "ticks": {
                            "stepSize": 1
                        }
                    }
                }
            }
        )
    
    def _create_benchmark_comparison_chart(self, perf_results: PerformanceTestResults) -> ChartData:
        """Create chart comparing benchmark results against targets
        
        Args:
            perf_results: Performance test results
            
        Returns:
            ChartData for benchmark comparison visualization
        """
        benchmarks = []
        actual_times = []
        target_times = []
        colors = []
        
        if perf_results.benchmark_720p:
            benchmarks.append("720p Generation")
            actual_times.append(perf_results.benchmark_720p.generation_time)
            target_times.append(perf_results.benchmark_720p.target_time)
            colors.append("rgba(75, 192, 192, 0.6)" if perf_results.benchmark_720p.meets_target else "rgba(255, 99, 132, 0.6)")
        
        if perf_results.benchmark_1080p:
            benchmarks.append("1080p Generation")
            actual_times.append(perf_results.benchmark_1080p.generation_time)
            target_times.append(perf_results.benchmark_1080p.target_time)
            colors.append("rgba(75, 192, 192, 0.6)" if perf_results.benchmark_1080p.meets_target else "rgba(255, 99, 132, 0.6)")
        
        return ChartData(
            chart_type="bar",
            title="Benchmark Performance vs Targets",
            labels=benchmarks,
            datasets=[
                {
                    "label": "Actual Time (minutes)",
                    "data": actual_times,
                    "backgroundColor": colors,
                    "borderColor": [color.replace("0.6", "1") for color in colors],
                    "borderWidth": 1
                },
                {
                    "label": "Target Time (minutes)",
                    "data": target_times,
                    "backgroundColor": "rgba(201, 203, 207, 0.3)",
                    "borderColor": "rgba(201, 203, 207, 1)",
                    "borderWidth": 1,
                    "type": "line"
                }
            ],
            options={
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Generation Time vs Performance Targets"
                    },
                    "legend": {
                        "display": True
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Time (minutes)"
                        }
                    }
                }
            }
        )
    
    def _create_vram_optimization_chart(self, vram_result: OptimizationResult) -> ChartData:
        """Create chart for VRAM optimization results
        
        Args:
            vram_result: VRAM optimization results
            
        Returns:
            ChartData for VRAM optimization visualization
        """
        return ChartData(
            chart_type="bar",
            title="VRAM Optimization Results",
            labels=["Baseline VRAM", "Optimized VRAM", "Target Reduction"],
            datasets=[{
                "label": "VRAM Usage (GB)",
                "data": [
                    vram_result.baseline_vram_mb / 1024,
                    vram_result.optimized_vram_mb / 1024,
                    (vram_result.baseline_vram_mb * (1 - vram_result.target_reduction_percent / 100)) / 1024
                ],
                "backgroundColor": [
                    "rgba(255, 99, 132, 0.6)",  # Red for baseline
                    "rgba(75, 192, 192, 0.6)" if vram_result.meets_target else "rgba(255, 206, 86, 0.6)",  # Green if meets target, yellow if not
                    "rgba(201, 203, 207, 0.6)"  # Gray for target
                ],
                "borderColor": [
                    "rgba(255, 99, 132, 1)",
                    "rgba(75, 192, 192, 1)" if vram_result.meets_target else "rgba(255, 206, 86, 1)",
                    "rgba(201, 203, 207, 1)"
                ],
                "borderWidth": 1
            }],
            options={
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"VRAM Reduction: {vram_result.reduction_percent:.1f}% (Target: {vram_result.target_reduction_percent}%)"
                    },
                    "legend": {
                        "display": False
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "VRAM Usage (GB)"
                        }
                    }
                }
            }
        )

    def _create_resource_metrics_chart(self) -> ChartData:
        """Create placeholder chart for resource metrics over time
        
        Returns:
            ChartData for resource metrics visualization
        """
        # This would be populated with actual metrics data in a real implementation
        sample_times = ["00:00", "00:05", "00:10", "00:15", "00:20"]
        
        return ChartData(
            chart_type="line",
            title="Resource Usage Over Time",
            labels=sample_times,
            datasets=[
                {
                    "label": "CPU %",
                    "data": [25, 30, 28, 35, 32],
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "Memory %",
                    "data": [45, 48, 52, 50, 55],
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "GPU %",
                    "data": [80, 85, 90, 88, 92],
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "VRAM %",
                    "data": [60, 65, 70, 68, 72],
                    "borderColor": "rgba(255, 206, 86, 1)",
                    "backgroundColor": "rgba(255, 206, 86, 0.2)",
                    "tension": 0.1
                }
            ],
            options={
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "System Resource Usage"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100,
                        "ticks": {
                            "callback": "function(value) { return value + '%'; }"
                        }
                    }
                }
            }
        )
    
    def _generate_html_content(self, results: TestResults, charts: List[ChartData]) -> str:
        """Generate complete HTML content for report
        
        Args:
            results: Test results to include in report
            charts: Chart data for visualization
            
        Returns:
            Complete HTML content string
        """
        # Generate chart scripts
        chart_scripts = self._generate_chart_scripts(charts)
        
        # Generate environment section
        env_section = self._generate_environment_section(results.environment_results)
        
        # Generate performance section
        perf_section = self._generate_performance_section(results.performance_results)
        
        # Generate failure analysis section
        failure_section = self._generate_failure_analysis_section(results)
        
        # Generate summary section
        summary_section = self._generate_summary_section(results)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {results.session_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .header .timestamp {{
            color: #666;
            font-size: 14px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #444;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .status-warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .status-error {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 4px;
        }}
        .validation-item {{
            margin: 10px 0;
            padding: 15px;
            border-left: 4px solid #ddd;
            background-color: #f9f9f9;
        }}
        .validation-item.passed {{
            border-left-color: #28a745;
        }}
        .validation-item.failed {{
            border-left-color: #dc3545;
        }}
        .validation-item.warning {{
            border-left-color: #ffc107;
        }}
        .validation-details {{
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }}
        .remediation-steps {{
            margin-top: 10px;
        }}
        .remediation-steps ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
        .remediation-steps li {{
            margin: 5px 0;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Local Testing Framework Report</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="timestamp">Session ID: {results.session_id}</div>
        </div>
        
        {summary_section}
        
        <div class="section">
            <h2>Performance Visualization</h2>
            <div class="chart-container">
                <canvas id="environmentChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="benchmarkChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="vramChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="resourceChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        {env_section}
        
        {perf_section}
        
        {failure_section}
    </div>
    
    <script>
        {chart_scripts}
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _generate_chart_scripts(self, charts: List[ChartData]) -> str:
        """Generate JavaScript code for Chart.js charts
        
        Args:
            charts: List of chart data to generate scripts for
            
        Returns:
            JavaScript code string for chart rendering
        """
        scripts = []
        
        # Map chart indices to canvas IDs
        canvas_ids = ["environmentChart", "benchmarkChart", "vramChart", "resourceChart"]
        
        for i, chart in enumerate(charts):
            canvas_id = canvas_ids[i] if i < len(canvas_ids) else f"chart_{i}"
            
            # Convert chart data to JavaScript
            chart_config = {
                "type": chart.chart_type,
                "data": {
                    "labels": chart.labels,
                    "datasets": chart.datasets
                },
                "options": chart.options
            }
            
            script = f"""
            const ctx_{i} = document.getElementById('{canvas_id}').getContext('2d');
            const chart_{i} = new Chart(ctx_{i}, {json.dumps(chart_config)});
            """
            scripts.append(script)
        
        return "\n".join(scripts)
    
    def _generate_environment_section(self, env_results: Optional[EnvironmentValidationResults]) -> str:
        """Generate HTML section for environment validation results
        
        Args:
            env_results: Environment validation results
            
        Returns:
            HTML content string for environment section
        """
        if not env_results:
            return '<div class="section"><h2>Environment Validation</h2><p>No environment validation results available.</p></div>'
        
        validations = [
            ("Python Version", env_results.python_version),
            ("CUDA Availability", env_results.cuda_availability),
            ("Dependencies", env_results.dependencies),
            ("Configuration", env_results.configuration),
            ("Environment Variables", env_results.environment_variables)
        ]
        
        validation_items = []
        for name, validation in validations:
            status_class = validation.status.value
            status_badge = f'<span class="status-badge status-{status_class}">{validation.status.value}</span>'
            
            remediation_html = ""
            if validation.remediation_steps:
                remediation_html = f"""
                <div class="remediation-steps">
                    <strong>Remediation Steps:</strong>
                    <ul>
                        {''.join(f'<li>{step}</li>' for step in validation.remediation_steps)}
                    </ul>
                </div>
                """
            
            validation_item = f"""
            <div class="validation-item {status_class}">
                <strong>{name}</strong> {status_badge}
                <div class="validation-details">{validation.message}</div>
                {remediation_html}
            </div>
            """
            validation_items.append(validation_item)
        
        overall_status_badge = f'<span class="status-badge status-{env_results.overall_status.value}">{env_results.overall_status.value}</span>'
        
        return f"""
        <div class="section">
            <h2>Environment Validation {overall_status_badge}</h2>
            {''.join(validation_items)}
        </div>
        """
    
    def _generate_performance_section(self, perf_results: Optional[PerformanceTestResults]) -> str:
        """Generate HTML section for performance test results
        
        Args:
            perf_results: Performance test results
            
        Returns:
            HTML content string for performance section
        """
        if not perf_results:
            return '<div class="section"><h2>Performance Testing</h2><p>No performance test results available.</p></div>'
        
        status_badge = f'<span class="status-badge status-{perf_results.overall_status.value}">{perf_results.overall_status.value}</span>'
        
        benchmark_items = []
        
        # 720p benchmark
        if perf_results.benchmark_720p:
            benchmark = perf_results.benchmark_720p
            status_class = "passed" if benchmark.meets_target else "failed"
            status_text = "PASSED" if benchmark.meets_target else "FAILED"
            
            benchmark_item = f"""
            <div class="validation-item {status_class}">
                <strong>720p Generation Benchmark</strong> <span class="status-badge status-{status_class}">{status_text}</span>
                <div class="validation-details">
                    Time: {benchmark.generation_time:.1f} minutes (Target: < {benchmark.target_time} minutes)<br>
                    VRAM Usage: {benchmark.vram_usage:.1f} GB<br>
                    CPU Usage: {benchmark.cpu_usage:.1f}%<br>
                    Memory Usage: {benchmark.memory_usage:.1f}%<br>
                    Optimization Level: {benchmark.optimization_level}
                </div>
            </div>
            """
            benchmark_items.append(benchmark_item)
        
        # 1080p benchmark
        if perf_results.benchmark_1080p:
            benchmark = perf_results.benchmark_1080p
            status_class = "passed" if benchmark.meets_target else "failed"
            status_text = "PASSED" if benchmark.meets_target else "FAILED"
            
            benchmark_item = f"""
            <div class="validation-item {status_class}">
                <strong>1080p Generation Benchmark</strong> <span class="status-badge status-{status_class}">{status_text}</span>
                <div class="validation-details">
                    Time: {benchmark.generation_time:.1f} minutes (Target: < {benchmark.target_time} minutes)<br>
                    VRAM Usage: {benchmark.vram_usage:.1f} GB<br>
                    CPU Usage: {benchmark.cpu_usage:.1f}%<br>
                    Memory Usage: {benchmark.memory_usage:.1f}%<br>
                    Optimization Level: {benchmark.optimization_level}
                </div>
            </div>
            """
            benchmark_items.append(benchmark_item)
        
        # VRAM optimization
        if perf_results.vram_optimization:
            vram = perf_results.vram_optimization
            status_class = "passed" if vram.meets_target else "failed"
            status_text = "PASSED" if vram.meets_target else "FAILED"
            
            optimizations_list = ", ".join(vram.optimizations_applied) if vram.optimizations_applied else "None"
            
            vram_item = f"""
            <div class="validation-item {status_class}">
                <strong>VRAM Optimization</strong> <span class="status-badge status-{status_class}">{status_text}</span>
                <div class="validation-details">
                    Baseline VRAM: {vram.baseline_vram_mb / 1024:.1f} GB<br>
                    Optimized VRAM: {vram.optimized_vram_mb / 1024:.1f} GB<br>
                    Reduction: {vram.reduction_percent:.1f}% (Target: {vram.target_reduction_percent}%)<br>
                    Optimizations Applied: {optimizations_list}
                </div>
            </div>
            """
            benchmark_items.append(vram_item)
        
        recommendations_html = ""
        if perf_results.recommendations:
            recommendations_html = f"""
            <div class="remediation-steps">
                <strong>Performance Recommendations:</strong>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in perf_results.recommendations)}
                </ul>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>Performance Testing {status_badge}</h2>
            {''.join(benchmark_items)}
            {recommendations_html}
        </div>
        """
    
    def _generate_failure_analysis_section(self, results: TestResults) -> str:
        """Generate HTML section for failure analysis with error codes and remediation
        
        Args:
            results: Test results to analyze for failures
            
        Returns:
            HTML content string for failure analysis section
        """
        failures = []
        
        # Collect environment failures
        if results.environment_results:
            env_failures = results.environment_results.get_failed_validations()
            for failure in env_failures:
                failures.append({
                    "category": "Environment",
                    "component": failure.component,
                    "error_code": f"ENV_{failure.component.upper()}_FAILED",
                    "message": failure.message,
                    "remediation_steps": failure.remediation_steps
                })
        
        # Collect performance failures
        if results.performance_results:
            perf_failures = results.performance_results.get_failed_benchmarks()
            for failure in perf_failures:
                failures.append({
                    "category": "Performance",
                    "component": f"{failure.resolution} Generation",
                    "error_code": f"PERF_{failure.resolution.upper()}_TIMEOUT",
                    "message": f"Generation time {failure.generation_time:.1f} minutes exceeds target of {failure.target_time} minutes",
                    "remediation_steps": [
                        "Enable attention slicing in configuration",
                        "Reduce VAE tile size",
                        "Enable CPU offload for models",
                        "Consider using lower resolution for testing"
                    ]
                })
            
            # VRAM optimization failure
            if results.performance_results.vram_optimization and not results.performance_results.vram_optimization.meets_target:
                vram = results.performance_results.vram_optimization
                failures.append({
                    "category": "Performance",
                    "component": "VRAM Optimization",
                    "error_code": "PERF_VRAM_OPTIMIZATION_INSUFFICIENT",
                    "message": f"VRAM reduction {vram.reduction_percent:.1f}% below target of {vram.target_reduction_percent}%",
                    "remediation_steps": [
                        "Enable additional memory optimizations",
                        "Use smaller model variants",
                        "Increase attention slicing factor",
                        "Enable VAE tiling with smaller tile size"
                    ]
                })
        
        if not failures:
            return '<div class="section"><h2>Failure Analysis</h2><p>No failures detected. All tests passed successfully!</p></div>'
        
        failure_items = []
        for failure in failures:
            remediation_html = ""
            if failure["remediation_steps"]:
                remediation_html = f"""
                <div class="remediation-steps">
                    <strong>Remediation Steps:</strong>
                    <ul>
                        {''.join(f'<li>{step}</li>' for step in failure["remediation_steps"])}
                    </ul>
                </div>
                """
            
            failure_item = f"""
            <div class="validation-item failed">
                <strong>[{failure["error_code"]}] {failure["category"]} - {failure["component"]}</strong>
                <div class="validation-details">{failure["message"]}</div>
                {remediation_html}
            </div>
            """
            failure_items.append(failure_item)
        
        return f"""
        <div class="section">
            <h2>Failure Analysis</h2>
            <p>Found {len(failures)} issue(s) that require attention:</p>
            {''.join(failure_items)}
        </div>
        """
    
    def _generate_summary_section(self, results: TestResults) -> str:
        """Generate HTML section for test summary
        
        Args:
            results: Test results to summarize
            
        Returns:
            HTML content string for summary section
        """
        status_badge = f'<span class="status-badge status-{results.overall_status.value}">{results.overall_status.value}</span>'
        
        duration = "N/A"
        if results.end_time and results.start_time:
            duration_seconds = (results.end_time - results.start_time).total_seconds()
            duration = f"{duration_seconds:.1f} seconds"
        
        recommendations_html = ""
        if results.recommendations:
            recommendations_html = f"""
            <div class="remediation-steps">
                <strong>Recommendations:</strong>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in results.recommendations)}
                </ul>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>Test Summary {status_badge}</h2>
            <p><strong>Start Time:</strong> {results.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Duration:</strong> {duration}</p>
            {recommendations_html}
        </div>
        """
    
    def _generate_summary(self, results: TestResults) -> Dict[str, Any]:
        """Generate summary data for JSON report
        
        Args:
            results: Test results to summarize
            
        Returns:
            Dictionary containing summary information
        """
        summary = {
            "overall_status": results.overall_status.value,
            "start_time": results.start_time.isoformat(),
            "recommendations_count": len(results.recommendations)
        }
        
        if results.end_time:
            summary["end_time"] = results.end_time.isoformat()
            summary["duration_seconds"] = (results.end_time - results.start_time).total_seconds()
        
        if results.environment_results:
            env_summary = {
                "overall_status": results.environment_results.overall_status.value,
                "failed_validations": len(results.environment_results.get_failed_validations()),
                "remediation_steps_count": len(results.environment_results.remediation_steps)
            }
            summary["environment_validation"] = env_summary
        
        if results.performance_results:
            perf_summary = {
                "overall_status": results.performance_results.overall_status.value,
                "failed_benchmarks": len(results.performance_results.get_failed_benchmarks()),
                "recommendations_count": len(results.performance_results.recommendations)
            }
            
            # Add benchmark details
            if results.performance_results.benchmark_720p:
                perf_summary["benchmark_720p_passed"] = results.performance_results.benchmark_720p.meets_target
            if results.performance_results.benchmark_1080p:
                perf_summary["benchmark_1080p_passed"] = results.performance_results.benchmark_1080p.meets_target
            if results.performance_results.vram_optimization:
                perf_summary["vram_optimization_passed"] = results.performance_results.vram_optimization.meets_target
                
            summary["performance_testing"] = perf_summary
        
        return summary
