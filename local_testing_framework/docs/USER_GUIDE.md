# Local Testing Framework - User Guide

## Overview

The Local Testing Framework for Wan2.2 UI Variant provides comprehensive automated testing capabilities to validate performance optimizations, system functionality, and deployment readiness. This guide covers installation, basic usage, and common workflows.

## Quick Start

### Installation

1. Ensure you have Python 3.8+ installed
2. Install the framework dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify installation:
   ```bash
   python -m local_testing_framework --help
   ```

### Basic Usage

The framework provides several commands for different testing scenarios:

```bash
# Validate your environment
python -m local_testing_framework validate-env

# Run performance tests
python -m local_testing_framework test-performance

# Run integration tests
python -m local_testing_framework test-integration

# Run diagnostics
python -m local_testing_framework diagnose

# Generate sample data
python -m local_testing_framework generate-samples

# Run full test suite
python -m local_testing_framework run-all
```

## Command Reference

### Environment Validation

Validates system prerequisites, dependencies, and configuration files.

```bash
python -m local_testing_framework validate-env [OPTIONS]
```

**Options:**

- `--fix`: Attempt to automatically fix issues
- `--report`: Generate detailed validation report
- `--platform`: Specify platform (windows, linux, macos)

**Example Output:**

```
Environment Validation Report
=============================
✓ Python Version: 3.9.7 (Required: 3.8+)
✓ CUDA Availability: 11.8 (Compatible)
✓ Dependencies: 45/45 packages installed
✗ Configuration: Missing HF_TOKEN in .env
✓ Multi-GPU: 2 GPUs detected (RTX 3080, RTX 3090)

Remediation Steps:
1. Add HF_TOKEN to .env file: echo "HF_TOKEN=your_token" >> .env
2. Restart application to load new environment variables

Overall Status: NEEDS_ATTENTION (1 issue found)
```

### Performance Testing

Executes automated performance benchmarks and validates optimization targets.

```bash
python -m local_testing_framework test-performance [OPTIONS]
```

**Options:**

- `--resolution {720p,1080p,all}`: Test specific resolution (default: all)
- `--benchmark`: Run comprehensive benchmarks
- `--vram-test`: Test VRAM optimization specifically
- `--targets`: Validate against performance targets

**Example Output:**

```
Performance Benchmark Results
============================
720p Generation Test:
  ✓ Time: 7.2 minutes (Target: <9 minutes)
  ✓ VRAM Usage: 8.4GB (Target: <12GB)
  ✓ VRAM Reduction: 82% (Target: 80%)

1080p Generation Test:
  ✓ Time: 14.8 minutes (Target: <17 minutes)
  ✓ VRAM Usage: 10.1GB (Target: <12GB)
  ✓ VRAM Reduction: 79% (Target: 80%)

Optimization Recommendations:
- Enable attention slicing for better memory efficiency
- Consider VAE tiling for larger resolutions

Overall Status: PASSED (All targets met)
```

### Integration Testing

Runs comprehensive integration tests and validates system components.

```bash
python -m local_testing_framework test-integration [OPTIONS]
```

**Options:**

- `--ui`: Test UI functionality
- `--api`: Test API endpoints
- `--full`: Run complete integration test suite
- `--browser {chrome,firefox,edge}`: Specify browser for UI tests

**Example Output:**

```
Integration Test Results
=======================
Video Generation Workflow: ✓ PASSED
UI Functionality: ✓ PASSED
  - Browser Access: ✓ PASSED
  - Form Submission: ✓ PASSED
  - File Upload: ✓ PASSED
API Endpoints: ✓ PASSED
  - Health Check: ✓ PASSED
  - Authentication: ✓ PASSED
Error Handling: ✓ PASSED
Resource Monitoring: ✓ PASSED

Overall Status: PASSED (12/12 tests passed)
```

### Diagnostics

Provides automated troubleshooting and system analysis capabilities.

```bash
python -m local_testing_framework diagnose [OPTIONS]
```

**Options:**

- `--system`: Analyze system resources
- `--cuda`: Diagnose CUDA issues
- `--memory`: Analyze memory issues
- `--logs`: Analyze error logs
- `--auto-fix`: Attempt automatic recovery

**Example Output:**

```
Diagnostic Analysis Report
=========================
System Resources:
  CPU Usage: 45% (Normal)
  Memory Usage: 8.2GB / 32GB (Normal)
  GPU Memory: 6.1GB / 12GB (Normal)

CUDA Analysis:
  ✓ CUDA Available: Yes (11.8)
  ✓ GPU Count: 2
  ✗ Issue Found: GPU memory fragmentation detected

Memory Analysis:
  ✓ System Memory: Sufficient
  ✓ GPU Memory: Within limits
  ✓ Memory Leaks: None detected

Recommendations:
1. Clear GPU cache: torch.cuda.empty_cache()
2. Enable memory optimization in config.json
3. Consider restarting application

Overall Status: MINOR_ISSUES (1 issue found)
```

### Sample Generation

Generates and manages test data, configurations, and sample inputs.

```bash
python -m local_testing_framework generate-samples [OPTIONS]
```

**Options:**

- `--config`: Generate configuration templates
- `--data`: Generate test data samples
- `--all`: Generate all sample types
- `--overwrite`: Overwrite existing files

**Example Output:**

```
Sample Generation Results
========================
Configuration Files:
  ✓ config.json template created
  ✓ .env template created
  ✓ sample_input.json created

Test Data:
  ✓ 720p test prompts: 10 samples
  ✓ 1080p test prompts: 10 samples
  ✓ Edge case samples: 15 samples

Validation:
  ✓ All files validated successfully
  ✓ JSON schema validation passed

Overall Status: SUCCESS (35 files generated)
```

### Continuous Monitoring

Provides real-time monitoring capabilities for extended testing sessions.

```bash
python -m local_testing_framework monitor [OPTIONS]
```

**Options:**

- `--duration SECONDS`: Monitoring duration (default: 3600)
- `--alerts`: Enable threshold alerts
- `--interval SECONDS`: Monitoring interval (default: 5)
- `--output FILE`: Save monitoring data to file

**Example Output:**

```
Continuous Monitoring Active
===========================
Duration: 1 hour
Interval: 5 seconds
Alerts: Enabled

[14:30:15] CPU: 42% | Memory: 8.1GB | GPU: 65% | VRAM: 7.2GB
[14:30:20] CPU: 45% | Memory: 8.3GB | GPU: 68% | VRAM: 7.4GB
[14:30:25] CPU: 43% | Memory: 8.2GB | GPU: 66% | VRAM: 7.3GB

⚠️  ALERT: VRAM usage approaching threshold (90%)
   Current: 10.8GB / 12GB (90%)
   Recommendation: Enable attention slicing

Monitoring complete. Report saved to monitoring_report.json
```

### Full Test Suite

Runs the complete test suite with comprehensive reporting.

```bash
python -m local_testing_framework run-all [OPTIONS]
```

**Options:**

- `--report-format {html,json,pdf}`: Report format (default: html)
- `--output DIR`: Output directory for reports
- `--parallel`: Run tests in parallel
- `--skip-slow`: Skip slow tests

**Example Output:**

```
Full Test Suite Execution
=========================
Environment Validation: ✓ PASSED
Performance Testing: ✓ PASSED
Integration Testing: ✓ PASSED
Diagnostics: ✓ PASSED

Test Summary:
  Total Tests: 47
  Passed: 45
  Failed: 2
  Skipped: 0
  Duration: 23 minutes

Failed Tests:
  - UI Test: Browser compatibility (Firefox)
  - Performance: 1080p target missed by 30 seconds

Reports Generated:
  - HTML Report: reports/test_report_20240130_143022.html
  - JSON Report: reports/test_report_20240130_143022.json

Overall Status: MOSTLY_PASSED (95.7% success rate)
```

## Common Workflows

### Daily Development Testing

```bash
# Quick environment check
python -m local_testing_framework validate-env

# Run performance tests for current changes
python -m local_testing_framework test-performance --resolution 720p

# Generate test report
python -m local_testing_framework run-all --report-format html
```

### Pre-Deployment Validation

```bash
# Full environment validation with fixes
python -m local_testing_framework validate-env --fix --report

# Comprehensive performance testing
python -m local_testing_framework test-performance --benchmark --targets

# Complete integration testing
python -m local_testing_framework test-integration --full

# Production readiness check
python -m local_testing_framework run-all --report-format pdf
```

### Troubleshooting Workflow

```bash
# Run diagnostics
python -m local_testing_framework diagnose --system --cuda --memory

# Generate fresh samples if needed
python -m local_testing_framework generate-samples --all --overwrite

# Monitor system during issue reproduction
python -m local_testing_framework monitor --duration 1800 --alerts
```

## Configuration

### Framework Configuration

The framework uses `config.json` for configuration. Key sections:

```json
{
  "testing_framework": {
    "performance_targets": {
      "target_720p_time_minutes": 9.0,
      "target_1080p_time_minutes": 17.0,
      "max_vram_usage_gb": 12.0,
      "vram_warning_threshold": 0.9,
      "cpu_warning_threshold": 80.0
    },
    "environment_requirements": {
      "min_python_version": "3.8.0",
      "required_cuda_version": "11.0",
      "min_memory_gb": 16
    },
    "reporting_options": {
      "default_format": "html",
      "include_charts": true,
      "chart_types": ["line", "bar", "gauge"]
    }
  }
}
```

### Environment Variables

Required environment variables:

- `HF_TOKEN`: Hugging Face authentication token
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation configuration

## Best Practices

### Performance Testing

1. **Consistent Environment**: Run tests in a consistent environment
2. **Baseline Measurements**: Establish baseline performance metrics
3. **Regular Testing**: Run performance tests regularly during development
4. **Resource Monitoring**: Monitor system resources during tests

### Integration Testing

1. **Isolated Tests**: Ensure tests don't interfere with each other
2. **Clean State**: Start each test from a clean state
3. **Error Handling**: Test error conditions and recovery
4. **Cross-Platform**: Test on multiple platforms when possible

### Troubleshooting

1. **Systematic Approach**: Use diagnostics before manual investigation
2. **Log Analysis**: Check error logs for specific issues
3. **Resource Monitoring**: Monitor resources during problem reproduction
4. **Incremental Testing**: Test components individually before integration

## Troubleshooting Common Issues

### Environment Issues

**Issue**: Missing dependencies

```
Solution: pip install -r requirements.txt
```

**Issue**: CUDA not available

```
Solution: Install CUDA toolkit and PyTorch with CUDA support
```

**Issue**: Configuration file errors

```
Solution: python -m local_testing_framework generate-samples --config
```

### Performance Issues

**Issue**: Tests exceed time targets

```
Solution: Enable optimization features in config.json
```

**Issue**: High VRAM usage

```
Solution: Enable attention slicing and VAE tiling
```

### Integration Issues

**Issue**: UI tests fail

```
Solution: Ensure browser drivers are installed and updated
```

**Issue**: API tests fail

```
Solution: Verify application is running on correct port
```

## Support and Resources

- **Documentation**: See `docs/` directory for detailed documentation
- **Examples**: Check `examples/` directory for sample configurations
- **Troubleshooting**: Refer to `docs/TROUBLESHOOTING.md` for common issues
- **Developer Guide**: See `docs/DEVELOPER_GUIDE.md` for extending the framework

For additional support, run diagnostics and include the output when reporting issues:

```bash
python -m local_testing_framework diagnose --system --cuda --memory > diagnostic_report.txt
```
