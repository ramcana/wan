# Local Testing Framework - Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when using the Local Testing Framework. Issues are organized by category with step-by-step resolution instructions.

## Quick Diagnostic Commands

Before diving into specific issues, run these diagnostic commands to gather information:

```bash
# General system diagnostics
python -m local_testing_framework diagnose --system --cuda --memory

# Environment validation
python -m local_testing_framework validate-env --report

# Generate diagnostic report
python -m local_testing_framework diagnose --system --cuda --memory > diagnostic_report.txt
```

## Environment Issues

### Python Version Issues

**Issue**: "Python version 3.7.x is not supported"

```
Error: Python version 3.7.5 detected. Minimum required: 3.8.0
```

**Solution**:

1. Install Python 3.8 or higher:

   ```bash
   # Windows (using chocolatey)
   choco install python --version=3.9.7

   # macOS (using homebrew)
   brew install python@3.9

   # Linux (Ubuntu/Debian)
   sudo apt update
   sudo apt install python3.9 python3.9-pip
   ```

2. Update your PATH to use the new Python version
3. Recreate your virtual environment:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### CUDA Issues

**Issue**: "CUDA not available" or "torch.cuda.is_available() returns False"

```
Error: CUDA validation failed - CUDA not available
```

**Solutions**:

1. **Install CUDA Toolkit**:

   ```bash
   # Check current CUDA version
   nvidia-smi

   # Install CUDA toolkit (version 11.8 recommended)
   # Download from: https://developer.nvidia.com/cuda-downloads
   ```

2. **Install PyTorch with CUDA support**:

   ```bash
   # Uninstall CPU-only PyTorch
   pip uninstall torch torchvision torchaudio

   # Install CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

**Issue**: "CUDA out of memory" during testing

```
Error: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

1. **Enable memory optimizations in config.json**:

   ```json
   {
     "optimization": {
       "enable_attention_slicing": true,
       "enable_cpu_offload": true,
       "enable_vae_tiling": true,
       "vae_tile_size": 128
     }
   }
   ```

2. **Clear GPU cache before tests**:

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Reduce batch size or resolution**:
   ```bash
   python -m local_testing_framework test-performance --resolution 720p
   ```

### Dependency Issues

**Issue**: "ModuleNotFoundError" for required packages

```
Error: ModuleNotFoundError: No module named 'psutil'
```

**Solution**:

1. **Install missing dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **For specific missing packages**:

   ```bash
   pip install psutil>=5.8.0
   pip install requests>=2.25.0
   pip install selenium>=4.0.0
   ```

3. **Verify installation**:
   ```bash
   python -c "import psutil; print(psutil.__version__)"
   ```

**Issue**: "Package version conflicts"

```
Error: Package 'torch' requires version >=1.12.0, but 1.10.0 is installed
```

**Solution**:

1. **Update conflicting packages**:

   ```bash
   pip install --upgrade torch>=1.12.0
   ```

2. **Use pip-tools for dependency resolution**:
   ```bash
   pip install pip-tools
   pip-compile requirements.in
   pip-sync requirements.txt
   ```

### Configuration Issues

**Issue**: "Configuration file not found" or "Invalid configuration"

```
Error: Configuration file 'config.json' not found
```

**Solution**:

1. **Generate default configuration**:

   ```bash
   python -m local_testing_framework generate-samples --config
   ```

2. **Validate existing configuration**:

   ```bash
   python -m local_testing_framework validate-env --report
   ```

3. **Check configuration format**:
   ```json
   {
     "system": {
       "platform": "auto",
       "gpu_memory_fraction": 0.9
     },
     "directories": {
       "models": "./models",
       "outputs": "./outputs"
     },
     "optimization": {
       "enable_attention_slicing": true,
       "enable_cpu_offload": false
     }
   }
   ```

**Issue**: "Missing HF_TOKEN in environment"

```
Error: HF_TOKEN not found in environment variables
```

**Solution**:

1. **Create .env file**:

   ```bash
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

2. **Set environment variable directly**:

   ```bash
   # Windows
   setx HF_TOKEN "your_token_here"

   # Linux/macOS
   export HF_TOKEN="your_token_here"
   echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
   ```

3. **Verify token is set**:
   ```bash
   python -c "import os; print('HF_TOKEN:', os.getenv('HF_TOKEN', 'Not set'))"
   ```

## Performance Issues

### Slow Test Execution

**Issue**: Tests take significantly longer than expected

```
Warning: 720p test took 15 minutes (target: <9 minutes)
```

**Solutions**:

1. **Enable all optimizations**:

   ```json
   {
     "optimization": {
       "enable_attention_slicing": true,
       "enable_cpu_offload": true,
       "enable_vae_tiling": true,
       "enable_sequential_cpu_offload": true,
       "vae_tile_size": 128
     }
   }
   ```

2. **Check system resources**:

   ```bash
   python -m local_testing_framework monitor --duration 300 --alerts
   ```

3. **Verify GPU utilization**:

   ```bash
   nvidia-smi -l 1
   ```

4. **Clear system cache**:

   ```bash
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"

   # Clear system cache (Linux)
   sudo sync && sudo sysctl vm.drop_caches=3
   ```

### High Memory Usage

**Issue**: System runs out of memory during tests

```
Error: System memory usage exceeded 95%
```

**Solutions**:

1. **Enable memory optimizations**:

   ```json
   {
     "optimization": {
       "enable_cpu_offload": true,
       "enable_sequential_cpu_offload": true,
       "low_mem_mode": true
     }
   }
   ```

2. **Monitor memory usage**:

   ```bash
   python -m local_testing_framework monitor --duration 600 --alerts
   ```

3. **Reduce concurrent operations**:
   ```bash
   # Run tests sequentially instead of parallel
   python -m local_testing_framework run-all --no-parallel
   ```

### VRAM Issues

**Issue**: VRAM usage exceeds targets

```
Warning: VRAM usage 14.2GB exceeds target of 12GB
```

**Solutions**:

1. **Enable VRAM optimizations**:

   ```json
   {
     "optimization": {
       "enable_attention_slicing": true,
       "enable_vae_tiling": true,
       "vae_tile_size": 64,
       "attention_slice_size": 1
     }
   }
   ```

2. **Test VRAM optimization**:

   ```bash
   python -m local_testing_framework test-performance --vram-test
   ```

3. **Monitor VRAM usage**:
   ```bash
   nvidia-smi -l 1
   ```

## Integration Test Issues

### UI Test Failures

**Issue**: "WebDriver not found" or browser automation fails

```
Error: selenium.common.exceptions.WebDriverException: 'chromedriver' executable needs to be in PATH
```

**Solutions**:

1. **Install browser drivers**:

   ```bash
   # Install ChromeDriver
   # Windows (using chocolatey)
   choco install chromedriver

   # macOS (using homebrew)
   brew install chromedriver

   # Linux
   sudo apt install chromium-chromedriver
   ```

2. **Use webdriver-manager for automatic driver management**:

   ```bash
   pip install webdriver-manager
   ```

3. **Verify driver installation**:
   ```bash
   chromedriver --version
   ```

**Issue**: "Browser fails to connect to application"

```
Error: Failed to connect to http://localhost:7860
```

**Solutions**:

1. **Verify application is running**:

   ```bash
   # Start application in background
   python main.py --port 7860 &

   # Check if port is open
   netstat -an | grep 7860
   ```

2. **Check firewall settings**:

   ```bash
   # Windows
   netsh advfirewall firewall add rule name="Allow Port 7860" dir=in action=allow protocol=TCP localport=7860

   # Linux
   sudo ufw allow 7860
   ```

3. **Test connection manually**:
   ```bash
   curl http://localhost:7860/health
   ```

### API Test Failures

**Issue**: "API endpoint returns unexpected response"

```
Error: Expected status 200, got 500
```

**Solutions**:

1. **Check application logs**:

   ```bash
   tail -f wan22_errors.log
   ```

2. **Verify API endpoints manually**:

   ```bash
   # Test health endpoint
   curl -X GET http://localhost:7860/health

   # Test with verbose output
   curl -v http://localhost:7860/health
   ```

3. **Check authentication**:
   ```bash
   # Test with authentication
   curl -H "Authorization: Bearer your_token" http://localhost:7860/api/endpoint
   ```

### Workflow Test Failures

**Issue**: "End-to-end workflow fails at specific step"

```
Error: Video generation workflow failed at step 3: Model loading
```

**Solutions**:

1. **Run workflow steps individually**:

   ```bash
   python -m local_testing_framework test-integration --step-by-step
   ```

2. **Check model availability**:

   ```bash
   python -c "from transformers import AutoModel; print('Models accessible')"
   ```

3. **Verify file permissions**:
   ```bash
   # Check output directory permissions
   ls -la outputs/
   chmod 755 outputs/
   ```

## Diagnostic and Monitoring Issues

### Diagnostic Tool Issues

**Issue**: "Diagnostic tool fails to collect metrics"

```
Error: Failed to collect system metrics
```

**Solutions**:

1. **Install required monitoring packages**:

   ```bash
   pip install psutil>=5.8.0
   pip install GPUtil>=1.4.0
   ```

2. **Check permissions**:

   ```bash
   # Linux: May need elevated permissions for some metrics
   sudo python -m local_testing_framework diagnose --system
   ```

3. **Run basic diagnostics**:
   ```bash
   python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%')"
   ```

### Monitoring Issues

**Issue**: "Continuous monitoring stops unexpectedly"

```
Error: Monitoring session terminated after 5 minutes
```

**Solutions**:

1. **Check system resources**:

   ```bash
   # Monitor framework overhead
   python -m local_testing_framework monitor --duration 300 --framework-overhead
   ```

2. **Increase monitoring interval**:

   ```bash
   python -m local_testing_framework monitor --interval 10 --duration 3600
   ```

3. **Check for memory leaks**:
   ```bash
   python -m local_testing_framework diagnose --memory --detailed
   ```

## Report Generation Issues

### HTML Report Issues

**Issue**: "Charts not displaying in HTML reports"

```
Error: Chart.js failed to load or render
```

**Solutions**:

1. **Check internet connection** (for CDN resources):

   ```bash
   curl -I https://cdn.jsdelivr.net/npm/chart.js
   ```

2. **Use local Chart.js**:

   ```bash
   # Download Chart.js locally
   mkdir -p local_testing_framework/static/js
   wget https://cdn.jsdelivr.net/npm/chart.js -O local_testing_framework/static/js/chart.js
   ```

3. **Verify report generation**:
   ```bash
   python -m local_testing_framework run-all --report-format html --debug
   ```

### PDF Report Issues

**Issue**: "PDF generation fails"

```
Error: WeasyPrint failed to generate PDF
```

**Solutions**:

1. **Install PDF dependencies**:

   ```bash
   # Windows
   pip install weasyprint

   # Linux
   sudo apt install python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
   pip install weasyprint

   # macOS
   brew install pango
   pip install weasyprint
   ```

2. **Use alternative PDF generator**:

   ```bash
   pip install pdfkit
   # Also install wkhtmltopdf system package
   ```

3. **Generate HTML first, then convert**:
   ```bash
   python -m local_testing_framework run-all --report-format html
   # Then manually convert HTML to PDF
   ```

## Platform-Specific Issues

### Windows Issues

**Issue**: "PowerShell execution policy prevents script execution"

```
Error: Execution of scripts is disabled on this system
```

**Solution**:

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue**: "Long path names cause issues"

```
Error: Path too long for Windows filesystem
```

**Solution**:

1. **Enable long path support**:

   ```powershell
   # Run as Administrator
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

2. **Use shorter paths**:
   ```bash
   # Move project to shorter path
   C:\ltf\  # instead of C:\Users\Username\Documents\Projects\local_testing_framework\
   ```

### Linux Issues

**Issue**: "Permission denied for GPU access"

```
Error: Permission denied: /dev/nvidia0
```

**Solution**:

```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login again
```

**Issue**: "Display not available for UI tests"

```
Error: selenium.common.exceptions.WebDriverException: unknown error: no display
```

**Solution**:

```bash
# Install and use Xvfb for headless display
sudo apt install xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

### macOS Issues

**Issue**: "Gatekeeper prevents execution"

```
Error: "python" cannot be opened because the developer cannot be verified
```

**Solution**:

```bash
# Allow execution
xattr -d com.apple.quarantine /path/to/python
# Or use System Preferences > Security & Privacy
```

**Issue**: "Metal Performance Shaders not available"

```
Warning: MPS backend not available
```

**Solution**:

```bash
# Install MPS-enabled PyTorch
pip install torch torchvision torchaudio
# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Advanced Troubleshooting

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment variable
export LTF_DEBUG=1

# Run with debug output
python -m local_testing_framework run-all --debug --verbose

# Save debug output
python -m local_testing_framework run-all --debug 2>&1 | tee debug.log
```

### Log Analysis

Analyze framework logs:

```bash
# View recent errors
tail -n 100 wan22_errors.log

# Search for specific errors
grep -i "cuda" wan22_errors.log
grep -i "memory" wan22_errors.log

# Analyze log patterns
python -m local_testing_framework diagnose --logs --analyze
```

### Performance Profiling

Profile framework performance:

```bash
# Profile test execution
python -m cProfile -o profile.stats -m local_testing_framework run-all

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
pip install memory-profiler
python -m memory_profiler -m local_testing_framework test-performance
```

### Network Issues

Debug network-related problems:

```bash
# Test connectivity
ping google.com
curl -I https://huggingface.co

# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Test with proxy bypass
export NO_PROXY="localhost,127.0.0.1"
```

## Getting Help

### Collecting Diagnostic Information

When reporting issues, include this diagnostic information:

```bash
# System information
python -m local_testing_framework diagnose --system --detailed > system_info.txt

# Environment validation
python -m local_testing_framework validate-env --report > env_validation.txt

# Recent logs
tail -n 200 wan22_errors.log > recent_errors.txt

# Package versions
pip list > package_versions.txt
```

### Creating Minimal Reproduction

Create a minimal example that reproduces the issue:

```python
# minimal_repro.py
from local_testing_framework.environment_validator import EnvironmentValidator

def reproduce_issue():
    validator = EnvironmentValidator()
    try:
        result = validator.validate_python_version()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    reproduce_issue()
```

### Support Channels

- **Documentation**: Check `docs/` directory for detailed guides
- **Issue Tracker**: Report bugs and request features
- **Discussion Forum**: Community support and questions
- **Stack Overflow**: Tag questions with `local-testing-framework`

### Emergency Recovery

If the framework is completely broken:

```bash
# Reset to clean state
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Reset configuration
rm -f config.json .env
python -m local_testing_framework generate-samples --config --overwrite

# Verify basic functionality
python -m local_testing_framework validate-env
```

## Prevention Tips

### Regular Maintenance

```bash
# Weekly maintenance
python -m local_testing_framework validate-env --fix
python -m local_testing_framework diagnose --system --cleanup

# Update dependencies monthly
pip list --outdated
pip install --upgrade -r requirements.txt

# Clean temporary files
python -m local_testing_framework cleanup --temp-files --cache
```

### Monitoring Setup

```bash
# Set up automated monitoring
crontab -e
# Add: 0 */6 * * * /path/to/python -m local_testing_framework monitor --duration 300 --alerts --log
```

### Backup Important Files

```bash
# Backup configuration
cp config.json config.json.backup
cp .env .env.backup

# Backup custom modifications
tar -czf custom_modifications.tar.gz local_testing_framework/custom/
```

This troubleshooting guide covers the most common issues. For additional help, refer to the user guide and developer documentation, or contact support with the diagnostic information collected above.
