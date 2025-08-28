# WAN22 Startup Manager User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Common Scenarios](#common-scenarios)
7. [Performance Optimization](#performance-optimization)
8. [FAQ](#faq)

## Quick Start

### First Time Setup

1. **Ensure Prerequisites**

   ```bash
   # Check Python installation
   python --version  # Should be 3.8+

   # Check Node.js installation
   node --version    # Should be 16+
   ```

2. **Start Servers**

   ```bash
   # Navigate to project directory
   cd wan2.2

   # Start both servers with intelligent management
   start_both_servers.bat
   ```

3. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### What Happens During Startup

The startup manager automatically:

- ✅ Validates your development environment
- ✅ Checks for port conflicts and resolves them
- ✅ Starts backend and frontend servers
- ✅ Monitors server health
- ✅ Provides clear status updates and error messages

## Basic Usage

### Standard Startup

```bash
# Basic startup (recommended for most users)
start_both_servers.bat
```

**What you'll see:**

```
========================================
   WAN22 Server Startup Manager v2.0
   Intelligent Server Management
========================================

Using intelligent startup manager...

Phase 1: Environment Validation
✓ Environment validation passed

Phase 2: Port Management
✓ Backend port 8000 is available
✓ Frontend port 3000 is available

Phase 3: Server Startup
✓ Backend server started successfully
✓ Frontend server started successfully

Phase 4: Health Verification
✓ Backend health check passed
✓ Frontend health check passed

✓ Startup Complete!

Backend Server:  http://localhost:8000
Frontend Server: http://localhost:3000
API Documentation: http://localhost:8000/docs
```

### Verbose Mode

```bash
# Get detailed information about the startup process
start_both_servers.bat --verbose
```

**Additional information shown:**

- Environment details (Python version, Node.js version, dependencies)
- Port scanning details
- Process startup details
- Performance metrics
- Optimization suggestions

### Debug Mode

```bash
# Get comprehensive debugging information
start_both_servers.bat --debug
```

**Additional debugging information:**

- Detailed system information
- Step-by-step operation logs
- Error stack traces
- Resource usage monitoring

## Advanced Features

### Custom Port Configuration

```bash
# Specify custom ports
start_both_servers.bat --backend-port 8080 --frontend-port 3001

# The startup manager will:
# 1. Check if specified ports are available
# 2. Find alternatives if they're occupied
# 3. Update configuration accordingly
```

### Force Basic Mode

```bash
# Bypass startup manager and use basic startup
start_both_servers.bat --basic

# Useful when:
# - Troubleshooting startup manager issues
# - Working in constrained environments
# - Preferring manual control
```

### Diagnostic Mode

```bash
# Run comprehensive system diagnostics
python scripts/startup_manager.py --diagnostics
```

**Diagnostic report includes:**

- System information (OS, hardware, software versions)
- Environment validation results
- Port availability analysis
- Configuration validation
- Performance benchmarks
- Recommendations for optimization

### Configuration Validation

```bash
# Validate startup configuration
python scripts/startup_manager.py --validate-config
```

### Environment Testing

```bash
# Test environment setup
python scripts/startup_manager.py --test-environment
```

## Configuration

### Startup Configuration File

Create `startup_config.json` in the project root for custom settings:

```json
{
  "backend": {
    "host": "localhost",
    "port": 8000,
    "auto_port": true,
    "timeout": 30,
    "reload": true,
    "log_level": "info",
    "workers": 1
  },
  "frontend": {
    "host": "localhost",
    "port": 3000,
    "auto_port": true,
    "timeout": 30,
    "open_browser": true,
    "hot_reload": true
  },
  "retry_attempts": 3,
  "retry_delay": 2.0,
  "verbose_logging": false,
  "auto_fix_issues": true,
  "performance_monitoring": {
    "enabled": true,
    "collect_metrics": true,
    "analytics": true
  },
  "windows_optimizations": {
    "firewall_exceptions": true,
    "process_priority": "normal",
    "service_integration": false
  }
}
```

### Environment Variables

Override configuration with environment variables:

```bash
# Port configuration
set WAN22_BACKEND_PORT=8080
set WAN22_FRONTEND_PORT=3001

# Logging configuration
set WAN22_VERBOSE_LOGGING=true
set WAN22_LOG_LEVEL=debug

# Feature toggles
set WAN22_AUTO_FIX_ISSUES=false
set WAN22_PERFORMANCE_MONITORING=true
```

### Configuration Options Explained

| Option            | Description                                                 | Default | Example |
| ----------------- | ----------------------------------------------------------- | ------- | ------- |
| `auto_port`       | Automatically find alternative ports if default is occupied | `true`  | `false` |
| `timeout`         | Server startup timeout in seconds                           | `30`    | `60`    |
| `retry_attempts`  | Number of retry attempts for failed operations              | `3`     | `5`     |
| `verbose_logging` | Enable detailed logging by default                          | `false` | `true`  |
| `auto_fix_issues` | Automatically attempt to fix common issues                  | `true`  | `false` |

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Conflicts

**Problem**: "Port 8000 is already in use"

**Automatic Solution**: Startup manager finds alternative port

```
Backend port 8000 is in use
Using alternative backend port 8001
```

**Manual Solution**:

```bash
# Specify different port
start_both_servers.bat --backend-port 8080

# Or kill process using the port
netstat -ano | findstr :8000
taskkill /PID [process_id] /F
```

#### 2. Permission Errors

**Problem**: "WinError 10013: An attempt was made to access a socket in a way forbidden"

**Automatic Solution**: Startup manager suggests solutions

```
Permission denied error detected
Suggested solutions:
1. Run as administrator
2. Add firewall exception for Python/Node.js
3. Try different port range (8080-8090)
```

**Manual Solution**:

```bash
# Run as administrator
# Right-click Command Prompt -> "Run as administrator"
start_both_servers.bat

# Or use basic mode
start_both_servers.bat --basic
```

#### 3. Environment Issues

**Problem**: "Python not found" or "Node.js not found"

**Automatic Solution**: Startup manager provides installation guidance

```
Environment validation failed
Issues found:
- Python 3.8+ not found in PATH
- Node.js 16+ not found in PATH

Suggested fixes:
1. Install Python from python.org
2. Install Node.js from nodejs.org
3. Ensure both are added to PATH
```

**Manual Solution**:

```bash
# Check installations
python --version
node --version

# Add to PATH if installed but not found
# Windows: System Properties -> Environment Variables -> PATH
```

#### 4. Dependency Issues

**Problem**: Import errors or missing packages

**Automatic Solution**: Startup manager detects and suggests fixes

```
Dependency validation failed
Missing packages detected:
- fastapi
- uvicorn
- react dependencies

Suggested fixes:
1. pip install -r backend/requirements.txt
2. cd frontend && npm install
```

**Manual Solution**:

```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

### Diagnostic Tools

#### 1. Verbose Mode

```bash
start_both_servers.bat --verbose
```

Shows detailed information about each step.

#### 2. Debug Mode

```bash
start_both_servers.bat --debug
```

Shows comprehensive debugging information.

#### 3. Diagnostic Report

```bash
python scripts/startup_manager.py --diagnostics
```

Generates comprehensive system diagnostic report.

#### 4. Log Files

```bash
# Check startup logs
type logs\startup_*.log

# Check error logs
type logs\errors.log
```

### Getting Help

1. **Check logs**: Look at `logs/startup_*.log` for detailed information
2. **Run diagnostics**: Use `--diagnostics` mode for system analysis
3. **Use verbose mode**: Add `--verbose` to see detailed operation information
4. **Check documentation**: Review this guide and other documentation files
5. **Contact support**: Provide diagnostic report and log files

## Common Scenarios

### Scenario 1: Daily Development

**Goal**: Quick, reliable server startup for daily development work

**Solution**:

```bash
# Simple daily startup
start_both_servers.bat

# With verbose output for monitoring
start_both_servers.bat --verbose
```

**Benefits**:

- Automatic port conflict resolution
- Environment validation
- Health monitoring
- Performance tracking

### Scenario 2: First-Time Setup

**Goal**: Set up development environment on new machine

**Solution**:

```bash
# Run diagnostic to check system
python scripts/startup_manager.py --diagnostics

# Start with verbose mode to see setup details
start_both_servers.bat --verbose
```

**What happens**:

- System requirements validation
- Automatic dependency checking
- Configuration file creation
- Firewall exception setup (if admin rights available)

### Scenario 3: Troubleshooting Issues

**Goal**: Diagnose and fix startup problems

**Solution**:

```bash
# Step 1: Run diagnostics
python scripts/startup_manager.py --diagnostics

# Step 2: Try verbose startup
start_both_servers.bat --verbose

# Step 3: If issues persist, use debug mode
start_both_servers.bat --debug

# Step 4: Check logs
type logs\startup_*.log
```

### Scenario 4: Team Development

**Goal**: Consistent startup experience across team members

**Solution**:

1. **Standardize configuration**: Share `startup_config.json`
2. **Document team settings**: Create team-specific documentation
3. **Use environment variables**: Set team-wide defaults

```json
// Team startup_config.json
{
  "backend": { "port": 8000, "auto_port": true },
  "frontend": { "port": 3000, "auto_port": true },
  "verbose_logging": true,
  "auto_fix_issues": true
}
```

### Scenario 5: Testing Environment

**Goal**: Start servers for automated testing

**Solution**:

```bash
# Use different ports for testing
start_both_servers.bat --backend-port 8001 --frontend-port 3001

# Or use environment variables
set WAN22_BACKEND_PORT=8001
set WAN22_FRONTEND_PORT=3001
start_both_servers.bat
```

### Scenario 6: Production-like Environment

**Goal**: Test in production-like configuration

**Solution**:

```json
// production_startup_config.json
{
  "backend": {
    "port": 8000,
    "reload": false,
    "workers": 4,
    "log_level": "warning"
  },
  "frontend": {
    "port": 3000,
    "hot_reload": false,
    "open_browser": false
  },
  "performance_monitoring": true
}
```

## Performance Optimization

### Startup Performance Metrics

The startup manager tracks:

- **Startup time**: Total time from start to servers running
- **Success rate**: Percentage of successful startups
- **Error patterns**: Common failure modes
- **Resource usage**: CPU and memory usage during startup

### Viewing Performance Data

```bash
# Start with performance monitoring
start_both_servers.bat --verbose

# Check performance logs
type logs\performance.log

# Generate performance report
python scripts/startup_manager.py --performance-report
```

### Optimization Suggestions

The startup manager provides automatic optimization suggestions:

```
Performance Metrics:
• Success Rate: 95%
• Average Duration: 12.3s
• Trend: Improving

Optimization Suggestions:
1. Enable SSD optimization for faster file access
   Expected improvement: 2-3s faster startup
2. Increase process priority for development
   Expected improvement: 1-2s faster startup
3. Pre-warm dependency cache
   Expected improvement: 1s faster startup
```

### Manual Optimizations

#### 1. SSD Optimization

```json
// startup_config.json
{
  "windows_optimizations": {
    "ssd_optimization": true,
    "process_priority": "above_normal"
  }
}
```

#### 2. Dependency Caching

```bash
# Pre-install and cache dependencies
pip install -r backend/requirements.txt
cd frontend && npm install
```

#### 3. Firewall Optimization

```bash
# Run as administrator to set up firewall exceptions
start_both_servers.bat
```

#### 4. Resource Allocation

```json
// startup_config.json
{
  "backend": {
    "workers": 1, // Reduce for development
    "reload": true // Enable for development
  },
  "performance_monitoring": {
    "resource_limits": {
      "max_cpu_percent": 80,
      "max_memory_mb": 2048
    }
  }
}
```

## FAQ

### General Questions

**Q: Do I need to change my existing workflow?**
A: No, the startup manager is designed to work with existing workflows. Just use `start_both_servers.bat` as before.

**Q: What if the startup manager doesn't work?**
A: The batch file automatically falls back to basic mode. You can also force basic mode with `--basic`.

**Q: Can I still start servers manually?**
A: Yes, manual startup methods continue to work unchanged.

### Configuration Questions

**Q: Where should I put the startup_config.json file?**
A: In the project root directory (same level as start_both_servers.bat).

**Q: Can I use environment variables instead of config file?**
A: Yes, environment variables override config file settings. Use `WAN22_` prefix.

**Q: How do I disable specific features?**
A: Set the feature to `false` in startup_config.json or use environment variables.

### Troubleshooting Questions

**Q: How do I get more detailed error information?**
A: Use `--verbose` or `--debug` mode, and check log files in the `logs/` directory.

**Q: What if ports are always conflicting?**
A: Enable `auto_port` in configuration, or specify different default ports.

**Q: How do I report bugs or issues?**
A: Run diagnostic mode and include the report with your bug report.

### Performance Questions

**Q: Why is startup slower than before?**
A: Initial startup includes environment validation. Subsequent startups are faster due to caching.

**Q: How can I speed up startup?**
A: Follow optimization suggestions, enable caching, and use SSD optimizations.

**Q: Can I disable performance monitoring?**
A: Yes, set `performance_monitoring.enabled` to `false` in configuration.

### Advanced Questions

**Q: Can I extend the startup manager?**
A: Yes, see the Developer Guide for information on extending functionality.

**Q: How do I integrate with CI/CD?**
A: Use environment variables and the `--basic` mode for automated environments.

**Q: Can I use this in Docker?**
A: Yes, the startup manager works in containerized environments with appropriate configuration.

## Next Steps

1. **Try basic usage**: Start with simple `start_both_servers.bat`
2. **Explore verbose mode**: Use `--verbose` to see detailed information
3. **Customize configuration**: Create `startup_config.json` for your preferences
4. **Read other guides**: Check out the Integration Guide and Developer Guide
5. **Share feedback**: Help improve the startup manager with your experience

For more advanced usage and customization, see:

- **Developer Guide**: `docs/STARTUP_MANAGER_DEVELOPER_GUIDE.md`
- **Integration Guide**: `docs/STARTUP_MANAGER_INTEGRATION_GUIDE.md`
- **Migration Guide**: `docs/STARTUP_MANAGER_MIGRATION_GUIDE.md`
