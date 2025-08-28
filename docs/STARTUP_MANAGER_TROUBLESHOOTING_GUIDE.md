# WAN22 Startup Manager Troubleshooting Guide

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Error Messages](#error-messages)
4. [Platform-Specific Issues](#platform-specific-issues)
5. [Performance Issues](#performance-issues)
6. [Network and Port Issues](#network-and-port-issues)
7. [Environment Issues](#environment-issues)
8. [Advanced Troubleshooting](#advanced-troubleshooting)
9. [Getting Help](#getting-help)

## Quick Diagnostics

### First Steps

When encountering issues, start with these diagnostic commands:

```bash
# 1. Run comprehensive diagnostics
python scripts/startup_manager.py --diagnostics

# 2. Try verbose startup to see detailed information
start_both_servers.bat --verbose

# 3. Check system requirements
python scripts/startup_manager.py --check-requirements

# 4. Validate configuration
python scripts/startup_manager.py --validate-config
```

### Emergency Fallback

If the startup manager is completely broken:

```bash
# Force basic mode (bypasses startup manager)
start_both_servers.bat --basic

# Or start servers manually
cd backend && python start_server.py
cd frontend && npm run dev
```

### Log File Locations

Check these log files for detailed information:

```
logs/startup_*.log          # Startup session logs
logs/errors.log             # Error logs
logs/performance.log        # Performance logs
logs/diagnostics.log        # Diagnostic information
```

## Common Issues

### Issue 1: "Python not found"

**Symptoms:**

```
Python startup manager not available - using basic mode
[WARN] Python not found in PATH
```

**Causes:**

- Python not installed
- Python not in system PATH
- Wrong Python version

**Solutions:**

1. **Install Python:**

   ```bash
   # Download from python.org (3.8+ required)
   # During installation, check "Add Python to PATH"
   ```

2. **Add Python to PATH:**

   ```bash
   # Windows: System Properties -> Environment Variables -> PATH
   # Add Python installation directory
   ```

3. **Verify installation:**
   ```bash
   python --version
   # Should show Python 3.8 or higher
   ```

### Issue 2: "Port already in use"

**Symptoms:**

```
Backend port 8000 is in use
Port conflict detected
```

**Automatic Resolution:**
The startup manager automatically finds alternative ports.

**Manual Solutions:**

1. **Use different ports:**

   ```bash
   start_both_servers.bat --backend-port 8080 --frontend-port 3001
   ```

2. **Kill process using port:**

   ```bash
   # Find process using port
   netstat -ano | findstr :8000

   # Kill process (replace PID with actual process ID)
   taskkill /PID 1234 /F
   ```

3. **Configure automatic port resolution:**
   ```json
   {
     "backend": { "auto_port": true },
     "frontend": { "auto_port": true }
   }
   ```

### Issue 3: "Permission denied" / "WinError 10013"

**Symptoms:**

```
WinError 10013: An attempt was made to access a socket in a way forbidden
Permission denied error detected
```

**Causes:**

- Windows Firewall blocking connections
- Insufficient privileges
- Antivirus software interference

**Solutions:**

1. **Run as Administrator:**

   ```bash
   # Right-click Command Prompt -> "Run as administrator"
   start_both_servers.bat
   ```

2. **Add Firewall Exceptions:**

   ```bash
   # Automatic (requires admin rights)
   start_both_servers.bat

   # Manual
   # Windows Defender Firewall -> Allow an app -> Add Python and Node.js
   ```

3. **Use alternative ports:**

   ```bash
   start_both_servers.bat --backend-port 8080 --frontend-port 3001
   ```

4. **Disable antivirus temporarily** (for testing)

### Issue 4: "Module not found" / Import Errors

**Symptoms:**

```
ModuleNotFoundError: No module named 'fastapi'
Import errors detected
```

**Causes:**

- Missing Python dependencies
- Wrong virtual environment
- Corrupted installation

**Solutions:**

1. **Install dependencies:**

   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Check virtual environment:**

   ```bash
   # Activate virtual environment if using one
   venv\Scripts\activate

   # Then install dependencies
   pip install -r backend/requirements.txt
   ```

3. **Reinstall dependencies:**
   ```bash
   pip uninstall -r backend/requirements.txt -y
   pip install -r backend/requirements.txt
   ```

### Issue 5: "Node.js not found" / Frontend Issues

**Symptoms:**

```
Node.js not found in PATH
Frontend startup failed
npm command not recognized
```

**Causes:**

- Node.js not installed
- Node.js not in PATH
- Wrong Node.js version

**Solutions:**

1. **Install Node.js:**

   ```bash
   # Download from nodejs.org (16+ required)
   # Use LTS version for stability
   ```

2. **Verify installation:**

   ```bash
   node --version
   npm --version
   ```

3. **Install frontend dependencies:**

   ```bash
   cd frontend
   npm install
   ```

4. **Clear npm cache if issues persist:**
   ```bash
   npm cache clean --force
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

## Error Messages

### Startup Manager Errors

#### "Startup manager dependencies not available"

**Solution:**

```bash
# Install startup manager dependencies
pip install rich click psutil requests

# Or install all requirements
pip install -r requirements.txt
```

#### "Configuration validation failed"

**Solution:**

```bash
# Check configuration syntax
python scripts/startup_manager.py --validate-config

# Use default configuration
del startup_config.json
start_both_servers.bat
```

#### "Environment validation failed"

**Solution:**

```bash
# Run detailed environment check
python scripts/startup_manager.py --check-environment

# Fix issues automatically
start_both_servers.bat --auto-fix
```

### Process Startup Errors

#### "Backend server failed to start"

**Diagnostic Steps:**

```bash
# Check backend logs
cd backend
python start_server.py

# Check for port conflicts
netstat -ano | findstr :8000

# Try different port
python start_server.py --port 8080
```

#### "Frontend server failed to start"

**Diagnostic Steps:**

```bash
# Check frontend logs
cd frontend
npm run dev

# Check for port conflicts
netstat -ano | findstr :3000

# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules
npm install
```

### Health Check Errors

#### "Backend health check failed"

**Solutions:**

```bash
# Check if backend is actually running
curl http://localhost:8000/health

# Check backend logs for errors
type logs\backend.log

# Restart backend with debug logging
cd backend
python start_server.py --log-level debug
```

#### "Frontend health check failed"

**Solutions:**

```bash
# Check if frontend is accessible
curl http://localhost:3000

# Check frontend build
cd frontend
npm run build

# Check for JavaScript errors in browser console
```

## Platform-Specific Issues

### Windows Issues

#### Windows Defender / Antivirus

**Symptoms:**

- Slow startup
- Random connection failures
- Permission errors

**Solutions:**

```bash
# Add exclusions to Windows Defender
# Settings -> Update & Security -> Windows Security -> Virus & threat protection
# Add exclusions for:
# - Project directory
# - Python installation
# - Node.js installation
```

#### Long Path Names

**Symptoms:**

```
FileNotFoundError: [Errno 2] No such file or directory
Path too long errors
```

**Solutions:**

```bash
# Enable long path support (requires admin)
# Group Policy: Computer Configuration -> Administrative Templates -> System -> Filesystem
# Enable "Enable Win32 long paths"

# Or use shorter project path
# Move project to C:\wan22 instead of deep nested folders
```

#### PowerShell Execution Policy

**Symptoms:**

```
Execution of scripts is disabled on this system
```

**Solutions:**

```bash
# Set execution policy (run PowerShell as admin)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use Command Prompt instead of PowerShell
```

### WSL (Windows Subsystem for Linux) Issues

#### Network Connectivity

**Symptoms:**

- Can't access servers from Windows browser
- Port forwarding issues

**Solutions:**

```bash
# Use 0.0.0.0 instead of localhost
start_both_servers.bat --backend-host 0.0.0.0 --frontend-host 0.0.0.0

# Check WSL network configuration
ip addr show eth0

# Access via WSL IP address
# Find WSL IP: ip route show default
```

#### File System Performance

**Symptoms:**

- Very slow startup
- File watching issues

**Solutions:**

```bash
# Move project to WSL file system (/home/user/) instead of Windows (/mnt/c/)
# Use WSL 2 instead of WSL 1
# Enable file system caching
```

## Performance Issues

### Slow Startup

**Symptoms:**

- Startup takes more than 60 seconds
- Timeout errors

**Diagnostic Steps:**

```bash
# Run performance analysis
python scripts/startup_manager.py --performance-analysis

# Check system resources
python scripts/startup_manager.py --system-info

# Use fast startup mode
start_both_servers.bat --fast-mode
```

**Solutions:**

1. **Enable performance optimizations:**

   ```json
   {
     "performance_optimization": {
       "fast_mode": true,
       "parallel_startup": true,
       "cache_validations": true
     }
   }
   ```

2. **Reduce timeout values:**

   ```json
   {
     "backend": { "timeout": 15 },
     "frontend": { "timeout": 15 }
   }
   ```

3. **Skip non-critical checks:**
   ```json
   {
     "validation": {
       "minimal_checks": true,
       "skip_optional": true
     }
   }
   ```

### High Memory Usage

**Symptoms:**

- System becomes slow during startup
- Out of memory errors

**Solutions:**

1. **Enable low memory mode:**

   ```json
   {
     "resource_constraints": {
       "low_memory_mode": true,
       "minimal_logging": true
     }
   }
   ```

2. **Reduce worker processes:**

   ```json
   {
     "backend": { "workers": 1 }
   }
   ```

3. **Disable performance monitoring:**
   ```json
   {
     "performance_monitoring": { "enabled": false }
   }
   ```

### High CPU Usage

**Symptoms:**

- CPU usage stays high during startup
- System becomes unresponsive

**Solutions:**

1. **Lower process priority:**

   ```json
   {
     "windows_optimizations": {
       "process_priority": "below_normal"
     }
   }
   ```

2. **Reduce parallel operations:**
   ```json
   {
     "performance_optimization": {
       "parallel_startup": false
     }
   }
   ```

## Network and Port Issues

### Firewall Issues

**Symptoms:**

- Connection refused errors
- Timeout when accessing servers

**Solutions:**

1. **Automatic firewall configuration:**

   ```bash
   # Run as administrator
   start_both_servers.bat
   ```

2. **Manual firewall configuration:**

   ```bash
   # Windows Firewall -> Advanced Settings -> Inbound Rules -> New Rule
   # Allow TCP ports 8000 and 3000
   ```

3. **Temporary firewall disable (for testing):**
   ```bash
   # Windows Defender Firewall -> Turn Windows Defender Firewall on or off
   # Disable temporarily for testing
   ```

### Network Interface Issues

**Symptoms:**

- Servers start but not accessible
- Connection refused from other machines

**Solutions:**

1. **Bind to all interfaces:**

   ```bash
   start_both_servers.bat --backend-host 0.0.0.0 --frontend-host 0.0.0.0
   ```

2. **Check network configuration:**

   ```bash
   ipconfig /all
   netstat -an | findstr :8000
   ```

3. **Test connectivity:**

   ```bash
   # Local test
   curl http://localhost:8000

   # Network test
   curl http://[your-ip]:8000
   ```

### DNS Issues

**Symptoms:**

- localhost doesn't resolve
- Intermittent connection issues

**Solutions:**

1. **Use IP address instead:**

   ```bash
   start_both_servers.bat --backend-host 127.0.0.1 --frontend-host 127.0.0.1
   ```

2. **Check hosts file:**

   ```bash
   # Check C:\Windows\System32\drivers\etc\hosts
   # Should contain: 127.0.0.1 localhost
   ```

3. **Flush DNS cache:**
   ```bash
   ipconfig /flushdns
   ```

## Environment Issues

### Virtual Environment Issues

**Symptoms:**

- Wrong Python packages loaded
- Import errors despite installation

**Solutions:**

1. **Activate virtual environment:**

   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. **Recreate virtual environment:**

   ```bash
   # Remove old environment
   rmdir /s venv

   # Create new environment
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Check Python path:**
   ```bash
   python -c "import sys; print(sys.executable)"
   # Should point to virtual environment
   ```

### PATH Issues

**Symptoms:**

- Commands not found
- Wrong versions executed

**Solutions:**

1. **Check PATH:**

   ```bash
   echo %PATH%
   where python
   where node
   ```

2. **Fix PATH order:**

   ```bash
   # System Properties -> Environment Variables -> PATH
   # Move Python and Node.js to top of list
   ```

3. **Use full paths:**
   ```bash
   C:\Python39\python.exe scripts/startup_manager.py
   ```

### Configuration Issues

**Symptoms:**

- Settings not applied
- Unexpected behavior

**Solutions:**

1. **Validate configuration:**

   ```bash
   python scripts/startup_manager.py --validate-config
   ```

2. **Check configuration precedence:**

   ```bash
   # Order: Command line > Environment variables > Config file > Defaults
   ```

3. **Reset to defaults:**
   ```bash
   # Rename or delete startup_config.json
   ren startup_config.json startup_config.json.backup
   start_both_servers.bat
   ```

## Advanced Troubleshooting

### Debug Mode

**Enable comprehensive debugging:**

```bash
start_both_servers.bat --debug --verbose
```

**Debug configuration:**

```json
{
  "debug_mode": {
    "enabled": true,
    "trace_calls": true,
    "detailed_errors": true,
    "step_by_step": true
  },
  "logging": {
    "level": "debug",
    "include_traceback": true,
    "save_debug_logs": true
  }
}
```

### System Information Collection

**Collect comprehensive system info:**

```bash
python scripts/startup_manager.py --system-info --save-report
```

**Manual system checks:**

```bash
# System information
systeminfo

# Network configuration
ipconfig /all
netstat -an

# Process information
tasklist | findstr python
tasklist | findstr node

# Disk space
dir C:\ /-c

# Memory usage
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory
```

### Log Analysis

**Analyze startup logs:**

```bash
# View recent startup logs
type logs\startup_*.log | findstr ERROR
type logs\startup_*.log | findstr WARNING

# Search for specific errors
findstr /i "permission" logs\*.log
findstr /i "timeout" logs\*.log
findstr /i "port" logs\*.log
```

**Enable detailed logging:**

```json
{
  "logging": {
    "level": "debug",
    "detailed_timing": true,
    "include_system_calls": true,
    "trace_network_calls": true
  }
}
```

### Performance Profiling

**Profile startup performance:**

```bash
python scripts/startup_manager.py --profile --save-profile
```

**Analyze performance bottlenecks:**

```bash
# Check startup timing
python scripts/analyze_performance.py logs/startup_*.log

# Monitor resource usage
python scripts/monitor_resources.py --during-startup
```

### Network Debugging

**Test network connectivity:**

```bash
# Test backend connectivity
curl -v http://localhost:8000/health

# Test frontend connectivity
curl -v http://localhost:3000

# Test with different hosts
curl -v http://127.0.0.1:8000/health
curl -v http://0.0.0.0:8000/health
```

**Network troubleshooting tools:**

```bash
# Port scanning
nmap -p 8000,3000 localhost

# Network monitoring
netstat -an | findstr :8000
netstat -an | findstr :3000

# Process monitoring
netstat -ano | findstr :8000
```

## Getting Help

### Before Asking for Help

1. **Run diagnostics:**

   ```bash
   python scripts/startup_manager.py --diagnostics --save-report
   ```

2. **Collect logs:**

   ```bash
   # Zip all log files
   # Include: logs/*.log, startup_config.json, diagnostic report
   ```

3. **Try basic mode:**

   ```bash
   start_both_servers.bat --basic
   ```

4. **Document steps to reproduce:**
   - What command did you run?
   - What did you expect to happen?
   - What actually happened?
   - What error messages did you see?

### Information to Include

When reporting issues, include:

1. **System information:**

   - Operating system and version
   - Python version (`python --version`)
   - Node.js version (`node --version`)
   - Project directory path

2. **Error details:**

   - Complete error message
   - Stack trace (if available)
   - Log files

3. **Configuration:**

   - startup_config.json (if used)
   - Environment variables
   - Command line arguments

4. **Diagnostic report:**
   ```bash
   python scripts/startup_manager.py --diagnostics --save-report
   ```

### Support Channels

1. **Documentation:** Check all documentation files in `docs/`
2. **GitHub Issues:** Create detailed issue with reproduction steps
3. **Team Chat:** Contact team members with diagnostic information
4. **Stack Overflow:** Tag questions with relevant technologies

### Self-Help Resources

1. **Verbose mode:** `start_both_servers.bat --verbose`
2. **Debug mode:** `start_both_servers.bat --debug`
3. **Diagnostic mode:** `python scripts/startup_manager.py --diagnostics`
4. **Configuration validation:** `python scripts/startup_manager.py --validate-config`
5. **System check:** `python scripts/startup_manager.py --check-system`

Remember: Most issues can be resolved by running in verbose mode and carefully reading the error messages. The startup manager is designed to provide clear, actionable error messages and automatic recovery for common issues.
