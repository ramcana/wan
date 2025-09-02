# 🔧 WAN22 Comprehensive Troubleshooting Guide

## Quick Diagnostic Commands

Before diving into specific issues, run these diagnostic commands:

```bash
# System diagnostic (comprehensive)
python start.py --diagnostics

# Check Python environment
python --version
python -c "import sys; print(sys.executable)"

# Check Node.js environment
node --version
npm --version

# Check GPU status (NVIDIA)
nvidia-smi

# Check available ports
netstat -an | findstr ":8000\|:3000"
```

## Startup Issues

### Issue: "Python not found" or "python is not recognized"

**Symptoms:**

- Error when running `python start.py`
- Command prompt says "python is not recognized"

**Solutions:**

1. **Install Python:**

   ```bash
   # Download from python.org and install
   # Make sure to check "Add Python to PATH"
   ```

2. **Fix PATH issues:**

   ```bash
   # Windows: Add to System PATH
   # Control Panel > System > Advanced > Environment Variables
   # Add: C:\Python39\ and C:\Python39\Scripts\
   ```

3. **Use full path:**

   ```bash
   # If PATH not working, use full path
   C:\Python39\python.exe start.py
   ```

4. **Check installation:**
   ```bash
   # Verify Python is installed correctly
   python --version
   pip --version
   ```

### Issue: "Node.js not found" or npm errors

**Symptoms:**

- Frontend fails to start
- npm command not recognized
- Package installation failures

**Solutions:**

1. **Install Node.js:**

   ```bash
   # Download LTS version from nodejs.org
   # Verify installation
   node --version  # Should show v16.x, v18.x, or v20.x
   npm --version
   ```

2. **Clear npm cache:**

   ```bash
   npm cache clean --force
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Use different package manager:**
   ```bash
   # Try yarn instead of npm
   npm install -g yarn
   cd frontend
   yarn install
   ```

### Issue: Port conflicts (8000 or 3000 already in use)

**Symptoms:**

- "Port 8000 is already in use"
- "EADDRINUSE" errors
- Servers fail to start

**Automatic Solution:**

```bash
# The start.py script automatically finds available ports
python start.py  # Will use 8001, 3001, etc. if needed
```

**Manual Solutions:**

1. **Kill processes using ports:**

   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID [process_id] /F

   # Linux/Mac
   lsof -ti:8000 | xargs kill -9
   ```

2. **Use different ports:**

   ```bash
   # Backend on different port
   cd backend
   python -m uvicorn app:app --port 8080

   # Frontend on different port
   cd frontend
   npm run dev -- --port 3001
   ```

### Issue: Permission denied errors

**Symptoms:**

- "WinError 10013: An attempt was made to access a socket"
- "Permission denied" when starting servers
- Firewall blocking connections

**Solutions:**

1. **Run as administrator:**

   ```bash
   # Right-click Command Prompt > "Run as administrator"
   python start.py
   ```

2. **Add firewall exceptions:**

   ```bash
   # Windows Firewall exceptions for Python and Node.js
   # Control Panel > Windows Defender Firewall > Allow an app
   ```

3. **Use different port range:**
   ```bash
   # Try ports above 8080 (less restricted)
   python start.py --backend-port 8080 --frontend-port 3001
   ```

## Dependency Issues

### Issue: Python package installation failures

**Symptoms:**

- "pip install" commands fail
- Import errors for fastapi, uvicorn, etc.
- "No module named 'fastapi'" errors

**Solutions:**

1. **Upgrade pip:**

   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install with verbose output:**

   ```bash
   pip install -r backend/requirements.txt -v
   ```

3. **Use virtual environment:**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

4. **Install individual packages:**

   ```bash
   pip install fastapi uvicorn python-multipart
   ```

5. **Clear pip cache:**
   ```bash
   pip cache purge
   pip install -r backend/requirements.txt --no-cache-dir
   ```

### Issue: Node.js dependency installation failures

**Symptoms:**

- "npm install" fails
- "ERESOLVE unable to resolve dependency tree"
- Missing frontend dependencies

**Solutions:**

1. **Clear cache and reinstall:**

   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm cache clean --force
   npm install
   ```

2. **Use legacy peer deps:**

   ```bash
   npm install --legacy-peer-deps
   ```

3. **Force install:**

   ```bash
   npm install --force
   ```

4. **Check Node.js version:**
   ```bash
   # Use Node.js LTS version (16.x, 18.x, or 20.x)
   node --version
   ```

## Runtime Errors

### Issue: Backend server crashes or fails to start

**Symptoms:**

- FastAPI server exits immediately
- Import errors in backend
- Database connection errors

**Diagnostic Steps:**

1. **Check detailed error:**

   ```bash
   cd backend
   python -m uvicorn app:app --reload --log-level debug
   ```

2. **Test imports:**

   ```bash
   cd backend
   python -c "import app; print('Backend imports OK')"
   ```

3. **Check database:**
   ```bash
   # Verify database file exists and is accessible
   ls -la wan22_tasks.db
   ```

**Solutions:**

1. **Fix import paths:**

   ```bash
   # Run import update script
   python utils_new/update_imports.py
   ```

2. **Recreate database:**

   ```bash
   cd backend
   rm wan22_tasks.db
   python app.py  # Will recreate database
   ```

3. **Check configuration:**
   ```bash
   # Verify config files exist and are valid
   python -c "import json; print(json.load(open('backend/config.json')))"
   ```

### Issue: Frontend build or runtime errors

**Symptoms:**

- React development server crashes
- TypeScript compilation errors
- "Module not found" errors

**Diagnostic Steps:**

1. **Check build process:**

   ```bash
   cd frontend
   npm run build
   ```

2. **Check TypeScript:**
   ```bash
   cd frontend
   npx tsc --noEmit
   ```

**Solutions:**

1. **Clear build cache:**

   ```bash
   cd frontend
   rm -rf .next dist build
   npm run dev
   ```

2. **Fix TypeScript errors:**

   ```bash
   # Check for type errors
   cd frontend
   npx tsc --noEmit --skipLibCheck
   ```

3. **Update dependencies:**
   ```bash
   cd frontend
   npm update
   ```

## GPU and Performance Issues

### Issue: GPU not detected or not being used

**Symptoms:**

- Video generation is very slow
- CPU usage high, GPU usage low
- "CUDA not available" warnings

**Diagnostic Steps:**

1. **Check GPU status:**

   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check CUDA installation:**
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

**Solutions:**

1. **Install/Update NVIDIA drivers:**

   ```bash
   # Download latest drivers from nvidia.com
   # Restart after installation
   ```

2. **Install CUDA toolkit:**

   ```bash
   # Download CUDA 11.8+ from nvidia.com
   # Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```

3. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Out of memory errors (CUDA OOM)

**Symptoms:**

- "CUDA out of memory" errors
- Video generation fails partway through
- System becomes unresponsive

**Solutions:**

1. **Reduce batch size:**

   ```python
   # In model configuration, set batch_size=1
   ```

2. **Enable memory optimization:**

   ```bash
   # Use RTX 4080 optimization script
   python backend/optimize_for_rtx4080.py
   ```

3. **Close other GPU applications:**

   ```bash
   # Close browsers, games, other AI applications
   nvidia-smi  # Check what's using GPU memory
   ```

4. **Use CPU fallback:**
   ```python
   # Force CPU mode if GPU memory insufficient
   device = "cpu"
   ```

## Network and Connectivity Issues

### Issue: Frontend cannot connect to backend

**Symptoms:**

- API calls fail with network errors
- "Connection refused" errors
- CORS errors in browser console

**Diagnostic Steps:**

1. **Check backend is running:**

   ```bash
   curl http://localhost:8000/health
   # Or visit in browser: http://localhost:8000/docs
   ```

2. **Check network connectivity:**
   ```bash
   telnet localhost 8000
   ```

**Solutions:**

1. **Check CORS configuration:**

   ```python
   # In backend/app.py, verify CORS settings
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Update API URL in frontend:**

   ```bash
   # Check frontend/.env or vite config
   VITE_API_URL=http://localhost:8000
   ```

3. **Disable firewall temporarily:**
   ```bash
   # Test if firewall is blocking connections
   # Windows: Turn off Windows Defender Firewall temporarily
   ```

## Model and AI-Specific Issues

### Issue: Models fail to download or load

**Symptoms:**

- "Model not found" errors
- Download timeouts or failures
- Corrupted model files

**Solutions:**

1. **Check internet connection:**

   ```bash
   ping huggingface.co
   ```

2. **Manual model download:**

   ```bash
   # Download models manually to models/ directory
   # Verify file integrity with checksums
   ```

3. **Clear model cache:**

   ```bash
   rm -rf models/
   python backend/core/enhanced_model_downloader.py
   ```

4. **Check disk space:**
   ```bash
   # Ensure 50+ GB free space for models
   df -h  # Linux/Mac
   dir   # Windows
   ```

### Issue: Video generation produces poor quality or fails

**Symptoms:**

- Generated videos are corrupted
- Low quality output
- Generation process hangs

**Solutions:**

1. **Check model compatibility:**

   ```bash
   # Verify model files are complete and uncorrupted
   python backend/diagnose_system.py
   ```

2. **Adjust generation parameters:**

   ```python
   # Reduce resolution, frame count, or complexity
   # Use simpler prompts for testing
   ```

3. **Monitor system resources:**
   ```bash
   # Check CPU, GPU, RAM usage during generation
   nvidia-smi -l 1  # Monitor GPU
   htop  # Monitor CPU/RAM (Linux)
   ```

## Advanced Troubleshooting

### Issue: Intermittent crashes or instability

**Diagnostic Steps:**

1. **Check system logs:**

   ```bash
   # Windows Event Viewer
   # Linux: journalctl -f
   # Look for hardware errors, driver issues
   ```

2. **Run memory test:**

   ```bash
   # Windows Memory Diagnostic
   # memtest86 for thorough testing
   ```

3. **Check temperatures:**
   ```bash
   # Monitor CPU/GPU temperatures
   # Ensure adequate cooling
   ```

**Solutions:**

1. **Update all drivers:**

   ```bash
   # GPU drivers, chipset drivers, etc.
   ```

2. **Check power supply:**

   ```bash
   # Ensure adequate PSU for GPU requirements
   # RTX 4080 needs 750W+ PSU
   ```

3. **Reduce overclocks:**
   ```bash
   # Reset GPU/CPU to stock settings
   # Test stability at default clocks
   ```

### Issue: Performance degradation over time

**Symptoms:**

- System gets slower during use
- Memory usage keeps increasing
- Generation times increase

**Solutions:**

1. **Monitor memory leaks:**

   ```bash
   # Check for processes with increasing memory usage
   # Restart application periodically
   ```

2. **Clear temporary files:**

   ```bash
   # Clean temp directories, model cache
   rm -rf temp/ cache/ logs/
   ```

3. **Restart services:**
   ```bash
   # Restart backend/frontend periodically
   # Full system restart if needed
   ```

## Getting Help

### Information to Collect Before Asking for Help

1. **System Information:**

   ```bash
   python start.py --diagnostics > system_info.txt
   ```

2. **Error Logs:**

   ```bash
   # Copy full error messages and stack traces
   # Include logs from both backend and frontend
   ```

3. **Configuration:**
   ```bash
   # Share relevant config files (remove sensitive data)
   # Include startup command used
   ```

### Where to Get Help

1. **Documentation:**

   - Check all README files in the project
   - Review configuration examples
   - Search existing documentation

2. **Logs and Diagnostics:**

   - Always check logs first
   - Run diagnostic commands
   - Try verbose/debug modes

3. **Community Support:**
   - Include system information
   - Provide complete error messages
   - Describe steps to reproduce issue

### Self-Help Checklist

Before asking for help, try:

- [ ] Restart the application
- [ ] Restart your computer
- [ ] Check system requirements
- [ ] Update drivers (especially GPU)
- [ ] Run diagnostic commands
- [ ] Check logs for error messages
- [ ] Try basic startup mode
- [ ] Verify internet connection
- [ ] Check available disk space
- [ ] Close other resource-intensive applications

Most issues can be resolved by following this troubleshooting guide systematically. Start with the quick diagnostic commands and work through the relevant sections based on your specific symptoms.
