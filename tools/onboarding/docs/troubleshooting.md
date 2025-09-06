# Troubleshooting Guide

This guide helps you resolve common issues encountered during WAN22 development. Issues are organized by category with step-by-step solutions.

## 🚨 Quick Diagnostics

Before diving into specific issues, run these diagnostic commands:

```bash
# Comprehensive environment check
python tools/dev-environment/environment_validator.py --validate

# Quick health status
python tools/dev-feedback/feedback_cli.py status

# Check recent errors
python tools/dev-feedback/debug_tools.py --errors 5
```

## 🐍 Python/Backend Issues

### Import Errors

#### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'xyz'`

**Solutions**:

1. **Check virtual environment**:

   ```bash
   # Verify you're in virtual environment
   which python  # Should show venv path

   # Activate if needed
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Install missing dependencies**:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Check Python path**:

   ```bash
   # Add project root to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **Verify package installation**:
   ```bash
   pip list | grep package-name
   pip show package-name
   ```

#### Relative Import Issues

**Problem**: `ImportError: attempted relative import with no known parent package`

**Solutions**:

1. **Run as module**:

   ```bash
   # Instead of: python backend/app.py
   python -m backend.app
   ```

2. **Fix import statements**:

   ```python
   # Good: Absolute imports
   from backend.services.video_service import VideoService
   from core.models.video import Video

   # Avoid: Relative imports in scripts
   from ..services.video_service import VideoService
   ```

### FastAPI Server Issues

#### Port Already in Use

**Problem**: `OSError: [Errno 48] Address already in use`

**Solutions**:

1. **Find and kill process**:

   ```bash
   # Find process using port 8000
   lsof -i :8000          # macOS/Linux
   netstat -ano | findstr :8000  # Windows

   # Kill process
   kill -9 <PID>          # macOS/Linux
   taskkill /PID <PID> /F # Windows
   ```

2. **Use different port**:

   ```bash
   # Set different port
   export API_PORT=8001
   python backend/start_server.py
   ```

3. **Use startup manager**:
   ```bash
   # Automatic port resolution
   python start.py
   ```

#### CORS Issues

**Problem**: Frontend can't connect to backend due to CORS

**Solutions**:

1. **Check CORS configuration**:

   ```python
   # In backend/app.py
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],  # Add frontend URL
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Verify frontend URL**:
   ```bash
   # Check frontend is running on expected port
   curl http://localhost:3000
   ```

### Database Issues

#### Connection Errors

**Problem**: Database connection failures

**Solutions**:

1. **Check database configuration**:

   ```python
   # Verify database URL in config
   DATABASE_URL = "sqlite:///./wan22_tasks.db"
   ```

2. **Initialize database**:

   ```bash
   python backend/init_db.py
   ```

3. **Check file permissions**:
   ```bash
   ls -la wan22_tasks.db
   chmod 664 wan22_tasks.db
   ```

### Memory Issues

#### Out of Memory (CUDA)

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:

   ```python
   # In model configuration
   batch_size = 1  # Reduce from default
   ```

2. **Clear GPU cache**:

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Monitor GPU usage**:

   ```bash
   nvidia-smi
   watch -n 1 nvidia-smi
   ```

4. **Use CPU fallback**:
   ```python
   # Force CPU usage
   device = "cpu"
   ```

## ⚛️ Frontend/React Issues

### Node.js Issues

#### Node Version Conflicts

**Problem**: `error: The engine "node" is incompatible with this module`

**Solutions**:

1. **Check Node version**:

   ```bash
   node --version  # Should be 16+
   ```

2. **Use Node Version Manager**:

   ```bash
   # Install nvm (if not installed)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

   # Install and use correct Node version
   nvm install 18
   nvm use 18
   ```

3. **Update package.json engines**:
   ```json
   {
     "engines": {
       "node": ">=16.0.0",
       "npm": ">=8.0.0"
     }
   }
   ```

#### npm Install Failures

**Problem**: `npm install` fails with various errors

**Solutions**:

1. **Clear npm cache**:

   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Use different registry**:

   ```bash
   npm install --registry https://registry.npmjs.org/
   ```

3. **Check disk space**:
   ```bash
   df -h  # Check available disk space
   ```

### Build Issues

#### TypeScript Compilation Errors

**Problem**: TypeScript compilation fails

**Solutions**:

1. **Check TypeScript configuration**:

   ```json
   // tsconfig.json
   {
     "compilerOptions": {
       "target": "ES2020",
       "lib": ["ES2020", "DOM", "DOM.Iterable"],
       "allowJs": false,
       "skipLibCheck": true,
       "esModuleInterop": false,
       "allowSyntheticDefaultImports": true,
       "strict": true,
       "forceConsistentCasingInFileNames": true,
       "module": "ESNext",
       "moduleResolution": "bundler",
       "resolveJsonModule": true,
       "isolatedModules": true,
       "noEmit": true,
       "jsx": "react-jsx"
     }
   }
   ```

2. **Fix type errors**:

   ```typescript
   // Add proper type annotations
   const [count, setCount] = useState<number>(0);

   // Use type assertions carefully
   const element = document.getElementById("root") as HTMLElement;
   ```

3. **Update dependencies**:
   ```bash
   npm update @types/react @types/react-dom typescript
   ```

#### Vite Build Issues

**Problem**: Vite build or dev server fails

**Solutions**:

1. **Clear Vite cache**:

   ```bash
   rm -rf node_modules/.vite
   npm run dev
   ```

2. **Check Vite configuration**:

   ```typescript
   // vite.config.ts
   import { defineConfig } from "vite";
   import react from "@vitejs/plugin-react";

   export default defineConfig({
     plugins: [react()],
     server: {
       port: 3000,
       proxy: {
         "/api": "http://localhost:8000",
       },
     },
   });
   ```

3. **Update Vite**:
   ```bash
   npm update vite @vitejs/plugin-react
   ```

### Runtime Issues

#### API Connection Failures

**Problem**: Frontend can't connect to backend API

**Solutions**:

1. **Check API URL configuration**:

   ```typescript
   // Check .env file
   VITE_API_URL=http://localhost:8000

   // Verify in code
   const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
   ```

2. **Test API directly**:

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/api/v1/status
   ```

3. **Check network tab in browser DevTools**:
   - Open DevTools (F12)
   - Go to Network tab
   - Look for failed requests
   - Check request/response details

## 🔧 Development Tools Issues

### Test Runner Issues

#### Tests Not Found

**Problem**: Test runner can't find tests

**Solutions**:

1. **Check test file naming**:

   ```bash
   # Correct naming patterns
   test_*.py
   *_test.py
   *.test.ts
   *.test.js
   ```

2. **Verify test directory structure**:

   ```
   tests/
   ├── unit/
   ├── integration/
   └── e2e/
   ```

3. **Check test configuration**:
   ```yaml
   # tests/config/test-config.yaml
   test_patterns:
     - "test_*.py"
     - "*_test.py"
   ```

#### Test Failures

**Problem**: Tests are failing unexpectedly

**Solutions**:

1. **Run tests in isolation**:

   ```bash
   # Run single test file
   python -m pytest tests/unit/test_video_service.py -v

   # Run single test
   python -m pytest tests/unit/test_video_service.py::test_generate_video -v
   ```

2. **Check test dependencies**:

   ```bash
   # Install test dependencies
   pip install pytest pytest-asyncio pytest-mock
   ```

3. **Clear test cache**:

   ```bash
   # Python
   rm -rf .pytest_cache __pycache__

   # Frontend
   rm -rf node_modules/.cache
   ```

### File Watcher Issues

#### File Changes Not Detected

**Problem**: File watcher not detecting changes

**Solutions**:

1. **Check file system limits**:

   ```bash
   # Linux: Increase inotify limits
   echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

2. **Restart file watcher**:

   ```bash
   # Stop and restart watcher
   python tools/dev-feedback/test_watcher.py
   ```

3. **Check ignored patterns**:
   ```python
   # Verify ignore patterns in watcher config
   ignore_patterns = [
       "*.pyc",
       "__pycache__/*",
       "node_modules/*",
       ".git/*"
   ]
   ```

## 🖥️ System-Level Issues

### Permission Issues

#### File Permission Errors

**Problem**: Permission denied errors

**Solutions**:

1. **Fix file permissions**:

   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   chmod +x tools/**/*.py

   # Fix directory permissions
   chmod 755 logs/ outputs/
   ```

2. **Check ownership**:

   ```bash
   # Change ownership if needed
   sudo chown -R $USER:$USER .
   ```

3. **Run with appropriate privileges**:
   ```bash
   # Windows: Run as Administrator
   # macOS/Linux: Use sudo only when necessary
   ```

### Environment Issues

#### Environment Variables Not Set

**Problem**: Configuration not loading from environment

**Solutions**:

1. **Check .env file location**:

   ```bash
   # Verify .env files exist
   ls -la backend/.env
   ls -la frontend/.env
   ```

2. **Load environment variables**:

   ```bash
   # Source .env file manually
   export $(cat backend/.env | xargs)

   # Or use python-dotenv
   pip install python-dotenv
   ```

3. **Verify environment loading**:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   print(os.getenv('DEBUG'))
   ```

### Network Issues

#### Firewall Blocking Connections

**Problem**: Firewall blocking development servers

**Solutions**:

1. **Check firewall settings**:

   ```bash
   # Windows
   netsh advfirewall show allprofiles

   # macOS
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

   # Linux
   sudo ufw status
   ```

2. **Allow ports through firewall**:

   ```bash
   # Windows
   netsh advfirewall firewall add rule name="WAN22 Backend" dir=in action=allow protocol=TCP localport=8000

   # Linux
   sudo ufw allow 8000
   sudo ufw allow 3000
   ```

3. **Use localhost instead of 0.0.0.0**:
   ```python
   # More restrictive, often works better
   uvicorn.run(app, host="127.0.0.1", port=8000)
   ```

## 🔍 Debugging Strategies

### Systematic Debugging

1. **Reproduce the issue**:

   - Document exact steps to reproduce
   - Note environment conditions
   - Check if issue is consistent

2. **Gather information**:

   ```bash
   # System information
   python tools/dev-environment/environment_validator.py --validate

   # Recent logs
   python tools/dev-feedback/debug_tools.py --errors 10

   # Health check
   python tools/health-checker/health_checker.py
   ```

3. **Isolate the problem**:

   - Test individual components
   - Use minimal reproduction case
   - Check dependencies one by one

4. **Use debugging tools**:

   ```bash
   # Enable debug logging
   python tools/dev-feedback/debug_tools.py --enable

   # Start debug session
   python tools/dev-feedback/feedback_cli.py debug
   ```

### Logging and Monitoring

1. **Enable verbose logging**:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use structured logging**:

   ```python
   logger.info("Processing video", extra={
       "video_id": video.id,
       "duration": video.duration,
       "model": model_name
   })
   ```

3. **Monitor resource usage**:

   ```bash
   # CPU and memory
   top
   htop

   # GPU usage
   nvidia-smi
   watch -n 1 nvidia-smi

   # Disk usage
   df -h
   du -sh *
   ```

## 🆘 Getting Help

### Before Asking for Help

1. **Search existing documentation**:

   - Check this troubleshooting guide
   - Review project documentation
   - Search issue tracker

2. **Gather diagnostic information**:

   ```bash
   # Generate comprehensive report
   python tools/dev-feedback/debug_tools.py --report debug_report.json

   # Environment validation
   python tools/dev-environment/environment_validator.py --export env_report.json
   ```

3. **Create minimal reproduction**:
   - Isolate the issue
   - Remove unnecessary code
   - Document exact steps

### How to Ask for Help

1. **Provide context**:

   - What were you trying to do?
   - What did you expect to happen?
   - What actually happened?

2. **Include relevant information**:

   - Error messages (full stack traces)
   - System information
   - Steps to reproduce
   - Diagnostic reports

3. **Share code snippets**:
   ````markdown
   ```python
   # Your code here
   def problematic_function():
       pass
   ```
   ````
   ```

   ```

### Where to Get Help

1. **Team Resources**:

   - Ask your mentor or team lead
   - Check team chat/Slack
   - Review team documentation

2. **Project Resources**:

   - GitHub Issues
   - Project wiki
   - Code comments and documentation

3. **External Resources**:
   - Stack Overflow
   - Framework documentation
   - Community forums

## 📚 Additional Resources

### Documentation Links

- [FastAPI Troubleshooting](https://fastapi.tiangolo.com/tutorial/debugging/)
- [React Debugging Guide](https://react.dev/learn/react-developer-tools)
- [Vite Troubleshooting](https://vitejs.dev/guide/troubleshooting.html)
- [Python Debugging](https://docs.python.org/3/library/pdb.html)

### Useful Commands Reference

```bash
# System diagnostics
python tools/dev-environment/environment_validator.py --validate
python tools/dev-feedback/feedback_cli.py status
python tools/health-checker/health_checker.py

# Development servers
python start.py
python backend/start_server.py
cd frontend && npm run dev

# Testing
python tools/test-runner/orchestrator.py --run-all
cd frontend && npm test

# Debugging
python tools/dev-feedback/debug_tools.py --enable
python tools/dev-feedback/feedback_cli.py debug

# File watching
python tools/dev-feedback/test_watcher.py
python tools/dev-feedback/config_watcher.py

# Health monitoring
python tools/health-checker/health_checker.py --continuous
```

Remember: Most issues have been encountered before. Take your time, read error messages carefully, and don't hesitate to ask for help! 🚀
