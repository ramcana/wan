---
category: reference
last_updated: '2025-09-15T22:50:00.829157'
original_path: tools\onboarding\docs\development-setup.md
tags:
- configuration
- api
- troubleshooting
- installation
- performance
title: Development Environment Setup
---

# Development Environment Setup

This guide provides detailed instructions for setting up your development environment for the WAN22 project.

## 📋 System Requirements

### Required Software

| Software    | Version | Purpose                           |
| ----------- | ------- | --------------------------------- |
| Python      | 3.8+    | Backend development               |
| Node.js     | 16+     | Frontend development              |
| Git         | Latest  | Version control                   |
| Code Editor | Any     | Development (VS Code recommended) |

### Optional Software

| Software     | Purpose                         |
| ------------ | ------------------------------- |
| Docker       | Containerized development       |
| CUDA Toolkit | GPU acceleration                |
| PostgreSQL   | Database (if using external DB) |

## 🔧 Manual Setup Instructions

### 1. Python Environment

```bash
# Check Python version
python --version  # Should be 3.8+

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Node.js Environment

```bash
# Check Node.js version
node --version  # Should be 16+

# Install frontend dependencies
cd frontend
npm install

# Verify installation
npm run build
```

### 3. Git Configuration

```bash
# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### 4. Development Tools

```bash
# Install additional development tools
pip install black isort mypy pytest

# Install VS Code extensions (if using VS Code)
code --install-extension ms-python.python
code --install-extension bradlc.vscode-tailwindcss
code --install-extension esbenp.prettier-vscode
```

## ⚙️ Configuration

### Environment Variables

Create environment files:

**Backend (.env)**:

```bash
# backend/.env
DEBUG=true
LOG_LEVEL=INFO
API_HOST=localhost
API_PORT=8000
```

**Frontend (.env)**:

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
VITE_DEBUG=true
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative"
}
```

#### VS Code Extensions

Recommended extensions:

- Python
- TypeScript and JavaScript Language Features
- Tailwind CSS IntelliSense
- Prettier - Code formatter
- GitLens
- Thunder Client (for API testing)

## 🚀 Starting Development Servers

### Method 1: Automated Startup

```bash
# Start both servers automatically
python start.py

# Or use the batch file (Windows)
start_both_servers.bat
```

### Method 2: Manual Startup

```bash
# Terminal 1: Backend server
cd backend
python start_server.py

# Terminal 2: Frontend server
cd frontend
npm run dev
```

### Method 3: Using Development Tools

```bash
# Start with file watching
python tools/dev-feedback/feedback_cli.py watch-tests &
python tools/dev-feedback/feedback_cli.py watch-config &

# Start servers
python start.py
```

## 🧪 Testing Your Setup

### 1. Run Health Check

```bash
# Comprehensive environment validation
python tools/dev-environment/environment_validator.py --validate

# Quick health check
python tools/dev-feedback/feedback_cli.py status
```

### 2. Run Test Suite

```bash
# Run all tests
python tools/test-runner/orchestrator.py --run-all

# Run specific test categories
python tools/test-runner/orchestrator.py --category unit
python tools/test-runner/orchestrator.py --category integration
```

### 3. Verify API Endpoints

```bash
# Check backend health
curl http://localhost:8000/health

# Check API documentation
open http://localhost:8000/docs
```

### 4. Verify Frontend

```bash
# Check frontend
open http://localhost:3000

# Run frontend tests
cd frontend
npm test
```

## 🔍 Development Workflow

### Daily Development

1. **Start your day**:

   ```bash
   # Pull latest changes
   git pull origin main

   # Update dependencies (if needed)
   pip install -r backend/requirements.txt
   cd frontend && npm install

   # Start development servers
   python start.py
   ```

2. **During development**:

   ```bash
   # Watch tests automatically
   python tools/dev-feedback/test_watcher.py

   # Watch configuration changes
   python tools/dev-feedback/config_watcher.py

   # Monitor project health
   python tools/health-checker/health_checker.py
   ```

3. **Before committing**:

   ```bash
   # Run pre-commit checks
   pre-commit run --all-files

   # Run full test suite
   python tools/test-runner/orchestrator.py --run-all

   # Check code quality
   python tools/health-checker/health_checker.py
   ```

### Code Quality Tools

```bash
# Format Python code
black backend/ core/ infrastructure/

# Sort imports
isort backend/ core/ infrastructure/

# Type checking
mypy backend/ core/ infrastructure/

# Lint frontend code
cd frontend
npm run lint

# Format frontend code
cd frontend
npm run format
```

## 🐛 Debugging Setup

### Python Debugging

1. **VS Code Debugging**:
   Create `.vscode/launch.json`:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: FastAPI",
         "type": "python",
         "request": "launch",
         "program": "${workspaceFolder}/backend/start_server.py",
         "console": "integratedTerminal",
         "justMyCode": true
       }
     ]
   }
   ```

2. **Debug Tools**:

   ```bash
   # Enable debug logging
   python tools/dev-feedback/debug_tools.py --enable

   # Start debug session
   python tools/dev-feedback/feedback_cli.py debug
   ```

### Frontend Debugging

1. **Browser DevTools**: Use Chrome/Firefox developer tools
2. **VS Code Debugging**: Install "Debugger for Chrome" extension
3. **React DevTools**: Install React Developer Tools browser extension

## 📊 Performance Monitoring

### Development Performance

```bash
# Monitor test performance
python tools/test-runner/orchestrator.py --benchmark

# Monitor application performance
python tools/health-checker/health_checker.py --performance

# Profile function execution
python tools/dev-feedback/debug_tools.py --profile
```

### Resource Monitoring

```bash
# Check system resources
python tools/dev-environment/environment_validator.py --gpu

# Monitor memory usage
python tools/health-checker/health_checker.py --memory

# Check disk space
df -h  # Unix/Linux/macOS
dir   # Windows
```

## 🔧 Troubleshooting

### Common Issues

1. **Port conflicts**:

   ```bash
   # Check what's using ports
   netstat -tulpn | grep :8000  # Linux/macOS
   netstat -ano | findstr :8000  # Windows

   # Use different ports
   API_PORT=8001 python backend/start_server.py
   ```

2. **Permission errors**:

   ```bash
   # Fix file permissions (Unix/Linux/macOS)
   chmod +x scripts/*.sh

   # Run as administrator (Windows)
   # Right-click Command Prompt -> "Run as administrator"
   ```

3. **Dependency conflicts**:

   ```bash
   # Clean Python environment
   pip freeze > requirements_backup.txt
   pip uninstall -r requirements_backup.txt -y
   pip install -r backend/requirements.txt

   # Clean Node.js environment
   rm -rf frontend/node_modules
   cd frontend && npm install
   ```

4. **Git issues**:

   ```bash
   # Reset to clean state
   git stash
   git pull origin main
   git stash pop

   # Fix line endings (Windows)
   git config --global core.autocrlf true
   ```

### Getting Help

1. **Check logs**:

   ```bash
   # Application logs
   tail -f logs/wan22_ui.log

   # Debug logs
   python tools/dev-feedback/debug_tools.py --errors
   ```

2. **Run diagnostics**:

   ```bash
   # Comprehensive diagnosis
   python tools/dev-feedback/feedback_cli.py status

   # Environment validation
   python tools/dev-environment/environment_validator.py --validate
   ```

3. **Ask for help**:
   - Check the [troubleshooting guide](troubleshooting.md)
   - Contact your mentor or team lead
   - Create an issue in the project repository

## 🎯 Next Steps

Once your development environment is set up:

1. **Read the [Project Overview](project-overview.md)** to understand the architecture
2. **Review [Coding Standards](coding-standards.md)** to follow best practices
3. **Complete the [Developer Checklist](../developer_checklist.py)** to track your progress
4. **Start with your first task** - ask your mentor for a good starter issue

Happy coding! 🚀
