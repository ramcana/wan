# Developer Onboarding Guide

Welcome to the WAN22 project! This guide will help you understand the project structure and get started with development in 30 minutes or less.

## 🚀 Quick Start (5 minutes)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd wan22
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python start.py --help
python -m local_testing_framework --quick-test
```

### 3. Run Your First Test
```bash
python backend/test_real_ai_ready.py
```

## 📁 Project Structure Overview (10 minutes)

The project is organized into several main areas:

### 📁 `utils_new/`

**Purpose:** Python Package
**Files:** 305

### 🏠 `local_installation/`

**Purpose:** Local Installation Package
**Files:** 112

### 📁 `docs/`

**Purpose:** Documentation
**Files:** 103

### 🧪 `wan/`

**Purpose:** Test Directory
**Files:** 84

### 🧪 `tests/`

**Purpose:** Test Suite
**Files:** 72

**Test types:**
- `unit/` - Unit tests
- `integration/` - Integration tests
- `e2e/` - End-to-end tests

### 📜 `scripts/`

**Purpose:** Automation Scripts
**Files:** 59

### 🔧 `backend/`

**Purpose:** Backend Application Code
**Files:** 29

**Key areas:**
- `api/` - REST API endpoints
- `core/` - Business logic
- `services/` - Service layer
- `models/` - Data models

### 🎨 `frontend/`

**Purpose:** Frontend Application Code
**Files:** 36

**Key areas:**
- `src/` - React/TypeScript source
- `public/` - Static assets
- UI components and styling


## 🧠 Key Concepts (10 minutes)

### WAN22 System
WAN22 is an AI-powered video generation system that:
- Converts text prompts to videos (T2V)
- Converts images to videos (I2V)
- Handles text+image to video (TI2V)

### Architecture Pattern
The system follows a **layered architecture**:

```
Frontend (React/TypeScript)
    ↓
Backend API (FastAPI/Python)
    ↓
Core Services (AI Models)
    ↓
Infrastructure (GPU/Storage)
```

### Key Technologies
- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: React, TypeScript, Vite
- **AI Models**: Diffusion models, Transformers
- **Infrastructure**: CUDA, Docker (optional)

### Critical Components
These components are used by many others - be careful when modifying:

- `config`
- `backend`
- `scripts`
- `core`
- `tests`

## 🔄 Development Workflow (5 minutes)

### Making Changes
1. **Create a branch**: `git checkout -b feature/your-feature`
2. **Make your changes** in the appropriate component
3. **Test locally**: `python -m local_testing_framework`
4. **Run unit tests**: `pytest tests/`
5. **Check code quality**: `python tools/health-checker/cli.py`
6. **Commit and push**: Standard git workflow

### Testing Strategy
- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Local Testing Framework**: Test the complete system
- **Manual testing**: Use the UI for end-to-end validation

### Common Tasks

**Adding a new API endpoint:**
1. Add route in `backend/api/`
2. Add business logic in `backend/core/` or `backend/services/`
3. Add tests in `backend/tests/`
4. Update frontend if needed

**Modifying AI models:**
1. Update model code in `backend/core/`
2. Test with Local Testing Framework
3. Update configuration if needed
4. Validate performance impact

## 🔧 Troubleshooting

### Common Issues

**Import errors:**
- Check virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python path configuration

**Model loading errors:**
- Ensure models are downloaded: `python backend/scripts/download_models.py`
- Check GPU availability: `python backend/test_cuda_detection.py`
- Verify disk space for model storage

**Test failures:**
- Run tests individually to isolate issues
- Check test configuration in `tests/config/`
- Review test logs in `test_logs/`

### Getting Help
- Check existing documentation in `docs/`
- Review similar issues in git history
- Run diagnostic tools: `python backend/diagnose_system.py`
- Use the health checker: `python tools/health-checker/cli.py`

## 🎯 Next Steps

Now that you understand the basics:

1. **Explore the codebase**: Start with the component most relevant to your work
2. **Read detailed docs**: Check `docs/` for specific guides
3. **Join development**: Pick up a task from the issue tracker
4. **Ask questions**: Don't hesitate to ask team members

### Recommended Reading Order
1. This onboarding guide (you're here!)
2. `docs/SYSTEM_REQUIREMENTS.md` - System setup
3. `docs/USER_GUIDE.md` - How to use the system
4. Component-specific documentation in each directory

### Development Resources
- **API Documentation**: `docs/api/`
- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md`
- **Troubleshooting**: `docs/COMPREHENSIVE_TROUBLESHOOTING.md`
- **Performance Guide**: `docs/WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md`
