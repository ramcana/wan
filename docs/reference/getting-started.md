---
category: reference
last_updated: '2025-09-15T22:50:00.829157'
original_path: tools\onboarding\docs\getting-started.md
tags:
- configuration
- api
- troubleshooting
- installation
title: Getting Started with WAN22 Development
---

# Getting Started with WAN22 Development

Welcome to the WAN22 video generation system! This guide will help you get up and running quickly as a new developer on the project.

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites Check

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Node.js 16+** installed
- **Git** installed
- A **code editor** (VS Code recommended)

### 2. One-Command Setup

Run our automated setup script:

```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd wan22

# Run the automated onboarding
python tools/onboarding/setup_wizard.py
```

This script will:

- ✅ Check your system requirements
- ✅ Install all dependencies
- ✅ Set up your development environment
- ✅ Run initial tests to verify everything works
- ✅ Open the application in your browser

### 3. Verify Installation

After setup completes, you should see:

- Backend server running at http://localhost:8000
- Frontend application at http://localhost:3000
- All tests passing

## 📚 What's Next?

Once your environment is set up, continue with:

1. **[Project Overview](project-overview.md)** - Understand the architecture
2. **[Development Setup](development-setup.md)** - Detailed development workflow
3. **[Coding Standards](coding-standards.md)** - Follow our coding guidelines
4. **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🆘 Need Help?

If you encounter any issues:

1. **Check the troubleshooting guide**: [troubleshooting.md](troubleshooting.md)
2. **Run the diagnostic tool**: `python tools/dev-environment/environment_validator.py --validate`
3. **Ask for help**: Contact your mentor or team lead

## 📋 Developer Checklist

Track your onboarding progress:

```bash
# Check your progress
python tools/onboarding/developer_checklist.py --status

# Mark items as complete
python tools/onboarding/developer_checklist.py --complete "Environment Setup"
```

## 🎯 First Tasks

Ready to contribute? Here are some good first tasks:

1. **Run the test suite**: `python tools/test-runner/orchestrator.py --run-all`
2. **Explore the codebase**: Start with `backend/app.py` and `frontend/src/App.tsx`
3. **Make a small change**: Try updating a UI component or adding a test
4. **Submit your first PR**: Follow our contribution guidelines

## 🏗️ Project Structure Overview

```
wan22/
├── backend/           # FastAPI backend server
├── frontend/          # React frontend application
├── core/             # Core business logic
├── infrastructure/   # Infrastructure and configuration
├── tests/            # Test suites
├── tools/            # Development tools
├── docs/             # Documentation
└── config/           # Configuration files
```

## 🔧 Development Workflow

1. **Start development servers**:

   ```bash
   # Backend
   cd backend && python start_server.py

   # Frontend (new terminal)
   cd frontend && npm run dev
   ```

2. **Run tests**:

   ```bash
   # All tests
   python tools/test-runner/orchestrator.py --run-all

   # Watch mode
   python tools/dev-feedback/test_watcher.py
   ```

3. **Check code quality**:

   ```bash
   # Health check
   python tools/health-checker/health_checker.py

   # Pre-commit hooks
   pre-commit run --all-files
   ```

## 🎉 Welcome to the Team!

You're now ready to start contributing to WAN22! Remember:

- **Ask questions** - We're here to help
- **Follow the coding standards** - Consistency is key
- **Write tests** - Quality code is tested code
- **Document your changes** - Help future developers (including yourself!)

Happy coding! 🚀
