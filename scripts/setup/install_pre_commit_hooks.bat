@echo off
REM Setup script for installing pre-commit hooks on Windows
REM This script installs and configures pre-commit hooks for the project

echo 🚀 Setting up pre-commit hooks for WAN22 Video Generation System
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.8 or higher and add it to PATH
    pause
    exit /b 1
)

echo ✅ Python is available

REM Install pre-commit
echo 📦 Installing pre-commit...
python -m pip install pre-commit
if errorlevel 1 (
    echo ❌ Failed to install pre-commit
    echo 💡 Try running: pip install pre-commit
    pause
    exit /b 1
)

echo ✅ pre-commit installed successfully

REM Check if .pre-commit-config.yaml exists
if not exist ".pre-commit-config.yaml" (
    echo ❌ .pre-commit-config.yaml not found
    pause
    exit /b 1
)

REM Install hooks
echo 🔧 Installing pre-commit hooks...
pre-commit install
if errorlevel 1 (
    echo ❌ Failed to install pre-commit hooks
    pause
    exit /b 1
)

REM Install pre-push hooks (optional)
pre-commit install --hook-type pre-push
if errorlevel 1 (
    echo ⚠️  Failed to install pre-push hooks (optional)
)

echo ✅ Pre-commit hooks installed successfully

REM Set up git configuration
echo ⚙️  Setting up git configuration...
git config core.hooksPath .git/hooks
git config pre-commit.enabled true
echo ✅ Git configuration updated

REM Run initial check
echo 🧪 Running initial pre-commit check...
pre-commit run --all-files
if errorlevel 1 (
    echo ⚠️  Initial pre-commit check found issues
    echo 💡 This is normal for first setup. Run 'pre-commit run --all-files' to fix auto-fixable issues
) else (
    echo ✅ Initial pre-commit check passed
)

echo.
echo ============================================================
echo ✅ Pre-commit hooks setup completed successfully!
echo.
echo 📋 What happens now:
echo   • Pre-commit hooks will run automatically on 'git commit'
echo   • Hooks will check code quality, tests, config, and documentation
echo   • Some issues will be auto-fixed, others will need manual attention
echo.
echo 🔧 Useful commands:
echo   • Run hooks manually: pre-commit run --all-files
echo   • Skip hooks (emergency): git commit --no-verify
echo   • Update hooks: pre-commit autoupdate
echo   • Uninstall hooks: pre-commit uninstall
echo.
pause