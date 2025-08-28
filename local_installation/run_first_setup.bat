@echo off
title WAN2.2 First-Run Setup
echo ========================================
echo WAN2.2 First-Run Configuration Wizard
echo ========================================
echo.

REM Change to installation directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating environment...
call "venv\Scripts\activate.bat"

REM Check if setup script exists
if not exist "scripts\post_install_setup.py" (
    echo Error: Setup script not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Run post-installation setup
python "scripts\post_install_setup.py" "%~dp0"

REM Keep window open
echo.
echo Press any key to close.
pause >nul