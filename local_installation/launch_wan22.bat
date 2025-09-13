@echo off
title WAN2.2 Video Generator
echo ========================================
echo WAN2.2 Video Generation System
echo ========================================
echo.

REM Change to installation directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "%~dp0venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating environment...
call "%~dp0venv\Scripts\activate.bat"

REM Check if main application exists
if not exist "application\main.py" (
    echo Error: Main application not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Launch main application
echo Starting WAN2.2...
echo.
python "application\main.py"

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the logs for details.
    echo Press any key to close.
    pause >nul
)
