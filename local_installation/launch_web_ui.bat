@echo off
REM ============================================================================
REM WAN2.2 Web UI Launcher
REM Launches the WAN2.2 web-based user interface
REM ============================================================================

setlocal enabledelayedexpansion

REM Set console properties
title WAN2.2 - Web UI Server
color 0F

REM Get installation directory
set "INSTALL_DIR=%~dp0"
set "PYTHON_CMD=python"
set "VENV_DIR=%INSTALL_DIR%venv"
set "APP_DIR=%INSTALL_DIR%application"

REM Check if virtual environment exists
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYTHON_CMD=%VENV_DIR%\Scripts\python.exe"
    echo Using virtual environment Python: %PYTHON_CMD%
) else (
    echo Virtual environment not found, using system Python
)

REM Check if application directory exists
if not exist "%APP_DIR%" (
    echo Error: Application directory not found at %APP_DIR%
    echo Please ensure WAN2.2 is properly installed.
    pause
    exit /b 1
)

REM Install Flask if not present
echo Checking Flask installation...
"%PYTHON_CMD%" -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    "%PYTHON_CMD%" -m pip install flask werkzeug
    if errorlevel 1 (
        echo Error: Failed to install Flask
        echo Please install manually: pip install flask werkzeug
        pause
        exit /b 1
    )
)

REM Launch the web UI
echo Starting WAN2.2 Web UI Server...
echo.
echo The web interface will open in your browser automatically.
echo If it doesn't open, navigate to: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server.
echo.

cd /d "%INSTALL_DIR%"

REM Launch the web UI
"%PYTHON_CMD%" "%APP_DIR%\web_ui.py"

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo Error launching WAN2.2 Web UI
    echo ============================================================================
    echo.
    echo Possible solutions:
    echo 1. Ensure the installation completed successfully
    echo 2. Install Flask: pip install flask werkzeug
    echo 3. Check that all dependencies are installed
    echo 4. Try running the installation again
    echo.
    pause
    exit /b 1
)

echo.
echo WAN2.2 Web UI server stopped.
pause