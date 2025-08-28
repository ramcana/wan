@echo off
title WAN2.2 UI
echo ========================================
echo WAN2.2 User Interface
echo ========================================
echo.

REM Change to parent directory where the Gradio UI is located
cd /d "%~dp0.."

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

REM Check if Gradio UI application exists
if not exist "main.py" (
    echo Error: Gradio UI application not found!
    echo Please ensure main.py exists in the root directory.
    echo.
    pause
    exit /b 1
)

REM Launch Gradio UI application
echo Starting WAN2.2 Gradio UI...
echo.
echo ========================================
echo  WAN2.2 Gradio Web Interface
echo ========================================
echo.
echo The web interface will be available at:
echo   http://localhost:7860
echo.
echo Features:
echo   * Generation Tab - T2V, I2V, TI2V video generation
echo   * Optimizations Tab - VRAM and performance settings  
echo   * Queue and Stats Tab - Task management and monitoring
echo   * Outputs Tab - Video gallery and file management
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python main.py --host 127.0.0.1 --port 7860

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo ========================================
    echo An error occurred during startup
    echo ========================================
    echo.
    echo This may be due to:
    echo   * Missing GPU drivers or CUDA installation
    echo   * Incomplete dependency installation
    echo   * System compatibility issues
    echo.
    echo Solutions:
    echo   * Check the installation guide
    echo   * Run the installer again
    echo   * Check logs in wan22_ui.log
    echo.
    echo Press any key to close.
    pause >nul
)