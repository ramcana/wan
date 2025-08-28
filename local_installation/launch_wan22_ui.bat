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

REM Run PyTorch diagnostics
echo Running PyTorch diagnostics...
python -c "import torch; print('PyTorch OK')" 2>nul
if errorlevel 1 (
    echo.
    echo ========================================
    echo PyTorch Import Error Detected
    echo ========================================
    echo.
    echo Running automated fix script...
    python "local_installation\fix_pytorch_dll.py"
    if errorlevel 1 (
        echo.
        echo Automated fix failed. Please run the fix manually:
        echo   python local_installation\fix_pytorch_dll.py
        echo.
        pause
        exit /b 1
    )
    echo.
    echo PyTorch fix completed. Continuing startup...
    echo.
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
echo   - Generation Tab - T2V, I2V, TI2V video generation
echo   - Optimizations Tab - VRAM and performance settings  
echo   - Queue ^& Stats Tab - Task management and monitoring
echo   - Outputs Tab - Video gallery and file management
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Apply runtime fixes before starting
echo Applying runtime fixes...
python runtime_fixes.py
if errorlevel 1 (
    echo ⚠️  Some runtime fixes failed, but continuing...
)

REM Quick model check
echo Checking models...
python -c "import os; models_dir='models'; required_models=['WAN2.2-T2V-A14B','WAN2.2-I2V-A14B','WAN2.2-TI2V-5B']; missing=[m for m in required_models if not os.path.exists(os.path.join(models_dir,m))]; print('⚠️  Missing models:',missing,'Please move models from local_installation/models/ to models/') if missing else print('✅ All required models found')"

REM Resolve port conflicts
echo Checking port availability...
python port_manager.py --port 7860 --auto-kill
if errorlevel 1 (
    echo ⚠️  Port conflict detected, trying alternative port...
    python port_manager.py --port 7861
)

REM Launch with resolved port (environment variable set by port_manager.py)
python main.py --host 127.0.0.1

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo ========================================
    echo An error occurred during startup
    echo ========================================
    echo.
    echo This may be due to:
    echo   - Missing GPU drivers or CUDA installation
    echo   - Incomplete dependency installation
    echo   - System compatibility issues
    echo.
    echo Solutions:
    echo   - Check the installation guide
    echo   - Run the installer again
    echo   - Check logs in wan22_ui.log
    echo.
    echo Press any key to close.
    pause >nul
)
