@echo off
title PyTorch DLL Fix
echo ========================================
echo PyTorch DLL Loading Fix
echo ========================================
echo.

REM Change to parent directory
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run the installer first.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating environment...
call "venv\Scripts\activate.bat"

REM Run the Python fix script
echo Running PyTorch DLL fix...
python "local_installation\fix_pytorch_dll.py"

echo.
echo Fix completed. You can now try launching the UI again.
echo.
pause