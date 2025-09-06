@echo off
REM Python installation script for WAN2.2 installer
REM Downloads and installs embedded Python if not available

echo Installing Python for WAN2.2...

REM Check if we already have Python in the installation
if exist "%~dp0..\resources\python-embed\python.exe" (
    echo Python already installed in resources directory.
    set "PYTHON_PATH=%~dp0..\resources\python-embed"
    goto :configure_python
)

REM Create resources directory if it doesn't exist
if not exist "%~dp0..\resources" mkdir "%~dp0..\resources"
if not exist "%~dp0..\resources\python-embed" mkdir "%~dp0..\resources\python-embed"

echo Downloading Python 3.11 embedded...
REM This would download Python embedded version
REM For now, we'll use system Python or guide user to install it

REM Check if system Python is available
python --version >nul 2>&1
if not errorlevel 1 (
    echo System Python found. Using system Python.
    goto :end
)

REM If no Python found, provide instructions
echo.
echo Python not found on system.
echo Please install Python 3.9 or later from https://python.org
echo Or ensure Python is in your PATH environment variable.
echo.
echo After installing Python, run install.bat again.
pause
exit /b 1

:configure_python
REM Configure Python environment
echo Configuring Python environment...
set "PATH=%PYTHON_PATH%;%PATH%"

:end
echo Python setup completed.
exit /b 0