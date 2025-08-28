@echo off
REM WAN2.2 Installation Management CLI
REM Wrapper for the Python CLI utility

setlocal enabledelayedexpansion

set "INSTALL_DIR=%~dp0"
set "SCRIPTS_DIR=%INSTALL_DIR%scripts"
set "PYTHON_CMD=python"

REM Check if Python is available
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    REM Try embedded Python
    if exist "%INSTALL_DIR%python\python.exe" (
        set "PYTHON_CMD=%INSTALL_DIR%python\python.exe"
    ) else (
        echo Error: Python not found. Please run install.bat first.
        exit /b 1
    )
)

REM Run the CLI utility
%PYTHON_CMD% "%SCRIPTS_DIR%\installation_cli.py" --installation-path "%INSTALL_DIR%" %*

exit /b %errorlevel%