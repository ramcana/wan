@echo off
REM System port validation script for WAN2.2 Video Generation System

echo Validating system port configurations...
cd /d "%~dp0"
python validate-system-ports.py

if %ERRORLEVEL% EQU 0 (
    echo System port validation completed successfully.
) else (
    echo System port validation found issues. Please review the report above.
    exit /b %ERRORLEVEL%
)