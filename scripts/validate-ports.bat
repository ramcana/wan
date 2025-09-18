@echo off
REM Port validation script for WAN2.2 Video Generation System

echo Validating port configurations...
cd /d "%~dp0"
python validate-ports.py

if %ERRORLEVEL% EQU 0 (
    echo Port validation completed successfully.
) else (
    echo Port validation found issues. Please review the report above.
    exit /b %ERRORLEVEL%
)