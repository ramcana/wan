@echo off
REM Script to automatically fix common port configuration issues

echo Applying automatic port configuration fixes...
cd /d "%~dp0"
python fix-port-issues.py

if %ERRORLEVEL% EQU 0 (
    echo Port fixes completed successfully.
) else (
    echo Port fixes applied. Please review changes and validate.
    exit /b %ERRORLEVEL%
)