@echo off
REM Kiro CLI - Unified Project Quality Tool
REM Windows batch wrapper for the unified CLI tool

set SCRIPT_DIR=%~dp0
set TOOLS_DIR=%SCRIPT_DIR%..

python "%SCRIPT_DIR%cli.py" %*