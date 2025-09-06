@echo off
REM WAN2.2 Release Preparation Tool
REM This script prepares complete releases for distribution

setlocal enabledelayedexpansion

echo ========================================
echo WAN2.2 Release Preparation Tool
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Set default values
set VERSION=
set RELEASE_NOTES=
set SKIP_TESTS=false
set SKIP_VALIDATION=false
set VERBOSE=false
set COMMAND=prepare

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--version" (
    set VERSION=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--release-notes" (
    set RELEASE_NOTES=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--skip-tests" (
    set SKIP_TESTS=true
    shift
    goto :parse_args
)
if "%~1"=="--skip-validation" (
    set SKIP_VALIDATION=true
    shift
    goto :parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto :parse_args
)
if "%~1"=="--list" (
    set COMMAND=list
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_help
)
shift
goto :parse_args

:args_done

REM Handle different commands
if "%COMMAND%"=="list" goto :list_releases

REM Prepare release command
if "%VERSION%"=="" (
    echo Error: Version is required for release preparation
    echo Use --version to specify the release version
    echo.
    goto :show_help
)

echo Preparing release v%VERSION%...
echo.

REM Build the Python command
set PYTHON_CMD=python scripts\prepare_release.py prepare --version %VERSION%

if not "%RELEASE_NOTES%"=="" (
    set PYTHON_CMD=!PYTHON_CMD! --release-notes "%RELEASE_NOTES%"
)

if "%SKIP_TESTS%"=="true" (
    set PYTHON_CMD=!PYTHON_CMD! --skip-tests
)

if "%SKIP_VALIDATION%"=="true" (
    set PYTHON_CMD=!PYTHON_CMD! --no-validate
)

if "%VERBOSE%"=="true" (
    set PYTHON_CMD=!PYTHON_CMD! --verbose
)

echo [1/4] Starting release preparation...
echo Command: !PYTHON_CMD!
echo.

REM Execute the release preparation
!PYTHON_CMD!

if errorlevel 1 (
    echo.
    echo Error: Release preparation failed
    echo Check release_preparation.log for details
    pause
    exit /b 1
)

echo.
echo [2/4] Release preparation completed successfully!
echo.

echo [3/4] Checking release artifacts...
if exist "dist\release-v%VERSION%" (
    echo ✓ Release directory created: dist\release-v%VERSION%
    
    REM List the contents
    echo.
    echo Release contents:
    dir /b "dist\release-v%VERSION%"
) else (
    echo ✗ Release directory not found
)

echo.
echo [4/4] Release v%VERSION% is ready!
echo.
echo Next steps:
echo 1. Review the release artifacts in dist\release-v%VERSION%
echo 2. Test the installer package on target systems
echo 3. Upload the release package for distribution
echo 4. Update release documentation
echo.

pause
exit /b 0

:list_releases
echo Listing available releases...
echo.

set LIST_CMD=python scripts\prepare_release.py list

if "%VERBOSE%"=="true" (
    set LIST_CMD=!LIST_CMD! --verbose
)

!LIST_CMD!

if errorlevel 1 (
    echo.
    echo Error: Failed to list releases
    pause
    exit /b 1
)

echo.
pause
exit /b 0

:show_help
echo Usage: prepare_release.bat [options]
echo.
echo Commands:
echo   (default)           Prepare a new release
echo   --list              List available releases
echo.
echo Options for release preparation:
echo   --version VERSION   Release version (required, e.g., 1.0.0)
echo   --release-notes "NOTES"  Release notes text
echo   --skip-tests        Skip compatibility tests
echo   --skip-validation   Skip release validation
echo   --verbose           Enable verbose output
echo   --help              Show this help message
echo.
echo Examples:
echo   prepare_release.bat --version 1.0.0
echo   prepare_release.bat --version 1.1.0 --release-notes "Bug fixes and improvements"
echo   prepare_release.bat --version 2.0.0 --skip-tests --verbose
echo   prepare_release.bat --list --verbose
echo.
pause
exit /b 0