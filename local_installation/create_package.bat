@echo off
REM WAN2.2 Installation Package Creator
REM This script creates distributable installation packages

setlocal enabledelayedexpansion

echo ========================================
echo WAN2.2 Installation Package Creator
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
set VERSION=1.0.0
set PACKAGE_NAME=WAN22-Installer
set VERIFY_PACKAGE=true
set LOG_LEVEL=INFO

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--version" (
    set VERSION=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--name" (
    set PACKAGE_NAME=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-verify" (
    set VERIFY_PACKAGE=false
    shift
    goto :parse_args
)
if "%~1"=="--verbose" (
    set LOG_LEVEL=DEBUG
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_help
)
shift
goto :parse_args

:args_done

echo Creating package: %PACKAGE_NAME% v%VERSION%
echo.

REM Create the package
echo [1/3] Creating installation package...
if "%VERIFY_PACKAGE%"=="true" (
    python scripts\package_installer.py create --version %VERSION% --name %PACKAGE_NAME% --log-level %LOG_LEVEL% --verify
) else (
    python scripts\package_installer.py create --version %VERSION% --name %PACKAGE_NAME% --log-level %LOG_LEVEL%
)

if errorlevel 1 (
    echo.
    echo Error: Package creation failed
    echo Check packaging.log for details
    pause
    exit /b 1
)

echo.
echo [2/3] Listing created packages...
python scripts\package_installer.py list packages

echo.
echo [3/3] Package creation complete!
echo.
echo Package location: dist\%PACKAGE_NAME%-v%VERSION%.zip
echo Integrity file: dist\%PACKAGE_NAME%-v%VERSION%.zip.integrity.json
echo Checksum file: dist\%PACKAGE_NAME%-v%VERSION%.zip.sha256
echo.
echo To distribute this package:
echo 1. Share the .zip file with users
echo 2. Include the .integrity.json file for verification
echo 3. Users can run install.bat from the extracted package
echo.

pause
exit /b 0

:show_help
echo Usage: create_package.bat [options]
echo.
echo Options:
echo   --version VERSION    Package version (default: 1.0.0)
echo   --name NAME         Package name (default: WAN22-Installer)
echo   --no-verify         Skip package verification
echo   --verbose           Enable verbose logging
echo   --help              Show this help message
echo.
echo Examples:
echo   create_package.bat --version 1.2.0
echo   create_package.bat --version 2.0.0 --name WAN22-Pro-Installer
echo   create_package.bat --verbose --no-verify
echo.
pause
exit /b 0