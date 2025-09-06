@echo off
title WAN2.2 UI - Test Mode
echo ========================================
echo WAN2.2 User Interface - Test Mode
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

REM Check if test files exist
if not exist "test_ui_integration.py" (
    echo Error: UI test files not found!
    echo Please ensure test files exist in the root directory.
    echo.
    pause
    exit /b 1
)

REM Run UI tests instead of launching the full application
echo Running WAN2.2 Gradio UI Tests...
echo.
echo ========================================
echo  WAN2.2 Gradio UI Test Suite
echo ========================================
echo.
echo This will test the UI structure and functionality
echo without requiring GPU dependencies.
echo.

REM Run the UI integration tests
python -m pytest test_ui_integration.py -v

REM Check test results
if errorlevel 1 (
    echo.
    echo ========================================
    echo Some tests failed
    echo ========================================
    echo.
    echo Please check the test output above for details.
    echo.
) else (
    echo.
    echo ========================================
    echo All UI tests passed successfully!
    echo ========================================
    echo.
    echo The Gradio UI structure is working correctly.
    echo For full functionality, ensure GPU dependencies are installed.
    echo.
)

echo Press any key to close.
pause >nul