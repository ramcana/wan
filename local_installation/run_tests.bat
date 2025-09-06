@echo off
REM WAN2.2 Local Installation Automated Test Runner
REM This batch file runs the comprehensive test suite for the local installation system

echo ================================================================================
echo WAN2.2 Local Installation Automated Test Framework
echo ================================================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "scripts" (
    echo ERROR: Please run this script from the local_installation directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Set default arguments
set SUITE=all
set VERBOSE=
set HTML_REPORT=
set NO_CLEANUP=
set OUTPUT_DIR=test_results

REM Parse command line arguments
:parse_args
if "%1"=="" goto run_tests
if "%1"=="--unit" set SUITE=unit
if "%1"=="--integration" set SUITE=integration
if "%1"=="--hardware" set SUITE=hardware
if "%1"=="--verbose" set VERBOSE=--verbose
if "%1"=="--html" set HTML_REPORT=--html-report
if "%1"=="--no-cleanup" set NO_CLEANUP=--no-cleanup
if "%1"=="--help" goto show_help
shift
goto parse_args

:show_help
echo Usage: run_tests.bat [options]
echo.
echo Options:
echo   --unit         Run only unit tests
echo   --integration  Run only integration tests
echo   --hardware     Run only hardware simulation tests
echo   --verbose      Enable verbose output
echo   --html         Generate HTML report
echo   --no-cleanup   Don't cleanup test files
echo   --help         Show this help message
echo.
echo Examples:
echo   run_tests.bat                    # Run all tests
echo   run_tests.bat --unit --verbose   # Run unit tests with verbose output
echo   run_tests.bat --html             # Run all tests and generate HTML report
pause
exit /b 0

:run_tests
echo Running test suite: %SUITE%
echo Output directory: %OUTPUT_DIR%
echo.

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Run the tests
python run_automated_tests.py --suite %SUITE% %VERBOSE% %HTML_REPORT% %NO_CLEANUP% --output-dir "%OUTPUT_DIR%"

set TEST_EXIT_CODE=%ERRORLEVEL%

echo.
echo ================================================================================
echo Test execution completed with exit code: %TEST_EXIT_CODE%
echo ================================================================================

REM Interpret exit codes
if %TEST_EXIT_CODE%==0 (
    echo ✅ All tests passed successfully!
    echo The installation system is ready for deployment.
) else if %TEST_EXIT_CODE%==1 (
    echo ⚠️  Some tests failed, but core functionality appears to work.
    echo Review the test report for details.
) else if %TEST_EXIT_CODE%==2 (
    echo ❌ Many tests failed. Please review and fix issues before deployment.
    echo Check the test report for detailed error information.
) else if %TEST_EXIT_CODE%==130 (
    echo ⚠️  Test execution was interrupted by user.
) else (
    echo ❌ Test framework encountered an error.
    echo Check the console output for details.
)

echo.
echo Test reports saved to: %OUTPUT_DIR%
if exist "%OUTPUT_DIR%\test_report.html" (
    echo 🌐 HTML report: %OUTPUT_DIR%\test_report.html
)
if exist "%OUTPUT_DIR%\test_report.json" (
    echo 📊 JSON report: %OUTPUT_DIR%\test_report.json
)

echo.
echo Press any key to exit...
pause >nul

exit /b %TEST_EXIT_CODE%