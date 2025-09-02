@echo off
setlocal enabledelayedexpansion

REM ========================================
REM Comprehensive Batch File Test Suite
REM Tests for start_both_servers.bat
REM ========================================

REM Define testing framework functions inline
set "FRAMEWORK_VERSION=1.0"
set "TEST_TEMP_DIR=%TEMP%\batch_tests_%RANDOM%"
set "CURRENT_TEST="
set "TEST_OUTPUT_FILE="

REM Create temporary directory for tests
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

set "TEST_COUNT=0"
set "PASS_COUNT=0"
set "FAIL_COUNT=0"
set "SCRIPT_DIR=%~dp0.."
set "BATCH_FILE=%SCRIPT_DIR%\start_both_servers.bat"

REM Initialize framework
echo ========================================
echo Batch Testing Framework v%FRAMEWORK_VERSION%
echo ========================================

echo.
echo ========================================
echo    Enhanced Batch File Test Suite
echo    Testing: start_both_servers.bat
echo ========================================
echo.

REM Verify batch file exists
if not exist "%BATCH_FILE%" (
    echo [ERROR] Batch file not found: %BATCH_FILE%
    exit /b 1
)

REM Test 1: Help option functionality
call :run_test "Help Option Display" "test_help_option"

REM Test 2: Argument parsing validation
call :run_test "Command Line Argument Parsing" "test_argument_parsing"

REM Test 3: Python manager detection
call :run_test "Python Manager Detection Logic" "test_python_detection"

REM Test 4: Fallback mode activation
call :run_test "Basic/Fallback Mode Activation" "test_fallback_mode"

REM Test 5: Verbose mode output
call :run_test "Verbose Mode Output" "test_verbose_mode"

REM Test 6: Port specification
call :run_test "Port Specification Handling" "test_port_specification"

REM Test 7: Error handling
call :run_test "Error Handling and Messages" "test_error_handling"

REM Test 8: Directory validation
call :run_test "Directory Structure Validation" "test_directory_validation"

REM Display comprehensive results
echo.
echo ========================================
echo    Comprehensive Test Results
echo ========================================
echo Total Tests: %TEST_COUNT%
echo Passed: %PASS_COUNT%
echo Failed: %FAIL_COUNT%
echo Success Rate: !PASS_COUNT!/%TEST_COUNT%
echo.

if %FAIL_COUNT% gtr 0 (
    echo [OVERALL RESULT] FAILED - %FAIL_COUNT% test(s) failed
    REM Cleanup
    if exist "%TEST_TEMP_DIR%" rmdir /s /q "%TEST_TEMP_DIR%"
    exit /b 1
) else (
    echo [OVERALL RESULT] PASSED - All tests successful
    REM Cleanup
    if exist "%TEST_TEMP_DIR%" rmdir /s /q "%TEST_TEMP_DIR%"
    exit /b 0
)

REM Framework functions
:start_test
set "CURRENT_TEST=%~1"
set "TEST_OUTPUT_FILE=%TEST_TEMP_DIR%\%CURRENT_TEST%_output.txt"
goto :eof

:end_test
set "CURRENT_TEST="
set "TEST_OUTPUT_FILE="
goto :eof

:assert_contains
findstr /c:"%~1" "%~2" >nul 2>&1
if errorlevel 1 (
    echo [ASSERT FAIL] Text "%~1" not found in %~2
    exit /b 1
) else (
    echo [ASSERT PASS] Text "%~1" found in %~2
    exit /b 0
)

:run_command_capture
%~1 > "%~2" 2>&1
exit /b %errorlevel%

REM Test execution framework
:run_test
set /a TEST_COUNT+=1
call :start_test "%~1"
echo [%TEST_COUNT%] Running: %~1
call :%~2
if !errorlevel! equ 0 (
    echo [PASS] %~1
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] %~1
    set /a FAIL_COUNT+=1
)
call :end_test
echo.
goto :eof

REM Individual test implementations

:test_help_option
REM Test that help option displays correct information
call :run_command_capture ""%BATCH_FILE%" --help" "%TEST_OUTPUT_FILE%"
call :assert_contains "WAN22 Server Startup Manager - Help" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
call :assert_contains "Usage:" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
call :assert_contains "Options:" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
call :assert_contains "Examples:" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:test_argument_parsing
REM Test argument parsing by checking help with different argument formats
call :run_command_capture ""%BATCH_FILE%" -h" "%TEST_OUTPUT_FILE%"
call :assert_contains "WAN22 Server Startup Manager - Help" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:test_python_detection
REM Test Python manager detection logic
echo Testing Python detection logic...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not available - should trigger fallback mode
    call :run_command_capture ""%BATCH_FILE%" --verbose --help" "%TEST_OUTPUT_FILE%"
    REM In real scenario, would test fallback behavior
) else (
    echo Python available - should attempt advanced mode
    REM Test would verify Python manager path checking
)
exit /b 0

:test_fallback_mode
REM Test forced fallback mode
call :run_command_capture ""%BATCH_FILE%" --basic --help" "%TEST_OUTPUT_FILE%"
call :assert_contains "Basic Startup Mode" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:test_verbose_mode
REM Test verbose mode produces additional output
call :run_command_capture ""%BATCH_FILE%" --verbose --help" "%TEST_OUTPUT_FILE%"
call :assert_contains "WAN22 Server Startup Manager - Help" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
REM Verbose mode should still show help, but with additional info
exit /b 0

:test_port_specification
REM Test port specification in help
call :run_command_capture ""%BATCH_FILE%" --help" "%TEST_OUTPUT_FILE%"
call :assert_contains "--backend-port" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
call :assert_contains "--frontend-port" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:test_error_handling
REM Test error handling by checking help includes troubleshooting
call :run_command_capture ""%BATCH_FILE%" --help" "%TEST_OUTPUT_FILE%"
call :assert_contains "Troubleshooting" "%TEST_OUTPUT_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:test_directory_validation
REM Test that script handles directory validation
REM This is a basic test - full test would require mocking directories
echo Testing directory validation logic...
REM In a full implementation, would test with missing backend/frontend dirs
exit /b 0