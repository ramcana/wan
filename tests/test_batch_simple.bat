@echo off
setlocal enabledelayedexpansion

REM ========================================
REM Simple Batch File Tests
REM Tests for start_both_servers.bat
REM ========================================

set "SCRIPT_DIR=%~dp0.."
set "BATCH_FILE=%SCRIPT_DIR%\start_both_servers.bat"
set "TEST_COUNT=0"
set "PASS_COUNT=0"

echo.
echo ========================================
echo    Simple Batch File Test Suite
echo ========================================
echo.

REM Verify batch file exists
if not exist "%BATCH_FILE%" (
    echo [ERROR] Batch file not found: %BATCH_FILE%
    exit /b 1
)

REM Test 1: Help option (non-interactive)
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing help option...
echo exit | "%BATCH_FILE%" --help > temp_help.txt 2>&1
findstr /c:"WAN22 Server Startup Manager - Help" temp_help.txt >nul
if errorlevel 1 (
    echo [FAIL] Help option test failed
) else (
    echo [PASS] Help option test passed
    set /a PASS_COUNT+=1
)
del temp_help.txt >nul 2>&1

REM Test 2: Basic argument recognition
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing basic mode help...
echo exit | "%BATCH_FILE%" --basic --help > temp_basic.txt 2>&1
findstr /c:"Basic Mode Features" temp_basic.txt >nul
if errorlevel 1 (
    echo [FAIL] Basic mode test failed
) else (
    echo [PASS] Basic mode test passed
    set /a PASS_COUNT+=1
)
del temp_basic.txt >nul 2>&1

REM Test 3: Verbose flag recognition
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing verbose flag...
echo exit | "%BATCH_FILE%" --verbose --help > temp_verbose.txt 2>&1
findstr /c:"WAN22 Server Startup Manager - Help" temp_verbose.txt >nul
if errorlevel 1 (
    echo [FAIL] Verbose flag test failed
) else (
    echo [PASS] Verbose flag test passed
    set /a PASS_COUNT+=1
)
del temp_verbose.txt >nul 2>&1

REM Test 4: Port argument recognition
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing port arguments in help...
echo exit | "%BATCH_FILE%" --help > temp_ports.txt 2>&1
findstr /c:"--backend-port" temp_ports.txt >nul && findstr /c:"--frontend-port" temp_ports.txt >nul
if errorlevel 1 (
    echo [FAIL] Port arguments test failed
) else (
    echo [PASS] Port arguments test passed
    set /a PASS_COUNT+=1
)
del temp_ports.txt >nul 2>&1

REM Display results
echo.
echo ========================================
echo    Test Results Summary
echo ========================================
echo Total Tests: %TEST_COUNT%
echo Passed: %PASS_COUNT%
echo Failed: !TEST_COUNT! - !PASS_COUNT!
echo.

if %PASS_COUNT% equ %TEST_COUNT% (
    echo [SUCCESS] All tests passed!
    exit /b 0
) else (
    echo [FAILURE] Some tests failed
    exit /b 1
)