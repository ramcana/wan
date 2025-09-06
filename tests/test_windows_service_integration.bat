@echo off
setlocal enabledelayedexpansion

REM ========================================
REM Windows Service Integration Tests
REM Tests Windows-specific features
REM ========================================

set "SCRIPT_DIR=%~dp0.."
set "BATCH_FILE=%SCRIPT_DIR%\start_both_servers.bat"
set "TEST_COUNT=0"
set "PASS_COUNT=0"

echo.
echo ========================================
echo    Windows Service Integration Tests
echo ========================================
echo.

REM Test 1: Windows optimization detection
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing Windows optimization detection...
echo exit | "%BATCH_FILE%" --verbose > temp_windows.txt 2>&1
findstr /c:"Checking Windows optimizations" temp_windows.txt >nul
if errorlevel 1 (
    echo [FAIL] Windows optimization detection failed
) else (
    echo [PASS] Windows optimization detection passed
    set /a PASS_COUNT+=1
)
del temp_windows.txt >nul 2>&1

REM Test 2: Administrator privilege detection
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing administrator privilege detection...
echo exit | "%BATCH_FILE%" --verbose > temp_admin.txt 2>&1
findstr /c:"Not running as administrator" temp_admin.txt >nul
if errorlevel 1 (
    echo [FAIL] Administrator detection failed
) else (
    echo [PASS] Administrator detection passed
    set /a PASS_COUNT+=1
)
del temp_admin.txt >nul 2>&1

REM Test 3: Firewall status checking
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing firewall status checking...
echo exit | "%BATCH_FILE%" --verbose > temp_firewall.txt 2>&1
findstr /c:"Checking Windows Firewall status" temp_firewall.txt >nul
if errorlevel 1 (
    echo [FAIL] Firewall status checking failed
) else (
    echo [PASS] Firewall status checking passed
    set /a PASS_COUNT+=1
)
del temp_firewall.txt >nul 2>&1

REM Test 4: Python Windows utilities availability
set /a TEST_COUNT+=1
echo [%TEST_COUNT%] Testing Python Windows utilities...
python -c "import sys; sys.path.insert(0, r'scripts'); from startup_manager.windows_utils import WindowsOptimizer; print('Windows utilities available')" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python Windows utilities not available
) else (
    echo [PASS] Python Windows utilities available
    set /a PASS_COUNT+=1
)

REM Display results
echo.
echo ========================================
echo    Windows Integration Test Results
echo ========================================
echo Total Tests: %TEST_COUNT%
echo Passed: %PASS_COUNT%
echo Failed: !TEST_COUNT! - !PASS_COUNT!
echo.

if %PASS_COUNT% equ %TEST_COUNT% (
    echo [SUCCESS] All Windows integration tests passed!
    exit /b 0
) else (
    echo [FAILURE] Some Windows integration tests failed
    exit /b 1
)