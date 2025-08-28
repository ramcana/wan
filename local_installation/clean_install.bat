@echo off
REM Clean installation script - removes previous installation state and starts fresh

echo 🧹 Cleaning previous installation state...

REM Remove installation state file
if exist "logs\installation_state.json" (
    del "logs\installation_state.json"
    echo ✅ Removed installation state file
)

REM Clear old log files (optional - keeps for debugging)
REM if exist "logs\*.log" (
REM     del "logs\*.log"
REM     echo ✅ Cleared old log files
REM )

echo.
echo 🚀 Starting fresh installation...
echo.

REM Start fresh installation without models (fastest test)
install.bat --force-reinstall --skip-models --verbose

echo.
echo ✅ Clean installation completed!
pause