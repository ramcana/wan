@echo off
setlocal enabledelayedexpansion

REM ========================================
REM WAN22 Enhanced Server Startup Manager v2.0
REM Intelligent batch file with Python integration
REM Backward compatible with legacy workflows
REM ========================================

REM Initialize variables
set "SCRIPT_DIR=%~dp0"
set "PYTHON_MANAGER=%SCRIPT_DIR%scripts\startup_manager.py"
set "FALLBACK_MODE=0"
set "VERBOSE_MODE=0"
set "DEBUG_MODE=0"
set "BACKEND_PORT="
set "FRONTEND_PORT="
set "FORCE_BASIC=0"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--verbose" set "VERBOSE_MODE=1"
if /i "%~1"=="-v" set "VERBOSE_MODE=1"
if /i "%~1"=="--debug" set "DEBUG_MODE=1"
if /i "%~1"=="-d" set "DEBUG_MODE=1"
if /i "%~1"=="--backend-port" (
    shift
    set "BACKEND_PORT=%~1"
)
if /i "%~1"=="--frontend-port" (
    shift
    set "FRONTEND_PORT=%~1"
)
if /i "%~1"=="--basic" set "FORCE_BASIC=1"
if /i "%~1"=="--fallback" set "FORCE_BASIC=1"
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
shift
goto :parse_args

:args_done

REM Display banner
echo.
echo ========================================
echo    WAN22 Server Startup Manager v2.0
echo    Intelligent Server Management
echo ========================================
echo.

REM Check if forced to use basic mode
if "%FORCE_BASIC%"=="1" (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Forced basic mode requested
    goto :basic_startup
)

REM Windows-specific optimizations
call :check_windows_optimizations

REM Check for Python startup manager availability
if "%VERBOSE_MODE%"=="1" echo [INFO] Checking for Python startup manager...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    if "%VERBOSE_MODE%"=="1" echo [WARN] Python not found in PATH
    set "FALLBACK_MODE=1"
    goto :check_fallback
)

REM Check if startup manager script exists
if not exist "%PYTHON_MANAGER%" (
    if "%VERBOSE_MODE%"=="1" echo [WARN] Python startup manager not found at: %PYTHON_MANAGER%
    set "FALLBACK_MODE=1"
    goto :check_fallback
)

REM Check if startup manager dependencies are available
if "%VERBOSE_MODE%"=="1" echo [INFO] Validating startup manager dependencies...
python -c "import sys; sys.path.insert(0, r'%SCRIPT_DIR%scripts'); from startup_manager import cli" >nul 2>&1
if errorlevel 1 (
    if "%VERBOSE_MODE%"=="1" echo [WARN] Startup manager dependencies not available
    set "FALLBACK_MODE=1"
    goto :check_fallback
)

REM Use Python startup manager
if "%VERBOSE_MODE%"=="1" echo [INFO] Using advanced Python startup manager
echo Using intelligent startup manager...
echo.

REM Build Python command with arguments
set "PYTHON_CMD=python "%PYTHON_MANAGER%""
if "%VERBOSE_MODE%"=="1" set "PYTHON_CMD=!PYTHON_CMD! --verbose"
if "%DEBUG_MODE%"=="1" set "PYTHON_CMD=!PYTHON_CMD! --debug"
if not "%BACKEND_PORT%"=="" set "PYTHON_CMD=!PYTHON_CMD! --backend-port %BACKEND_PORT%"
if not "%FRONTEND_PORT%"=="" set "PYTHON_CMD=!PYTHON_CMD! --frontend-port %FRONTEND_PORT%"

REM Execute Python startup manager
if "%DEBUG_MODE%"=="1" echo [DEBUG] Executing: !PYTHON_CMD!
!PYTHON_CMD!

REM Check Python startup manager result
if errorlevel 1 (
    echo.
    echo [ERROR] Python startup manager failed with exit code %errorlevel%
    echo Falling back to basic startup mode...
    echo.
    set "FALLBACK_MODE=1"
    goto :basic_startup
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Python startup manager completed successfully
    goto :end
)

:check_fallback
if "%FALLBACK_MODE%"=="1" (
    echo Python startup manager not available - using basic mode
    echo.
    goto :basic_startup
)
goto :end

:basic_startup
echo ========================================
echo    Basic Startup Mode
echo ========================================
echo.

REM Set default ports if not specified
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=3000"

if "%VERBOSE_MODE%"=="1" (
    echo [INFO] Starting servers in basic mode
    echo [INFO] Backend port: %BACKEND_PORT%
    echo [INFO] Frontend port: %FRONTEND_PORT%
    echo.
)

REM Check if backend directory exists
if not exist "%SCRIPT_DIR%backend" (
    echo [ERROR] Backend directory not found: %SCRIPT_DIR%backend
    echo Please ensure you're running this script from the project root directory.
    goto :error_exit
)

REM Check if frontend directory exists
if not exist "%SCRIPT_DIR%frontend" (
    echo [ERROR] Frontend directory not found: %SCRIPT_DIR%frontend
    echo Please ensure you're running this script from the project root directory.
    goto :error_exit
)

echo Starting FastAPI Backend Server on port %BACKEND_PORT%...
if "%BACKEND_PORT%"=="8000" (
    start "WAN22 Backend" cmd /k "cd /d "%SCRIPT_DIR%backend" && python start_server.py"
) else (
    start "WAN22 Backend" cmd /k "cd /d "%SCRIPT_DIR%backend" && python start_server.py --port %BACKEND_PORT%"
)

echo Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak > nul

echo Starting React Frontend Server on port %FRONTEND_PORT%...
if "%FRONTEND_PORT%"=="3000" (
    start "WAN22 Frontend" cmd /k "cd /d "%SCRIPT_DIR%frontend" && npm run dev"
) else (
    REM Note: React dev server port configuration would need to be handled differently
    REM This is a simplified version for basic mode
    start "WAN22 Frontend" cmd /k "cd /d "%SCRIPT_DIR%frontend" && npm run dev"
)

echo.
echo ========================================
echo    Servers Starting...
echo ========================================
echo.
echo Backend Server:  http://localhost:%BACKEND_PORT%
echo Frontend Server: http://localhost:%FRONTEND_PORT%
echo API Documentation: http://localhost:%BACKEND_PORT%/docs
echo.
echo [INFO] Servers are starting in separate windows
echo [INFO] Close those windows to stop the servers
echo [INFO] Check the server windows for any error messages
echo.

if "%VERBOSE_MODE%"=="1" (
    echo [INFO] Basic startup mode completed
    echo [INFO] For advanced features, install Python dependencies
    echo [INFO] Run 'pip install -r requirements.txt' in the project root
    echo.
)

goto :end

:show_help
echo.
echo WAN22 Server Startup Manager - Help
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --help, -h           Show this help message
echo   --verbose, -v        Enable verbose output
echo   --debug, -d          Enable debug output
echo   --basic, --fallback  Force basic startup mode
echo   --backend-port PORT  Specify backend port (default: 8000)
echo   --frontend-port PORT Specify frontend port (default: 3000)
echo.
echo Examples:
echo   %~nx0                          Start with default settings
echo   %~nx0 --verbose                Start with verbose output
echo   %~nx0 --backend-port 8080      Start backend on port 8080
echo   %~nx0 --basic                  Force basic mode (no Python manager)
echo.
echo Advanced Features (Python Manager):
echo   - Automatic port conflict resolution
echo   - Environment validation
echo   - Intelligent error recovery
echo   - Process health monitoring
echo   - Detailed logging and diagnostics
echo.
echo Basic Mode Features:
echo   - Simple server startup
echo   - Manual port specification
echo   - Fallback when Python manager unavailable
echo.
goto :end

:error_exit
echo.
echo [ERROR] Startup failed. Please check the error messages above.
echo.
echo Troubleshooting:
echo   1. Ensure you're in the correct project directory
echo   2. Check that backend and frontend directories exist
echo   3. Verify Python and Node.js are installed
echo   4. Try running with --verbose for more information
echo.
pause
exit /b 1

:check_windows_optimizations
REM Check and apply Windows-specific optimizations
if "%VERBOSE_MODE%"=="1" echo [INFO] Checking Windows optimizations...

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Not running as administrator
    set "ADMIN_RIGHTS=0"
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Running with administrator privileges
    set "ADMIN_RIGHTS=1"
)

REM Check Windows Defender Firewall status
if "%VERBOSE_MODE%"=="1" echo [INFO] Checking Windows Firewall status...
netsh advfirewall show allprofiles state >nul 2>&1
if errorlevel 1 (
    if "%VERBOSE_MODE%"=="1" echo [WARN] Could not check firewall status
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Firewall status checked successfully
)

REM Offer to apply optimizations if admin rights available
if "%ADMIN_RIGHTS%"=="1" (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Administrator rights available for optimizations
    call :apply_windows_optimizations
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Limited optimizations available without admin rights
)

goto :eof

:apply_windows_optimizations
REM Apply Windows-specific optimizations with admin rights
if "%VERBOSE_MODE%"=="1" echo [INFO] Applying Windows optimizations...

REM Add firewall exceptions for development ports
call :add_firewall_exception 3000 "WAN22 Frontend Development"
call :add_firewall_exception 8000 "WAN22 Backend Development"

REM Check for Python firewall exception
python -c "import sys; print(sys.executable)" >temp_python_path.txt 2>nul
if exist temp_python_path.txt (
    set /p PYTHON_PATH=<temp_python_path.txt
    call :add_program_firewall_exception "!PYTHON_PATH!" "WAN22 Python Development"
    del temp_python_path.txt
)

goto :eof

:add_firewall_exception
REM Add firewall exception for a port
REM Usage: call :add_firewall_exception PORT RULE_NAME
set "PORT=%~1"
set "RULE_NAME=%~2"

if "%VERBOSE_MODE%"=="1" echo [INFO] Adding firewall exception for port %PORT%...

netsh advfirewall firewall show rule name="%RULE_NAME%" >nul 2>&1
if errorlevel 1 (
    REM Rule doesn't exist, create it
    netsh advfirewall firewall add rule name="%RULE_NAME%" dir=in action=allow protocol=TCP localport=%PORT% enable=yes >nul 2>&1
    if errorlevel 1 (
        if "%VERBOSE_MODE%"=="1" echo [WARN] Failed to add firewall exception for port %PORT%
    ) else (
        if "%VERBOSE_MODE%"=="1" echo [INFO] Added firewall exception for port %PORT%
    )
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Firewall exception already exists for port %PORT%
)

goto :eof

:add_program_firewall_exception
REM Add firewall exception for a program
REM Usage: call :add_program_firewall_exception "PROGRAM_PATH" "RULE_NAME"
set "PROGRAM_PATH=%~1"
set "RULE_NAME=%~2"

if "%VERBOSE_MODE%"=="1" echo [INFO] Adding firewall exception for program: %PROGRAM_PATH%...

netsh advfirewall firewall show rule name="%RULE_NAME%" >nul 2>&1
if errorlevel 1 (
    REM Rule doesn't exist, create it
    netsh advfirewall firewall add rule name="%RULE_NAME%" dir=in action=allow program="%PROGRAM_PATH%" enable=yes >nul 2>&1
    if errorlevel 1 (
        if "%VERBOSE_MODE%"=="1" echo [WARN] Failed to add firewall exception for program
    ) else (
        if "%VERBOSE_MODE%"=="1" echo [INFO] Added firewall exception for program
    )
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Firewall exception already exists for program
)

goto :eof

:request_elevation
REM Request UAC elevation if needed
if "%VERBOSE_MODE%"=="1" echo [INFO] Requesting administrator elevation...

REM Check if we're already elevated
net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo    Administrator Rights Required
    echo ========================================
    echo.
    echo Some optimizations require administrator privileges.
    echo The script will now restart with elevated permissions.
    echo.
    echo Please click "Yes" in the UAC prompt to continue.
    echo.
    
    REM Restart with elevation
    powershell -Command "Start-Process '%~f0' -ArgumentList '%*' -Verb RunAs"
    exit /b 0
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] Already running with administrator privileges
)

goto :eof

:setup_windows_service
REM Set up Windows service for background management
if "%VERBOSE_MODE%"=="1" echo [INFO] Setting up Windows service...

REM Check if service already exists
sc query "WAN22Service" >nul 2>&1
if errorlevel 1 (
    REM Service doesn't exist, offer to create it
    echo.
    echo Would you like to install WAN22 as a Windows service?
    echo This allows automatic startup and background management.
    echo.
    choice /C YN /M "Install Windows service"
    if errorlevel 2 goto :eof
    
    REM Use Python to set up the service
    if exist "%PYTHON_MANAGER%" (
        python "%PYTHON_MANAGER%" --setup-service
        if errorlevel 1 (
            echo [WARN] Failed to set up Windows service
        ) else (
            echo [INFO] Windows service installed successfully
        )
    )
) else (
    if "%VERBOSE_MODE%"=="1" echo [INFO] WAN22 Windows service already installed
)

goto :eof

:end
echo.
if "%VERBOSE_MODE%"=="1" echo [INFO] Startup script completed
echo Press any key to close this window...
pause > nul
exit /b 0