@echo off
REM ============================================================================
REM WAN2.2 Local Installation System
REM Automated installation with hardware detection and optimization
REM ============================================================================

setlocal enabledelayedexpansion

REM Set console properties for better display
title WAN2.2 Installation System
color 0F
mode con: cols=80 lines=30

REM Initialize variables
set "INSTALL_DIR=%~dp0"
set "LOGS_DIR=%INSTALL_DIR%logs"
set "SCRIPTS_DIR=%INSTALL_DIR%scripts"
set "PYTHON_CMD=python"
set "INSTALLATION_LOG=%LOGS_DIR%\installation.log"
set "ERROR_LOG=%LOGS_DIR%\error.log"
set "INSTALL_SUCCESS=0"

REM Parse command line arguments
set "SILENT_MODE=0"
set "VERBOSE_MODE=0"
set "DEV_MODE=0"
set "SKIP_MODELS=0"
set "FORCE_REINSTALL=0"
set "DRY_RUN=0"
set "CUSTOM_PATH="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--silent" set "SILENT_MODE=1"
if /i "%~1"=="--verbose" set "VERBOSE_MODE=1"
if /i "%~1"=="--dev-mode" set "DEV_MODE=1"
if /i "%~1"=="--skip-models" set "SKIP_MODELS=1"
if /i "%~1"=="--force-reinstall" set "FORCE_REINSTALL=1"
if /i "%~1"=="--dry-run" set "DRY_RUN=1"
if /i "%~1"=="--custom-path" (
    shift
    set "CUSTOM_PATH=%~1"
)
if /i "%~1"=="--help" goto show_help
shift
goto parse_args

:args_done

REM Create logs directory
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

REM Initialize log files
echo [%date% %time%] WAN2.2 Installation Started > "%INSTALLATION_LOG%"
echo [%date% %time%] Command line: %* >> "%INSTALLATION_LOG%"

REM Display header
if "%SILENT_MODE%"=="0" (
    cls
    call :display_header
)

REM Check for administrator privileges if not in dry-run mode
if "%DRY_RUN%"=="0" (
    call :check_admin_privileges
    if errorlevel 1 goto error_exit
)

REM Pre-installation checks
call :pre_installation_checks
if errorlevel 1 goto error_exit

REM Check for existing installation
if "%FORCE_REINSTALL%"=="0" (
    call :check_existing_installation
    if errorlevel 1 goto error_exit
)

REM Phase 1: System Detection
call :log_info "Starting Phase 1: System Detection"
call :display_phase "SYSTEM DETECTION" "Analyzing your hardware configuration..."
call :run_python_phase "detect_system" "System detection completed"
if errorlevel 1 goto phase_error

REM Phase 2: Dependency Management
call :log_info "Starting Phase 2: Dependency Management"
call :display_phase "DEPENDENCY SETUP" "Installing Python packages and dependencies..."
call :run_python_phase "setup_dependencies" "Dependencies installed successfully"
if errorlevel 1 goto phase_error

REM Phase 3: Model Download (if not skipped)
if "%SKIP_MODELS%"=="0" (
    call :log_info "Starting Phase 3: Model Download"
    call :display_phase "MODEL DOWNLOAD" "Downloading WAN2.2 models (this may take a while)..."
    call :run_python_phase "download_models" "Models downloaded successfully"
    if errorlevel 1 goto phase_error
) else (
    call :log_info "Skipping Phase 3: Model Download (--skip-models specified)"
    if "%SILENT_MODE%"=="0" echo ⚠️  Skipping model download as requested
)

REM Phase 4: Configuration Generation
call :log_info "Starting Phase 4: Configuration Generation"
call :display_phase "CONFIGURATION" "Generating optimized configuration for your hardware..."
call :run_python_phase "generate_config" "Configuration generated successfully"
if errorlevel 1 goto phase_error

REM Phase 5: Installation Validation
call :log_info "Starting Phase 5: Installation Validation"
call :display_phase "VALIDATION" "Validating installation and running tests..."
call :run_python_phase "validate_installation" "Installation validated successfully"
if errorlevel 1 goto phase_error

REM Post-installation setup
call :log_info "Running post-installation setup"
call :display_phase "FINALIZATION" "Creating shortcuts and finalizing setup..."
call :create_shortcuts
if errorlevel 1 goto phase_error

REM Installation completed successfully
set "INSTALL_SUCCESS=1"
call :log_info "Installation completed successfully"
call :display_success
goto cleanup_exit

:phase_error
call :log_error "Installation phase failed with error code %errorlevel%"
call :display_error "Installation failed during current phase" "Check %INSTALLATION_LOG% for details"
goto error_exit

:error_exit
call :log_error "Installation failed with error code %errorlevel%"
if "%SILENT_MODE%"=="0" (
    echo.
    echo ❌ Installation failed. Check the following:
    echo    • %INSTALLATION_LOG%
    echo    • %ERROR_LOG%
    echo.
    echo 💡 Common solutions:
    echo    • Run as Administrator
    echo    • Check internet connection
    echo    • Ensure sufficient disk space
    echo    • Close other applications
    echo.
    pause
)
exit /b 1

:cleanup_exit
if "%SILENT_MODE%"=="0" pause
exit /b 0

REM ============================================================================
REM HELPER FUNCTIONS
REM ============================================================================

:display_header
echo.
echo ████████████████████████████████████████████████████████████████████████████
echo ██                                                                        ██
echo ██                    WAN2.2 Local Installation System                   ██
echo ██                                                                        ██
echo ██    Automated installation with hardware detection and optimization    ██
echo ██                                                                        ██
echo ████████████████████████████████████████████████████████████████████████████
echo.
if "%DRY_RUN%"=="1" (
    echo 🔍 DRY RUN MODE - No changes will be made
    echo.
)
goto :eof

:display_phase
if "%SILENT_MODE%"=="1" goto :eof
echo.
echo ═══════════════════════════════════════════════════════════════════════════
echo  %~1
echo ═══════════════════════════════════════════════════════════════════════════
echo %~2
echo.
goto :eof

:display_success
if "%SILENT_MODE%"=="1" goto :eof
echo.
echo ████████████████████████████████████████████████████████████████████████████
echo ██                                                                        ██
echo ██                    🎉 INSTALLATION COMPLETED! 🎉                      ██
echo ██                                                                        ██
echo ████████████████████████████████████████████████████████████████████████████
echo.
echo ✅ WAN2.2 has been successfully installed and configured for your system
echo.
echo 📋 Next Steps:
echo    • Desktop shortcuts have been created
echo    • Check the User Guide for usage instructions  
echo    • Run validation tests to ensure everything works
echo.
echo 📁 Important Files:
echo    • Installation log: %INSTALLATION_LOG%
echo    • Configuration: config.json
echo    • Models directory: models\
echo.
echo 🆘 For support:
echo    • Check documentation in the docs\ folder
echo    • Review logs for troubleshooting
echo    • Visit the project repository for updates
echo.
goto :eof

:display_error
if "%SILENT_MODE%"=="1" goto :eof
echo.
echo ❌ ERROR: %~1
echo 💡 %~2
echo.
goto :eof

:check_admin_privileges
call :log_info "Checking administrator privileges"
net session >nul 2>&1
if errorlevel 1 (
    call :log_warning "Administrator privileges not detected"
    if "%SILENT_MODE%"=="0" (
        echo ⚠️  Administrator privileges recommended for optimal installation
        echo    Some features may not work without elevated permissions
        echo.
        choice /c YN /m "Continue anyway? (Y/N)"
        if errorlevel 2 (
            call :log_info "User chose to exit due to insufficient privileges"
            exit /b 1
        )
    )
) else (
    call :log_info "Administrator privileges confirmed"
)
goto :eof

:pre_installation_checks
call :log_info "Running pre-installation checks"

REM Check disk space (require at least 50GB)
call :check_disk_space
if errorlevel 1 goto :eof

REM Check internet connectivity
call :check_internet_connection
if errorlevel 1 goto :eof

REM Verify Python availability
call :check_python_installation
if errorlevel 1 goto :eof

call :log_info "Pre-installation checks completed successfully"
goto :eof

:check_disk_space
call :log_info "Checking available disk space"
for /f "tokens=3" %%a in ('dir /-c "%INSTALL_DIR%" ^| find "bytes free"') do set "FREE_BYTES=%%a"
set /a "FREE_GB=%FREE_BYTES:~0,-9%"
if %FREE_GB% LSS 50 (
    call :log_error "Insufficient disk space: %FREE_GB%GB available, 50GB required"
    call :display_error "Insufficient disk space" "At least 50GB free space required"
    exit /b 1
)
call :log_info "Disk space check passed: %FREE_GB%GB available"
goto :eof

:check_internet_connection
call :log_info "Checking internet connectivity"
ping -n 1 8.8.8.8 >nul 2>&1
if errorlevel 1 (
    call :log_warning "Internet connectivity check failed"
    if "%SILENT_MODE%"=="0" (
        echo ⚠️  Internet connection not detected
        echo    Model downloads and package installation may fail
        echo.
        choice /c YN /m "Continue anyway? (Y/N)"
        if errorlevel 2 (
            call :log_info "User chose to exit due to no internet connection"
            exit /b 1
        )
    )
) else (
    call :log_info "Internet connectivity confirmed"
)
goto :eof

:check_python_installation
call :log_info "Checking Python installation"
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    call :log_info "Python not found, will install embedded Python"
    if "%SILENT_MODE%"=="0" echo 🐍 Python not found - will install embedded Python
    
    REM Check if embedded Python installer exists
    if not exist "%SCRIPTS_DIR%\install_python.bat" (
        call :log_error "Python installer script not found"
        call :display_error "Python installer missing" "install_python.bat not found in scripts directory"
        exit /b 1
    )
    
    REM Install embedded Python
    call :log_info "Installing embedded Python"
    call "%SCRIPTS_DIR%\install_python.bat"
    if errorlevel 1 (
        call :log_error "Failed to install embedded Python"
        call :display_error "Python installation failed" "Could not install embedded Python"
        exit /b 1
    )
    
    REM Update Python command to use embedded version
    set "PYTHON_CMD=%INSTALL_DIR%python\python.exe"
) else (
    call :log_info "Python installation confirmed"
    if "%VERBOSE_MODE%"=="1" (
        for /f "tokens=*" %%a in ('%PYTHON_CMD% --version 2^>^&1') do (
            call :log_info "Python version: %%a"
            if "%SILENT_MODE%"=="0" echo 🐍 Found: %%a
        )
    )
)
goto :eof

:check_existing_installation
call :log_info "Checking for existing installation"
if exist "%LOGS_DIR%\installation_state.json" (
    call :log_info "Found existing installation state"
    if "%SILENT_MODE%"=="0" (
        echo 🔄 Found incomplete installation
        choice /c YN /m "Resume previous installation? (Y/N)"
        if errorlevel 1 (
            call :log_info "User chose to resume existing installation"
            REM The Python script will handle resuming
        ) else (
            call :log_info "User chose to start fresh installation"
            del "%LOGS_DIR%\installation_state.json" >nul 2>&1
        )
    ) else (
        call :log_info "Silent mode: auto-resuming existing installation"
    )
)
goto :eof

:run_python_phase
set "PHASE_NAME=%~1"
set "SUCCESS_MESSAGE=%~2"

call :log_info "Executing Python phase: %PHASE_NAME%"

REM Build Python command with arguments
set "PYTHON_ARGS="
if "%SILENT_MODE%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --silent"
if "%VERBOSE_MODE%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --verbose"
if "%DEV_MODE%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --dev-mode"
if "%SKIP_MODELS%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --skip-models"
if "%FORCE_REINSTALL%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --force-reinstall"
if "%DRY_RUN%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --dry-run"
if not "%CUSTOM_PATH%"=="" set "PYTHON_ARGS=%PYTHON_ARGS% --custom-path "%CUSTOM_PATH%""

REM Execute the main installer
%PYTHON_CMD% "%SCRIPTS_DIR%\main_installer.py" %PYTHON_ARGS%
set "PYTHON_EXIT_CODE=%errorlevel%"

if "%PYTHON_EXIT_CODE%"=="0" (
    call :log_info "Phase %PHASE_NAME% completed successfully"
    if "%SILENT_MODE%"=="0" echo ✅ %SUCCESS_MESSAGE%
) else (
    call :log_error "Phase %PHASE_NAME% failed with exit code %PYTHON_EXIT_CODE%"
    if "%SILENT_MODE%"=="0" echo ❌ Phase %PHASE_NAME% failed
)

exit /b %PYTHON_EXIT_CODE%

:create_shortcuts
call :log_info "Creating desktop shortcuts and start menu entries"
if exist "%SCRIPTS_DIR%\create_shortcuts.py" (
    %PYTHON_CMD% "%SCRIPTS_DIR%\create_shortcuts.py"
    if errorlevel 1 (
        call :log_warning "Failed to create shortcuts"
        if "%SILENT_MODE%"=="0" echo ⚠️  Warning: Could not create desktop shortcuts
    ) else (
        call :log_info "Shortcuts created successfully"
    )
) else (
    call :log_warning "Shortcut creation script not found"
)
goto :eof

:log_info
echo [%date% %time%] INFO: %~1 >> "%INSTALLATION_LOG%"
if "%VERBOSE_MODE%"=="1" if "%SILENT_MODE%"=="0" echo [INFO] %~1
goto :eof

:log_warning
echo [%date% %time%] WARNING: %~1 >> "%INSTALLATION_LOG%"
echo [%date% %time%] WARNING: %~1 >> "%ERROR_LOG%"
if "%VERBOSE_MODE%"=="1" if "%SILENT_MODE%"=="0" echo [WARNING] %~1
goto :eof

:log_error
echo [%date% %time%] ERROR: %~1 >> "%INSTALLATION_LOG%"
echo [%date% %time%] ERROR: %~1 >> "%ERROR_LOG%"
if "%VERBOSE_MODE%"=="1" if "%SILENT_MODE%"=="0" echo [ERROR] %~1
goto :eof

:show_help
echo.
echo WAN2.2 Local Installation System
echo.
echo Usage: install.bat [options]
echo.
echo Options:
echo   --silent          Run installation in silent mode (no user interaction)
echo   --verbose         Enable verbose logging and output
echo   --dev-mode        Install development dependencies
echo   --skip-models     Skip model download (for testing)
echo   --force-reinstall Force complete reinstallation
echo   --dry-run         Simulate installation without making changes
echo   --custom-path     Specify custom installation path
echo   --help            Show this help message
echo.
echo Examples:
echo   install.bat                    # Standard installation
echo   install.bat --silent          # Silent installation
echo   install.bat --verbose         # Verbose installation
echo   install.bat --skip-models     # Install without downloading models
echo   install.bat --dry-run         # Test installation without changes
echo.
exit /b 0