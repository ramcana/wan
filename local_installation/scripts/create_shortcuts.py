"""
Utility script for creating desktop shortcuts and start menu entries.
This will be used in the final installation phase.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from base_classes import BaseInstallationComponent
from interfaces import InstallationError, ErrorCategory


class ShortcutCreator(BaseInstallationComponent):
    """Creates desktop shortcuts and start menu entries."""
    
    def __init__(self, installation_path: str):
        super().__init__(installation_path)
        self.desktop_path = Path.home() / "Desktop"
        self.start_menu_path = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        self.venv_path = Path(installation_path) / "venv"
        self.python_exe = self.venv_path / "Scripts" / "python.exe"
    
    def create_desktop_shortcuts(self) -> bool:
        """Create desktop shortcuts for WAN2.2 applications."""
        try:
            shortcuts = [
                {
                    "name": "WAN2.2 Desktop UI",
                    "target": "launch_wan22.bat",
                    "icon": "resources/wan22_icon.ico",
                    "description": "WAN2.2 Desktop Video Generation Interface",
                    "working_dir": str(self.installation_path),
                    "is_batch": True
                },
                {
                    "name": "WAN2.2 Web UI",
                    "target": "launch_web_ui.bat", 
                    "icon": "resources/wan22_web_icon.ico",
                    "description": "WAN2.2 Web-based Video Generation Interface",
                    "working_dir": str(self.installation_path),
                    "is_batch": True
                }
            ]
            
            for shortcut in shortcuts:
                if shortcut.get("is_batch"):
                    # Create shortcut to batch file
                    self._create_batch_shortcut(
                        shortcut["name"],
                        shortcut["target"],
                        self.desktop_path,
                        shortcut.get("description")
                    )
                else:
                    success = self._create_windows_shortcut(
                        shortcut["name"],
                        shortcut["target"],
                        self.desktop_path,
                        shortcut.get("icon"),
                        shortcut.get("description"),
                        shortcut.get("working_dir")
                    )
                    if not success:
                        # Fallback to batch file
                        self._create_batch_launcher(
                            shortcut["name"],
                            shortcut["target"],
                            self.desktop_path,
                            shortcut.get("description")
                        )
            
            self.logger.info("Desktop shortcuts created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create desktop shortcuts: {e}")
            return False
    
    def create_start_menu_entries(self) -> bool:
        """Create start menu entries for WAN2.2 applications."""
        try:
            # Create WAN2.2 folder in start menu
            wan22_folder = self.start_menu_path / "WAN2.2"
            wan22_folder.mkdir(exist_ok=True)
            
            shortcuts = [
                {
                    "name": "WAN2.2 Desktop UI",
                    "target": "launch_wan22.bat",
                    "icon": "resources/wan22_icon.ico",
                    "description": "WAN2.2 Desktop Video Generation Interface",
                    "working_dir": str(self.installation_path),
                    "is_batch": True
                },
                {
                    "name": "WAN2.2 Web UI",
                    "target": "launch_web_ui.bat",
                    "icon": "resources/wan22_web_icon.ico", 
                    "description": "WAN2.2 Web-based Video Generation Interface",
                    "working_dir": str(self.installation_path),
                    "is_batch": True
                },
                {
                    "name": "WAN2.2 Configuration",
                    "target": "run_first_setup.bat",
                    "icon": "resources/config_icon.ico",
                    "description": "WAN2.2 Configuration and First-time Setup",
                    "working_dir": str(self.installation_path),
                    "is_batch": True
                },
                {
                    "name": "Uninstall WAN2.2",
                    "target": "uninstall.bat",
                    "description": "Uninstall WAN2.2 System",
                    "working_dir": str(self.installation_path),
                    "is_batch": True
                }
            ]
            
            for shortcut in shortcuts:
                if shortcut.get("is_batch"):
                    self._create_batch_shortcut(
                        shortcut["name"],
                        shortcut["target"],
                        wan22_folder,
                        shortcut.get("description")
                    )
                else:
                    success = self._create_windows_shortcut(
                        shortcut["name"],
                        shortcut["target"],
                        wan22_folder,
                        shortcut.get("icon"),
                        shortcut.get("description"),
                        shortcut.get("working_dir")
                    )
                    if not success:
                        # Fallback to batch file
                        self._create_batch_launcher(
                            shortcut["name"],
                            shortcut["target"],
                            wan22_folder,
                            shortcut.get("description")
                        )
            
            self.logger.info("Start menu entries created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create start menu entries: {e}")
            return False
    
    def _create_windows_shortcut(self, name: str, target: str, location: Path,
                               icon: Optional[str] = None, description: Optional[str] = None,
                               working_dir: Optional[str] = None) -> bool:
        """Create a proper Windows .lnk shortcut file using PowerShell."""
        try:
            shortcut_path = location / f"{name}.lnk"
            target_path = Path(working_dir or self.installation_path) / target
            
            # PowerShell script to create .lnk file
            ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{self.python_exe}"
$Shortcut.Arguments = '"{target_path}"'
$Shortcut.WorkingDirectory = "{working_dir or self.installation_path}"
$Shortcut.Description = "{description or name}"
'''
            
            if icon and Path(self.installation_path) / icon:
                ps_script += f'$Shortcut.IconLocation = "{Path(self.installation_path) / icon}"\n'
            
            ps_script += '$Shortcut.Save()'
            
            # Execute PowerShell script
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Created Windows shortcut: {shortcut_path}")
                return True
            else:
                self.logger.warning(f"PowerShell shortcut creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to create Windows shortcut {name}: {e}")
            return False
    
    def _create_batch_launcher(self, name: str, target: str, location: Path,
                             description: Optional[str] = None) -> None:
        """Create a batch file launcher with environment activation."""
        try:
            shortcut_path = location / f"{name}.bat"
            target_path = Path(self.installation_path) / target
            
            # Create batch file that activates environment and launches target
            batch_content = f'''@echo off
title {name}
echo Starting {name}...
echo.

REM Change to installation directory
cd /d "{self.installation_path}"

REM Activate virtual environment
call "{self.venv_path}\\Scripts\\activate.bat"

REM Launch application
python "{target_path}"

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to close.
    pause >nul
)
'''
            
            with open(shortcut_path, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            
            self.logger.debug(f"Created batch launcher: {shortcut_path}")
            
        except Exception as e:
            raise InstallationError(
                f"Failed to create batch launcher {name}: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check file permissions", "Ensure target directory exists"]
            )
    
    def _create_batch_shortcut(self, name: str, target: str, location: Path,
                             description: Optional[str] = None) -> None:
        """Create a shortcut to an existing batch file."""
        try:
            shortcut_path = location / f"{name}.lnk"
            target_path = Path(self.installation_path) / target
            
            # PowerShell script to create .lnk file pointing to batch file
            ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target_path}"
$Shortcut.WorkingDirectory = "{self.installation_path}"
$Shortcut.Description = "{description or name}"
$Shortcut.Save()
'''
            
            # Execute PowerShell script
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Created batch shortcut: {shortcut_path}")
            else:
                self.logger.warning(f"PowerShell batch shortcut creation failed: {result.stderr}")
                # Fallback: just copy the batch file
                import shutil
                fallback_path = location / f"{name}.bat"
                shutil.copy2(target_path, fallback_path)
                self.logger.debug(f"Created fallback batch copy: {fallback_path}")
                
        except Exception as e:
            self.logger.warning(f"Failed to create batch shortcut {name}: {e}")
            # Try to copy the batch file as fallback
            try:
                import shutil
                fallback_path = location / f"{name}.bat"
                shutil.copy2(Path(self.installation_path) / target, fallback_path)
                self.logger.debug(f"Created fallback batch copy: {fallback_path}")
            except Exception as fallback_error:
                raise InstallationError(
                    f"Failed to create batch shortcut {name}: {str(e)} (fallback also failed: {fallback_error})",
                    ErrorCategory.SYSTEM,
                    ["Check file permissions", "Ensure target batch file exists"]
                )
    
    def create_application_launcher(self) -> bool:
        """Create main application launcher script."""
        try:
            launcher_path = self.installation_path / "launch_wan22.bat"
            
            launcher_content = f'''@echo off
title WAN2.2 Video Generator
echo ========================================
echo WAN2.2 Video Generation System
echo ========================================
echo.

REM Change to installation directory
cd /d "{self.installation_path}"

REM Check if virtual environment exists
if not exist "{self.venv_path}\\Scripts\\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating environment...
call "{self.venv_path}\\Scripts\\activate.bat"

REM Check if main application exists
if not exist "application\\main.py" (
    echo Error: Main application not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Launch main application
echo Starting WAN2.2...
echo.
python "application\\main.py"

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the logs for details.
    echo Press any key to close.
    pause >nul
)
'''
            
            with open(launcher_path, 'w', encoding='utf-8') as f:
                f.write(launcher_content)
            
            # Also create UI launcher
            ui_launcher_path = self.installation_path / "launch_wan22_ui.bat"
            
            ui_launcher_content = f'''@echo off
title WAN2.2 UI
echo ========================================
echo WAN2.2 User Interface
echo ========================================
echo.

REM Change to installation directory
cd /d "{self.installation_path}"

REM Check if virtual environment exists
if not exist "{self.venv_path}\\Scripts\\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating environment...
call "{self.venv_path}\\Scripts\\activate.bat"

REM Check if UI application exists
if not exist "application\\ui.py" (
    echo Error: UI application not found!
    echo Please run the installer again.
    echo.
    pause
    exit /b 1
)

REM Launch UI application
echo Starting WAN2.2 UI...
echo.
python "application\\ui.py"

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the logs for details.
    echo Press any key to close.
    pause >nul
)
'''
            
            with open(ui_launcher_path, 'w', encoding='utf-8') as f:
                f.write(ui_launcher_content)
            
            self.logger.info("Application launchers created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create application launchers: {e}")
            return False
    
    def create_uninstaller(self) -> bool:
        """Create uninstaller script."""
        try:
            uninstall_script = self.installation_path / "uninstall.bat"
            
            uninstall_content = '''@echo off
echo WAN2.2 Uninstaller
echo ==================
echo.
echo This will remove WAN2.2 shortcuts and start menu entries.
echo The installation files will remain for manual deletion.
echo.
set /p confirm="Are you sure you want to uninstall? (y/n): "
if /i not "%confirm%"=="y" goto :cancel

echo.
echo Removing WAN2.2 shortcuts...

REM Remove desktop shortcuts (.lnk and .bat files)
del "%USERPROFILE%\\Desktop\\WAN2.2 Video Generator.lnk" 2>nul
del "%USERPROFILE%\\Desktop\\WAN2.2 Video Generator.bat" 2>nul
del "%USERPROFILE%\\Desktop\\WAN2.2 UI.lnk" 2>nul
del "%USERPROFILE%\\Desktop\\WAN2.2 UI.bat" 2>nul

REM Remove start menu entries
rmdir /s /q "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\WAN2.2" 2>nul

echo.
echo WAN2.2 shortcuts have been removed.
echo.
echo To completely remove WAN2.2, you can manually delete this folder:
echo %~dp0
echo.
echo Thank you for using WAN2.2!
echo.
pause
goto :end

:cancel
echo Uninstall cancelled.
pause

:end'''
            
            with open(uninstall_script, 'w', encoding='utf-8') as f:
                f.write(uninstall_content)
            
            self.logger.info("Uninstaller created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create uninstaller: {e}")
            return False
    
    def create_all_shortcuts(self) -> bool:
        """Create all shortcuts, launchers, and integration components."""
        try:
            self.logger.info("Creating application integration components...")
            
            # Create application launchers
            if not self.create_application_launcher():
                self.logger.warning("Failed to create application launchers")
                return False
            
            # Create desktop shortcuts
            if not self.create_desktop_shortcuts():
                self.logger.warning("Failed to create desktop shortcuts")
                return False
            
            # Create start menu entries
            if not self.create_start_menu_entries():
                self.logger.warning("Failed to create start menu entries")
                return False
            
            # Create uninstaller
            if not self.create_uninstaller():
                self.logger.warning("Failed to create uninstaller")
                return False
            
            self.logger.info("All application integration components created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create application integration: {e}")
            return False