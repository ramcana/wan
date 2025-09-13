#!/usr/bin/env python3
"""
Windows-specific utilities for the startup manager.
Handles UAC, Windows services, firewall exceptions, and other Windows-specific features.
"""

import os
import sys
import subprocess
import ctypes
import winreg
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class WindowsPermissionManager:
    """Manages Windows permissions and UAC elevation"""
    
    @staticmethod
    def is_admin() -> bool:
        """Check if the current process is running with administrator privileges"""
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception as e:
            logger.warning(f"Could not check admin status: {e}")
            return False
    
    @staticmethod
    def request_elevation(script_path: str, args: List[str] = None) -> bool:
        """Request UAC elevation for the current script"""
        try:
            if WindowsPermissionManager.is_admin():
                return True
            
            # Build command for elevation
            if args is None:
                args = []
            
            # Use ShellExecute with 'runas' verb to trigger UAC
            params = ' '.join(args) if args else ''
            
            result = ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                f'"{script_path}" {params}',
                None,
                1  # SW_SHOWNORMAL
            )
            
            # ShellExecute returns > 32 on success
            return result > 32
            
        except Exception as e:
            logger.error(f"Failed to request elevation: {e}")
            return False
    
    @staticmethod
    def can_bind_privileged_ports() -> bool:
        """Check if we can bind to privileged ports (< 1024)"""
        # On Windows, this typically requires admin rights
        return WindowsPermissionManager.is_admin()


class WindowsFirewallManager:
    """Manages Windows Firewall exceptions"""
    
    @staticmethod
    def check_firewall_exception(program_path: str) -> bool:
        """Check if a program has a firewall exception"""
        try:
            # Use netsh to check firewall rules
            result = subprocess.run([
                'netsh', 'advfirewall', 'firewall', 'show', 'rule',
                f'name=all', 'dir=in'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Check if the program path appears in the output
                return str(program_path).lower() in result.stdout.lower()
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not check firewall exception: {e}")
            return False
    
    @staticmethod
    def add_firewall_exception(program_path: str, rule_name: str, 
                             description: str = None) -> bool:
        """Add a firewall exception for a program"""
        try:
            if not WindowsPermissionManager.is_admin():
                logger.warning("Admin rights required to add firewall exception")
                return False
            
            cmd = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name={rule_name}',
                'dir=in',
                'action=allow',
                f'program={program_path}',
                'enable=yes'
            ]
            
            if description:
                cmd.extend(['description', description])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Added firewall exception for {program_path}")
                return True
            else:
                logger.error(f"Failed to add firewall exception: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding firewall exception: {e}")
            return False
    
    @staticmethod
    def add_port_exception(port: int, protocol: str = 'TCP', 
                          rule_name: str = None) -> bool:
        """Add a firewall exception for a specific port"""
        try:
            if not WindowsPermissionManager.is_admin():
                logger.warning("Admin rights required to add port exception")
                return False
            
            if rule_name is None:
                rule_name = f"WAN22 {protocol} Port {port}"
            
            cmd = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name={rule_name}',
                'dir=in',
                'action=allow',
                'protocol=TCP',
                f'localport={port}',
                'enable=yes'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Added firewall exception for port {port}")
                return True
            else:
                logger.error(f"Failed to add port exception: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding port exception: {e}")
            return False


class WindowsServiceManager:
    """Manages Windows services for background server management"""
    
    @staticmethod
    def is_service_installed(service_name: str) -> bool:
        """Check if a Windows service is installed"""
        try:
            result = subprocess.run([
                'sc', 'query', service_name
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.warning(f"Could not check service status: {e}")
            return False
    
    @staticmethod
    def install_service(service_name: str, display_name: str, 
                       executable_path: str, description: str = None) -> bool:
        """Install a Windows service"""
        try:
            if not WindowsPermissionManager.is_admin():
                logger.warning("Admin rights required to install service")
                return False
            
            # Create the service
            cmd = [
                'sc', 'create', service_name,
                f'binPath={executable_path}',
                f'DisplayName={display_name}',
                'start=demand'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Failed to create service: {result.stderr}")
                return False
            
            # Set description if provided
            if description:
                desc_cmd = [
                    'sc', 'description', service_name, description
                ]
                subprocess.run(desc_cmd, capture_output=True, text=True, timeout=10)
            
            logger.info(f"Installed Windows service: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing service: {e}")
            return False
    
    @staticmethod
    def start_service(service_name: str) -> bool:
        """Start a Windows service"""
        try:
            result = subprocess.run([
                'sc', 'start', service_name
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            return False
    
    @staticmethod
    def stop_service(service_name: str) -> bool:
        """Stop a Windows service"""
        try:
            result = subprocess.run([
                'sc', 'stop', service_name
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            return False
    
    @staticmethod
    def uninstall_service(service_name: str) -> bool:
        """Uninstall a Windows service"""
        try:
            if not WindowsPermissionManager.is_admin():
                logger.warning("Admin rights required to uninstall service")
                return False
            
            # Stop service first
            WindowsServiceManager.stop_service(service_name)
            
            # Delete the service
            result = subprocess.run([
                'sc', 'delete', service_name
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Uninstalled Windows service: {service_name}")
                return True
            else:
                logger.error(f"Failed to uninstall service: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error uninstalling service: {e}")
            return False


class WindowsRegistryManager:
    """Manages Windows Registry operations for startup configuration"""
    
    @staticmethod
    def get_startup_entry(app_name: str) -> Optional[str]:
        """Get a startup entry from the Windows Registry"""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                              r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                value, _ = winreg.QueryValueEx(key, app_name)
                return value
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Could not read startup entry: {e}")
            return None
    
    @staticmethod
    def set_startup_entry(app_name: str, command: str) -> bool:
        """Set a startup entry in the Windows Registry"""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                              r"Software\Microsoft\Windows\CurrentVersion\Run", 
                              0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, command)
                logger.info(f"Added startup entry: {app_name}")
                return True
        except Exception as e:
            logger.error(f"Could not set startup entry: {e}")
            return False
    
    @staticmethod
    def remove_startup_entry(app_name: str) -> bool:
        """Remove a startup entry from the Windows Registry"""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                              r"Software\Microsoft\Windows\CurrentVersion\Run", 
                              0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, app_name)
                logger.info(f"Removed startup entry: {app_name}")
                return True
        except FileNotFoundError:
            return True  # Already doesn't exist
        except Exception as e:
            logger.error(f"Could not remove startup entry: {e}")
            return False


class WindowsSystemInfo:
    """Provides Windows system information"""
    
    @staticmethod
    def get_windows_version() -> Dict[str, Any]:
        """Get Windows version information"""
        try:
            import platform
            return {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
        except Exception as e:
            logger.warning(f"Could not get Windows version: {e}")
            return {}
    
    @staticmethod
    def get_defender_status() -> Dict[str, Any]:
        """Get Windows Defender status"""
        try:
            # Use PowerShell to get Defender status
            result = subprocess.run([
                'powershell', '-Command',
                'Get-MpComputerStatus | Select-Object RealTimeProtectionEnabled, FirewallEnabled | ConvertTo-Json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
            
            return {}
            
        except Exception as e:
            logger.warning(f"Could not get Defender status: {e}")
            return {}
    
    @staticmethod
    def is_wsl_available() -> bool:
        """Check if Windows Subsystem for Linux is available"""
        try:
            result = subprocess.run([
                'wsl', '--list'
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0
            
        except Exception:
            return False


class WindowsOptimizer:
    """Windows-specific optimizations for the startup manager"""
    
    def __init__(self):
        self.permission_manager = WindowsPermissionManager()
        self.firewall_manager = WindowsFirewallManager()
        self.service_manager = WindowsServiceManager()
        self.registry_manager = WindowsRegistryManager()
        self.system_info = WindowsSystemInfo()
    
    def optimize_for_development(self, ports: List[int] = None) -> Dict[str, Any]:
        """Apply Windows optimizations for development environment"""
        results = {
            'firewall_exceptions': [],
            'permissions_elevated': False,
            'optimizations_applied': []
        }
        
        try:
            # Check if we need elevation
            if not self.permission_manager.is_admin():
                logger.info("Checking if elevation is needed for optimizations...")
                results['elevation_needed'] = True
            else:
                results['permissions_elevated'] = True
            
            # Add firewall exceptions for common development ports
            if ports is None:
                ports = [3000, 8000, 8080, 3001, 8001]
            
            for port in ports:
                if self.permission_manager.is_admin():
                    success = self.firewall_manager.add_port_exception(
                        port, 'TCP', f'WAN22 Development Port {port}'
                    )
                    results['firewall_exceptions'].append({
                        'port': port,
                        'success': success
                    })
            
            # Add Python and Node.js firewall exceptions
            python_path = sys.executable
            if self.permission_manager.is_admin():
                success = self.firewall_manager.add_firewall_exception(
                    python_path, 'WAN22 Python Development',
                    'Allow Python for WAN22 development'
                )
                results['firewall_exceptions'].append({
                    'program': 'Python',
                    'path': python_path,
                    'success': success
                })
            
            results['optimizations_applied'].append('firewall_configuration')
            
            return results
            
        except Exception as e:
            logger.error(f"Error during Windows optimization: {e}")
            results['error'] = str(e)
            return results
    
    def setup_service_mode(self, service_name: str = 'WAN22Service') -> bool:
        """Set up Windows service for background server management"""
        try:
            if not self.permission_manager.is_admin():
                logger.warning("Admin rights required for service setup")
                return False
            
            # Create a service wrapper script
            service_script = Path(__file__).parent / 'service_wrapper.py'
            
            if not service_script.exists():
                self._create_service_wrapper(service_script)
            
            # Install the service
            success = self.service_manager.install_service(
                service_name,
                'WAN22 Server Manager',
                f'"{sys.executable}" "{service_script}"',
                'WAN22 Video Generation Server Management Service'
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting up service mode: {e}")
            return False
    
    def _create_service_wrapper(self, script_path: Path):
        """Create a Windows service wrapper script"""
        wrapper_code = '''#!/usr/bin/env python3
"""
Windows Service wrapper for WAN22 Server Manager
"""

import sys
import time
import logging
from pathlib import Path

# Add startup manager to path
sys.path.insert(0, str(Path(__file__).parent))

from startup_manager import StartupManager
from startup_manager.config import load_config
from startup_manager.cli import InteractiveCLI, CLIOptions

class WAN22Service:
    def __init__(self):
        self.running = False
        self.startup_manager = None
    
    def start(self):
        """Start the service"""
        try:
            self.running = True
            
            # Load configuration
            config = load_config()
            
            # Create CLI interface in service mode
            cli_options = CLIOptions(interactive=False, verbosity="INFO")
            cli_interface = InteractiveCLI(cli_options)
            
            # Create startup manager
            self.startup_manager = StartupManager(cli_interface, config)
            
            # Run startup sequence
            success = self.startup_manager.run_startup_sequence()
            
            if success:
                # Keep service running
                while self.running:
                    time.sleep(10)
                    # Could add health checks here
            
        except Exception as e:
            logging.error(f"Service error: {e}")
    
    def stop(self):
        """Stop the service"""
        self.running = False
        if self.startup_manager:
            # Cleanup processes
            pass

if __name__ == '__main__':
    service = WAN22Service()
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
'''
        
        with open(script_path, 'w') as f:
            f.write(wrapper_code)
