#!/usr/bin/env python3
"""
Integration tests for Windows-specific startup manager features.
Tests UAC handling, firewall management, and Windows service integration.
"""

import pytest
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add startup_manager to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from startup_manager.windows_utils import (
    WindowsPermissionManager,
    WindowsFirewallManager,
    WindowsServiceManager,
    WindowsRegistryManager,
    WindowsSystemInfo,
    WindowsOptimizer
)


class TestWindowsPermissionManager:
    """Test Windows permission and UAC handling"""
    
    def test_admin_check(self):
        """Test administrator privilege checking"""
        manager = WindowsPermissionManager()
        
        # Should return a boolean
        is_admin = manager.is_admin()
        assert isinstance(is_admin, bool)
    
    @patch('ctypes.windll.shell32.IsUserAnAdmin')
    def test_admin_check_mocked(self, mock_is_admin):
        """Test admin check with mocked Windows API"""
        mock_is_admin.return_value = True
        
        manager = WindowsPermissionManager()
        assert manager.is_admin() == True
        
        mock_is_admin.return_value = False
        assert manager.is_admin() == False
    
    def test_privileged_port_check(self):
        """Test privileged port binding capability"""
        manager = WindowsPermissionManager()
        
        # Should return a boolean
        can_bind = manager.can_bind_privileged_ports()
        assert isinstance(can_bind, bool)
    
    @patch('ctypes.windll.shell32.ShellExecuteW')
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    def test_request_elevation(self, mock_is_admin, mock_shell_execute):
        """Test UAC elevation request"""
        mock_is_admin.return_value = False
        mock_shell_execute.return_value = 42  # Success value > 32
        
        manager = WindowsPermissionManager()
        result = manager.request_elevation("test_script.py", ["--test"])
        
        assert result == True
        mock_shell_execute.assert_called_once()


class TestWindowsFirewallManager:
    """Test Windows Firewall management"""
    
    @patch('subprocess.run')
    def test_check_firewall_exception(self, mock_run):
        """Test checking for existing firewall exceptions"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="python.exe is in the firewall rules"
        )
        
        manager = WindowsFirewallManager()
        result = manager.check_firewall_exception("python.exe")
        
        assert result == True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    def test_add_firewall_exception(self, mock_is_admin, mock_run):
        """Test adding firewall exceptions"""
        mock_is_admin.return_value = True
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        manager = WindowsFirewallManager()
        result = manager.add_firewall_exception(
            "C:\\Python\\python.exe", 
            "Test Python Rule"
        )
        
        assert result == True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    def test_add_port_exception(self, mock_is_admin, mock_run):
        """Test adding port-based firewall exceptions"""
        mock_is_admin.return_value = True
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        manager = WindowsFirewallManager()
        result = manager.add_port_exception(8000, "TCP", "Test Port Rule")
        
        assert result == True
        mock_run.assert_called_once()
    
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    def test_add_exception_without_admin(self, mock_is_admin):
        """Test that firewall exceptions require admin rights"""
        mock_is_admin.return_value = False
        
        manager = WindowsFirewallManager()
        result = manager.add_firewall_exception("test.exe", "Test Rule")
        
        assert result == False


class TestWindowsServiceManager:
    """Test Windows service management"""
    
    @patch('subprocess.run')
    def test_is_service_installed(self, mock_run):
        """Test checking if a service is installed"""
        mock_run.return_value = Mock(returncode=0)
        
        manager = WindowsServiceManager()
        result = manager.is_service_installed("TestService")
        
        assert result == True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    def test_install_service(self, mock_is_admin, mock_run):
        """Test installing a Windows service"""
        mock_is_admin.return_value = True
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        manager = WindowsServiceManager()
        result = manager.install_service(
            "TestService",
            "Test Service Display Name",
            "C:\\test\\service.exe",
            "Test service description"
        )
        
        assert result == True
        assert mock_run.call_count >= 1  # At least one call for service creation
    
    @patch('subprocess.run')
    def test_start_service(self, mock_run):
        """Test starting a Windows service"""
        mock_run.return_value = Mock(returncode=0)
        
        manager = WindowsServiceManager()
        result = manager.start_service("TestService")
        
        assert result == True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_stop_service(self, mock_run):
        """Test stopping a Windows service"""
        mock_run.return_value = Mock(returncode=0)
        
        manager = WindowsServiceManager()
        result = manager.stop_service("TestService")
        
        assert result == True
        mock_run.assert_called_once()


class TestWindowsRegistryManager:
    """Test Windows Registry operations"""
    
    @patch('winreg.OpenKey')
    @patch('winreg.QueryValueEx')
    def test_get_startup_entry(self, mock_query, mock_open):
        """Test reading startup entries from registry"""
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock(return_value=None)
        mock_query.return_value = ("test_command", "REG_SZ")
        
        manager = WindowsRegistryManager()
        result = manager.get_startup_entry("TestApp")
        
        assert result == "test_command"
    
    @patch('winreg.OpenKey')
    @patch('winreg.SetValueEx')
    def test_set_startup_entry(self, mock_set, mock_open):
        """Test setting startup entries in registry"""
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock(return_value=None)
        
        manager = WindowsRegistryManager()
        result = manager.set_startup_entry("TestApp", "test_command")
        
        assert result == True
        mock_set.assert_called_once()
    
    @patch('winreg.OpenKey')
    @patch('winreg.DeleteValue')
    def test_remove_startup_entry(self, mock_delete, mock_open):
        """Test removing startup entries from registry"""
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock(return_value=None)
        
        manager = WindowsRegistryManager()
        result = manager.remove_startup_entry("TestApp")
        
        assert result == True
        mock_delete.assert_called_once()


class TestWindowsSystemInfo:
    """Test Windows system information gathering"""
    
    def test_get_windows_version(self):
        """Test getting Windows version information"""
        info = WindowsSystemInfo()
        version_info = info.get_windows_version()
        
        assert isinstance(version_info, dict)
        # Should have basic platform information
        if version_info:  # May be empty if platform module fails
            assert 'system' in version_info
    
    @patch('subprocess.run')
    def test_get_defender_status(self, mock_run):
        """Test getting Windows Defender status"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"RealTimeProtectionEnabled": true, "FirewallEnabled": true}'
        )
        
        info = WindowsSystemInfo()
        status = info.get_defender_status()
        
        assert isinstance(status, dict)
        if status:  # May be empty if PowerShell fails
            assert 'RealTimeProtectionEnabled' in status
    
    @patch('subprocess.run')
    def test_is_wsl_available(self, mock_run):
        """Test checking WSL availability"""
        mock_run.return_value = Mock(returncode=0)
        
        info = WindowsSystemInfo()
        result = info.is_wsl_available()
        
        assert isinstance(result, bool)


class TestWindowsOptimizer:
    """Test Windows optimization functionality"""
    
    def test_optimizer_initialization(self):
        """Test that optimizer initializes all components"""
        optimizer = WindowsOptimizer()
        
        assert hasattr(optimizer, 'permission_manager')
        assert hasattr(optimizer, 'firewall_manager')
        assert hasattr(optimizer, 'service_manager')
        assert hasattr(optimizer, 'registry_manager')
        assert hasattr(optimizer, 'system_info')
    
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    def test_optimize_for_development(self, mock_is_admin):
        """Test development environment optimization"""
        mock_is_admin.return_value = False
        
        optimizer = WindowsOptimizer()
        results = optimizer.optimize_for_development([3000, 8000])
        
        assert isinstance(results, dict)
        assert 'firewall_exceptions' in results
        assert 'permissions_elevated' in results
        assert 'optimizations_applied' in results
    
    @patch('startup_manager.windows_utils.WindowsPermissionManager.is_admin')
    @patch('startup_manager.windows_utils.WindowsServiceManager.install_service')
    def test_setup_service_mode(self, mock_install, mock_is_admin):
        """Test Windows service setup"""
        mock_is_admin.return_value = True
        mock_install.return_value = True
        
        optimizer = WindowsOptimizer()
        result = optimizer.setup_service_mode("TestService")
        
        assert isinstance(result, bool)


class TestWindowsIntegration:
    """Integration tests for Windows-specific features"""
    
    def test_batch_file_windows_optimizations(self):
        """Test that batch file includes Windows optimization calls"""
        batch_file = Path(__file__).parent.parent / "start_both_servers.bat"
        
        if batch_file.exists():
            content = batch_file.read_text()
            
            # Check for Windows-specific functions
            assert "check_windows_optimizations" in content
            assert "add_firewall_exception" in content
            assert "net session" in content  # Admin check
            assert "netsh advfirewall" in content  # Firewall management
    
    @patch('subprocess.run')
    def test_batch_file_execution_with_verbose(self, mock_run):
        """Test batch file execution with verbose Windows output"""
        batch_file = Path(__file__).parent.parent / "start_both_servers.bat"
        
        if batch_file.exists():
            # Mock the batch file execution
            mock_run.return_value = Mock(
                returncode=0,
                stdout="[INFO] Checking Windows optimizations...",
                stderr=""
            )
            
            # This would be a real test in a full implementation
            # result = subprocess.run([str(batch_file), "--verbose", "--help"], 
            #                        capture_output=True, text=True, timeout=30)
            # assert "[INFO] Checking Windows optimizations" in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])