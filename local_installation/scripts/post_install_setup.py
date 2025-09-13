"""
Post-installation setup wizard for WAN2.2.
Handles first-run configuration, usage instructions, and optional application launch.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from base_classes import BaseInstallationComponent
from interfaces import InstallationError, ErrorCategory


class PostInstallationSetup(BaseInstallationComponent):
    """Handles post-installation setup and first-run configuration."""
    
    def __init__(self, installation_path: str):
        super().__init__(installation_path)
        self.config_path = Path(installation_path) / "config.json"
        self.first_run_marker = Path(installation_path) / ".first_run_complete"
    
    def run_first_run_wizard(self) -> bool:
        """Run the first-run configuration wizard."""
        try:
            self.logger.info("Starting first-run configuration wizard...")
            
            print("\n" + "="*60)
            print("WAN2.2 First-Run Configuration Wizard")
            print("="*60)
            print()
            
            # Check if this is actually the first run
            if self.first_run_marker.exists():
                print("First-run setup has already been completed.")
                return self._show_usage_instructions()
            
            # Load current configuration
            config = self._load_current_config()
            
            # Run configuration steps
            config = self._configure_basic_settings(config)
            config = self._configure_performance_settings(config)
            config = self._configure_output_settings(config)
            
            # Save updated configuration
            if self._save_configuration(config):
                self._mark_first_run_complete()
                print("\n✓ First-run configuration completed successfully!")
                return True
            else:
                print("\n✗ Failed to save configuration.")
                return False
                
        except Exception as e:
            self.logger.error(f"First-run wizard failed: {e}")
            print(f"\n✗ Configuration wizard failed: {e}")
            return False
    
    def _load_current_config(self) -> Dict[str, Any]:
        """Load the current configuration file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Return default configuration
                return {
                    "system": {
                        "default_quantization": "fp16",
                        "enable_offload": True,
                        "vae_tile_size": 256,
                        "max_queue_size": 5,
                        "worker_threads": 4
                    },
                    "optimization": {
                        "max_vram_usage_gb": 8,
                        "cpu_threads": 8,
                        "memory_pool_gb": 4
                    },
                    "output": {
                        "default_resolution": "720p",
                        "output_format": "mp4",
                        "quality": "high"
                    },
                    "user_preferences": {
                        "auto_launch": False,
                        "show_advanced_options": False,
                        "enable_logging": True
                    }
                }
        except Exception as e:
            self.logger.warning(f"Failed to load config, using defaults: {e}")
            return {}
    
    def _configure_basic_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure basic user preferences."""
        print("Basic Settings Configuration")
        print("-" * 30)
        
        # Auto-launch preference
        current_auto_launch = config.get("user_preferences", {}).get("auto_launch", False)
        print(f"\nCurrent auto-launch setting: {'Enabled' if current_auto_launch else 'Disabled'}")
        
        while True:
            response = input("Enable auto-launch after installation? (y/n) [current]: ").strip().lower()
            if response == '':
                break
            elif response in ['y', 'yes']:
                config.setdefault("user_preferences", {})["auto_launch"] = True
                break
            elif response in ['n', 'no']:
                config.setdefault("user_preferences", {})["auto_launch"] = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        # Advanced options preference
        current_advanced = config.get("user_preferences", {}).get("show_advanced_options", False)
        print(f"\nCurrent advanced options setting: {'Enabled' if current_advanced else 'Disabled'}")
        
        while True:
            response = input("Show advanced options in UI? (y/n) [current]: ").strip().lower()
            if response == '':
                break
            elif response in ['y', 'yes']:
                config.setdefault("user_preferences", {})["show_advanced_options"] = True
                break
            elif response in ['n', 'no']:
                config.setdefault("user_preferences", {})["show_advanced_options"] = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        return config
    
    def _configure_performance_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure performance-related settings."""
        print("\nPerformance Settings Configuration")
        print("-" * 35)
        
        # VRAM usage setting
        current_vram = config.get("optimization", {}).get("max_vram_usage_gb", 8)
        print(f"\nCurrent VRAM usage limit: {current_vram} GB")
        
        while True:
            response = input(f"Maximum VRAM usage in GB (4-16) [{current_vram}]: ").strip()
            if response == '':
                break
            try:
                vram_gb = int(response)
                if 4 <= vram_gb <= 16:
                    config.setdefault("optimization", {})["max_vram_usage_gb"] = vram_gb
                    break
                else:
                    print("Please enter a value between 4 and 16.")
            except ValueError:
                print("Please enter a valid number.")
        
        # CPU threads setting
        current_threads = config.get("optimization", {}).get("cpu_threads", 8)
        print(f"\nCurrent CPU threads: {current_threads}")
        
        while True:
            response = input(f"Number of CPU threads (1-32) [{current_threads}]: ").strip()
            if response == '':
                break
            try:
                threads = int(response)
                if 1 <= threads <= 32:
                    config.setdefault("optimization", {})["cpu_threads"] = threads
                    break
                else:
                    print("Please enter a value between 1 and 32.")
            except ValueError:
                print("Please enter a valid number.")
        
        return config
    
    def _configure_output_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure output-related settings."""
        print("\nOutput Settings Configuration")
        print("-" * 30)
        
        # Default resolution
        current_resolution = config.get("output", {}).get("default_resolution", "720p")
        print(f"\nCurrent default resolution: {current_resolution}")
        print("Available resolutions: 720p, 1080p")
        
        while True:
            response = input(f"Default resolution (720p/1080p) [{current_resolution}]: ").strip().lower()
            if response == '':
                break
            elif response in ['720p', '1080p']:
                config.setdefault("output", {})["default_resolution"] = response
                break
            else:
                print("Please enter '720p' or '1080p'.")
        
        # Output quality
        current_quality = config.get("output", {}).get("quality", "high")
        print(f"\nCurrent quality setting: {current_quality}")
        print("Available qualities: low, medium, high")
        
        while True:
            response = input(f"Output quality (low/medium/high) [{current_quality}]: ").strip().lower()
            if response == '':
                break
            elif response in ['low', 'medium', 'high']:
                config.setdefault("output", {})["quality"] = response
                break
            else:
                print("Please enter 'low', 'medium', or 'high'.")
        
        return config
    
    def _save_configuration(self, config: Dict[str, Any]) -> bool:
        """Save the configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _mark_first_run_complete(self) -> None:
        """Mark first-run setup as complete."""
        try:
            with open(self.first_run_marker, 'w', encoding='utf-8') as f:
                f.write("First-run setup completed successfully.\n")
        except Exception as e:
            self.logger.warning(f"Failed to create first-run marker: {e}")
    
    def show_usage_instructions(self) -> bool:
        """Display usage instructions and getting started guide."""
        return self._show_usage_instructions()
    
    def _show_usage_instructions(self) -> bool:
        """Display comprehensive usage instructions."""
        try:
            print("\n" + "="*60)
            print("WAN2.2 Usage Instructions")
            print("="*60)
            
            print("""
Getting Started:
================

1. Launch WAN2.2:
   • Double-click the desktop shortcut "WAN2.2 Video Generator"
   • Or use Start Menu → WAN2.2 → WAN2.2 Video Generator
   • Or run: launch_wan22.bat from the installation folder

2. Launch WAN2.2 UI:
   • Double-click the desktop shortcut "WAN2.2 UI"
   • Or use Start Menu → WAN2.2 → WAN2.2 UI
   • Or run: launch_wan22_ui.bat from the installation folder

3. Configuration:
   • Use Start Menu → WAN2.2 → WAN2.2 Configuration
   • Or edit config.json in the installation folder
   • Or run this wizard again with: python scripts/post_install_setup.py

Basic Usage:
============

Text-to-Video Generation:
• Open WAN2.2 and select "Text-to-Video" mode
• Enter your text prompt
• Choose resolution (720p or 1080p)
• Click "Generate" and wait for processing

Image-to-Video Generation:
• Open WAN2.2 and select "Image-to-Video" mode
• Upload your source image
• Enter optional text prompt for guidance
• Choose settings and click "Generate"

Tips for Best Results:
======================

• Use descriptive, detailed prompts
• Start with 720p for faster generation
• Monitor VRAM usage in Task Manager
• Check logs if generation fails
• Ensure sufficient disk space for outputs

Troubleshooting:
================

• If application won't start: Check logs/installation.log
• If generation is slow: Reduce resolution or adjust CPU threads
• If out of memory: Lower VRAM usage limit in configuration
• For support: Check the documentation or GitHub issues

File Locations:
===============

• Installation: {self.installation_path}
• Configuration: {self.config_path}
• Outputs: {self.installation_path}/outputs/
• Logs: {self.installation_path}/logs/
• Models: {self.installation_path}/models/

""")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to show usage instructions: {e}")
            return False
    
    def offer_application_launch(self) -> bool:
        """Offer to launch the application after setup."""
        try:
            print("\nLaunch Application")
            print("-" * 18)
            
            while True:
                response = input("Would you like to launch WAN2.2 now? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    return self._launch_application()
                elif response in ['n', 'no']:
                    print("\nYou can launch WAN2.2 later using the desktop shortcuts or start menu.")
                    return True
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
                    
        except Exception as e:
            self.logger.error(f"Failed to offer application launch: {e}")
            return False
    
    def _launch_application(self) -> bool:
        """Launch the main WAN2.2 application."""
        try:
            launcher_path = self.installation_path / "launch_wan22.bat"
            
            if not launcher_path.exists():
                print("✗ Application launcher not found. Please use desktop shortcuts.")
                return False
            
            print("\n✓ Launching WAN2.2...")
            
            # Launch the application in a new process
            subprocess.Popen([str(launcher_path)], shell=True)
            
            print("✓ WAN2.2 has been launched in a new window.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch application: {e}")
            print(f"✗ Failed to launch application: {e}")
            return False
    
    def run_complete_post_install_setup(self) -> bool:
        """Run the complete post-installation setup process."""
        try:
            self.logger.info("Starting complete post-installation setup...")
            
            # Run first-run wizard
            if not self.run_first_run_wizard():
                return False
            
            # Show usage instructions
            if not self.show_usage_instructions():
                return False
            
            # Offer to launch application
            if not self.offer_application_launch():
                return False
            
            print("\n" + "="*60)
            print("Post-installation setup completed successfully!")
            print("Thank you for installing WAN2.2!")
            print("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Post-installation setup failed: {e}")
            return False


def main():
    """Main entry point for post-installation setup."""
    if len(sys.argv) > 1:
        installation_path = sys.argv[1].strip('"')  # Remove quotes from path
    else:
        installation_path = Path(__file__).parent.parent
    
    setup = PostInstallationSetup(str(installation_path))
    success = setup.run_complete_post_install_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
