#!/usr/bin/env python3
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
