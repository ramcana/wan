#!/usr/bin/env python3
"""
Alert Management CLI

Command-line interface for managing health monitoring alerts.
"""

import sys
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools" / "health-checker"))

from automated_alerting import AutomatedAlertingSystem, create_alerting_cli

if __name__ == "__main__":
    # Use the CLI from automated_alerting module
    asyncio.run(main())
