#!/usr/bin/env python3

print("Starting import test...")

try:
    print("Importing vram_manager...")
    from vram_manager import VRAMManager, GPUInfo, VRAMConfig, VRAMDetectionError
    print("vram_manager imported successfully")
    
    print("Importing other modules...")
    import json
import logging
import os
from pathlib import Path
    from typing import Dict, List, Optional, Tuple, Any, Union
    from dataclasses import dataclass, asdict
    from datetime import datetime
    import shutil
print("All imports successful")
    
    print("Defining VRAMConfigProfile...")
    @dataclass
    class VRAMConfigProfile:
        name: str
        description: str
        manual_vram_gb: Dict[int, int]
        preferred_gpu: Optional[int] = None
    print("VRAMConfigProfile defined")
    
    print("Defining VRAMConfigManager...")
    class VRAMConfigManager:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            print("VRAMConfigManager initialized")
    print("VRAMConfigManager defined")
    
    print("Creating instance...")
    manager = VRAMConfigManager()
    print("Instance created successfully")
    
except Exception as e:
    print(f"Error during import/definition: {e}")
    import traceback
traceback.print_exc()