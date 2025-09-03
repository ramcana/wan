#!/usr/bin/env python3

try:
    print("Testing imports...")
    
    print("1. Testing basic imports...")
    import logging
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
    from dataclasses import dataclass, field
    from enum import Enum
    import json
from pathlib import Path
    import threading
import time
print("   Basic imports successful")
    
    print("2. Testing hardware_parameter_recommender import...")
    import hardware_parameter_recommender as hpr
print(f"   Module imported: {hpr}")
    print(f"   Module attributes: {[x for x in dir(hpr) if not x.startswith('_')]}")
    
    print("3. Testing specific class imports...")
    try:
        from hardware_parameter_recommender import HardwareClass
        print("   HardwareClass imported successfully")
    except ImportError as e:
        print(f"   HardwareClass import failed: {e}")
    
    try:
        from hardware_parameter_recommender import HardwareParameterRecommender
        print("   HardwareParameterRecommender imported successfully")
    except ImportError as e:
        print(f"   HardwareParameterRecommender import failed: {e}")
    
    print("4. Testing module execution...")
    exec(open('hardware_parameter_recommender.py').read())
    print("   Module executed successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
traceback.print_exc()