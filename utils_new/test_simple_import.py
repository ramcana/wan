#!/usr/bin/env python3

# Simple test to check if we can import the classes
try:
    from hardware_optimizer import HardwareOptimizer, HardwareProfile, OptimalSettings
    print("Import successful!")
    print(f"HardwareOptimizer: {HardwareOptimizer}")
    print(f"HardwareProfile: {HardwareProfile}")
    print(f"OptimalSettings: {OptimalSettings}")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")

# Try to import the module and check its contents
try:
    import hardware_optimizer
    print(f"Module attributes: {[attr for attr in dir(hardware_optimizer) if not attr.startswith('_')]}")
except Exception as e:
    print(f"Module import error: {e}")
