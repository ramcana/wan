#!/usr/bin/env python3

# Test if we can execute the hardware_optimizer.py file directly
import sys
import traceback

try:
    exec(open('hardware_optimizer.py').read())
    print("File executed successfully")
    print(f"HardwareOptimizer defined: {'HardwareOptimizer' in locals()}")
    print(f"HardwareProfile defined: {'HardwareProfile' in locals()}")
    print(f"OptimalSettings defined: {'OptimalSettings' in locals()}")
    
    if 'HardwareOptimizer' in locals():
        optimizer = HardwareOptimizer()
        print(f"HardwareOptimizer instance created: {optimizer}")
        
except Exception as e:
    print(f"Error executing file: {e}")
    traceback.print_exc()
