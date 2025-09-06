#!/usr/bin/env python3

print("Starting debug...")

try:
    from dataclasses import dataclass
    print("✓ dataclass import successful")
except Exception as e:
    print(f"❌ dataclass import failed: {e}")

try:
    @dataclass
    class TestProfile:
        name: str
        value: int
    
    print("✓ dataclass definition successful")
    
    test = TestProfile("test", 42)
    print(f"✓ dataclass instantiation successful: {test}")
    
except Exception as e:
    print(f"❌ dataclass definition/instantiation failed: {e}")

print("Reading hardware_optimizer.py content...")
try:
    with open('hardware_optimizer.py', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    print("Checking for class definitions...")
    
    if '@dataclass' in content:
        print("✓ Found @dataclass decorators")
    else:
        print("❌ No @dataclass decorators found")
    
    if 'class HardwareProfile:' in content:
        print("✓ Found HardwareProfile class definition")
    else:
        print("❌ No HardwareProfile class definition found")
    
    if 'class HardwareOptimizer:' in content:
        print("✓ Found HardwareOptimizer class definition")
    else:
        print("❌ No HardwareOptimizer class definition found")
        
except Exception as e:
    print(f"❌ Error reading file: {e}")

print("Attempting to execute file...")
try:
    exec(open('hardware_optimizer.py').read())
    print("✓ File executed without errors")
    
    # Check if classes are defined in local scope
    if 'HardwareProfile' in locals():
        print("✓ HardwareProfile is in local scope")
    else:
        print("❌ HardwareProfile not in local scope")
        
    if 'HardwareOptimizer' in locals():
        print("✓ HardwareOptimizer is in local scope")
    else:
        print("❌ HardwareOptimizer not in local scope")
        
except Exception as e:
    print(f"❌ Error executing file: {e}")
    import traceback
    traceback.print_exc()