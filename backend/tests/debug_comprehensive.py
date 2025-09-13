"""
Debug script to check comprehensive integration suite
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

print("Attempting to import comprehensive integration suite...")

try:
    import test_comprehensive_integration_suite as t
    print("Import successful!")
    print("Available attributes:", [x for x in dir(t) if not x.startswith('_')])
    print("Test classes:", [x for x in dir(t) if x.startswith('Test')])
except Exception as e:
    print("Import failed:", e)
    import traceback
    traceback.print_exc()

print("\nTrying to execute the file directly...")
try:
    with open('test_comprehensive_integration_suite.py', 'r') as f:
        content = f.read()
    
    # Check for syntax errors
    compile(content, 'test_comprehensive_integration_suite.py', 'exec')
    print("File compiles successfully!")
    
    # Execute in a namespace
    namespace = {}
    exec(content, namespace)
    print("Execution successful!")
    print("Classes in namespace:", [x for x in namespace if x.startswith('Test')])
    
except Exception as e:
    print("Execution failed:", e)
    import traceback
    traceback.print_exc()
