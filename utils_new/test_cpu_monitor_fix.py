#!/usr/bin/env python3
"""
Test the centralized CPU monitor fix
"""

import time
import threading
from cpu_monitor import get_cpu_percent

def test_cpu_monitor():
    """Test the centralized CPU monitor"""
    print("🔍 Testing Centralized CPU Monitor")
    print("=" * 40)
    
    # Test single-threaded usage
    print("\n1. Single-threaded test:")
    for i in range(5):
        cpu = get_cpu_percent()
        print(f"   Reading {i+1}: {cpu:.1f}%")
        time.sleep(0.5)
    
    # Test multi-threaded usage (simulating multiple monitors)
    print("\n2. Multi-threaded test (simulating performance profiler + resource monitor):")
    
    results = []
    
    def monitor_thread(name, count):
        """Simulate a monitoring thread"""
        thread_results = []
        for i in range(count):
            cpu = get_cpu_percent()
            thread_results.append(cpu)
            print(f"   {name} reading {i+1}: {cpu:.1f}%")
            time.sleep(0.3)
        results.append((name, thread_results))
    
    # Start multiple threads
    thread1 = threading.Thread(target=monitor_thread, args=("Performance", 5))
    thread2 = threading.Thread(target=monitor_thread, args=("Resource", 5))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    # Analyze results
    print("\n3. Results analysis:")
    for name, readings in results:
        avg_cpu = sum(readings) / len(readings)
        max_cpu = max(readings)
        min_cpu = min(readings)
        print(f"   {name} Monitor - Avg: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%, Min: {min_cpu:.1f}%")
        
        # Check for the 100% issue
        if max_cpu > 90:
            print(f"   ⚠️  {name} Monitor still showing high readings!")
        else:
            print(f"   ✅ {name} Monitor readings look normal")
    
    print("\n✅ CPU monitor test completed")

if __name__ == "__main__":
    test_cpu_monitor()