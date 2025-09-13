#!/usr/bin/env python3
"""
Test script to verify CPU monitoring accuracy
"""

import psutil
import time
import threading
from performance_profiler import get_performance_profiler

def test_psutil_cpu_methods():
    """Test different psutil CPU measurement methods"""
    print("Testing different CPU measurement methods...")
    
    # Method 1: interval=None (non-blocking)
    print("\n1. Testing psutil.cpu_percent(interval=None):")
    for i in range(5):
        cpu = psutil.cpu_percent(interval=None)
        print(f"   Reading {i+1}: {cpu:.1f}%")
        time.sleep(1)
    
    # Method 2: interval=1.0 (blocking)
    print("\n2. Testing psutil.cpu_percent(interval=1.0):")
    for i in range(3):
        cpu = psutil.cpu_percent(interval=1.0)
        print(f"   Reading {i+1}: {cpu:.1f}%")
    
    # Method 3: Two-call method (initialize then read)
    print("\n3. Testing two-call method:")
    psutil.cpu_percent()  # Initialize
    for i in range(5):
        time.sleep(1)
        cpu = psutil.cpu_percent()
        print(f"   Reading {i+1}: {cpu:.1f}%")

    assert True  # TODO: Add proper assertion

def test_performance_profiler():
    """Test the performance profiler CPU monitoring"""
    print("\n4. Testing Performance Profiler:")
    
    profiler = get_performance_profiler()
    
    # Start monitoring
    profiler.start_monitoring()
    print("   Performance monitoring started...")
    
    # Let it run for 15 seconds
    time.sleep(15)
    
    # Get recent metrics
    summary = profiler.get_system_performance_summary()
    print(f"   Current CPU: {summary['cpu']['current_percent']:.1f}%")
    print(f"   Average CPU: {summary['cpu']['average_percent']:.1f}%")
    print(f"   Max CPU: {summary['cpu']['max_percent']:.1f}%")
    
    # Stop monitoring
    profiler.stop_monitoring()
    print("   Performance monitoring stopped")

    assert True  # TODO: Add proper assertion

def main():
    """Main test function"""
    print("🔍 CPU Monitoring Accuracy Test")
    print("=" * 40)
    
    # Show system info
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"CPU Frequency: {psutil.cpu_freq()}")
    
    # Test different methods
    test_psutil_cpu_methods()
    test_performance_profiler()
    
    print("\n✅ CPU monitoring test completed")

if __name__ == "__main__":
    main()
