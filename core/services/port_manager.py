#!/usr/bin/env python3
"""
Port Management for WAN2.2 UI
Handles port conflicts and finds available ports automatically
"""

import socket
import subprocess
import sys
import os

def is_port_in_use(port, host='127.0.0.1'):
    """Check if a port is currently in use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False

def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return None

def kill_process_on_port(port):
    """Kill any process using the specified port (Windows)"""
    try:
        # Find process using the port
        result = subprocess.run([
            'netstat', '-ano'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            # Kill the process
                            subprocess.run([
                                'taskkill', '/F', '/PID', pid
                            ], capture_output=True, shell=True)
                            print(f"✅ Killed process {pid} using port {port}")
                            return True
                        except Exception as e:
                            print(f"⚠️  Could not kill process {pid}: {e}")
                            return False
        return False
    except Exception as e:
        print(f"⚠️  Error killing process on port {port}: {e}")
        return False

def get_port_info(port):
    """Get information about what's using a port"""
    try:
        result = subprocess.run([
            'netstat', '-ano'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        
                        # Get process name
                        try:
                            proc_result = subprocess.run([
                                'tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'
                            ], capture_output=True, text=True, shell=True)
                            
                            if proc_result.returncode == 0:
                                lines = proc_result.stdout.split('\n')
                                if len(lines) > 1:
                                    process_name = lines[1].split(',')[0].strip('"')
                                    return {
                                        'pid': pid,
                                        'process_name': process_name,
                                        'port': port
                                    }
                        except Exception:
                            pass
                        
                        return {
                            'pid': pid,
                            'process_name': 'Unknown',
                            'port': port
                        }
        return None
    except Exception as e:
        print(f"⚠️  Error getting port info: {e}")
        return None

def resolve_port_conflict(preferred_port=7860, auto_kill=False):
    """Resolve port conflicts by finding alternative or killing existing process"""
    print(f"🔍 Checking port {preferred_port}...")
    
    if not is_port_in_use(preferred_port):
        print(f"✅ Port {preferred_port} is available")
        return preferred_port
    
    # Port is in use, get info about what's using it
    port_info = get_port_info(preferred_port)
    if port_info:
        print(f"⚠️  Port {preferred_port} is in use by {port_info['process_name']} (PID: {port_info['pid']})")
        
        # Check if it's likely another instance of our app
        if 'python' in port_info['process_name'].lower() or 'wan' in port_info['process_name'].lower():
            print("🤔 Detected possible previous WAN2.2 instance")
            
            if auto_kill:
                print("🔧 Attempting to kill previous instance...")
                if kill_process_on_port(preferred_port):
                    if not is_port_in_use(preferred_port):
                        print(f"✅ Port {preferred_port} is now available")
                        return preferred_port
    
    # Find alternative port
    print("🔍 Searching for alternative port...")
    alternative_port = find_available_port(preferred_port + 1)
    
    if alternative_port:
        print(f"✅ Found available port: {alternative_port}")
        return alternative_port
    else:
        print("❌ No available ports found in range")
        return None

def main():
    """Main port management function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WAN2.2 Port Manager')
    parser.add_argument('--port', type=int, default=7860, help='Preferred port (default: 7860)')
    parser.add_argument('--auto-kill', action='store_true', help='Automatically kill processes on preferred port')
    parser.add_argument('--check-only', action='store_true', help='Only check port status, don\'t resolve conflicts')
    
    args = parser.parse_args()
    
    print("WAN2.2 Port Manager")
    print("=" * 30)
    
    if args.check_only:
        if is_port_in_use(args.port):
            port_info = get_port_info(args.port)
            if port_info:
                print(f"❌ Port {args.port} is in use by {port_info['process_name']} (PID: {port_info['pid']})")
            else:
                print(f"❌ Port {args.port} is in use")
            return 1
        else:
            print(f"✅ Port {args.port} is available")
            return 0
    
    available_port = resolve_port_conflict(args.port, args.auto_kill)
    
    if available_port:
        print(f"\n🎉 Ready to use port: {available_port}")
        
        # Set environment variable for the application
        os.environ['GRADIO_SERVER_PORT'] = str(available_port)
        print(f"✅ Set GRADIO_SERVER_PORT={available_port}")
        
        return 0
    else:
        print("\n❌ Could not resolve port conflict")
        return 1

if __name__ == "__main__":
    sys.exit(main())
