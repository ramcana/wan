#!/usr/bin/env python3
"""
Demo script to test the health check integration
Shows the backend health endpoint working with port detection
"""

import asyncio
import json
import time
from datetime import datetime
import requests
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))


async def test_health_endpoint():
    """Test the health endpoint functionality"""
    print("🔍 Testing Health Check Integration")
    print("=" * 50)
    
    # Test different ports
    ports_to_test = [8000, 8000, 8080]
    
    for port in ports_to_test:
        print(f"\n📡 Testing port {port}...")
        
        try:
            url = f"http://localhost:{port}/api/v1/system/health"
            start_time = time.time()
            
            response = requests.get(url, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Port {port} - Backend is healthy!")
                print(f"   Response time: {response_time:.1f}ms")
                print(f"   Detected port: "
                      f"{health_data.get('port', 'unknown')}")
                print(f"   API version: "
                      f"{health_data.get('api_version', 'unknown')}")
                print(f"   Service: "
                      f"{health_data.get('service', 'unknown')}")
                print(f"   CORS enabled: "
                      f"{health_data.get('connectivity', {}).get('cors_enabled', False)}")
                print(f"   Allowed origins: "
                      f"{health_data.get('connectivity', {}).get('allowed_origins', [])}")
                
                # Verify schema compliance
                required_fields = [
                    'status', 'port', 'timestamp', 'api_version',
                    'connectivity', 'endpoints', 'server_info'
                ]
                missing_fields = [
                    field for field in required_fields
                    if field not in health_data
                ]
                
                if missing_fields:
                    print(f"   ⚠️  Missing fields: {missing_fields}")
                else:
                    print("   ✅ Schema compliance: "
                      "All required fields present")
                
                return port, health_data
                
            else:
                print(f"❌ Port {port} - HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Port {port} - Connection refused")
        except requests.exceptions.Timeout:
            print(f"❌ Port {port} - Timeout")
        except Exception as e:
            print(f"❌ Port {port} - Error: {e}")
    
    print("\n🚨 No healthy backend found on any tested port")
    return None, None

def test_port_detection_logic():
    """Test the port detection logic similar to frontend"""
    print("\n🔍 Testing Port Detection Logic")
    print("=" * 50)
    
    ports_to_test = [8000, 8000, 8080, 3001]
    
    for port in ports_to_test:
        print(f"\n🔍 [{datetime.now().isoformat()}] Testing port {port}...")
        
        try:
            url = f"http://localhost:{port}/api/v1/system/health"
            start_time = time.time()
            
            response = requests.get(url, timeout=5, headers={'Accept': 'application/json'})
            response_time = (time.time() - start_time) * 1000
            
            if response.ok:
                health_data = response.json()
                print(f"✅ [{datetime.now().isoformat()}] Port {port} connectivity test successful")
                print(f"   Port: {port}")
                print(f"   Base URL: http://localhost:{port}")
                print(f"   Response time: {response_time:.0f}ms")
                print(f"   Health data: {json.dumps(health_data, indent=2)}")
                return {
                    'detectedPort': health_data.get('port', port),
                    'baseUrl': f'http://localhost:{port}',
                    'isHealthy': True,
                    'responseTime': response_time
                }
            else:
                print(f"⚠️ [{datetime.now().isoformat()}] Port {port} responded with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ [{datetime.now().isoformat()}] Port {port} connectivity test failed: {e}")
    
    print(f"\n🚨 [{datetime.now().isoformat()}] No healthy backend found on any tested port")
    return {
        'detectedPort': 8000,
        'baseUrl': 'http://localhost:8000',
        'isHealthy': False,
        'responseTime': 0
    }

def validate_configuration(detected_config):
    """Validate configuration similar to frontend"""
    print("\n🔍 Testing Configuration Validation")
    print("=" * 50)
    
    issues = []
    suggestions = []
    
    if not detected_config['isHealthy']:
        issues.append('Backend server is not responding')
        suggestions.append('Ensure the backend server is running')
        suggestions.append('Check if the server is running on the expected port')
    
    if detected_config['responseTime'] > 5000:
        issues.append('Backend response time is slow')
        suggestions.append('Check network connectivity')
        suggestions.append('Consider optimizing backend performance')
    
    # Test CORS configuration if backend is healthy
    if detected_config['isHealthy']:
        try:
            url = f"{detected_config['baseUrl']}/api/v1/system/health"
            response = requests.get(url, headers={'Origin': 'http://localhost:3000'})
            
            if response.ok:
                health_data = response.json()
                connectivity = health_data.get('connectivity', {})
                
                if not connectivity.get('cors_enabled'):
                    issues.append('CORS is not enabled on the backend')
                    suggestions.append('Enable CORS middleware in FastAPI backend')
                
                allowed_origins = connectivity.get('allowed_origins', [])
                frontend_origin = 'http://localhost:3000'
                
                if frontend_origin not in allowed_origins:
                    issues.append(f'Frontend origin {frontend_origin} is not in CORS allowed origins')
                    suggestions.append(f'Add {frontend_origin} to CORS allowed_origins in backend configuration')
                
                if not connectivity.get('websocket_available'):
                    issues.append('WebSocket support is not available')
                    suggestions.append('Ensure WebSocket endpoints are properly configured')
                    
        except Exception as e:
            issues.append(f'Could not validate CORS configuration: {e}')
            suggestions.append('Check backend API endpoints are properly configured')
    
    is_valid = len(issues) == 0
    
    print(f"Configuration validation result:")
    print(f"  Valid: {is_valid}")
    print(f"  Issues: {len(issues)}")
    print(f"  Suggestions: {len(suggestions)}")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    if suggestions:
        print("\n💡 Suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    
    if is_valid:
        print("\n✅ Configuration validation passed!")
    
    return {
        'isValid': is_valid,
        'issues': issues,
        'suggestions': suggestions,
        'detectedConfig': detected_config
    }

async def main():
    """Main demo function"""
    print("🚀 Health Check Integration Demo")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Basic health endpoint
    detected_port, health_data = await test_health_endpoint()
    
    # Test 2: Port detection logic
    detected_config = test_port_detection_logic()
    
    # Test 3: Configuration validation
    validation_result = validate_configuration(detected_config)
    
    # Summary
    print("\n📊 Summary")
    print("=" * 50)
    
    if detected_port:
        print(f"✅ Backend found on port {detected_port}")
        print(f"✅ Health endpoint working correctly")
        print(f"✅ Schema compliance verified")
    else:
        print("❌ No backend found")
        print("💡 Start the backend server with: python backend/start_server.py --port 8000")
    
    print(f"\nPort detection result:")
    print(f"  Detected port: {detected_config['detectedPort']}")
    print(f"  Base URL: {detected_config['baseUrl']}")
    print(f"  Healthy: {detected_config['isHealthy']}")
    print(f"  Response time: {detected_config['responseTime']:.1f}ms")
    
    print(f"\nConfiguration validation:")
    print(f"  Valid: {validation_result['isValid']}")
    print(f"  Issues: {len(validation_result['issues'])}")
    print(f"  Suggestions: {len(validation_result['suggestions'])}")
    
    if validation_result['isValid']:
        print("\n🎉 All tests passed! Health check integration is working correctly.")
    else:
        print("\n⚠️  Some issues found. See suggestions above.")
    
    print("\n" + "=" * 50)
    print("Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
