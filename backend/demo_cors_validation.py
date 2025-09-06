#!/usr/bin/env python3
"""
Demo script to test CORS validation functionality
"""

import asyncio
import aiohttp
import json
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_cors_endpoints():
    """Test CORS validation endpoints"""
    base_url = "http://localhost:9000"
    
    # Test endpoints to check
    endpoints = [
        "/api/v1/system/cors/validate",
        "/api/v1/system/cors/test",
        "/api/v1/system/health"
    ]
    
    print("🔍 Testing CORS validation endpoints...")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            
            try:
                print(f"\n📍 Testing: {endpoint}")
                
                # Test GET request
                async with session.get(url, headers={
                    "Origin": "http://localhost:3000",
                    "Content-Type": "application/json"
                }) as response:
                    print(f"   GET Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        if "cors_valid" in data:
                            print(f"   CORS Valid: {data['cors_valid']}")
                        if "errors" in data and data["errors"]:
                            print(f"   Errors: {data['errors']}")
                        if "configuration" in data:
                            print(f"   Allowed Origins: {data['configuration'].get('allowed_origins', [])}")
                    
                    # Check CORS headers
                    cors_headers = {k: v for k, v in response.headers.items() 
                                  if k.lower().startswith('access-control')}
                    if cors_headers:
                        print(f"   CORS Headers: {cors_headers}")
                
                # Test OPTIONS (preflight) request for POST endpoints
                if endpoint == "/api/v1/system/cors/test":
                    async with session.options(url, headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type"
                    }) as preflight_response:
                        print(f"   OPTIONS Status: {preflight_response.status}")
                        preflight_headers = {k: v for k, v in preflight_response.headers.items() 
                                           if k.lower().startswith('access-control')}
                        if preflight_headers:
                            print(f"   Preflight Headers: {preflight_headers}")
                
                # Test POST request for test endpoint
                if endpoint == "/api/v1/system/cors/test":
                    async with session.post(url, 
                                          json={"test": "data"},
                                          headers={
                                              "Origin": "http://localhost:3000",
                                              "Content-Type": "application/json"
                                          }) as post_response:
                        print(f"   POST Status: {post_response.status}")
                        if post_response.status == 200:
                            post_data = await post_response.json()
                            print(f"   POST Response: {post_data.get('message', 'No message')}")
                
            except aiohttp.ClientError as e:
                print(f"   ❌ Connection Error: {e}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ CORS validation testing completed!")

def test_cors_validator_directly():
    """Test CORS validator functionality directly"""
    from core.cors_validator import CORSValidator, generate_cors_error_response
    
    print("\n🧪 Testing CORS validator directly...")
    print("=" * 60)
    
    validator = CORSValidator()
    
    # Test origin validation
    test_origins = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:8080",
        "https://example.com",
        "invalid-url",
        "*"
    ]
    
    print("\n📋 Origin Validation Results:")
    for origin in test_origins:
        is_valid = validator._is_valid_origin(origin)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"   {origin:<25} {status}")
    
    # Test error message generation
    print("\n💬 Error Message Examples:")
    test_cases = [
        ("unknown", "GET"),
        ("http://localhost:8080", "POST"),
        ("http://localhost:3000", "OPTIONS")
    ]
    
    for origin, method in test_cases:
        message = validator.generate_cors_error_message(origin, method)
        print(f"   {origin} + {method}:")
        print(f"      {message}")
    
    # Test resolution steps
    print("\n🔧 Resolution Steps Example:")
    response = generate_cors_error_response("http://localhost:8080", "POST")
    print(f"   Error Type: {response['error']}")
    print(f"   Steps Count: {len(response['resolution_steps'])}")
    for i, step in enumerate(response['resolution_steps'][:2], 1):
        print(f"   Step {i}: {step['step']} (Priority: {step['priority']})")
    
    # Test configuration suggestions
    print("\n⚙️  Configuration Suggestions:")
    suggestions = validator.get_cors_configuration_suggestions()
    print(f"   Allowed Origins: {suggestions['allow_origins']}")
    print(f"   Allow Credentials: {suggestions['allow_credentials']}")
    print(f"   Allow Methods: {suggestions['allow_methods']}")
    print(f"   Allow Headers: {suggestions['allow_headers']}")

if __name__ == "__main__":
    print("🚀 CORS Validation Demo")
    print("=" * 60)
    
    # Test validator directly first
    test_cors_validator_directly()
    
    # Test endpoints if server is running
    print("\n🌐 Testing live endpoints...")
    print("Note: Make sure the FastAPI server is running on localhost:9000")
    
    try:
        asyncio.run(test_cors_endpoints())
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error testing endpoints: {e}")
        print("Make sure the FastAPI server is running on localhost:9000")