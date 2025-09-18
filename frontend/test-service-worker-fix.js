// Test script to verify service worker stream handling fix
// Run this in browser console to test the fix

async function testServiceWorkerFix() {
  console.log('🔧 Testing Service Worker Stream Handling Fix...');
  console.log('='.repeat(50));
  
  // Test 1: Verify service worker is registered
  console.log('1️⃣ Checking Service Worker registration...');
  if ('serviceWorker' in navigator) {
    const registration = await navigator.serviceWorker.getRegistration();
    if (registration) {
      console.log('✅ Service Worker is registered');
      console.log('   Scope:', registration.scope);
      console.log('   State:', registration.active?.state);
    } else {
      console.log('❌ Service Worker not registered');
      console.log('💡 Try refreshing the page or running clear-service-worker.js first');
      return;
    }
  } else {
    console.log('❌ Service Worker not supported in this browser');
    return;
  }
  
  // Test 2: Check backend connectivity
  console.log('\n2️⃣ Testing backend connectivity...');
  try {
    const healthResponse = await fetch('/api/v1/system/health');
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('✅ Backend is reachable');
      console.log('   Health status:', healthData.status);
      console.log('   API version:', healthData.api_version);
    } else {
      console.log('⚠️ Backend health check failed with status:', healthResponse.status);
    }
  } catch (error) {
    console.log('❌ Backend is not reachable:', error.message);
    console.log('💡 Make sure the backend server is running on port 8000');
    console.log('💡 Check if VITE_API_URL is set correctly in frontend/.env');
  }
  
  // Test 3: Test POST request that previously failed
  console.log('\n3️⃣ Testing POST request with stream handling...');
  try {
    const testData = {
      prompt: "Test prompt for stream handling fix",
      options: { test: true, timestamp: Date.now() }
    };
    
    console.log('   Sending POST request to /api/v1/prompt/enhance...');
    console.log('   Request data:', testData);
    
    const response = await fetch('/api/v1/prompt/enhance', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });
    
    console.log('   Response status:', response.status);
    console.log('   Response headers:', Object.fromEntries(response.headers.entries()));
    
    if (response.ok) {
      console.log('✅ POST request succeeded - stream handling fixed!');
      const data = await response.json();
      console.log('   Response data:', data);
    } else {
      console.log('⚠️ POST request failed with status:', response.status);
      const errorText = await response.text();
      console.log('   Error response:', errorText);
    }
    
  } catch (error) {
    if (error.message.includes('body stream already read')) {
      console.log('❌ Stream consumption error still exists:', error.message);
      console.log('💡 The service worker fix may not be applied correctly');
    } else if (error.message.includes('Failed to fetch')) {
      console.log('❌ Network error:', error.message);
      console.log('💡 Check if backend is running and CORS is configured');
    } else {
      console.log('⚠️ Other error occurred:', error.message);
    }
  }
  
  // Test 4: Check service worker control
  console.log('\n4️⃣ Checking Service Worker control...');
  const swController = navigator.serviceWorker.controller;
  if (swController) {
    console.log('✅ Service Worker is controlling this page');
    console.log('   Script URL:', swController.scriptURL);
  } else {
    console.log('⚠️ Service Worker is not controlling this page');
    console.log('💡 Try refreshing the page or clearing cache');
  }
  
  // Test 5: Test multiple rapid requests (stress test)
  console.log('\n5️⃣ Testing multiple rapid requests...');
  try {
    const promises = [];
    for (let i = 0; i < 3; i++) {
      promises.push(
        fetch('/api/v1/system/health').then(r => ({ index: i, status: r.status }))
      );
    }
    
    const results = await Promise.all(promises);
    const successCount = results.filter(r => r.status === 200).length;
    console.log(`✅ ${successCount}/3 rapid requests succeeded`);
    
    if (successCount === 3) {
      console.log('✅ Stream handling is working correctly under load');
    } else {
      console.log('⚠️ Some requests failed - may indicate remaining issues');
    }
    
  } catch (error) {
    console.log('❌ Rapid request test failed:', error.message);
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('🏁 Service Worker test completed');
  console.log('💡 If tests fail, try:');
  console.log('   1. Run clear-service-worker.js');
  console.log('   2. Refresh the page');
  console.log('   3. Check browser console for errors');
  console.log('   4. Verify backend is running on port 8000');
}

// Auto-run the test
console.log('🚀 Starting Service Worker Fix Test...');
testServiceWorkerFix().catch(console.error);