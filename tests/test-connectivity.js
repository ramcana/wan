// Node.js test script to verify backend connectivity
const https = require('http');

async function testBackendConnectivity() {
  console.log('🔧 Testing Backend Connectivity...');
  console.log('='.repeat(50));
  
  // Test 1: Health endpoint
  console.log('1️⃣ Testing health endpoint...');
  try {
    const healthResponse = await fetch('http://127.0.0.1:9000/api/v1/system/health');
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('✅ Backend health check passed');
      console.log('   Status:', healthData.status);
      console.log('   API Version:', healthData.api_version);
    } else {
      console.log('❌ Health check failed with status:', healthResponse.status);
    }
  } catch (error) {
    console.log('❌ Health check error:', error.message);
  }
  
  // Test 2: POST endpoint that was failing
  console.log('\n2️⃣ Testing POST endpoint...');
  try {
    const testData = {
      prompt: "Test prompt for connectivity check",
      options: { test: true, timestamp: Date.now() }
    };
    
    const response = await fetch('http://127.0.0.1:9000/api/v1/prompt/enhance', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ POST request succeeded');
      console.log('   Original prompt:', data.original_prompt);
      console.log('   Enhanced prompt:', data.enhanced_prompt?.substring(0, 100) + '...');
    } else {
      console.log('❌ POST request failed with status:', response.status);
      const errorText = await response.text();
      console.log('   Error:', errorText);
    }
  } catch (error) {
    console.log('❌ POST request error:', error.message);
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('🏁 Backend connectivity test completed');
}

// Run the test
testBackendConnectivity().catch(console.error);