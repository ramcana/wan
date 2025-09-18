// Simple test to verify frontend-backend connection
const API_BASE = import.meta?.env?.VITE_API_URL || 'http://localhost:8000';

async function testConnection() {
  console.log('Testing backend connection...');
  
  try {
    // Test health endpoint
    const healthResponse = await fetch(`${API_BASE}/health`);
    const healthData = await healthResponse.json();
    console.log('✅ Health check:', healthData);
    
    // Test system health endpoint
    const systemResponse = await fetch(`${API_BASE}/api/v1/system/health`);
    const systemData = await systemResponse.json();
    console.log('✅ System health:', systemData);
    
    console.log('🎉 Backend connection successful!');
    return true;
  } catch (error) {
    console.error('❌ Backend connection failed:', error);
    return false;
  }
}

// Run the test
testConnection();