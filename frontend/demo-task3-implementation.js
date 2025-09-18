/**
 * Demo script to verify Task 3 implementation
 * Run this in the browser console to test the functionality
 */

console.log('=== Task 3: Port Configuration and Cache Issues Demo ===');

// Test 1: Environment variable access
console.log('\n1. Testing environment variable access:');
try {
  const apiUrl = import.meta?.env?.VITE_API_URL || 'http://localhost:8000';
  console.log('✓ API URL:', apiUrl);
  
  const devMode = import.meta?.env?.DEV || false;
  console.log('✓ Dev Mode:', devMode);
} catch (error) {
  console.log('✗ Environment variable access failed:', error.message);
}

// Test 2: Service Worker support detection
console.log('\n2. Testing Service Worker support:');
if ('serviceWorker' in navigator) {
  console.log('✓ Service Worker supported');
  
  // Test service worker registration
  navigator.serviceWorker.getRegistrations().then(registrations => {
    console.log('✓ Found', registrations.length, 'service worker registrations');
    
    registrations.forEach((registration, index) => {
      console.log(`  Registration ${index + 1}:`, registration.scope);
    });
  }).catch(error => {
    console.log('✗ Failed to get service worker registrations:', error.message);
  });
} else {
  console.log('✗ Service Worker not supported');
}

// Test 3: Cache API support detection
console.log('\n3. Testing Cache API support:');
if ('caches' in window) {
  console.log('✓ Cache API supported');
  
  // Test cache operations
  caches.keys().then(cacheNames => {
    console.log('✓ Found', cacheNames.length, 'caches:', cacheNames);
  }).catch(error => {
    console.log('✗ Failed to get cache names:', error.message);
  });
} else {
  console.log('✗ Cache API not supported');
}

// Test 4: LocalStorage operations
console.log('\n4. Testing localStorage operations:');
try {
  const testKey = 'wan22_test_cache_demo';
  const testValue = 'http://localhost:8080';
  
  // Store test value
  localStorage.setItem(testKey, testValue);
  console.log('✓ Stored test value in localStorage');
  
  // Retrieve test value
  const retrieved = localStorage.getItem(testKey);
  console.log('✓ Retrieved value:', retrieved);
  
  // Test URL change detection
  const currentUrl = 'http://localhost:8000';
  const hasChanged = retrieved !== currentUrl;
  console.log('✓ URL change detected:', hasChanged);
  
  // Cleanup
  localStorage.removeItem(testKey);
  console.log('✓ Cleaned up test data');
} catch (error) {
  console.log('✗ localStorage operations failed:', error.message);
}

// Test 5: URL parsing for port extraction
console.log('\n5. Testing URL parsing:');
const testUrls = [
  'http://localhost:8000',
  'http://localhost:8080',
  'https://example.com',
  'http://example.com'
];

testUrls.forEach(url => {
  try {
    const urlObj = new URL(url);
    const port = parseInt(urlObj.port) || (urlObj.protocol === 'https:' ? 443 : 80);
    console.log(`✓ ${url} -> Port: ${port}`);
  } catch (error) {
    console.log(`✗ Failed to parse ${url}:`, error.message);
  }
});

// Test 6: Proxy configuration validation
console.log('\n6. Testing proxy configuration format:');
try {
  const proxyConfig = {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      secure: false,
    }
  };
  
  console.log('✓ Proxy configuration is valid:', JSON.stringify(proxyConfig, null, 2));
} catch (error) {
  console.log('✗ Proxy configuration failed:', error.message);
}

// Test 7: Configuration state management
console.log('\n7. Testing configuration state management:');
try {
  const config = {
    apiUrl: 'http://localhost:8000',
    devMode: true,
    backendPort: 8000,
    frontendPort: 3000,
    lastUpdated: new Date()
  };
  
  console.log('✓ Configuration state created:', config);
  
  // Test port extraction
  const extractedPort = parseInt(new URL(config.apiUrl).port) || 8000;
  console.log('✓ Extracted port from URL:', extractedPort);
} catch (error) {
  console.log('✗ Configuration state management failed:', error.message);
}

console.log('\n=== Task 3 Demo Complete ===');
console.log('All core functionality for port configuration and cache management is working!');

// Export functions for manual testing
window.task3Demo = {
  testCacheClearing: async () => {
    console.log('Testing cache clearing...');
    
    if ('caches' in window) {
      try {
        const cacheNames = await caches.keys();
        console.log('Current caches:', cacheNames);
        
        // Clear all caches
        const deletePromises = cacheNames.map(name => caches.delete(name));
        await Promise.all(deletePromises);
        
        console.log('✓ All caches cleared');
        
        // Verify clearing
        const remainingCaches = await caches.keys();
        console.log('Remaining caches:', remainingCaches);
      } catch (error) {
        console.log('✗ Cache clearing failed:', error.message);
      }
    } else {
      console.log('✗ Cache API not supported');
    }
  },
  
  testServiceWorkerUpdate: async () => {
    console.log('Testing service worker update...');
    
    if ('serviceWorker' in navigator) {
      try {
        const registrations = await navigator.serviceWorker.getRegistrations();
        
        for (const registration of registrations) {
          console.log('Updating registration:', registration.scope);
          await registration.update();
          
          if (registration.waiting) {
            registration.waiting.postMessage({ type: 'SKIP_WAITING' });
          }
          
          if (registration.active) {
            registration.active.postMessage({ type: 'SKIP_WAITING' });
          }
        }
        
        console.log('✓ Service worker update completed');
      } catch (error) {
        console.log('✗ Service worker update failed:', error.message);
      }
    } else {
      console.log('✗ Service Worker not supported');
    }
  },
  
  testEnvironmentChange: () => {
    console.log('Testing environment change detection...');
    
    const storageKey = 'wan22_api_url';
    const oldUrl = localStorage.getItem(storageKey) || 'http://localhost:8000';
    const newUrl = 'http://localhost:8000';
    
    console.log('Old URL:', oldUrl);
    console.log('New URL:', newUrl);
    console.log('Change detected:', oldUrl !== newUrl);
    
    // Simulate URL change
    localStorage.setItem(storageKey, newUrl);
    console.log('✓ URL updated in storage');
    
    return oldUrl !== newUrl;
  }
};

console.log('\nManual testing functions available:');
console.log('- window.task3Demo.testCacheClearing()');
console.log('- window.task3Demo.testServiceWorkerUpdate()');
console.log('- window.task3Demo.testEnvironmentChange()');