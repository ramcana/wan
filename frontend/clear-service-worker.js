// Script to clear service worker cache and force update
// Run this in browser console to clear cached service workers

async function clearServiceWorkerCache() {
  console.log('Clearing Service Worker cache...');
  
  try {
    // 1. Unregister all service workers
    if ('serviceWorker' in navigator) {
      const registrations = await navigator.serviceWorker.getRegistrations();
      console.log(`Found ${registrations.length} service worker registrations`);
      
      for (const registration of registrations) {
        console.log('Unregistering service worker:', registration.scope);
        await registration.unregister();
      }
      
      console.log('✅ All service workers unregistered');
    }
    
    // 2. Clear all caches
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      console.log(`Found ${cacheNames.length} caches:`, cacheNames);
      
      for (const cacheName of cacheNames) {
        console.log('Deleting cache:', cacheName);
        await caches.delete(cacheName);
      }
      
      console.log('✅ All caches cleared');
    }
    
    // 3. Clear localStorage and sessionStorage
    localStorage.clear();
    sessionStorage.clear();
    console.log('✅ Local storage cleared');
    
    // 4. Force reload to register new service worker
    console.log('🔄 Reloading page to register updated service worker...');
    setTimeout(() => {
      window.location.reload(true);
    }, 1000);
    
  } catch (error) {
    console.error('❌ Error clearing service worker cache:', error);
  }
}

// Run the cleanup
clearServiceWorkerCache();