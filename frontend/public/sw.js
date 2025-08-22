// Service Worker for Wan2.2 React Frontend
// Provides offline functionality and request queuing

const CACHE_NAME = 'wan22-v1';
const STATIC_CACHE_NAME = 'wan22-static-v1';
const DYNAMIC_CACHE_NAME = 'wan22-dynamic-v1';

// Files to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html',
  // Add other static assets as needed
];

// API endpoints that should be cached
const CACHEABLE_APIS = [
  '/api/v1/system/stats',
  '/api/v1/outputs',
  '/api/v1/queue',
];

// Queue for offline requests
let offlineQueue = [];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...');
  
  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE_NAME).then((cache) => {
        console.log('Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      }),
      // Create offline page
      caches.open(STATIC_CACHE_NAME).then((cache) => {
        return cache.add('/offline.html');
      })
    ])
  );
  
  // Force activation of new service worker
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE_NAME && cacheName !== DYNAMIC_CACHE_NAME) {
            console.log('Service Worker: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  
  // Take control of all pages
  self.clients.claim();
});

// Fetch event - handle requests
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Handle different types of requests
  if (request.method === 'GET') {
    if (url.pathname.startsWith('/api/')) {
      // Handle API requests
      event.respondWith(handleApiRequest(request));
    } else {
      // Handle static assets and pages
      event.respondWith(handleStaticRequest(request));
    }
  } else {
    // Handle POST/PUT/DELETE requests (potentially offline)
    event.respondWith(handleMutationRequest(request));
  }
});

// Handle static requests (HTML, CSS, JS, images)
async function handleStaticRequest(request) {
  try {
    // Try network first for HTML pages
    if (request.destination === 'document') {
      try {
        const networkResponse = await fetch(request);
        // Cache successful responses
        if (networkResponse.ok) {
          const cache = await caches.open(DYNAMIC_CACHE_NAME);
          cache.put(request, networkResponse.clone());
        }
        return networkResponse;
      } catch (error) {
        // Fallback to cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
          return cachedResponse;
        }
        // Return offline page
        return caches.match('/offline.html');
      }
    }
    
    // For other static assets, try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Try network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('Service Worker: Static request failed:', error);
    
    // Return cached version or offline page
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return caches.match('/offline.html');
  }
}

// Handle API requests
async function handleApiRequest(request) {
  const url = new URL(request.url);
  
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    // Cache successful GET responses for specific endpoints
    if (networkResponse.ok && CACHEABLE_APIS.some(api => url.pathname.includes(api))) {
      const cache = await caches.open(DYNAMIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('Service Worker: API request failed, trying cache:', error);
    
    // Try to return cached response
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      // Add offline indicator to response
      const response = cachedResponse.clone();
      const data = await response.json();
      data._offline = true;
      data._cachedAt = new Date().toISOString();
      
      return new Response(JSON.stringify(data), {
        status: 200,
        statusText: 'OK (Cached)',
        headers: {
          'Content-Type': 'application/json',
          'X-Offline-Response': 'true'
        }
      });
    }
    
    // Return offline response
    return new Response(JSON.stringify({
      error: 'offline',
      message: 'This request is not available offline',
      _offline: true
    }), {
      status: 503,
      statusText: 'Service Unavailable',
      headers: {
        'Content-Type': 'application/json',
        'X-Offline-Response': 'true'
      }
    });
  }
}

// Handle mutation requests (POST, PUT, DELETE)
async function handleMutationRequest(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    return networkResponse;
  } catch (error) {
    console.log('Service Worker: Mutation request failed, queuing for later:', error);
    
    // Queue the request for when online
    const requestData = {
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: request.method !== 'GET' ? await request.text() : null,
      timestamp: Date.now(),
      id: generateRequestId()
    };
    
    // Store in IndexedDB for persistence
    await storeOfflineRequest(requestData);
    
    // Return queued response
    return new Response(JSON.stringify({
      success: false,
      queued: true,
      message: 'Request queued for when connection is restored',
      requestId: requestData.id,
      _offline: true
    }), {
      status: 202,
      statusText: 'Accepted (Queued)',
      headers: {
        'Content-Type': 'application/json',
        'X-Offline-Response': 'true'
      }
    });
  }
}

// Store offline request in IndexedDB
async function storeOfflineRequest(requestData) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('wan22-offline', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['requests'], 'readwrite');
      const store = transaction.objectStore('requests');
      
      store.add(requestData);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    };
    
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('requests')) {
        const store = db.createObjectStore('requests', { keyPath: 'id' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
  });
}

// Process queued requests when online
async function processOfflineQueue() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('wan22-offline', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = async () => {
      const db = request.result;
      const transaction = db.transaction(['requests'], 'readwrite');
      const store = transaction.objectStore('requests');
      
      const getAllRequest = store.getAll();
      getAllRequest.onsuccess = async () => {
        const requests = getAllRequest.result;
        
        for (const requestData of requests) {
          try {
            // Recreate the request
            const fetchRequest = new Request(requestData.url, {
              method: requestData.method,
              headers: requestData.headers,
              body: requestData.body
            });
            
            // Try to send the request
            const response = await fetch(fetchRequest);
            
            if (response.ok) {
              // Remove from queue if successful
              store.delete(requestData.id);
              
              // Notify the client
              self.clients.matchAll().then(clients => {
                clients.forEach(client => {
                  client.postMessage({
                    type: 'OFFLINE_REQUEST_PROCESSED',
                    requestId: requestData.id,
                    success: true,
                    response: response.status
                  });
                });
              });
            }
          } catch (error) {
            console.log('Service Worker: Failed to process queued request:', error);
            
            // Notify client of failure
            self.clients.matchAll().then(clients => {
              clients.forEach(client => {
                client.postMessage({
                  type: 'OFFLINE_REQUEST_FAILED',
                  requestId: requestData.id,
                  error: error.message
                });
              });
            });
          }
        }
        
        resolve();
      };
    };
  });
}

// Listen for online events
self.addEventListener('online', () => {
  console.log('Service Worker: Back online, processing queued requests');
  processOfflineQueue();
});

// Handle messages from the main thread
self.addEventListener('message', (event) => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_OFFLINE_QUEUE':
      getOfflineQueueCount().then(count => {
        event.ports[0].postMessage({ count });
      });
      break;
      
    case 'CLEAR_OFFLINE_QUEUE':
      clearOfflineQueue().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    case 'PROCESS_OFFLINE_QUEUE':
      processOfflineQueue().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
  }
});

// Get offline queue count
async function getOfflineQueueCount() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('wan22-offline', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['requests'], 'readonly');
      const store = transaction.objectStore('requests');
      
      const countRequest = store.count();
      countRequest.onsuccess = () => resolve(countRequest.result);
      countRequest.onerror = () => reject(countRequest.error);
    };
  });
}

// Clear offline queue
async function clearOfflineQueue() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('wan22-offline', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['requests'], 'readwrite');
      const store = transaction.objectStore('requests');
      
      const clearRequest = store.clear();
      clearRequest.onsuccess = () => resolve();
      clearRequest.onerror = () => reject(clearRequest.error);
    };
  });
}

// Generate unique request ID
function generateRequestId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Background sync for processing queued requests
self.addEventListener('sync', (event) => {
  if (event.tag === 'process-offline-queue') {
    event.waitUntil(processOfflineQueue());
  }
});

console.log('Service Worker: Loaded');