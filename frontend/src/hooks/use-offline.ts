import { useState, useEffect, useCallback } from 'react';

interface OfflineRequest {
  id: string;
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: string;
  timestamp: number;
  retryCount: number;
}

interface OfflineState {
  isOnline: boolean;
  isServiceWorkerReady: boolean;
  queuedRequestsCount: number;
  lastSyncAttempt?: Date;
  syncInProgress: boolean;
}

export const useOffline = () => {
  const [state, setState] = useState<OfflineState>({
    isOnline: navigator.onLine,
    isServiceWorkerReady: false,
    queuedRequestsCount: 0,
    syncInProgress: false,
  });

  // Register service worker
  const registerServiceWorker = useCallback(async () => {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js');
        
        // Handle service worker updates
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                // New service worker is available
                showUpdateAvailableNotification();
              }
            });
          }
        });

        setState(prev => ({ ...prev, isServiceWorkerReady: true }));
        console.log('Service Worker registered successfully');
      } catch (error) {
        console.error('Service Worker registration failed:', error);
      }
    }
  }, []);

  // Show update notification
  const showUpdateAvailableNotification = useCallback(() => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('App Update Available', {
        body: 'A new version of the app is available. Refresh to update.',
        icon: '/favicon.ico',
        tag: 'app-update'
      });
    }
  }, []);

  // Update service worker
  const updateServiceWorker = useCallback(async () => {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({ type: 'SKIP_WAITING' });
      window.location.reload();
    }
  }, []);

  // Get queued requests count
  const getQueuedRequestsCount = useCallback(async (): Promise<number> => {
    if (!state.isServiceWorkerReady || !navigator.serviceWorker.controller) {
      return 0;
    }

    return new Promise((resolve) => {
      const messageChannel = new MessageChannel();
      messageChannel.port1.onmessage = (event) => {
        resolve(event.data.count || 0);
      };

      navigator.serviceWorker.controller!.postMessage(
        { type: 'GET_OFFLINE_QUEUE' },
        [messageChannel.port2]
      );
    });
  }, [state.isServiceWorkerReady]);

  // Clear offline queue
  const clearOfflineQueue = useCallback(async (): Promise<boolean> => {
    if (!state.isServiceWorkerReady || !navigator.serviceWorker.controller) {
      return false;
    }

    return new Promise((resolve) => {
      const messageChannel = new MessageChannel();
      messageChannel.port1.onmessage = (event) => {
        resolve(event.data.success || false);
      };

      navigator.serviceWorker.controller!.postMessage(
        { type: 'CLEAR_OFFLINE_QUEUE' },
        [messageChannel.port2]
      );
    });
  }, [state.isServiceWorkerReady]);

  // Process offline queue manually
  const processOfflineQueue = useCallback(async (): Promise<boolean> => {
    if (!state.isServiceWorkerReady || !navigator.serviceWorker.controller) {
      return false;
    }

    setState(prev => ({ ...prev, syncInProgress: true }));

    return new Promise((resolve) => {
      const messageChannel = new MessageChannel();
      messageChannel.port1.onmessage = (event) => {
        setState(prev => ({ 
          ...prev, 
          syncInProgress: false,
          lastSyncAttempt: new Date()
        }));
        resolve(event.data.success || false);
      };

      navigator.serviceWorker.controller!.postMessage(
        { type: 'PROCESS_OFFLINE_QUEUE' },
        [messageChannel.port2]
      );
    });
  }, [state.isServiceWorkerReady]);

  // Queue request for offline processing
  const queueRequest = useCallback(async (
    url: string,
    options: RequestInit = {}
  ): Promise<{ queued: boolean; requestId?: string }> => {
    if (state.isOnline) {
      // If online, make the request normally
      try {
        const response = await fetch(url, options);
        return { queued: false };
      } catch (error) {
        // Network error, queue the request
      }
    }

    // Queue the request
    const requestData: OfflineRequest = {
      id: generateRequestId(),
      url,
      method: options.method || 'GET',
      headers: options.headers as Record<string, string> || {},
      body: options.body as string,
      timestamp: Date.now(),
      retryCount: 0,
    };

    // Store in localStorage as backup
    const existingQueue = JSON.parse(localStorage.getItem('offline-queue') || '[]');
    existingQueue.push(requestData);
    localStorage.setItem('offline-queue', JSON.stringify(existingQueue));

    setState(prev => ({ 
      ...prev, 
      queuedRequestsCount: prev.queuedRequestsCount + 1 
    }));

    return { queued: true, requestId: requestData.id };
  }, [state.isOnline]);

  // Check if response is from cache/offline
  const isOfflineResponse = useCallback((response: Response): boolean => {
    return response.headers.get('X-Offline-Response') === 'true';
  }, []);

  // Get offline-capable data with fallback
  const getOfflineData = useCallback(async <T>(
    url: string,
    fallbackData?: T
  ): Promise<{ data: T | null; isOffline: boolean }> => {
    try {
      const response = await fetch(url);
      const data = await response.json();
      
      return {
        data,
        isOffline: isOfflineResponse(response)
      };
    } catch (error) {
      // Return fallback data if available
      if (fallbackData !== undefined) {
        return {
          data: fallbackData,
          isOffline: true
        };
      }

      // Try to get from localStorage cache
      const cacheKey = `offline-cache-${btoa(url)}`;
      const cachedData = localStorage.getItem(cacheKey);
      
      if (cachedData) {
        return {
          data: JSON.parse(cachedData),
          isOffline: true
        };
      }

      return {
        data: null,
        isOffline: true
      };
    }
  }, [isOfflineResponse]);

  // Cache data for offline use
  const cacheData = useCallback(<T>(url: string, data: T): void => {
    const cacheKey = `offline-cache-${btoa(url)}`;
    localStorage.setItem(cacheKey, JSON.stringify(data));
  }, []);

  // Request notification permission
  const requestNotificationPermission = useCallback(async (): Promise<boolean> => {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return false;
  }, []);

  // Show offline notification
  const showOfflineNotification = useCallback((message: string) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Offline Mode', {
        body: message,
        icon: '/favicon.ico',
        tag: 'offline-status'
      });
    }
  }, []);

  // Update queued requests count
  const updateQueuedRequestsCount = useCallback(async () => {
    const count = await getQueuedRequestsCount();
    setState(prev => ({ ...prev, queuedRequestsCount: count }));
  }, [getQueuedRequestsCount]);

  // Handle online/offline events
  useEffect(() => {
    const handleOnline = () => {
      setState(prev => ({ ...prev, isOnline: true }));
      showOfflineNotification('Connection restored! Syncing queued requests...');
      
      // Process queued requests
      processOfflineQueue();
    };

    const handleOffline = () => {
      setState(prev => ({ ...prev, isOnline: false }));
      showOfflineNotification('You are now offline. Some features may be limited.');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [processOfflineQueue, showOfflineNotification]);

  // Listen for service worker messages
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      const handleMessage = (event: MessageEvent) => {
        const { type, requestId, success } = event.data;

        switch (type) {
          case 'OFFLINE_REQUEST_PROCESSED':
            if (success) {
              setState(prev => ({ 
                ...prev, 
                queuedRequestsCount: Math.max(0, prev.queuedRequestsCount - 1)
              }));
            }
            break;

          case 'OFFLINE_REQUEST_FAILED':
            console.warn('Offline request failed:', requestId);
            break;
        }
      };

      navigator.serviceWorker.addEventListener('message', handleMessage);

      return () => {
        navigator.serviceWorker.removeEventListener('message', handleMessage);
      };
    }
  }, []);

  // Initialize service worker and update queue count
  useEffect(() => {
    registerServiceWorker();
    updateQueuedRequestsCount();

    // Update queue count periodically
    const interval = setInterval(updateQueuedRequestsCount, 30000);
    return () => clearInterval(interval);
  }, [registerServiceWorker, updateQueuedRequestsCount]);

  return {
    ...state,
    queueRequest,
    getOfflineData,
    cacheData,
    isOfflineResponse,
    clearOfflineQueue,
    processOfflineQueue,
    updateServiceWorker,
    requestNotificationPermission,
    showOfflineNotification,
    updateQueuedRequestsCount,
  };
};

// Utility function to generate request IDs
function generateRequestId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Hook for offline-aware API calls
export const useOfflineApi = () => {
  const { queueRequest, getOfflineData, cacheData, isOnline } = useOffline();

  const get = useCallback(async <T>(url: string, fallback?: T) => {
    const result = await getOfflineData<T>(url, fallback);
    
    // Cache successful responses
    if (result.data && !result.isOffline) {
      cacheData(url, result.data);
    }
    
    return result;
  }, [getOfflineData, cacheData]);

  const post = useCallback(async (url: string, data: any) => {
    const options: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    };

    if (isOnline) {
      try {
        const response = await fetch(url, options);
        return await response.json();
      } catch (error) {
        // Queue if network error
        const result = await queueRequest(url, options);
        return { queued: result.queued, requestId: result.requestId };
      }
    } else {
      // Queue immediately if offline
      const result = await queueRequest(url, options);
      return { queued: result.queued, requestId: result.requestId };
    }
  }, [isOnline, queueRequest]);

  const put = useCallback(async (url: string, data: any) => {
    const options: RequestInit = {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    };

    if (isOnline) {
      try {
        const response = await fetch(url, options);
        return await response.json();
      } catch (error) {
        const result = await queueRequest(url, options);
        return { queued: result.queued, requestId: result.requestId };
      }
    } else {
      const result = await queueRequest(url, options);
      return { queued: result.queued, requestId: result.requestId };
    }
  }, [isOnline, queueRequest]);

  const del = useCallback(async (url: string) => {
    const options: RequestInit = {
      method: 'DELETE',
    };

    if (isOnline) {
      try {
        const response = await fetch(url, options);
        return await response.json();
      } catch (error) {
        const result = await queueRequest(url, options);
        return { queued: result.queued, requestId: result.requestId };
      }
    } else {
      const result = await queueRequest(url, options);
      return { queued: result.queued, requestId: result.requestId };
    }
  }, [isOnline, queueRequest]);

  return {
    get,
    post,
    put,
    delete: del,
  };
};