/**
 * Tests for Cache Manager
 * Verifies Task 3 implementation for port configuration and cache issues
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cacheManager } from '@/lib/cache-manager';

// Mock navigator.serviceWorker
const mockServiceWorker = {
  getRegistrations: vi.fn(),
  register: vi.fn(),
  controller: null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
};

const mockRegistration = {
  waiting: {
    postMessage: vi.fn(),
  },
  active: {
    postMessage: vi.fn(),
  },
  update: vi.fn(),
  unregister: vi.fn(),
  scope: 'http://localhost:3000/',
  addEventListener: vi.fn(),
};

// Mock caches API
const mockCaches = {
  keys: vi.fn(),
  delete: vi.fn(),
  open: vi.fn(),
};

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};

// Mock sessionStorage
const mockSessionStorage = {
  clear: vi.fn(),
};

// Mock window.location
const mockLocation = {
  reload: vi.fn(),
};

Object.defineProperty(global, 'navigator', {
  value: {
    serviceWorker: mockServiceWorker,
  },
  writable: true,
});

Object.defineProperty(global, 'caches', {
  value: mockCaches,
  writable: true,
});

Object.defineProperty(global, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

Object.defineProperty(global, 'sessionStorage', {
  value: mockSessionStorage,
  writable: true,
});

Object.defineProperty(global, 'window', {
  value: {
    location: mockLocation,
  },
  writable: true,
});

// Mock import.meta.env
vi.mock('import.meta', () => ({
  env: {
    VITE_API_URL: 'http://localhost:8000',
  },
}));

describe('CacheManager', () => {
  let cacheManagerInstance: any;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    
    // Reset localStorage mock
    mockLocalStorage.getItem.mockReturnValue(null);
    mockLocalStorage.setItem.mockImplementation(() => {});
    mockLocalStorage.removeItem.mockImplementation(() => {});
    
    // Reset service worker mocks
    mockServiceWorker.getRegistrations.mockResolvedValue([mockRegistration]);
    mockRegistration.update.mockResolvedValue(undefined);
    mockRegistration.unregister.mockResolvedValue(true);
    
    // Reset caches mock
    mockCaches.keys.mockResolvedValue(['cache1', 'cache2']);
    mockCaches.delete.mockResolvedValue(true);
    
    // Get fresh instance
    cacheManagerInstance = cacheManager;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('checkAndClearCacheOnUrlChange', () => {
    it('should clear caches when API URL changes', async () => {
      // Setup: stored URL is different from current
      mockLocalStorage.getItem.mockReturnValue('http://localhost:8000');
      
      const result = await cacheManagerInstance.checkAndClearCacheOnUrlChange();
      
      expect(result).toBe(true);
      expect(mockCaches.keys).toHaveBeenCalled();
      expect(mockCaches.delete).toHaveBeenCalledWith('cache1');
      expect(mockCaches.delete).toHaveBeenCalledWith('cache2');
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('wan22_api_url', 'http://localhost:8000');
    });

    it('should not clear caches when API URL is unchanged', async () => {
      // Setup: stored URL is same as current
      mockLocalStorage.getItem.mockReturnValue('http://localhost:8000');
      
      const result = await cacheManagerInstance.checkAndClearCacheOnUrlChange();
      
      expect(result).toBe(false);
      expect(mockCaches.keys).not.toHaveBeenCalled();
    });

    it('should store URL when not previously stored', async () => {
      // Setup: no stored URL
      mockLocalStorage.getItem.mockReturnValue(null);
      
      const result = await cacheManagerInstance.checkAndClearCacheOnUrlChange();
      
      expect(result).toBe(false);
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('wan22_api_url', 'http://localhost:8000');
    });
  });

  describe('clearServiceWorkerCache', () => {
    it('should clear service worker cache successfully', async () => {
      const result = await cacheManagerInstance.clearServiceWorkerCache();
      
      expect(result).toBe(true);
      expect(mockServiceWorker.getRegistrations).toHaveBeenCalled();
      expect(mockRegistration.waiting.postMessage).toHaveBeenCalledWith({ type: 'SKIP_WAITING' });
      expect(mockRegistration.active.postMessage).toHaveBeenCalledWith({ type: 'SKIP_WAITING' });
      expect(mockRegistration.update).toHaveBeenCalled();
    });

    it('should handle service worker not supported', async () => {
      // Mock service worker not supported
      Object.defineProperty(global, 'navigator', {
        value: {},
        writable: true,
      });
      
      const result = await cacheManagerInstance.clearServiceWorkerCache();
      
      expect(result).toBe(false);
    });

    it('should handle service worker errors', async () => {
      mockServiceWorker.getRegistrations.mockRejectedValue(new Error('Service worker error'));
      
      const result = await cacheManagerInstance.clearServiceWorkerCache();
      
      expect(result).toBe(false);
    });
  });

  describe('clearBrowserCache', () => {
    it('should clear browser cache successfully', async () => {
      const result = await cacheManagerInstance.clearBrowserCache();
      
      expect(result).toBe(true);
      expect(mockCaches.keys).toHaveBeenCalled();
      expect(mockCaches.delete).toHaveBeenCalledWith('cache1');
      expect(mockCaches.delete).toHaveBeenCalledWith('cache2');
      expect(mockSessionStorage.clear).toHaveBeenCalled();
    });

    it('should handle cache API errors', async () => {
      mockCaches.keys.mockRejectedValue(new Error('Cache error'));
      
      const result = await cacheManagerInstance.clearBrowserCache();
      
      expect(result).toBe(false);
    });
  });

  describe('unregisterServiceWorkers', () => {
    it('should unregister all service workers', async () => {
      const result = await cacheManagerInstance.unregisterServiceWorkers();
      
      expect(result).toBe(true);
      expect(mockServiceWorker.getRegistrations).toHaveBeenCalled();
      expect(mockRegistration.unregister).toHaveBeenCalled();
    });

    it('should handle unregistration errors', async () => {
      mockRegistration.unregister.mockRejectedValue(new Error('Unregister error'));
      
      const result = await cacheManagerInstance.unregisterServiceWorkers();
      
      expect(result).toBe(false);
    });
  });

  describe('clearAllCaches', () => {
    it('should clear all caches successfully', async () => {
      const result = await cacheManagerInstance.clearAllCaches();
      
      expect(result).toBe(true);
      expect(mockServiceWorker.getRegistrations).toHaveBeenCalled();
      expect(mockCaches.keys).toHaveBeenCalled();
    });

    it('should handle partial failures', async () => {
      // Make service worker clearing fail
      mockServiceWorker.getRegistrations.mockRejectedValue(new Error('SW error'));
      
      const result = await cacheManagerInstance.clearAllCaches();
      
      expect(result).toBe(false);
    });
  });

  describe('updateApiUrl', () => {
    it('should update API URL and clear caches', async () => {
      const newUrl = 'http://localhost:8080';
      
      const result = await cacheManagerInstance.updateApiUrl(newUrl);
      
      expect(result).toBe(true);
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('wan22_api_url', newUrl);
      expect(mockCaches.keys).toHaveBeenCalled();
    });

    it('should not clear caches if URL is unchanged', async () => {
      const currentUrl = cacheManagerInstance.getCurrentApiUrl();
      
      const result = await cacheManagerInstance.updateApiUrl(currentUrl);
      
      expect(result).toBe(false);
      expect(mockCaches.keys).not.toHaveBeenCalled();
    });
  });

  describe('forceReload', () => {
    it('should force reload the application', () => {
      cacheManagerInstance.forceReload();
      
      expect(mockLocation.reload).toHaveBeenCalled();
    });
  });

  describe('initialize', () => {
    it('should initialize and check for URL changes', async () => {
      mockLocalStorage.getItem.mockReturnValue('http://localhost:8000');
      
      await cacheManagerInstance.initialize();
      
      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('wan22_api_url');
      expect(mockCaches.keys).toHaveBeenCalled();
    });

    it('should handle initialization errors', async () => {
      mockLocalStorage.getItem.mockImplementation(() => {
        throw new Error('Storage error');
      });
      
      // Should not throw
      await expect(cacheManagerInstance.initialize()).resolves.toBeUndefined();
    });
  });
});