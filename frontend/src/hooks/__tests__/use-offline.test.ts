import { renderHook, act, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { useOffline, useOfflineApi } from '../use-offline';

// Mock service worker
const mockServiceWorker = {
  register: vi.fn(),
  controller: {
    postMessage: vi.fn(),
  },
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
};

// Mock navigator
Object.defineProperty(global, 'navigator', {
  value: {
    onLine: true,
    serviceWorker: mockServiceWorker,
  },
  writable: true,
});

// Mock Notification API
Object.defineProperty(global, 'Notification', {
  value: {
    permission: 'default',
    requestPermission: vi.fn().mockResolvedValue('granted'),
  },
  writable: true,
});

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
};
Object.defineProperty(global, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

// Mock fetch
global.fetch = vi.fn();

// Mock MessageChannel
global.MessageChannel = vi.fn().mockImplementation(() => ({
  port1: { onmessage: null },
  port2: {},
}));

beforeEach(() => {
  vi.clearAllMocks();
  mockLocalStorage.getItem.mockReturnValue('[]');
  (global.fetch as any).mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({ data: 'test' }),
    headers: new Map(),
  });
});

describe('useOffline', () => {
  it('should initialize with correct default state', () => {
    const { result } = renderHook(() => useOffline());
    
    expect(result.current.isOnline).toBe(true);
    expect(result.current.isServiceWorkerReady).toBe(false);
    expect(result.current.queuedRequestsCount).toBe(0);
    expect(result.current.syncInProgress).toBe(false);
  });

  it('should register service worker', async () => {
    mockServiceWorker.register.mockResolvedValue({
      addEventListener: vi.fn(),
    });

    const { result } = renderHook(() => useOffline());
    
    await waitFor(() => {
      expect(result.current.isServiceWorkerReady).toBe(true);
    });

    expect(mockServiceWorker.register).toHaveBeenCalledWith('/sw.js');
  });

  it('should handle online/offline events', () => {
    const { result } = renderHook(() => useOffline());
    
    // Simulate going offline
    Object.defineProperty(navigator, 'onLine', {
      value: false,
      writable: true,
    });
    
    act(() => {
      window.dispatchEvent(new Event('offline'));
    });

    expect(result.current.isOnline).toBe(false);

    // Simulate going online
    Object.defineProperty(navigator, 'onLine', {
      value: true,
      writable: true,
    });
    
    act(() => {
      window.dispatchEvent(new Event('online'));
    });

    expect(result.current.isOnline).toBe(true);
  });

  it('should queue requests when offline', async () => {
    const { result } = renderHook(() => useOffline());
    
    // Set offline
    Object.defineProperty(navigator, 'onLine', {
      value: false,
      writable: true,
    });

    let queueResult;
    await act(async () => {
      queueResult = await result.current.queueRequest('/api/test', {
        method: 'POST',
        body: JSON.stringify({ test: 'data' }),
      });
    });

    expect(queueResult.queued).toBe(true);
    expect(queueResult.requestId).toBeDefined();
    expect(mockLocalStorage.setItem).toHaveBeenCalled();
  });

  it('should get offline data with fallback', async () => {
    const { result } = renderHook(() => useOffline());
    
    // Mock fetch failure
    (global.fetch as any).mockRejectedValue(new Error('Network error'));
    
    const fallbackData = { message: 'fallback' };
    
    let offlineData;
    await act(async () => {
      offlineData = await result.current.getOfflineData('/api/test', fallbackData);
    });

    expect(offlineData.data).toEqual(fallbackData);
    expect(offlineData.isOffline).toBe(true);
  });

  it('should cache data for offline use', () => {
    const { result } = renderHook(() => useOffline());
    
    const testData = { id: 1, name: 'test' };
    
    act(() => {
      result.current.cacheData('/api/test', testData);
    });

    expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
      expect.stringContaining('offline-cache-'),
      JSON.stringify(testData)
    );
  });

  it('should request notification permission', async () => {
    const { result } = renderHook(() => useOffline());
    
    let permissionGranted;
    await act(async () => {
      permissionGranted = await result.current.requestNotificationPermission();
    });

    expect(Notification.requestPermission).toHaveBeenCalled();
    expect(permissionGranted).toBe(true);
  });
});

describe('useOfflineApi', () => {
  it('should provide API methods', () => {
    const { result } = renderHook(() => useOfflineApi());
    
    expect(result.current.get).toBeDefined();
    expect(result.current.post).toBeDefined();
    expect(result.current.put).toBeDefined();
    expect(result.current.delete).toBeDefined();
  });

  it('should make GET requests and cache responses', async () => {
    const { result } = renderHook(() => useOfflineApi());
    
    const mockResponse = { data: 'test' };
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
      headers: new Map(),
    });

    let response;
    await act(async () => {
      response = await result.current.get('/api/test');
    });

    expect(response.data).toEqual(mockResponse);
    expect(response.isOffline).toBe(false);
    expect(mockLocalStorage.setItem).toHaveBeenCalled();
  });

  it('should queue POST requests when offline', async () => {
    const { result } = renderHook(() => useOfflineApi());
    
    // Set offline
    Object.defineProperty(navigator, 'onLine', {
      value: false,
      writable: true,
    });

    let response;
    await act(async () => {
      response = await result.current.post('/api/test', { data: 'test' });
    });

    expect(response.queued).toBe(true);
    expect(response.requestId).toBeDefined();
  });
});