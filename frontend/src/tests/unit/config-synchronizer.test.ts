/**
 * Tests for Configuration Synchronizer
 * Verifies robust singleton implementation with typed state and listener support
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { 
  ConfigSynchronizer, 
  type CacheManager 
} from '@/lib/config-synchronizer';

// Mock cache manager implementation
const mockCacheManager: CacheManager = {
  get: vi.fn().mockResolvedValue(null),
  set: vi.fn().mockResolvedValue(undefined),
  delete: vi.fn().mockResolvedValue(undefined),
  clear: vi.fn().mockResolvedValue(undefined),
};

describe('ConfigSynchronizer', () => {
  let configSynchronizer: ConfigSynchronizer;
  let mockSetInterval: ReturnType<typeof vi.fn>;
  let mockClearInterval: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Reset singleton instance
    ConfigSynchronizer.resetInstance();
    
    // Mock timers
    mockSetInterval = vi.fn().mockReturnValue(123);
    mockClearInterval = vi.fn();
    
    vi.stubGlobal('setInterval', mockSetInterval);
    vi.stubGlobal('clearInterval', mockClearInterval);

    // Reset mocks
    vi.clearAllMocks();
    
    // Get fresh instance with test options
    configSynchronizer = ConfigSynchronizer.getInstance({
      autoSync: false, // Disable auto sync for tests
      cacheManager: mockCacheManager,
    });
  });

  afterEach(() => {
    configSynchronizer.destroy();
    ConfigSynchronizer.resetInstance();
    vi.restoreAllMocks();
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance', () => {
      const instance1 = ConfigSynchronizer.getInstance();
      const instance2 = ConfigSynchronizer.getInstance();
      
      expect(instance1).toBe(instance2);
    });

    it('should reset instance properly', () => {
      const instance1 = ConfigSynchronizer.getInstance();
      ConfigSynchronizer.resetInstance();
      const instance2 = ConfigSynchronizer.getInstance();
      
      expect(instance1).not.toBe(instance2);
    });
  });

  describe('Configuration Management', () => {
    it('should return empty config initially', () => {
      const config = configSynchronizer.getConfig();
      
      expect(config).toEqual({});
    });

    it('should update config correctly', () => {
      const updates = { apiUrl: 'http://localhost:8080', port: 8080 };
      
      configSynchronizer.updateConfig(updates);
      const config = configSynchronizer.getConfig();
      
      expect(config).toEqual(updates);
    });

    it('should merge config updates', () => {
      configSynchronizer.updateConfig({ apiUrl: 'http://localhost:8080' });
      configSynchronizer.updateConfig({ port: 8080 });
      
      const config = configSynchronizer.getConfig();
      
      expect(config).toEqual({
        apiUrl: 'http://localhost:8080',
        port: 8080,
      });
    });
  });

  describe('Change Detection', () => {
    it('should detect changes correctly', () => {
      configSynchronizer.updateConfig({ apiUrl: 'http://localhost:8080' });
      
      const newConfig = { apiUrl: 'http://localhost:9000', port: 9000 };
      const changes = configSynchronizer.detectChanges(newConfig);
      
      expect(changes).toHaveLength(2);
      expect(changes[0]).toMatchObject({
        key: 'apiUrl',
        oldValue: 'http://localhost:8080',
        newValue: 'http://localhost:9000',
      });
      expect(changes[1]).toMatchObject({
        key: 'port',
        oldValue: undefined,
        newValue: 9000,
      });
    });

    it('should return empty array when no changes', () => {
      const config = { apiUrl: 'http://localhost:8080' };
      configSynchronizer.updateConfig(config);
      
      const changes = configSynchronizer.detectChanges(config);
      
      expect(changes).toHaveLength(0);
    });
  });

  describe('Listener Management', () => {
    it('should add and notify listeners', () => {
      const listener = vi.fn();
      const unsubscribe = configSynchronizer.addListener(listener);
      
      configSynchronizer.updateConfig({ apiUrl: 'http://localhost:8080' });
      
      expect(listener).toHaveBeenCalledWith({
        key: 'apiUrl',
        oldValue: undefined,
        newValue: 'http://localhost:8080',
        timestamp: expect.any(Number),
      });
      
      expect(typeof unsubscribe).toBe('function');
    });

    it('should remove listeners correctly', () => {
      const listener = vi.fn();
      configSynchronizer.addListener(listener);
      configSynchronizer.removeListener(listener);
      
      configSynchronizer.updateConfig({ apiUrl: 'http://localhost:8080' });
      
      expect(listener).not.toHaveBeenCalled();
    });

    it('should unsubscribe using returned function', () => {
      const listener = vi.fn();
      const unsubscribe = configSynchronizer.addListener(listener);
      
      unsubscribe();
      configSynchronizer.updateConfig({ apiUrl: 'http://localhost:8080' });
      
      expect(listener).not.toHaveBeenCalled();
    });

    it('should handle listener errors gracefully', () => {
      const errorListener = vi.fn().mockImplementation(() => {
        throw new Error('Listener error');
      });
      const goodListener = vi.fn();
      
      configSynchronizer.addListener(errorListener);
      configSynchronizer.addListener(goodListener);
      
      // Should not throw
      expect(() => {
        configSynchronizer.updateConfig({ apiUrl: 'http://localhost:8080' });
      }).not.toThrow();
      
      expect(goodListener).toHaveBeenCalled();
    });
  });

  describe('Auto Sync', () => {
    it('should start auto sync when enabled', () => {
      ConfigSynchronizer.resetInstance();
      vi.clearAllMocks();
      
      const instance = ConfigSynchronizer.getInstance({ autoSync: true });
      
      expect(mockSetInterval).toHaveBeenCalledWith(
        expect.any(Function),
        30000
      );
      
      instance.destroy();
    });

    it('should not start auto sync when disabled', () => {
      ConfigSynchronizer.resetInstance();
      vi.clearAllMocks();
      
      ConfigSynchronizer.getInstance({ autoSync: false });
      
      expect(mockSetInterval).not.toHaveBeenCalled();
    });

    it('should stop auto sync', () => {
      // First start auto sync to have something to stop
      ConfigSynchronizer.resetInstance();
      vi.clearAllMocks();
      
      const instance = ConfigSynchronizer.getInstance({ autoSync: true });
      vi.clearAllMocks(); // Clear the initial setInterval call
      
      instance.stopAutoSync();
      
      expect(mockClearInterval).toHaveBeenCalledWith(123);
      instance.destroy();
    });

    it('should resume auto sync', () => {
      // Create instance with autoSync: true to test resumeAutoSync
      ConfigSynchronizer.resetInstance();
      vi.clearAllMocks();
      
      const instance = ConfigSynchronizer.getInstance({ autoSync: true });
      instance.stopAutoSync(); // Stop it first
      vi.clearAllMocks(); // Clear the stop call
      
      instance.resumeAutoSync();
      
      expect(mockSetInterval).toHaveBeenCalledWith(
        expect.any(Function),
        30000
      );
      
      instance.destroy();
    });
  });

  describe('Sync Operations', () => {
    it('should sync config successfully', async () => {
      await expect(configSynchronizer.syncConfig()).resolves.not.toThrow();
      
      const metrics = configSynchronizer.getMetrics();
      expect(metrics.totalSyncs).toBe(1);
      expect(metrics.successfulSyncs).toBe(1);
      expect(metrics.failedSyncs).toBe(0);
    });

    it('should handle sync failures with retry', async () => {
      // Mock performSync to fail twice then succeed
      const originalPerformSync = (configSynchronizer as any).performSync;
      let callCount = 0;
      
      (configSynchronizer as any).performSync = vi.fn().mockImplementation(async () => {
        callCount++;
        if (callCount <= 2) {
          throw new Error('Sync failed');
        }
        return originalPerformSync.call(configSynchronizer);
      });

      await expect(configSynchronizer.syncConfig()).resolves.not.toThrow();
      
      const metrics = configSynchronizer.getMetrics();
      expect(metrics.totalSyncs).toBe(3); // Initial + 2 retries
      expect(metrics.successfulSyncs).toBe(1);
      expect(metrics.failedSyncs).toBe(2);
      expect(metrics.retryCount).toBe(2);
    });

    it('should prevent concurrent syncs', async () => {
      const syncPromise1 = configSynchronizer.syncConfig();
      const syncPromise2 = configSynchronizer.syncConfig();
      
      await Promise.all([syncPromise1, syncPromise2]);
      
      const metrics = configSynchronizer.getMetrics();
      expect(metrics.totalSyncs).toBe(1); // Only one sync should have run
    });
  });

  describe('Cache Integration', () => {
    it('should use cache manager when available', async () => {
      const cachedConfig = { apiUrl: 'http://cached:8080' };
      (mockCacheManager.get as any).mockResolvedValueOnce(cachedConfig);
      
      await configSynchronizer.syncConfig();
      
      expect(mockCacheManager.get).toHaveBeenCalledWith('config');
      expect(configSynchronizer.getConfig()).toEqual(cachedConfig);
    });

    it('should update cache on config changes', () => {
      const updates = { apiUrl: 'http://localhost:8080' };
      
      configSynchronizer.updateConfig(updates);
      
      expect(mockCacheManager.set).toHaveBeenCalledWith('config', updates);
    });
  });

  describe('Health Monitoring', () => {
    it('should report healthy when syncing successfully', async () => {
      await configSynchronizer.syncConfig();
      
      expect(configSynchronizer.isHealthy()).toBe(true);
    });

    it('should report unhealthy when destroyed', () => {
      configSynchronizer.destroy();
      
      expect(configSynchronizer.isHealthy()).toBe(false);
    });

    it('should provide sync metrics', async () => {
      await configSynchronizer.syncConfig();
      
      const metrics = configSynchronizer.getMetrics();
      
      expect(metrics).toMatchObject({
        totalSyncs: 1,
        successfulSyncs: 1,
        failedSyncs: 0,
        retryCount: 0,
        lastSyncTime: expect.any(Number),
        lastSyncDuration: expect.any(Number),
        averageSyncDuration: expect.any(Number),
      });
    });
  });

  describe('Cleanup', () => {
    it('should cleanup resources on destroy', () => {
      const listener = vi.fn();
      
      // Create instance with auto sync to have timer to clear
      ConfigSynchronizer.resetInstance();
      vi.clearAllMocks();
      
      const instance = ConfigSynchronizer.getInstance({ 
        autoSync: true,
        cacheManager: mockCacheManager 
      });
      instance.addListener(listener);
      
      vi.clearAllMocks(); // Clear the initial setInterval call
      
      instance.destroy();
      
      expect(mockClearInterval).toHaveBeenCalled();
      expect(mockCacheManager.clear).toHaveBeenCalled();
      
      // Should not notify listeners after destroy
      instance.updateConfig({ test: 'value' });
      expect(listener).not.toHaveBeenCalled();
    });

    it('should throw error when using destroyed instance', async () => {
      configSynchronizer.destroy();
      
      await expect(configSynchronizer.syncConfig()).rejects.toThrow(
        'ConfigSynchronizer has been destroyed'
      );
    });
  });
});