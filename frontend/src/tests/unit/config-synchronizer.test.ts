/**
 * Tests for Configuration Synchronizer
 * Verifies Task 3 implementation for environment variable synchronization
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ConfigSynchronizer } from '../../lib/config-synchronizer';

// Mock import.meta.env
const mockEnv = {
  VITE_API_URL: 'http://localhost:9000',
  VITE_DEV_MODE: 'true',
  DEV: true,
};

vi.mock('import.meta', () => ({
  env: mockEnv,
}));

// Mock cache manager
vi.mock('../../lib/cache-manager', () => ({
  cacheManager: {
    clearAllCaches: vi.fn().mockResolvedValue(true),
    updateApiUrl: vi.fn().mockResolvedValue(true),
    forceReload: vi.fn(),
  },
}));

describe('ConfigSynchronizer', () => {
  let configSynchronizer: ConfigSynchronizer;
  let mockSetInterval: ReturnType<typeof vi.fn>;
  let mockClearInterval: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Mock timers
    mockSetInterval = vi.fn();
    mockClearInterval = vi.fn();
    
    Object.defineProperty(global, 'setInterval', {
      value: mockSetInterval,
      writable: true,
    });
    
    Object.defineProperty(global, 'clearInterval', {
      value: mockClearInterval,
      writable: true,
    });

    // Reset mocks
    vi.clearAllMocks();
    
    // Get fresh instance
    configSynchronizer = ConfigSynchronizer.getInstance();
  });

  afterEach(() => {
    configSynchronizer.cleanup();
    vi.restoreAllMocks();
  });

  describe('getCurrentConfig', () => {
    it('should return current configuration', () => {
      const config = configSynchronizer.getCurrentConfig();
      
      expect(config).toEqual({
        apiUrl: 'http://localhost:9000',
        devMode: true,
        backendPort: 9000,
        frontendPort: 3000,
        lastUpdated: expect.any(Date),
      });
    });
  });

  describe('checkForChanges', () => {
    it('should detect API URL changes', async () => {
      // Change the mock environment
      mockEnv.VITE_API_URL = 'http://localhost:8080';
      
      const changes = await configSynchronizer.checkForChanges();
      
      expect(changes).toHaveLength(2); // API URL and port change
      expect(changes[0]).toEqual({
        type: 'api_url_changed',
        oldValue: 'http://localhost:9000',
        newValue: 'http://localhost:8080',
        timestamp: expect.any(Date),
      });
      expect(changes[1]).toEqual({
        type: 'port_changed',
        oldValue: 9000,
        newValue: 8080,
        timestamp: expect.any(Date),
      });
    });

    it('should detect dev mode changes', async () => {
      // Change the mock environment
      mockEnv.VITE_DEV_MODE = 'false';
      mockEnv.DEV = false;
      
      const changes = await configSynchronizer.checkForChanges();
      
      expect(changes).toHaveLength(1);
      expect(changes[0]).toEqual({
        type: 'dev_mode_changed',
        oldValue: true,
        newValue: false,
        timestamp: expect.any(Date),
      });
    });

    it('should return empty array when no changes', async () => {
      const changes = await configSynchronizer.checkForChanges();
      
      expect(changes).toHaveLength(0);
    });
  });

  describe('startMonitoring', () => {
    it('should start monitoring with default interval', () => {
      configSynchronizer.startMonitoring();
      
      expect(mockSetInterval).toHaveBeenCalledWith(
        expect.any(Function),
        5000
      );
    });

    it('should start monitoring with custom interval', () => {
      configSynchronizer.startMonitoring(3000);
      
      expect(mockSetInterval).toHaveBeenCalledWith(
        expect.any(Function),
        3000
      );
    });

    it('should stop existing monitoring before starting new', () => {
      const intervalId = 123;
      mockSetInterval.mockReturnValue(intervalId);
      
      configSynchronizer.startMonitoring();
      configSynchronizer.startMonitoring();
      
      expect(mockClearInterval).toHaveBeenCalledWith(intervalId);
      expect(mockSetInterval).toHaveBeenCalledTimes(2);
    });
  });

  describe('stopMonitoring', () => {
    it('should stop monitoring', () => {
      const intervalId = 123;
      mockSetInterval.mockReturnValue(intervalId);
      
      configSynchronizer.startMonitoring();
      configSynchronizer.stopMonitoring();
      
      expect(mockClearInterval).toHaveBeenCalledWith(intervalId);
    });

    it('should handle stopping when not monitoring', () => {
      configSynchronizer.stopMonitoring();
      
      expect(mockClearInterval).not.toHaveBeenCalled();
    });
  });

  describe('addChangeListener', () => {
    it('should add change listener', async () => {
      const listener = vi.fn();
      configSynchronizer.addChangeListener(listener);
      
      // Trigger a change
      mockEnv.VITE_API_URL = 'http://localhost:8080';
      await configSynchronizer.checkForChanges();
      
      expect(listener).toHaveBeenCalledWith({
        type: 'api_url_changed',
        oldValue: 'http://localhost:9000',
        newValue: 'http://localhost:8080',
        timestamp: expect.any(Date),
      });
    });
  });

  describe('removeChangeListener', () => {
    it('should remove change listener', async () => {
      const listener = vi.fn();
      configSynchronizer.addChangeListener(listener);
      configSynchronizer.removeChangeListener(listener);
      
      // Trigger a change
      mockEnv.VITE_API_URL = 'http://localhost:8080';
      await configSynchronizer.checkForChanges();
      
      expect(listener).not.toHaveBeenCalled();
    });
  });

  describe('forceSync', () => {
    it('should force configuration sync', async () => {
      mockEnv.VITE_API_URL = 'http://localhost:8080';
      
      const changes = await configSynchronizer.forceSync();
      
      expect(changes).toHaveLength(2); // API URL and port change
    });
  });

  describe('initialize', () => {
    it('should initialize and start monitoring in dev mode', async () => {
      await configSynchronizer.initialize();
      
      expect(mockSetInterval).toHaveBeenCalledWith(
        expect.any(Function),
        3000
      );
    });

    it('should not start monitoring in production mode', async () => {
      mockEnv.VITE_DEV_MODE = 'false';
      mockEnv.DEV = false;
      
      await configSynchronizer.initialize();
      
      expect(mockSetInterval).not.toHaveBeenCalled();
    });
  });

  describe('cleanup', () => {
    it('should cleanup resources', () => {
      const listener = vi.fn();
      configSynchronizer.addChangeListener(listener);
      configSynchronizer.startMonitoring();
      
      configSynchronizer.cleanup();
      
      expect(mockClearInterval).toHaveBeenCalled();
      
      // Verify listeners are cleared
      mockEnv.VITE_API_URL = 'http://localhost:8080';
      configSynchronizer.checkForChanges();
      expect(listener).not.toHaveBeenCalled();
    });
  });
});