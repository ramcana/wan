// Cache Manager for handling service worker caches
class CacheManager {
  private cacheName: string;

  constructor(cacheName: string = 'wan22-cache') {
    this.cacheName = cacheName;
  }

  async initialize(): Promise<void> {
    try {
      // Nothing to do here for now
      console.log('Cache manager initialized');
    } catch (error) {
      console.error('Failed to initialize cache manager:', error);
    }
  }

  async clearAllCaches(): Promise<void> {
    try {
      const cacheNames = await caches.keys();
      await Promise.all(
        cacheNames.map((cacheName) => caches.delete(cacheName))
      );
      console.log('All caches cleared');
    } catch (error) {
      console.error('Failed to clear caches:', error);
    }
  }

  async clearCache(cacheName: string): Promise<void> {
    try {
      await caches.delete(cacheName);
      console.log(`Cache ${cacheName} cleared`);
    } catch (error) {
      console.error(`Failed to clear cache ${cacheName}:`, error);
    }
  }
}

// Export singleton instance
export const cacheManager = new CacheManager();