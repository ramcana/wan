export interface ConfigSyncOptions {
  autoSync?: boolean;
  syncInterval?: number;
  retryAttempts?: number;
  retryDelay?: number;
  enableChangeDetection?: boolean;
  cacheManager?: CacheManager;
}

export interface ConfigState {
  [key: string]: any;
}

export interface ConfigChangeEvent {
  key: string;
  oldValue: any;
  newValue: any;
  timestamp: number;
}

export interface SyncMetrics {
  totalSyncs: number;
  successfulSyncs: number;
  failedSyncs: number;
  lastSyncTime?: number;
  lastSyncDuration?: number;
  averageSyncDuration: number;
  retryCount: number;
}

export interface CacheManager {
  get(key: string): Promise<any>;
  set(key: string, value: any, ttl?: number): Promise<void>;
  delete(key: string): Promise<void>;
  clear(): Promise<void>;
}

export type ConfigListener = (event: ConfigChangeEvent) => void;

export class ConfigSynchronizer {
  private static instance: ConfigSynchronizer | null = null;
  private config: ConfigState = {};
  private options: Required<Omit<ConfigSyncOptions, 'cacheManager'>> & { cacheManager?: CacheManager };
  private syncTimer?: NodeJS.Timeout;
  private listeners: Set<ConfigListener> = new Set();
  private isDestroyed = false;
  private isSyncing = false;
  private metrics: SyncMetrics = {
    totalSyncs: 0,
    successfulSyncs: 0,
    failedSyncs: 0,
    averageSyncDuration: 0,
    retryCount: 0,
  };
  private syncDurations: number[] = [];
  private maxDurationHistory = 10;

  private constructor(options: ConfigSyncOptions = {}) {
    this.options = {
      autoSync: true,
      syncInterval: 30000, // 30 seconds
      retryAttempts: 3,
      retryDelay: 1000, // 1 second
      enableChangeDetection: true,
      ...options,
    };

    if (this.options.autoSync) {
      this.startAutoSync();
    }
  }

  public static getInstance(options?: ConfigSyncOptions): ConfigSynchronizer {
    if (!ConfigSynchronizer.instance) {
      ConfigSynchronizer.instance = new ConfigSynchronizer(options);
    }
    return ConfigSynchronizer.instance;
  }

  public static resetInstance(): void {
    if (ConfigSynchronizer.instance) {
      ConfigSynchronizer.instance.destroy();
      ConfigSynchronizer.instance = null;
    }
  }

  public addListener(listener: ConfigListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public removeListener(listener: ConfigListener): void {
    this.listeners.delete(listener);
  }

  private notifyListeners(event: ConfigChangeEvent): void {
    if (!this.options.enableChangeDetection) return;
    
    this.listeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in config change listener:', error);
      }
    });
  }

  public async syncConfig(): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('ConfigSynchronizer has been destroyed');
    }

    if (this.isSyncing) {
      console.warn('Sync already in progress, skipping...');
      return;
    }

    this.isSyncing = true;
    const startTime = Date.now();
    let attempt = 0;

    while (attempt <= this.options.retryAttempts) {
      try {
        this.metrics.totalSyncs++;
        
        await this.performSync();
        
        this.metrics.successfulSyncs++;
        const duration = Date.now() - startTime;
        this.updateSyncMetrics(duration);
        
        console.log(`Config sync completed successfully in ${duration}ms`);
        break;
      } catch (error) {
        attempt++;
        this.metrics.failedSyncs++;
        
        if (attempt > this.options.retryAttempts) {
          console.error(`Config sync failed after ${this.options.retryAttempts} attempts:`, error);
          throw error;
        }
        
        this.metrics.retryCount++;
        console.warn(`Config sync attempt ${attempt} failed, retrying in ${this.options.retryDelay}ms...`, error);
        await this.delay(this.options.retryDelay * attempt);
      }
    }
    
    this.isSyncing = false;
  }

  private async performSync(): Promise<void> {
    if (this.options.cacheManager) {
      try {
        const cachedConfig = await this.options.cacheManager.get('config');
        if (cachedConfig) {
          this.updateConfigInternal(cachedConfig, false);
        }
      } catch (error) {
        console.warn('Failed to load config from cache:', error);
      }
    }
    
    await this.delay(100);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private updateSyncMetrics(duration: number): void {
    this.metrics.lastSyncTime = Date.now();
    this.metrics.lastSyncDuration = duration;
    
    this.syncDurations.push(duration);
    if (this.syncDurations.length > this.maxDurationHistory) {
      this.syncDurations.shift();
    }
    
    this.metrics.averageSyncDuration = 
      this.syncDurations.reduce((sum, d) => sum + d, 0) / this.syncDurations.length;
  }

  public getConfig(): ConfigState {
    return { ...this.config };
  }

  public getMetrics(): SyncMetrics {
    return { ...this.metrics };
  }

  public updateConfig(updates: Partial<ConfigState>, notify = true): void {
    this.updateConfigInternal(updates, notify);
  }

  private updateConfigInternal(updates: Partial<ConfigState>, notify = true): void {
    const oldConfig = { ...this.config };
    this.config = { ...this.config, ...updates };

    if (notify && this.options.enableChangeDetection) {
      Object.keys(updates).forEach(key => {
        if (oldConfig[key] !== updates[key]) {
          this.notifyListeners({
            key,
            oldValue: oldConfig[key],
            newValue: updates[key],
            timestamp: Date.now(),
          });
        }
      });
    }

    if (this.options.cacheManager) {
      try {
        const cachePromise = this.options.cacheManager.set('config', this.config);
        if (cachePromise && typeof cachePromise.catch === 'function') {
          cachePromise.catch(error => {
            console.warn('Failed to update config cache:', error);
          });
        }
      } catch (error) {
        console.warn('Failed to update config cache:', error);
      }
    }
  }

  public detectChanges(newConfig: ConfigState): ConfigChangeEvent[] {
    const changes: ConfigChangeEvent[] = [];
    const timestamp = Date.now();

    Object.keys(newConfig).forEach(key => {
      if (this.config[key] !== newConfig[key]) {
        changes.push({
          key,
          oldValue: this.config[key],
          newValue: newConfig[key],
          timestamp,
        });
      }
    });

    return changes;
  }

  private startAutoSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
    }

    this.syncTimer = setInterval(() => {
      if (!this.isDestroyed) {
        this.syncConfig().catch(console.error);
      }
    }, this.options.syncInterval);
  }

  public stopAutoSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = undefined;
    }
  }

  public resumeAutoSync(): void {
    if (!this.isDestroyed && this.options.autoSync) {
      this.startAutoSync();
    }
  }

  public isHealthy(): boolean {
    const recentFailureThreshold = 5 * 60 * 1000;
    const now = Date.now();
    
    return (
      !this.isDestroyed &&
      this.metrics.totalSyncs > 0 &&
      (this.metrics.lastSyncTime ? (now - this.metrics.lastSyncTime) < recentFailureThreshold : false) &&
      this.metrics.successfulSyncs > this.metrics.failedSyncs
    );
  }

  public destroy(): void {
    if (this.isDestroyed) return;
    
    this.isDestroyed = true;
    this.stopAutoSync();
    this.listeners.clear();
    
    if (this.options.cacheManager) {
      try {
        const clearPromise = this.options.cacheManager.clear();
        if (clearPromise && typeof clearPromise.catch === 'function') {
          clearPromise.catch(console.error);
        }
      } catch (error) {
        console.error('Failed to clear cache:', error);
      }
    }
  }
}

// Export shared instance
export const configSynchronizer = ConfigSynchronizer.getInstance();
