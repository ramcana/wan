export interface ConfigSyncOptions {
  autoSync?: boolean
  syncInterval?: number
  retryAttempts?: number
  cacheManager?: CacheManagerLike
}

export type ConfigChangeType =
  | 'api_url_changed'
  | 'port_changed'
  | 'dev_mode_changed'

export interface ConfigChange {
  type: ConfigChangeType
  oldValue: unknown
  newValue: unknown
  timestamp: Date
}

export interface ConfigState {
  apiUrl: string
  devMode: boolean
  backendPort: number
  frontendPort: number
  lastUpdated: Date
}

interface EnvironmentConfig {
  apiUrl: string
  devMode: boolean
  backendPort: number
}

interface CacheManagerLike {
  clearAllCaches: () => Promise<unknown> | unknown
  updateApiUrl: (url: string) => Promise<unknown> | unknown
  forceReload: () => void
}

const DEFAULT_API_URL = 'http://localhost:9000'
const DEFAULT_BACKEND_PORT = 9000
const DEFAULT_FRONTEND_PORT = 3000
const DEFAULT_MONITOR_INTERVAL = 5000
const DEV_MODE_MONITOR_INTERVAL = 3000
const DEFAULT_RETRY_ATTEMPTS = 3

export class ConfigSynchronizer {
  private static instance: ConfigSynchronizer | undefined

  private config: ConfigState
  private options: ConfigSyncOptions
  private monitorTimer?: ReturnType<typeof setInterval>
  private listeners: Set<(change: ConfigChange) => void> = new Set()
  private retryAttempts: number
  private cacheManager: CacheManagerLike | null

  private constructor(options: ConfigSyncOptions = {}) {
    this.options = {
      autoSync: false,
      syncInterval: DEFAULT_MONITOR_INTERVAL,
      retryAttempts: DEFAULT_RETRY_ATTEMPTS,
      ...options,
    }

    this.retryAttempts = this.options.retryAttempts ?? DEFAULT_RETRY_ATTEMPTS
    this.cacheManager = this.options.cacheManager ?? null
    this.config = this.createConfigFromEnv()

    if (this.options.autoSync) {
      this.startMonitoring(this.options.syncInterval)
    }
  }

  static getInstance(options: ConfigSyncOptions = {}): ConfigSynchronizer {
    if (!ConfigSynchronizer.instance) {
      ConfigSynchronizer.instance = new ConfigSynchronizer(options)
    } else if (Object.keys(options).length > 0) {
      ConfigSynchronizer.instance.updateOptions(options)
    }

    return ConfigSynchronizer.instance
  }

  private updateOptions(options: ConfigSyncOptions): void {
    this.options = { ...this.options, ...options }

    if (typeof options.retryAttempts === 'number') {
      this.retryAttempts = options.retryAttempts
    }

    if (options.cacheManager) {
      this.cacheManager = options.cacheManager
    }

    if (this.options.autoSync && !this.monitorTimer) {
      this.startMonitoring(this.options.syncInterval)
    }
  }

  getCurrentConfig(): ConfigState {
    return {
      ...this.config,
      lastUpdated: new Date(this.config.lastUpdated),
    }
  }

  // Backwards compatibility with previous API
  getConfig(): ConfigState {
    return this.getCurrentConfig()
  }

  addChangeListener(listener: (change: ConfigChange) => void): void {
    this.listeners.add(listener)
  }

  removeChangeListener(listener: (change: ConfigChange) => void): void {
    this.listeners.delete(listener)
  }

  updateConfig(updates: Partial<ConfigState>): void {
    this.config = {
      ...this.config,
      ...updates,
      lastUpdated: updates.lastUpdated ?? new Date(),
    }
  }

  async checkForChanges(): Promise<ConfigChange[]> {
    return this.executeWithRetry(() => this.performCheckForChanges())
  }

  startMonitoring(interval?: number): void {
    const effectiveInterval =
      interval ?? this.options.syncInterval ?? DEFAULT_MONITOR_INTERVAL

    this.stopMonitoring()

    this.monitorTimer = setInterval(() => {
      this.executeWithRetry(() => this.performCheckForChanges()).catch(
        (error) => {
          console.error('Config monitoring failed', error)
        },
      )
    }, effectiveInterval)
  }

  stopMonitoring(): void {
    if (this.monitorTimer) {
      clearInterval(this.monitorTimer)
      this.monitorTimer = undefined
    }
  }

  async forceSync(): Promise<ConfigChange[]> {
    return this.checkForChanges()
  }

  async initialize(): Promise<void> {
    await this.checkForChanges()

    if (this.config.devMode) {
      this.startMonitoring(DEV_MODE_MONITOR_INTERVAL)
    }
  }

  cleanup(): void {
    this.stopMonitoring()
    this.listeners.clear()
    this.config = this.createConfigFromEnv()

    if (ConfigSynchronizer.instance === this) {
      ConfigSynchronizer.instance = undefined
    }
  }

  private async performCheckForChanges(): Promise<ConfigChange[]> {
    const previousConfig = this.config
    const envConfig = this.readEnvironmentConfig()
    const detectedChanges: ConfigChange[] = []

    if (envConfig.apiUrl !== previousConfig.apiUrl) {
      detectedChanges.push({
        type: 'api_url_changed',
        oldValue: previousConfig.apiUrl,
        newValue: envConfig.apiUrl,
        timestamp: new Date(),
      })
    }

    if (envConfig.backendPort !== previousConfig.backendPort) {
      detectedChanges.push({
        type: 'port_changed',
        oldValue: previousConfig.backendPort,
        newValue: envConfig.backendPort,
        timestamp: new Date(),
      })
    }

    if (envConfig.devMode !== previousConfig.devMode) {
      detectedChanges.push({
        type: 'dev_mode_changed',
        oldValue: previousConfig.devMode,
        newValue: envConfig.devMode,
        timestamp: new Date(),
      })
    }

    if (!detectedChanges.length) {
      this.config = { ...previousConfig, lastUpdated: new Date() }
      return []
    }

    await this.applySynchronizationEffects(detectedChanges, envConfig)

    const updatedTimestamp = new Date()
    this.config = {
      apiUrl: envConfig.apiUrl,
      devMode: envConfig.devMode,
      backendPort: envConfig.backendPort,
      frontendPort: previousConfig.frontendPort,
      lastUpdated: updatedTimestamp,
    }

    const normalizedChanges = detectedChanges.map((change) => ({
      ...change,
      timestamp: updatedTimestamp,
    }))

    normalizedChanges.forEach((change) => this.notifyListeners(change))

    return normalizedChanges
  }

  private async applySynchronizationEffects(
    changes: ConfigChange[],
    envConfig: EnvironmentConfig,
  ): Promise<void> {
    const manager = await this.resolveCacheManager()

    if (!manager) {
      return
    }

    const hasUrlChange = changes.some(
      (change) =>
        change.type === 'api_url_changed' || change.type === 'port_changed',
    )

    const hasDevModeChange = changes.some(
      (change) => change.type === 'dev_mode_changed',
    )

    try {
      if (hasUrlChange) {
        await manager.updateApiUrl?.(envConfig.apiUrl)
        await manager.clearAllCaches?.()
      }

      if (hasDevModeChange) {
        manager.forceReload?.()
      }
    } catch (error) {
      console.error('Failed to apply configuration changes', error)
      throw error
    }
  }

  private notifyListeners(change: ConfigChange): void {
    this.listeners.forEach((listener) => {
      try {
        listener(change)
      } catch (error) {
        console.error('Config change listener failed', error)
      }
    })
  }

  private readEnvironmentConfig(): EnvironmentConfig {
    const env =
      (import.meta as unknown as { env?: Record<string, unknown> }).env ?? {}

    const currentApiUrl =
      typeof env.VITE_API_URL === 'string' && env.VITE_API_URL.trim().length > 0
        ? (env.VITE_API_URL as string).trim()
        : this.config?.apiUrl ?? DEFAULT_API_URL

    const devMode = this.parseBoolean(
      env.VITE_DEV_MODE ?? env.DEV,
      this.config?.devMode ?? false,
    )

    const backendPort = this.extractPort(
      currentApiUrl,
      this.config?.backendPort ?? DEFAULT_BACKEND_PORT,
    )

    return {
      apiUrl: currentApiUrl,
      devMode,
      backendPort,
    }
  }

  private createConfigFromEnv(): ConfigState {
    const envConfig = this.readEnvironmentConfig()

    return {
      apiUrl: envConfig.apiUrl,
      devMode: envConfig.devMode,
      backendPort: envConfig.backendPort,
      frontendPort: DEFAULT_FRONTEND_PORT,
      lastUpdated: new Date(),
    }
  }

  private parseBoolean(value: unknown, fallback: boolean): boolean {
    if (typeof value === 'boolean') {
      return value
    }

    if (typeof value === 'string') {
      const normalized = value.toLowerCase()

      if (normalized === 'true') {
        return true
      }

      if (normalized === 'false') {
        return false
      }
    }

    return fallback
  }

  private extractPort(apiUrl: string, fallback: number): number {
    try {
      const url = new URL(apiUrl)

      if (url.port) {
        return Number(url.port)
      }

      return url.protocol === 'https:' ? 443 : 80
    } catch {
      return fallback
    }
  }

  private async executeWithRetry<T>(operation: () => Promise<T>): Promise<T> {
    let attempt = 0
    let lastError: unknown

    while (attempt < this.retryAttempts) {
      try {
        return await operation()
      } catch (error) {
        lastError = error
        attempt += 1

        if (attempt >= this.retryAttempts) {
          break
        }
      }
    }

    throw lastError
  }

  private async resolveCacheManager(): Promise<CacheManagerLike | null> {
    if (this.cacheManager) {
      return this.cacheManager
    }

    const globalManager = (globalThis as unknown as {
      cacheManager?: CacheManagerLike
    }).cacheManager

    if (globalManager) {
      this.cacheManager = globalManager
      return this.cacheManager
    }

    try {
      const dynamicImport = new Function(
        'path',
        'return import(path);',
      ) as (path: string) => Promise<{ cacheManager?: CacheManagerLike }>

      const module = await dynamicImport('./cache-manager')

      if (module?.cacheManager) {
        this.cacheManager = module.cacheManager
        return this.cacheManager
      }
    } catch {
      // Ignore module resolution errors and continue without cache manager support
    }

    return null
  }
}

export const configSynchronizer = ConfigSynchronizer.getInstance()
