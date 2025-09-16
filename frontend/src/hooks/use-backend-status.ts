import { useState, useEffect, useCallback } from 'react';
import { getSystemHealth } from '@/lib/api-client';

export interface BackendStatus {
  isOnline: boolean;
  isLoading: boolean;
  lastCheck: Date | null;
  error: string | null;
  port: number | null;
  version: string | null;
  uptime: number | null;
}

export const useBackendStatus = (checkInterval = 30000) => {
  const [status, setStatus] = useState<BackendStatus>({
    isOnline: false,
    isLoading: true,
    lastCheck: null,
    error: null,
    port: null,
    version: null,
    uptime: null,
  });

  const checkBackendHealth = useCallback(async () => {
    try {
      setStatus((prev: BackendStatus) => ({ ...prev, isLoading: true, error: null }));
      
      const response = await getSystemHealth();
      
      setStatus({
        isOnline: true,
        isLoading: false,
        lastCheck: new Date(),
        error: null,
        port: 9000, // Default port, could be detected dynamically
        version: response.version || null,
        uptime: response.uptime || null,
      });
    } catch (error) {
      setStatus({
        isOnline: false,
        isLoading: false,
        lastCheck: new Date(),
        error: error instanceof Error ? error.message : 'Connection failed',
        port: null,
        version: null,
        uptime: null,
      });
    }
  }, []);

  useEffect(() => {
    // Initial check
    checkBackendHealth();

    // Set up interval for periodic checks
    const interval = setInterval(checkBackendHealth, checkInterval);

    return () => clearInterval(interval);
  }, [checkBackendHealth, checkInterval]);

  const retry = useCallback(() => {
    checkBackendHealth();
  }, [checkBackendHealth]);

  return {
    ...status,
    retry,
    checkHealth: checkBackendHealth,
  };
};
