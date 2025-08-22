import { useQuery } from 'react-query';
import { get, ApiError } from '@/lib/api-client';

// Query keys
export const systemKeys = {
  all: ['system'] as const,
  stats: () => [...systemKeys.all, 'stats'] as const,
  health: () => [...systemKeys.all, 'health'] as const,
};

// Get system stats query
export const useSystemStats = () => {
  return useQuery({
    queryKey: systemKeys.stats(),
    queryFn: () => get('/system/stats'),
    refetchInterval: 10000, // Poll every 10 seconds
    staleTime: 5000, // Consider data stale after 5 seconds
    retry: (failureCount, error) => {
      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

// Get system health query
export const useSystemHealth = () => {
  return useQuery({
    queryKey: systemKeys.health(),
    queryFn: () => get('/system/health'),
    refetchInterval: 30000, // Poll every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
    retry: (failureCount, error) => {
      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

// Custom hook for monitoring system resources with alerts
export const useSystemMonitoring = () => {
  const { data: stats, isLoading, error } = useSystemStats();
  
  // Calculate alert levels
  const alerts = [];
  
  if (stats) {
    if (stats.vram_percent > 90) {
      alerts.push({
        type: 'error' as const,
        message: `VRAM usage critical: ${stats.vram_percent.toFixed(1)}%`,
        suggestion: 'Consider reducing model precision or batch size'
      });
    } else if (stats.vram_percent > 80) {
      alerts.push({
        type: 'warning' as const,
        message: `VRAM usage high: ${stats.vram_percent.toFixed(1)}%`,
        suggestion: 'Monitor memory usage closely'
      });
    }
    
    if (stats.ram_percent > 90) {
      alerts.push({
        type: 'error' as const,
        message: `RAM usage critical: ${stats.ram_percent.toFixed(1)}%`,
        suggestion: 'Close unnecessary applications'
      });
    } else if (stats.ram_percent > 80) {
      alerts.push({
        type: 'warning' as const,
        message: `RAM usage high: ${stats.ram_percent.toFixed(1)}%`,
        suggestion: 'Consider freeing up system memory'
      });
    }
    
    if (stats.cpu_percent > 95) {
      alerts.push({
        type: 'warning' as const,
        message: `CPU usage very high: ${stats.cpu_percent.toFixed(1)}%`,
        suggestion: 'System may be under heavy load'
      });
    }
  }
  
  return {
    stats,
    alerts,
    isLoading,
    error,
    hasAlerts: alerts.length > 0,
    criticalAlerts: alerts.filter(a => a.type === 'error'),
    warningAlerts: alerts.filter(a => a.type === 'warning'),
  };
};