import React from 'react';
import { AlertCircle, CheckCircle, Wifi, RefreshCw } from 'lucide-react';
import { Button } from './button';
import { Alert, AlertDescription } from './alert';
import { Badge } from './badge';
import { useBackendStatus } from '@/hooks/use-backend-status';
import { cn } from '@/lib/utils';

interface ConnectionStatusProps {
  className?: string;
  showDetails?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ 
  className, 
  showDetails = false 
}) => {
  const { isOnline, isLoading, error, retry, port, version, uptime, lastCheck } = useBackendStatus();

  const getStatusIcon = () => {
    if (isLoading) return <RefreshCw className="h-4 w-4 animate-spin" />;
    if (isOnline) return <CheckCircle className="h-4 w-4 text-green-500" />;
    return <AlertCircle className="h-4 w-4 text-red-500" />;
  };

  const getStatusText = () => {
    if (isLoading) return 'Checking connection...';
    if (isOnline) return `Backend connected${port ? ` (port ${port})` : ''}`;
    return 'Backend disconnected';
  };

  const getStatusVariant = () => {
    if (isLoading) return 'secondary';
    if (isOnline) return 'default';
    return 'destructive';
  };

  if (!showDetails) {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        {getStatusIcon()}
        <Badge variant={getStatusVariant()}>{getStatusText()}</Badge>
        {!isOnline && !isLoading && (
          <Button size="sm" variant="outline" onClick={retry}>
            Retry
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className="font-medium">{getStatusText()}</span>
        </div>
        {!isOnline && !isLoading && (
          <Button size="sm" variant="outline" onClick={retry}>
            <RefreshCw className="h-3 w-3 mr-1" />
            Retry
          </Button>
        )}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Connection Error: {error}
          </AlertDescription>
        </Alert>
      )}

      {isOnline && (
        <div className="text-sm text-muted-foreground space-y-1">
          {version && <div>Version: {version}</div>}
          {uptime && <div>Uptime: {Math.floor(uptime / 60)} minutes</div>}
          {lastCheck && (
            <div>Last check: {lastCheck.toLocaleTimeString()}</div>
          )}
        </div>
      )}

      {!isOnline && !isLoading && (
        <Alert>
          <Wifi className="h-4 w-4" />
          <AlertDescription>
            Make sure the backend server is running:
            <br />
            <code className="text-xs bg-muted px-1 py-0.5 rounded">
              $env:WAN_MODELS_ROOT="D:\AI\models"; python -m backend --host 0.0.0.0 --port 8000
            </code>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};
