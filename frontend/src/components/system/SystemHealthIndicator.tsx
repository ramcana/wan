import React from "react";
import { Badge } from "@/components/ui/badge";
import { useSystemHealth } from "@/hooks/api/use-system";
import { CheckCircle, AlertTriangle, XCircle, Loader2 } from "lucide-react";

interface SystemHealthIndicatorProps {
  showDetails?: boolean;
  className?: string;
}

const SystemHealthIndicator: React.FC<SystemHealthIndicatorProps> = ({
  showDetails = false,
  className = "",
}) => {
  const { data: health, isLoading, error } = useSystemHealth();

  if (isLoading) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-sm text-muted-foreground">Checking...</span>
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <XCircle className="h-4 w-4 text-red-500" />
        <Badge variant="destructive">Offline</Badge>
        {showDetails && (
          <span className="text-sm text-muted-foreground">
            Unable to connect to system
          </span>
        )}
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (health.status) {
      case "healthy":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case "critical":
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = () => {
    switch (health.status) {
      case "healthy":
        return (
          <Badge
            variant="default"
            className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
          >
            Healthy
          </Badge>
        );
      case "warning":
        return (
          <Badge
            variant="secondary"
            className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
          >
            Warning
          </Badge>
        );
      case "critical":
        return <Badge variant="destructive">Critical</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {getStatusIcon()}
      {getStatusBadge()}
      {showDetails && (
        <div className="flex flex-col">
          <span className="text-sm text-foreground">{health.message}</span>
          {health.system_info?.warnings &&
            health.system_info.warnings.length > 0 && (
              <span className="text-xs text-muted-foreground">
                {health.system_info.warnings.length} warning
                {health.system_info.warnings.length > 1 ? "s" : ""}
              </span>
            )}
          {health.system_info?.issues &&
            health.system_info.issues.length > 0 && (
              <span className="text-xs text-red-600 dark:text-red-400">
                {health.system_info.issues.length} issue
                {health.system_info.issues.length > 1 ? "s" : ""}
              </span>
            )}
        </div>
      )}
    </div>
  );
};

export default SystemHealthIndicator;
