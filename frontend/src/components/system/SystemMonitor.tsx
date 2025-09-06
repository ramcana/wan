import React from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useSystemMonitoring } from "@/hooks/api/use-system";
import { AlertTriangle, CheckCircle, XCircle, Activity } from "lucide-react";

interface ResourceBarProps {
  label: string;
  value: number;
  total?: number;
  unit: string;
  color: "green" | "yellow" | "red";
}

const ResourceBar: React.FC<ResourceBarProps> = ({
  label,
  value,
  total,
  unit,
  color,
}) => {
  const percentage = total ? ((value || 0) / (total || 1)) * 100 : (value || 0);

  const colorClasses = {
    green: "bg-green-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
  };

  const bgColorClasses = {
    green: "bg-green-100 dark:bg-green-900/20",
    yellow: "bg-yellow-100 dark:bg-yellow-900/20",
    red: "bg-red-100 dark:bg-red-900/20",
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-sm text-muted-foreground">
          {total
            ? `${(value || 0).toFixed(1)} / ${(total || 0).toFixed(1)} ${unit}`
            : `${(value || 0).toFixed(1)}${unit}`}
        </span>
      </div>
      <div className={`w-full h-3 rounded-full ${bgColorClasses[color]}`}>
        <div
          className={`h-full rounded-full transition-all duration-300 ${colorClasses[color]}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="text-xs text-muted-foreground text-right">
        {(percentage || 0).toFixed(1)}%
      </div>
    </div>
  );
};

interface AlertCardProps {
  alerts: Array<{
    type: "error" | "warning";
    message: string;
    suggestion: string;
  }>;
}

const AlertCard: React.FC<AlertCardProps> = ({ alerts }) => {
  if (alerts.length === 0) {
    return (
      <Card className="p-4 border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20">
        <div className="flex items-center gap-2">
          <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
          <span className="text-green-800 dark:text-green-200 font-medium">
            System Running Normally
          </span>
        </div>
        <p className="text-green-700 dark:text-green-300 text-sm mt-1">
          All system resources are within normal operating ranges.
        </p>
      </Card>
    );
  }

  const criticalAlerts = alerts.filter((a) => a.type === "error");
  const warningAlerts = alerts.filter((a) => a.type === "warning");

  return (
    <div className="space-y-3">
      {criticalAlerts.map((alert, index) => (
        <Card
          key={`critical-${index}`}
          className="p-4 border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20"
        >
          <div className="flex items-start gap-2">
            <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-red-800 dark:text-red-200 font-medium">
                {alert.message}
              </p>
              <p className="text-red-700 dark:text-red-300 text-sm mt-1">
                {alert.suggestion}
              </p>
            </div>
          </div>
        </Card>
      ))}

      {warningAlerts.map((alert, index) => (
        <Card
          key={`warning-${index}`}
          className="p-4 border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20"
        >
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-yellow-800 dark:text-yellow-200 font-medium">
                {alert.message}
              </p>
              <p className="text-yellow-700 dark:text-yellow-300 text-sm mt-1">
                {alert.suggestion}
              </p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};

const SystemMonitor: React.FC = () => {
  const { stats, alerts, isLoading, error, hasAlerts } = useSystemMonitoring();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Activity className="h-6 w-6 animate-pulse" />
          <h2 className="text-xl font-semibold">Loading System Stats...</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="p-6">
              <div className="animate-pulse space-y-4">
                <div className="h-4 bg-muted rounded w-1/2"></div>
                <div className="h-3 bg-muted rounded"></div>
                <div className="h-2 bg-muted rounded w-1/4"></div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-6 border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20">
        <div className="flex items-center gap-2">
          <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
          <h2 className="text-xl font-semibold text-red-800 dark:text-red-200">
            Failed to Load System Stats
          </h2>
        </div>
        <p className="text-red-700 dark:text-red-300 mt-2">
          Unable to retrieve system monitoring data. Please check your
          connection and try again.
        </p>
      </Card>
    );
  }

  if (!stats) {
    return (
      <Card className="p-6">
        <p className="text-center text-muted-foreground">
          No system data available
        </p>
      </Card>
    );
  }

  // Determine resource bar colors based on usage
  const getResourceColor = (percentage: number): "green" | "yellow" | "red" => {
    const safePercentage = percentage || 0;
    if (safePercentage >= 90) return "red";
    if (safePercentage >= 80) return "yellow";
    return "green";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-6 w-6" />
          <h2 className="text-xl font-semibold">System Monitor</h2>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={hasAlerts ? "destructive" : "default"}>
            {hasAlerts
              ? `${alerts.length} Alert${alerts.length > 1 ? "s" : ""}`
              : "Healthy"}
          </Badge>
          <span className="text-sm text-muted-foreground">
            Updated: {new Date(stats.timestamp).toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Alerts */}
      <AlertCard alerts={alerts} />

      {/* Resource Usage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* CPU Usage */}
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">CPU Usage</h3>
          <ResourceBar
            label="CPU"
            value={stats.cpu_percent || 0}
            unit="%"
            color={getResourceColor(stats.cpu_percent || 0)}
          />
        </Card>

        {/* RAM Usage */}
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">System Memory</h3>
          <ResourceBar
            label="RAM"
            value={stats.ram_used_gb || 0}
            total={stats.ram_total_gb || 1}
            unit="GB"
            color={getResourceColor(stats.ram_percent || 0)}
          />
        </Card>

        {/* GPU Usage */}
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">GPU Usage</h3>
          <ResourceBar
            label="GPU"
            value={stats.gpu_percent || 0}
            unit="%"
            color={getResourceColor(stats.gpu_percent || 0)}
          />
        </Card>

        {/* VRAM Usage */}
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Video Memory</h3>
          <ResourceBar
            label="VRAM"
            value={(stats.vram_used_mb || 0) / 1024}
            total={(stats.vram_total_mb || 1024) / 1024}
            unit="GB"
            color={getResourceColor(stats.vram_percent || 0)}
          />
        </Card>
      </div>

      {/* System Health Summary */}
      <Card className="p-6">
        <h3 className="text-lg font-medium mb-4">System Health Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-foreground">
              {(stats.cpu_percent ?? 0).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">CPU</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-foreground">
              {(stats.ram_percent ?? 0).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">RAM</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-foreground">
              {(stats.gpu_percent ?? 0).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">GPU</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-foreground">
              {(stats.vram_percent ?? 0).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">VRAM</div>
          </div>
        </div>
      </Card>

      {/* Optimization Suggestions */}
      {hasAlerts && (
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Optimization Suggestions</h3>
          <div className="space-y-2">
            {stats.vram_percent > 80 && (
              <div className="flex items-start gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0"></div>
                <p className="text-sm text-muted-foreground">
                  Enable model offloading to reduce VRAM usage
                </p>
              </div>
            )}
            {stats.vram_percent > 85 && (
              <div className="flex items-start gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0"></div>
                <p className="text-sm text-muted-foreground">
                  Use int8 quantization for better memory efficiency
                </p>
              </div>
            )}
            {stats.ram_percent > 80 && (
              <div className="flex items-start gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0"></div>
                <p className="text-sm text-muted-foreground">
                  Close unnecessary applications to free up system memory
                </p>
              </div>
            )}
            {stats.cpu_percent > 90 && (
              <div className="flex items-start gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0"></div>
                <p className="text-sm text-muted-foreground">
                  Reduce concurrent generation tasks to lower CPU load
                </p>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};

export default SystemMonitor;
