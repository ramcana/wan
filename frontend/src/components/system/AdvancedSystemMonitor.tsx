import React, { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSystemMonitoring } from "@/hooks/api/use-system";
import { useWebSocket } from "@/hooks/use-websocket";
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  TrendingUp,
  Settings,
  Zap,
} from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import "chartjs-adapter-date-fns";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

interface SystemMetrics {
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  gpu_percent: number;
  vram_used_mb: number;
  vram_total_mb: number;
  vram_percent: number;
  timestamp: string;
}

interface TimeRange {
  label: string;
  hours: number;
  updateInterval: number; // in milliseconds
}

const TIME_RANGES: TimeRange[] = [
  { label: "Last 5 minutes", hours: 0.083, updateInterval: 1000 },
  { label: "Last 15 minutes", hours: 0.25, updateInterval: 2000 },
  { label: "Last hour", hours: 1, updateInterval: 5000 },
  { label: "Last 6 hours", hours: 6, updateInterval: 30000 },
  { label: "Last 24 hours", hours: 24, updateInterval: 60000 },
  { label: "Last week", hours: 168, updateInterval: 300000 },
];

interface OptimizationPreset {
  name: string;
  description: string;
  settings: {
    quantization: "fp16" | "bf16" | "int8";
    enable_offload: boolean;
    vae_tile_size: number;
    max_vram_usage_gb: number;
  };
  vram_savings_gb: number;
  performance_impact: "low" | "medium" | "high";
}

const OPTIMIZATION_PRESETS: OptimizationPreset[] = [
  {
    name: "Balanced",
    description: "Good balance of quality and performance",
    settings: {
      quantization: "fp16",
      enable_offload: false,
      vae_tile_size: 512,
      max_vram_usage_gb: 12.0,
    },
    vram_savings_gb: 2.0,
    performance_impact: "low",
  },
  {
    name: "Memory Efficient",
    description: "Optimized for lower VRAM usage",
    settings: {
      quantization: "bf16",
      enable_offload: true,
      vae_tile_size: 256,
      max_vram_usage_gb: 8.0,
    },
    vram_savings_gb: 4.5,
    performance_impact: "medium",
  },
  {
    name: "Ultra Efficient",
    description: "Maximum memory savings for low-end GPUs",
    settings: {
      quantization: "int8",
      enable_offload: true,
      vae_tile_size: 128,
      max_vram_usage_gb: 6.0,
    },
    vram_savings_gb: 6.0,
    performance_impact: "high",
  },
  {
    name: "High Performance",
    description: "Best quality for high-end GPUs",
    settings: {
      quantization: "fp16",
      enable_offload: false,
      vae_tile_size: 512,
      max_vram_usage_gb: 20.0,
    },
    vram_savings_gb: 1.0,
    performance_impact: "low",
  },
];

const AdvancedSystemMonitor: React.FC = () => {
  const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>(
    TIME_RANGES[2]
  ); // Default to 1 hour
  const [historicalData, setHistoricalData] = useState<SystemMetrics[]>([]);
  const [selectedPreset, setSelectedPreset] =
    useState<OptimizationPreset | null>(null);
  const [showOptimizationPanel, setShowOptimizationPanel] = useState(false);

  const { stats, alerts, isLoading, error, hasAlerts } = useSystemMonitoring();

  // WebSocket for real-time updates
  const {
    isConnected: wsConnected,
    subscribe,
    unsubscribe,
    lastMessage,
  } = useWebSocket(
    `ws://${
      import.meta.env.VITE_API_URL?.replace("http://", "") || "localhost:8000"
    }/ws`
  );

  const chartRef = useRef<ChartJS<"line", any, any>>(null);

  // Subscribe to real-time system stats
  useEffect(() => {
    if (wsConnected) {
      subscribe("system_stats");
      return () => unsubscribe("system_stats");
    }
  }, [wsConnected, subscribe, unsubscribe]);

  // Handle real-time WebSocket updates
  useEffect(() => {
    if (lastMessage?.type === "system_stats_update") {
      const newMetrics: SystemMetrics = lastMessage.data;

      setHistoricalData((prev) => {
        const updated = [...prev, newMetrics];

        // Keep only data within selected time range
        const cutoffTime = new Date(
          Date.now() - selectedTimeRange.hours * 60 * 60 * 1000
        );
        const filtered = updated.filter(
          (metric) => new Date(metric.timestamp) > cutoffTime
        );

        // Limit to reasonable number of points for performance
        const maxPoints = Math.min(
          1000,
          Math.max(100, selectedTimeRange.hours * 60)
        );
        return filtered.slice(-maxPoints);
      });
    }
  }, [lastMessage, selectedTimeRange.hours]);

  // Fetch historical data when time range changes
  useEffect(() => {
    const fetchHistoricalData = async () => {
      try {
        const response = await fetch(
          `/api/v1/system/stats/history?hours=${selectedTimeRange.hours}`
        );
        if (response.ok) {
          const data = await response.json();
          setHistoricalData(data.stats || []);
        }
      } catch (error) {
        console.error("Failed to fetch historical data:", error);
      }
    };

    fetchHistoricalData();
  }, [selectedTimeRange]);

  // Chart configuration
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index" as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: `System Resources - ${selectedTimeRange.label}`,
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            const label = context.dataset.label || "";
            const value = context.parsed.y;
            const unit = context.dataset.unit || "%";
            return `${label}: ${value.toFixed(1)}${unit}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: "time" as const,
        time: {
          displayFormats: {
            minute: "HH:mm",
            hour: "HH:mm",
            day: "MMM dd",
          },
        },
        title: {
          display: true,
          text: "Time",
        },
      },
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: "Usage (%)",
        },
      },
    },
    animation: {
      duration: 300,
      easing: "easeInOutQuart" as const,
    },
  };

  const chartData = {
    labels: historicalData.map((d) => new Date(d.timestamp)),
    datasets: [
      {
        label: "CPU",
        data: historicalData.map((d) => d.cpu_percent),
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        fill: false,
        tension: 0.1,
        unit: "%",
      },
      {
        label: "RAM",
        data: historicalData.map((d) => d.ram_percent),
        borderColor: "rgb(16, 185, 129)",
        backgroundColor: "rgba(16, 185, 129, 0.1)",
        fill: false,
        tension: 0.1,
        unit: "%",
      },
      {
        label: "GPU",
        data: historicalData.map((d) => d.gpu_percent),
        borderColor: "rgb(245, 158, 11)",
        backgroundColor: "rgba(245, 158, 11, 0.1)",
        fill: false,
        tension: 0.1,
        unit: "%",
      },
      {
        label: "VRAM",
        data: historicalData.map((d) => d.vram_percent),
        borderColor: "rgb(239, 68, 68)",
        backgroundColor: "rgba(239, 68, 68, 0.1)",
        fill: false,
        tension: 0.1,
        unit: "%",
      },
    ],
  };

  const handlePresetApply = async (preset: OptimizationPreset) => {
    try {
      const response = await fetch("/api/v1/system/optimization", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(preset.settings),
      });

      if (response.ok) {
        setSelectedPreset(preset);
        // Show success notification
      }
    } catch (error) {
      console.error("Failed to apply optimization preset:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Activity className="h-6 w-6 animate-pulse" />
          <h2 className="text-xl font-semibold">
            Loading Advanced System Monitor...
          </h2>
        </div>
        <div className="grid grid-cols-1 gap-6">
          <Card className="p-6 h-96">
            <div className="animate-pulse space-y-4">
              <div className="h-4 bg-muted rounded w-1/2"></div>
              <div className="h-64 bg-muted rounded"></div>
            </div>
          </Card>
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
            Failed to Load Advanced System Monitor
          </h2>
        </div>
        <p className="text-red-700 dark:text-red-300 mt-2">
          Unable to retrieve system monitoring data. Please check your
          connection and try again.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-6 w-6" />
          <h2 className="text-xl font-semibold">Advanced System Monitor</h2>
          <Badge variant={wsConnected ? "default" : "secondary"}>
            {wsConnected ? "Live" : "Offline"}
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          <Select
            value={selectedTimeRange.label}
            onValueChange={(value) => {
              const range = TIME_RANGES.find((r) => r.label === value);
              if (range) setSelectedTimeRange(range);
            }}
          >
            <SelectTrigger className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TIME_RANGES.map((range) => (
                <SelectItem key={range.label} value={range.label}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowOptimizationPanel(!showOptimizationPanel)}
          >
            <Settings className="h-4 w-4 mr-2" />
            Optimize
          </Button>
        </div>
      </div>

      {/* Real-time Chart */}
      <Card className="p-6">
        <div className="h-96">
          <Line ref={chartRef} data={chartData} options={chartOptions} />
        </div>
      </Card>

      {/* Current Stats Summary */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">CPU</p>
                <p className="text-2xl font-bold">
                  {stats.cpu_percent.toFixed(1)}%
                </p>
              </div>
              <div
                className={`w-3 h-3 rounded-full ${
                  stats.cpu_percent > 90
                    ? "bg-red-500"
                    : stats.cpu_percent > 80
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
              />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">RAM</p>
                <p className="text-2xl font-bold">
                  {stats.ram_percent.toFixed(1)}%
                </p>
              </div>
              <div
                className={`w-3 h-3 rounded-full ${
                  stats.ram_percent > 90
                    ? "bg-red-500"
                    : stats.ram_percent > 80
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
              />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">GPU</p>
                <p className="text-2xl font-bold">
                  {stats.gpu_percent.toFixed(1)}%
                </p>
              </div>
              <div
                className={`w-3 h-3 rounded-full ${
                  stats.gpu_percent > 90
                    ? "bg-red-500"
                    : stats.gpu_percent > 80
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
              />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">VRAM</p>
                <p className="text-2xl font-bold">
                  {stats.vram_percent.toFixed(1)}%
                </p>
              </div>
              <div
                className={`w-3 h-3 rounded-full ${
                  stats.vram_percent > 90
                    ? "bg-red-500"
                    : stats.vram_percent > 80
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
              />
            </div>
          </Card>
        </div>
      )}

      {/* Optimization Panel */}
      {showOptimizationPanel && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="h-5 w-5" />
            <h3 className="text-lg font-semibold">Optimization Presets</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {OPTIMIZATION_PRESETS.map((preset) => (
              <Card
                key={preset.name}
                className={`p-4 cursor-pointer transition-colors hover:bg-muted/50 ${
                  selectedPreset?.name === preset.name
                    ? "ring-2 ring-primary"
                    : ""
                }`}
                onClick={() => handlePresetApply(preset)}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">{preset.name}</h4>
                    <Badge
                      variant={
                        preset.performance_impact === "low"
                          ? "default"
                          : preset.performance_impact === "medium"
                          ? "secondary"
                          : "destructive"
                      }
                    >
                      {preset.performance_impact}
                    </Badge>
                  </div>

                  <p className="text-sm text-muted-foreground">
                    {preset.description}
                  </p>

                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>VRAM Savings:</span>
                      <span className="font-medium text-green-600">
                        -{preset.vram_savings_gb}GB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Quantization:</span>
                      <span className="font-medium">
                        {preset.settings.quantization}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Offloading:</span>
                      <span className="font-medium">
                        {preset.settings.enable_offload ? "Yes" : "No"}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>
      )}

      {/* Alerts */}
      {hasAlerts && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500" />
            System Alerts
          </h3>
          <div className="space-y-3">
            {alerts.map((alert, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  alert.type === "error"
                    ? "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20"
                    : "border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20"
                }`}
              >
                <div className="flex items-start gap-2">
                  {alert.type === "error" ? (
                    <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <p
                      className={`font-medium ${
                        alert.type === "error"
                          ? "text-red-800 dark:text-red-200"
                          : "text-yellow-800 dark:text-yellow-200"
                      }`}
                    >
                      {alert.message}
                    </p>
                    <p
                      className={`text-sm mt-1 ${
                        alert.type === "error"
                          ? "text-red-700 dark:text-red-300"
                          : "text-yellow-700 dark:text-yellow-300"
                      }`}
                    >
                      {alert.suggestion}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default AdvancedSystemMonitor;
