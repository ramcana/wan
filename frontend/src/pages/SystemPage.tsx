import React from "react";
import SystemMonitor from "@/components/system/SystemMonitor";
import SystemHealthIndicator from "@/components/system/SystemHealthIndicator";

const SystemPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">System Monitor</h1>
          <p className="text-muted-foreground">
            Monitor system resources and performance
          </p>
        </div>
        <SystemHealthIndicator showDetails />
      </div>

      <SystemMonitor />
    </div>
  );
};

export default SystemPage;
