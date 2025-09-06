import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Calendar, Clock, TrendingUp, BarChart3 } from "lucide-react";

interface TimeRange {
  label: string;
  hours: number;
  updateInterval: number;
  description: string;
  icon: React.ReactNode;
}

const TIME_RANGES: TimeRange[] = [
  {
    label: "Real-time",
    hours: 0.017, // ~1 minute
    updateInterval: 500,
    description: "Live updates every 500ms",
    icon: <TrendingUp className="h-4 w-4" />,
  },
  {
    label: "5 minutes",
    hours: 0.083,
    updateInterval: 1000,
    description: "Recent activity",
    icon: <Clock className="h-4 w-4" />,
  },
  {
    label: "15 minutes",
    hours: 0.25,
    updateInterval: 2000,
    description: "Short-term trends",
    icon: <Clock className="h-4 w-4" />,
  },
  {
    label: "1 hour",
    hours: 1,
    updateInterval: 5000,
    description: "Hourly patterns",
    icon: <BarChart3 className="h-4 w-4" />,
  },
  {
    label: "6 hours",
    hours: 6,
    updateInterval: 30000,
    description: "Daily patterns",
    icon: <BarChart3 className="h-4 w-4" />,
  },
  {
    label: "24 hours",
    hours: 24,
    updateInterval: 60000,
    description: "Full day analysis",
    icon: <Calendar className="h-4 w-4" />,
  },
  {
    label: "1 week",
    hours: 168,
    updateInterval: 300000,
    description: "Weekly trends",
    icon: <Calendar className="h-4 w-4" />,
  },
];

interface TimeRangeSelectorProps {
  selectedRange: TimeRange;
  onRangeChange: (range: TimeRange) => void;
  className?: string;
}

const TimeRangeSelector: React.FC<TimeRangeSelectorProps> = ({
  selectedRange,
  onRangeChange,
  className = "",
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Compact selector */}
      <div className="flex items-center gap-2 flex-wrap">
        {TIME_RANGES.map((range) => (
          <Button
            key={range.label}
            variant={
              selectedRange.label === range.label ? "default" : "outline"
            }
            size="sm"
            onClick={() => onRangeChange(range)}
            className="flex items-center gap-1"
          >
            {range.icon}
            {range.label}
          </Button>
        ))}

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="ml-2"
        >
          {isExpanded ? "Less" : "More"}
        </Button>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <Card className="p-4">
          <h3 className="text-sm font-medium mb-3">Time Range Details</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {TIME_RANGES.map((range) => (
              <div
                key={range.label}
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedRange.label === range.label
                    ? "border-primary bg-primary/5"
                    : "border-border hover:bg-muted/50"
                }`}
                onClick={() => onRangeChange(range)}
              >
                <div className="flex items-center gap-2 mb-1">
                  {range.icon}
                  <span className="font-medium text-sm">{range.label}</span>
                  {selectedRange.label === range.label && (
                    <Badge variant="default" className="text-xs">
                      Active
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  {range.description}
                </p>
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span>Duration:</span>
                    <span className="font-medium">
                      {range.hours < 1
                        ? `${Math.round(range.hours * 60)}m`
                        : range.hours < 24
                        ? `${range.hours}h`
                        : `${Math.round(range.hours / 24)}d`}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Update rate:</span>
                    <span className="font-medium">
                      {range.updateInterval < 1000
                        ? `${range.updateInterval}ms`
                        : range.updateInterval < 60000
                        ? `${range.updateInterval / 1000}s`
                        : `${range.updateInterval / 60000}m`}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Current selection summary */}
          <div className="mt-4 p-3 bg-muted/50 rounded-lg">
            <h4 className="text-sm font-medium mb-2">Current Selection</h4>
            <div className="flex items-center gap-2 mb-2">
              {selectedRange.icon}
              <span className="font-medium">{selectedRange.label}</span>
              <Badge variant="outline">{selectedRange.description}</Badge>
            </div>
            <div className="text-xs text-muted-foreground space-y-1">
              <div>
                <strong>Data points:</strong> Approximately{" "}
                {Math.min(1000, Math.max(100, selectedRange.hours * 60))}{" "}
                samples
              </div>
              <div>
                <strong>Resolution:</strong>{" "}
                {selectedRange.updateInterval < 1000
                  ? "Sub-second precision"
                  : selectedRange.updateInterval < 60000
                  ? "Second-level precision"
                  : "Minute-level precision"}
              </div>
              <div>
                <strong>Best for:</strong>{" "}
                {selectedRange.hours < 0.1
                  ? "Real-time monitoring and immediate feedback"
                  : selectedRange.hours < 1
                  ? "Troubleshooting and immediate analysis"
                  : selectedRange.hours < 24
                  ? "Performance analysis and trend identification"
                  : "Long-term pattern analysis and capacity planning"}
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export { TimeRangeSelector, TIME_RANGES };
export type { TimeRange };
