import React from "react";
import { QueueStatus } from "@/lib/api-schemas";
import { useTaskStatistics } from "@/hooks/api/use-queue";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Clock,
  Play,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Activity,
} from "lucide-react";

interface QueueStatsProps {
  queueStatus?: QueueStatus;
  isLoading?: boolean;
}

const QueueStats: React.FC<QueueStatsProps> = ({ queueStatus, isLoading }) => {
  const stats = useTaskStatistics(queueStatus);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader className="pb-2">
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            </CardHeader>
            <CardContent>
              <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-full"></div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const statCards = [
    {
      title: "Total Tasks",
      value: stats.totalTasks,
      icon: <BarChart3 className="h-4 w-4" />,
      color: "text-blue-600",
      bgColor: "bg-blue-50",
    },
    {
      title: "Active Tasks",
      value: stats.activeTasks,
      icon: <Activity className="h-4 w-4" />,
      color: "text-orange-600",
      bgColor: "bg-orange-50",
    },
    {
      title: "Completed",
      value: stats.completedTasks,
      icon: <CheckCircle className="h-4 w-4" />,
      color: "text-green-600",
      bgColor: "bg-green-50",
    },
    {
      title: "Failed",
      value: stats.failedTasks,
      icon: <AlertCircle className="h-4 w-4" />,
      color: "text-red-600",
      bgColor: "bg-red-50",
    },
  ];

  return (
    <div className="space-y-4">
      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {stat.title}
              </CardTitle>
              <div className={`p-2 rounded-full ${stat.bgColor} ${stat.color}`}>
                {stat.icon}
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Detailed Status Breakdown */}
      {queueStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Queue Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Completion Rate */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">Completion Rate</span>
                  <span>{stats.completionRate}%</span>
                </div>
                <Progress value={stats.completionRate} className="h-2" />
              </div>

              {/* Status Breakdown */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="flex items-center space-x-2">
                  <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>Pending</span>
                  </Badge>
                  <span className="text-sm font-medium">
                    {queueStatus.pending_tasks}
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <Badge className="bg-blue-100 text-blue-800 border-blue-200 flex items-center space-x-1">
                    <Play className="h-3 w-3" />
                    <span>Processing</span>
                  </Badge>
                  <span className="text-sm font-medium">
                    {queueStatus.processing_tasks}
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <Badge className="bg-green-100 text-green-800 border-green-200 flex items-center space-x-1">
                    <CheckCircle className="h-3 w-3" />
                    <span>Completed</span>
                  </Badge>
                  <span className="text-sm font-medium">
                    {queueStatus.completed_tasks}
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <Badge className="bg-red-100 text-red-800 border-red-200 flex items-center space-x-1">
                    <AlertCircle className="h-3 w-3" />
                    <span>Failed</span>
                  </Badge>
                  <span className="text-sm font-medium">
                    {queueStatus.failed_tasks}
                  </span>
                </div>

                {queueStatus.cancelled_tasks > 0 && (
                  <div className="flex items-center space-x-2">
                    <Badge className="bg-gray-100 text-gray-800 border-gray-200 flex items-center space-x-1">
                      <span>Cancelled</span>
                    </Badge>
                    <span className="text-sm font-medium">
                      {queueStatus.cancelled_tasks}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default QueueStats;
