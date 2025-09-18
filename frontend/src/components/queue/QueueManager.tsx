import React, { useState, useEffect } from "react";
import { useQueueStatus, useClearCompletedTasks } from "@/hooks/api/use-queue";
import { TaskInfo } from "@/lib/api-schemas";
import TaskCard from "./TaskCard";
import QueueStats from "./QueueStats";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Trash2,
  RefreshCw,
  Filter,
  SortAsc,
  SortDesc,
  Bell,
  BellOff,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useWebSocket } from "@/hooks/use-websocket";
import { cn } from "@/lib/utils";

type SortOption = "created_at" | "status" | "progress";
type SortDirection = "asc" | "desc";
type FilterOption =
  | "all"
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

const QueueManager: React.FC = () => {
  const { toast } = useToast();
  const [sortBy, setSortBy] = useState<SortOption>("created_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [filterBy, setFilterBy] = useState<FilterOption>("all");
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [lastCompletedCount, setLastCompletedCount] = useState(0);

  const { data: queueStatus, isLoading, error, refetch } = useQueueStatus();
  const clearCompleted = useClearCompletedTasks();

  // WebSocket for real-time updates
  const { isConnected: wsConnected, lastMessage } = useWebSocket(
    `ws://${
      import.meta.env.VITE_API_URL?.replace("http://", "") || "localhost:8000"
    }/ws`
  );

  // Request notification permission
  useEffect(() => {
    if ("Notification" in window) {
      if (Notification.permission === "granted") {
        setNotificationsEnabled(true);
      } else if (Notification.permission === "default") {
        Notification.requestPermission().then((permission) => {
          setNotificationsEnabled(permission === "granted");
        });
      }
    }
  }, []);

  // Handle WebSocket messages for real-time updates
  useEffect(() => {
    if (lastMessage) {
      const message = lastMessage;

      if (message.type === "task_update") {
        // Refresh queue when task updates are received
        refetch();

        // Show notification for completed tasks
        if (
          message.data?.status === "completed" &&
          notificationsEnabled &&
          "Notification" in window
        ) {
          new Notification("Video Generation Complete", {
            body: `Your video has finished generating!`,
            icon: "/favicon.ico",
          });
        }

        // Show toast for task updates
        if (message.data?.message) {
          toast({
            title: "Task Update",
            description: message.data.message,
            variant:
              message.data.status === "failed" ? "destructive" : "default",
          });
        }
      } else if (message.type === "queue_update") {
        // Refresh queue when queue updates are received
        refetch();
      }
    }
  }, [lastMessage, refetch, notificationsEnabled, toast]);

  // Show completion notifications
  useEffect(() => {
    if (queueStatus && notificationsEnabled && "Notification" in window) {
      const currentCompletedCount = queueStatus.completed_tasks;

      if (
        lastCompletedCount > 0 &&
        currentCompletedCount > lastCompletedCount
      ) {
        const newCompletions = currentCompletedCount - lastCompletedCount;

        new Notification("Video Generation Complete", {
          body: `${newCompletions} video${
            newCompletions > 1 ? "s" : ""
          } finished generating!`,
          icon: "/favicon.ico",
        });
      }

      setLastCompletedCount(currentCompletedCount);
    }
  }, [queueStatus?.completed_tasks, notificationsEnabled, lastCompletedCount]);

  const handleClearCompleted = async () => {
    try {
      await clearCompleted.mutateAsync();
      toast({
        title: "Tasks Cleared",
        description: "All completed tasks have been removed from the queue.",
      });
    } catch (error) {
      toast({
        title: "Clear Failed",
        description: "Failed to clear completed tasks. Please try again.",
        variant: "destructive",
      });
    }
  };

  const toggleNotifications = () => {
    if ("Notification" in window) {
      if (Notification.permission === "granted") {
        setNotificationsEnabled(!notificationsEnabled);
      } else {
        Notification.requestPermission().then((permission) => {
          setNotificationsEnabled(permission === "granted");
        });
      }
    }
  };

  const sortTasks = (tasks: TaskInfo[]): TaskInfo[] => {
    return [...tasks].sort((a, b) => {
      let aValue: any;
      let bValue: any;

      switch (sortBy) {
        case "created_at":
          aValue = new Date(a.created_at).getTime();
          bValue = new Date(b.created_at).getTime();
          break;
        case "status":
          // Custom status order: processing, pending, completed, failed, cancelled
          const statusOrder = {
            processing: 0,
            pending: 1,
            completed: 2,
            failed: 3,
            cancelled: 4,
          };
          aValue = statusOrder[a.status] ?? 5;
          bValue = statusOrder[b.status] ?? 5;
          break;
        case "progress":
          aValue = a.progress;
          bValue = b.progress;
          break;
        default:
          return 0;
      }

      if (sortDirection === "asc") {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });
  };

  const filterTasks = (tasks: TaskInfo[]): TaskInfo[] => {
    if (filterBy === "all") return tasks;
    return tasks.filter((task) => task.status === filterBy);
  };

  const getFilteredAndSortedTasks = (): TaskInfo[] => {
    if (!queueStatus?.tasks) return [];
    return sortTasks(filterTasks(queueStatus.tasks));
  };

  const filteredTasks = getFilteredAndSortedTasks();
  const hasCompletedTasks =
    queueStatus?.completed_tasks > 0 || queueStatus?.failed_tasks > 0;

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center space-y-4">
            <div className="text-red-600">
              Failed to load queue status. Please try again.
            </div>
            <Button onClick={() => refetch()} variant="outline">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Queue Statistics */}
      <QueueStats queueStatus={queueStatus} isLoading={isLoading} />

      {/* Controls */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
            <CardTitle className="text-lg">Task Queue</CardTitle>

            <div className="flex flex-wrap items-center gap-2">
              {/* WebSocket Connection Status */}
              <Badge
                variant={wsConnected ? "default" : "destructive"}
                className="flex items-center gap-1"
              >
                <div
                  className={cn(
                    "w-2 h-2 rounded-full",
                    wsConnected ? "bg-green-500" : "bg-red-500"
                  )}
                />
                {wsConnected ? "Live Updates" : "Offline"}
              </Badge>

              {/* Notifications Toggle */}
              <Button
                variant="outline"
                size="sm"
                onClick={toggleNotifications}
                className={cn(
                  "flex items-center space-x-2",
                  notificationsEnabled && "bg-blue-50 border-blue-200"
                )}
              >
                {notificationsEnabled ? (
                  <Bell className="h-4 w-4" />
                ) : (
                  <BellOff className="h-4 w-4" />
                )}
                <span>Notifications</span>
              </Button>

              {/* Clear Completed */}
              {hasCompletedTasks && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleClearCompleted}
                  disabled={clearCompleted.isLoading}
                  className="flex items-center space-x-2"
                >
                  <Trash2 className="h-4 w-4" />
                  <span>Clear Completed</span>
                </Button>
              )}

              {/* Refresh */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetch()}
                disabled={isLoading}
                className="flex items-center space-x-2"
              >
                <RefreshCw
                  className={cn("h-4 w-4", isLoading && "animate-spin")}
                />
                <span>Refresh</span>
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {/* Filters and Sorting */}
          <div className="flex flex-wrap items-center gap-4 mb-6">
            {/* Filter */}
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Filter:</span>
              <div className="flex flex-wrap gap-1">
                {(
                  [
                    "all",
                    "pending",
                    "processing",
                    "completed",
                    "failed",
                    "cancelled",
                  ] as FilterOption[]
                ).map((filter) => (
                  <Badge
                    key={filter}
                    variant={filterBy === filter ? "default" : "outline"}
                    className="cursor-pointer capitalize"
                    onClick={() => setFilterBy(filter)}
                  >
                    {filter}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Sort */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Sort:</span>
              <div className="flex items-center space-x-1">
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortOption)}
                  className="text-sm border rounded px-2 py-1"
                >
                  <option value="created_at">Created</option>
                  <option value="status">Status</option>
                  <option value="progress">Progress</option>
                </select>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setSortDirection(sortDirection === "asc" ? "desc" : "asc")
                  }
                  className="p-1"
                >
                  {sortDirection === "asc" ? (
                    <SortAsc className="h-4 w-4" />
                  ) : (
                    <SortDesc className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* Task List */}
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[...Array(6)].map((_, i) => (
                <Card key={i} className="animate-pulse">
                  <CardHeader>
                    <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="h-3 bg-gray-200 rounded"></div>
                      <div className="h-3 bg-gray-200 rounded w-2/3"></div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : filteredTasks.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-muted-foreground">
                {filterBy === "all"
                  ? "No tasks in queue. Start generating videos to see them here!"
                  : `No ${filterBy} tasks found.`}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTasks.map((task) => (
                <TaskCard key={task.id} task={task} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default QueueManager;
