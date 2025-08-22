import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import {
  restrictToVerticalAxis,
  restrictToWindowEdges,
} from "@dnd-kit/modifiers";
import {
  useQueueStatus,
  useClearCompletedTasks,
  useReorderTasks,
} from "@/hooks/api/use-queue";
import { TaskInfo } from "@/lib/api-schemas";
import DraggableTaskCard from "./DraggableTaskCard";
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
  GripVertical,
  Move,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

type SortOption = "created_at" | "status" | "progress" | "manual";
type SortDirection = "asc" | "desc";
type FilterOption =
  | "all"
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

const DraggableQueueManager: React.FC = () => {
  const { toast } = useToast();
  const [sortBy, setSortBy] = useState<SortOption>("created_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [filterBy, setFilterBy] = useState<FilterOption>("all");
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [lastCompletedCount, setLastCompletedCount] = useState(0);
  const [isDragMode, setIsDragMode] = useState(false);
  const [localTasks, setLocalTasks] = useState<TaskInfo[]>([]);

  const { data: queueStatus, isLoading, error, refetch } = useQueueStatus();
  const clearCompleted = useClearCompletedTasks();
  const reorderTasks = useReorderTasks();

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Update local tasks when queue status changes
  useEffect(() => {
    if (queueStatus?.tasks) {
      setLocalTasks(queueStatus.tasks);
    }
  }, [queueStatus?.tasks]);

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

  const handleDragEnd = async (event: DragEndEvent) => {
    const { active, over } = event;

    if (!over || active.id === over.id) {
      return;
    }

    const oldIndex = localTasks.findIndex((task) => task.id === active.id);
    const newIndex = localTasks.findIndex((task) => task.id === over.id);

    if (oldIndex !== -1 && newIndex !== -1) {
      const newTasks = arrayMove(localTasks, oldIndex, newIndex);
      setLocalTasks(newTasks);

      // Only reorder pending tasks
      const pendingTasks = newTasks.filter((task) => task.status === "pending");
      const taskIds = pendingTasks.map((task) => task.id);

      try {
        await reorderTasks.mutateAsync(taskIds);
        toast({
          title: "Queue Reordered",
          description: "Task order has been updated successfully.",
        });
      } catch (error) {
        // Revert on error
        setLocalTasks(queueStatus?.tasks || []);
        toast({
          title: "Reorder Failed",
          description: "Failed to reorder tasks. Please try again.",
          variant: "destructive",
        });
      }
    }
  };

  const sortTasks = (tasks: TaskInfo[]): TaskInfo[] => {
    if (sortBy === "manual") {
      return tasks; // Keep manual order
    }

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
    if (!localTasks.length) return [];
    return sortTasks(filterTasks(localTasks));
  };

  const filteredTasks = getFilteredAndSortedTasks();
  const hasCompletedTasks =
    queueStatus?.completed_tasks > 0 || queueStatus?.failed_tasks > 0;
  const canReorder = sortBy === "manual" && filterBy === "pending";

  const toggleDragMode = () => {
    if (!isDragMode) {
      setSortBy("manual");
      setFilterBy("pending");
    }
    setIsDragMode(!isDragMode);
  };

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
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
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Queue Statistics */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <QueueStats queueStatus={queueStatus} isLoading={isLoading} />
      </motion.div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
              <CardTitle className="text-lg">Task Queue</CardTitle>

              <div className="flex flex-wrap items-center gap-2">
                {/* Drag Mode Toggle */}
                <Button
                  variant={isDragMode ? "default" : "outline"}
                  size="sm"
                  onClick={toggleDragMode}
                  className={cn(
                    "flex items-center space-x-2",
                    isDragMode && "bg-blue-50 border-blue-200"
                  )}
                >
                  {isDragMode ? (
                    <GripVertical className="h-4 w-4" />
                  ) : (
                    <Move className="h-4 w-4" />
                  )}
                  <span>{isDragMode ? "Drag Mode" : "Reorder"}</span>
                </Button>

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
            {/* Drag Mode Info */}
            <AnimatePresence>
              {isDragMode && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg"
                >
                  <div className="flex items-center gap-2 text-blue-800">
                    <GripVertical className="h-4 w-4" />
                    <span className="font-medium">Drag Mode Active</span>
                  </div>
                  <p className="text-sm text-blue-600 mt-1">
                    Drag and drop pending tasks to reorder them. Only pending
                    tasks can be reordered.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Filters and Sorting */}
            {!isDragMode && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-wrap items-center gap-4 mb-6"
              >
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
                      <option value="manual">Manual Order</option>
                    </select>
                    {sortBy !== "manual" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setSortDirection(
                            sortDirection === "asc" ? "desc" : "asc"
                          )
                        }
                        className="p-1"
                      >
                        {sortDirection === "asc" ? (
                          <SortAsc className="h-4 w-4" />
                        ) : (
                          <SortDesc className="h-4 w-4" />
                        )}
                      </Button>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Task List */}
            {isLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {[...Array(6)].map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                  >
                    <Card className="animate-pulse">
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
                  </motion.div>
                ))}
              </div>
            ) : filteredTasks.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center py-12"
              >
                <div className="text-muted-foreground">
                  {filterBy === "all"
                    ? "No tasks in queue. Start generating videos to see them here!"
                    : `No ${filterBy} tasks found.`}
                </div>
              </motion.div>
            ) : (
              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={handleDragEnd}
                modifiers={[restrictToVerticalAxis, restrictToWindowEdges]}
              >
                <SortableContext
                  items={filteredTasks.map((task) => task.id)}
                  strategy={verticalListSortingStrategy}
                >
                  <motion.div
                    layout
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
                  >
                    <AnimatePresence mode="popLayout">
                      {filteredTasks.map((task, index) => (
                        <motion.div
                          key={task.id}
                          layout
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          transition={{
                            duration: 0.2,
                            delay: index * 0.05,
                          }}
                        >
                          <DraggableTaskCard
                            task={task}
                            isDragMode={isDragMode && task.status === "pending"}
                          />
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </motion.div>
                </SortableContext>
              </DndContext>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default DraggableQueueManager;
