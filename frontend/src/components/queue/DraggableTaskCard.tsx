import React from "react";
import { motion } from "framer-motion";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { TaskInfo } from "@/lib/api-schemas";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Clock,
  Play,
  CheckCircle,
  XCircle,
  Pause,
  GripVertical,
  X,
  RotateCcw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";
import { useCancelTask } from "@/hooks/api/use-queue";
import { useToast } from "@/hooks/use-toast";

interface DraggableTaskCardProps {
  task: TaskInfo;
  isDragMode?: boolean;
}

const DraggableTaskCard: React.FC<DraggableTaskCardProps> = ({
  task,
  isDragMode = false,
}) => {
  const { toast } = useToast();
  const cancelTask = useCancelTask();

  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({
    id: task.id,
    disabled: !isDragMode,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  const getStatusIcon = () => {
    switch (task.status) {
      case "pending":
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case "processing":
        return <Play className="h-4 w-4 text-blue-500" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "cancelled":
        return <Pause className="h-4 w-4 text-gray-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (task.status) {
      case "pending":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "processing":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "completed":
        return "bg-green-100 text-green-800 border-green-200";
      case "failed":
        return "bg-red-100 text-red-800 border-red-200";
      case "cancelled":
        return "bg-gray-100 text-gray-800 border-gray-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const handleCancel = async () => {
    if (task.status !== "pending" && task.status !== "processing") return;

    try {
      await cancelTask.mutateAsync(task.id);
      toast({
        title: "Task Cancelled",
        description: "The task has been cancelled successfully.",
      });
    } catch (error) {
      toast({
        title: "Cancel Failed",
        description: "Failed to cancel the task. Please try again.",
        variant: "destructive",
      });
    }
  };

  const canCancel = task.status === "pending" || task.status === "processing";

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      {...attributes}
      className={cn(
        "relative",
        isDragging && "z-50",
        isDragMode && "cursor-grab active:cursor-grabbing"
      )}
      whileHover={!isDragging ? { scale: 1.02 } : undefined}
      whileTap={!isDragging ? { scale: 0.98 } : undefined}
    >
      <Card
        className={cn(
          "transition-all duration-200",
          isDragging && "shadow-lg rotate-2",
          isDragMode &&
            task.status === "pending" &&
            "border-blue-200 bg-blue-50/30"
        )}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              {/* Drag Handle */}
              {isDragMode && task.status === "pending" && (
                <div
                  {...listeners}
                  className="flex items-center justify-center w-6 h-6 text-gray-400 hover:text-gray-600 cursor-grab active:cursor-grabbing"
                >
                  <GripVertical className="h-4 w-4" />
                </div>
              )}

              <div className="flex-1 min-w-0">
                <CardTitle className="text-sm font-medium truncate">
                  {task.model_type}
                </CardTitle>
                <div className="flex items-center gap-2 mt-1">
                  {getStatusIcon()}
                  <Badge
                    variant="outline"
                    className={cn("text-xs", getStatusColor())}
                  >
                    {task.status}
                  </Badge>
                </div>
              </div>
            </div>

            {/* Cancel Button */}
            {canCancel && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCancel}
                disabled={cancelTask.isLoading}
                className="h-6 w-6 p-0 text-gray-400 hover:text-red-500"
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          <div className="space-y-3">
            {/* Prompt */}
            {task.prompt && (
              <div>
                <p className="text-xs text-muted-foreground line-clamp-2">
                  {task.prompt}
                </p>
              </div>
            )}

            {/* Progress */}
            {task.status === "processing" && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span>Progress</span>
                  <span>{task.progress}%</span>
                </div>
                <Progress value={task.progress} className="h-2" />
              </div>
            )}

            {/* Metadata */}
            <div className="space-y-1 text-xs text-muted-foreground">
              {task.resolution && (
                <div className="flex justify-between">
                  <span>Resolution:</span>
                  <span>{task.resolution}</span>
                </div>
              )}
              {task.steps && (
                <div className="flex justify-between">
                  <span>Steps:</span>
                  <span>{task.steps}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span>Created:</span>
                <span>
                  {formatDistanceToNow(new Date(task.created_at))} ago
                </span>
              </div>
              {task.started_at && (
                <div className="flex justify-between">
                  <span>Started:</span>
                  <span>
                    {formatDistanceToNow(new Date(task.started_at))} ago
                  </span>
                </div>
              )}
              {task.completed_at && (
                <div className="flex justify-between">
                  <span>Completed:</span>
                  <span>
                    {formatDistanceToNow(new Date(task.completed_at))} ago
                  </span>
                </div>
              )}
            </div>

            {/* Error Message */}
            {task.status === "failed" && task.error_message && (
              <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                <div className="font-medium mb-1">Error:</div>
                <div className="line-clamp-2">{task.error_message}</div>
              </div>
            )}

            {/* Estimated Time */}
            {task.status === "processing" && task.estimated_completion && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                <span>
                  Est. completion:{" "}
                  {formatDistanceToNow(new Date(task.estimated_completion))}
                </span>
              </div>
            )}

            {/* Output Path */}
            {task.status === "completed" && task.output_path && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  Output ready
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    // Open output file or navigate to gallery
                    window.open(task.output_path, "_blank");
                  }}
                  className="h-6 px-2 text-xs"
                >
                  View
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Drag Overlay Effect */}
      {isDragging && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute inset-0 bg-blue-100/50 border-2 border-blue-300 border-dashed rounded-lg pointer-events-none"
        />
      )}
    </motion.div>
  );
};

export default DraggableTaskCard;
