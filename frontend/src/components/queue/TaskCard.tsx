import React from "react";
import { TaskInfo } from "@/lib/api-schemas";
import {
  useCancelTask,
  useDeleteTask,
  useEstimatedCompletionTime,
} from "@/hooks/api/use-queue";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  AlertCircle,
  CheckCircle,
  Clock,
  Play,
  X,
  Trash2,
  Image,
  Video,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

interface TaskCardProps {
  task: TaskInfo;
  className?: string;
}

const TaskCard: React.FC<TaskCardProps> = ({ task, className }) => {
  const { toast } = useToast();
  const cancelTask = useCancelTask();
  const deleteTask = useDeleteTask();
  const estimatedTime = useEstimatedCompletionTime(task);

  const handleCancel = async () => {
    if (!cancelTask?.mutateAsync) return;

    try {
      await cancelTask.mutateAsync(task.id);
      toast({
        title: "Task Cancelled",
        description: "The generation task has been cancelled successfully.",
      });
    } catch (error) {
      toast({
        title: "Cancellation Failed",
        description: "Failed to cancel the task. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDelete = async () => {
    if (!deleteTask?.mutateAsync) return;

    try {
      await deleteTask.mutateAsync(task.id);
      toast({
        title: "Task Deleted",
        description: "The task has been removed from the queue.",
      });
    } catch (error) {
      toast({
        title: "Deletion Failed",
        description: "Failed to delete the task. Please try again.",
        variant: "destructive",
      });
    }
  };

  const getStatusIcon = () => {
    switch (task.status) {
      case "pending":
        return <Clock className="h-4 w-4" />;
      case "processing":
        return <Play className="h-4 w-4" />;
      case "completed":
        return <CheckCircle className="h-4 w-4" />;
      case "failed":
        return <AlertCircle className="h-4 w-4" />;
      case "cancelled":
        return <X className="h-4 w-4" />;
      default:
        return <Clock className="h-4 w-4" />;
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

  const formatTime = (dateString?: string) => {
    if (!dateString) return null;
    return new Date(dateString).toLocaleTimeString();
  };

  const formatDuration = (minutes?: number) => {
    if (!minutes) return null;
    if (minutes < 60) {
      return `${Math.round(minutes)}m`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  const canCancel = task.status === "pending" || task.status === "processing";
  const canDelete =
    task.status === "completed" ||
    task.status === "failed" ||
    task.status === "cancelled";

  return (
    <Card
      className={cn("transition-all duration-200 hover:shadow-md", className)}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-2">
            <Badge
              className={cn("flex items-center space-x-1", getStatusColor())}
            >
              {getStatusIcon()}
              <span className="capitalize">{task.status}</span>
            </Badge>
            <Badge variant="outline" className="flex items-center space-x-1">
              {task.image_path ? (
                <Image className="h-3 w-3" />
              ) : (
                <Video className="h-3 w-3" />
              )}
              <span>{task.model_type}</span>
            </Badge>
          </div>

          <div className="flex items-center space-x-1">
            {canCancel && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCancel}
                disabled={cancelTask?.isLoading || false}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
            {canDelete && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDelete}
                disabled={deleteTask?.isLoading || false}
                className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        <CardTitle className="text-sm font-medium line-clamp-2">
          {task.prompt}
        </CardTitle>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="space-y-3">
          {/* Progress Bar */}
          {task.status === "processing" && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Progress</span>
                <span>{task.progress}%</span>
              </div>
              <Progress value={task.progress} className="h-2" />
            </div>
          )}

          {/* Task Details */}
          <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
            <div>
              <span className="font-medium">Resolution:</span> {task.resolution}
            </div>
            <div>
              <span className="font-medium">Steps:</span> {task.steps}
            </div>
            {task.lora_path && (
              <div className="col-span-2">
                <span className="font-medium">LoRA:</span> {task.lora_path} (
                {task.lora_strength}x)
              </div>
            )}
          </div>

          {/* Timing Information */}
          <div className="space-y-1 text-xs text-muted-foreground">
            <div className="flex justify-between">
              <span>Created:</span>
              <span>{formatTime(task.created_at)}</span>
            </div>

            {task.started_at && (
              <div className="flex justify-between">
                <span>Started:</span>
                <span>{formatTime(task.started_at)}</span>
              </div>
            )}

            {task.completed_at && (
              <div className="flex justify-between">
                <span>Completed:</span>
                <span>{formatTime(task.completed_at)}</span>
              </div>
            )}

            {estimatedTime && task.status !== "completed" && (
              <div className="flex justify-between">
                <span>Est. Time:</span>
                <span>{formatDuration(estimatedTime)}</span>
              </div>
            )}
          </div>

          {/* Error Message */}
          {task.error_message && (
            <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
              <div className="font-medium">Error:</div>
              <div>{task.error_message}</div>
            </div>
          )}

          {/* Output Path */}
          {task.output_path && task.status === "completed" && (
            <div className="p-2 bg-green-50 border border-green-200 rounded text-xs text-green-700">
              <div className="font-medium">Output:</div>
              <div className="truncate">{task.output_path}</div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TaskCard;
