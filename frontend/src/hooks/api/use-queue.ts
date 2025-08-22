import { useQuery, useMutation, useQueryClient } from "react-query";
import { get, post, del } from "@/lib/api-client";
import { 
  QueueStatus, 
  QueueStatusSchema,
  TaskInfo,
  validateApiResponse 
} from "@/lib/api-schemas";
import { ApiError } from "@/lib/api-client";

// Hook for fetching queue status
export const useQueueStatus = (options?: {
  refetchInterval?: number;
  enabled?: boolean;
}) => {
  return useQuery<QueueStatus, ApiError>({
    queryKey: ["queue"],
    queryFn: async () => {
      const response = await get<QueueStatus>("/queue");
      return validateApiResponse(QueueStatusSchema, response);
    },
    refetchInterval: options?.refetchInterval || 5000, // Poll every 5 seconds
    enabled: options?.enabled !== false,
    staleTime: 0, // Always consider data stale for real-time updates
    cacheTime: 1000 * 60, // Keep in cache for 1 minute
  });
};

// Hook for polling queue updates (optimized endpoint)
export const useQueuePolling = (options?: {
  enabled?: boolean;
  activeOnly?: boolean;
}) => {
  return useQuery({
    queryKey: ["queue-poll", options?.activeOnly],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (options?.activeOnly) {
        params.append("active_only", "true");
      }
      
      const response = await get(`/queue/poll?${params.toString()}`);
      return response;
    },
    refetchInterval: 5000, // Poll every 5 seconds
    enabled: options?.enabled !== false,
    staleTime: 0,
    cacheTime: 30000, // Keep in cache for 30 seconds
  });
};

// Hook for cancelling a task
export const useCancelTask = () => {
  const queryClient = useQueryClient();

  return useMutation<{ message: string }, ApiError, string>({
    mutationFn: async (taskId: string) => {
      const response = await post<{ message: string }>(`/queue/${taskId}/cancel`, {});
      return response;
    },
    onSuccess: () => {
      // Invalidate queue queries to refresh the queue
      queryClient.invalidateQueries(["queue"]);
      queryClient.invalidateQueries(["queue-poll"]);
    },
    onError: (error: ApiError) => {
      console.error("Failed to cancel task:", error);
    },
  });
};

// Hook for deleting a task
export const useDeleteTask = () => {
  const queryClient = useQueryClient();

  return useMutation<{ message: string }, ApiError, string>({
    mutationFn: async (taskId: string) => {
      const response = await del<{ message: string }>(`/queue/${taskId}`);
      return response;
    },
    onSuccess: () => {
      // Invalidate queue queries to refresh the queue
      queryClient.invalidateQueries(["queue"]);
      queryClient.invalidateQueries(["queue-poll"]);
    },
    onError: (error: ApiError) => {
      console.error("Failed to delete task:", error);
    },
  });
};

// Hook for clearing completed tasks
export const useClearCompletedTasks = () => {
  const queryClient = useQueryClient();

  return useMutation<{ message: string }, ApiError, void>({
    mutationFn: async () => {
      const response = await post<{ message: string }>("/queue/clear", {});
      return response;
    },
    onSuccess: () => {
      // Invalidate queue queries to refresh the queue
      queryClient.invalidateQueries(["queue"]);
      queryClient.invalidateQueries(["queue-poll"]);
    },
    onError: (error: ApiError) => {
      console.error("Failed to clear completed tasks:", error);
    },
  });
};

// Hook for reordering tasks
export const useReorderTasks = () => {
  const queryClient = useQueryClient();

  return useMutation<{ message: string }, ApiError, string[]>({
    mutationFn: async (taskIds: string[]) => {
      const response = await post<{ message: string }>("/queue/reorder", {
        task_ids: taskIds
      });
      return response;
    },
    onSuccess: () => {
      // Invalidate queue queries to refresh the queue
      queryClient.invalidateQueries(["queue"]);
      queryClient.invalidateQueries(["queue-poll"]);
    },
    onError: (error: ApiError) => {
      console.error("Failed to reorder tasks:", error);
    },
  });
};

// Hook for getting estimated completion time
export const useEstimatedCompletionTime = (task: TaskInfo) => {
  const calculateEstimatedCompletion = () => {
    if (task.status === "completed" || task.status === "failed" || task.status === "cancelled") {
      return null;
    }

    if (task.status === "processing" && task.progress > 0) {
      // Calculate based on current progress
      const elapsedTime = task.started_at 
        ? (Date.now() - new Date(task.started_at).getTime()) / (1000 * 60) // minutes
        : 0;
      
      const estimatedTotal = elapsedTime / (task.progress / 100);
      const remaining = estimatedTotal - elapsedTime;
      
      return Math.max(0, Math.round(remaining));
    }

    // Use the estimated time from the task
    return task.estimated_time_minutes || null;
  };

  return calculateEstimatedCompletion();
};

// Hook for task statistics
export const useTaskStatistics = (queueStatus?: QueueStatus) => {
  if (!queueStatus) {
    return {
      totalTasks: 0,
      activeTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      completionRate: 0,
    };
  }

  const activeTasks = queueStatus.pending_tasks + queueStatus.processing_tasks;
  const completionRate = queueStatus.total_tasks > 0 
    ? (queueStatus.completed_tasks / queueStatus.total_tasks) * 100 
    : 0;

  return {
    totalTasks: queueStatus.total_tasks,
    activeTasks,
    completedTasks: queueStatus.completed_tasks,
    failedTasks: queueStatus.failed_tasks,
    completionRate: Math.round(completionRate),
  };
};