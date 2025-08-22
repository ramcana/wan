import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import QueueManager from "../QueueManager";
import { QueueStatus, TaskInfo } from "@/lib/api-schemas";

// Mock the API hooks
vi.mock("@/hooks/api/use-queue", () => ({
  useQueueStatus: vi.fn(),
  useClearCompletedTasks: vi.fn(),
  useCancelTask: vi.fn(),
  useDeleteTask: vi.fn(),
  useEstimatedCompletionTime: vi.fn(),
  useTaskStatistics: vi.fn(),
}));

// Mock the toast hook
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

// Mock Notification API
Object.defineProperty(window, "Notification", {
  value: {
    permission: "granted",
    requestPermission: vi.fn().mockResolvedValue("granted"),
  },
  writable: true,
});

const mockQueueStatus: QueueStatus = {
  total_tasks: 5,
  pending_tasks: 2,
  processing_tasks: 1,
  completed_tasks: 2,
  failed_tasks: 0,
  cancelled_tasks: 0,
  tasks: [
    {
      id: "task-1",
      model_type: "T2V-A14B",
      prompt: "A beautiful sunset over mountains",
      resolution: "1280x720",
      steps: 50,
      status: "processing",
      progress: 75,
      created_at: "2024-01-01T10:00:00Z",
      started_at: "2024-01-01T10:01:00Z",
      lora_strength: 1.0,
    },
    {
      id: "task-2",
      model_type: "I2V-A14B",
      prompt: "Ocean waves crashing",
      image_path: "/uploads/image.jpg",
      resolution: "1280x720",
      steps: 50,
      status: "pending",
      progress: 0,
      created_at: "2024-01-01T10:05:00Z",
      lora_strength: 1.0,
    },
    {
      id: "task-3",
      model_type: "TI2V-5B",
      prompt: "City skyline at night",
      resolution: "1920x1080",
      steps: 75,
      status: "completed",
      progress: 100,
      created_at: "2024-01-01T09:30:00Z",
      started_at: "2024-01-01T09:31:00Z",
      completed_at: "2024-01-01T09:45:00Z",
      output_path: "/outputs/task-3.mp4",
      lora_strength: 1.0,
    },
  ] as TaskInfo[],
};

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
};

describe("QueueManager", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render queue statistics correctly", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    (useQueueStatus as any).mockReturnValue({
      data: mockQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 5,
      activeTasks: 3,
      completedTasks: 2,
      failedTasks: 0,
      completionRate: 40,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Check if statistics are displayed
    expect(screen.getByText("5")).toBeInTheDocument(); // Total tasks
    expect(screen.getByText("3")).toBeInTheDocument(); // Active tasks
    expect(screen.getByText("2")).toBeInTheDocument(); // Completed tasks
  });

  it("should display task cards with correct information", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    (useQueueStatus as any).mockReturnValue({
      data: mockQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 5,
      activeTasks: 3,
      completedTasks: 2,
      failedTasks: 0,
      completionRate: 40,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Check if task prompts are displayed
    expect(
      screen.getByText("A beautiful sunset over mountains")
    ).toBeInTheDocument();
    expect(screen.getByText("Ocean waves crashing")).toBeInTheDocument();
    expect(screen.getByText("City skyline at night")).toBeInTheDocument();

    // Check if status badges are displayed
    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });

  it("should show progress bar for processing tasks", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    (useQueueStatus as any).mockReturnValue({
      data: mockQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 5,
      activeTasks: 3,
      completedTasks: 2,
      failedTasks: 0,
      completionRate: 40,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Check if progress percentage is displayed
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("should handle task filtering", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    (useQueueStatus as any).mockReturnValue({
      data: mockQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 5,
      activeTasks: 3,
      completedTasks: 2,
      failedTasks: 0,
      completionRate: 40,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Click on "completed" filter
    const completedFilter = screen.getByText("completed");
    fireEvent.click(completedFilter);

    // Should only show completed tasks
    expect(screen.getByText("City skyline at night")).toBeInTheDocument();
    expect(
      screen.queryByText("A beautiful sunset over mountains")
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Ocean waves crashing")).not.toBeInTheDocument();
  });

  it("should handle notifications toggle", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    (useQueueStatus as any).mockReturnValue({
      data: mockQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 5,
      activeTasks: 3,
      completedTasks: 2,
      failedTasks: 0,
      completionRate: 40,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Find and click notifications button
    const notificationsButton = screen.getByText("Notifications");
    expect(notificationsButton).toBeInTheDocument();

    fireEvent.click(notificationsButton);
    // Should toggle notification state
  });

  it("should show loading state", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    (useQueueStatus as any).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 0,
      activeTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      completionRate: 0,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Should show loading skeleton
    expect(document.querySelectorAll(".animate-pulse")).toHaveLength(10); // 4 stat cards + 6 task cards
  });

  it("should show error state and retry button", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    const mockRefetch = vi.fn();

    (useQueueStatus as any).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Network error"),
      refetch: mockRefetch,
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 0,
      activeTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      completionRate: 0,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Should show error message
    expect(
      screen.getByText("Failed to load queue status. Please try again.")
    ).toBeInTheDocument();

    // Should show retry button
    const retryButton = screen.getByText("Retry");
    expect(retryButton).toBeInTheDocument();

    // Click retry button
    fireEvent.click(retryButton);
    expect(mockRefetch).toHaveBeenCalled();
  });

  it("should show empty state when no tasks", async () => {
    const { useQueueStatus, useTaskStatistics, useClearCompletedTasks } =
      await import("@/hooks/api/use-queue");

    const emptyQueueStatus = {
      ...mockQueueStatus,
      total_tasks: 0,
      pending_tasks: 0,
      processing_tasks: 0,
      completed_tasks: 0,
      failed_tasks: 0,
      cancelled_tasks: 0,
      tasks: [],
    };

    (useQueueStatus as any).mockReturnValue({
      data: emptyQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    (useTaskStatistics as any).mockReturnValue({
      totalTasks: 0,
      activeTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      completionRate: 0,
    });

    (useClearCompletedTasks as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });

    renderWithQueryClient(<QueueManager />);

    // Should show empty state message
    expect(
      screen.getByText(
        "No tasks in queue. Start generating videos to see them here!"
      )
    ).toBeInTheDocument();
  });
});
