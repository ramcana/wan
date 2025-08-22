import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import QueueManager from "../QueueManager";

// Mock all the hooks with proper return values
vi.mock("@/hooks/api/use-queue", () => ({
  useQueueStatus: () => ({
    data: {
      total_tasks: 3,
      pending_tasks: 1,
      processing_tasks: 1,
      completed_tasks: 1,
      failed_tasks: 0,
      cancelled_tasks: 0,
      tasks: [
        {
          id: "task-1",
          model_type: "T2V-A14B",
          prompt: "A beautiful sunset",
          resolution: "1280x720",
          steps: 50,
          status: "processing",
          progress: 75,
          created_at: "2024-01-01T10:00:00Z",
          lora_strength: 1.0,
        },
        {
          id: "task-2",
          model_type: "I2V-A14B",
          prompt: "Ocean waves",
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
          prompt: "City skyline",
          resolution: "1920x1080",
          steps: 75,
          status: "completed",
          progress: 100,
          created_at: "2024-01-01T09:30:00Z",
          completed_at: "2024-01-01T09:45:00Z",
          output_path: "/outputs/task-3.mp4",
          lora_strength: 1.0,
        },
      ],
    },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useClearCompletedTasks: () => ({
    mutateAsync: vi.fn(),
    isLoading: false,
  }),
  useCancelTask: () => ({
    mutateAsync: vi.fn(),
    isLoading: false,
  }),
  useDeleteTask: () => ({
    mutateAsync: vi.fn(),
    isLoading: false,
  }),
  useEstimatedCompletionTime: () => 15,
  useTaskStatistics: () => ({
    totalTasks: 3,
    activeTasks: 2,
    completedTasks: 1,
    failedTasks: 0,
    completionRate: 33,
  }),
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

describe("Queue Integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render complete queue interface with all components", async () => {
    renderWithQueryClient(<QueueManager />);

    // Wait for component to render
    await waitFor(() => {
      // Check if main components are present
      expect(screen.getByText("Task Queue")).toBeInTheDocument();
      expect(screen.getByText("Total Tasks")).toBeInTheDocument();
      expect(screen.getByText("Active Tasks")).toBeInTheDocument();
      expect(screen.getByText("Completed")).toBeInTheDocument();
    });

    // Check if task cards are rendered
    expect(screen.getByText("A beautiful sunset")).toBeInTheDocument();
    expect(screen.getByText("Ocean waves")).toBeInTheDocument();
    expect(screen.getByText("City skyline")).toBeInTheDocument();

    // Check if status badges are present
    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();

    // Check if progress is shown for processing task
    expect(screen.getByText("75%")).toBeInTheDocument();

    // Check if controls are present
    expect(screen.getByText("Notifications")).toBeInTheDocument();
    expect(screen.getByText("Clear Completed")).toBeInTheDocument();
    expect(screen.getByText("Refresh")).toBeInTheDocument();

    // Check if filters are present
    expect(screen.getByText("all")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("processing")).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
  });

  it("should show queue statistics correctly", async () => {
    renderWithQueryClient(<QueueManager />);

    await waitFor(() => {
      // Check statistics values
      expect(screen.getByText("3")).toBeInTheDocument(); // Total tasks
      expect(screen.getByText("2")).toBeInTheDocument(); // Active tasks
      expect(screen.getByText("1")).toBeInTheDocument(); // Completed tasks
    });

    // Check completion rate
    expect(screen.getByText("33%")).toBeInTheDocument();
  });

  it("should display task details correctly", async () => {
    renderWithQueryClient(<QueueManager />);

    await waitFor(() => {
      // Check task details are displayed
      expect(screen.getByText("1280x720")).toBeInTheDocument();
      expect(screen.getByText("1920x1080")).toBeInTheDocument();
      expect(screen.getByText("50")).toBeInTheDocument(); // steps
      expect(screen.getByText("75")).toBeInTheDocument(); // steps

      // Check model types
      expect(screen.getByText("T2V-A14B")).toBeInTheDocument();
      expect(screen.getByText("I2V-A14B")).toBeInTheDocument();
      expect(screen.getByText("TI2V-5B")).toBeInTheDocument();
    });
  });

  it("should show estimated completion time for processing tasks", async () => {
    renderWithQueryClient(<QueueManager />);

    await waitFor(() => {
      // Check if estimated time is shown
      expect(screen.getByText("15m")).toBeInTheDocument();
    });
  });

  it("should show output path for completed tasks", async () => {
    renderWithQueryClient(<QueueManager />);

    await waitFor(() => {
      // Check if output path is shown for completed task
      expect(screen.getByText("Output:")).toBeInTheDocument();
      expect(screen.getByText("/outputs/task-3.mp4")).toBeInTheDocument();
    });
  });
});
