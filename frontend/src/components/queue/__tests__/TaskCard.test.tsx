import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import TaskCard from "../TaskCard";
import { TaskInfo } from "@/lib/api-schemas";

// Mock the API hooks
vi.mock("@/hooks/api/use-queue", () => ({
  useCancelTask: vi.fn(),
  useDeleteTask: vi.fn(),
  useEstimatedCompletionTime: vi.fn(),
}));

// Mock the toast hook
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

const mockProcessingTask: TaskInfo = {
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
};

const mockCompletedTask: TaskInfo = {
  id: "task-2",
  model_type: "I2V-A14B",
  prompt: "Ocean waves crashing",
  image_path: "/uploads/image.jpg",
  resolution: "1920x1080",
  steps: 75,
  status: "completed",
  progress: 100,
  created_at: "2024-01-01T09:30:00Z",
  started_at: "2024-01-01T09:31:00Z",
  completed_at: "2024-01-01T09:45:00Z",
  output_path: "/outputs/task-2.mp4",
  lora_strength: 1.0,
};

const mockFailedTask: TaskInfo = {
  id: "task-3",
  model_type: "TI2V-5B",
  prompt: "City skyline at night",
  resolution: "1280x720",
  steps: 50,
  status: "failed",
  progress: 25,
  created_at: "2024-01-01T11:00:00Z",
  started_at: "2024-01-01T11:01:00Z",
  completed_at: "2024-01-01T11:05:00Z",
  error_message: "VRAM exhausted during generation",
  lora_strength: 1.0,
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

describe("TaskCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render processing task correctly", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(15); // 15 minutes

    renderWithQueryClient(<TaskCard task={mockProcessingTask} />);

    // Check task information
    expect(
      screen.getByText("A beautiful sunset over mountains")
    ).toBeInTheDocument();
    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("T2V-A14B")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("1280x720")).toBeInTheDocument();
    expect(screen.getByText("50")).toBeInTheDocument(); // steps
    expect(screen.getByText("15m")).toBeInTheDocument(); // estimated time
  });

  it("should render completed task correctly", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(null);

    renderWithQueryClient(<TaskCard task={mockCompletedTask} />);

    // Check task information
    expect(screen.getByText("Ocean waves crashing")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getByText("I2V-A14B")).toBeInTheDocument();
    expect(screen.getByText("1920x1080")).toBeInTheDocument();
    expect(screen.getByText("75")).toBeInTheDocument(); // steps

    // Should show output path
    expect(screen.getByText("Output:")).toBeInTheDocument();
    expect(screen.getByText("/outputs/task-2.mp4")).toBeInTheDocument();

    // Should show delete button (not cancel)
    expect(screen.queryByRole("button", { name: /cancel/i })).toBeNull();
    expect(screen.getByRole("button")).toBeInTheDocument(); // delete button
  });

  it("should render failed task with error message", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(null);

    renderWithQueryClient(<TaskCard task={mockFailedTask} />);

    // Check task information
    expect(screen.getByText("City skyline at night")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("TI2V-5B")).toBeInTheDocument();

    // Should show error message
    expect(screen.getByText("Error:")).toBeInTheDocument();
    expect(
      screen.getByText("VRAM exhausted during generation")
    ).toBeInTheDocument();
  });

  it("should handle task cancellation", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    const mockCancelMutation = vi.fn().mockResolvedValue({});
    (useCancelTask as any).mockReturnValue({
      mutateAsync: mockCancelMutation,
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(15);

    renderWithQueryClient(<TaskCard task={mockProcessingTask} />);

    // Find and click cancel button
    const cancelButton = screen.getByRole("button");
    fireEvent.click(cancelButton);

    await waitFor(() => {
      expect(mockCancelMutation).toHaveBeenCalledWith("task-1");
    });
  });

  it("should handle task deletion", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    const mockDeleteMutation = vi.fn().mockResolvedValue({});
    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: mockDeleteMutation,
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(null);

    renderWithQueryClient(<TaskCard task={mockCompletedTask} />);

    // Find and click delete button
    const deleteButton = screen.getByRole("button");
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(mockDeleteMutation).toHaveBeenCalledWith("task-2");
    });
  });

  it("should show LoRA information when present", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(null);

    const taskWithLora = {
      ...mockProcessingTask,
      lora_path: "style_anime.safetensors",
      lora_strength: 0.8,
    };

    renderWithQueryClient(<TaskCard task={taskWithLora} />);

    // Should show LoRA information
    expect(screen.getByText("LoRA:")).toBeInTheDocument();
    expect(
      screen.getByText("style_anime.safetensors (0.8x)")
    ).toBeInTheDocument();
  });

  it("should format time correctly", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(null);

    renderWithQueryClient(<TaskCard task={mockCompletedTask} />);

    // Should show formatted times
    expect(screen.getByText("Created:")).toBeInTheDocument();
    expect(screen.getByText("Started:")).toBeInTheDocument();
    expect(screen.getByText("Completed:")).toBeInTheDocument();
  });

  it("should show image icon for I2V/TI2V tasks", async () => {
    const { useCancelTask, useDeleteTask, useEstimatedCompletionTime } =
      await import("@/hooks/api/use-queue");

    (useCancelTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useDeleteTask as any).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useEstimatedCompletionTime as any).mockReturnValue(null);

    renderWithQueryClient(<TaskCard task={mockCompletedTask} />);

    // Should show image icon for I2V task
    const badge = screen.getByText("I2V-A14B").closest(".flex");
    expect(badge).toBeInTheDocument();
  });
});
