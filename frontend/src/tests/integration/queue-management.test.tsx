import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { QueueManager } from "../../components/queue/QueueManager";
import { apiClient } from "../../lib/api-client";

vi.mock("../../lib/api-client");

const mockApiClient = vi.mocked(apiClient);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("Queue Management Integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("handles queue updates in real-time", async () => {
    let queueData = [
      {
        id: "task-1",
        modelType: "T2V-A14B",
        prompt: "Test task",
        status: "pending",
        progress: 0,
        createdAt: new Date().toISOString(),
      },
    ];

    mockApiClient.get.mockImplementation(() =>
      Promise.resolve({ data: queueData })
    );

    const { rerender } = render(<QueueManager />, { wrapper: createWrapper() });

    // Initial state
    await waitFor(() => {
      expect(screen.getByText("Test task")).toBeInTheDocument();
      expect(screen.getByText(/pending/i)).toBeInTheDocument();
    });

    // Update queue data to simulate progress
    queueData = [
      {
        ...queueData[0],
        status: "processing",
        progress: 50,
      },
    ];

    rerender(<QueueManager />);

    await waitFor(() => {
      expect(screen.getByText(/processing/i)).toBeInTheDocument();
      expect(screen.getByRole("progressbar")).toHaveAttribute(
        "aria-valuenow",
        "50"
      );
    });

    // Complete task
    queueData = [
      {
        ...queueData[0],
        status: "completed",
        progress: 100,
      },
    ];

    rerender(<QueueManager />);

    await waitFor(() => {
      expect(screen.getByText(/completed/i)).toBeInTheDocument();
      expect(screen.getByRole("progressbar")).toHaveAttribute(
        "aria-valuenow",
        "100"
      );
    });
  });

  test("handles multiple task operations", async () => {
    const queueData = [
      {
        id: "task-1",
        modelType: "T2V-A14B",
        prompt: "Task 1",
        status: "processing",
        progress: 25,
        createdAt: new Date().toISOString(),
      },
      {
        id: "task-2",
        modelType: "I2V-A14B",
        prompt: "Task 2",
        status: "pending",
        progress: 0,
        createdAt: new Date().toISOString(),
      },
      {
        id: "task-3",
        modelType: "TI2V-5B",
        prompt: "Task 3",
        status: "completed",
        progress: 100,
        createdAt: new Date().toISOString(),
      },
    ];

    mockApiClient.get.mockResolvedValue({ data: queueData });
    mockApiClient.delete.mockResolvedValue({ data: { success: true } });

    render(<QueueManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText("Task 1")).toBeInTheDocument();
      expect(screen.getByText("Task 2")).toBeInTheDocument();
      expect(screen.getByText("Task 3")).toBeInTheDocument();
    });

    // Test bulk operations
    const selectAllCheckbox = screen.getByLabelText(/select all/i);
    fireEvent.click(selectAllCheckbox);

    // Clear completed tasks
    const clearCompletedButton = screen.getByRole("button", {
      name: /clear completed/i,
    });
    fireEvent.click(clearCompletedButton);

    await waitFor(() => {
      expect(mockApiClient.delete).toHaveBeenCalledWith("/api/v1/queue/task-3");
    });
  });

  test("handles queue errors gracefully", async () => {
    mockApiClient.get.mockRejectedValue(new Error("Queue service unavailable"));

    render(<QueueManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(
        screen.getByText(/queue service unavailable/i)
      ).toBeInTheDocument();
    });

    // Test retry functionality
    const retryButton = screen.getByRole("button", { name: /retry/i });

    // Mock successful retry
    mockApiClient.get.mockResolvedValue({ data: [] });

    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText(/no tasks in queue/i)).toBeInTheDocument();
    });
  });

  test("handles task cancellation with confirmation", async () => {
    const queueData = [
      {
        id: "task-1",
        modelType: "T2V-A14B",
        prompt: "Long running task",
        status: "processing",
        progress: 75,
        createdAt: new Date().toISOString(),
      },
    ];

    mockApiClient.get.mockResolvedValue({ data: queueData });
    mockApiClient.delete.mockResolvedValue({ data: { success: true } });

    render(<QueueManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText("Long running task")).toBeInTheDocument();
    });

    // Click cancel button
    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    fireEvent.click(cancelButton);

    // Confirm cancellation in dialog
    await waitFor(() => {
      expect(screen.getByText(/are you sure/i)).toBeInTheDocument();
    });

    const confirmButton = screen.getByRole("button", { name: /confirm/i });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(mockApiClient.delete).toHaveBeenCalledWith("/api/v1/queue/task-1");
    });
  });

  test("handles queue reordering", async () => {
    const queueData = [
      {
        id: "task-1",
        modelType: "T2V-A14B",
        prompt: "Task 1",
        status: "pending",
        progress: 0,
        priority: 1,
        createdAt: new Date().toISOString(),
      },
      {
        id: "task-2",
        modelType: "I2V-A14B",
        prompt: "Task 2",
        status: "pending",
        progress: 0,
        priority: 2,
        createdAt: new Date().toISOString(),
      },
    ];

    mockApiClient.get.mockResolvedValue({ data: queueData });
    mockApiClient.put.mockResolvedValue({ data: { success: true } });

    render(<QueueManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText("Task 1")).toBeInTheDocument();
      expect(screen.getByText("Task 2")).toBeInTheDocument();
    });

    // Simulate drag and drop reordering
    const task1 = screen.getByText("Task 1").closest('[draggable="true"]');
    const task2 = screen.getByText("Task 2").closest('[draggable="true"]');

    // Mock drag events
    fireEvent.dragStart(task1!);
    fireEvent.dragOver(task2!);
    fireEvent.drop(task2!);

    await waitFor(() => {
      expect(mockApiClient.put).toHaveBeenCalledWith("/api/v1/queue/reorder", {
        taskOrder: ["task-2", "task-1"],
      });
    });
  });

  test("displays queue statistics", async () => {
    const queueData = [
      {
        id: "task-1",
        modelType: "T2V-A14B",
        prompt: "Task 1",
        status: "completed",
        progress: 100,
        createdAt: new Date().toISOString(),
      },
      {
        id: "task-2",
        modelType: "I2V-A14B",
        prompt: "Task 2",
        status: "processing",
        progress: 50,
        createdAt: new Date().toISOString(),
      },
      {
        id: "task-3",
        modelType: "TI2V-5B",
        prompt: "Task 3",
        status: "pending",
        progress: 0,
        createdAt: new Date().toISOString(),
      },
    ];

    mockApiClient.get.mockResolvedValue({ data: queueData });

    render(<QueueManager />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText("3 tasks")).toBeInTheDocument();
      expect(screen.getByText("1 completed")).toBeInTheDocument();
      expect(screen.getByText("1 processing")).toBeInTheDocument();
      expect(screen.getByText("1 pending")).toBeInTheDocument();
    });
  });
});
