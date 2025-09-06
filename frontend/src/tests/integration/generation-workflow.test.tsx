import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "../../App";
import { apiClient } from "../../lib/api-client";

// Mock API client
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
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe("Generation Workflow Integration", () => {
  beforeEach(() => {
    // Mock successful API responses
    mockApiClient.post.mockImplementation((url) => {
      if (url === "/api/v1/generate") {
        return Promise.resolve({
          data: {
            task_id: "test-task-1",
            status: "pending",
            message: "Task added to queue",
          },
        });
      }
      return Promise.reject(new Error("Unknown endpoint"));
    });

    mockApiClient.get.mockImplementation((url) => {
      if (url === "/api/v1/queue") {
        return Promise.resolve({
          data: [
            {
              id: "test-task-1",
              modelType: "T2V-A14B",
              prompt: "A beautiful sunset over mountains",
              status: "processing",
              progress: 50,
              createdAt: new Date().toISOString(),
            },
          ],
        });
      }
      if (url === "/api/v1/outputs") {
        return Promise.resolve({ data: [] });
      }
      if (url === "/api/v1/system/stats") {
        return Promise.resolve({
          data: {
            cpu: 45.0,
            ram: { used: 8.0, total: 16.0 },
            gpu: 60.0,
            vram: { used: 4.0, total: 8.0 },
            timestamp: new Date().toISOString(),
          },
        });
      }
      return Promise.reject(new Error("Unknown endpoint"));
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test("complete T2V generation workflow", async () => {
    render(<App />, { wrapper: createWrapper() });

    // Navigate to generation page
    fireEvent.click(screen.getByText(/generate/i));

    // Fill out generation form
    await waitFor(() => {
      expect(screen.getByLabelText(/model type/i)).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/model type/i), {
      target: { value: "T2V-A14B" },
    });

    fireEvent.change(screen.getByLabelText(/prompt/i), {
      target: { value: "A beautiful sunset over mountains" },
    });

    fireEvent.change(screen.getByLabelText(/resolution/i), {
      target: { value: "1280x720" },
    });

    // Submit generation request
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    // Verify API call was made
    await waitFor(() => {
      expect(mockApiClient.post).toHaveBeenCalledWith("/api/v1/generate", {
        model_type: "T2V-A14B",
        prompt: "A beautiful sunset over mountains",
        resolution: "1280x720",
        steps: 50,
      });
    });

    // Navigate to queue to see task
    fireEvent.click(screen.getByText(/queue/i));

    // Verify task appears in queue
    await waitFor(() => {
      expect(
        screen.getByText("A beautiful sunset over mountains")
      ).toBeInTheDocument();
      expect(screen.getByText("T2V-A14B")).toBeInTheDocument();
      expect(screen.getByText(/processing/i)).toBeInTheDocument();
    });

    // Verify progress is shown
    const progressBar = screen.getByRole("progressbar");
    expect(progressBar).toHaveAttribute("aria-valuenow", "50");
  });

  test("I2V generation workflow with image upload", async () => {
    render(<App />, { wrapper: createWrapper() });

    // Navigate to generation page
    fireEvent.click(screen.getByText(/generate/i));

    await waitFor(() => {
      expect(screen.getByLabelText(/model type/i)).toBeInTheDocument();
    });

    // Select I2V model
    fireEvent.change(screen.getByLabelText(/model type/i), {
      target: { value: "I2V-A14B" },
    });

    // Verify image upload appears
    await waitFor(() => {
      expect(screen.getByText(/upload image/i)).toBeInTheDocument();
    });

    // Create mock file
    const file = new File(["test"], "test.jpg", { type: "image/jpeg" });

    // Upload image
    const fileInput = screen.getByLabelText(/upload image/i);
    fireEvent.change(fileInput, { target: { files: [file] } });

    // Fill prompt
    fireEvent.change(screen.getByLabelText(/prompt/i), {
      target: { value: "Transform this image into a video" },
    });

    // Submit
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    // Verify API call includes image
    await waitFor(() => {
      expect(mockApiClient.post).toHaveBeenCalledWith(
        "/api/v1/generate",
        expect.any(FormData)
      );
    });
  });

  test("queue management workflow", async () => {
    // Mock queue with multiple tasks
    mockApiClient.get.mockImplementation((url) => {
      if (url === "/api/v1/queue") {
        return Promise.resolve({
          data: [
            {
              id: "task-1",
              modelType: "T2V-A14B",
              prompt: "Task 1",
              status: "processing",
              progress: 75,
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
          ],
        });
      }
      return Promise.resolve({ data: [] });
    });

    mockApiClient.delete.mockResolvedValue({ data: { success: true } });

    render(<App />, { wrapper: createWrapper() });

    // Navigate to queue
    fireEvent.click(screen.getByText(/queue/i));

    // Verify tasks are displayed
    await waitFor(() => {
      expect(screen.getByText("Task 1")).toBeInTheDocument();
      expect(screen.getByText("Task 2")).toBeInTheDocument();
    });

    // Cancel a task
    const cancelButtons = screen.getAllByRole("button", { name: /cancel/i });
    fireEvent.click(cancelButtons[0]);

    // Verify cancellation API call
    await waitFor(() => {
      expect(mockApiClient.delete).toHaveBeenCalledWith("/api/v1/queue/task-1");
    });
  });

  test("system monitoring integration", async () => {
    render(<App />, { wrapper: createWrapper() });

    // Navigate to monitoring
    fireEvent.click(screen.getByText(/monitor/i));

    // Verify system stats are displayed
    await waitFor(() => {
      expect(screen.getByText("CPU")).toBeInTheDocument();
      expect(screen.getByText("45.0%")).toBeInTheDocument();
      expect(screen.getByText("8.0 / 16.0 GB")).toBeInTheDocument();
    });

    // Verify API call was made
    expect(mockApiClient.get).toHaveBeenCalledWith("/api/v1/system/stats");
  });

  test("error handling workflow", async () => {
    // Mock API error
    mockApiClient.post.mockRejectedValue(new Error("Generation failed"));

    render(<App />, { wrapper: createWrapper() });

    // Navigate to generation
    fireEvent.click(screen.getByText(/generate/i));

    await waitFor(() => {
      expect(screen.getByLabelText(/prompt/i)).toBeInTheDocument();
    });

    // Fill form and submit
    fireEvent.change(screen.getByLabelText(/prompt/i), {
      target: { value: "Test prompt" },
    });

    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    // Verify error is displayed
    await waitFor(() => {
      expect(screen.getByText(/generation failed/i)).toBeInTheDocument();
    });
  });

  test("offline handling workflow", async () => {
    // Mock network error
    mockApiClient.get.mockRejectedValue(new Error("Network Error"));

    render(<App />, { wrapper: createWrapper() });

    // Verify offline indicator appears
    await waitFor(() => {
      expect(screen.getByText(/offline/i)).toBeInTheDocument();
    });

    // Try to navigate - should show offline message
    fireEvent.click(screen.getByText(/generate/i));

    await waitFor(() => {
      expect(
        screen.getByText(/you are currently offline/i)
      ).toBeInTheDocument();
    });
  });
});
