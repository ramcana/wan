import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErrorDisplay } from "../ErrorDisplay";
import { ApiError } from "@/lib/api-client";

describe("ErrorDisplay", () => {
  const mockOnRetry = vi.fn();

  beforeEach(() => {
    mockOnRetry.mockClear();
  });

  it("should not render when error is null", () => {
    const { container } = render(<ErrorDisplay error={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("should render string error", () => {
    render(<ErrorDisplay error="Something went wrong" />);

    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("should render generic Error object", () => {
    const error = new Error("Test error message");
    render(<ErrorDisplay error={error} />);

    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("Test error message")).toBeInTheDocument();
  });

  it("should render VRAM error with specific suggestions", () => {
    const error = new ApiError(
      400,
      "Insufficient VRAM available",
      {},
      "INSUFFICIENT_VRAM"
    );
    render(<ErrorDisplay error={error} />);

    expect(screen.getByText("Insufficient VRAM")).toBeInTheDocument();
    expect(
      screen.getByText("Try reducing the resolution to 720p or lower")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Enable INT8 quantization in advanced settings")
    ).toBeInTheDocument();
  });

  it("should render model loading error with retry button", () => {
    const error = new ApiError(
      500,
      "Failed to load model",
      {},
      "MODEL_LOADING_ERROR"
    );
    render(<ErrorDisplay error={error} onRetry={mockOnRetry} />);

    expect(screen.getByText("Model Loading Failed")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("should call onRetry when retry button is clicked", async () => {
    const user = userEvent.setup();
    const error = new ApiError(500, "Server error", {}, "SERVER_ERROR");
    render(<ErrorDisplay error={error} onRetry={mockOnRetry} />);

    const retryButton = screen.getByRole("button", { name: /retry/i });
    await user.click(retryButton);

    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it("should render validation error without retry button", () => {
    const error = new ApiError(
      422,
      "Invalid input data",
      {},
      "VALIDATION_ERROR"
    );
    render(<ErrorDisplay error={error} onRetry={mockOnRetry} />);

    expect(screen.getByText("Invalid Input")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /retry/i })
    ).not.toBeInTheDocument();
  });

  it("should render network error with connection suggestions", () => {
    const error = new ApiError(0, "Network error", {}, "NETWORK_ERROR");
    render(<ErrorDisplay error={error} />);

    expect(screen.getByText("Connection Error")).toBeInTheDocument();
    expect(
      screen.getByText("Check your internet connection")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Verify the server is running")
    ).toBeInTheDocument();
  });

  it("should show error details for server errors", () => {
    const error = new ApiError(
      500,
      "Internal server error",
      {},
      "SERVER_ERROR"
    );
    render(<ErrorDisplay error={error} />);

    expect(screen.getByText(/Error ID: 500/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /copy error details/i })
    ).toBeInTheDocument();
  });

  it("should handle queue full error", () => {
    const error = new ApiError(
      429,
      "Queue is at maximum capacity",
      {},
      "QUEUE_FULL"
    );
    render(<ErrorDisplay error={error} />);

    expect(screen.getByText("Queue is Full")).toBeInTheDocument();
    expect(
      screen.getByText("Wait for current tasks to complete")
    ).toBeInTheDocument();
    expect(screen.getByText("Cancel some pending tasks")).toBeInTheDocument();
  });
});
