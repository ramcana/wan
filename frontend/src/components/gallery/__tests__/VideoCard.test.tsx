import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi } from "vitest";
import VideoCard from "../VideoCard";
import { VideoMetadata } from "@/lib/api-schemas";

// Mock the hooks
vi.mock("@/hooks/api/use-outputs", () => ({
  useDeleteVideo: () => ({
    mutateAsync: vi.fn(),
  }),
}));

// Mock the toast hook
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

// Mock the API
vi.mock("@/lib/api", () => ({
  outputsApi: {
    getDownloadUrl: vi.fn(
      (id) => `http://localhost:8000/api/v1/outputs/${id}/download`
    ),
    getThumbnailUrl: vi.fn(
      (id) => `http://localhost:8000/api/v1/outputs/${id}/thumbnail`
    ),
  },
}));

const mockVideo: VideoMetadata = {
  id: "1",
  filename: "test-video.mp4",
  file_path: "/outputs/test-video.mp4",
  thumbnail_path: "/thumbnails/1_thumb.jpg",
  prompt: "A beautiful sunset over mountains",
  model_type: "T2V-A14B",
  resolution: "1280x720",
  duration_seconds: 5,
  file_size_mb: 10.5,
  created_at: "2024-01-15T10:00:00Z",
  generation_time_minutes: 3.5,
};

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
};

describe("VideoCard", () => {
  const mockOnSelect = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders video information correctly", () => {
    renderWithProviders(
      <VideoCard video={mockVideo} onSelect={mockOnSelect} />
    );

    expect(screen.getByText("test-video.mp4")).toBeInTheDocument();
    expect(
      screen.getByText("A beautiful sunset over mountains")
    ).toBeInTheDocument();
    expect(screen.getByText("T2V-A14B")).toBeInTheDocument();
    expect(screen.getByText("1280x720")).toBeInTheDocument();
    expect(screen.getByText("10.5 MB")).toBeInTheDocument();
    expect(screen.getByText("4 min generation")).toBeInTheDocument();
  });

  it("calls onSelect when play button is clicked", async () => {
    renderWithProviders(
      <VideoCard video={mockVideo} onSelect={mockOnSelect} />
    );

    const buttons = screen.getAllByRole("button");
    const playButton = buttons[0]; // First button should be the play button
    await userEvent.click(playButton);
    expect(mockOnSelect).toHaveBeenCalledWith(mockVideo);
  });

  it("formats file size correctly for small files", () => {
    const smallVideo = { ...mockVideo, file_size_mb: 0.5 };
    renderWithProviders(
      <VideoCard video={smallVideo} onSelect={mockOnSelect} />
    );

    expect(screen.getByText("512 KB")).toBeInTheDocument();
  });

  it("formats duration correctly", () => {
    const longVideo = { ...mockVideo, duration_seconds: 125 };
    renderWithProviders(
      <VideoCard video={longVideo} onSelect={mockOnSelect} />
    );

    expect(screen.getByText("2:05")).toBeInTheDocument();
  });

  it("shows correct model type badge", () => {
    renderWithProviders(
      <VideoCard video={mockVideo} onSelect={mockOnSelect} />
    );

    const badge = screen.getByText("T2V-A14B");
    expect(badge).toBeInTheDocument();
  });

  it("handles missing thumbnail gracefully", () => {
    const videoWithoutThumbnail = { ...mockVideo, thumbnail_path: undefined };
    renderWithProviders(
      <VideoCard video={videoWithoutThumbnail} onSelect={mockOnSelect} />
    );

    // Should still render the video card
    expect(screen.getByText("test-video.mp4")).toBeInTheDocument();
  });

  it("handles missing duration gracefully", () => {
    const videoWithoutDuration = { ...mockVideo, duration_seconds: undefined };
    renderWithProviders(
      <VideoCard video={videoWithoutDuration} onSelect={mockOnSelect} />
    );

    // Should still render the video card
    expect(screen.getByText("test-video.mp4")).toBeInTheDocument();
    // Duration badge should not be present
    expect(screen.queryByText(/:/)).not.toBeInTheDocument();
  });

  it("shows generation time when available", () => {
    renderWithProviders(
      <VideoCard video={mockVideo} onSelect={mockOnSelect} />
    );

    expect(screen.getByText("4 min generation")).toBeInTheDocument();
  });

  it("handles missing generation time gracefully", () => {
    const videoWithoutGenTime = {
      ...mockVideo,
      generation_time_minutes: undefined,
    };
    renderWithProviders(
      <VideoCard video={videoWithoutGenTime} onSelect={mockOnSelect} />
    );

    // Should still render the video card
    expect(screen.getByText("test-video.mp4")).toBeInTheDocument();
    // Generation time should not be present
    expect(screen.queryByText(/min generation/)).not.toBeInTheDocument();
  });
});
