import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi } from "vitest";
import { LoRABrowser } from "../LoRABrowser";
import * as loraHooks from "../../../hooks/api/use-lora";

// Mock the hooks
vi.mock("../../../hooks/api/use-lora");

// Mock the child components
vi.mock("../LoRACard", () => ({
  LoRACard: ({ lora, onPreview, onSelect }: any) => (
    <div data-testid={`lora-card-${lora.name}`}>
      <span>{lora.name}</span>
      <button onClick={onPreview}>Preview</button>
      {onSelect && <button onClick={() => onSelect(lora)}>Select</button>}
    </div>
  ),
}));

vi.mock("../LoRAUploadDialog", () => ({
  LoRAUploadDialog: ({ open, onOpenChange }: any) =>
    open ? <div data-testid="upload-dialog">Upload Dialog</div> : null,
}));

vi.mock("../LoRAPreviewDialog", () => ({
  LoRAPreviewDialog: ({ lora, open, onOpenChange }: any) =>
    open ? <div data-testid="preview-dialog">Preview: {lora?.name}</div> : null,
}));

vi.mock("../../gallery/LoadingGrid", () => ({
  LoadingGrid: () => <div data-testid="loading-grid">Loading...</div>,
}));

vi.mock("../../gallery/EmptyState", () => ({
  EmptyState: ({ title, description, action }: any) => (
    <div data-testid="empty-state">
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  ),
}));

const mockLoRAs = [
  {
    name: "anime-style",
    filename: "anime-style.safetensors",
    path: "/loras/anime-style.safetensors",
    size_mb: 150.5,
    modified_time: "2024-01-15T10:30:00Z",
    is_loaded: false,
    is_applied: false,
    current_strength: 0.0,
  },
  {
    name: "realistic-portrait",
    filename: "realistic-portrait.pt",
    path: "/loras/realistic-portrait.pt",
    size_mb: 200.2,
    modified_time: "2024-01-14T15:45:00Z",
    is_loaded: true,
    is_applied: false,
    current_strength: 0.0,
  },
  {
    name: "cartoon-style",
    filename: "cartoon-style.safetensors",
    path: "/loras/cartoon-style.safetensors",
    size_mb: 120.8,
    modified_time: "2024-01-13T09:15:00Z",
    is_loaded: true,
    is_applied: true,
    current_strength: 1.2,
  },
];

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
};

describe("LoRABrowser", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: [],
      totalCount: 0,
      isLoading: true,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 0,
        totalSizeMB: 0,
        loadedCount: 0,
        appliedCount: 0,
        categories: { style: 0, character: 0, quality: 0 },
      },
      isLoading: true,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    expect(screen.getByTestId("loading-grid")).toBeInTheDocument();
  });

  it("renders LoRA list with statistics", () => {
    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    // Check statistics
    expect(screen.getByText("3")).toBeInTheDocument(); // Total count
    expect(screen.getByText("471.5MB")).toBeInTheDocument(); // Total size
    expect(screen.getByText("2")).toBeInTheDocument(); // Loaded count
    expect(screen.getByText("1")).toBeInTheDocument(); // Applied count

    // Check LoRA cards
    expect(screen.getByTestId("lora-card-anime-style")).toBeInTheDocument();
    expect(
      screen.getByTestId("lora-card-realistic-portrait")
    ).toBeInTheDocument();
    expect(screen.getByTestId("lora-card-cartoon-style")).toBeInTheDocument();
  });

  it("renders empty state when no LoRAs found", () => {
    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: [],
      totalCount: 0,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 0,
        totalSizeMB: 0,
        loadedCount: 0,
        appliedCount: 0,
        categories: { style: 0, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(screen.getByText("No LoRA files found")).toBeInTheDocument();
  });

  it("handles search functionality", async () => {
    const mockUseLoRASearch = vi.mocked(loraHooks.useLoRASearch);

    // Initial render with all LoRAs
    mockUseLoRASearch.mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    const searchInput = screen.getByPlaceholderText("Search LoRA files...");

    // Type in search
    fireEvent.change(searchInput, { target: { value: "anime" } });

    // Verify search was called with the term
    await waitFor(() => {
      expect(mockUseLoRASearch).toHaveBeenCalledWith("anime", "all");
    });
  });

  it("handles category filtering", async () => {
    const mockUseLoRASearch = vi.mocked(loraHooks.useLoRASearch);

    mockUseLoRASearch.mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    // Find and click the filter dropdown
    const filterTrigger = screen.getByRole("combobox");
    fireEvent.click(filterTrigger);

    // Select style category
    const styleOption = screen.getByText("Style (2)");
    fireEvent.click(styleOption);

    // Verify filter was applied
    await waitFor(() => {
      expect(mockUseLoRASearch).toHaveBeenCalledWith("", "style");
    });
  });

  it("opens upload dialog when upload button is clicked", () => {
    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    const uploadButton = screen.getByText("Upload LoRA");
    fireEvent.click(uploadButton);

    expect(screen.getByTestId("upload-dialog")).toBeInTheDocument();
  });

  it("opens preview dialog when preview is clicked", () => {
    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    const previewButton = screen.getAllByText("Preview")[0];
    fireEvent.click(previewButton);

    expect(screen.getByTestId("preview-dialog")).toBeInTheDocument();
    expect(screen.getByText("Preview: anime-style")).toBeInTheDocument();
  });

  it("handles LoRA selection when showSelection is true", () => {
    const mockOnLoRASelect = vi.fn();

    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(
      <LoRABrowser onLoRASelect={mockOnLoRASelect} showSelection={true} />
    );

    const selectButton = screen.getAllByText("Select")[0];
    fireEvent.click(selectButton);

    expect(mockOnLoRASelect).toHaveBeenCalledWith(mockLoRAs[0]);
  });

  it("handles refresh functionality", () => {
    const mockRefresh = vi.fn();

    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: mockLoRAs,
      totalCount: 3,
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 3,
        totalSizeMB: 471.5,
        loadedCount: 2,
        appliedCount: 1,
        categories: { style: 2, character: 0, quality: 0 },
      },
      isLoading: false,
      error: null,
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(mockRefresh);

    renderWithQueryClient(<LoRABrowser />);

    const refreshButton = screen.getByText("Refresh");
    fireEvent.click(refreshButton);

    expect(mockRefresh).toHaveBeenCalled();
  });

  it("renders error state", () => {
    vi.mocked(loraHooks.useLoRASearch).mockReturnValue({
      loras: [],
      totalCount: 0,
      isLoading: false,
      error: new Error("Failed to load LoRAs"),
    });

    vi.mocked(loraHooks.useLoRAStatistics).mockReturnValue({
      statistics: {
        totalCount: 0,
        totalSizeMB: 0,
        loadedCount: 0,
        appliedCount: 0,
        categories: { style: 0, character: 0, quality: 0 },
      },
      isLoading: false,
      error: new Error("Failed to load LoRAs"),
    });

    vi.mocked(loraHooks.useRefreshLoRAList).mockReturnValue(vi.fn());

    renderWithQueryClient(<LoRABrowser />);

    expect(screen.getByText("Failed to load LoRA files")).toBeInTheDocument();
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });
});
