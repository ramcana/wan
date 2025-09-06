import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { GenerationPanel } from "../../../../components/generation/GenerationPanel";
import { useGeneration } from "../../../../hooks/api/use-generation";

// Mock the API hook
vi.mock("../../../../hooks/api/use-generation");

const mockUseGeneration = vi.mocked(useGeneration);

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

describe("GenerationPanel", () => {
  beforeEach(() => {
    mockUseGeneration.mockReturnValue({
      generateVideo: vi.fn(),
      isLoading: false,
      error: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test("renders all form fields correctly", () => {
    render(<GenerationPanel />, { wrapper: createWrapper() });

    expect(screen.getByLabelText(/model type/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/prompt/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/resolution/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/steps/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /generate/i })
    ).toBeInTheDocument();
  });

  test("shows image upload when I2V model is selected", async () => {
    render(<GenerationPanel />, { wrapper: createWrapper() });

    const modelSelect = screen.getByLabelText(/model type/i);
    fireEvent.change(modelSelect, { target: { value: "I2V-A14B" } });

    await waitFor(() => {
      expect(screen.getByText(/upload image/i)).toBeInTheDocument();
    });
  });

  test("validates prompt length", async () => {
    render(<GenerationPanel />, { wrapper: createWrapper() });

    const promptInput = screen.getByLabelText(/prompt/i);
    fireEvent.change(promptInput, {
      target: { value: "a".repeat(501) },
    });

    await waitFor(() => {
      expect(
        screen.getByText(/prompt must be 500 characters or less/i)
      ).toBeInTheDocument();
    });
  });

  test("submits form with correct data", async () => {
    const mockGenerate = vi.fn();
    mockUseGeneration.mockReturnValue({
      generateVideo: mockGenerate,
      isLoading: false,
      error: null,
    });

    render(<GenerationPanel />, { wrapper: createWrapper() });

    // Fill form
    fireEvent.change(screen.getByLabelText(/model type/i), {
      target: { value: "T2V-A14B" },
    });
    fireEvent.change(screen.getByLabelText(/prompt/i), {
      target: { value: "A beautiful sunset" },
    });
    fireEvent.change(screen.getByLabelText(/resolution/i), {
      target: { value: "1280x720" },
    });

    // Submit
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    await waitFor(() => {
      expect(mockGenerate).toHaveBeenCalledWith({
        modelType: "T2V-A14B",
        prompt: "A beautiful sunset",
        resolution: "1280x720",
        steps: 50,
      });
    });
  });

  test("displays loading state during generation", () => {
    mockUseGeneration.mockReturnValue({
      generateVideo: vi.fn(),
      isLoading: true,
      error: null,
    });

    render(<GenerationPanel />, { wrapper: createWrapper() });

    expect(screen.getByRole("button", { name: /generating/i })).toBeDisabled();
  });

  test("displays error message when generation fails", () => {
    mockUseGeneration.mockReturnValue({
      generateVideo: vi.fn(),
      isLoading: false,
      error: new Error("Generation failed"),
    });

    render(<GenerationPanel />, { wrapper: createWrapper() });

    expect(screen.getByText(/generation failed/i)).toBeInTheDocument();
  });
});
