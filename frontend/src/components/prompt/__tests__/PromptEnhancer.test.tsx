import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import PromptEnhancer from "../PromptEnhancer";
import * as usePromptHook from "@/hooks/api/use-prompt";

// Mock the prompt hooks
vi.mock("@/hooks/api/use-prompt");
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

const mockUsePromptEnhancement = vi.mocked(usePromptHook.usePromptEnhancement);
const mockUsePromptPreview = vi.mocked(usePromptHook.usePromptPreview);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("PromptEnhancer", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    mockUsePromptEnhancement.mockReturnValue({
      enhancePrompt: vi.fn(),
      isEnhancing: false,
      enhanceError: null,
      lastEnhancement: null,
      reset: vi.fn(),
    });

    mockUsePromptPreview.mockReturnValue({
      data: null,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: false,
      status: "idle",
    } as any);
  });

  it("renders prompt enhancer interface", () => {
    render(<PromptEnhancer />, { wrapper: createWrapper() });

    expect(screen.getByText("Prompt Enhancement")).toBeInTheDocument();
    expect(screen.getByText("Preview")).toBeInTheDocument();
    expect(screen.getByText("Enhance")).toBeInTheDocument();

    // Check enhancement options
    expect(screen.getByLabelText("Cinematic Style")).toBeInTheDocument();
    expect(screen.getByLabelText("Style-Specific")).toBeInTheDocument();
    expect(screen.getByLabelText("Force VACE")).toBeInTheDocument();
  });

  it("displays character count", () => {
    render(<PromptEnhancer initialPrompt="Test prompt" />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByText("11/500 characters")).toBeInTheDocument();
  });

  it("calls enhance function when enhance button is clicked", async () => {
    const mockEnhancePrompt = vi.fn().mockResolvedValue({
      original_prompt: "Test prompt",
      enhanced_prompt: "Test prompt, cinematic lighting, high quality",
      enhancements_applied: ["Cinematic Style"],
      character_count: { original: 11, enhanced: 45, difference: 34 },
      vace_detected: false,
    });

    mockUsePromptEnhancement.mockReturnValue({
      enhancePrompt: mockEnhancePrompt,
      isEnhancing: false,
      enhanceError: null,
      lastEnhancement: null,
      reset: vi.fn(),
    });

    render(<PromptEnhancer initialPrompt="Test prompt" />, {
      wrapper: createWrapper(),
    });

    const enhanceButton = screen.getByText("Enhance");
    fireEvent.click(enhanceButton);

    await waitFor(() => {
      expect(mockEnhancePrompt).toHaveBeenCalledWith("Test prompt", {
        apply_vace: undefined,
        apply_cinematic: true,
        apply_style: true,
      });
    });
  });

  it("shows loading state when enhancing", () => {
    mockUsePromptEnhancement.mockReturnValue({
      enhancePrompt: vi.fn(),
      isEnhancing: true,
      enhanceError: null,
      lastEnhancement: null,
      reset: vi.fn(),
    });

    render(<PromptEnhancer initialPrompt="Test prompt" />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByRole("button", { name: /enhance/i })).toBeDisabled();
  });

  it("shows preview when preview button is clicked", async () => {
    const mockPreview = {
      original_prompt: "Test prompt",
      preview_enhanced: "Test prompt, cinematic lighting",
      suggested_enhancements: ["Cinematic Style", "Quality Keywords"],
      detected_style: "cinematic",
      vace_detected: false,
      character_count: { original: 11, preview: 35, difference: 24 },
    };

    mockUsePromptPreview.mockReturnValue({
      data: mockPreview,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<PromptEnhancer initialPrompt="Test prompt" />, {
      wrapper: createWrapper(),
    });

    const previewButton = screen.getByText("Preview");
    fireEvent.click(previewButton);

    await waitFor(() => {
      expect(screen.getByText("Enhancement Preview")).toBeInTheDocument();
      expect(screen.getByText("cinematic Style")).toBeInTheDocument();
      expect(screen.getByText("Suggested Enhancements:")).toBeInTheDocument();
      expect(screen.getAllByText("Cinematic Style")).toHaveLength(2); // Checkbox + suggestion
      expect(screen.getAllByText("Quality Keywords")).toHaveLength(1);
    });
  });

  it("shows enhancement result with apply/reject options", () => {
    const mockEnhancement = {
      original_prompt: "Test prompt",
      enhanced_prompt: "Test prompt, cinematic lighting, high quality",
      enhancements_applied: ["Cinematic Style", "Quality Keywords"],
      character_count: { original: 11, enhanced: 45, difference: 34 },
      detected_style: "cinematic",
      vace_detected: false,
    };

    mockUsePromptEnhancement.mockReturnValue({
      enhancePrompt: vi.fn(),
      isEnhancing: false,
      enhanceError: null,
      lastEnhancement: mockEnhancement,
      reset: vi.fn(),
    });

    render(<PromptEnhancer />, { wrapper: createWrapper() });

    expect(screen.getByText("Enhancement Complete")).toBeInTheDocument();
    expect(screen.getByText("Apply")).toBeInTheDocument();
    expect(screen.getByText("Reject")).toBeInTheDocument();

    // Check enhancement badges (using getAllByText since text appears multiple times)
    expect(screen.getAllByText("Cinematic Style")).toHaveLength(2); // Checkbox label + badge
    expect(screen.getByText("Quality Keywords")).toBeInTheDocument();
    expect(screen.getByText("cinematic Style")).toBeInTheDocument();

    // Check before/after comparison
    expect(screen.getByText("Original:")).toBeInTheDocument();
    expect(screen.getByText("Enhanced:")).toBeInTheDocument();
    expect(screen.getByText("11 → 45 characters (+34)")).toBeInTheDocument();
  });

  it("calls onPromptChange when prompt is applied", () => {
    const mockOnPromptChange = vi.fn();
    const mockOnEnhancementApplied = vi.fn();

    const mockEnhancement = {
      original_prompt: "Test prompt",
      enhanced_prompt: "Test prompt, cinematic lighting, high quality",
      enhancements_applied: ["Cinematic Style"],
      character_count: { original: 11, enhanced: 45, difference: 34 },
      vace_detected: false,
    };

    mockUsePromptEnhancement.mockReturnValue({
      enhancePrompt: vi.fn(),
      isEnhancing: false,
      enhanceError: null,
      lastEnhancement: mockEnhancement,
      reset: vi.fn(),
    });

    render(
      <PromptEnhancer
        onPromptChange={mockOnPromptChange}
        onEnhancementApplied={mockOnEnhancementApplied}
      />,
      { wrapper: createWrapper() }
    );

    const applyButton = screen.getByText("Apply");
    fireEvent.click(applyButton);

    expect(mockOnPromptChange).toHaveBeenCalledWith(
      "Test prompt, cinematic lighting, high quality"
    );
    expect(mockOnEnhancementApplied).toHaveBeenCalledWith(
      "Test prompt, cinematic lighting, high quality"
    );
  });

  it("shows error when enhancement fails", () => {
    const mockError = new Error("Enhancement failed");

    mockUsePromptEnhancement.mockReturnValue({
      enhancePrompt: vi.fn(),
      isEnhancing: false,
      enhanceError: mockError,
      lastEnhancement: null,
      reset: vi.fn(),
    });

    render(<PromptEnhancer />, { wrapper: createWrapper() });

    expect(screen.getByText("Enhancement Failed")).toBeInTheDocument();
    expect(screen.getByText("Enhancement failed")).toBeInTheDocument();
  });

  it("updates enhancement options when checkboxes are changed", () => {
    render(<PromptEnhancer initialPrompt="Test prompt" />, {
      wrapper: createWrapper(),
    });

    const cinematicCheckbox = screen.getByLabelText("Cinematic Style");
    const styleCheckbox = screen.getByLabelText("Style-Specific");
    const vaceCheckbox = screen.getByLabelText("Force VACE");

    // Initially cinematic and style should be checked
    expect(cinematicCheckbox).toBeChecked();
    expect(styleCheckbox).toBeChecked();
    expect(vaceCheckbox).not.toBeChecked();

    // Uncheck cinematic
    fireEvent.click(cinematicCheckbox);
    expect(cinematicCheckbox).not.toBeChecked();

    // Check VACE
    fireEvent.click(vaceCheckbox);
    expect(vaceCheckbox).toBeChecked();
  });

  it("disables enhance button for short prompts", () => {
    render(<PromptEnhancer initialPrompt="Hi" />, { wrapper: createWrapper() });

    const enhanceButton = screen.getByText("Enhance");
    expect(enhanceButton).toBeDisabled();
  });

  it("shows VACE detected badge in preview", () => {
    const mockPreview = {
      original_prompt: "Test prompt",
      preview_enhanced: "Test prompt, VACE aesthetic",
      suggested_enhancements: ["VACE Enhancement"],
      vace_detected: true,
      character_count: { original: 11, preview: 30, difference: 19 },
    };

    mockUsePromptPreview.mockReturnValue({
      data: mockPreview,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<PromptEnhancer initialPrompt="Test prompt" />, {
      wrapper: createWrapper(),
    });

    const previewButton = screen.getByText("Preview");
    fireEvent.click(previewButton);

    expect(screen.getByText("VACE Detected")).toBeInTheDocument();
  });
});
