import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "react-query";
import { GenerationForm } from "../GenerationForm";

// Mock the toast hook
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

// Test wrapper with React Query
const createTestWrapper = () => {
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

describe("GenerationForm", () => {
  let mockOnSubmit: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockOnSubmit = vi.fn().mockResolvedValue(undefined);
  });

  const renderForm = (props = {}) => {
    const Wrapper = createTestWrapper();
    return render(
      <Wrapper>
        <GenerationForm onSubmit={mockOnSubmit} {...props} />
      </Wrapper>
    );
  };

  describe("Form Rendering", () => {
    it("should render all required form fields", () => {
      renderForm();

      expect(screen.getByText("Model Type")).toBeInTheDocument();
      expect(screen.getByText("Prompt")).toBeInTheDocument();
      expect(screen.getByText("Resolution")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /generate video/i })
      ).toBeInTheDocument();
    });

    it("should show character counter for prompt", () => {
      renderForm();

      expect(screen.getByText("0/500")).toBeInTheDocument();
    });
  });

  describe("Form Validation", () => {
    it("should require prompt input", async () => {
      const user = userEvent.setup();
      renderForm();

      const submitButton = screen.getByRole("button", {
        name: /generate video/i,
      });
      expect(submitButton).toBeDisabled();

      const promptInput = screen.getByRole("textbox");
      await user.type(promptInput, "A beautiful sunset");

      await waitFor(() => {
        expect(submitButton).toBeEnabled();
      });
    });

    it("should update character counter as user types", async () => {
      const user = userEvent.setup();
      renderForm();

      const promptInput = screen.getByRole("textbox");
      await user.type(promptInput, "Hello world");

      await waitFor(() => {
        expect(screen.getByText("11/500")).toBeInTheDocument();
      });
    });
  });

  describe("Form Submission", () => {
    it("should show loading state during submission", () => {
      renderForm({ isSubmitting: true });

      const submitButton = screen.getByRole("button", { name: /generating/i });
      expect(submitButton).toBeDisabled();
      expect(screen.getByText(/generating/i)).toBeInTheDocument();
    });

    it("should display error message", () => {
      const errorMessage = "Generation failed: Insufficient VRAM";
      renderForm({ error: errorMessage });

      expect(screen.getByText(/generation failed/i)).toBeInTheDocument();
    });

    it("should show validation errors", async () => {
      const user = userEvent.setup();
      renderForm();

      const promptInput = screen.getByRole("textbox");
      await user.type(promptInput, "a".repeat(501));

      await waitFor(() => {
        expect(
          screen.getByText(/please fix the following issues/i)
        ).toBeInTheDocument();
      });
    });

    it("should show success indicator when form is valid", async () => {
      const user = userEvent.setup();
      renderForm();

      const promptInput = screen.getByRole("textbox");
      await user.type(promptInput, "A beautiful sunset over mountains");

      await waitFor(() => {
        expect(screen.getByText(/ready to generate/i)).toBeInTheDocument();
      });
    });

    it("should disable submit button when form is invalid", async () => {
      const user = userEvent.setup();
      renderForm();

      const submitButton = screen.getByRole("button", {
        name: /generate video/i,
      });
      expect(submitButton).toBeDisabled();

      const promptInput = screen.getByRole("textbox");
      await user.type(promptInput, "a".repeat(501));

      await waitFor(() => {
        expect(submitButton).toBeDisabled();
      });
    });

    it("should enable submit button when form is valid", async () => {
      const user = userEvent.setup();
      renderForm();

      const promptInput = screen.getByRole("textbox");
      await user.type(promptInput, "A beautiful sunset");

      await waitFor(() => {
        const submitButton = screen.getByRole("button", {
          name: /generate video/i,
        });
        expect(submitButton).toBeEnabled();
      });
    });
  });

  describe("Advanced Settings", () => {
    it("should toggle advanced settings panel", async () => {
      const user = userEvent.setup();
      renderForm();

      const advancedButton = screen.getByRole("button", {
        name: /advanced settings/i,
      });

      // Panel should be hidden initially
      expect(screen.queryByText(/generation steps/i)).not.toBeInTheDocument();

      // Click to show panel
      await user.click(advancedButton);
      expect(screen.getByText(/generation steps/i)).toBeInTheDocument();
    });
  });
});
