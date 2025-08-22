import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Generation form state
interface GenerationFormData {
  modelType: 'T2V-A14B' | 'I2V-A14B' | 'TI2V-5B';
  prompt: string;
  image?: File;
  resolution: string;
  steps: number;
  loraPath?: string;
  loraStrength: number;
}

interface GenerationState {
  formData: GenerationFormData;
  isSubmitting: boolean;
  errors: Record<string, string>;
  enhancedPrompt?: string;
  
  // Actions
  updateFormData: (data: Partial<GenerationFormData>) => void;
  setSubmitting: (isSubmitting: boolean) => void;
  setErrors: (errors: Record<string, string>) => void;
  clearErrors: () => void;
  setEnhancedPrompt: (prompt: string) => void;
  resetForm: () => void;
}

const initialFormData: GenerationFormData = {
  modelType: 'T2V-A14B',
  prompt: '',
  resolution: '1280x720',
  steps: 50,
  loraStrength: 1.0,
};

export const useGenerationStore = create<GenerationState>()(
  devtools(
    (set) => ({
      // Initial state
      formData: initialFormData,
      isSubmitting: false,
      errors: {},
      enhancedPrompt: undefined,

      // Actions
      updateFormData: (data) =>
        set((state) => ({
          formData: { ...state.formData, ...data }
        })),

      setSubmitting: (isSubmitting) => set({ isSubmitting }),

      setErrors: (errors) => set({ errors }),

      clearErrors: () => set({ errors: {} }),

      setEnhancedPrompt: (enhancedPrompt) => set({ enhancedPrompt }),

      resetForm: () => set({
        formData: initialFormData,
        isSubmitting: false,
        errors: {},
        enhancedPrompt: undefined,
      }),
    }),
    { name: 'GenerationStore' }
  )
);