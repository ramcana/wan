import { useMutation, useQuery } from 'react-query';
import { post, get, ApiError } from '@/lib/api-client';

// Types for prompt enhancement
export interface PromptEnhanceRequest {
  prompt: string;
  options?: {
    apply_vace?: boolean;
    apply_cinematic?: boolean;
    apply_style?: boolean;
  };
}

export interface PromptEnhanceResponse {
  original_prompt: string;
  enhanced_prompt: string;
  enhancements_applied: string[];
  character_count: {
    original: number;
    enhanced: number;
    difference: number;
  };
  detected_style?: string;
  vace_detected: boolean;
}

export interface PromptPreviewResponse {
  original_prompt: string;
  preview_enhanced: string;
  suggested_enhancements: string[];
  detected_style?: string;
  vace_detected: boolean;
  character_count: {
    original: number;
    preview: number;
    difference: number;
  };
  quality_score?: number;
}

export interface PromptValidationResponse {
  is_valid: boolean;
  message: string;
  character_count: number;
  suggestions?: string[];
}

export interface StyleInfo {
  name: string;
  display_name: string;
  description: string;
}

// API functions
const promptApi = {
  enhance: async (data: PromptEnhanceRequest): Promise<PromptEnhanceResponse> => {
    return await post('/api/v1/prompt/enhance', {
      prompt: data.prompt,
      options: data.options || {}
    });
  },

  preview: async (prompt: string): Promise<PromptPreviewResponse> => {
    return await post('/api/v1/prompt/preview', {
      prompt: prompt,
      options: {}
    });
  },

  validate: async (prompt: string): Promise<PromptValidationResponse> => {
    return await post('/api/v1/prompt/validate', {
      prompt: prompt
    });
  },

  getStyles: async (): Promise<{ styles: StyleInfo[]; total_count: number }> => {
    return await get('/api/v1/prompt/styles');
  },
};

// Query keys
export const promptKeys = {
  all: ['prompt'] as const,
  styles: () => [...promptKeys.all, 'styles'] as const,
  preview: (prompt: string) => [...promptKeys.all, 'preview', prompt] as const,
  validation: (prompt: string) => [...promptKeys.all, 'validation', prompt] as const,
};

// Hooks
export const usePromptEnhance = () => {
  return useMutation({
    mutationFn: promptApi.enhance,
    retry: (failureCount, error) => {
      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status !== undefined && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

export const usePromptPreview = (prompt: string, enabled: boolean = true) => {
  return useQuery({
    queryKey: promptKeys.preview(prompt),
    queryFn: () => promptApi.preview(prompt),
    enabled: enabled && prompt.length > 0,
    staleTime: 30000, // Cache for 30 seconds
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status !== undefined && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

export const usePromptValidation = (prompt: string, enabled: boolean = true) => {
  return useQuery({
    queryKey: promptKeys.validation(prompt),
    queryFn: () => promptApi.validate(prompt),
    enabled: enabled && prompt.length > 0,
    staleTime: 10000, // Cache for 10 seconds
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status !== undefined && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

export const usePromptStyles = () => {
  return useQuery({
    queryKey: promptKeys.styles(),
    queryFn: promptApi.getStyles,
    staleTime: 300000, // Cache for 5 minutes
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status !== undefined && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

// Custom hook for prompt enhancement with preview
export const usePromptEnhancement = () => {
  const enhanceMutation = usePromptEnhance();
  
  const enhancePrompt = async (
    prompt: string,
    options: {
      apply_vace?: boolean;
      apply_cinematic?: boolean;
      apply_style?: boolean;
    } = {}
  ) => {
    try {
      const result = await enhanceMutation.mutateAsync({
        prompt,
        options,
      });
      return result;
    } catch (error) {
      throw error;
    }
  };

  return {
    enhancePrompt,
    isEnhancing: enhanceMutation.isLoading,
    enhanceError: enhanceMutation.error,
    lastEnhancement: enhanceMutation.data,
    reset: enhanceMutation.reset,
  };
};