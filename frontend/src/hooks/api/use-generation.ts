import { useMutation, useQueryClient } from "react-query";
import { post } from "@/lib/api-client";
import { 
  GenerationResponse, 
  GenerationFormData,
  validateApiResponse,
  GenerationResponseSchema
} from "@/lib/api-schemas";
import { ApiError } from "@/lib/api-client";

// Hook for submitting generation requests
export const useGenerateVideo = () => {
  const queryClient = useQueryClient();

  return useMutation<GenerationResponse, ApiError, { formData: GenerationFormData; imageFile?: File; endImageFile?: File }>({
    mutationFn: async ({ formData, imageFile, endImageFile }) => {
      // Always use FormData since backend expects multipart/form-data
      const uploadData = new FormData();
      
      // Add image file if provided
      if (imageFile) {
        uploadData.append("image", imageFile);
      }
      
      // Add end image file if provided
      if (endImageFile) {
        uploadData.append("end_image", endImageFile);
      }
      
      // Add other form fields
      uploadData.append("model_type", formData.modelType);
      uploadData.append("prompt", formData.prompt);
      uploadData.append("resolution", formData.resolution);
      uploadData.append("steps", formData.steps.toString());
      
      // Always append lora_path, even if empty
      uploadData.append("lora_path", formData.loraPath || "");
      uploadData.append("lora_strength", formData.loraStrength.toString());
      
      // Debug log
      console.log("Sending FormData with:", {
        modelType: formData.modelType,
        prompt: formData.prompt,
        resolution: formData.resolution,
        steps: formData.steps,
        uploadDataType: uploadData.constructor.name
      });
      
      // Submit to API with multipart data and extended timeout
      const response = await post<GenerationResponse>("/generation/submit", uploadData, {
        headers: {
          // Don't set Content-Type manually for FormData - let axios set it with boundary
        },
        timeout: 300000, // 5 minutes for generation submission
      });
      
      // Validate response
      return validateApiResponse(GenerationResponseSchema, response);
    },
    onSuccess: (data) => {
      // Invalidate queue queries to refresh the queue
      queryClient.invalidateQueries(["queue"]);
      queryClient.invalidateQueries(["queue-status"]);
      
      console.log("Generation request submitted:", data);
    },
    onError: (error: ApiError) => {
      console.error("Generation request failed:", error);
    },
  });
};

// Hook for submitting generation requests with file upload (for I2V/TI2V)
export const useGenerateVideoWithImage = () => {
  const queryClient = useQueryClient();

  return useMutation<GenerationResponse, ApiError, { formData: GenerationFormData; imageFile: File }>({
    mutationFn: async ({ formData, imageFile }) => {
      // Create FormData for multipart upload
      const uploadData = new FormData();
      
      // Add image file
      uploadData.append("image", imageFile);
      
      // Add other form fields
      uploadData.append("model_type", formData.modelType);
      uploadData.append("prompt", formData.prompt);
      uploadData.append("resolution", formData.resolution);
      uploadData.append("steps", formData.steps.toString());
      
      // Always append lora_path, even if empty
      uploadData.append("lora_path", formData.loraPath || "");
      uploadData.append("lora_strength", formData.loraStrength.toString());
      
      // Submit to API with multipart data and extended timeout
      const response = await post<GenerationResponse>("/generation/submit", uploadData, {
        headers: {
          // Don't set Content-Type manually for FormData - let axios set it with boundary
        },
        timeout: 300000, // 5 minutes for generation submission
      });
      
      // Validate response
      return validateApiResponse(GenerationResponseSchema, response);
    },
    onSuccess: (data) => {
      // Invalidate queue queries to refresh the queue
      queryClient.invalidateQueries(["queue"]);
      queryClient.invalidateQueries(["queue-status"]);
      
      console.log("Generation request with image submitted:", data);
    },
    onError: (error: ApiError) => {
      console.error("Generation request with image failed:", error);
    },
  });
};

// Estimated generation times based on resolution and model type
export const getEstimatedGenerationTime = (
  modelType: string,
  resolution: string,
  steps: number
): number => {
  // Base times in minutes for different resolutions (T2V-A14B on RTX 4080)
  const baseTimes: Record<string, number> = {
    "854x480": 2,    // 480p - 2 minutes
    "1024x576": 3,   // 576p - 3 minutes  
    "1280x720": 6,   // 720p - 6 minutes
    "1920x1080": 17, // 1080p - 17 minutes
  };

  const baseTime = baseTimes[resolution] || baseTimes["1280x720"];
  
  // Adjust for steps (50 is baseline)
  const stepsMultiplier = steps / 50;
  
  // Adjust for model type
  let modelMultiplier = 1;
  if (modelType === "I2V-A14B") {
    modelMultiplier = 0.8; // I2V is slightly faster
  } else if (modelType === "TI2V-5B") {
    modelMultiplier = 1.2; // TI2V is slightly slower
  }
  
  return Math.round(baseTime * stepsMultiplier * modelMultiplier);
};

// Hook to get estimated generation time
export const useEstimatedTime = (formData: GenerationFormData) => {
  return getEstimatedGenerationTime(
    formData.modelType,
    formData.resolution,
    formData.steps
  );
};