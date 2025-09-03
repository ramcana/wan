import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Loader2,
  Settings,
  Wand2,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import {
  GenerationFormSchema,
  type GenerationFormData,
  type ModelType,
} from "@/lib/api-schemas";
import { cn } from "@/lib/utils";
import { ErrorDisplay } from "./ErrorDisplay";
import { ImageUpload } from "./ImageUpload";
import { ApiError } from "@/lib/api-client";
import PromptEnhancer from "@/components/prompt/PromptEnhancer";
import { LoRASelector } from "./LoRASelector";

interface GenerationFormProps {
  onSubmit: (
    data: GenerationFormData,
    imageFile?: File,
    endImageFile?: File
  ) => Promise<void>;
  isSubmitting?: boolean;
  error?: string | ApiError | Error | null;
  onRetry?: () => void;
}

// Resolution options with visual indicators
const RESOLUTION_OPTIONS = [
  {
    value: "1280x720",
    label: "720p HD",
    description: "1280 × 720",
    recommended: true,
  },
  {
    value: "1920x1080",
    label: "1080p Full HD",
    description: "1920 × 1080",
    premium: true,
  },
  { value: "854x480", label: "480p SD", description: "854 × 480", fast: true },
  { value: "1024x576", label: "576p", description: "1024 × 576" },
];

// Enhanced model options for Phase 1 MVP with auto-detection
const MODEL_OPTIONS = [
  {
    value: "auto" as ModelType | "auto",
    label: "Auto-Detect (Recommended)",
    description: "Automatically choose the best model based on your inputs",
    badge: "AUTO",
    requiresImage: false,
    isAutoDetect: true,
  },
  {
    value: "T2V-A14B" as ModelType,
    label: "Text-to-Video",
    description: "Generate videos from text prompts only",
    badge: "T2V",
    requiresImage: false,
    estimatedTime: "2-3 minutes",
    vramUsage: "~8GB",
  },
  {
    value: "I2V-A14B" as ModelType,
    label: "Image-to-Video",
    description: "Animate images into videos",
    badge: "I2V",
    requiresImage: true,
    estimatedTime: "2.5-3.5 minutes",
    vramUsage: "~8.5GB",
  },
  {
    value: "TI2V-5B" as ModelType,
    label: "Text + Image to Video",
    description: "Combine text and images for guided generation",
    badge: "TI2V",
    requiresImage: true,
    estimatedTime: "1.5-2.5 minutes",
    vramUsage: "~6GB",
  },
];

// Steps presets
const STEPS_PRESETS = [
  { value: 25, label: "Fast", description: "Quick generation" },
  { value: 50, label: "Balanced", description: "Good quality/speed balance" },
  { value: 75, label: "High Quality", description: "Better quality, slower" },
];

export const GenerationForm: React.FC<GenerationFormProps> = ({
  onSubmit,
  isSubmitting = false,
  error = null,
  onRetry,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [promptLength, setPromptLength] = useState(0);
  const [isFormValid, setIsFormValid] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedEndImage, setSelectedEndImage] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [endImageError, setEndImageError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    trigger,
    clearErrors,
    formState: { errors, isValid, isDirty, isValidating },
  } = useForm<GenerationFormData>({
    resolver: zodResolver(GenerationFormSchema) as any,
    defaultValues: {
      modelType: "auto" as any, // Phase 1: Default to auto-detection
      prompt: "",
      resolution: "1280x720",
      steps: 50,
      loraStrength: 1.0,
    },
    mode: "onChange",
    reValidateMode: "onChange",
  });

  const watchedPrompt = watch("prompt");
  const watchedModelType = watch("modelType");
  const watchedResolution = watch("resolution");
  const watchedSteps = watch("steps");

  // Get current model option
  const currentModelOption = MODEL_OPTIONS.find(
    (opt) => opt.value === watchedModelType
  );

  // Phase 1: Auto-detection state
  const [detectedModel, setDetectedModel] = useState<string | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number>(0);
  const [detectionExplanation, setDetectionExplanation] = useState<string[]>(
    []
  );
  const [isDetecting, setIsDetecting] = useState(false);

  // Update prompt length and validation when prompt changes
  React.useEffect(() => {
    setPromptLength(watchedPrompt?.length || 0);
  }, [watchedPrompt]);

  // Phase 1: Auto-detection logic
  React.useEffect(() => {
    const runAutoDetection = async () => {
      if (
        watchedModelType === "auto" &&
        watchedPrompt &&
        watchedPrompt.length > 3
      ) {
        setIsDetecting(true);
        try {
          // Call auto-detection API
          const response = await fetch(
            `/api/v1/generation/models/detect?prompt=${encodeURIComponent(
              watchedPrompt
            )}&has_image=${selectedImage !== null}&has_end_image=${
              selectedEndImage !== null
            }`
          );
          if (response.ok) {
            const data = await response.json();
            setDetectedModel(data.detected_model_type);
            setDetectionConfidence(data.confidence);
            setDetectionExplanation(data.explanation || []);
          }
        } catch (error) {
          console.warn("Auto-detection failed:", error);
        } finally {
          setIsDetecting(false);
        }
      } else if (watchedModelType !== "auto") {
        // Clear detection when manual model is selected
        setDetectedModel(null);
        setDetectionConfidence(0);
        setDetectionExplanation([]);
      }
    };

    const timeoutId = setTimeout(runAutoDetection, 500); // Debounce
    return () => clearTimeout(timeoutId);
  }, [watchedPrompt, watchedModelType, selectedImage, selectedEndImage]);

  // Update form validation state with auto-detection support
  React.useEffect(() => {
    // For auto-detection, use detected model requirements
    const effectiveModelType =
      watchedModelType === "auto" ? detectedModel : watchedModelType;
    const effectiveModelOption = MODEL_OPTIONS.find(
      (opt) => opt.value === effectiveModelType
    );

    // Check if image is required and provided
    const imageRequired = effectiveModelOption?.requiresImage || false;
    const imageProvided = selectedImage !== null;
    const imageValid = !imageRequired || imageProvided;

    // For auto mode, require either detection completion or manual override
    const autoDetectionValid =
      watchedModelType !== "auto" || detectedModel !== null;

    setIsFormValid(
      isValid && !isValidating && imageValid && autoDetectionValid
    );

    // Collect validation errors for display
    const errorMessages: string[] = [];
    if (errors.prompt)
      errorMessages.push(String(errors.prompt.message) || "Invalid prompt");
    if (errors.steps)
      errorMessages.push(String(errors.steps.message) || "Invalid steps value");
    if (errors.modelType)
      errorMessages.push(
        String(errors.modelType.message) || "Invalid model type"
      );
    if (errors.resolution)
      errorMessages.push(
        String(errors.resolution.message) || "Invalid resolution"
      );
    if (imageRequired && !imageProvided)
      errorMessages.push("Image is required for this model type");
    if (imageError) errorMessages.push(imageError);
    if (endImageError) errorMessages.push(endImageError);
    if (
      watchedModelType === "auto" &&
      !detectedModel &&
      watchedPrompt &&
      watchedPrompt.length > 3
    )
      errorMessages.push("Auto-detection in progress...");

    setValidationErrors(errorMessages);
  }, [
    isValid,
    isValidating,
    errors,
    currentModelOption,
    selectedImage,
    imageError,
    selectedEndImage,
    endImageError,
    watchedModelType,
    detectedModel,
    watchedPrompt,
  ]);

  // Auto-validate on field changes with debouncing
  React.useEffect(() => {
    if (isDirty) {
      const timeoutId = setTimeout(() => {
        trigger();
      }, 300);
      return () => clearTimeout(timeoutId);
    }
  }, [watchedPrompt, watchedSteps, isDirty, trigger]);

  const handleFormSubmit = async (data: GenerationFormData) => {
    try {
      // Clear any previous errors
      clearErrors();
      setImageError(null);
      setEndImageError(null);

      // Additional client-side validation
      if (!data.prompt.trim()) {
        throw new Error("Prompt cannot be empty");
      }

      if (data.prompt.length > 500) {
        throw new Error("Prompt must be 500 characters or less");
      }

      if (data.steps < 1 || data.steps > 100) {
        throw new Error("Steps must be between 1 and 100");
      }

      // Handle auto-detection: use detected model if in auto mode
      const finalModelType =
        data.modelType === "auto" ? detectedModel : data.modelType;
      if (data.modelType === "auto" && !detectedModel) {
        throw new Error(
          "Auto-detection is still in progress. Please wait or select a model manually."
        );
      }

      // Check image requirements for the final model type
      const finalModelOption = MODEL_OPTIONS.find(
        (opt) => opt.value === finalModelType
      );
      const imageRequired = finalModelOption?.requiresImage || false;
      if (imageRequired && !selectedImage) {
        throw new Error(`Image is required for ${finalModelType} model`);
      }

      // Create final form data with detected model type
      const finalData = {
        ...data,
        modelType: finalModelType as any,
      };

      // Submit the form with optional images
      await onSubmit(
        finalData,
        selectedImage || undefined,
        selectedEndImage || undefined
      );
    } catch (err) {
      // Error handling is managed by parent component
      console.error("Form submission error:", err);
      throw err; // Re-throw to let parent handle it
    }
  };

  const handleRetry = () => {
    if (onRetry) {
      onRetry();
    } else {
      // If no retry handler, try to resubmit the form
      const currentData = {
        modelType: watchedModelType,
        prompt: watchedPrompt,
        resolution: watchedResolution,
        steps: watchedSteps,
        loraStrength: 1.0,
      };
      handleFormSubmit(currentData);
    }
  };

  const getResolutionOption = (value: string) => {
    return RESOLUTION_OPTIONS.find((opt) => opt.value === value);
  };

  const getStepsPreset = (value: number) => {
    return STEPS_PRESETS.find((preset) => preset.value === value);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Wand2 className="h-5 w-5" />
          Video Generation
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form
          onSubmit={handleSubmit(handleFormSubmit as any)}
          className="space-y-6"
        >
          {/* Error Display */}
          {error && (
            <ErrorDisplay
              error={error}
              onRetry={handleRetry}
              className="mb-4"
            />
          )}

          {/* Validation Errors */}
          {validationErrors.length > 0 && !error && (
            <Alert variant="warning" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <p className="font-medium mb-2">
                  Please fix the following issues:
                </p>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  {validationErrors.map((errorMsg, index) => (
                    <li key={index}>{String(errorMsg)}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {/* Success Indicator */}
          {isFormValid && isDirty && !error && (
            <Alert variant="success" className="mb-4">
              <CheckCircle2 className="h-4 w-4" />
              <AlertDescription>Form is ready for submission</AlertDescription>
            </Alert>
          )}

          {/* Model Type Selector */}
          <div className="space-y-2">
            <Label htmlFor="modelType">Model Type</Label>
            <Select
              value={watchedModelType}
              onValueChange={(value: ModelType) => setValue("modelType", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select model type" />
              </SelectTrigger>
              <SelectContent>
                {MODEL_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        {option.badge}
                      </Badge>
                      <div>
                        <div className="font-medium">{option.label}</div>
                        <div className="text-xs text-muted-foreground">
                          {option.description}
                        </div>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.modelType && (
              <p className="text-sm text-destructive">
                {String(errors.modelType.message || "Invalid model type")}
              </p>
            )}

            {/* Auto-Detection Feedback */}
            {watchedModelType === "auto" && (
              <div className="mt-2 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-md border border-blue-200 dark:border-blue-800">
                {isDetecting ? (
                  <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700 dark:border-blue-300"></div>
                    <span className="text-sm">Analyzing your inputs...</span>
                  </div>
                ) : detectedModel ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-green-700 dark:text-green-300">
                      <CheckCircle2 className="h-4 w-4" />
                      <span className="text-sm font-medium">
                        Detected: {detectedModel} (
                        {Math.round(detectionConfidence * 100)}% confidence)
                      </span>
                    </div>
                    {detectionExplanation.length > 0 && (
                      <div className="text-xs text-blue-600 dark:text-blue-400">
                        {detectionExplanation.map((explanation, index) => (
                          <div key={index}>• {explanation}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : watchedPrompt && watchedPrompt.length > 3 ? (
                  <div className="flex items-center gap-2 text-yellow-700 dark:text-yellow-300">
                    <AlertCircle className="h-4 w-4" />
                    <span className="text-sm">
                      Enter more details for better detection
                    </span>
                  </div>
                ) : (
                  <div className="text-sm text-blue-600 dark:text-blue-400">
                    Start typing your prompt to automatically detect the best
                    model
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Prompt Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="prompt">Prompt</Label>
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "text-xs",
                    promptLength > 450
                      ? "text-destructive"
                      : promptLength > 400
                      ? "text-yellow-600"
                      : "text-muted-foreground"
                  )}
                >
                  {promptLength}/500
                </span>
              </div>
            </div>
            <Textarea
              {...register("prompt")}
              placeholder="Describe the video you want to generate..."
              className="min-h-[100px] resize-none"
              maxLength={500}
            />
            {errors.prompt && (
              <p className="text-sm text-destructive">
                {String(errors.prompt.message || "Invalid prompt")}
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              Be descriptive and specific for better results. Include details
              about style, mood, and visual elements.
            </p>
          </div>

          {/* Prompt Enhancement */}
          <PromptEnhancer
            initialPrompt={watchedPrompt}
            onPromptChange={(prompt) => setValue("prompt", prompt)}
            onEnhancementApplied={(enhancedPrompt) => {
              setValue("prompt", enhancedPrompt);
              trigger("prompt");
            }}
          />

          {/* Image Upload (for I2V and TI2V) */}
          {currentModelOption?.requiresImage && (
            <div className="space-y-4">
              {/* Start/Source Image */}
              <div className="space-y-2">
                <Label htmlFor="image">
                  {watchedModelType === "I2V-A14B"
                    ? "Source Image"
                    : watchedModelType === "TI2V-5B"
                    ? "Start Image"
                    : "Reference Image"}
                </Label>
                <ImageUpload
                  onImageSelect={(file) => {
                    setSelectedImage(file);
                    setImageError(null);
                  }}
                  selectedImage={selectedImage}
                  error={imageError}
                  disabled={isSubmitting}
                />
                <p className="text-xs text-muted-foreground">
                  {watchedModelType === "I2V-A14B"
                    ? "Upload an image to generate a video from. The video will animate this image."
                    : watchedModelType === "TI2V-5B"
                    ? "Upload the starting frame image for your video generation."
                    : "Upload a reference image to guide the video generation along with your text prompt."}
                </p>
              </div>

              {/* End Image (for I2V and TI2V) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="endImage">
                    {watchedModelType === "I2V-A14B"
                      ? "End Image (Optional)"
                      : "End Image (Optional)"}
                  </Label>
                  <Badge variant="secondary" className="text-xs">
                    Optional
                  </Badge>
                </div>
                <ImageUpload
                  onImageSelect={(file) => {
                    setSelectedEndImage(file);
                    setEndImageError(null);
                  }}
                  selectedImage={selectedEndImage}
                  error={endImageError}
                  disabled={isSubmitting}
                />
                <p className="text-xs text-muted-foreground">
                  {watchedModelType === "I2V-A14B"
                    ? "Upload an optional end frame to control how the video should end."
                    : "Upload an optional end frame to define the final state of your video."}
                </p>
              </div>
            </div>
          )}

          {/* Resolution Selector */}
          <div className="space-y-2">
            <Label htmlFor="resolution">Resolution</Label>
            <Select
              value={watchedResolution}
              onValueChange={(value) => setValue("resolution", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select resolution" />
              </SelectTrigger>
              <SelectContent>
                {RESOLUTION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex items-center justify-between w-full">
                      <div>
                        <div className="font-medium">{option.label}</div>
                        <div className="text-xs text-muted-foreground">
                          {option.description}
                        </div>
                      </div>
                      <div className="flex gap-1">
                        {option.recommended && (
                          <Badge variant="default" className="text-xs">
                            Recommended
                          </Badge>
                        )}
                        {option.premium && (
                          <Badge variant="secondary" className="text-xs">
                            Premium
                          </Badge>
                        )}
                        {option.fast && (
                          <Badge variant="outline" className="text-xs">
                            Fast
                          </Badge>
                        )}
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.resolution && (
              <p className="text-sm text-destructive">
                {String(errors.resolution.message || "Invalid resolution")}
              </p>
            )}
            {getResolutionOption(watchedResolution) && (
              <p className="text-xs text-muted-foreground">
                Higher resolutions produce better quality but take longer to
                generate.
              </p>
            )}
          </div>

          {/* Advanced Settings Toggle */}
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2"
            >
              <Settings className="h-4 w-4" />
              Advanced Settings
            </Button>
          </div>

          {/* Advanced Settings Panel */}
          {showAdvanced && (
            <Card className="bg-muted/50">
              <CardContent className="pt-6 space-y-4">
                {/* Steps Selector */}
                <div className="space-y-2">
                  <Label htmlFor="steps">Generation Steps</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {STEPS_PRESETS.map((preset) => (
                      <Button
                        key={preset.value}
                        type="button"
                        variant={
                          watchedSteps === preset.value ? "default" : "outline"
                        }
                        size="sm"
                        onClick={() => setValue("steps", preset.value)}
                        className="flex flex-col h-auto py-3"
                      >
                        <span className="font-medium">{preset.label}</span>
                        <span className="text-xs opacity-70">
                          {preset.value} steps
                        </span>
                      </Button>
                    ))}
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      {...register("steps", { valueAsNumber: true })}
                      type="number"
                      min={1}
                      max={100}
                      className="w-20"
                    />
                    <span className="text-sm text-muted-foreground">
                      Custom steps (1-100)
                    </span>
                  </div>
                  {errors.steps && (
                    <p className="text-sm text-destructive">
                      {String(errors.steps.message || "Invalid steps value")}
                    </p>
                  )}
                  {getStepsPreset(watchedSteps) && (
                    <p className="text-xs text-muted-foreground">
                      {getStepsPreset(watchedSteps)?.description}
                    </p>
                  )}
                </div>

                {/* LoRA Selection */}
                <div className="space-y-2">
                  <Label>LoRA (Style Enhancement)</Label>
                  <LoRASelector
                    selectedLoRA={watch("loraPath")}
                    onLoRASelect={(loraPath) => setValue("loraPath", loraPath)}
                    onLoRAStrengthChange={(strength) =>
                      setValue("loraStrength", strength)
                    }
                    loraStrength={watch("loraStrength")}
                  />
                </div>

                {/* Quantization Options */}
                <div className="space-y-2">
                  <Label>Quantization (VRAM Optimization)</Label>
                  <div className="grid grid-cols-3 gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="flex flex-col h-auto py-3"
                      disabled
                    >
                      <span className="font-medium">FP16</span>
                      <span className="text-xs opacity-70">High Quality</span>
                    </Button>
                    <Button
                      type="button"
                      variant="default"
                      size="sm"
                      className="flex flex-col h-auto py-3"
                    >
                      <span className="font-medium">BF16</span>
                      <span className="text-xs opacity-70">Balanced</span>
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="flex flex-col h-auto py-3"
                      disabled
                    >
                      <span className="font-medium">INT8</span>
                      <span className="text-xs opacity-70">Memory Saver</span>
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    BF16 is recommended for RTX 4080. Lower precision saves VRAM
                    but may reduce quality.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Submit Button */}
          <div className="flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              {isSubmitting && (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Submitting to queue...
                </span>
              )}
              {!isSubmitting && !isFormValid && isDirty && (
                <span className="flex items-center gap-2 text-yellow-600">
                  <AlertCircle className="h-3 w-3" />
                  Please fix validation errors
                </span>
              )}
              {!isSubmitting && isFormValid && isDirty && (
                <span className="flex items-center gap-2 text-green-600">
                  <CheckCircle2 className="h-3 w-3" />
                  Ready to generate
                </span>
              )}
            </div>

            <Button
              type="submit"
              disabled={!isFormValid || isSubmitting}
              className={cn(
                "min-w-[140px] transition-all duration-200",
                isFormValid &&
                  !isSubmitting &&
                  "bg-green-600 hover:bg-green-700"
              )}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Wand2 className="mr-2 h-4 w-4" />
                  Generate Video
                </>
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};

export default GenerationForm;
