import { z } from "zod";

/**
 * Model type options supported by the backend generation pipeline.
 */
export const ModelTypeSchema = z.enum([
  "T2V-A14B",
  "I2V-A14B",
  "TI2V-5B",
]);
export type ModelType = z.infer<typeof ModelTypeSchema>;

/**
 * Task lifecycle states that may be returned by queue endpoints.
 */
export const TaskStatusSchema = z.enum([
  "pending",
  "processing",
  "completed",
  "failed",
  "cancelled",
]);
export type TaskStatus = z.infer<typeof TaskStatusSchema>;

const ResolutionSchema = z
  .string()
  .regex(/^[0-9]+x[0-9]+$/, "Resolution must follow WIDTHxHEIGHT format");

const LoRAPathSchema = z
  .union([
    z.string().trim().min(1, "LoRA path cannot be empty"),
    z.literal(""),
  ])
  .optional()
  .transform((value) => {
    if (!value) {
      return undefined;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  });

/**
 * Schema for the generation form used throughout the UI. Includes support for
 * the "auto" placeholder model which is resolved client-side before
 * submission.
 */
export const GenerationFormSchema = z.object({
  modelType: z.union([ModelTypeSchema, z.literal("auto")]),
  prompt: z
    .string()
    .trim()
    .min(1, "Prompt is required")
    .max(500, "Prompt must be 500 characters or less"),
  resolution: ResolutionSchema,
  steps: z
    .number({ invalid_type_error: "Steps must be a number" })
    .int("Steps must be an integer")
    .min(1, "Steps must be at least 1")
    .max(100, "Steps must be 100 or less"),
  loraPath: LoRAPathSchema,
  loraStrength: z
    .number({ invalid_type_error: "LoRA strength must be a number" })
    .min(0, "LoRA strength must be at least 0")
    .max(2, "LoRA strength must be at most 2"),
});
export type GenerationFormData = z.infer<typeof GenerationFormSchema>;

/**
 * Schema describing the response returned when a generation task is queued.
 */
export const GenerationResponseSchema = z.object({
  task_id: z.string(),
  status: TaskStatusSchema,
  message: z.string(),
  estimated_time_minutes: z.number().int().nonnegative().optional().nullable(),
});
export type GenerationResponse = z.infer<typeof GenerationResponseSchema>;

/**
 * Task information as returned from the queue endpoints.
 */
export const TaskInfoSchema = z.object({
  id: z.string(),
  model_type: ModelTypeSchema,
  prompt: z.string(),
  image_path: z.string().optional().nullable(),
  resolution: ResolutionSchema,
  steps: z.number().int(),
  lora_path: z.string().optional().nullable(),
  lora_strength: z.number(),
  status: TaskStatusSchema,
  progress: z.number().int().min(0).max(100),
  created_at: z.string(),
  started_at: z.string().optional().nullable(),
  completed_at: z.string().optional().nullable(),
  output_path: z.string().optional().nullable(),
  error_message: z.string().optional().nullable(),
  estimated_time_minutes: z.number().int().optional().nullable(),
});
export type TaskInfo = z.infer<typeof TaskInfoSchema>;

/**
 * Schema for queue overview responses that bundle aggregate counts with task
 * details.
 */
export const QueueStatusSchema = z.object({
  total_tasks: z.number().int().nonnegative(),
  pending_tasks: z.number().int().nonnegative(),
  processing_tasks: z.number().int().nonnegative(),
  completed_tasks: z.number().int().nonnegative(),
  failed_tasks: z.number().int().nonnegative(),
  cancelled_tasks: z.number().int().nonnegative().optional().default(0),
  tasks: z.array(TaskInfoSchema),
});
export type QueueStatus = z.infer<typeof QueueStatusSchema>;

/**
 * LoRA metadata shared across multiple responses.
 */
export const LoRAInfoSchema = z.object({
  name: z.string(),
  filename: z.string(),
  path: z.string(),
  size_mb: z.number().nonnegative(),
  modified_time: z.string(),
  is_loaded: z.boolean().optional().default(false),
  is_applied: z.boolean().optional().default(false),
  current_strength: z.number().optional().default(0),
});
export type LoRAInfo = z.infer<typeof LoRAInfoSchema>;

export const LoRAListResponseSchema = z.object({
  loras: z.array(LoRAInfoSchema),
  total_count: z.number().int().nonnegative(),
  total_size_mb: z.number().nonnegative(),
});
export type LoRAListResponse = z.infer<typeof LoRAListResponseSchema>;

export const LoRAUploadResponseSchema = z.object({
  success: z.boolean(),
  message: z.string(),
  lora_name: z.string(),
  file_path: z.string(),
  size_mb: z.number().nonnegative(),
});
export type LoRAUploadResponse = z.infer<typeof LoRAUploadResponseSchema>;

export const LoRAStatusResponseSchema = z.object({
  name: z.string(),
  exists: z.boolean(),
  path: z.string().optional().nullable(),
  size_mb: z.number().optional().nullable(),
  is_loaded: z.boolean(),
  is_applied: z.boolean(),
  current_strength: z.number(),
  modified_time: z.string().optional().nullable(),
});
export type LoRAStatusResponse = z.infer<typeof LoRAStatusResponseSchema>;

export const LoRAPreviewResponseSchema = z.object({
  lora_name: z.string(),
  base_prompt: z.string(),
  enhanced_prompt: z.string(),
  style_indicators: z.array(z.string()),
  preview_note: z.string(),
});
export type LoRAPreviewResponse = z.infer<typeof LoRAPreviewResponseSchema>;

const MemoryImpactLevelSchema = z.enum(["low", "medium", "high"]);

export const LoRAMemoryImpactResponseSchema = z.object({
  lora_name: z.string(),
  file_size_mb: z.number().nonnegative(),
  estimated_memory_mb: z.number().nonnegative(),
  vram_available_mb: z.number().nonnegative(),
  can_load: z.boolean(),
  memory_impact: MemoryImpactLevelSchema,
  recommendation: z.string(),
});
export type LoRAMemoryImpactResponse = z.infer<
  typeof LoRAMemoryImpactResponseSchema
>;

/**
 * Maximum LoRA upload size enforced client-side (mirrors backend limit).
 */
export const MAX_LORA_FILE_SIZE_BYTES = 500 * 1024 * 1024; // 500MB

const SUPPORTED_LORA_EXTENSIONS = new Set([
  ".safetensors",
  ".pt",
  ".pth",
  ".bin",
]);

const BaseFileSchema: z.ZodType<File> =
  typeof File !== "undefined"
    ? z.instanceof(File)
    : z.custom<File>((value) => {
        return (
          typeof value === "object" &&
          value !== null &&
          "name" in value &&
          "size" in value &&
          typeof (value as { name: unknown }).name === "string" &&
          typeof (value as { size: unknown }).size === "number"
        );
      }, {
        message: "A valid file is required",
      });

const LoRAFileSchema = BaseFileSchema.superRefine((file, ctx) => {
  if (file.size <= 0) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "File cannot be empty",
    });
  }

  if (file.size > MAX_LORA_FILE_SIZE_BYTES) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "File must be 500MB or smaller",
    });
  }

  const extension = file.name.includes(".")
    ? `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`
    : "";

  if (!SUPPORTED_LORA_EXTENSIONS.has(extension)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Unsupported file format",
    });
  }
});

export const LoRAUploadSchema = z.object({
  file: LoRAFileSchema,
  name: z
    .string()
    .trim()
    .max(100, "Name must be 100 characters or fewer")
    .optional()
    .transform((value) => (value && value.length > 0 ? value : undefined)),
});
export type LoRAUploadInput = z.infer<typeof LoRAUploadSchema>;

/**
 * Validate LoRA upload inputs and return the parsed value for convenience.
 */
export const validateLoRAUpload = (input: LoRAUploadInput) =>
  LoRAUploadSchema.parse(input);

/**
 * Utility helper that validates API responses against the expected schema.
 */
export const validateApiResponse = <T>(schema: z.ZodType<T>, data: unknown): T => {
  const result = schema.safeParse(data);
  if (!result.success) {
    throw result.error;
  }
  return result.data;
};
