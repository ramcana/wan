import { z } from 'zod'

// Task Status Enum
export enum TaskStatus {
  PENDING = 'pending',
  PROCESSING = 'processing', 
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

// Generation Request Schema
export const GenerationFormSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  model_type: z.enum(['t2v', 'i2v', 'ti2v']),
  num_frames: z.number().min(1).max(240).default(120),
  fps: z.number().min(1).max(60).default(24),
  width: z.number().min(64).max(2048).default(1024),
  height: z.number().min(64).max(2048).default(576),
  guidance_scale: z.number().min(1).max(20).default(7.5),
  num_inference_steps: z.number().min(1).max(100).default(50),
  seed: z.number().optional(),
  image_file: z.any().optional(), // File input
  video_file: z.any().optional(), // File input
})

export type GenerationFormData = z.infer<typeof GenerationFormSchema>

// Image Upload Validation
export const ImageUploadSchema = z.object({
  type: z.string().refine(
    (type) => ['image/jpeg', 'image/png', 'image/webp'].includes(type),
    { message: 'Only JPEG, PNG, and WebP images are supported' }
  ),
  size: z.number().max(10 * 1024 * 1024, 'File size must be less than 10MB'), // 10MB
})

export type ImageUpload = z.infer<typeof ImageUploadSchema>

export function validateImageUpload(file: File): void {
  const result = ImageUploadSchema.safeParse({
    type: file.type,
    size: file.size,
  });
  
  if (!result.success) {
    throw new Error(result.error.errors[0].message);
  }
}

// LoRA Upload Validation
export const LoRAUploadSchema = z.object({
  file: z.object({
    type: z.string().refine(
      (type) => ['.safetensors', '.pt', '.pth', '.bin'].some(ext => type.endsWith(ext)),
      { message: 'Only .safetensors, .pt, .pth, and .bin files are supported' }
    ),
    size: z.number().max(500 * 1024 * 1024, 'File size must be less than 500MB'), // 500MB
  }),
  name: z.string().max(100, 'Name must be less than 100 characters').optional(),
})

export type LoRAUpload = z.infer<typeof LoRAUploadSchema>

export function validateLoRAUpload(data: { file: File; name?: string }): void {
  const result = LoRAUploadSchema.safeParse({
    file: {
      type: data.file.name,
      size: data.file.size,
    },
    name: data.name,
  });
  
  if (!result.success) {
    throw new Error(result.error.errors[0].message);
  }
}

// Generation Response Schema
export const GenerationResponseSchema = z.object({
  task_id: z.string(),
  status: z.nativeEnum(TaskStatus),
  message: z.string(),
  estimated_time: z.number().optional(),
})

export type GenerationResponse = z.infer<typeof GenerationResponseSchema>

// Task Info Schema
export const TaskInfoSchema = z.object({
  id: z.string(),
  status: z.nativeEnum(TaskStatus),
  prompt: z.string(),
  model_type: z.enum(['t2v', 'i2v', 'ti2v']),
  created_at: z.string(),
  started_at: z.string().optional(),
  completed_at: z.string().optional(),
  progress: z.number().min(0).max(100).default(0),
  estimated_time_remaining: z.number().optional(),
  output_path: z.string().optional(),
  error_message: z.string().optional(),
  parameters: z.record(z.any()).optional(),
})

export type TaskInfo = z.infer<typeof TaskInfoSchema>

// Queue Status Schema
export const QueueStatusSchema = z.object({
  total_tasks: z.number(),
  pending_tasks: z.number(),
  processing_tasks: z.number(),
  completed_tasks: z.number(),
  failed_tasks: z.number(),
  cancelled_tasks: z.number(),
  current_task: TaskInfoSchema.optional(),
  queue: z.array(TaskInfoSchema),
})

export type QueueStatus = z.infer<typeof QueueStatusSchema>

// System Health Schema - Updated to match actual backend response
export const SystemHealthSchema = z.object({
  status: z.string(), // Changed from enum to string to accept "ok"
  port: z.number().optional(), // Added missing field
  timestamp: z.string().optional(), // Added missing field
  api_version: z.string().optional(), // Added missing field
  system: z.string().optional(), // Added missing field
  service: z.string().optional(), // Added missing field
  endpoints: z.object({
    health: z.string(),
    docs: z.string(),
    websocket: z.string(),
    api_base: z.string(),
  }).optional(), // Added missing field
  connectivity: z.object({
    cors_enabled: z.boolean().optional(),
    allowed_origins: z.array(z.string()).optional(),
    websocket_available: z.boolean().optional(),
    request_origin: z.string().optional(),
    host_header: z.string().optional(),
  }).optional(), // Added missing field
  server_info: z.object({
    configured_port: z.number().optional(),
    detected_port: z.number().optional(),
    environment: z.string().optional(),
  }).optional(), // Added missing field
  // Made these fields optional since they're not in the actual response
  backend_online: z.boolean().optional(),
  database_connected: z.boolean().optional(),
  gpu_available: z.boolean().optional(),
  memory_usage: z.number().optional(),
  gpu_memory_usage: z.number().optional(),
  active_tasks: z.number().optional(),
  uptime: z.number().optional(),
})

export type SystemHealth = z.infer<typeof SystemHealthSchema>

// Port Detection Response Schema
export const PortDetectionResponseSchema = z.object({
  backend_port: z.number(),
  frontend_port: z.number(),
  status: z.string(),
})

export type PortDetectionResponse = z.infer<typeof PortDetectionResponseSchema>

// API Response Validation Helper
export function validateApiResponse<T>(schema: z.ZodSchema<T>, data: unknown): T {
  try {
    return schema.parse(data)
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new Error(`API response validation failed: ${error.errors.map(e => e.message).join(', ')}`)
    }
    throw error
  }
}

// Output File Schema
export const OutputFileSchema = z.object({
  id: z.string(),
  filename: z.string(),
  path: z.string(),
  size: z.number(),
  created_at: z.string(),
  task_id: z.string().optional(),
  thumbnail_path: z.string().optional(),
  duration: z.number().optional(),
  width: z.number().optional(),
  height: z.number().optional(),
})

export type OutputFile = z.infer<typeof OutputFileSchema>

// Prompt Enhancement Schemas
export const PromptEnhanceRequestSchema = z.object({
  prompt: z.string().min(1),
  style: z.enum(['cinematic', 'artistic', 'realistic', 'anime']).optional(),
  enhance_creativity: z.boolean().default(true),
})

export type PromptEnhanceRequest = z.infer<typeof PromptEnhanceRequestSchema>

export const PromptEnhanceResponseSchema = z.object({
  enhanced_prompt: z.string(),
  suggested_model: z.enum(['t2v', 'i2v', 'ti2v']).optional(),
  confidence: z.number().min(0).max(1).optional(),
})

export type PromptEnhanceResponse = z.infer<typeof PromptEnhanceResponseSchema>

// LoRA API Response Schemas
export const LoRAInfoSchema = z.object({
  name: z.string(),
  filename: z.string(),
  path: z.string(),
  size_mb: z.number(),
  is_loaded: z.boolean(),
  is_applied: z.boolean(),
  created_at: z.string(),
  last_used: z.string().optional(),
  compatible_models: z.array(z.string()),
  description: z.string().optional(),
})

export type LoRAInfo = z.infer<typeof LoRAInfoSchema>

export const LoRAListResponseSchema = z.object({
  loras: z.array(LoRAInfoSchema),
  total_count: z.number(),
  total_size_mb: z.number(),
})

export type LoRAListResponse = z.infer<typeof LoRAListResponseSchema>

export const LoRAUploadResponseSchema = z.object({
  name: z.string(),
  filename: z.string(),
  path: z.string(),
  size_mb: z.number(),
  message: z.string(),
})

export type LoRAUploadResponse = z.infer<typeof LoRAUploadResponseSchema>

export const LoRAStatusResponseSchema = z.object({
  name: z.string(),
  is_loaded: z.boolean(),
  is_applied: z.boolean(),
  memory_usage_mb: z.number().optional(),
})

export type LoRAStatusResponse = z.infer<typeof LoRAStatusResponseSchema>

export const LoRAPreviewResponseSchema = z.object({
  task_id: z.string(),
  preview_url: z.string(),
  estimated_time: z.number(),
})

export type LoRAPreviewResponse = z.infer<typeof LoRAPreviewResponseSchema>

export const LoRAMemoryImpactResponseSchema = z.object({
  name: z.string(),
  estimated_vram_increase_mb: z.number(),
  compatible_with_current_setup: z.boolean(),
})

export type LoRAMemoryImpactResponse = z.infer<typeof LoRAMemoryImpactResponseSchema>