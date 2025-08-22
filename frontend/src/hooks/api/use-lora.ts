import { useQuery, useMutation, useQueryClient } from 'react-query';
import { loraApi } from '../../lib/api-client';
import { useToast } from '../use-toast';
import type {
  LoRAListResponse,
  LoRAUploadResponse,
  LoRAStatusResponse,
  LoRAPreviewResponse,
  LoRAMemoryImpactResponse,
} from '../../lib/api-schemas';

// Query keys
const LORA_QUERY_KEYS = {
  list: ['lora', 'list'] as const,
  status: (name: string) => ['lora', 'status', name] as const,
  preview: (name: string, prompt: string) => ['lora', 'preview', name, prompt] as const,
  memoryImpact: (name: string) => ['lora', 'memory-impact', name] as const,
};

// Hook to list all LoRAs
export function useLoRAList() {
  return useQuery<LoRAListResponse>({
    queryKey: LORA_QUERY_KEYS.list,
    queryFn: () => loraApi.list(),
    staleTime: 30000, // 30 seconds
    cacheTime: 300000, // 5 minutes
    refetchOnWindowFocus: false,
    retry: 2,
  });
}

// Hook to get LoRA status
export function useLoRAStatus(loraName: string, enabled = true) {
  return useQuery<LoRAStatusResponse>({
    queryKey: LORA_QUERY_KEYS.status(loraName),
    queryFn: () => loraApi.getStatus(loraName),
    enabled: enabled && !!loraName,
    staleTime: 10000, // 10 seconds
    cacheTime: 60000, // 1 minute
    retry: 1,
  });
}

// Hook to generate LoRA preview
export function useLoRAPreview(loraName: string, basePrompt: string, enabled = false) {
  return useQuery<LoRAPreviewResponse>({
    queryKey: LORA_QUERY_KEYS.preview(loraName, basePrompt),
    queryFn: () => loraApi.generatePreview(loraName, basePrompt),
    enabled: enabled && !!loraName && !!basePrompt,
    staleTime: 60000, // 1 minute
    cacheTime: 300000, // 5 minutes
    retry: 1,
  });
}

// Hook to estimate memory impact
export function useLoRAMemoryImpact(loraName: string, enabled = true) {
  return useQuery<LoRAMemoryImpactResponse>({
    queryKey: LORA_QUERY_KEYS.memoryImpact(loraName),
    queryFn: () => loraApi.estimateMemoryImpact(loraName),
    enabled: enabled && !!loraName,
    staleTime: 30000, // 30 seconds
    cacheTime: 300000, // 5 minutes
    retry: 1,
  });
}

// Hook to upload LoRA
export function useLoRAUpload() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation<
    LoRAUploadResponse,
    Error,
    { file: File; name?: string; onProgress?: (progress: number) => void }
  >({
    mutationFn: ({ file, name, onProgress }) => loraApi.upload(file, name, onProgress),
    onSuccess: (data) => {
      // Invalidate and refetch LoRA list
      queryClient.invalidateQueries(LORA_QUERY_KEYS.list);
      
      // Show success toast
      toast({
        title: 'LoRA Uploaded',
        description: data.message,
        variant: 'default',
      });
    },
    onError: (error) => {
      // Show error toast
      toast({
        title: 'Upload Failed',
        description: error.message || 'Failed to upload LoRA file',
        variant: 'destructive',
      });
    },
  });
}

// Hook to delete LoRA
export function useLoRADelete() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation<any, Error, string>({
    mutationFn: (loraName: string) => loraApi.delete(loraName),
    onSuccess: (data, loraName) => {
      // Invalidate and refetch LoRA list
      queryClient.invalidateQueries(LORA_QUERY_KEYS.list);
      
      // Remove specific LoRA queries from cache
      queryClient.removeQueries(LORA_QUERY_KEYS.status(loraName));
      queryClient.removeQueries(['lora', 'preview', loraName]);
      queryClient.removeQueries(LORA_QUERY_KEYS.memoryImpact(loraName));
      
      // Show success toast
      toast({
        title: 'LoRA Deleted',
        description: data.message || `Successfully deleted ${loraName}`,
        variant: 'default',
      });
    },
    onError: (error, loraName) => {
      // Show error toast
      toast({
        title: 'Delete Failed',
        description: error.message || `Failed to delete ${loraName}`,
        variant: 'destructive',
      });
    },
  });
}

// Hook to refresh LoRA list
export function useRefreshLoRAList() {
  const queryClient = useQueryClient();

  return () => {
    queryClient.invalidateQueries(LORA_QUERY_KEYS.list);
  };
}

// Hook to get LoRA by name from the list
export function useLoRAByName(loraName: string) {
  const { data: loraList } = useLoRAList();
  
  return loraList?.loras.find(lora => lora.name === loraName);
}

// Hook to get LoRA statistics
export function useLoRAStatistics() {
  const { data: loraList, isLoading, error } = useLoRAList();

  const statistics = {
    totalCount: loraList?.total_count || 0,
    totalSizeMB: loraList?.total_size_mb || 0,
    loadedCount: loraList?.loras.filter(lora => lora.is_loaded).length || 0,
    appliedCount: loraList?.loras.filter(lora => lora.is_applied).length || 0,
    categories: {
      style: loraList?.loras.filter(lora => 
        lora.name.toLowerCase().includes('style') || 
        lora.name.toLowerCase().includes('anime') ||
        lora.name.toLowerCase().includes('art')
      ).length || 0,
      character: loraList?.loras.filter(lora => 
        lora.name.toLowerCase().includes('character') || 
        lora.name.toLowerCase().includes('person')
      ).length || 0,
      quality: loraList?.loras.filter(lora => 
        lora.name.toLowerCase().includes('detail') || 
        lora.name.toLowerCase().includes('quality')
      ).length || 0,
    },
  };

  return {
    statistics,
    isLoading,
    error,
  };
}

// Hook for LoRA search and filtering
export function useLoRASearch(searchTerm: string, category?: string) {
  const { data: loraList, isLoading, error } = useLoRAList();

  const filteredLoras = loraList?.loras.filter(lora => {
    const matchesSearch = !searchTerm || 
      lora.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      lora.filename.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesCategory = !category || category === 'all' || (() => {
      const name = lora.name.toLowerCase();
      switch (category) {
        case 'style':
          return name.includes('style') || name.includes('anime') || name.includes('art');
        case 'character':
          return name.includes('character') || name.includes('person');
        case 'quality':
          return name.includes('detail') || name.includes('quality');
        case 'loaded':
          return lora.is_loaded;
        case 'applied':
          return lora.is_applied;
        default:
          return true;
      }
    })();

    return matchesSearch && matchesCategory;
  }) || [];

  return {
    loras: filteredLoras,
    totalCount: filteredLoras.length,
    isLoading,
    error,
  };
}