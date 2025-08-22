import { useQuery, useMutation, useQueryClient } from 'react-query';
import { get, del, ApiError } from '@/lib/api-client';
import { useToast } from '@/hooks/use-toast';

// Query keys
export const outputsKeys = {
  all: ['outputs'] as const,
  list: () => [...outputsKeys.all, 'list'] as const,
  video: (id: string) => [...outputsKeys.all, 'video', id] as const,
};

// Get outputs query
export const useOutputs = () => {
  return useQuery({
    queryKey: outputsKeys.list(),
    queryFn: () => get('/outputs'),
    staleTime: 30000, // Consider data stale after 30 seconds
    retry: (failureCount, error) => {
      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
        return false;
      }
      return failureCount < 3;
    },
  });
};

// Get video info query
export const useVideoInfo = (videoId: string, enabled = true) => {
  return useQuery({
    queryKey: outputsKeys.video(videoId),
    queryFn: () => get(`/outputs/${videoId}`),
    enabled: enabled && !!videoId,
    staleTime: 60000, // Consider data stale after 1 minute
  });
};

// Delete video mutation
export const useDeleteVideo = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation<{ message: string }, ApiError, string>({
    mutationFn: (videoId: string) => del(`/outputs/${videoId}`),
    onSuccess: (_, videoId) => {
      toast({
        title: 'Video Deleted',
        description: 'The video has been successfully deleted.',
      });
      
      // Remove from cache and invalidate list
      queryClient.removeQueries(outputsKeys.video(videoId));
      queryClient.invalidateQueries(outputsKeys.list());
    },
    onError: (error) => {
      toast({
        title: 'Delete Failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
};

// Custom hook for video gallery with filtering and sorting
export const useVideoGallery = () => {
  const { data, isLoading, error } = useOutputs();
  
  const videos = data?.videos || [];
  
  // Helper functions for filtering and sorting
  const filterByModel = (modelType: string) => {
    return videos.filter(video => video.model_type === modelType);
  };
  
  const filterByDateRange = (startDate: Date, endDate: Date) => {
    return videos.filter(video => {
      const videoDate = new Date(video.created_at);
      return videoDate >= startDate && videoDate <= endDate;
    });
  };
  
  const sortByDate = (ascending = false) => {
    return [...videos].sort((a, b) => {
      const dateA = new Date(a.created_at).getTime();
      const dateB = new Date(b.created_at).getTime();
      return ascending ? dateA - dateB : dateB - dateA;
    });
  };
  
  const sortBySize = (ascending = false) => {
    return [...videos].sort((a, b) => {
      return ascending ? a.file_size_mb - b.file_size_mb : b.file_size_mb - a.file_size_mb;
    });
  };
  
  const searchVideos = (query: string) => {
    const lowercaseQuery = query.toLowerCase();
    return videos.filter(video => 
      video.prompt.toLowerCase().includes(lowercaseQuery) ||
      video.filename.toLowerCase().includes(lowercaseQuery) ||
      video.model_type.toLowerCase().includes(lowercaseQuery)
    );
  };
  
  return {
    videos,
    totalCount: data?.total_count || 0,
    totalSize: data?.total_size_mb || 0,
    isLoading,
    error,
    // Helper functions
    filterByModel,
    filterByDateRange,
    sortByDate,
    sortBySize,
    searchVideos,
  };
};