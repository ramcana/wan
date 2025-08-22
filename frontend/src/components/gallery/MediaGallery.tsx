import React, { useState, useMemo } from "react";
import { VideoMetadata } from "@/lib/api-schemas";
import { useVideoGallery } from "@/hooks/api/use-outputs";
import VideoCard from "./VideoCard";
import GalleryFilters from "./GalleryFilters";
import GalleryStats from "./GalleryStats";
import VideoPlayer from "./VideoPlayer";
import EmptyState from "./EmptyState";
import LoadingGrid from "./LoadingGrid";
import ErrorDisplay from "@/components/generation/ErrorDisplay";

export interface MediaGalleryProps {
  className?: string;
}

export interface GalleryFilters {
  search: string;
  modelType: string;
  sortBy: "date" | "size" | "name";
  sortOrder: "asc" | "desc";
  dateRange?: {
    start: Date;
    end: Date;
  };
}

const MediaGallery: React.FC<MediaGalleryProps> = ({ className = "" }) => {
  const {
    videos,
    totalCount,
    totalSize,
    isLoading,
    error,
    searchVideos,
    filterByModel,
    sortByDate,
    sortBySize,
  } = useVideoGallery();

  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(
    null
  );
  const [filters, setFilters] = useState<GalleryFilters>({
    search: "",
    modelType: "all",
    sortBy: "date",
    sortOrder: "desc",
  });

  // Apply filters and sorting
  const filteredVideos = useMemo(() => {
    let result = videos;

    // Apply search filter
    if (filters.search.trim()) {
      result = searchVideos(filters.search.trim());
    }

    // Apply model type filter
    if (filters.modelType !== "all") {
      result = result.filter((video) => video.model_type === filters.modelType);
    }

    // Apply date range filter
    if (filters.dateRange) {
      result = result.filter((video) => {
        const videoDate = new Date(video.created_at);
        return (
          videoDate >= filters.dateRange!.start &&
          videoDate <= filters.dateRange!.end
        );
      });
    }

    // Apply sorting
    switch (filters.sortBy) {
      case "date":
        result = [...result].sort((a, b) => {
          const dateA = new Date(a.created_at).getTime();
          const dateB = new Date(b.created_at).getTime();
          return filters.sortOrder === "asc" ? dateA - dateB : dateB - dateA;
        });
        break;
      case "size":
        result = [...result].sort((a, b) => {
          return filters.sortOrder === "asc"
            ? a.file_size_mb - b.file_size_mb
            : b.file_size_mb - a.file_size_mb;
        });
        break;
      case "name":
        result = [...result].sort((a, b) => {
          const nameA = a.filename.toLowerCase();
          const nameB = b.filename.toLowerCase();
          return filters.sortOrder === "asc"
            ? nameA.localeCompare(nameB)
            : nameB.localeCompare(nameA);
        });
        break;
    }

    return result;
  }, [videos, filters, searchVideos]);

  const handleVideoSelect = (video: VideoMetadata) => {
    setSelectedVideo(video);
  };

  const handleVideoClose = () => {
    setSelectedVideo(null);
  };

  const handleFiltersChange = (newFilters: Partial<GalleryFilters>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  };

  if (error) {
    return (
      <div className={`space-y-6 ${className}`}>
        <div>
          <h1 className="text-2xl font-bold text-foreground">
            Generated Videos
          </h1>
          <p className="text-muted-foreground">
            Browse and manage your generated video content
          </p>
        </div>
        <ErrorDisplay error={error} onRetry={() => window.location.reload()} />
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Generated Videos</h1>
        <p className="text-muted-foreground">
          Browse and manage your generated video content
        </p>
      </div>

      {/* Stats */}
      <GalleryStats
        totalVideos={totalCount}
        totalSize={totalSize}
        filteredCount={filteredVideos.length}
      />

      {/* Filters */}
      <GalleryFilters
        filters={filters}
        onFiltersChange={handleFiltersChange}
        videoCount={filteredVideos.length}
      />

      {/* Gallery Content */}
      {isLoading ? (
        <LoadingGrid />
      ) : filteredVideos.length === 0 ? (
        <EmptyState
          hasVideos={videos.length > 0}
          hasFilters={
            filters.search.trim() !== "" || filters.modelType !== "all"
          }
          onClearFilters={() =>
            setFilters({
              search: "",
              modelType: "all",
              sortBy: "date",
              sortOrder: "desc",
            })
          }
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredVideos.map((video) => (
            <VideoCard
              key={video.id}
              video={video}
              onSelect={handleVideoSelect}
            />
          ))}
        </div>
      )}

      {/* Video Player Modal */}
      {selectedVideo && (
        <VideoPlayer video={selectedVideo} onClose={handleVideoClose} />
      )}
    </div>
  );
};

export default MediaGallery;
