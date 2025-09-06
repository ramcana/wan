import React, { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { VideoMetadata } from "@/lib/api-schemas";
import { useVideoGallery } from "@/hooks/api/use-outputs";
import VideoCard from "./VideoCard";
import GalleryFilters from "./GalleryFilters";
import GalleryStats from "./GalleryStats";
import EnhancedVideoPlayer from "./EnhancedVideoPlayer";
import EmptyState from "./EmptyState";
import LoadingGrid from "./LoadingGrid";
import ErrorDisplay from "@/components/generation/ErrorDisplay";
import BulkOperations from "./BulkOperations";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Grid3X3,
  List,
  Search,
  Filter,
  Download,
  Trash2,
  Share2,
  Eye,
  EyeOff,
} from "lucide-react";
import { cn } from "@/lib/utils";

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

type ViewMode = "grid" | "list";

const EnhancedMediaGallery: React.FC<MediaGalleryProps> = ({
  className = "",
}) => {
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
  const [selectedVideos, setSelectedVideos] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [showFilters, setShowFilters] = useState(false);
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

  const handleVideoSelect = useCallback((video: VideoMetadata) => {
    setSelectedVideo(video);
  }, []);

  const handleVideoClose = useCallback(() => {
    setSelectedVideo(null);
  }, []);

  const handleFiltersChange = useCallback(
    (newFilters: Partial<GalleryFilters>) => {
      setFilters((prev) => ({ ...prev, ...newFilters }));
    },
    []
  );

  const handleVideoToggle = useCallback((videoId: string) => {
    setSelectedVideos((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(videoId)) {
        newSet.delete(videoId);
      } else {
        newSet.add(videoId);
      }
      return newSet;
    });
  }, []);

  const handleSelectAll = useCallback(() => {
    if (selectedVideos.size === filteredVideos.length) {
      setSelectedVideos(new Set());
    } else {
      setSelectedVideos(new Set(filteredVideos.map((v) => v.id)));
    }
  }, [selectedVideos.size, filteredVideos]);

  const handleBulkDelete = useCallback(async (videoIds: string[]) => {
    // Implementation for bulk delete
    console.log("Bulk delete:", videoIds);
    setSelectedVideos(new Set());
  }, []);

  const handleBulkDownload = useCallback(async (videoIds: string[]) => {
    // Implementation for bulk download
    console.log("Bulk download:", videoIds);
  }, []);

  const handleBulkShare = useCallback(async (videoIds: string[]) => {
    // Implementation for bulk share
    console.log("Bulk share:", videoIds);
  }, []);

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`space-y-6 ${className}`}
      >
        <div>
          <h1 className="text-2xl font-bold text-foreground">
            Generated Videos
          </h1>
          <p className="text-muted-foreground">
            Browse and manage your generated video content
          </p>
        </div>
        <ErrorDisplay error={error} onRetry={() => window.location.reload()} />
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`space-y-6 ${className}`}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h1 className="text-2xl font-bold text-foreground">Generated Videos</h1>
        <p className="text-muted-foreground">
          Browse and manage your generated video content
        </p>
      </motion.div>

      {/* Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <GalleryStats
          totalVideos={totalCount}
          totalSize={totalSize}
          filteredCount={filteredVideos.length}
        />
      </motion.div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4"
      >
        <div className="flex items-center gap-2">
          {/* View Mode Toggle */}
          <div className="flex items-center border rounded-lg p-1">
            <Button
              variant={viewMode === "grid" ? "default" : "ghost"}
              size="sm"
              onClick={() => setViewMode("grid")}
              className="px-3"
            >
              <Grid3X3 className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "list" ? "default" : "ghost"}
              size="sm"
              onClick={() => setViewMode("list")}
              className="px-3"
            >
              <List className="h-4 w-4" />
            </Button>
          </div>

          {/* Filter Toggle */}
          <Button
            variant={showFilters ? "default" : "outline"}
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2"
          >
            {showFilters ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
            Filters
          </Button>

          {/* Selection Info */}
          <AnimatePresence>
            {selectedVideos.size > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <Badge variant="secondary">
                  {selectedVideos.size} selected
                </Badge>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Bulk Operations */}
        <AnimatePresence>
          {selectedVideos.size > 0 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <BulkOperations
                selectedCount={selectedVideos.size}
                onDelete={() => handleBulkDelete(Array.from(selectedVideos))}
                onDownload={() =>
                  handleBulkDownload(Array.from(selectedVideos))
                }
                onShare={() => handleBulkShare(Array.from(selectedVideos))}
                onSelectAll={handleSelectAll}
                onClearSelection={() => setSelectedVideos(new Set())}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Filters */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <GalleryFilters
              filters={filters}
              onFiltersChange={handleFiltersChange}
              videoCount={filteredVideos.length}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Gallery Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
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
          <motion.div
            layout
            className={cn(
              viewMode === "grid"
                ? "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
                : "space-y-4"
            )}
          >
            <AnimatePresence mode="popLayout">
              {filteredVideos.map((video, index) => (
                <motion.div
                  key={video.id}
                  layout
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{
                    duration: 0.3,
                    delay: index * 0.05,
                  }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <VideoCard
                    video={video}
                    onSelect={handleVideoSelect}
                    isSelected={selectedVideos.has(video.id)}
                    onToggleSelect={() => handleVideoToggle(video.id)}
                    viewMode={viewMode}
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )}
      </motion.div>

      {/* Enhanced Video Player Modal */}
      <AnimatePresence>
        {selectedVideo && (
          <EnhancedVideoPlayer
            video={selectedVideo}
            onClose={handleVideoClose}
            onNext={() => {
              const currentIndex = filteredVideos.findIndex(
                (v) => v.id === selectedVideo.id
              );
              const nextIndex = (currentIndex + 1) % filteredVideos.length;
              setSelectedVideo(filteredVideos[nextIndex]);
            }}
            onPrevious={() => {
              const currentIndex = filteredVideos.findIndex(
                (v) => v.id === selectedVideo.id
              );
              const prevIndex =
                currentIndex === 0
                  ? filteredVideos.length - 1
                  : currentIndex - 1;
              setSelectedVideo(filteredVideos[prevIndex]);
            }}
            hasNext={filteredVideos.length > 1}
            hasPrevious={filteredVideos.length > 1}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default EnhancedMediaGallery;
