import React, { useState } from "react";
import { motion } from "framer-motion";
import { VideoMetadata } from "@/lib/api-schemas";
import { useDeleteVideo } from "@/hooks/api/use-outputs";
import { outputsApi } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Play,
  Download,
  Trash2,
  MoreVertical,
  Clock,
  HardDrive,
  Calendar,
  Image as ImageIcon,
  Check,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

export interface VideoCardProps {
  video: VideoMetadata;
  onSelect: (video: VideoMetadata) => void;
  isSelected?: boolean;
  onToggleSelect?: () => void;
  viewMode?: "grid" | "list";
  className?: string;
}

const VideoCard: React.FC<VideoCardProps> = ({
  video,
  onSelect,
  isSelected = false,
  onToggleSelect,
  viewMode = "grid",
  className = "",
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [thumbnailError, setThumbnailError] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const deleteVideoMutation = useDeleteVideo();

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect(video);
  };

  const handleSelectToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    onToggleSelect?.();
  };

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    const downloadUrl = outputsApi.getDownloadUrl(video.id);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = video.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (
      window.confirm(`Are you sure you want to delete "${video.filename}"?`)
    ) {
      setIsDeleting(true);
      try {
        await deleteVideoMutation.mutateAsync(video.id);
      } catch (error) {
        console.error("Failed to delete video:", error);
      } finally {
        setIsDeleting(false);
      }
    }
  };

  const formatFileSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) {
      return `${Math.round(sizeInMB * 1024)} KB`;
    } else if (sizeInMB < 1024) {
      return `${Math.round(sizeInMB * 10) / 10} MB`;
    } else {
      return `${Math.round((sizeInMB / 1024) * 10) / 10} GB`;
    }
  };

  const formatDuration = (seconds?: number): string => {
    if (!seconds) return "Unknown";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getModelTypeColor = (modelType: string): string => {
    switch (modelType) {
      case "T2V-A14B":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "I2V-A14B":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "TI2V-5B":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  // Generate thumbnail URL or use actual thumbnail path
  const thumbnailSrc = video.thumbnail_path
    ? `${
        import.meta.env.VITE_API_URL || "http://localhost:8000"
      }/thumbnails/${video.thumbnail_path.split("/").pop()}`
    : outputsApi.getThumbnailUrl(video.id);

  if (viewMode === "list") {
    return (
      <motion.div
        layout
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <Card
          className={cn(
            "group cursor-pointer transition-all duration-200 hover:shadow-md",
            isSelected && "ring-2 ring-primary bg-primary/5",
            isDeleting && "opacity-50 pointer-events-none",
            className
          )}
          onClick={handlePlay}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-4">
              {/* Selection Checkbox */}
              {onToggleSelect && (
                <div onClick={handleSelectToggle}>
                  <Checkbox checked={isSelected} />
                </div>
              )}

              {/* Thumbnail */}
              <div className="relative w-24 h-16 bg-muted rounded overflow-hidden flex-shrink-0">
                {thumbnailSrc && !thumbnailError ? (
                  <img
                    src={thumbnailSrc}
                    alt={`Thumbnail for ${video.filename}`}
                    className="w-full h-full object-cover"
                    onError={() => setThumbnailError(true)}
                    loading="lazy"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                    <ImageIcon className="w-6 h-6 text-muted-foreground/50" />
                  </div>
                )}

                {/* Duration Badge */}
                {video.duration_seconds && (
                  <div className="absolute bottom-1 right-1 bg-black/70 text-white text-xs px-1 py-0.5 rounded">
                    {formatDuration(video.duration_seconds)}
                  </div>
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3
                      className="font-medium text-sm truncate"
                      title={video.filename}
                    >
                      {video.filename}
                    </h3>
                    <p
                      className="text-xs text-muted-foreground line-clamp-1 mt-1"
                      title={video.prompt}
                    >
                      {video.prompt}
                    </p>

                    <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                      <Badge className={getModelTypeColor(video.model_type)}>
                        {video.model_type}
                      </Badge>
                      <div className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        <span>
                          {formatDistanceToNow(new Date(video.created_at), {
                            addSuffix: true,
                          })}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <HardDrive className="w-3 h-3" />
                        <span>{formatFileSize(video.file_size_mb)}</span>
                      </div>
                      <span>{video.resolution}</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8 w-8 p-0"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={handlePlay}>
                          <Play className="w-4 h-4 mr-2" />
                          Play
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={handleDownload}>
                          <Download className="w-4 h-4 mr-2" />
                          Download
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={handleDelete}
                          className="text-destructive focus:text-destructive"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div layout whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
      <Card
        className={cn(
          "group cursor-pointer transition-all duration-200 hover:shadow-lg",
          isSelected && "ring-2 ring-primary bg-primary/5",
          isDeleting && "opacity-50 pointer-events-none",
          className
        )}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={handlePlay}
      >
        <CardContent className="p-0">
          {/* Thumbnail/Preview */}
          <div className="relative aspect-video bg-muted rounded-t-lg overflow-hidden">
            {/* Selection Checkbox */}
            {onToggleSelect && (
              <div
                className="absolute top-2 left-2 z-10"
                onClick={handleSelectToggle}
              >
                <div
                  className={cn(
                    "w-6 h-6 rounded border-2 bg-white/90 flex items-center justify-center transition-all",
                    isSelected
                      ? "border-primary bg-primary text-white"
                      : "border-gray-300 hover:border-primary"
                  )}
                >
                  {isSelected && <Check className="h-3 w-3" />}
                </div>
              </div>
            )}
            {thumbnailSrc && !thumbnailError ? (
              <img
                src={thumbnailSrc}
                alt={`Thumbnail for ${video.filename}`}
                className="w-full h-full object-cover"
                onError={() => setThumbnailError(true)}
                loading="lazy"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                <ImageIcon
                  className="w-12 h-12 text-muted-foreground/50"
                  data-testid="image-icon"
                />
              </div>
            )}

            {/* Play overlay */}
            <div
              className={cn(
                "absolute inset-0 bg-black/50 flex items-center justify-center transition-opacity duration-200",
                isHovered ? "opacity-100" : "opacity-0"
              )}
            >
              <Button
                size="lg"
                className="rounded-full bg-white/20 hover:bg-white/30 text-white border-white/30"
                onClick={handlePlay}
              >
                <Play className="w-6 h-6 ml-1" />
              </Button>
            </div>

            {/* Duration badge */}
            {video.duration_seconds && (
              <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                {formatDuration(video.duration_seconds)}
              </div>
            )}

            {/* Model type badge */}
            <div
              className={cn(
                "absolute top-2",
                onToggleSelect ? "right-2" : "left-2"
              )}
            >
              <Badge className={getModelTypeColor(video.model_type)}>
                {video.model_type}
              </Badge>
            </div>

            {/* Actions menu */}
            <div
              className={cn(
                "absolute top-2",
                onToggleSelect ? "right-10" : "right-2"
              )}
            >
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    size="sm"
                    variant="ghost"
                    className={cn(
                      "h-8 w-8 p-0 bg-black/50 hover:bg-black/70 text-white transition-opacity duration-200",
                      isHovered ? "opacity-100" : "opacity-0"
                    )}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <MoreVertical className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={handlePlay}>
                    <Play className="w-4 h-4 mr-2" />
                    Play
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={handleDownload}>
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={handleDelete}
                    className="text-destructive focus:text-destructive"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Video Info */}
          <div className="p-4 space-y-3">
            {/* Title */}
            <div>
              <h3
                className="font-medium text-sm truncate"
                title={video.filename}
              >
                {video.filename}
              </h3>
              <p
                className="text-xs text-muted-foreground line-clamp-2 mt-1"
                title={video.prompt}
              >
                {video.prompt}
              </p>
            </div>

            {/* Metadata */}
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1">
                  <HardDrive className="w-3 h-3" />
                  <span>{formatFileSize(video.file_size_mb)}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Calendar className="w-3 h-3" />
                  <span>
                    {formatDistanceToNow(new Date(video.created_at), {
                      addSuffix: true,
                    })}
                  </span>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                {video.resolution}
              </div>
            </div>

            {/* Generation time */}
            {video.generation_time_minutes && (
              <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                <Clock className="w-3 h-3" />
                <span>
                  {Math.round(video.generation_time_minutes)} min generation
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default VideoCard;
