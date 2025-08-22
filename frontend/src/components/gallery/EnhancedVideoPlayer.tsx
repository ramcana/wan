import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { VideoMetadata } from "@/lib/api-schemas";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  X,
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Minimize,
  SkipBack,
  SkipForward,
  Download,
  Share2,
  Info,
  Settings,
  RotateCcw,
  Repeat,
  Repeat1,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";

interface EnhancedVideoPlayerProps {
  video: VideoMetadata;
  onClose: () => void;
  onNext?: () => void;
  onPrevious?: () => void;
  hasNext?: boolean;
  hasPrevious?: boolean;
}

type RepeatMode = "none" | "one" | "all";

const EnhancedVideoPlayer: React.FC<EnhancedVideoPlayerProps> = ({
  video,
  onClose,
  onNext,
  onPrevious,
  hasNext = false,
  hasPrevious = false,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [showInfo, setShowInfo] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [repeatMode, setRepeatMode] = useState<RepeatMode>("none");
  const [controlsTimeout, setControlsTimeout] = useState<NodeJS.Timeout | null>(
    null
  );

  // Auto-hide controls
  useEffect(() => {
    const resetTimeout = () => {
      if (controlsTimeout) {
        clearTimeout(controlsTimeout);
      }
      setShowControls(true);
      const timeout = setTimeout(() => {
        if (isPlaying) {
          setShowControls(false);
        }
      }, 3000);
      setControlsTimeout(timeout);
    };

    resetTimeout();

    return () => {
      if (controlsTimeout) {
        clearTimeout(controlsTimeout);
      }
    };
  }, [isPlaying]);

  // Video event handlers
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleDurationChange = () => setDuration(video.duration);
    const handleEnded = () => {
      setIsPlaying(false);
      if (repeatMode === "one") {
        video.currentTime = 0;
        video.play();
        setIsPlaying(true);
      } else if (repeatMode === "all" && hasNext && onNext) {
        onNext();
      }
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("durationchange", handleDurationChange);
    video.addEventListener("ended", handleEnded);

    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("durationchange", handleDurationChange);
      video.removeEventListener("ended", handleEnded);
    };
  }, [repeatMode, hasNext, onNext]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlayPause();
          break;
        case "ArrowLeft":
          e.preventDefault();
          seekRelative(-10);
          break;
        case "ArrowRight":
          e.preventDefault();
          seekRelative(10);
          break;
        case "ArrowUp":
          e.preventDefault();
          setVolume((prev) => Math.min(1, prev + 0.1));
          break;
        case "ArrowDown":
          e.preventDefault();
          setVolume((prev) => Math.max(0, prev - 0.1));
          break;
        case "m":
          toggleMute();
          break;
        case "f":
          toggleFullscreen();
          break;
        case "Escape":
          if (isFullscreen) {
            toggleFullscreen();
          } else {
            onClose();
          }
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  const seekTo = (time: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = time;
    setCurrentTime(time);
  };

  const seekRelative = (seconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    const newTime = Math.max(0, Math.min(duration, currentTime + seconds));
    seekTo(newTime);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (newVolume: number[]) => {
    const video = videoRef.current;
    if (!video) return;
    const vol = newVolume[0];
    video.volume = vol;
    setVolume(vol);
    setIsMuted(vol === 0);
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const handlePlaybackRateChange = (rate: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = rate;
    setPlaybackRate(rate);
  };

  const cycleRepeatMode = () => {
    const modes: RepeatMode[] = ["none", "one", "all"];
    const currentIndex = modes.indexOf(repeatMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    setRepeatMode(modes[nextIndex]);
  };

  const getRepeatIcon = () => {
    switch (repeatMode) {
      case "one":
        return <Repeat1 className="h-4 w-4" />;
      case "all":
        return <Repeat className="h-4 w-4 text-primary" />;
      default:
        return <Repeat className="h-4 w-4" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
      onMouseMove={() => {
        setShowControls(true);
        if (controlsTimeout) {
          clearTimeout(controlsTimeout);
        }
        const timeout = setTimeout(() => {
          if (isPlaying) {
            setShowControls(false);
          }
        }, 3000);
        setControlsTimeout(timeout);
      }}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="relative w-full h-full max-w-6xl max-h-[90vh] mx-4"
      >
        {/* Video */}
        <video
          ref={videoRef}
          src={video.file_path}
          className="w-full h-full object-contain"
          onClick={togglePlayPause}
        />

        {/* Controls Overlay */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/60"
            >
              {/* Top Controls */}
              <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-start">
                <div className="flex items-center gap-4">
                  <h3 className="text-white font-medium truncate max-w-md">
                    {video.filename}
                  </h3>
                  <Badge variant="secondary" className="bg-black/50 text-white">
                    {video.model_type}
                  </Badge>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowInfo(!showInfo)}
                    className="text-white hover:bg-white/20"
                  >
                    <Info className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-white hover:bg-white/20"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Center Play Button */}
              {!isPlaying && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute inset-0 flex items-center justify-center"
                >
                  <Button
                    variant="ghost"
                    size="lg"
                    onClick={togglePlayPause}
                    className="w-20 h-20 rounded-full bg-white/20 hover:bg-white/30 text-white"
                  >
                    <Play className="h-8 w-8 ml-1" />
                  </Button>
                </motion.div>
              )}

              {/* Bottom Controls */}
              <div className="absolute bottom-0 left-0 right-0 p-4 space-y-4">
                {/* Progress Bar */}
                <div className="flex items-center gap-4">
                  <span className="text-white text-sm min-w-[40px]">
                    {formatTime(currentTime)}
                  </span>
                  <Slider
                    value={[currentTime]}
                    max={duration}
                    step={0.1}
                    onValueChange={(value) => seekTo(value[0])}
                    className="flex-1"
                  />
                  <span className="text-white text-sm min-w-[40px]">
                    {formatTime(duration)}
                  </span>
                </div>

                {/* Control Buttons */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {/* Previous */}
                    {hasPrevious && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={onPrevious}
                        className="text-white hover:bg-white/20"
                      >
                        <SkipBack className="h-4 w-4" />
                      </Button>
                    )}

                    {/* Play/Pause */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={togglePlayPause}
                      className="text-white hover:bg-white/20"
                    >
                      {isPlaying ? (
                        <Pause className="h-4 w-4" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                    </Button>

                    {/* Next */}
                    {hasNext && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={onNext}
                        className="text-white hover:bg-white/20"
                      >
                        <SkipForward className="h-4 w-4" />
                      </Button>
                    )}

                    {/* Volume */}
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={toggleMute}
                        className="text-white hover:bg-white/20"
                      >
                        {isMuted || volume === 0 ? (
                          <VolumeX className="h-4 w-4" />
                        ) : (
                          <Volume2 className="h-4 w-4" />
                        )}
                      </Button>
                      <Slider
                        value={[isMuted ? 0 : volume]}
                        max={1}
                        step={0.1}
                        onValueChange={handleVolumeChange}
                        className="w-20"
                      />
                    </div>

                    {/* Repeat */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={cycleRepeatMode}
                      className={cn(
                        "text-white hover:bg-white/20",
                        repeatMode !== "none" && "text-primary"
                      )}
                    >
                      {getRepeatIcon()}
                    </Button>
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Playback Speed */}
                    <select
                      value={playbackRate}
                      onChange={(e) =>
                        handlePlaybackRateChange(Number(e.target.value))
                      }
                      className="bg-black/50 text-white border border-white/20 rounded px-2 py-1 text-sm"
                    >
                      <option value={0.25}>0.25x</option>
                      <option value={0.5}>0.5x</option>
                      <option value={0.75}>0.75x</option>
                      <option value={1}>1x</option>
                      <option value={1.25}>1.25x</option>
                      <option value={1.5}>1.5x</option>
                      <option value={2}>2x</option>
                    </select>

                    {/* Download */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        const a = document.createElement("a");
                        a.href = video.file_path;
                        a.download = video.filename;
                        a.click();
                      }}
                      className="text-white hover:bg-white/20"
                    >
                      <Download className="h-4 w-4" />
                    </Button>

                    {/* Fullscreen */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={toggleFullscreen}
                      className="text-white hover:bg-white/20"
                    >
                      {isFullscreen ? (
                        <Minimize className="h-4 w-4" />
                      ) : (
                        <Maximize className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Info Panel */}
        <AnimatePresence>
          {showInfo && (
            <motion.div
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              className="absolute top-0 right-0 h-full w-80 bg-black/90 p-6 overflow-y-auto"
            >
              <Card className="bg-transparent border-white/20">
                <CardHeader>
                  <CardTitle className="text-white">
                    Video Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-white">
                  <div>
                    <h4 className="font-medium mb-2">Details</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Filename:</span>
                        <span className="truncate ml-2">{video.filename}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Model:</span>
                        <span>{video.model_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Resolution:</span>
                        <span>{video.resolution}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Duration:</span>
                        <span>{video.duration}s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Size:</span>
                        <span>{video.file_size_mb.toFixed(1)} MB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Created:</span>
                        <span>
                          {formatDistanceToNow(new Date(video.created_at))} ago
                        </span>
                      </div>
                    </div>
                  </div>

                  {video.prompt && (
                    <div>
                      <h4 className="font-medium mb-2">Prompt</h4>
                      <p className="text-sm text-gray-300 bg-black/30 p-3 rounded">
                        {video.prompt}
                      </p>
                    </div>
                  )}

                  {video.generation_params && (
                    <div>
                      <h4 className="font-medium mb-2">
                        Generation Parameters
                      </h4>
                      <div className="space-y-1 text-sm">
                        {Object.entries(video.generation_params).map(
                          ([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-400 capitalize">
                                {key.replace(/_/g, " ")}:
                              </span>
                              <span>{String(value)}</span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
};

export default EnhancedVideoPlayer;
