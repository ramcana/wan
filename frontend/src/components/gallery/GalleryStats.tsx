import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Video, HardDrive, Filter } from "lucide-react";

export interface GalleryStatsProps {
  totalVideos: number;
  totalSize: number;
  filteredCount: number;
  className?: string;
}

const GalleryStats: React.FC<GalleryStatsProps> = ({
  totalVideos,
  totalSize,
  filteredCount,
  className = "",
}) => {
  const formatFileSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) {
      return `${Math.round(sizeInMB * 1024)} KB`;
    } else if (sizeInMB < 1024) {
      return `${Math.round(sizeInMB * 10) / 10} MB`;
    } else {
      return `${Math.round((sizeInMB / 1024) * 10) / 10} GB`;
    }
  };

  const isFiltered = filteredCount !== totalVideos;

  return (
    <div className={`grid grid-cols-1 md:grid-cols-3 gap-4 ${className}`}>
      {/* Total Videos */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
              <Video className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                Total Videos
              </p>
              <p className="text-2xl font-bold">
                {totalVideos.toLocaleString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Total Storage */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
              <HardDrive className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                Total Storage
              </p>
              <p className="text-2xl font-bold">{formatFileSize(totalSize)}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filtered Count */}
      <Card className={isFiltered ? "ring-2 ring-primary/20" : ""}>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div
              className={`p-2 rounded-lg ${
                isFiltered ? "bg-primary/10" : "bg-gray-100 dark:bg-gray-800"
              }`}
            >
              <Filter
                className={`w-5 h-5 ${
                  isFiltered
                    ? "text-primary"
                    : "text-gray-600 dark:text-gray-400"
                }`}
              />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                {isFiltered ? "Filtered Results" : "Showing All"}
              </p>
              <p className="text-2xl font-bold">
                {filteredCount.toLocaleString()}
                {isFiltered && (
                  <span className="text-sm font-normal text-muted-foreground ml-1">
                    of {totalVideos}
                  </span>
                )}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GalleryStats;
