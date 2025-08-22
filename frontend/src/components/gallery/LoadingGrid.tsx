import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export interface LoadingGridProps {
  count?: number;
  className?: string;
}

const LoadingGrid: React.FC<LoadingGridProps> = ({
  count = 12,
  className = "",
}) => {
  return (
    <div
      className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 ${className}`}
    >
      {Array.from({ length: count }).map((_, index) => (
        <Card key={index} className="overflow-hidden">
          <CardContent className="p-0">
            {/* Thumbnail skeleton */}
            <div className="relative aspect-video">
              <Skeleton className="w-full h-full" />

              {/* Badge skeleton */}
              <div className="absolute top-2 left-2">
                <Skeleton className="h-5 w-16 rounded-full" />
              </div>

              {/* Duration skeleton */}
              <div className="absolute bottom-2 right-2">
                <Skeleton className="h-4 w-10 rounded" />
              </div>
            </div>

            {/* Content skeleton */}
            <div className="p-4 space-y-3">
              {/* Title */}
              <div className="space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-2/3" />
              </div>

              {/* Metadata */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Skeleton className="h-3 w-12" />
                  <Skeleton className="h-3 w-16" />
                </div>
                <Skeleton className="h-3 w-16" />
              </div>

              {/* Generation time */}
              <Skeleton className="h-3 w-20" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export { LoadingGrid };
export default LoadingGrid;
