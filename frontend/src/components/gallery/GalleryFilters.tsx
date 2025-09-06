import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import {
  Search,
  Filter,
  SortAsc,
  SortDesc,
  X,
  Calendar,
  Video,
} from "lucide-react";
import { GalleryFilters as GalleryFiltersType } from "./MediaGallery";

export interface GalleryFiltersProps {
  filters: GalleryFiltersType;
  onFiltersChange: (filters: Partial<GalleryFiltersType>) => void;
  videoCount: number;
  className?: string;
}

const GalleryFilters: React.FC<GalleryFiltersProps> = ({
  filters,
  onFiltersChange,
  videoCount,
  className = "",
}) => {
  const hasActiveFilters =
    filters.search.trim() !== "" || filters.modelType !== "all";

  const clearFilters = () => {
    onFiltersChange({
      search: "",
      modelType: "all",
      dateRange: undefined,
    });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({ search: e.target.value });
  };

  const handleModelTypeChange = (value: string) => {
    onFiltersChange({ modelType: value });
  };

  const handleSortByChange = (value: string) => {
    onFiltersChange({ sortBy: value as "date" | "size" | "name" });
  };

  const handleSortOrderChange = () => {
    onFiltersChange({
      sortOrder: filters.sortOrder === "asc" ? "desc" : "asc",
    });
  };

  return (
    <Card className={className}>
      <CardContent className="p-4">
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <Label htmlFor="search" className="sr-only">
              Search videos
            </Label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                id="search"
                type="text"
                placeholder="Search videos by prompt, filename, or model..."
                value={filters.search}
                onChange={handleSearchChange}
                className="pl-10"
              />
              {filters.search && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onFiltersChange({ search: "" })}
                  className="absolute right-1 top-1/2 transform -translate-y-1/2 h-7 w-7 p-0"
                >
                  <X className="w-3 h-3" />
                </Button>
              )}
            </div>
          </div>

          {/* Model Type Filter */}
          <div className="w-full lg:w-48">
            <Label htmlFor="model-type" className="sr-only">
              Filter by model type
            </Label>
            <Select
              value={filters.modelType}
              onValueChange={handleModelTypeChange}
            >
              <SelectTrigger id="model-type">
                <div className="flex items-center space-x-2">
                  <Video className="w-4 h-4" />
                  <SelectValue placeholder="All Models" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Models</SelectItem>
                <SelectItem value="T2V-A14B">T2V-A14B</SelectItem>
                <SelectItem value="I2V-A14B">I2V-A14B</SelectItem>
                <SelectItem value="TI2V-5B">TI2V-5B</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Sort By */}
          <div className="w-full lg:w-40">
            <Label htmlFor="sort-by" className="sr-only">
              Sort by
            </Label>
            <Select value={filters.sortBy} onValueChange={handleSortByChange}>
              <SelectTrigger id="sort-by">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="date">Date Created</SelectItem>
                <SelectItem value="size">File Size</SelectItem>
                <SelectItem value="name">Name</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Sort Order */}
          <Button
            variant="outline"
            size="default"
            onClick={handleSortOrderChange}
            className="w-full lg:w-auto"
          >
            {filters.sortOrder === "asc" ? (
              <>
                <SortAsc className="w-4 h-4 mr-2" />
                Ascending
              </>
            ) : (
              <>
                <SortDesc className="w-4 h-4 mr-2" />
                Descending
              </>
            )}
          </Button>

          {/* Clear Filters */}
          {hasActiveFilters && (
            <Button
              variant="ghost"
              onClick={clearFilters}
              className="w-full lg:w-auto"
            >
              <X className="w-4 h-4 mr-2" />
              Clear Filters
            </Button>
          )}
        </div>

        {/* Results count */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            {videoCount === 0
              ? "No videos found"
              : videoCount === 1
              ? "1 video"
              : `${videoCount} videos`}
            {hasActiveFilters && " (filtered)"}
          </div>

          {/* Quick filter chips */}
          <div className="flex items-center space-x-2">
            {filters.search && (
              <div className="flex items-center space-x-1 bg-primary/10 text-primary px-2 py-1 rounded-md text-xs">
                <Search className="w-3 h-3" />
                <span>"{filters.search}"</span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onFiltersChange({ search: "" })}
                  className="h-4 w-4 p-0 ml-1"
                >
                  <X className="w-2 h-2" />
                </Button>
              </div>
            )}

            {filters.modelType !== "all" && (
              <div className="flex items-center space-x-1 bg-primary/10 text-primary px-2 py-1 rounded-md text-xs">
                <Video className="w-3 h-3" />
                <span>{filters.modelType}</span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onFiltersChange({ modelType: "all" })}
                  className="h-4 w-4 p-0 ml-1"
                >
                  <X className="w-2 h-2" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default GalleryFilters;
