import React, { useState } from "react";
import {
  Search,
  Upload,
  Filter,
  RefreshCw,
  Trash2,
  Eye,
  HardDrive,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import {
  useLoRAList,
  useLoRASearch,
  useLoRAStatistics,
  useRefreshLoRAList,
} from "../../hooks/api/use-lora";
import { LoRACard } from "./LoRACard";
import { LoRAUploadDialog } from "./LoRAUploadDialog";
import { LoRAPreviewDialog } from "./LoRAPreviewDialog";
import { LoadingGrid } from "../gallery/LoadingGrid";
import { EmptyState } from "../gallery/EmptyState";
import type { LoRAInfo } from "../../lib/api-schemas";

interface LoRABrowserProps {
  onLoRASelect?: (lora: LoRAInfo) => void;
  selectedLoRA?: string;
  showSelection?: boolean;
}

export function LoRABrowser({
  onLoRASelect,
  selectedLoRA,
  showSelection = false,
}: LoRABrowserProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [previewLoRA, setPreviewLoRA] = useState<LoRAInfo | null>(null);

  const { loras, totalCount, isLoading, error } = useLoRASearch(
    searchTerm,
    selectedCategory
  );
  const { statistics } = useLoRAStatistics();
  const refreshLoRAList = useRefreshLoRAList();

  const handleRefresh = () => {
    refreshLoRAList();
  };

  const handlePreview = (lora: LoRAInfo) => {
    setPreviewLoRA(lora);
  };

  const handleSelect = (lora: LoRAInfo) => {
    if (onLoRASelect) {
      onLoRASelect(lora);
    }
  };

  if (error) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="text-center">
            <p className="text-red-600 mb-4">Failed to load LoRA files</p>
            <Button onClick={handleRefresh} variant="outline">
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <HardDrive className="w-4 h-4 text-blue-500" />
              <div>
                <p className="text-sm font-medium">Total LoRAs</p>
                <p className="text-2xl font-bold">{statistics.totalCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <HardDrive className="w-4 h-4 text-green-500" />
              <div>
                <p className="text-sm font-medium">Total Size</p>
                <p className="text-2xl font-bold">
                  {statistics.totalSizeMB.toFixed(1)}MB
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Eye className="w-4 h-4 text-orange-500" />
              <div>
                <p className="text-sm font-medium">Loaded</p>
                <p className="text-2xl font-bold">{statistics.loadedCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Badge className="w-4 h-4 bg-purple-500" />
              <div>
                <p className="text-sm font-medium">Applied</p>
                <p className="text-2xl font-bold">{statistics.appliedCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>LoRA Files</span>
            <div className="flex items-center space-x-2">
              <Button
                onClick={handleRefresh}
                variant="outline"
                size="sm"
                disabled={isLoading}
              >
                <RefreshCw
                  className={`w-4 h-4 mr-2 ${isLoading ? "animate-spin" : ""}`}
                />
                Refresh
              </Button>
              <Button onClick={() => setIsUploadDialogOpen(true)} size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Upload LoRA
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search and Filter */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search LoRA files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select
              value={selectedCategory}
              onValueChange={setSelectedCategory}
            >
              <SelectTrigger className="w-full sm:w-48">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue placeholder="Filter by category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                <SelectItem value="style">
                  Style ({statistics.categories.style})
                </SelectItem>
                <SelectItem value="character">
                  Character ({statistics.categories.character})
                </SelectItem>
                <SelectItem value="quality">
                  Quality ({statistics.categories.quality})
                </SelectItem>
                <SelectItem value="loaded">
                  Loaded ({statistics.loadedCount})
                </SelectItem>
                <SelectItem value="applied">
                  Applied ({statistics.appliedCount})
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Results Count */}
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>
              {totalCount} LoRA{totalCount !== 1 ? "s" : ""} found
              {searchTerm && ` for "${searchTerm}"`}
              {selectedCategory !== "all" && ` in ${selectedCategory}`}
            </span>
          </div>

          {/* LoRA Grid */}
          {isLoading ? (
            <LoadingGrid itemCount={6} />
          ) : totalCount === 0 ? (
            <EmptyState
              title="No LoRA files found"
              description={
                searchTerm || selectedCategory !== "all"
                  ? "Try adjusting your search or filter criteria"
                  : "Upload your first LoRA file to get started"
              }
              action={
                <Button onClick={() => setIsUploadDialogOpen(true)}>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload LoRA
                </Button>
              }
            />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {loras.map((lora) => (
                <LoRACard
                  key={lora.name}
                  lora={lora}
                  onPreview={() => handlePreview(lora)}
                  onSelect={
                    showSelection ? () => handleSelect(lora) : undefined
                  }
                  isSelected={selectedLoRA === lora.name}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Upload Dialog */}
      <LoRAUploadDialog
        open={isUploadDialogOpen}
        onOpenChange={setIsUploadDialogOpen}
      />

      {/* Preview Dialog */}
      {previewLoRA && (
        <LoRAPreviewDialog
          lora={previewLoRA}
          open={!!previewLoRA}
          onOpenChange={(open) => !open && setPreviewLoRA(null)}
        />
      )}
    </div>
  );
}
