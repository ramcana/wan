import React, { useState } from "react";
import {
  ChevronDown,
  X,
  Eye,
  AlertCircle,
  CheckCircle,
  Sparkles,
} from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Card, CardContent } from "../ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { useLoRAList, useLoRAMemoryImpact } from "../../hooks/api/use-lora";
import { LoRAStrengthControl } from "../lora/LoRAStrengthControl";
import { LoRAPreviewDialog } from "../lora/LoRAPreviewDialog";
import type { LoRAInfo } from "../../lib/api-schemas";

interface LoRASelectorProps {
  selectedLoRA?: string;
  onLoRASelect: (loraPath?: string) => void;
  loraStrength: number;
  onLoRAStrengthChange: (strength: number) => void;
}

export function LoRASelector({
  selectedLoRA,
  onLoRASelect,
  loraStrength,
  onLoRAStrengthChange,
}: LoRASelectorProps) {
  const [showPreview, setShowPreview] = useState(false);
  const { data: loraList, isLoading, error } = useLoRAList();

  // Find the selected LoRA info
  const selectedLoRAInfo = loraList?.loras.find(
    (lora) => lora.path === selectedLoRA
  );
  const selectedLoRAName = selectedLoRAInfo?.name;

  const { data: memoryImpact } = useLoRAMemoryImpact(
    selectedLoRAName || "",
    !!selectedLoRAName
  );

  const handleLoRASelect = (value: string) => {
    if (value === "none") {
      onLoRASelect(undefined);
    } else {
      onLoRASelect(value);
    }
  };

  const handleClearSelection = () => {
    onLoRASelect(undefined);
  };

  const handlePreview = () => {
    if (selectedLoRAInfo) {
      setShowPreview(true);
    }
  };

  const getMemoryImpactIndicator = () => {
    if (!memoryImpact) return null;

    const color = memoryImpact.can_load
      ? memoryImpact.memory_impact === "low"
        ? "text-green-600"
        : memoryImpact.memory_impact === "medium"
        ? "text-yellow-600"
        : "text-orange-600"
      : "text-red-600";

    const icon = memoryImpact.can_load ? CheckCircle : AlertCircle;
    const IconComponent = icon;

    return (
      <div className={`flex items-center space-x-1 text-xs ${color}`}>
        <IconComponent className="w-3 h-3" />
        <span>
          {memoryImpact.can_load
            ? `${memoryImpact.memory_impact} impact`
            : "Insufficient VRAM"}
        </span>
      </div>
    );
  };

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent className="p-4">
          <div className="flex items-center space-x-2 text-red-600">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Failed to load LoRA files</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* LoRA Selection */}
      <div className="space-y-2">
        <Select
          value={selectedLoRA || "none"}
          onValueChange={handleLoRASelect}
          disabled={isLoading}
        >
          <SelectTrigger className="w-full">
            <SelectValue
              placeholder={
                isLoading ? "Loading LoRAs..." : "Select a LoRA (optional)"
              }
            />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">
              <div className="flex items-center space-x-2">
                <span>No LoRA</span>
                <Badge variant="outline" className="text-xs">
                  Default
                </Badge>
              </div>
            </SelectItem>
            {loraList?.loras.map((lora) => (
              <SelectItem key={lora.path} value={lora.path}>
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium">{lora.name}</span>
                    {lora.is_applied && (
                      <Badge className="text-xs bg-green-100 text-green-800">
                        Applied
                      </Badge>
                    )}
                    {lora.is_loaded && !lora.is_applied && (
                      <Badge className="text-xs bg-blue-100 text-blue-800">
                        Loaded
                      </Badge>
                    )}
                  </div>
                  <span className="text-xs text-gray-500">
                    {lora.size_mb.toFixed(1)}MB
                  </span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Selected LoRA Info */}
        {selectedLoRAInfo && (
          <Card className="bg-blue-50 border-blue-200">
            <CardContent className="p-3">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-blue-600" />
                    <span className="font-medium text-sm truncate">
                      {selectedLoRAInfo.name}
                    </span>
                    <Badge className="text-xs bg-blue-100 text-blue-800">
                      {selectedLoRAInfo.size_mb.toFixed(1)}MB
                    </Badge>
                  </div>
                  <div className="mt-1">{getMemoryImpactIndicator()}</div>
                </div>
                <div className="flex items-center space-x-1 ml-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handlePreview}
                    className="h-7 px-2"
                  >
                    <Eye className="w-3 h-3" />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleClearSelection}
                    className="h-7 px-2 text-red-600 hover:text-red-700"
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* LoRA Strength Control */}
      {selectedLoRA && (
        <div className="space-y-2">
          <LoRAStrengthControl
            value={loraStrength}
            onChange={onLoRAStrengthChange}
            label="LoRA Strength"
            description="Controls how strongly the LoRA affects the generation"
            showPresets={true}
          />
        </div>
      )}

      {/* Memory Warning */}
      {memoryImpact && !memoryImpact.can_load && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-3">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-red-800">
                <p className="font-medium">VRAM Warning</p>
                <p>{memoryImpact.recommendation}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* No LoRAs Available */}
      {loraList && loraList.loras.length === 0 && (
        <Card className="border-gray-200 bg-gray-50">
          <CardContent className="p-4 text-center">
            <div className="text-sm text-gray-600">
              <p className="font-medium mb-1">No LoRA files available</p>
              <p>
                Upload LoRA files to enhance your generations with custom
                styles.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Preview Dialog */}
      {selectedLoRAInfo && (
        <LoRAPreviewDialog
          lora={selectedLoRAInfo}
          open={showPreview}
          onOpenChange={setShowPreview}
        />
      )}
    </div>
  );
}
