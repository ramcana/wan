import React, { useState } from "react";
import {
  Eye,
  Trash2,
  Download,
  AlertCircle,
  CheckCircle,
  Clock,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardHeader } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { useLoRADelete, useLoRAMemoryImpact } from "../../hooks/api/use-lora";
import { useToast } from "../../hooks/use-toast";
import { formatDistanceToNow } from "date-fns";
import type { LoRAInfo } from "../../lib/api-schemas";

interface LoRACardProps {
  lora: LoRAInfo;
  onPreview?: () => void;
  onSelect?: () => void;
  isSelected?: boolean;
}

export function LoRACard({
  lora,
  onPreview,
  onSelect,
  isSelected = false,
}: LoRACardProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const { toast } = useToast();

  const deleteLoRA = useLoRADelete();
  const { data: memoryImpact } = useLoRAMemoryImpact(lora.name);

  const handleDelete = async () => {
    if (!showDeleteConfirm) {
      setShowDeleteConfirm(true);
      setTimeout(() => setShowDeleteConfirm(false), 3000); // Auto-hide after 3 seconds
      return;
    }

    try {
      await deleteLoRA.mutateAsync(lora.name);
      setShowDeleteConfirm(false);
    } catch (error) {
      // Error handling is done in the hook
      setShowDeleteConfirm(false);
    }
  };

  const getStatusIcon = () => {
    if (lora.is_applied) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    }
    if (lora.is_loaded) {
      return <Clock className="w-4 h-4 text-blue-500" />;
    }
    return null;
  };

  const getStatusText = () => {
    if (lora.is_applied) {
      return `Applied (${lora.current_strength.toFixed(1)}x)`;
    }
    if (lora.is_loaded) {
      return "Loaded";
    }
    return "Available";
  };

  const getStatusColor = () => {
    if (lora.is_applied) return "bg-green-100 text-green-800";
    if (lora.is_loaded) return "bg-blue-100 text-blue-800";
    return "bg-gray-100 text-gray-800";
  };

  const getMemoryImpactColor = () => {
    if (!memoryImpact) return "bg-gray-100 text-gray-800";

    switch (memoryImpact.memory_impact) {
      case "low":
        return "bg-green-100 text-green-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "high":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const formatFileSize = (sizeMB: number) => {
    if (sizeMB < 1) {
      return `${(sizeMB * 1024).toFixed(0)}KB`;
    }
    return `${sizeMB.toFixed(1)}MB`;
  };

  const modifiedTime = new Date(lora.modified_time);
  const timeAgo = formatDistanceToNow(modifiedTime, { addSuffix: true });

  return (
    <Card
      className={`transition-all duration-200 hover:shadow-md ${
        isSelected ? "ring-2 ring-blue-500 bg-blue-50" : ""
      } ${onSelect ? "cursor-pointer" : ""}`}
      onClick={onSelect}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm truncate" title={lora.name}>
              {lora.name}
            </h3>
            <p className="text-xs text-gray-500 truncate" title={lora.filename}>
              {lora.filename}
            </p>
          </div>
          <div className="flex items-center space-x-1 ml-2">
            {getStatusIcon()}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0 space-y-3">
        {/* Status and Size */}
        <div className="flex items-center justify-between">
          <Badge className={`text-xs ${getStatusColor()}`}>
            {getStatusText()}
          </Badge>
          <span className="text-xs text-gray-500">
            {formatFileSize(lora.size_mb)}
          </span>
        </div>

        {/* Memory Impact */}
        {memoryImpact && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-600">Memory Impact</span>
              <Badge className={`text-xs ${getMemoryImpactColor()}`}>
                {memoryImpact.memory_impact}
              </Badge>
            </div>
            {memoryImpact.can_load ? (
              <div className="flex items-center text-xs text-green-600">
                <CheckCircle className="w-3 h-3 mr-1" />
                Can load ({memoryImpact.estimated_memory_mb.toFixed(0)}MB)
              </div>
            ) : (
              <div className="flex items-center text-xs text-red-600">
                <AlertCircle className="w-3 h-3 mr-1" />
                Insufficient VRAM
              </div>
            )}
          </div>
        )}

        {/* Applied Strength Progress */}
        {lora.is_applied && lora.current_strength > 0 && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-600">Strength</span>
              <span className="font-medium">
                {lora.current_strength.toFixed(1)}x
              </span>
            </div>
            <Progress
              value={(lora.current_strength / 2.0) * 100}
              className="h-1"
            />
          </div>
        )}

        {/* Modified Time */}
        <div className="text-xs text-gray-500">Modified {timeAgo}</div>

        {/* Actions */}
        <div className="flex items-center justify-between pt-2 border-t">
          <div className="flex items-center space-x-1">
            {onPreview && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onPreview();
                }}
                className="h-8 px-2"
              >
                <Eye className="w-3 h-3" />
              </Button>
            )}
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              handleDelete();
            }}
            className={`h-8 px-2 ${
              showDeleteConfirm
                ? "text-red-600 hover:text-red-700 hover:bg-red-50"
                : "text-gray-500 hover:text-red-600"
            }`}
            disabled={deleteLoRA.isLoading}
          >
            <Trash2 className="w-3 h-3" />
            {showDeleteConfirm && (
              <span className="ml-1 text-xs">Confirm?</span>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
