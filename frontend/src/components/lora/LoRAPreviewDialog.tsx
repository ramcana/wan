import React, { useState } from "react";
import { Eye, Sparkles, AlertCircle, Copy, Check } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Badge } from "../ui/badge";
import { Textarea } from "../ui/textarea";
import { useLoRAPreview, useLoRAMemoryImpact } from "../../hooks/api/use-lora";
import { useToast } from "../../hooks/use-toast";
import type { LoRAInfo } from "../../lib/api-schemas";

interface LoRAPreviewDialogProps {
  lora: LoRAInfo;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function LoRAPreviewDialog({
  lora,
  open,
  onOpenChange,
}: LoRAPreviewDialogProps) {
  const [basePrompt, setBasePrompt] = useState("a beautiful landscape");
  const [copiedPrompt, setCopiedPrompt] = useState<string | null>(null);
  const { toast } = useToast();

  const {
    data: preview,
    isLoading: isPreviewLoading,
    refetch: generatePreview,
  } = useLoRAPreview(lora.name, basePrompt, false);

  const { data: memoryImpact } = useLoRAMemoryImpact(lora.name);

  const handleGeneratePreview = () => {
    if (basePrompt.trim()) {
      generatePreview();
    }
  };

  const handleCopyPrompt = async (
    prompt: string,
    type: "original" | "enhanced"
  ) => {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopiedPrompt(type);
      setTimeout(() => setCopiedPrompt(null), 2000);

      toast({
        title: "Copied to clipboard",
        description: `${
          type === "original" ? "Original" : "Enhanced"
        } prompt copied`,
        variant: "default",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Could not copy prompt to clipboard",
        variant: "destructive",
      });
    }
  };

  const formatFileSize = (sizeMB: number) => {
    if (sizeMB < 1) {
      return `${(sizeMB * 1024).toFixed(0)}KB`;
    }
    return `${sizeMB.toFixed(1)}MB`;
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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <Eye className="w-5 h-5 mr-2" />
            LoRA Preview: {lora.name}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* LoRA Information */}
          <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
            <div>
              <Label className="text-sm font-medium text-gray-600">
                File Name
              </Label>
              <p className="text-sm font-mono">{lora.filename}</p>
            </div>
            <div>
              <Label className="text-sm font-medium text-gray-600">
                File Size
              </Label>
              <p className="text-sm">{formatFileSize(lora.size_mb)}</p>
            </div>
            <div>
              <Label className="text-sm font-medium text-gray-600">
                Status
              </Label>
              <div className="flex items-center space-x-2">
                <Badge
                  className={
                    lora.is_applied
                      ? "bg-green-100 text-green-800"
                      : lora.is_loaded
                      ? "bg-blue-100 text-blue-800"
                      : "bg-gray-100 text-gray-800"
                  }
                >
                  {lora.is_applied
                    ? `Applied (${lora.current_strength.toFixed(1)}x)`
                    : lora.is_loaded
                    ? "Loaded"
                    : "Available"}
                </Badge>
              </div>
            </div>
            <div>
              <Label className="text-sm font-medium text-gray-600">
                Memory Impact
              </Label>
              {memoryImpact ? (
                <Badge className={`text-xs ${getMemoryImpactColor()}`}>
                  {memoryImpact.memory_impact} (
                  {memoryImpact.estimated_memory_mb.toFixed(0)}MB)
                </Badge>
              ) : (
                <p className="text-sm text-gray-500">Calculating...</p>
              )}
            </div>
          </div>

          {/* Memory Impact Details */}
          {memoryImpact && (
            <div className="p-4 border rounded-lg">
              <h3 className="font-medium mb-2">Memory Impact Analysis</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Estimated Memory Usage:</span>
                  <span className="font-medium">
                    {memoryImpact.estimated_memory_mb.toFixed(1)}MB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Available VRAM:</span>
                  <span className="font-medium">
                    {memoryImpact.vram_available_mb.toFixed(1)}MB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Can Load:</span>
                  <span
                    className={`font-medium ${
                      memoryImpact.can_load ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {memoryImpact.can_load ? "Yes" : "No"}
                  </span>
                </div>
                <div className="pt-2 border-t">
                  <p className="text-gray-600">{memoryImpact.recommendation}</p>
                </div>
              </div>
            </div>
          )}

          {/* Prompt Preview Generator */}
          <div className="space-y-4">
            <div>
              <Label htmlFor="base-prompt">Base Prompt</Label>
              <div className="flex space-x-2 mt-1">
                <Input
                  id="base-prompt"
                  placeholder="Enter a base prompt to see how this LoRA might enhance it..."
                  value={basePrompt}
                  onChange={(e) => setBasePrompt(e.target.value)}
                  onKeyPress={(e) =>
                    e.key === "Enter" && handleGeneratePreview()
                  }
                />
                <Button
                  onClick={handleGeneratePreview}
                  disabled={!basePrompt.trim() || isPreviewLoading}
                >
                  {isPreviewLoading ? (
                    <Sparkles className="w-4 h-4 animate-pulse" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </div>

            {/* Preview Results */}
            {preview && (
              <div className="space-y-4">
                {/* Style Indicators */}
                {preview.style_indicators.length > 0 && (
                  <div>
                    <Label className="text-sm font-medium">
                      Style Indicators
                    </Label>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {preview.style_indicators.map((indicator, index) => (
                        <Badge
                          key={index}
                          variant="outline"
                          className="text-xs"
                        >
                          {indicator}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Original Prompt */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <Label className="text-sm font-medium">
                      Original Prompt
                    </Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        handleCopyPrompt(preview.base_prompt, "original")
                      }
                      className="h-6 px-2"
                    >
                      {copiedPrompt === "original" ? (
                        <Check className="w-3 h-3 text-green-500" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </Button>
                  </div>
                  <Textarea
                    value={preview.base_prompt}
                    readOnly
                    className="min-h-[60px] text-sm bg-gray-50"
                  />
                </div>

                {/* Enhanced Prompt */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <Label className="text-sm font-medium">
                      Enhanced Prompt
                    </Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        handleCopyPrompt(preview.enhanced_prompt, "enhanced")
                      }
                      className="h-6 px-2"
                    >
                      {copiedPrompt === "enhanced" ? (
                        <Check className="w-3 h-3 text-green-500" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </Button>
                  </div>
                  <Textarea
                    value={preview.enhanced_prompt}
                    readOnly
                    className="min-h-[80px] text-sm bg-blue-50 border-blue-200"
                  />
                </div>

                {/* Preview Note */}
                <div className="flex items-start space-x-2 p-3 bg-amber-50 border border-amber-200 rounded-md">
                  <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-amber-800">
                    {preview.preview_note}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end space-x-2 pt-4 border-t">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
