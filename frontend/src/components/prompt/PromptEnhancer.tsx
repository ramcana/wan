import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { usePromptEnhancement, usePromptPreview } from "@/hooks/api/use-prompt";
import { Sparkles, Eye, Check, X, Loader2, Info, Wand2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface PromptEnhancerProps {
  initialPrompt?: string;
  onPromptChange?: (prompt: string) => void;
  onEnhancementApplied?: (enhancedPrompt: string) => void;
  className?: string;
}

const PromptEnhancer: React.FC<PromptEnhancerProps> = ({
  initialPrompt = "",
  onPromptChange,
  onEnhancementApplied,
  className = "",
}) => {
  const [currentPrompt, setCurrentPrompt] = useState(initialPrompt);
  const [showPreview, setShowPreview] = useState(false);
  const [enhancementOptions, setEnhancementOptions] = useState({
    apply_vace: undefined as boolean | undefined,
    apply_cinematic: true,
    apply_style: true,
  });

  const { toast } = useToast();
  const { enhancePrompt, isEnhancing, enhanceError, lastEnhancement } =
    usePromptEnhancement() as any;

  // Preview enhancement (debounced)
  const { data: preview, isLoading: isLoadingPreview } = usePromptPreview(
    currentPrompt,
    showPreview && currentPrompt.length > 3
  ) as { data: any; isLoading: boolean };

  useEffect(() => {
    setCurrentPrompt(initialPrompt);
  }, [initialPrompt]);


  const handleEnhance = async () => {
    if (!currentPrompt.trim()) {
      toast({
        title: "No prompt provided",
        description: "Please enter a prompt to enhance.",
        variant: "destructive",
      });
      return;
    }

    try {
      const result = await enhancePrompt(currentPrompt, enhancementOptions);

      if (result?.enhanced_prompt) {
        toast({
          title: "Prompt enhanced successfully",
          description: `Generated ${
            result.enhancements_applied?.length || 0
          } enhancement${(result.enhancements_applied?.length || 0) !== 1 ? "s" : ""}. Click Apply to use the enhanced prompt.`,
        });
      }

      // Show the enhancement result
      setShowPreview(false);
    } catch (error) {
      toast({
        title: "Enhancement failed",
        description:
          error instanceof Error ? error.message : "Failed to enhance prompt",
        variant: "destructive",
      });
    }
  };

  const handleApplyEnhancement = () => {
    if (lastEnhancement) {
      const enhancedPrompt = lastEnhancement.enhanced_prompt;
      setCurrentPrompt(enhancedPrompt);
      onPromptChange?.(enhancedPrompt);
      onEnhancementApplied?.(enhancedPrompt);

      toast({
        title: "Enhancement applied",
        description: "The enhanced prompt has been applied to your input.",
      });
    }
  };

  const handleRejectEnhancement = () => {
    toast({
      title: "Enhancement rejected",
      description: "Keeping your original prompt.",
    });
  };

  const togglePreview = () => {
    setShowPreview(!showPreview);
  };

  const getDiffHighlight = (original: string, enhanced: string): string => {
    // Simple diff highlighting - find added content
    if (enhanced && enhanced.includes(original)) {
      const addedContent = enhanced.replace(original, "").trim();
      if (addedContent.startsWith(",")) {
        return addedContent.substring(1).trim();
      }
      return addedContent;
    }
    return enhanced || "";
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Main Enhancement Card */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Wand2 className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-medium">Prompt Enhancement</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={togglePreview}
              disabled={currentPrompt.length < 3}
            >
              <Eye className="h-4 w-4 mr-1" />
              {showPreview ? "Hide Preview" : "Preview"}
            </Button>
            <Button
              onClick={handleEnhance}
              disabled={isEnhancing || currentPrompt.length < 3}
              size="sm"
            >
              {isEnhancing ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 mr-1" />
              )}
              Enhance
            </Button>
          </div>
        </div>

        {/* Enhancement Options */}
        <div className="flex flex-wrap gap-2 mb-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={enhancementOptions.apply_cinematic}
              onChange={(e) =>
                setEnhancementOptions((prev) => ({
                  ...prev,
                  apply_cinematic: e.target.checked,
                }))
              }
              className="rounded"
            />
            Cinematic Style
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={enhancementOptions.apply_style}
              onChange={(e) =>
                setEnhancementOptions((prev) => ({
                  ...prev,
                  apply_style: e.target.checked,
                }))
              }
              className="rounded"
            />
            Style-Specific
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={enhancementOptions.apply_vace === true}
              onChange={(e) =>
                setEnhancementOptions((prev) => ({
                  ...prev,
                  apply_vace: e.target.checked ? true : undefined,
                }))
              }
              className="rounded"
            />
            Force VACE
          </label>
        </div>

        {/* Current Prompt Display */}
        <div className="space-y-2">
          <textarea
            value={currentPrompt}
            onChange={(e) => {
              setCurrentPrompt(e.target.value);
              onPromptChange?.(e.target.value);
            }}
            placeholder="Enter your image generation prompt..."
            className="w-full h-24 p-3 border rounded-md resize-none focus:ring-2 focus:ring-primary focus:border-transparent"
            maxLength={500}
          />
        </div>
        <div className="text-sm text-muted-foreground mb-2">
          {currentPrompt.length}/500 characters
        </div>
      </Card>

      {/* Preview Card */}
      {showPreview && (
        <Card className="p-4 border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20">
          <div className="flex items-center gap-2 mb-3">
            <Eye className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            <h4 className="font-medium text-blue-800 dark:text-blue-200">
              Enhancement Preview
            </h4>
            {isLoadingPreview && (
              <Loader2 className="h-4 w-4 animate-spin text-blue-600 dark:text-blue-400" />
            )}
          </div>

          {preview && (
            <div className="space-y-3">
              {/* Detected Information */}
              <div className="flex flex-wrap gap-2">
                {(preview as any)?.vace_detected && (
                  <Badge
                    variant="secondary"
                    className="bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                  >
                    VACE Detected
                  </Badge>
                )}
                {(preview as any)?.detected_style && (
                  <Badge
                    variant="secondary"
                    className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                  >
                    {String((preview as any).detected_style)} Style
                  </Badge>
                )}
              </div>

              {/* Suggested Enhancements */}
              {(preview as any)?.suggested_enhancements?.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">
                    Suggested Enhancements:
                  </p>
                  <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                    {(preview as any).suggested_enhancements.map(
                      (enhancement: string, index: number) => (
                        <li key={index} className="flex items-start gap-2">
                          <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                          {String(enhancement)}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              )}

              {/* Preview Text */}
              <div>
                <p className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">
                  Enhanced Preview:
                </p>
                <div className="bg-white dark:bg-gray-800 p-3 rounded border text-sm">
                  <span className="text-muted-foreground">{currentPrompt}</span>
                  {(preview as any)?.character_count?.difference > 0 && (
                    <span className="text-blue-600 dark:text-blue-400 font-medium">
                      {getDiffHighlight(
                        currentPrompt,
                        String((preview as any).preview_enhanced || "")
                      )}
                    </span>
                  )}
                </div>
              </div>

              {/* Character Count Change */}
              {(preview as any)?.character_count?.difference !== 0 && (
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  {Number((preview as any).character_count.difference) > 0 ? "+" : ""}
                  {String((preview as any).character_count.difference)} characters
                </p>
              )}
            </div>
          )}
          
          {!preview && !isLoadingPreview && (
            <div className="text-center text-muted-foreground py-4">
              <p>No preview available</p>
            </div>
          )}
        </Card>
      )}

      {/* Enhancement Result */}
      {lastEnhancement && (
        <Card className="p-4 border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-green-600 dark:text-green-400" />
              <h4 className="font-medium text-green-800 dark:text-green-200">
                Enhancement Complete
              </h4>
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={handleRejectEnhancement}
                className="border-red-200 text-red-700 hover:bg-red-50 dark:border-red-800 dark:text-red-300 dark:hover:bg-red-900/20"
              >
                <X className="h-3 w-3 mr-1" />
                Reject
              </Button>
              <Button
                size="sm"
                onClick={handleApplyEnhancement}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <Check className="h-3 w-3 mr-1" />
                Apply
              </Button>
            </div>
          </div>

          {/* Applied Enhancements */}
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {lastEnhancement?.enhancements_applied?.map(
                (enhancement: string, index: number) => (
                  <Badge
                    key={index}
                    variant="secondary"
                    className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                  >
                    {String(enhancement)}
                  </Badge>
                )
              )}
              {lastEnhancement?.vace_detected && (
                <Badge
                  variant="secondary"
                  className="bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                >
                  VACE Style
                </Badge>
              )}
              {lastEnhancement?.detected_style && (
                <Badge
                  variant="secondary"
                  className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                >
                  {lastEnhancement?.detected_style} Style
                </Badge>
              )}
            </div>

            {/* Before/After Comparison */}
            <div className="space-y-2">
              <div>
                <p className="text-xs font-medium text-green-700 dark:text-green-300 mb-1">
                  Original:
                </p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded border text-sm text-muted-foreground">
                  {lastEnhancement?.original_prompt || ""}
                </div>
              </div>
              <div>
                <p className="text-xs font-medium text-green-700 dark:text-green-300 mb-1">
                  Enhanced:
                </p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded border text-sm">
                  <span className="text-muted-foreground">
                    {lastEnhancement?.original_prompt || ""}
                  </span>
                  <span className="text-green-600 dark:text-green-400 font-medium">
                    {getDiffHighlight(
                      lastEnhancement?.original_prompt || "",
                      lastEnhancement?.enhanced_prompt || ""
                    )}
                  </span>
                </div>
              </div>
            </div>

            {/* Character Count */}
            <p className="text-xs text-green-600 dark:text-green-400">
              {lastEnhancement?.character_count?.original || 0} →{" "}
              {lastEnhancement?.character_count?.enhanced || 0} characters (
              {(lastEnhancement?.character_count?.difference || 0) > 0 ? "+" : ""}
              {lastEnhancement?.character_count?.difference || 0})
            </p>
          </div>
        </Card>
      )}

      {/* Error Display */}
      {enhanceError && (
        <Card className="p-4 border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20">
          <div className="flex items-center gap-2">
            <X className="h-4 w-4 text-red-600 dark:text-red-400" />
            <p className="text-red-800 dark:text-red-200 font-medium">
              Enhancement Failed
            </p>
          </div>
          <p className="text-red-700 dark:text-red-300 text-sm mt-1">
            {enhanceError instanceof Error
              ? enhanceError.message
              : "An error occurred while enhancing the prompt"}
          </p>
        </Card>
      )}
    </div>
  );
};

export default PromptEnhancer;
