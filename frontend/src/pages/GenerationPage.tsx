import React, { useState } from "react";
import { GenerationForm } from "@/components/generation/GenerationForm";
import { useGenerateVideo } from "@/hooks/api/use-generation";
import { GenerationFormData } from "@/lib/api-schemas";
import { useToast } from "@/hooks/use-toast";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock, Zap, Cpu } from "lucide-react";

const GenerationPage: React.FC = () => {
  const { toast } = useToast();
  const generateVideoMutation = useGenerateVideo();
  const [lastSubmittedData, setLastSubmittedData] =
    useState<GenerationFormData | null>(null);

  const handleGenerationSubmit = async (
    formData: GenerationFormData,
    imageFile?: File,
    endImageFile?: File
  ) => {
    setLastSubmittedData(formData);
    try {
      const result = await generateVideoMutation.mutateAsync({
        formData,
        imageFile,
        endImageFile,
      });

      toast({
        title: "Generation Started",
        description: `Your video has been added to the queue. Task ID: ${result.task_id}`,
      });

      // Clear the last submitted data on success
      setLastSubmittedData(null);
    } catch (error) {
      // Don't show toast for errors - let the form handle error display
      console.error("Generation submission failed:", error);
      throw error; // Re-throw to let form handle it
    }
  };

  const handleRetry = () => {
    if (lastSubmittedData) {
      handleGenerationSubmit(lastSubmittedData);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Video Generation</h1>
        <p className="text-muted-foreground">
          Create videos from text prompts using advanced AI models
        </p>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant="default" className="text-xs">
                Ready
              </Badge>
              <span className="text-sm text-muted-foreground">T2V-A14B</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Queue Status</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-xs">
                0 Pending
              </Badge>
              <span className="text-sm text-muted-foreground">
                Ready for new tasks
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Hardware</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                RTX 4080
              </Badge>
              <span className="text-sm text-muted-foreground">Optimized</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Generation Form */}
      <GenerationForm
        onSubmit={handleGenerationSubmit}
        isSubmitting={generateVideoMutation.isLoading}
        error={generateVideoMutation.error || null}
        onRetry={handleRetry}
      />

      {/* Help Section */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">Tips for Better Results</CardTitle>
          <CardDescription>
            Follow these guidelines to create amazing videos
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-medium mb-2">Prompt Writing</h4>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Be specific about visual details</li>
                <li>• Include style and mood descriptors</li>
                <li>• Mention camera movements if desired</li>
                <li>• Use descriptive adjectives</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Performance Tips</h4>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Start with 720p for faster results</li>
                <li>• Use 25-50 steps for good quality</li>
                <li>• Enable advanced settings for fine-tuning</li>
                <li>• Monitor VRAM usage in system panel</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GenerationPage;
