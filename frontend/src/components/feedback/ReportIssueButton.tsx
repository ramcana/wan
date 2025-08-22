/**
 * Report Issue Button Component
 * Provides a simple feedback mechanism for users to report issues
 * Requirements: 1.4, 3.4, 6.5
 */

import React, { useState } from "react";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import { Textarea } from "../ui/textarea";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { useToast } from "../../hooks/use-toast";
import { AlertTriangle, Send } from "lucide-react";

interface ReportIssueButtonProps {
  context?: {
    page?: string;
    action?: string;
    error?: string;
  };
  variant?: "default" | "outline" | "ghost";
  size?: "default" | "sm" | "lg" | "icon";
}

interface IssueReport {
  type: string;
  description: string;
  context: string;
  userAgent: string;
  timestamp: string;
}

export const ReportIssueButton: React.FC<ReportIssueButtonProps> = ({
  context,
  variant = "outline",
  size = "sm",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [issueType, setIssueType] = useState("");
  const [description, setDescription] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!issueType || !description.trim()) {
      toast({
        title: "Validation Error",
        description: "Please select an issue type and provide a description.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      const report: IssueReport = {
        type: issueType,
        description: description.trim(),
        context: JSON.stringify({
          ...context,
          url: window.location.href,
          timestamp: new Date().toISOString(),
        }),
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
      };

      // In a real implementation, this would send to a backend endpoint
      // For now, we'll log it and show success
      console.log("Issue Report:", report);

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      toast({
        title: "Issue Reported",
        description: "Thank you for your feedback. We'll look into this issue.",
        variant: "default",
      });

      // Reset form
      setIssueType("");
      setDescription("");
      setIsOpen(false);
    } catch (error) {
      toast({
        title: "Report Failed",
        description: "Failed to submit issue report. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant={variant}
          size={size}
          data-testid="report-issue-button"
          className="gap-2"
        >
          <AlertTriangle className="h-4 w-4" />
          Report Issue
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Report an Issue</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="issue-type">Issue Type</Label>
            <Select value={issueType} onValueChange={setIssueType}>
              <SelectTrigger data-testid="issue-type-select">
                <SelectValue placeholder="Select issue type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bug">Bug / Error</SelectItem>
                <SelectItem value="performance">Performance Issue</SelectItem>
                <SelectItem value="ui">UI / Design Issue</SelectItem>
                <SelectItem value="feature">Feature Request</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              data-testid="issue-description"
              placeholder="Please describe the issue you encountered..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              maxLength={1000}
            />
            <div className="text-sm text-muted-foreground text-right">
              {description.length}/1000
            </div>
          </div>

          {context && (
            <div className="space-y-2">
              <Label>Context Information</Label>
              <div className="text-sm text-muted-foreground bg-muted p-2 rounded">
                <div>Page: {context.page || "Unknown"}</div>
                <div>Action: {context.action || "Unknown"}</div>
                {context.error && (
                  <div>
                    Error:{" "}
                    {typeof context.error === "string"
                      ? context.error
                      : String(context.error)}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setIsOpen(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isSubmitting}
              data-testid="submit-issue-button"
              className="gap-2"
            >
              {isSubmitting ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Submitting...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  Submit Report
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default ReportIssueButton;
