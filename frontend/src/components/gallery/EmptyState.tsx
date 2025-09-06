import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Video, Search, Filter, Plus } from "lucide-react";
import { Link } from "react-router-dom";

export interface EmptyStateProps {
  hasVideos: boolean;
  hasFilters: boolean;
  onClearFilters: () => void;
  className?: string;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  hasVideos,
  hasFilters,
  onClearFilters,
  className = "",
}) => {
  if (!hasVideos) {
    // No videos at all
    return (
      <Card className={`${className}`}>
        <CardContent className="p-12 text-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="p-4 bg-muted rounded-full">
              <Video className="w-12 h-12 text-muted-foreground" />
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-semibold">No videos yet</h3>
              <p className="text-muted-foreground max-w-md">
                You haven't generated any videos yet. Start creating your first
                video to see it here.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 mt-6">
              <Button asChild>
                <Link to="/generate">
                  <Plus className="w-4 h-4 mr-2" />
                  Generate Your First Video
                </Link>
              </Button>

              <Button variant="outline" asChild>
                <Link to="/queue">View Queue</Link>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (hasFilters) {
    // Has videos but filtered results are empty
    return (
      <Card className={`${className}`}>
        <CardContent className="p-12 text-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="p-4 bg-muted rounded-full">
              <Search className="w-12 h-12 text-muted-foreground" />
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-semibold">No matching videos</h3>
              <p className="text-muted-foreground max-w-md">
                No videos match your current search criteria. Try adjusting your
                filters or search terms.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 mt-6">
              <Button onClick={onClearFilters}>
                <Filter className="w-4 h-4 mr-2" />
                Clear All Filters
              </Button>

              <Button variant="outline" asChild>
                <Link to="/generate">
                  <Plus className="w-4 h-4 mr-2" />
                  Generate New Video
                </Link>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Fallback (shouldn't happen)
  return (
    <Card className={`${className}`}>
      <CardContent className="p-12 text-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="p-4 bg-muted rounded-full">
            <Video className="w-12 h-12 text-muted-foreground" />
          </div>

          <div className="space-y-2">
            <h3 className="text-xl font-semibold">Something went wrong</h3>
            <p className="text-muted-foreground max-w-md">
              Unable to load videos. Please try refreshing the page.
            </p>
          </div>

          <Button onClick={() => window.location.reload()}>Refresh Page</Button>
        </div>
      </CardContent>
    </Card>
  );
};

export { EmptyState };
export default EmptyState;
