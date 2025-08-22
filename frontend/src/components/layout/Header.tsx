import React from "react";
import { useLocation } from "react-router-dom";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { Button } from "@/components/ui/button";
import { Settings, HelpCircle } from "lucide-react";

const Header: React.FC = () => {
  const location = useLocation();

  const getPageTitle = (pathname: string) => {
    switch (pathname) {
      case "/":
      case "/generation":
        return "Video Generation";
      case "/queue":
        return "Generation Queue";
      case "/system":
        return "System Monitor";
      case "/outputs":
        return "Generated Videos";
      default:
        return "Wan2.2 Studio";
    }
  };

  return (
    <header className="h-16 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-full items-center justify-between px-6">
        <div>
          <h2 className="text-xl font-semibold text-foreground">
            {getPageTitle(location.pathname)}
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0"
            aria-label="Help"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>

          <Button
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0"
            aria-label="Settings"
          >
            <Settings className="h-4 w-4" />
          </Button>

          <ThemeToggle />
        </div>
      </div>
    </header>
  );
};

export default Header;
