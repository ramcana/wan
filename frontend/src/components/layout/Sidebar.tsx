import React from "react";
import { NavLink } from "react-router-dom";
import {
  Video,
  ListTodo,
  Monitor,
  FolderOpen,
  Sparkles,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";

const navigation = [
  {
    name: "Generation",
    href: "/generation",
    icon: Video,
    description: "Create new videos (Alt+1)",
    shortcut: "Alt+1",
  },
  {
    name: "Queue",
    href: "/queue",
    icon: ListTodo,
    description: "Manage generation queue (Alt+2)",
    shortcut: "Alt+2",
  },
  {
    name: "LoRA",
    href: "/lora",
    icon: Sparkles,
    description: "Manage LoRA files (Alt+3)",
    shortcut: "Alt+3",
  },
  {
    name: "System",
    href: "/system",
    icon: Monitor,
    description: "Monitor system resources (Alt+4)",
    shortcut: "Alt+4",
  },
  {
    name: "Outputs",
    href: "/outputs",
    icon: FolderOpen,
    description: "Browse generated videos (Alt+5)",
    shortcut: "Alt+5",
  },
];

const Sidebar: React.FC = () => {
  const { sidebarCollapsed, toggleSidebar } = useAppStore();

  return (
    <div
      className={cn(
        "bg-card border-r border-border transition-all duration-300",
        sidebarCollapsed ? "w-16" : "w-64"
      )}
    >
      <div className="flex h-16 items-center justify-between px-4 border-b border-border">
        {!sidebarCollapsed && (
          <h1 className="text-lg font-semibold text-foreground">
            Wan2.2 Studio
          </h1>
        )}
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-md hover:bg-accent transition-colors"
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <ChevronRight
            className={cn(
              "h-4 w-4 transition-transform",
              sidebarCollapsed ? "rotate-0" : "rotate-180"
            )}
          />
        </button>
      </div>

      <nav className="p-4 space-y-2">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground"
              )
            }
            title={sidebarCollapsed ? item.description : undefined}
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!sidebarCollapsed && (
              <div className="flex-1 flex items-center justify-between">
                <span className="truncate">{item.name}</span>
                <span className="text-xs text-muted-foreground opacity-60">
                  {item.shortcut}
                </span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;
