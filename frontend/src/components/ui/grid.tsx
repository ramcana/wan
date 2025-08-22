import React from "react";
import { cn } from "@/lib/utils";
import { useBreakpoint, getGridCols } from "@/lib/responsive";

interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  cols?: {
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
    "2xl"?: number;
  };
  gap?: "sm" | "md" | "lg" | "xl";
  auto?: boolean;
}

const Grid: React.FC<GridProps> = ({
  children,
  className,
  cols,
  gap = "md",
  auto = false,
  ...props
}) => {
  const breakpoint = useBreakpoint();

  const getGridClass = () => {
    if (auto) return "grid-cols-auto-fit";

    if (cols) {
      const currentCols = cols[breakpoint] || cols.sm || 1;
      return `grid-cols-${currentCols}`;
    }

    const defaultCols = getGridCols(breakpoint);
    return `grid-cols-${defaultCols}`;
  };

  const getGapClass = () => {
    const gapMap = {
      sm: "gap-2",
      md: "gap-4",
      lg: "gap-6",
      xl: "gap-8",
    };
    return gapMap[gap];
  };

  return (
    <div
      className={cn("grid", getGridClass(), getGapClass(), className)}
      {...props}
    >
      {children}
    </div>
  );
};

interface GridItemProps extends React.HTMLAttributes<HTMLDivElement> {
  span?: {
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
    "2xl"?: number;
  };
}

const GridItem: React.FC<GridItemProps> = ({
  children,
  className,
  span,
  ...props
}) => {
  const breakpoint = useBreakpoint();

  const getSpanClass = () => {
    if (!span) return "";

    const currentSpan = span[breakpoint] || span.sm || 1;
    return `col-span-${currentSpan}`;
  };

  return (
    <div className={cn(getSpanClass(), className)} {...props}>
      {children}
    </div>
  );
};

export { Grid, GridItem };
