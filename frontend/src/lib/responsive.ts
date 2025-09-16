import { useEffect, useState } from 'react'

export type Breakpoint = 'sm' | 'md' | 'lg' | 'xl' | '2xl'

export const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
}

export function useBreakpoint(): Breakpoint {
  const [breakpoint, setBreakpoint] = useState<Breakpoint>('lg')

  useEffect(() => {
    const updateBreakpoint = () => {
      const width = window.innerWidth
      if (width >= breakpoints['2xl']) setBreakpoint('2xl')
      else if (width >= breakpoints.xl) setBreakpoint('xl')
      else if (width >= breakpoints.lg) setBreakpoint('lg')
      else if (width >= breakpoints.md) setBreakpoint('md')
      else setBreakpoint('sm')
    }

    updateBreakpoint()
    window.addEventListener('resize', updateBreakpoint)
    return () => window.removeEventListener('resize', updateBreakpoint)
  }, [])

  return breakpoint
}

export function getGridCols(breakpoint: Breakpoint): number {
  switch (breakpoint) {
    case 'sm': return 1
    case 'md': return 2
    case 'lg': return 3
    case 'xl': return 4
    case '2xl': return 5
    default: return 3
  }
}

export function getContainerMaxWidth(size: 'sm' | 'md' | 'lg' | 'xl' | 'full' = 'lg'): string {
  switch (size) {
    case 'sm': return 'max-w-2xl'
    case 'md': return 'max-w-4xl'
    case 'lg': return 'max-w-6xl'
    case 'xl': return 'max-w-7xl'
    case 'full': return 'max-w-full'
    default: return 'max-w-6xl'
  }
}

export function getResponsivePadding(size: 'sm' | 'md' | 'lg' = 'md'): string {
  switch (size) {
    case 'sm': return 'px-4 sm:px-6'
    case 'md': return 'px-4 sm:px-6 lg:px-8'
    case 'lg': return 'px-6 sm:px-8 lg:px-12'
    default: return 'px-4 sm:px-6 lg:px-8'
  }
}
