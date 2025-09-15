# Re-export the real loader so imports like backend.core.services.WanPipelineLoader work.
try:
    from backend.services.wan_pipeline_loader import WanPipelineLoader  # real loader
    __all__ = ["WanPipelineLoader"]
except Exception:
    # Keep the package importable even if loader import fails; the integrator will handle fallback.
    __all__ = []