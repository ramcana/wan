"""Debug low memory scenario"""

from core.services.optimization_manager import OptimizationManager, SystemResources, ModelRequirements

# Low-end system (from test)
system = SystemResources(
    total_vram_mb=6144, available_vram_mb=4096,
    total_ram_mb=16384, available_ram_mb=8192,
    gpu_name="RTX 3060", gpu_compute_capability=(8, 6),
    cpu_cores=8, supports_mixed_precision=True, supports_cpu_offload=True
)

# Large model that doesn't fit (from test)
model = ModelRequirements(
    min_vram_mb=8192, recommended_vram_mb=12288, model_size_mb=10240,
    supports_mixed_precision=True, supports_cpu_offload=True,
    supports_chunked_processing=True, component_sizes={}
)

manager = OptimizationManager()
plan = manager.recommend_optimizations(model, system)

print("System:")
print(f"  Available VRAM: {system.available_vram_mb}MB")
print(f"  Safety margin: {manager.config['vram_safety_margin_mb']}MB")
print(f"  Effective VRAM: {system.available_vram_mb - manager.config['vram_safety_margin_mb']}MB")

print("\nModel:")
print(f"  Recommended VRAM: {model.recommended_vram_mb}MB")

print("\nPlan:")
print(f"  Mixed precision: {plan.use_mixed_precision} ({plan.precision_type})")
print(f"  CPU offload: {plan.enable_cpu_offload} ({plan.offload_strategy})")
print(f"  Chunk frames: {plan.chunk_frames} (size: {plan.max_chunk_size})")
print(f"  VRAM reduction: {plan.estimated_vram_reduction:.1%}")

# Calculate what should happen
effective_vram = system.available_vram_mb - manager.config['vram_safety_margin_mb']
print(f"\nDebug calculations:")
print(f"  Effective VRAM: {effective_vram}MB")
print(f"  Model needs: {model.recommended_vram_mb}MB")

# After mixed precision (40% reduction for bf16)
after_mixed = model.recommended_vram_mb * 0.6
print(f"  After mixed precision: {after_mixed:.0f}MB")

# After CPU offload (60% additional reduction)
after_offload = after_mixed * 0.4
print(f"  After CPU offload: {after_offload:.0f}MB")

print(f"  Still need chunking? {after_offload > effective_vram}")
print(f"  Final VRAM need: {model.recommended_vram_mb * (1 - plan.estimated_vram_reduction):.0f}MB")