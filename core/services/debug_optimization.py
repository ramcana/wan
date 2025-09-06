"""Debug optimization recommendations"""

from core.services.optimization_manager import OptimizationManager, SystemResources, ModelRequirements

# Create minimal system (same as test)
minimal_system = SystemResources(
    total_vram_mb=4096,  # 4GB
    available_vram_mb=3072,  # 3GB available
    total_ram_mb=8192,
    available_ram_mb=4096,
    gpu_name="GTX 1660",
    gpu_compute_capability=(7, 5),
    cpu_cores=4,
    supports_mixed_precision=True,
    supports_cpu_offload=True
)

# Create large model (same as test)
large_model = ModelRequirements(
    min_vram_mb=8192,
    recommended_vram_mb=12288,
    model_size_mb=10240,
    supports_mixed_precision=True,
    supports_cpu_offload=True,
    supports_chunked_processing=True,
    component_sizes={"transformer": 6144, "vae": 2048, "text_encoder": 1024}
)

manager = OptimizationManager()
plan = manager.recommend_optimizations(large_model, minimal_system)

print("System:")
print(f"  Available VRAM: {minimal_system.available_vram_mb}MB")
print(f"  Safety margin: {manager.config['vram_safety_margin_mb']}MB")
print(f"  Effective VRAM: {minimal_system.available_vram_mb - manager.config['vram_safety_margin_mb']}MB")

print("\nModel:")
print(f"  Recommended VRAM: {large_model.recommended_vram_mb}MB")

print("\nPlan:")
print(f"  Mixed precision: {plan.use_mixed_precision} ({plan.precision_type})")
print(f"  CPU offload: {plan.enable_cpu_offload} ({plan.offload_strategy})")
print(f"  Chunk frames: {plan.chunk_frames} (size: {plan.max_chunk_size})")
print(f"  VRAM reduction: {plan.estimated_vram_reduction:.1%}")
print(f"  Performance impact: {plan.estimated_performance_impact:.1%}")

print("\nSteps:")
for step in plan.optimization_steps:
    print(f"  - {step}")

print("\nWarnings:")
for warning in plan.warnings:
    print(f"  - {warning}")

# Calculate what should happen
effective_vram = minimal_system.available_vram_mb - manager.config['vram_safety_margin_mb']
print(f"\nDebug calculations:")
print(f"  Effective VRAM: {effective_vram}MB")
print(f"  Model needs: {large_model.recommended_vram_mb}MB")

# After mixed precision (35% reduction)
after_mixed = large_model.recommended_vram_mb * 0.65
print(f"  After mixed precision: {after_mixed:.0f}MB")

# After CPU offload (60% additional reduction)
after_offload = after_mixed * 0.4
print(f"  After CPU offload: {after_offload:.0f}MB")

print(f"  Still need chunking? {after_offload > effective_vram}")