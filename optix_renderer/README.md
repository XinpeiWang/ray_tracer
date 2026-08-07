# optix_renderer

This directory holds only the MSBuild project file (`optix_renderer.vcxproj`)
for the GPU renderer. All of its source — the OptiX renderer, scene builder,
recursive/wavefront path-tracing strategies, and `.cu`/PTX device code —
lives under [`../gpu/optix/`](../gpu/optix/), which the `.vcxproj` references
directly (`..\gpu\optix\*`).

If you're looking for the actual OptiX implementation, start in `gpu/optix/`,
not here.
