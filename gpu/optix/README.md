# OptiX GPU Renderer

This directory holds **all three** GPU backends this project ships, plus the
scene-building code shared between them. It's one flat directory rather than
per-backend subdirectories deliberately: all three compile into a single
OptiX pipeline/PTX build (`build_optix.targets`), and splitting into
subdirectories would fight that shared build rather than clarify it. Files
are distinguished by prefix instead - see File Structure below.

## The three backends

- **Recursive** (default `--gpu`): a mega-kernel OptiX path tracer. Files
  with no backend prefix (`optix_programs.cu`, `optix_device_helpers.h`,
  `optix_intersection_*.h`, `optix_renderer*.cpp/.h`, ...).
- **Wavefront** (`--gpu --wavefront`): a queue-based path tracer, its own
  separately-compiled OptiX pipeline. `wavefront_*.cu/.h`. OptiX rejects
  combining intersection/closest-hit programs from separately-compiled
  modules with different `pipelineCompileOptions.numPayloadValues`, so this
  backend has its own copies of sphere/quad/triangle/disk/cylinder
  intersection logic rather than sharing the recursive backend's - a
  deliberate duplication, not an oversight, if you find yourself wondering
  why the same-looking logic exists twice.
- **SPPM** (`--gpu --sppm`, GPU photon mapping): `sppm_*.cu/.h`.

## File Structure (representative, not exhaustive)

- `optix_types.h` - shared host/device data structures (materials, geometry,
  launch params) - the common vocabulary all three backends read/write.
- `optix_interface.h`/`.cpp` - C API wrapper the CPU-side launcher/GUI call.
- `optix_renderer.h` + `optix_renderer_init.cpp`/`_scene.cpp`/`_render.cpp` -
  the recursive backend's host-side `OptiXRenderer` class, split by concern
  (init/scene-build/render) rather than one large file.
- `optix_programs.cu` - recursive backend's OptiX device programs (raygen,
  miss, closest-hit, intersection).
- `optix_device_helpers.h` - recursive backend's shared `__device__` helpers
  (material shading dispatch, NEE/light sampling).
- `scene_builder.cpp`/`.h` - converts both native hand-authored demo scenes
  and pbrt-v4-loaded scenes into GPU-uploadable `SceneData`.
- `pbrt_gpu_builder.h` - the pbrt-v4 loader's GPU-side half: takes a
  flattened pbrt scene (`src/shared/pbrt_flatten.h`'s `FlatScene`) and builds
  `SceneData` from it. The CPU-side equivalent is
  `src/TheRestOfYourLife/pbrt_cpu_builder.h`.

## Build System

OptiX/CUDA programs (`.cu` files) are compiled to PTX (recursive, wavefront)
or device-linked object code (wavefront's non-OptiX kernels, SPPM) at build
time via `build_optix.targets`. The host code loads PTX modules at runtime
and creates the OptiX pipeline(s).

## OptiX Pipeline (recursive backend)

1. **Ray Generation**: creates primary rays from the camera.
2. **Intersection**: custom programs for spheres/quads/bilinear
   patches/disks/cylinders; triangles use OptiX's own hardware primitive.
3. **Closest Hit**: shades the hit point based on material.
4. **Miss**: returns sky/background/infinite-light color.
5. **Any Hit**: used for shadow rays and transmissive-material shadowing.

## Materials

Every pbrt-v4 material class has a real implementation on all three
backends - see [`docs/FEATURE_INVENTORY.md`](../../docs/FEATURE_INVENTORY.md)
for the full, current list and per-backend caveats; this directory tracks
that document, not the other way around.

## Requirements

- OptiX SDK 7.0+ (tested with 9.1.0)
- CUDA 11.0+ (compiled with 13.2)
- NVIDIA GPU with RT cores (Turing/Ampere/Ada/Hopper/Blackwell)
- Windows 10/11 with recent NVIDIA drivers (535.00+)
