# Scene Selection System

## Overview

The renderer ships with a large, growing library of built-in scenes (140 at
last count, per `tests/unit/scene_registry_tests.cpp`'s own pinned
`builtin_scene_count()` check) covering the original "Ray Tracing in One Weekend" series, a
wide sweep of pbrt-v4 materials/lights/cameras/volumes/shapes, imported mesh
models, full textured environments, and curated demos of specific render
options. Scenes can also be loaded at runtime from `.pbrt` scene files found
on disk (`pbrt_scenes/`), with no code changes required.

**Every scene runs on both the CPU renderer and both GPU (OptiX) backends**
with real importance sampling (NEE + MIS) on all three - there is no
GPU-only-supports-Cornell-Box limitation. See
[`FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md) and
[`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) for the full current material/light/
shape support matrix per backend; a small number of individual scenes are
marked CPU-only or GPU-only in the registry when a specific feature they use
genuinely isn't ported to a given backend yet - that is scene-specific, not
a blanket GPU limitation.

## Scene IDs

Scenes are identified by a **category letter + a number within that
category** (e.g. `A1`, `B10`, `G25`) rather than a flat integer - this is
the id used everywhere: the CLI, the GUI, `scene_metadata.dll`, and tests.
The category letters, in GUI tab order:

| Letter | Category | Contents |
|--------|----------|----------|
| A | Basics | The original "Ray Tracing in One Weekend"/"Next Week" book scenes |
| B | Materials | BxDF / surface appearance sweep |
| C | Lights | Light types and sampling strategies |
| D | Cameras | Projection and lens models |
| E | Volumes | Participating media |
| F | Geometry | Shape primitives |
| G | Models | Single imported meshes |
| H | Large Scenes | Full textured environments |
| I | Education | Curated demos of specific Render Options tab controls (Sampler, Spectral rendering, Exposure, Tone mapping, Integrator) |
| J | Textures | Texture-system demos (encoding/wrap/invert, procedural texture classes, nested texture references) - split out of Materials once it grew past 25 scenes mixing both concerns |
| K | Custom Scenes | Loaded live from `.pbrt` files on disk (`pbrt_scenes/`) - empty until you add one |

The full, authoritative scene table (id, name, description, performance
hint, recommended SPP, GPU compatibility, camera) lives in
`src/TheRestOfYourLife/scene_registry.h`'s `get_builtin_scene_registry()` -
that is the single source of truth this document intentionally does not
duplicate (a hand-maintained copy here would just drift out of sync again,
which is exactly what happened to the previous version of this file).

## GUI Usage

The scene selector lives on the **Basic Settings** tab, in the "Scene"
group box:
1. **Availability tabs** - "Self-Contained" vs "Requires External Files"
   (mesh/texture scenes that need assets not guaranteed to be on disk)
2. **Category tabs** - one per letter above (Basics, Materials, Lights, ...)
3. **Search box** - filters the dropdown below by name/id/description
   substring, on top of the category/availability filters
4. **Scene dropdown** - the actual selector
5. **Info panel** - shows the scene's description, performance hint,
   recommended SPP, and GPU-compatibility badge, all read live from
   `scene_metadata.dll` (see below), never hardcoded in the GUI

Selecting a scene automatically applies its recommended SPP.

### Where the GUI gets scene metadata

The GUI does **not** hardcode scene names/descriptions/camera info. It
loads them at runtime from `scene_metadata.dll` (built from
`scene_metadata.vcxproj`), which wraps the same `scene_registry.h` table
the renderer itself uses. **This DLL must be rebuilt whenever
`scene_registry.h` changes**, or the GUI will show a stale scene list -
`RayTracerGUI.exe` holds a lock on the DLL while running, so close it
before rebuilding.

## Command-Line Usage

```
ray_tracer.exe [--cpu|--gpu] [--output PATH] width spp max_depth SCENE_ID [cam_x cam_y cam_z]
```

```bash
# Cornell Box (A1), GPU, defaults
ray_tracer.exe --gpu --output out.png 800 100 50 A1

# A Materials-category scene on CPU
ray_tracer.exe --cpu --output out.png 800 200 50 B10

# Override the camera position (falls back to the scene's own recommended
# camera - scene_registry.h's CameraConfig - if omitted)
ray_tracer.exe --cpu --output out.png 800 100 50 A1 278 278 -800
```

A scene's own recommended camera (position, lookat, vfov, DOF) is baked
into its `SceneDescriptor` entry and used automatically if you don't pass
`cam_x cam_y cam_z` explicitly.

## Adding a New Scene

This is already documented at the top of
`src/TheRestOfYourLife/scene_registry.h` itself (kept there, not
duplicated here, so it can't drift out of sync with the actual code):

1. Add a `SceneNames::` constant in `src/shared/scene_descriptor.h`
2. Add a builder function in `src/TheRestOfYourLife/scenes.h` (or
   `scenes_book.h`/`scenes_advanced.h`)
3. Add one `SceneDescriptor` entry in `get_builtin_scene_registry()`
   (`scene_registry.h`) using the `SceneNames::` constant

That's it - `cpu_interface.cpp`'s C API and `scene_metadata.dll` (and
through it, the GUI) pick it up automatically. Optionally add a case in
`gpu/optix/scene_builder.cpp` and set the registry entry's
`gpu_compatible = true` for GPU support; `scene_metadata.dll` picks that up
automatically too.

## Custom `.pbrt` Scenes

Any `.pbrt` file placed in `pbrt_scenes/` (or a path passed via the
relevant CLI/env mechanism - see `src/shared/pbrt_discover.h`) is
discovered at startup and appears under the **Custom Scenes** category with
no code changes. See [`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) for the current
pbrt-v4 directive support matrix.

## Troubleshooting

**"Requires external assets" warning** - the scene needs mesh/texture files
not present on disk (e.g. a large imported model). Check the scene's info
panel/description for what's needed; some assets are tracked via Git LFS
(`git lfs pull` if they show up as small pointer files).

**Scene renders too slowly** - reduce SPP or resolution for a faster
preview; check the info panel's recommended SPP first, since some scenes
(volumes, high-poly meshes) are inherently slower than others.

**GPU renders differently from CPU** - some individual scenes are marked
CPU-only or GPU-only in the registry for feature-specific reasons (rare,
not a blanket limitation) - the GUI's GPU-compatibility badge and CLI
`gpu_compatible` field flag these; everything else should be visually
consistent across CPU/GPU-recursive/GPU-wavefront within normal Monte
Carlo noise.
