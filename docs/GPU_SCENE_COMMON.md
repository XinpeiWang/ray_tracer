# Proposal: one scene-to-GPU conversion for OptiX and Metal

Status: **proposal, not started** (2026-10-08). It needs the Mac session, because half of it is Metal code that only builds and runs on a Mac.

## The problem

Both GPU backends start from the same parsed scene (`pbrt_flatten::FlatScene`), and then each turns it into its own GPU arrays with its own code:

| | OptiX (Windows) | Metal (macOS) |
|---|---|---|
| Where | `gpu/optix/pbrt_gpu_builder.h` (+ `pbrt_gpu_builder_materials.h`) | `gpu/metal/metal_poc_pbrt_loader.mm` (+ `metal_poc_pbrt_materials.mm`) |
| Size | about 1,800 + 1,200 lines | about 1,800 + 600 lines |
| Output | `SceneData` (OptiX structs) | `MetalPocApp` buffers (Metal structs) |
| Shapes it walks | spheres, bilinear patches, curves, disks, cylinders, triangles and quads, instances | the same list, in much the same order |
| Lights it walks | infinite light, punctual lights, camera medium | infinite light, punctual lights, medium |

Reading the two side by side (`build()` in OptiX is now one function per kind of thing it adds, and the Metal loader already has one per kind), they decide the same things independently:

* **Which material a shape gets**, including the mix-material walk, the procedural-texture average colour, and what to do with a texture-bound parameter.
* **Texture decoding**: image files, gamma, alpha masks, grayscale detection.
* **Tables**: measured BRDFs, subsurface (BSSRDF) tables, the portal light's rectified image, the sky's sampling distribution.
* **Media**: how a homogeneous medium's per-channel extinction is reduced to what the GPU stores.
* **Tessellation**: curves into tubes (both call `curve_tessellate.h`), and Metal also tessellates bilinear patches, cones and paraboloids.
* **Light lists**: which emitters are samplable, and their power.
* **Scene scale and camera**: the Metal loader rescales every scene to about 2 units and transforms the camera override to match; OptiX does neither.

Every one of these is a place where the two can quietly disagree. The parity sweeps catch a visible disagreement after the fact; they do not stop the next one being written.

## The proposal

Split the conversion in two. The first half is backend-neutral and lives in `src/shared`, where the CPU unit tests can reach it (no GPU, no Objective-C). The second half only packs the neutral result into one backend's structs.

```
FlatScene --(shared: gpu_scene_common)--> GpuSceneIR --(OptiX pack)--> SceneData
                                                     \-(Metal pack)--> MetalPocApp buffers
```

`GpuSceneIR` would hold, already decided:

* a de-duplicated **material table** (each entry: a kind, plain parameters, indices into a texture table), with mix materials resolved;
* a **texture table** (decoded pixels, wrap, gamma, alpha masks);
* the **shapes** as flat lists with a material index and an area-light index, curves and other non-native shapes already tessellated *when the backend asks for it* (a capability flag per backend, from `backend_capabilities.h`'s idea: the OptiX side keeps native curve-free shapes, Metal asks for everything tessellated);
* the **lights** (punctual, area, infinite, portal) with power and sampling tables;
* the **media** in their reduced form, and the **camera** with its scene transform.

The pieces that are already shared and stay as they are: `pbrt_flatten`, `curve_tessellate.h`, `pbrt_quadify.h`, `measured_bxdf_loader.h`, `portal_image_infinite_light.h`, `srgb_decode.h`.

## How to do it safely (in stages)

1. **Move the pure functions first.** Material resolution (`resolveMixColor`, `resolveProceduralReflectanceTexture`), image and alpha-mask decoding, the measured and BSSRDF table builders, chromatic-medium reduction, and the Metal loader's scene-scale rule go into `src/shared/gpu_scene_common.h`, each with a unit test on CPU. OptiX calls them; nothing else changes. Check: seeded OptiX renders are byte-identical before and after (`scripts/render_baseline.py`, as in the refactors).
2. **Make Metal call the same functions** where it has its own copy. Check: the Mac parity sweep and its golden snapshot do not move (the Mac session).
3. **Introduce the material and texture tables** as a neutral type; both backends pack from it. Same checks.
4. **Shapes and lights last**, one kind at a time, because they are where the two diverge most (Metal's tessellation, OptiX's native spheres, disks and cylinders).

Each stage is its own merge. After stage 2 the two backends can no longer disagree about materials or textures; after stage 4, about anything in the scene.

## What it does not change

The shaders and kernels, the GPU struct layouts, the pipelines, and the Live Preview paths (OptiX wavefront features, Metal's plain frames). Those stay per backend.

## Risks

* **Metal performance and memory layout.** Metal has no spare kernel buffers (a 31-slot limit), so its per-material parameters ride in spare `TriangleMaterial` fields. The packing step stays Metal-specific for that reason; only the decisions move.
* **A behaviour the Mac relies on.** The scene rescale and camera transform are Mac-only today. Moving them into shared code must keep them off for OptiX (a flag), or both backends' images change.
* **Size.** About 5,000 lines of loader code are touched across the stages. The byte-identical render checks (OptiX) and the golden snapshot (Metal) are what make that safe; neither needs new infrastructure.

## Who

Stages 1 and 2 need both sides, so the Mac session should review this before anyone starts. A first useful, low-risk step that is Windows-only: stage 1 for the OptiX functions, with the Mac side added once it has agreed on the names.
