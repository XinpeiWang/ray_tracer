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

## Mac session review (2026-10-08)

Read against `gpu/metal/metal_poc_pbrt_loader.mm` (1,789 lines) and `metal_poc_pbrt_materials.mm` (599) on main at 40f30a7.

**Verdict: agree with the split (decisions in shared, packing per backend), agree with most names, disagree with the stage order, and four of the "decided independently" items are not what they look like.**

### Where the Metal loader is not a copy of the OptiX one

1. **Mix materials are decided differently on purpose, not duplicated.** OptiX picks between the two sub-materials per hit (stochastically, `optix_device_mix_camera.h`) and uses `resolveMixColor` where it needs one colour. Metal assigns materials per triangle at load time, so `mapPbrtMaterial` deterministically takes sub-material B when `mixWeight >= 0.5` and A otherwise (the comment in `metal_poc_pbrt_materials.mm` says why; BACKEND_SUPPORT.md lists it as an approximation). `resolveMixColor` has no Metal counterpart. The IR should carry (A, B, weight) and each packer should decide. Making Metal use the OptiX rule would be a behaviour change (new shader work, new golden snapshot), not a refactor.
2. **Textures: Metal can bind one image per slot.** The kernel has nine texture slots (environment, goniometric, projection, area-light, diffuse reflectance, transmit...), and only the *first* matching material in a scene gets a real image lookup; a second one falls back to its colour. A neutral texture table with N entries is richer than anything Metal can consume. The Metal packer needs a slot policy ("first use wins"), so the IR should list textures in order of first use, and stage 3 should say so.
3. **There is no BSSRDF on Metal.** Subsurface is unsupported there. The measured-BRDF tables are *already* shared (`measured_bxdf_loader.h`); Metal's `appendMeasuredBrdf` / `putMeasuredPL2D` only flatten them into its float buffer, which is packing and stays in Metal.
4. **Shapes: Metal does not walk bilinear patches, curves, cones or paraboloids at all.** `tessellateUnsupportedShapes` (about 170 lines in the loader) rewrites the `FlatScene` into triangles *before* the bounding box, materials, area lights or instancing look at it. OptiX tessellates curves and patches but not cones or paraboloids (BACKEND_SUPPORT.md). So "the same list in much the same order" is true only after that pass. It is also the best first stage: a pure `FlatScene -> FlatScene` step, testable on the CPU, with exactly the capability flag the proposal already wants.

### Things the proposal does not mention

* **The scene frame is more than a function.** `loadPbrtScene` computes the bounding box (ignoring a "huge ground-like sphere"), the scale `2 / extent`, the centre and a +X offset. The scale is then threaded through lights, medium coefficients (sigma), material parameters (`PbrtMaterialMapState`) and **Live Preview** (`metal_live_preview.mm` converts the GUI's camera and the first-hit positions with `pbrtSceneScale`, `pbrtSceneOffset` and `pbrtBboxCenter`). "Live Preview stays per backend" is right, but it depends on this rule: moving it into shared code must keep returning scale, centre and offset.
* **Visit order is part of the result.** Metal loads area-light triangles first (`loadPbrtAreaLights` assigns the light ids and probabilities), then the remaining triangles, spheres, disks, cylinders, instances, punctual lights, media, the infinite light and the camera. Ids come from that order, so changing it changes images. Stage 4 has to keep it.
* **File size.** One `gpu_scene_common.h` for all of this would break the size ratchet (2,000 lines per file, 300 per function) quickly. One header per stage is easier to keep under it.

### Names

* `GpuSceneIR`, "material table", "texture table", `backend_capabilities.h` as the home of the per-backend flag: fine.
* Keep the existing OptiX names (`resolveMixColor`, `resolveProceduralReflectanceTexture`) where they stay OptiX-only; do not generalise them.
* Suggested shared names: `tessellateForBackend(FlatScene&, const TessellationCaps&)` (Metal's `tessellateUnsupportedShapes` generalised), `computeSceneFrame(const FlatScene&) -> {scale, centre, offset}` (instead of "the scene-scale rule"), and one header per stage: `gpu_scene_tessellate.h`, `gpu_scene_frame.h`, `gpu_scene_textures.h`, `gpu_scene_materials.h`.
* For the table in this file: Metal's `mapPbrtMaterial` corresponds to OptiX's `makeMaterial`.

### Stage order (revised)

0. **`tessellateForBackend` and `computeSceneFrame` first**, both pure and complete on the Metal side already. OptiX adopts patches and curves through the capability flag and keeps its native shapes. CPU unit tests on triangle counts, areas and bounds.
1. Image and alpha-mask decoding plus sRGB (`stb_load_large.h`, `srgb_decode.h`), shared; the slot policy stays per backend.
2. The material decisions that really are common (reflectance colour, procedural average colour, chromatic-medium reduction). **Not mix.**
3. The material and texture tables as the neutral type, with the first-use texture order.
4. Shapes and lights last, in Metal's current visit order.

### Checking each stage on the Mac

Metal's loader refactors do not touch the shaders, so a seeded Metal render is **bit-identical** before and after (checked on this Mac: the same `--seed` gives the same EXR for 109 of 111 renders; D8 and D12 are not reproducible even run to run). That is a stronger check than the golden snapshot's tolerances and should be the gate for stages 0-4. It is not true for a *shader* change: moving code between shader functions changed float rounding in the kernel split and only the statistics stayed equal.

`scripts/render_baseline.py` has no Metal backend (it expects `x64\Release` and the OptiX recursive and wavefront backends). The Mac side needs `--backends metal` there before stage 0; the seeded-hash approach above is about 30 lines.

### Open for the Windows session

* Is mix on Metal meant to stay an approximation? (Per-hit mixing needs shader work; this proposal should not decide it.)
* Should OptiX grow native cones and paraboloids, or take Metal's tessellation through the flag? That decides how big `TessellationCaps` is.

## Windows session reply (2026-10-08)

Agreed with the Mac review: the stage order below replaces the one in "How to do it safely", the shared names are the ones suggested (`tessellateForBackend`, `computeSceneFrame`), one header per stage, mix stays a per-backend decision (the IR carries (A, B, weight)), `resolveMixColor` stays OptiX-only, textures carry a first-use order, and Metal's visit order is kept.

**The two open questions**

* *Mix on Metal:* leave it as the load-time approximation. Per-hit mixing is shader work; this proposal does not decide it.
* *Cones and paraboloids on OptiX:* take Metal's tessellation through the capability flag (`TessellationCaps::conesAndParaboloids()`), not native shapes. It was the cheaper answer and it is done: F14 (cone and paraboloid gallery) went from a documented known gap, with the shapes dropped, to passing the CPU-versus-GPU parity sweep on both OptiX renderers (mean 0.4311 vs 0.4292). `TessellationCaps` is four booleans: bilinear patches, curves, cones, paraboloids; Metal asks for all four, OptiX for the last two (its native patches and curve dicing stay).

**Stage 0, Windows half: done**

* `src/shared/gpu_tessellate.h`: `tessellateForBackend(FlatScene&, TessellationCaps, FiberTangents*)`. It is `tessellateUnsupportedShapes` from `metal_poc_pbrt_loader.mm` moved over with the numbers unchanged (`PackedFloat3` became `std::array<float, 3>`).
* `src/shared/gpu_scene_frame.h`: `computeSceneFrame(const FlatScene&) -> Frame {scale, centre[3], offset[3], extent, ...}` with `toWorld()` and `fromWorld()`. The bounding box, the huge-ground-sphere rule, the `2 / extent` scale and the +60 X offset are `loadPbrtScene`'s arithmetic, in the same single-precision order.
* CPU unit tests for both (`tests/unit/gpu_tessellate_tests.cpp`, `gpu_scene_frame_tests.cpp`): triangle counts per shape, a flat patch keeps its area exactly, a cone is within 1% of pi * r * slant, interface-material shapes are skipped, material and area-light ids carry over, the Cornell-box-sized frame, the ground-sphere rule, the empty scene.
* OptiX uses `tessellateForBackend` for cones and paraboloids (`gpu/optix/scene_builder.cpp`).

**Stage 0, Mac half: for the Mac session**

The Metal loader still has its own copy of both. Replace them with calls to the shared ones and check that a seeded Metal render is byte-identical before and after:

1. In `loadPbrtScene()`: `tessellateUnsupportedShapes(result.scene, pbrtTriangleFiberTangent)` becomes `gpu_tessellate::tessellateForBackend(result.scene, gpu_tessellate::TessellationCaps::all(), &tangents)` (copy `tangents` into `pbrtTriangleFiberTangent` as `PackedFloat3`, or change that map's value type to `std::array<float, 3>`), and delete the local `tessellateGrid`, `normalize3` and `tessellateUnsupportedShapes`.
2. The bounding box / scale / centre / offset block becomes `const gpu_scene_frame::Frame frame = gpu_scene_frame::computeSceneFrame(scene);`; `sceneScale`, `bboxCenter` and `sceneOffset` are `frame.scale`, `frame.centre` and `frame.offset`, `toWorld` is `frame.toWorld(...)`, and the two `fprintf` lines can use `frame.extent`, `frame.fullExtent` and `frame.ignoredGiantSphere`. Live Preview's camera and position conversions use the same `Frame`.

Two things I fixed on the way, both MSVC-only name lookups that clang accepts: a `using namespace scene_doc;` in a test made `detail` and `shapes_detail` ambiguous, so `scene_shapes.h`'s namespace is now `scene_shapes_detail`, and the shape headers and two library headers qualify `::detail` / `::shapes_detail`.

## Checking a Metal loader change (Mac)

`scripts/metal_render_hash.sh capture <name>` renders the 95 scenes of the parity sweep plus option variants (adaptive, clamp, lock-step, bands, crop) at a fixed seed and stores a hash of each float EXR; `compare <name>` does the same again and lists every picture that changed. About 40 s each way, and all 108 renders reproduce exactly run to run. Use it as the gate for stages 0-4 (host-side moves must give identical files). It is not for shader changes (use the parity sweep and `scripts/update_metal_golden.sh`). `scripts/render_baseline.py` is the statistical version for the CPU and OptiX; it needs numpy, which this Mac's Python does not have, and has no Metal backend.

## Stage 0, Mac half: done (2026-10-08)

`MetalPocApp::loadPbrtScene()` now calls `gpu_tessellate::tessellateForBackend(scene, TessellationCaps::all(), &pbrtTriangleFiberTangent)` and `gpu_scene_frame::computeSceneFrame(scene)`; `tessellateGrid`, `normalize3`, `tessellateUnsupportedShapes` and the bounding-box / scale / centre / offset block are gone from `gpu/metal/metal_poc_pbrt_loader.mm` (about 215 lines fewer). `pbrtTriangleFiberTangent` is now `std::unordered_map<int, std::array<float, 3>>`. `toWorld` stays a `float3` lambda over the frame's three numbers, so the arithmetic is the one it was. Checked with `scripts/metal_render_hash.sh`: all 108 seeded renders (the 95 scenes plus option variants) are byte-identical to the pre-change binary; `ctest` 13/13. F7 (hair curves) renders differently from run to run on the unchanged binary too (3 distinct hashes in 6 runs), so the script retries a differing scene a few times before counting it as changed.

Next on the Mac side: stage 1 (image and alpha-mask decoding, sRGB) once the Windows half exists.
