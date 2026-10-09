# Metal backend: CPU parity status

How closely the macOS Metal renderer (`gpu/metal/`) matches the CPU renderer, how that is
measured, and what is still different. Start with `docs/METAL_BACKEND.md` (how the backend works and how to change it). Complements `docs/history/METAL_GPU_FEASIBILITY.md` (design
history) and `docs/PBRT_SUPPORT.md` (which pbrt features each backend supports).

## How it is measured

`gpu/metal/metal_cpu_gpu_parity_check.cpp` renders every pbrt-backed scene with both backends at
a small size and compares linear HDR (`.exr`) output:

* **whole-image brightness** and **per-channel means** within 30%;
* a **6x6 regional grid** (worst block within 50%; a miss up to 1.5x that is only "marginal");
* **no NaN/Inf pixels** in either image.

Both backends use a **fixed RNG seed** (`METAL_PARITY_SEED` overrides), so the sweep is reproducible.
`ctest` runs it with `METAL_PARITY_STRICT=1`: a scene that exceeds tolerance and is not a documented
known gap fails the test. Without that variable it just prints the per-scene report.

```
cd build_macos && ./metal_cpu_gpu_parity_check                    # report only
METAL_PARITY_ONLY_SCENE_ID=C4 ./metal_cpu_gpu_parity_check        # one scene
METAL_PARITY_MODELS=1 ./metal_cpu_gpu_parity_check                # also the Models scenes (need models/)
```

**Fast Mac dev loop** (measured on an 8-core M2): configure with Ninja (`cmake -S . -B build_macos -G Ninja -DCMAKE_BUILD_TYPE=Release -DRT_BUILD_METAL=ON`):
a clean build is ~65 s (Unix Makefiles `-j8`: ~81 s), editing a `.mm` rebuilds in ~5 s, `metal_poc_app.h` ~17 s, a shared CPU
header 35-45 s, and editing a `.metal` shader needs no rebuild at all (shaders are compiled at run time). The full `ctest`
is ~79 s, almost all of it the parity sweep, which runs as **2 worker processes** (the CPU reference render of one scene
overlaps the Metal render of another; ~98 s single-process, identical verdicts). `METAL_PARITY_WORKERS=N` changes that
(1 = single process; 3-4 barely help, the CPU renders are the bottleneck); a single-scene run is always one process.
A strict sweep fails if it skipped every scene (e.g. run from a directory where the scene files do not resolve), so
run it from the build directory or through `ctest`.

**Path regeneration and the divergence census.** A SIMD group (32 lanes, an 8x4 pixel tile) used to start sample s+1 only when
every lane had finished sample s, so each bounce cost as much as the longest path: `METAL_CENSUS=1` (prints per-scene lane
utilisation and distinct materials per group-bounce) measured only ~49-56% of lanes active even in A1, and ~1.5 distinct materials
per group-bounce - so lane idling, not material mixing, is the recoverable waste (a full wavefront split with material sorting was
judged not worth it for that). The kernel now regenerates: a lane whose path ends immediately starts its next sample. Every lane
draws its random numbers in exactly the same order, so the image is **bit-identical** (verified by `cmp` on the PNGs). Measured
(400px, 64spp, depth 8, M2): Cornell-style scenes 9-23% faster, but mesh-heavy scenes (>=2k triangles) 2-15% SLOWER (lanes at
different bounce depths trace incoherent rays through a big BVH), so the host picks per scene: regeneration below 1500 triangles,
the old lockstep above. `METAL_REGEN=0|1` forces a mode. Full strict ctest 79 s -> 66 s.

**Golden snapshot (catches what the CPU comparison cannot).** The CPU-vs-Metal tolerance (30% whole image, 50% per block) only
finds gross errors, and it cannot see a regression that CPU and Metal share. Measured by injecting bugs into the shaders: a
+15% bias on all diffuse direct lighting and removing the Russian-roulette compensation both PASSED it; +35% and a 40% Fresnel
error only just failed. Metal is deterministic and low-noise (a different seed moves a scene's channel means by <= 2.2% and its
4x4 block means by <= 11%, over 95 scenes x 3 seeds), so the sweep also compares every scene's Metal image with a committed
snapshot of Metal's own earlier output, `gpu/metal/parity_golden.txt` (R/G/B means + 4x4 block means, averaged over 3 seeds):
channel means may move by 6%, blocks (above a small brightness floor) by 20%. With the snapshot: **0** drifts on two unseen
seeds, and the same bugs are caught: +15% diffuse -> 40 scenes flagged, +8% -> 16, no RR compensation -> 2. Under
`METAL_PARITY_STRICT=1` (ctest) a drift fails the test.

* A drift you did not intend is a **bug** - find it. If the picture was MEANT to change (a new feature, a deliberate accuracy
  fix; known-gap scenes improving counts), run `scripts/update_metal_golden.sh` (~3.5 min) and review `git diff` of the snapshot:
  every scene whose numbers moved should be one you meant to change. Commit it with the change.
* A different Mac GPU family (M1/M3/M4) should stay inside the tolerance (it only changes float rounding, like a seed change);
  if it does not, that is worth knowing, but re-snapshot only after understanding why.
* `METAL_PARITY_SPP=<n>` renders both backends at n samples per pixel (a converged reference; not for the gate), `METAL_PARITY_TIMING=1` prints each Metal render's wall time, `METAL_PARITY_ADAPTIVE=1` turns Metal's adaptive sampling on for that render.
* `METAL_PARITY_GOLDEN=<file>` uses another snapshot, `METAL_PARITY_GOLDEN=off` disables the check, `METAL_PARITY_DUMP=<file>`
  appends the current numbers to a file. CI cannot run it (the GitHub macOS runner has no hardware ray tracing), so it protects
  a developer Mac, like the rest of the sweep.

Latest full run (all categories, METAL_PARITY_ALL=1): 141 scenes, **140 pass, 1 marginal (the noise-limited rough-glass-lamp), 0 un-triaged failures** - the portal-light scene (C17) now passes too (cpu 0.0766 vs metal 0.0765 at 3000 spp, worst block 0.7%): Metal implements pbrt-v4's portal restriction (see "Portal infinite light" below).
The standard sweep (what CI/ctest runs): 56 scenes, all pass. All 24 Models
scenes with assets present match CPU (23 pass, 1 marginal). CI skips the test, non-fatally, on a runner
whose Metal device cannot do hardware ray tracing, so the gate really protects a developer Mac.

**Host/shader contracts.** Two things that used to be held together only by comments are now checked. (1) `materialType` values have names, `METAL_MAT_*`, in `gpu/metal/metal_poc_material_ids.metal` - one file that the host #includes and that is also the first file of the concatenated shader source, so the loader and the kernel cannot disagree about what "25" means (the numbering is Metal's own; OptiX has a separate enum). (2) `metal_poc_shader_tests` asks the shader for its `sizeof`/`alignof` of every struct the host mirrors by hand (`Uniforms`, `TriangleMaterial`, the media, light, lens structs) and the byte offset of the last 17 `Uniforms` fields, and compares them with the host's (`test_structLayouts`, `metal_poc_shader_tests_layout.mm`): adding a field on one side only, or in a different place, fails that test instead of shifting every later field. Keep the probe lists in step when you add a mirrored struct.

## What the Metal loader supports

Shapes: triangle meshes, spheres, disks, cylinders; **bilinear patches, cones, paraboloids and curves are
tessellated into triangles at load time**. Materials: diffuse, conductor, dielectric (smooth and rough),
thin dielectric, diffuse transmission, **coated diffuse / coated conductor (pbrt-v4 LayeredBxDF)**, hair,
normalized-fresnel, principled, mix. Textures: image (a second image slot serves diffuse-transmission transmittance), 2D/3D/nested checkerboard, marble, fbm, windy,
wrinkled, dots, bilerp. Lights: point, spot, distant, goniometric, projection, infinite (constant and image),
quad / sphere / disk / cylinder / triangle area lights (all NEE-sampled). Media: camera medium (infinite),
homogeneous media bounded by interface-material spheres, and **glass spheres that bound a scattering medium**.

Because there are no spare kernel buffers (Metal's 31-slot limit), per-material parameters ride in spare
`TriangleMaterial` fields. The conventions (documented where each is set, in `metal_poc_pbrt_materials.mm` and `metal_poc_pbrt_loader.mm`):
materialType 25 is a family of textures selected by `conductorK.y` (0 = 2D checker, 1 = 3D checker, 2 = marble,
3 = fbm, 4 = windy, 5 = wrinkled, 6 = dots, 7 = bilerp, 8 = image-in-checker); glass with a medium carries
`conductorEta` = per-channel sigma_t, `conductorK` = (g, has-medium, chromatic), `transmitColor` = albedo.

## Behaviour worth knowing

* The loader rescales every scene to ~2 units across. The extent is taken from the scene's *content*, ignoring a
  huge ground-plane sphere (diameter >= 60% of the extent, shrinking it by > 4x), so ray epsilons stay meaningful.
* The Film `maxcomponentvalue` firefly clamp is unbounded by default (as on CPU). Paths that bounced off hair get
  a per-sample clamp of 40, because the float32 hair BSDF occasionally yields absurd weights.
* A shape with `Material "interface"` is transparent (it only bounds a medium), never an opaque gray mesh.
* Glass shadow rays: clear, thin and rough glass **block** shadow rays (as pbrt-v4 and, since commit 2f8dc3d, the CPU and OptiX do), including a glass sphere that bounds a medium: a scatter vertex inside it cannot see a light through its own shell, so the fog is lit only along specular chains. A boundary that should not block is Material "interface". (Metal used to let shadow rays through such glass with stochastic attenuation and to keep a scattered path's MIS state across the boundary, which only agreed with the CPU for sky lights; a small lamp read ~2x too bright.)
* Adaptive sampling is **off** unless `RenderOptions::adaptive_sampling` asks for it (as on the CPU). It used to be hard-wired on, and its "mean below 1e-4 after 16 samples" early stop froze every pixel that had not yet seen a rare light path (caustics, a lamp seen through glass): E12 read ~10x too dark in places and kept black pixels at any sample count. Measured with `METAL_PARITY_ADAPTIVE=1 METAL_PARITY_TIMING=1 METAL_PARITY_SPP=4000` (Metal wall time, 60x60, off -> on): it saves little where it matters - Cornell box A1 45.5 -> 44.4 s, fog E1 62.0 -> 61.5 s, F1 54.2 -> 47.3 s, C3 36.3 -> 34.1 s - and helps only scenes that converge quickly (B1 5.1 -> 2.5 s, the white-sky K22 17.1 -> 0.27 s), while costing accuracy where light is rare (E12 mean 0.1129 -> 0.1044, C3 0.1764 -> 0.1638, i.e. -7 to -8%). Timings are single runs on a shared machine (B20 and E12 even came out slower with it on), so read them as orders of magnitude. Default-off is the right call; opt in with `RenderOptions::adaptive_sampling` for flat scenes.

## Known gaps (`kKnownGapScenes`)

(none - `kKnownGapScenes` is empty. C17, the portal-light scene, used to be listed here for a missing sky image; it now uses a generated in-repo sky and passes.)

**Found by looking at pictures, not by the sweep (2026-10-08):** a pbrt scene's *cloud* medium was not rendered on Metal (`pbrt_scenes/cloud-medium.pbrt` = E5 and `cloud-medium-scene.pbrt` = E2 showed no cloud, the second a black disc): the loader had no case for it. Fixed: the loader builds the cloud (like the RGB grid), and three shader faults it exposed are fixed too - a ray that had just left the cloud's box stood still re-entering the shader (dark fringe), the environment-light NEE used the box's majorant instead of the real density (black cloud), and the environment light was counted twice after a cloud scatter (too bright). The sweep still cannot see most of this (the pictures are mostly sky), so E2's and E5's golden lines were re-snapshotted after comparing the pictures with the CPU's.

Smaller approximations not covered by a scene: shadow rays use shutter time 0 (a moving sphere casts its
shadow at its start position); the layered shaders use one roughness for both interfaces; a marble reflectance on a
*coated diffuse* material renders flat; animated (motion-blurred) bilinear patches and curves are dropped;
the large third-party scenes (Sponza, Bistro, the H-family environments, the pbrt-v4 folders) are not in the parity sweep: they are not in a checkout, and the GUI fetches them on demand ("Download missing files"). Barcelona Pavilion (H14) used to be the one scene that did not render: baking its ObjectInstance placements made 97M triangles. pbrt ObjectInstance now maps to hardware instances (see "ObjectInstance" below), and it renders.

## Packaging note (macOS)

`scripts/build_and_deploy_macos.sh [--arch native|arm64|x86_64|universal]` builds the whole app for one architecture choice (the default is this Mac's own CPU) and hands it to both CMake and qmake, since the GUI and `scene_metadata.dylib` must match or the GUI's `dlopen` fails. Qt 6 official installs are universal, so `--arch universal` (arm64 + x86_64 in every binary, one dmg for every Mac, about 4 minutes) works with no extra install; an earlier version took the first architecture `lipo -archs qmake` listed (x86_64) and so shipped Rosetta-only builds although Qt was universal all along. Build directories are per choice (`build_macos_arm64/`, `build_macos_universal/`, and `qt_gui/build_macos_<choice>/`), so a re-run is incremental and nothing is wiped. The root `CMakeLists.txt` also defaults `CMAKE_OSX_ARCHITECTURES` to the real CPU: an Intel-built `cmake` under Rosetta used to make every development build x86_64 too. On a native arm64 build the strict ctest still passes with the committed golden snapshot (123 s versus 126 s under Rosetta: the CPU reference renders are not the bottleneck the arch would change).

## ObjectInstance (hardware instancing)

A pbrt `ObjectInstance` used to be baked: every placement copied the group's triangles into the world-space arrays. Now each used group's triangles are stored ONCE (object space) after the scene's own triangles in the shared vertex/normal/uv/material arrays; each group gets its own acceleration structure (a slice of the shared vertex buffer), and each placement is a Metal instance of it (`pbrtGroupAS`, `buildInstancedMeshAS` in `metal_poc_gpu_resources.mm`). The instance table (`InstanceTransform`, buffer 12) carries, per instance, the group's first triangle (`triBase`) and the NORMAL matrix (inverse transpose of the object-to-scene linear part, so non-uniform scale and mirroring are right); `Uniforms::pbrtInstanceFirst` says from which instance id the placements start (Suzanne's demo-room instances come before). In the kernel a hit on such an instance sets `primId = triBase + primitive_id`, so every triangle-indexed lookup (uvs, materials, vertices) works unchanged in object space, and only the shading normal is transformed. Bump mapping on an instanced triangle is skipped (its tangent frame would be object-space). Instanced spheres are still skipped, as before. Existing instanced scenes agree with the baked renders to within float rounding (mean abs diff 0.00-0.04 of 255).

## The large third-party scenes (H1-H21) on Metal

They are not in a checkout (the GUI's "Download missing files" fetches them) and a CPU reference render of them is far too slow for the parity
sweep, so `scripts/metal_large_scenes_sweep.sh` renders each one on Metal only (320x180, 8 spp, depth 4) and reports load, time, triangle count
and a lit/dark verdict. Run on an M2 (2026-10-07), all 21 downloaded through the GUI's own downloader:

* **All 21 render**: 80k to 12.8M triangles, 0-18 s each (Bistro 2.8M: 18 s; San Miguel 10M: 12 s; Power Plant 12.8M: 13 s).
* **H14 Barcelona Pavilion**: 43 placements of 2 tree groups (5.2M triangles stored once; baking them was 96.9M and failed with "primAS is nil" after
  100 s) now render in about 8 s at 480x270, 32 spp.
* **H13 (Contemporary Bathroom) and H19 (Crown) come out dark** here: the GUI applies a curated per-scene exposure for them (see
  `scene_registry.h`), which this plain render does not.
* H7 (Breakfast Room) skips one material group: the scene file asks for `Material_005`, the upstream OBJ now spells it `Material_005.001`, so
  `pbrt_scenes/environment-breakfast-room.pbrt` (generated by `scripts/gen_environment_scenes.py`) is slightly stale.
* Unsupported pbrt features these scenes use are reported by the loader as warnings, not errors: a `coateddiffuse` roughness or displacement
  and a `mix` amount bound to a texture (Zero Day, Villa, Crown) fall back to a constant.
* Sweeping them found two image-loading problems that affect every backend, both fixed: the vendored stb_image (v2.06) could not decode 16-bit
  PNGs (Zero Day alone has ~100), so the upgrade to v2.30 was needed; and v2.30 in turn refuses an image whose FLOAT buffer would exceed 2 GB,
  which the Gallery's 16384x16384 JPEG (3.2 GB as floats) does - its picture changed from the tan texture to plain grey after the upgrade
  (mean 131.8 -> 227.7) until `src/shared/stb_load_large.h` added an 8-bit fallback that converts exactly like stb does (back to 131.8). The
  sweep is how that was noticed: compare a scene's numbers before and after any change to image loading.

## pbrt example (K) family - not in the default sweep

The default sweep (`ctest` / `METAL_PARITY_STRICT=1`) covers the hand-written feature categories. The ~73 pbrt example scenes (ids
K1..K160, one per `pbrt_scenes/*.pbrt`) were never compared against the CPU until `METAL_PARITY_ALL=1` was added:

    METAL_PARITY_ALL=1 METAL_PARITY_GOLDEN=off ./build_macos/metal_cpu_gpu_parity_check     # run from the repo root

First run: 19 of those scenes differed. Fixed since: K10, K11 (chromatic absorbers), K12 (chromatic camera medium), K75 (point light in
a fog sphere rendered black), K14 and K15 (RGB-grid medium: now per-channel spectral tracking with per-voxel sigma_a, like the CPU and
OptiX; the grid also reads zero beyond its outermost voxel centres, and no longer double-counts a constant sky). K43 (a rough conductor under a spot light: the CPU, not Metal, was wrong - a plain `conductor` with an "rgb reflectance" was built as the fuzz-mirror `metal`, which cannot show a point-light highlight; it is now the real GGX + complex-Fresnel conductor) and K5 (Metal now implements image bump mapping: a grayscale "texture displacement" height map, stored in the shared float buffer, perturbs the shading normal of triangle hits with the same pixel-footprint finite-difference step as the CPU and OptiX; an RGB normal map - "texture displacement" of a linear-RGB image, tangent-space (x, y, z) = 2*rgb-1 in the (dpdu, dpdv, n) frame - is read the same way, see normal-mapped-plane.pbrt). K94 (a rough silver floor under a small sphere light: a loader bug - the conductor shader squares the roughness-style value it is given, and the pbrt loader stored alpha itself, so every pbrt conductor rendered with alpha^2; roughness 0.04 got alpha 0.04 instead of 0.2, a far sharper highlight; the profile now matches both the CPU and a hand-computed GGX integral along the highlight). K145 (a camera inside rough glass: rough dielectrics had the same squared-alpha bug as the conductors, and Metal did no light sampling at rough-glass vertices, so a small bright light seen through frosted glass was found only by chance - the estimate was under-converged and 45% low; Metal now samples the area lights there with MIS, using a port of the CPU RoughDielectricBxDF f() and pdf()). K84 and K82 (hair: the hair shader started the continuation ray on the incoming side of the surface although hair scatters through to the far side too - a spurious second scattering event, seen as black speckle - and Metal had no path-throughput ceiling, which the CPU and OptiX cap at 50 per channel, so a closed hair sphere rendered about 2x too bright; the same B11 hair-fibers scene moved from 0.178 to 0.126 and now agrees with the CPU). textured-twosided-lights (an image-textured disk and cylinder light: Metal lit them with the image's flat average; the loader now lets several lights share the one image, the light sampler and direct hits map pbrt's (u, v) over the disk/cylinder from its own frame). rgbgrid-emission (a self-glowing RGB grid: pbrt's per-voxel "rgb Le" x "Lescale" is now emitted at the grid's real collisions, weighted by the event's absorption share like the CPU and OptiX; Metal used to render it black; CPU 0.3497 vs Metal 0.3493). K49/E6 (a homogeneous medium bounded by a cylinder, filling the solid tube like the CPU). NOTE: the K ids below shift whenever a pbrt file is added to pbrt_scenes/ - go by the file names. Fourteen of these scenes
(K10, K11, K12, K14, K15, K43, K5, K75, K84, K94, K145, textured-twosided-lights, normal-mapped-plane, rgbgrid-emission) are now part of the default sweep (matched by file name; not in the golden snapshot). Still differing (no fix yet - none of these is a crash or a black frame in the GUI):

| Scene | pbrt file | CPU vs Metal | Likely cause |
|---|---|---|---|
| K49 (= E6) | cylinder-medium | regional | passes (a homogeneous medium bounded by a cylinder fills the SOLID tube - wall plus open ends, like the CPU volume_bounds() - so rays through an open end see the fog; shadow rays are attenuated stochastically in cylinderIntersectionFunction) |
| K132 (= C17) | portal-light | passes | Metal implements the portal restriction now (see "Portal infinite light" below) |
| K87 (= infinite-light-image) | infinite-light-image | passes | fixed on both sides (see "Image sky orientation" below): the CPU sky light read the image upside down in `Le()` against its own importance-sampling table, and Metal used a vertically flipped, azimuth-mirrored mapping of its own. Both now put image row 0 straight up, like the OptiX backends. The 4x4 test image still differs a little: the CPU reads it nearest-neighbour (a hard blue/black horizon stripe), Metal filters bilinearly. |

**Image sky orientation (fixed 2026-10-06).** `sky_light::Le()` (the CPU) used to look the image up through `hdr_image_texture::value()`, which flips v, so straight up read the BOTTOM row while the importance-sampling table (`sample_Le`, `pdf_Li`) put row 0 at +y: light was sampled from one half of the sky and looked up in the other, and an image sky displayed upside down. The OptiX `gpu_sky_Le()` mirrored the quirk on purpose. All three now read row 0 at +y (`u = atan2(-z, x) / 2pi`, `v = theta / pi`); Metal had a third convention (`equirectangularUV()`: v = 1 at +y, longitude mirrored and shifted), so its environment lookups, importance-sampling inverse and pdf now go through `envMapUV()`. Every image-sky scene (C1, I3, K87 and the hand-written environment scenes) changes on CPU and OptiX; `SkyLightTest.LeReadsTheSameRowThePdfTableUses` guards it. The OptiX change is untested on this Mac (no CUDA). Metal-compatible scenes (`RT_GUI_SELFTEST=livepreview_sweep`, launched from "/"):
283 start with a well-lit picture; 9 (H13-H21) need external scene assets that are not bundled (the GUI downloads them); the rest are dark by design (a sphere on
a black background, light-only tests) and show the same lit fraction as the standalone Metal renderer.

Rough glass bounding a medium (E12, `rough-dielectric-medium.pbrt`) now matches the CPU to about 1% (0.1130 vs 0.1128, rows within 5% at 3000 spp): the glass takes direct-light samples like any other vertex, and adaptive sampling no longer freezes its rare-light pixels. The earlier note that K146 (a lamp inside a rough glass sphere) was noise-limited was the same adaptive-sampling bias: it now passes (0.8246 vs 0.8246).

Triage of the three scenes that sat just inside the tolerance (all with converged references, `METAL_PARITY_SPP`):
* `coated-conductor-glossy-lamp` (a coat with `interface.roughness 0` over a conductor with `conductor.roughness 0.3`): a REAL difference, Metal 1.5-1.8x too bright. The Metal layered shader used ONE alpha for both interfaces (the coat's), so the conductor came out mirror-smooth. The loader now passes the conductor's own alpha and the coat `thickness` (transmitColor = conductor alpha, thickness); rows now agree to 1.00 at 6000 spp. The scene joined the regression set.
* A9 (final-scene): the radius-5000 world-haze glass sphere contains the CAMERA, and Metal only entered a glass medium by refracting through its boundary, so a path that started inside never saw the haze (dark regions 0.4-0.7x the CPU). `Uniforms::cameraGlassPrim` now starts every path inside the outermost glass-medium sphere that contains the camera and returns to it on leaving a nested glass-medium sphere (the small blue fog sphere). A9 matches to within a few percent (worst block 68% -> 21%).
* J2 (diffusetransmission-texture): passes now (brightness cpu 0.2736 vs metal 0.2739, worst block 3.4%). It used to be marginal and was blamed on texture filtering; the diffuse-transmission lobe-weight fix (#287) is what closed it, so that explanation was wrong.

The three new `bdpt-box-room-*` scenes (clear glass, rough glass, rough metal in a diffuse box lit by an UP-facing plate) failed the all-category sweep and led to four real Metal fixes, each confirmed against the CPU and, for the first, an analytic integral:
* **Shadow rays blocked by the light they target.** A shadow ray started 0.001 off the receiver along its normal but stopped 0.002 short of the sampled point measured along the ORIGINAL direction, so for a light plane nearly parallel to the receiver (an up-facing plate lighting the ceiling above it) every ray more than ~60 degrees off the normal crossed the light's own plane before its end point. At an oblique ceiling point 87% of the light samples were reported blocked: 0.066 vs the analytic 0.203 (the CPU reads 0.204). `reaimShadowRay()` now re-aims each offset shadow ray at the sampled point from its offset origin (37 sites). The down-facing sibling scene never showed it because its receivers face the light.
* **Depth limit.** pbrt's convention (and the CPU's) is `maxDepth` scattering vertices plus ONE more step that only adds the emission or sky the last sampled ray sees, MIS-weighted against the light sample taken at the vertex that sampled it. Metal stopped at the last vertex: `path-depth-furnace` read 0 at depth 1 instead of 0.5. The kernel now runs one extra emission-only iteration (no light sampling, no scattering; a path that reaches a bounded medium or a global-fog scatter there simply ends).
* **Conductor energy compensation removed.** The old multi-scatter boost made a rough metal (roughness 0.5) read up to 1.4x brighter than the CPU, whose conductor, like pbrt-v4's, is single-scatter GGX. K4 now matches (0.1931 vs 0.1929).
* **Rough dielectric.** pbrt discards a sample that lands in the wrong hemisphere (a reflection below the surface, a refraction that stays on the incoming side); Metal kept it, so frosted glass lost less energy than the CPU as roughness grew (+9% at 0.25, +12% at 0.5). It is now discarded; K3 matches (0.1671 vs 0.1675).
* **Diffuse transmission.** The new upstream `diffuse-transmission-furnace` (a closed R = 0.2, T = 0.6 shell under a white sky, closed form 0.2 / 0.56 / 0.632 / 0.6464 -> 0.65) read 0.05 at depth 1 and 0.26 at depth 8 on Metal. Metal chose the reflect or transmit lobe with probability p = pr/(pr+pt) but weighted the sample by the bare lobe colour, losing a factor 1/p per bounce (4x on reflection here); pbrt (and, since upstream commit 16b11b0, the CPU and OptiX) divide by p. Metal now matches the closed form at every depth (0.2013, 0.5609, 0.6328, 0.6506 at depths 1, 2, 3, 8). The scene joined the regression set.

The three scenes' light plate had no `Material`, so pbrt gives it a default 0.5 diffuse whose underside is visible at the top of the frame, while the CPU treats it as an emitter only; they now say `Material "diffuse" "rgb reflectance" [ 0 0 0 ]` explicitly, as the other scenes do (no change on the CPU). `METAL_PARITY_DEPTH=<n>` overrides both renders' depth for this kind of diagnosis.



Measured BSDFs (pbrt "measured", a tabulated Dupuy-Jakob BRDF) now render on Metal (materialType 32, `shadeMeasured`): a port of `measured_bxdf.h` / `piecewise_linear_2d.h` via the OptiX device code, with the five warp tables and a descriptor kept in the shared `rgbGridData` float buffer. Importance-sampled continuation (weight f*|cos|/pdf, luminance-warped) plus direct-light sampling with MIS (area, point, spot, distant, projection, goniometric, environment) like the CPU. `measured-furnace`, `measured-lights`, `measured-lights-area` and the B14 showroom now match the CPU to 0.1-0.3% (before this Metal rendered every measured material as flat gray Lambertian). They are in the regression set (by file name).

**Portal infinite light (added 2026-10-07).** `LightSource "infinite" "point3 portal"` restricts an image environment light to the directions seen through a quad, and the window depends on the ORIGIN of each ray. Metal used to ignore `portal[4]` and treat the light as an ordinary environment map, so `portal-light` (whose open ceiling and front admit sky the portal blocks) came out about 2.5x too bright. The loader now builds a real `PortalImageInfiniteLightData` on the host (the same class the CPU uses: equal-area rectification plus the summed-area table), uploads its arrays through the existing `pbrtEnv*` buffers (distribution + SAT in the "marginal" slot, the rectified image in the "conditional" slot, so no new buffer slot), and `portalSampleLi()` / `portalPdfLi()` / `portalLe()` (`metal_poc_sampling.metal`, a float port of `gpu/optix/gpu_portal_light_shared.h`) do the sampling, pdf and radiance. Every environment-NEE site and the miss path go through `pbrtEnvSampleDirection()` / `pbrtEnvLeAt()` / `pbrtEnvPdfAt()`, which switch on `Uniforms::pbrtHasPortalLight`; plain image skies take the same code as before. The portal corners are put through the same rescale/recentre/offset as the geometry. Two caveats: the summed-area table is float on Metal (the CPU keeps double), so a very small window far from the shading point is less precise; and a portal with no usable image adds no light at all, like the CPU, rather than an unwindowed sky.

**Cloud follow-up (2026-10-09):** the darker top edge of the E5 cloud that remained after the first cloud fix was a bug of that fix, not a density or lighting difference: the new per-path flag `PathState::envLightSampled` was cleared at a surface hit but not at the start of each sample, and `PathState` lives across a pixel's samples, so after a cloud scatter every later sample of that pixel lost its sky light. A white furnace found it (a non-absorbing cloud under a uniform sky must be invisible: Metal was dark where the cloud was thin and scattered rarely, the CPU flat). `pbrt_scenes/cloud-furnace.pbrt` and the `metal_poc_cloud_furnace` test keep it fixed; E5 now agrees with the CPU to 0.5% everywhere. (Lesson: a new field in `PathState` must be reset in `initPathState`.)
