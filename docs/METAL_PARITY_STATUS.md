# Metal backend: CPU parity status

How closely the macOS Metal renderer (`gpu/metal/`) matches the CPU renderer, how that is
measured, and what is still different. Complements `docs/METAL_GPU_FEASIBILITY.md` (design
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
* `METAL_PARITY_GOLDEN=<file>` uses another snapshot, `METAL_PARITY_GOLDEN=off` disables the check, `METAL_PARITY_DUMP=<file>`
  appends the current numbers to a file. CI cannot run it (the GitHub macOS runner has no hardware ray tracing), so it protects
  a developer Mac, like the rest of the sweep.

Latest full run (Models included): 119 scenes, **104 pass, 6 marginal, 9 known gaps, 0 failed**.
The standard 95-scene sweep (what CI/ctest runs): ~81 pass, ~5 marginal, 9 known gaps. All 24 Models
scenes with assets present match CPU (23 pass, 1 marginal). CI skips the test, non-fatally, on a runner
whose Metal device cannot do hardware ray tracing, so the gate really protects a developer Mac.

## What the Metal loader supports

Shapes: triangle meshes, spheres, disks, cylinders; **bilinear patches, cones, paraboloids and curves are
tessellated into triangles at load time**. Materials: diffuse, conductor, dielectric (smooth and rough),
thin dielectric, diffuse transmission, **coated diffuse / coated conductor (pbrt-v4 LayeredBxDF)**, hair,
normalized-fresnel, principled, mix. Textures: image (a second image slot serves diffuse-transmission transmittance), 2D/3D/nested checkerboard, marble, fbm, windy,
wrinkled, dots, bilerp. Lights: point, spot, distant, goniometric, projection, infinite (constant and image),
quad / sphere / disk / cylinder / triangle area lights (all NEE-sampled). Media: camera medium (infinite),
homogeneous media bounded by interface-material spheres, and **glass spheres that bound a scattering medium**.

Because there are no spare kernel buffers (Metal's 31-slot limit), per-material parameters ride in spare
`TriangleMaterial` fields. The conventions (documented where each is set in `metal_poc_pbrt_loader.mm`):
materialType 25 is a family of textures selected by `conductorK.y` (0 = 2D checker, 1 = 3D checker, 2 = marble,
3 = fbm, 4 = windy, 5 = wrinkled, 6 = dots, 7 = bilerp, 8 = image-in-checker); glass with a medium carries
`conductorEta` = per-channel sigma_t, `conductorK` = (g, has-medium, chromatic), `transmitColor` = albedo.

## Behaviour worth knowing

* The loader rescales every scene to ~2 units across. The extent is taken from the scene's *content*, ignoring a
  huge ground-plane sphere (diameter >= 60% of the extent, shrinking it by > 4x), so ray epsilons stay meaningful.
* The Film `maxcomponentvalue` firefly clamp is unbounded by default (as on CPU). Paths that bounced off hair get
  a per-sample clamp of 40, because the float32 hair BSDF occasionally yields absurd weights.
* A shape with `Material "interface"` is transparent (it only bounds a medium), never an opaque gray mesh.
* Glass shadow rays: clear and rough glass **block** shadow rays (as pbrt-v4 does); a glass sphere bounding a
  medium lets them through with stochastic attenuation (as CPU does).

## Known gaps (`kKnownGapScenes`)

| scene(s) | cause | notes |
|---|---|---|
| C2, F2 | reflectance-only conductor | CPU renders a fuzzy mirror (no highlight from point/spot lights); Metal renders real GGX metal. A CPU-side decision. |
| B24 | CPU shadow rays pass through glass | CPU's `shadow_ray_hit` walks through glass; Metal blocks like pbrt-v4. Matching needs material-aware shadow tracing (~70 call sites). |
| A9, B13, E12 | glass + scattering medium, remaining differences | the medium is simulated (E3 matches CPU to 2%, E11 thin glass now passes after keeping the MIS state across the boundary); E12's rough glass is ~2-3x too bright inside (CPU sits between Metal's blocked and pass-through shadow behaviour, also at roughness 0); A9 also has a radius-5000 "world haze" sphere; B13 is just past the regional tolerance. |
| B11 | hair "black fur" | float32 hair BSDF instability for high absorption + narrow lobes; red channel ~1.4x CPU. |
| C17 | missing asset | `sssdragon/textures/small_rural_road_equiarea.exr` is not in the repo; both backends fall back differently. |

Smaller approximations not covered by a scene: shadow rays use shutter time 0 (a moving sphere casts its
shadow at its start position); the layered shaders use one roughness for both interfaces; a marble reflectance on a
*coated diffuse* material renders flat; animated (motion-blurred) bilinear patches and curves are dropped;
scenes that need external meshes beyond the Models set (Sponza, Bistro, the large environments) were never run on Metal.

## Packaging note (macOS)

`scripts/build_and_deploy_macos.sh` builds everything as the architecture of the installed `qmake`. The Qt install on
the dev Mac is x86_64-only, so releases are x86_64 (Rosetta). If the shell is native arm64, run the script as
`export PATH=$HOME/Qt/bin:$PATH; arch -x86_64 bash scripts/build_and_deploy_macos.sh` (the script now builds into per-architecture directories, `build_macos_x86_64/` and `qt_gui/build_macos_x86_64/`, so it never clobbers a native `build_macos/` and a re-run is incremental; no manual clearing needed anymore), or the link used to fail on mixed
architectures.

## pbrt example (K) family - not in the default sweep

The default sweep (`ctest` / `METAL_PARITY_STRICT=1`) covers the hand-written feature categories. The ~73 pbrt example scenes (ids
K1..K160, one per `pbrt_scenes/*.pbrt`) were never compared against the CPU until `METAL_PARITY_ALL=1` was added:

    METAL_PARITY_ALL=1 METAL_PARITY_GOLDEN=off ./build_macos/metal_cpu_gpu_parity_check     # run from the repo root

First run: 19 of those scenes differed. Fixed since: K10, K11 (chromatic absorbers), K12 (chromatic camera medium), K75 (point light in
a fog sphere rendered black), K14 and K15 (RGB-grid medium: now per-channel spectral tracking with per-voxel sigma_a, like the CPU and
OptiX; the grid also reads zero beyond its outermost voxel centres, and no longer double-counts a constant sky). K43 (a rough conductor under a spot light: the CPU, not Metal, was wrong - a plain `conductor` with an "rgb reflectance" was built as the fuzz-mirror `metal`, which cannot show a point-light highlight; it is now the real GGX + complex-Fresnel conductor) and K5 (Metal now implements image bump mapping: a grayscale "texture displacement" height map, stored in the shared float buffer, perturbs the shading normal of triangle hits with the same pixel-footprint finite-difference step as the CPU and OptiX; an RGB normal map - "texture displacement" of a linear-RGB image, tangent-space (x, y, z) = 2*rgb-1 in the (dpdu, dpdv, n) frame - is read the same way, see normal-mapped-plane.pbrt). K94 (a rough silver floor under a small sphere light: a loader bug - the conductor shader squares the roughness-style value it is given, and the pbrt loader stored alpha itself, so every pbrt conductor rendered with alpha^2; roughness 0.04 got alpha 0.04 instead of 0.2, a far sharper highlight; the profile now matches both the CPU and a hand-computed GGX integral along the highlight). K145 (a camera inside rough glass: rough dielectrics had the same squared-alpha bug as the conductors, and Metal did no light sampling at rough-glass vertices, so a small bright light seen through frosted glass was found only by chance - the estimate was under-converged and 45% low; Metal now samples the area lights there with MIS, using a port of the CPU RoughDielectricBxDF f() and pdf()). K84 and K82 (hair: the hair shader started the continuation ray on the incoming side of the surface although hair scatters through to the far side too - a spurious second scattering event, seen as black speckle - and Metal had no path-throughput ceiling, which the CPU and OptiX cap at 50 per channel, so a closed hair sphere rendered about 2x too bright; the same B11 hair-fibers scene moved from 0.178 to 0.126 and now agrees with the CPU). textured-twosided-lights (an image-textured disk and cylinder light: Metal lit them with the image's flat average; the loader now lets several lights share the one image, the light sampler and direct hits map pbrt's (u, v) over the disk/cylinder from its own frame). NOTE: the K ids below shift whenever a pbrt file is added to pbrt_scenes/ - go by the file names. Thirteen of these scenes
(K10, K11, K12, K14, K15, K43, K5, K75, K84, K94, K145, textured-twosided-lights, normal-mapped-plane) are now part of the default sweep (matched by file name; not in the golden snapshot). Still differing (no fix yet - none of these is a crash or a black frame in the GUI):

| Scene | pbrt file | CPU vs Metal | Likely cause |
|---|---|---|---|
| K49 (= E6) | cylinder-medium | regional | passes (a homogeneous medium bounded by a cylinder fills the SOLID tube - wall plus open ends, like the CPU volume_bounds() - so rays through an open end see the fog; shadow rays are attenuated stochastically in cylinderIntersectionFunction) |
| K132 | portal-light | CPU black | not a Metal bug: `sssdragon/textures/small_rural_road_equiarea.exr` is not in the checkout, so the CPU falls back to a black constant light |
| K87 | infinite-light-image | G channel, blue | not a Metal bug: the CPU sky light is inconsistent with itself for an image that is not vertically symmetric - `sky_light::Le()` reads the image through `hdr_image_texture::value()`, which flips v (so +y reads the BOTTOM rows), while its importance-sampling table (`sample_Le`, `pdf_Li`) puts row 0 at +y. Metal reads +y at the top rows throughout, as pbrt does; matching the CPU here would mean replicating that mismatch. Worth fixing on the CPU side |

Live Preview (macOS) was also run over all 312 Metal-compatible scenes (`RT_GUI_SELFTEST=livepreview_sweep`, launched from "/"):
283 start with a well-lit picture; 9 (H13-H21) need external scene assets that are not bundled; the rest are dark by design (a sphere on
a black background, light-only tests) and show the same lit fraction as the standalone Metal renderer.

Still open among the rough-glass scenes: K146 (a lamp inside a rough glass sphere: the means agree to 3% but light that crosses two glass surfaces is found only by path sampling in both renderers, so the 60x60 picture is speckled and the regional comparison is noise-limited) and K144 (rough glass bounding a scattering medium; marginal - that glass keeps the old no-light-sampling behaviour because the kernel carries the MIS state of a scattered path across its boundary).

New since: the three measured-BSDF scenes (`measured-furnace`, `measured-lights`, `measured-lights-area`) differ from the CPU (Metal 1.5-2.6x brighter) after the CPU/OptiX measured material was reworked upstream (path weight f*|cos|/pdf, luminance-warped sampling, direct-light sampling with MIS, point/spot/distant lights now light it). Metal's measured material (materialType 15) has not been brought along.
