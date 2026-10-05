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

Latest full run (Models included): 119 scenes, **104 pass, 6 marginal, 9 known gaps, 0 failed**.
The standard 95-scene sweep (what CI/ctest runs): ~81 pass, ~5 marginal, 9 known gaps. All 24 Models
scenes with assets present match CPU (23 pass, 1 marginal). CI skips the test, non-fatally, on a runner
whose Metal device cannot do hardware ray tracing, so the gate really protects a developer Mac.

## What the Metal loader supports

Shapes: triangle meshes, spheres, disks, cylinders; **bilinear patches, cones, paraboloids and curves are
tessellated into triangles at load time**. Materials: diffuse, conductor, dielectric (smooth and rough),
thin dielectric, diffuse transmission, **coated diffuse / coated conductor (pbrt-v4 LayeredBxDF)**, hair,
normalized-fresnel, principled, mix. Textures: image, 2D/3D/nested checkerboard, marble, fbm, windy,
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
| A9, B13, E11, E12 | glass + scattering medium, remaining differences | the medium is simulated now (E3 matches CPU to 2%); E12's *rough* glass blocks shadow rays while CPU lights through it; A9 also has a radius-5000 "world haze" sphere. |
| E6, E7 | cylinder medium / RGB-grid medium | the medium is not representable; the boundary is transparent. |
| C11 | image-textured disk/cylinder lights | only the image's average colour is used; the per-pixel pattern (and so the red channel, ~1/3 low) is not reproduced. |
| J2 | two different image textures on one material | Metal has a single diffuse-image slot (reflectance and transmittance both bound to images). |
| B11 | hair "black fur" | float32 hair BSDF instability for high absorption + narrow lobes; red channel ~1.4x CPU. |
| C17 | missing asset | `sssdragon/textures/small_rural_road_equiarea.exr` is not in the repo; both backends fall back differently. |

Smaller approximations not covered by a scene: shadow rays use shutter time 0 (a moving sphere casts its
shadow at its start position); the layered shaders use one roughness for both interfaces; a marble reflectance on a
*coated diffuse* material renders flat; animated (motion-blurred) bilinear patches and curves are dropped;
scenes that need external meshes beyond the Models set (Sponza, Bistro, the large environments) were never run on Metal.

## Packaging note (macOS)

`scripts/build_and_deploy_macos.sh` builds everything as the architecture of the installed `qmake`. The Qt install on
the dev Mac is x86_64-only, so releases are x86_64 (Rosetta). If the shell is native arm64, run the script as
`arch -x86_64 bash scripts/build_and_deploy_macos.sh` and clear `qt_gui/build_macos` first, or the link fails on mixed
architectures.
