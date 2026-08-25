# Feature Inventory & pbrt-v4 Gap Analysis

What this renderer actually has, backend by backend, and what it's still
missing relative to pbrt-v4. Built by reading the current source directly
(not from commit history or comments, several of which were found stale
while writing this) — see "Stale comments found" at the bottom for the ones
worth fixing.

**Scope note**: this is the broad "what features exist" survey. For the
narrower question of "what happens to each individual pbrt-v4 `.pbrt`
directive when loaded" (per-directive Full/Approx/Fallback/Unsupported,
CPU vs GPU), see [`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) — that table is the
source of truth for loader fidelity; this doc doesn't repeat it.

Backends referenced throughout: **CPU** (`src/TheRestOfYourLife`,
`cpu_renderer/`), **GPU-recursive** (`gpu/optix/recursive_path_tracer.*`,
default `--gpu` mode), **GPU-wavefront** (`gpu/optix/wavefront_*`, opt-in
via `--wavefront`).

---

## Complete feature table

Quick-reference index of every feature this doc covers, in one place.
"Status" packs CPU/GPU-recursive/GPU-wavefront coverage as `Y/Y/Y`-style
shorthand where that shape fits, or plain text where it doesn't (CLI flags,
orphaned code, loader-only limits). "Fallback / Counterpart" is what
actually happens when the feature isn't there — `N/A` means the feature
works as stated and there's no fallback concept to speak of. See the
numbered sections below for the narrative detail behind any row.

| Category | Feature | Status | Fallback / Counterpart |
|---|---|---|---|
| Shapes | Sphere | Y / Y / Y | N/A |
| Shapes | Quad | Y / Y / Y | N/A |
| Shapes | Disk | Y / Y / Y | N/A (medium on a disk is structurally not meaningful, not merely unimplemented) |
| Shapes | Cylinder | Y / Y / Y | N/A |
| Shapes | Triangle / triangle mesh | Y / Y / Y | N/A |
| Shapes | Bilinear patch | Y / Y / Y | N/A |
| Shapes | Curves (hair/fiber strands) | Y (analytic) / Y (tessellated) / Y (tessellated) | GPU tessellates to a bilinear-patch tube instead of native curve intersection — matches pbrt-v4's own GPU strategy, not a degraded fallback |
| Shapes | Loop-subdivision surfaces | Y / Y / Y | N/A (refined to triangles once at load, before either backend sees it) |
| Shapes | OBJ mesh loader | Y / Y / Y | N/A |
| Shapes | PLY mesh loader | Y / Y / Y | N/A |
| Shapes | Instancing (`ObjectInstance`) | Y / Y / Y | N/A |
| Shapes | Non-cubic / non-Bezier curve (`splitdepth`, other basis/degree) | N | Dropped with a "shape not supported" warning — nothing rendered in its place |
| Materials | Lambertian | Y / Y / Y | N/A |
| Materials | Metal (fuzz) / Conductor (real complex-IOR GGX) | Y / Y / Y | N/A |
| Materials | Dielectric (smooth) | Y / Y / Y | N/A |
| Materials | Rough dielectric | Y / Y / Y | N/A |
| Materials | Thin dielectric | Y / Y / Y | N/A |
| Materials | Coated diffuse | Y / Y / Y | N/A |
| Materials | Coated conductor | Y / Y / Y | N/A |
| Materials | Diffuse transmission | Y / Y / Y | N/A |
| Materials | Mix material | Y / Y / Y | N/A |
| Materials | Subsurface (tabulated BSSRDF) | Y / Y / Y | N/A |
| Materials | Hair (Marschner/Chiang) | Y / Y / Y | N/A |
| Materials | Measured (`.bsdf` tensor) | Y / Y / Y | Unresolved filename → falls back to Lambertian (same gate on both backends) |
| Materials | Principled | Y / Y / Y | N/A |
| Materials | Dispersion (`dielectric` only) | CPU only | GPU request → falls back to flat, non-dispersive IOR, silently (no warning — not a scene-load failure) |
| Materials | Dispersion (`rough_dielectric`) | N, no backend | Always flat, non-dispersive IOR |
| Materials | Unrecognized pbrt `Material` kind | N | Falls back to flat Lambertian using base color, warned by name |
| Textures | Procedural (checker/noise/marble/windy/dots/etc.) | Y | N/A |
| Textures | Image textures + mipmap/EWA filter | Y | N/A |
| Textures | Texture mapping (UV/spherical/cylindrical/planar) | Y | N/A |
| Textures | `mix` texture with texture-bound `amount` | N (loader only — class works via native API) | Falls back to flat colour with a warning when loaded from `.pbrt` |
| Textures | Texture nesting beyond one level | N (loader only) | Same — falls back with a warning |
| Lights | Point / spot / distant | Y / Y / Y | N/A |
| Lights | Goniometric | Y / Y / Y | Missing/non-square profile image → falls back to a uniform isotropic distribution |
| Lights | Projection | Y / Y / Y | Missing `filename` → warned (pbrt-v4 itself requires one) |
| Lights | Area (any NEE-samplable shape) | Y / Y / Y | N/A |
| Lights | Uniform infinite | Y / Y / Y | N/A |
| Lights | Image infinite (HDRI, importance-sampled) | Y / Y / Y | N/A |
| Lights | Portal light | Y / Y / Y | N/A |
| Lights | Unrecognized pbrt light kind | N | Dropped, **no fallback rendered** — the one case in the whole loader with no safety net |
| Light sampling | BVH light sampler (spatial + power) | CPU default | GPU counterpart is the flat `PowerLightSampler` — a real, permanent substitute, not a degraded fallback |
| Light sampling | Power light sampler (flat) | CPU (superseded) / GPU (default) | N/A |
| Light sampling | `UniformLightSampler` | Dead code, zero callers | N/A — orphaned, not part of any fallback chain |
| Light sampling | `BVHLightSampler2` | Dead code, zero callers | N/A — orphaned |
| Light sampling | `ExhaustiveLightSampler` | Dead code (only reachable from unwired `restir.h`) | N/A — orphaned |
| Media | Homogeneous | Y / Y | N/A |
| Media | Cloud (procedural Perlin-FBm) | Y / Y | N/A |
| Media | RGB grid | Y / Y | N/A |
| Media | Uniform grid | Y / Y | N/A |
| Media | NanoVDB | N | Falls back to homogeneous medium, warned |
| Media | Medium on disk / triangle mesh / etc. (GPU) | N | No fallback — GPU medium dispatch is sphere/cylinder-triggered only |
| Cameras | Perspective (+ depth of field) | Y / Y / Y | N/A |
| Cameras | Orthographic | Y / Y / Y | N/A |
| Cameras | Spherical (equirect + equal-area) | Y / Y / Y | N/A |
| Cameras | Realistic (lens-file simulation) | Y / Y / Y | Missing/unreadable lens file → falls back to perspective, warned |
| Cameras | Motion blur (`AnimatedTransform`, native API) | Y | N/A |
| Cameras | `ActiveTransform`/`TransformTimes` (`.pbrt`-authored animated camera) | N (loader only) | Directive skipped with a warning; native `AnimatedTransform` API still usable directly |
| Samplers | Sobol / Z-Sobol / padded Sobol / stratified / PMJ02BN / Halton | Y (CPU) | N/A |
| Samplers | Blue noise (bonus, non-pbrt-v4) | Y (CPU) | N/A |
| Samplers | Independent | N | Falls back to Sobol, silently |
| Samplers | `--sampler` under GPU/BDPT/MLT/SPPM/debug integrators | Out of scope by design | Warned and ignored, doesn't error |
| Integrators | Path (default) | Y / Y / Y | N/A |
| Integrators | VolPath (media-aware) | Y / folded into default / Y | N/A |
| Integrators | SPPM | Y / dedicated GPU pipeline / — | N/A |
| Integrators | BDPT | CPU only | `--gpu` request → forced onto CPU, warned, not an error |
| Integrators | MLT | CPU only | Same — forced onto CPU, warned |
| Integrators | RandomWalk / AO / SimplePath / SimpleVolPath / LightPath (debug) | CPU only | N/A (no GPU variant is ever attempted) |
| Integrators | `--denoise` (OptiX AI denoiser) | GPU-recursive only | Silently a no-op under `--wavefront` |
| Spectral | `--spectral` (hero-wavelength Monte Carlo) | CPU, default path tracer only | Combined with GPU/SPPM/BDPT/MLT/debug → flag silently dropped, warned; render proceeds without it |
| Spectral | GPU-wavefront's internal spectral pipeline | Always-on, GPU-wavefront | N/A (not a flag, not togglable — this is just how the integrator works) |
| Spectral | Real accumulating spectral film/sensor (`PixelSensor`/`SpectralFilm`) | N, dead code | N/A — not a fallback scenario, simply unused; every spectral computation reduces to RGB per-sample instead |
| Acceleration | CPU hand-rolled BVH (`bvh.h`) | Y | N/A |
| Acceleration | CPU SAH/HLBVH BVH (`bvh_aggregate.h`) | Y | N/A |
| Acceleration | GPU hardware BVH (OptiX `OptixTraversableHandle`) | Y | N/A |
| Acceleration | Light BVH on GPU | N | Falls back to the flat `PowerLightSampler` (see Light sampling row above) — permanent, not degraded |
| Acceleration | `kd_tree.h` | Present, primary consumer unconfirmed from a source scan | N/A — worth a follow-up to confirm live vs orphaned |
| Tooling | Interactive/real-time progressive preview | N | No fallback — Qt GUI only launches the CLI as a subprocess and parses output; pbrt-v4's own reference implementation also lacks this, so it tracks upstream |
| Tooling | Tone mapping (ACES / Reinhard / none) | Y | N/A |
| Tooling | Video generation (`--video`) | Y | N/A |
| Tooling | System-compatibility diagnostics (`--diagnose`) | Y | N/A |
| Orphaned scaffolding | ReSTIR / reservoir sampler (`restir.h`) | Present, unwired, beyond pbrt-v4 book scope | N/A — not part of any fallback chain |
| Orphaned scaffolding | `PixelSensor` / `SpectralFilm` | Present, unwired | N/A |
| Loader-only | `Accelerator`/`CoordinateSystem`/`ColorSpace` `.pbrt` directives | N | Warned and skipped; rest of scene still loads |

---

## 1. Shapes

| Shape | CPU | GPU-recursive | GPU-wavefront | Notes |
|---|:-:|:-:|:-:|---|
| Sphere | Y | Y | Y | `sphere.h` |
| Quad | Y | Y | Y | `quad.h` |
| Disk | Y | Y | Y | `disk_cylinder_hittable.h` |
| Cylinder | Y | Y | Y | `disk_cylinder_hittable.h` |
| Triangle / triangle mesh | Y | Y | Y | `triangle.h` |
| Bilinear patch | Y | Y | Y | `src/shared/bilinear_patch.h` |
| Curves (hair/fiber strands) | Y — real analytic ray-Bezier | Y — tessellated to bilinear-patch tube | Y — tessellated | Matches pbrt-v4's own CPU-vs-GPU strategy split, not a gap. `type` (flat/cylinder/ribbon) is ignored on GPU — all tessellate to the same round tube. |
| Loop-subdivision surfaces | Y | Y | Y | Refined to triangles once at load (`loop_subdivide.h`), so both backends just see triangles. |
| OBJ mesh loader | Y | Y | Y | `mesh.h` (+ `mesh_mtl.h` for `.mtl`) — hand-written parser. |
| PLY mesh loader | Y | Y | Y | `src/shared/ply_mesh.h` — ASCII + binary LE/BE, arbitrary vertex properties. |
| Instancing (`ObjectInstance`) | Y | Y | Y | `transform_instance.h` + `pbrt_scene.h`'s `ObjectBegin/End/ObjectInstance`. |

No shape gap versus pbrt-v4's own set.

## 2. Materials / BxDFs

Every pbrt-v4 material class has a real implementation on **every**
backend, including GPU-wavefront (the newest and most gap-prone target):
`lambertian`, `metal`(legacy fuzz model)/`conductor`(real complex-IOR GGX),
`dielectric`, `rough_dielectric`, `thin_dielectric`, `coated_diffuse`,
`coated_conductor`, `diffuse_transmission`, `mix_material`, `subsurface`
(real tabulated BSSRDF + device probe-walk), `hair` (real Marschner/Chiang
fiber scattering), `measured` (real Dupuy & Jakob tensor `.bsdf` loader +
eval on all three backends), `principled`, `diffuse_light`, `isotropic`
(medium phase-function material), normal/bump mapping wrappers.

**Dispersion** (wavelength-dependent IOR): `dielectric` only, via a
two-term Cauchy formula (`dielectric::make_dispersive(eta_d, abbe_number)`,
`material_simple.h`) — **CPU only**, and only reachable through
`--spectral`. `mix_material` forwards to a dispersive sub-material
correctly. **Gaps**: no `rough_dielectric` dispersion (flat IOR only even
under `--spectral`); no dispersion anywhere on GPU (recursive or
wavefront).

**Not a gap, just a scope note**: `--spectral`'s own material whitelist
(`cpu_interface.cpp`'s `spectral_scan_hittable()`) only covers 6 of the
materials above — `lambertian`/`metal`/`dielectric`/`rough_dielectric`/
`conductor`/`diffuse_light`. A scene using `coated_diffuse`, `mix_material`,
`subsurface`, `hair`, `measured`, or any medium fails `--spectral` loudly
at load time rather than silently rendering wrong. GPU-wavefront's own
internal spectral pipeline (see §9) has no such restriction — it's
always-on and covers every material GPU-wavefront supports at all.

## 3. Textures

Procedural: checker (2D + 3D), noise/FBm, marble (cubic-Bezier color ramp,
matches pbrt-v4's 9 control points), windy, wrinkled (turbulence), dots,
scale, mix, bilerp, solid color — `src/shared/procedural_textures.h` +
`src/TheRestOfYourLife/texture.h`.

Image textures: Y, with a real mipmap + EWA filter
(`src/shared/mipmap.h`, ported from pbrt-v4's `mipmap.cpp` — 128-entry
Gaussian LUT), point/bilinear/trilinear/EWA all selectable.

Texture mapping: all 4 pbrt-v4 2D mapping types ported — UV, spherical,
cylindrical, planar (`src/shared/texture_mapping.h`).

**Gap**: pbrt's `mix` texture with a texture-bound `amount` (rather than a
constant), and texture nesting past one level, aren't supported when
loading a `.pbrt` scene — see `PBRT_SUPPORT.md` for the exact scope cut
(this is a loader-parsing limit, not a missing texture *class*; the classes
exist and work when built through the native C++ scene API).

## 4. Lights & Light Sampling

All 5 pbrt-v4 punctual light kinds, plus infinite/portal, all real on both
CPU and GPU: point, spot, distant, goniometric (real IES-equirect image
decode), projection (real slide-image decode), area (any NEE-samplable
shape, with real per-point image emission + `twosided`), uniform infinite,
image-based infinite (real importance-sampled HDRI), portal (bounded image
infinite light).

**Light samplers** — this is where real backend asymmetry exists:

| Sampler | Where it's actually wired in |
|---|---|
| `bvh_light_sampler` (light BVH, spatial+power) | **CPU default**, `cpu_interface.cpp` |
| `power_light_sampler` | CPU, superseded by the BVH sampler as default |
| `PowerLightSampler` (flat, power-only) | **GPU default (both recursive and wavefront)** |
| `UniformLightSampler` | present, **zero callers** — dead code |
| `BVHLightSampler2` | present, **zero callers** — dead code |
| `ExhaustiveLightSampler` | only referenced by the unwired `restir.h` (see §11) — not reachable from any render path |

**Gap**: no light BVH on GPU — GPU light sampling is always the flat
power-weighted sampler, which scales worse (linear alias-table draw, no
spatial pruning) on scenes with many lights than CPU's spatial BVH sampler
does. This is a real perf-scaling gap, not a correctness one.

## 5. Media / Volumes

Homogeneous, cloud (procedural Perlin-FBm), RGB grid (per-voxel RGB
sigma_a/sigma_s), uniform grid (single-channel density × RGB sigma_s) — all
four real, on CPU and GPU, DDA majorant-grid delta tracking
(`src/shared/grid_medium.h`).

**Gap**: **NanoVDB is not implemented.** `MakeNamedMedium "nanovdb"`
silently falls back to homogeneous with a warning — this is the one medium
type pbrt-v4 treats as its primary real-world-volumetric-data path (VDB
files from Houdini/etc.) that this codebase has no equivalent for at all.

**Gap**: GPU medium dispatch (all types) is sphere-hit-triggered only — a
`MediumInterface` on a disk/cylinder/trianglemesh has no effect on GPU
(cylinder gained real homogeneous-medium support since; disk is
structurally not meaningful — zero-thickness, no "inside" to bound).

## 6. Cameras

All 4 pbrt-v4 camera types, full parity CPU/GPU: perspective (+ depth of
field), orthographic, spherical (equirect + equal-area), realistic
(lens-file-based, real multi-element simulation on both backends). Motion
blur via `AnimatedTransform` keyframe interpolation.

**Gap**: pbrt's `ActiveTransform`/`TransformTimes` directives (per-keyframe
animated-camera authoring inside a `.pbrt` file) aren't parsed — motion
blur is reachable through this codebase's own native scene-building API,
not from a loaded `.pbrt` file's animated-camera syntax.

No other camera gap.

## 7. Samplers

Sobol, Z-order Sobol, padded Sobol, stratified, PMJ02BN, Halton — all real,
selectable via `--sampler`. Bonus (non-pbrt-v4): blue-noise sampler.

**Gap**: pbrt-v4's `independent` sampler is not ported — an unrecognized
`--sampler` name (including `independent`) silently falls back to Sobol
(`camera.h`'s own comment names this explicitly). Low-priority gap since
Sobol strictly dominates independent sampling in practice, but it means a
`.pbrt` file's own `Sampler "independent"` directive can't be honored
faithfully.

**Gap**: `--sampler` is CPU-default-path-tracer-only — no effect on GPU, or
under BDPT/MLT/SPPM/debug integrators (warns, doesn't error).

## 8. Integrators / Render Modes

| Integrator | CPU | GPU-recursive | GPU-wavefront |
|---|:-:|:-:|:-:|
| Path (default) | Y | Y | Y |
| VolPath (media-aware default) | Y | folded into default | Y |
| SPPM | Y | dedicated GPU pipeline | — |
| BDPT | Y only | — | — |
| MLT | Y only | — | — |
| SimpleVolPath / RandomWalk / AmbientOcclusion / SimplePath / LightPath (debug integrators) | Y only | — | — |

**Gap**: BDPT, MLT, and every debug integrator are CPU-only — no GPU
implementation exists or is planned to exist for any of them (they're
diagnostic/reference-correctness tools, not the primary render path).
This mirrors pbrt-v4's own book scope reasonably closely (pbrt-v4's GPU
target, `pbrt_gpu`, is *also* path/volpath-only — it has no GPU BDPT/MLT
either), so this is arguably not a gap versus upstream at all.

**Note, not quite a gap**: `--denoise` (OptiX AI denoiser, AOV-guided) only
works on GPU-recursive — silently a no-op under `--wavefront`.

## 9. Spectral Rendering

Two independent, non-interacting spectral code paths exist:

- **CPU `--spectral`**: opt-in flag, default path tracer only, 6-material
  whitelist (fails closed on anything else), hero-wavelength Monte Carlo,
  reduced to RGB once per sample (not accumulated into a real spectral
  film). Only source of dispersion in the whole codebase (`dielectric`
  only). See §2.
- **GPU-wavefront's internal spectral pipeline**: always-on, not a flag,
  not user-togglable — this is simply how the wavefront integrator itself
  is implemented internally (CIE tables + D65 + sRGB-upsampling table on
  device). No dispersion.

**Gap**: no real accumulating spectral film/sensor. `PixelSensor`
(`src/shared/pixel_sensor.h`) and `SpectralFilm` (`src/shared/film.h`) are
complete-looking classes that are **dead code** — `camera.h`'s own comment
calls them "never-live-tested," and neither is called from any render
path. Every actual spectral computation reduces to RGB immediately, every
sample, rather than accumulating spectral radiance across the whole image
and reducing once at the end (pbrt-v4's own architecture).

**Gap**: GPU-recursive has no spectral path at all (RGB only).

**Gap**: no dispersion on GPU, and no `rough_dielectric` dispersion
anywhere (see §2).

## 10. Acceleration Structures

CPU has two independent BVH implementations: a hand-rolled book-style one
(`bvh.h`) and a full pbrt-v4-style SAH/HLBVH one
(`src/shared/bvh_aggregate.h`, `BvhSplitMethod::{SAH, Middle, EqualCounts,
HLBVH}`). GPU uses OptiX's own hardware-accelerated BVH (`OptixTraversableHandle`)
for primitive traversal — not this project's own BVH code. Light-sampling
BVH exists (`bvh_light_sampler.h`) but CPU-only (see §4's gap).

A `kd_tree.h` also exists in `src/shared/`; not confirmed as load-bearing
in the primary render path from a source scan alone — worth a targeted
follow-up if it turns out to be dead code too, matching the light-sampler
orphans in §11.

## 11. Present But Not Wired In (orphaned scaffolding)

These are real, apparently-complete implementations with **zero callers**
found anywhere in the render path — not gaps exactly (the code exists),
but not usable features either since nothing reaches them:

- `UniformLightSampler`, `BVHLightSampler2` (`src/shared/`) — superseded by
  `bvh_light_sampler.h`/`power_light_sampler.h`, left in place.
- `ExhaustiveLightSampler` — only referenced by `restir.h`, which is itself
  unwired.
- `restir.h` / `reservoir_sampler.h` (ReSTIR/RIS temporal+spatial reservoir
  reuse) — real-looking scaffolding, beyond pbrt-v4 book scope to begin
  with, never connected to any integrator.
- `PixelSensor` / `SpectralFilm` (§9) — real spectral-film classes, never
  called.

If any of these get finished and wired in, they'd be *additions beyond*
pbrt-v4 parity (ReSTIR especially — it's not part of the pbrt-v4 book at
all), not gap closures.

## 12. Tooling / CLI Surface

`--cpu`/`--gpu`, `--wavefront`, `--optix-validate`, `--denoise`, `--stats`,
`--diagnose` (system-compat report), `--exposure`, `--sampler`,
`--spectral`, `--sppm[-iterations|-photons]`, `--bdpt[-max-depth]`,
`--mlt[-bootstrap|-mutations|-max-depth]`, `--randomwalk`, `--ao[-max-dist|
-uniform|-illum-scale|-illum-rgb]`, `--simplepath[-no-lights|-no-bsdf]`,
`--simplevolpath`, `--lightpath`, `--video[-frames|-fps|-speed|
-camera-path|-preset]`. Full validation/mutual-exclusion logic in
`launcher/main.cpp`.

Tone mapping: ACES (Narkowicz approx, default), Reinhard, or none.

**Gap**: no real-time/interactive progressive preview. The Qt GUI launches
the CLI as a subprocess and parses its output — there's no live progressive
raster or interactive viewport (pbrt-v4 itself doesn't have one either in
the book/reference implementation, so this tracks upstream, not a gap
relative to it specifically).

---

## Summary: gaps from pbrt-v4, ranked by how much they'd actually matter

1. **NanoVDB heterogeneous media** (§5) — the clearest, most consequential
   gap. pbrt-v4's primary real-world volumetric path; this codebase has no
   equivalent, only procedural/flat-grid media.
2. **No real spectral film/sensor accumulation** (§9) — `--spectral`
   reduces to RGB every sample instead of accumulating spectral radiance
   pbrt-v4-style; the dead `PixelSensor`/`SpectralFilm` classes suggest
   this was planned and abandoned partway.
3. **No GPU dispersion, no `rough_dielectric` dispersion anywhere** (§2, §9)
   — dispersion exists only for smooth CPU `dielectric`.
4. **No GPU light BVH** (§4) — GPU light sampling doesn't spatially scale
   the way CPU's does on many-light scenes.
5. **`independent` sampler not ported** (§7) — minor; Sobol dominates it in
   practice.
6. **`ActiveTransform`/`TransformTimes`, `Accelerator`, `CoordinateSystem`,
   `ColorSpace` pbrt directives not parsed** — narrows what a *loaded*
   `.pbrt` file can express, even though the underlying native features
   (motion blur, BVH choice) exist and work through this codebase's own
   scene-building API.
7. **BDPT/MLT/debug integrators CPU-only** — arguably tracks pbrt-v4's own
   GPU scope (path/volpath only), so more a parity-with-upstream item than
   a true gap.

## Fallback behavior: what happens when a feature is missing

Most gaps above degrade gracefully rather than failing outright — but not
all of them, and it's worth knowing which is which before relying on one.

**Falls back to something functional (warns, keeps rendering):**

- `MakeNamedMedium "nanovdb"` → falls back to a homogeneous medium, with a
  warning. Not real VDB data, but the render doesn't break.
- An unrecognized `--sampler` name (including `independent`) → silently
  falls back to Sobol.
- An unrecognized pbrt `Material` kind → falls back to flat Lambertian
  using the material's base color, with a named warning.
- `realistic` camera with a missing/unreadable lens file → falls back to
  perspective, with a warning.
- Goniometric/projection light with a missing or non-square image → falls
  back to a uniform/isotropic distribution.
- `--bdpt`/`--mlt` combined with `--gpu` → forced onto CPU with a warning
  ("--gpu is ignored under --bdpt/--mlt"), not an error.
- `--spectral` combined with `--gpu`/`--sppm`/`--bdpt`/`--mlt`/any debug
  integrator → the flag is silently dropped with a warning; you still get
  an ordinary render on whatever backend/mode you asked for.
- A `.pbrt` directive this loader doesn't recognize at all (`Accelerator`,
  `CoordinateSystem`, `ActiveTransform`, `ColorSpace`, etc.) → warned and
  skipped; the rest of the scene still loads.

**No counterpart — genuinely absent, dropped rather than approximated:**

- An unrecognized pbrt *light* kind → dropped with a warning, and unlike
  materials there is no fallback light rendered in its place — it's simply
  invisible. This is the one case in the whole loader with no safety net:
  everything else either substitutes something visible or forces a backend/
  flag change loudly; an unsupported light just disappears.
- A `Shape` type this loader can't build (e.g. a non-cubic/non-Bezier
  `curve`) → dropped with a "shape not supported" warning; nothing is
  rendered in its place.
- Dispersion on GPU, or on `rough_dielectric` anywhere → no approximate
  dispersion; it's just flat, non-dispersive IOR, silently (no warning,
  since this isn't a scene-loading failure — it's simply a code path that
  was never built).
- The orphaned scaffolding (§11: `UniformLightSampler`, `BVHLightSampler2`,
  `ExhaustiveLightSampler`, ReSTIR, `PixelSensor`/`SpectralFilm`) — these
  aren't fallbacks *for* anything and don't *have* fallbacks either; they're
  unreachable code with zero callers, not part of any fallback chain.

**Working counterpart already in place (not a degraded fallback — just the
permanent behavior):**

- No GPU light BVH → GPU always uses the flat `PowerLightSampler` instead.
  That's a real, working counterpart, not a failure mode — it's the
  permanent GPU default, just with worse scaling on many-light scenes than
  CPU's spatial BVH sampler.
- No real spectral film accumulation → CPU `--spectral` reduces to RGB
  every sample rather than accumulating spectral radiance. This isn't a
  fallback triggered by failure — it's simply how the feature is
  architected; there's no better mode it's falling back *from*.

## Stale comments found while building this doc

Worth fixing separately (not done here — this is a survey, not a patch):

- `launcher/launcher_args.h`'s SPPM doc comment calls it "a separate
  CPU-only render mode" — false; `main.cpp` has a fully wired GPU SPPM path.
- `src/shared/pbrt_scene.h`'s top-of-file comment says a real pbrt scene
  "will always contain something we do not implement yet (subsurface,
  curves, instancing, media)" — stale; all four are implemented and
  directive-parsed now. Only NanoVDB media remains genuinely unimplemented.
