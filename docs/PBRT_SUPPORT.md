# pbrt-v4 Loader Support Matrix

What happens to each pbrt-v4 directive this codebase's loader recognizes,
on the CPU renderer and on the GPU (OptiX) renderer, once a `.pbrt` scene
file is loaded via `--scene <path-to-file>.pbrt`.

This exists because "will this scene look the same on GPU as on CPU" was
previously only answerable by reading `src/TheRestOfYourLife/pbrt_cpu_builder.h`,
`gpu/optix/pbrt_gpu_builder.h`, and `gpu/optix/scene_builder.cpp`'s camera
code together, and because two real CPU/GPU divergences this codebase hit
(`ThinDielectric` mapped to the wrong CPU class; GPU never reading a loaded
scene's `lensradius` at all) were exactly the shape of gap this table exists
to make visible before it turns into a rendering bug.

Four tiers, used consistently across all three tables below:

- **Full** — matches pbrt-v4 semantics on that backend.
- **Approx** — a documented, deliberate simplification. The Note column
  says what's simplified.
- **Fallback** — silently or loudly downgrades to a different, simpler
  representation. The Note column says what it downgrades to.
- **Unsupported** — not handled; the loader warns and the directive is
  dropped.

This table reflects the loader's actual current behavior, verified against
source (not comments — several `src/shared/pbrt_flatten.h` comments were
found to be stale relative to the code they describe; corrections are noted
inline where relevant). It is not auto-generated or test-enforced — if you
change a builder's handling of one of these, update this file in the same
change.

## Materials (`MaterialKind`, `src/shared/pbrt_flatten.h`)

CPU: `src/TheRestOfYourLife/pbrt_cpu_builder.h`'s `makeMaterial()`.
GPU: `gpu/optix/pbrt_gpu_builder.h`'s material switch.

| pbrt kind | CPU | GPU | Note |
|---|---|---|---|
| `diffuse` | Full | Full | Plain Lambertian on both. |
| `conductor` | Approx | Approx | A named metal spectrum (`"spectrum eta"`/`"spectrum k"` = `"metal-<Name>-eta"`/`"-k"`, e.g. Ag/Au/Al/Cu/Fe) resolves to the real complex-IOR GGX `conductor`/`MaterialType::Conductor` model with tabulated eta/k, on both backends. An explicit RGB `eta`/`k`, or an unrecognized spectrum name, still falls back to the fuzz-sphere metal model (roughness fed directly as fuzz, not GGX alpha) on both — symmetric either way. |
| `dielectric` | Full | Full | Smooth by default; a nonzero `roughness` routes to the real `rough_dielectric`/`MaterialType::RoughDielectric` GGX model on both. pbrt's separate `uroughness`/`vroughness` (anisotropic) aren't parsed independently — whichever one is present is used as a single isotropic roughness. |
| `thindielectric` | Full | Full | Both use the correct closed-form un-refracted transmission (`R_eff = R + T²R/(1-R²)`), not a solid-glass approximation. |
| `coateddiffuse` | Full | Full | Same layered rough-coat-over-Lambertian model, same 3 parameters (albedo, ior, roughness), on both. |
| `coatedconductor` | Approx | Approx | Symmetric approximation: base color reinterpreted as normal-incidence reflectance (eta=1, k solved from it). pbrt's real `conductor.eta`/`conductor.k` sub-parameters aren't parsed. |
| `diffusetransmission` | Full | Full | Separate reflectance/transmittance colors on both. |
| `subsurface` | Full | Full | Real tabulated BSSRDF with device probe-walk on both GPU backends (recursive and wavefront), matching CPU's own tabulated BSSRDF. |
| `measured` (real `.bsdf` file) | Full | Full | Both load and flatten the same tensor tables; both fall back to Lambertian on the same "unresolved filename" gate, so they can't disagree about when the fallback applies. |
| `mix` | Full | **Fallback** | CPU does a real recursive stochastic two-material blend. **GPU flattens it to a single flat Lambertian** whose albedo is the mix-weighted average of the two sub-materials' resolved colors — the largest remaining CPU/GPU material gap in this loader. |
| unrecognized / `hair` | Fallback | Fallback | Both fall back to flat Lambertian using the material's base color; the loader warns by name. |

Cross-cutting: a material parameter bound to a pbrt `texture` (rather than a
constant) is dropped to a constant color on both backends — no `MaterialKind`
here carries a texture through this loader.

## Lights

CPU: `pbrt_cpu_builder.h`'s light-building code.
GPU: `pbrt_gpu_builder.h`'s light-building code.

| pbrt light | CPU | GPU | Note |
|---|---|---|---|
| `point` | Full | Full | |
| `spot` | Full | Full | Same cone-angle/falloff semantics on both. |
| `distant` | Full | Full | |
| `goniometric` | Approx | Approx | Neither backend decodes a real IES profile file — both synthesize a uniform (isotropic) intensity distribution instead. The light's aim/orientation (CTM-derived rotation) is still correct on both; only the per-direction intensity shape is approximated. The loader warns when a scene actually names a profile file. |
| `projection` | Approx | Approx | Same story as goniometric: neither backend decodes the real projected slide image, both use a uniform white slide instead. Warns when a scene names a real image. |
| `infinite` (constant color) | Full | Full | |
| `infinite` (HDRI image) | Full | Full | Same equirectangular importance-sampling distribution (luminance-weighted, sin θ Jacobian) built and used on both. |
| `AreaLightSource "diffuse"` | Approx | Approx | Real NEE-samplable geometry (sphere/quad/triangle/bilinear patch) on both, one-sided on both. `twosided` is not parsed on either backend (silently always one-sided, no warning); `blackbody` emission is read as a raw number rather than converted, on both (warned). |
| anything else | Unsupported | Unsupported | Dropped with a warning; not visible on either backend. |

## Cameras (`Camera::type`)

CPU: `src/TheRestOfYourLife/scene_registry.h`'s `setup_camera` lambda for
loaded pbrt scenes. GPU: `gpu/optix/scene_builder.cpp`'s
`build_loaded_pbrt_scene()`.

| pbrt camera | CPU | GPU | Note |
|---|---|---|---|
| `perspective` (pinhole) | Full | Full | |
| `perspective` + `lensradius` (depth of field) | Full | Full | Both convert pbrt's `lensradius` (a world-space lens radius) through the same shared `defocusAngleDegreesFor()`/`focusDistanceFor()` helpers before applying it, so there's no unit mismatch between them. |
| `orthographic` | Full | Full | Both honor an explicit `screenwindow`, and fall back to the same computed default window otherwise. |
| `spherical` — equirectangular | Full | Full | `"environment"` accepted as an alias for `"spherical"` on both. |
| `spherical` — equalarea | Full | Full | Both do the real pbrt-v4 concentric-octahedral equal-area mapping (`EqualAreaSquareToSphere`); GPU keeps a small local device-side copy of the math on each backend rather than including the CPU header (same pattern as the rest of this codebase's device helpers). |
| `realistic` (lens file) | Full | Full | Both parse the same lens-file format and build a real multi-element lens simulation (GPU reuses the same host-side `RealisticCamera` and flattens it to device buffers); both fall back to perspective with a warning if the lens file is missing/unreadable. |

## Stale comments corrected while building this table

These `src/shared/pbrt_flatten.h` comments describe an earlier state of the
loader and no longer match the code:

- A comment near `MaterialKind::Subsurface` claiming "GPU has no BSSRDF
  implementation" — false; both GPU backends have real tabulated-BSSRDF
  probe-walk support.
- A comment near `MaterialKind::Measured` claiming "GPU has no measured-BRDF
  implementation" — false; GPU flattens and uploads the same tensor tables.
- A comment on `struct InfiniteLight` claiming "distant, point and spot
  lights are still dropped" — false; all five punctual light kinds are
  supported on both backends.

## Other known gaps (not backend-asymmetric, but worth knowing)

- `AreaLightSource`'s `twosided` parameter is parsed nowhere; every area
  light is one-sided on both backends regardless of what the scene requests.
- `Shape "disk"`/`Shape "cylinder"` are supported on CPU and the GPU-recursive
  backend; the GPU-wavefront backend does not build them yet (it refuses to
  render a scene containing either, rather than mis-rendering it — see
  `WavefrontPathTracer::render()`'s own guard). CPU keeps the CTM unbaked and
  is exactly correct under arbitrary rotation (see `disk_cylinder_hittable.h`);
  the GPU-recursive port carries the same unbaked object↔world transform in
  `DiskData`/`CylinderData` and applies it by hand in the intersection/
  closest-hit programs, so it's exactly correct under rotation too — but
  unlike CPU, a GPU-recursive disk/cylinder used as an `AreaLightSource` is
  not yet registered for explicit NEE sampling (no `GpuLightKind::Disk`/
  `::Cylinder`), so it still emits when directly hit but converges noisier
  than the CPU render of the same scene.

(The `dielectric roughness` and `conductor` routing gaps once listed here were
fixed — see the Materials table above, which is the source of truth for
per-`MaterialKind` behavior.)
