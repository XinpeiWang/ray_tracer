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
| `hair` | Full | Full | Real Marschner/Chiang fiber scattering (`HairBxDF<T>`) on both — `MaterialType::Hair` was already fully wired for GPU shading before this loader could reach it (see `pbrt_scenes/hair-material.pbrt`). `"sigma_a"` wins if given; else `"reflectance"`/`"color"`; else `"eumelanin"`/`"pheomelanin"`; else the default brown preset — all three resolution formulas (`SigmaAFromConcentration`, `SigmaAFromReflectance`) are pbrt-v4's own closed-form per-channel formulas (neither needs an iterative fit). Real Hair support now also reaches every GPU shape type (sphere, quad, triangle, disk, cylinder, bilinear patch), not just sphere — `Material "hair"` on an ordinary shape used to be unreachable from this loader, so those shapes' own `__trap()` guards were previously dead code; wiring `"hair"` up for real exposed them as a genuine crash until each got its own real (if tangent-proxy) Hair branch. Uses the shading normal as a fiber-tangent proxy on both backends for any non-curve shape (same simplification as this project's own native `build_hair_fibers()` demo) — but paired with real `Shape "curve"` geometry, both backends use the curve's own genuine tangent instead (see that entry above and `pbrt_scenes/curve-hair-tuft.pbrt`). |
| unrecognized | Fallback | Fallback | Falls back to flat Lambertian using the material's base color; the loader warns by name. |

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
| `AreaLightSource "diffuse"` | Approx | Approx | Real NEE-samplable geometry (sphere/quad/triangle/bilinear patch) on both. CPU parses and honors `filename` (spatially-varying image emission, point-sampled) and `twosided` on any shape. GPU honors both too, but triangle-only (both backends) - the only shape a raw pbrt trianglemesh light ever builds as; a `filename`/`twosided` on a `Shape "sphere"`/`"disk"`/`"cylinder"` area light silently falls back to flat `L`/one-sided on GPU. A GPU triangle light's `filename` image also samples a fixed texel rather than varying spatially, since pbrt trianglemesh `"point2 uv"` isn't parsed at all yet (see "Other known gaps" below) - real on CPU only by accident, via triangle.h's barycentric UV fallback. `blackbody` emission is read as a raw number rather than converted, on both (warned). |
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

- `AreaLightSource`'s `twosided` and `filename` parameters are honored on
  CPU for any shape, and on GPU for triangle lights only (see the
  backend-asymmetric table above) - a sphere/disk/cylinder area light with
  either param set falls back to flat/one-sided on GPU, tracked as
  follow-up work.
- A raw pbrt `Shape "trianglemesh"`'s `"point2 uv"` parameter is never
  parsed anywhere in this codebase's loader (`pbrt_scene.h`/
  `pbrt_flatten.h`), on either backend. CPU usually looks correct anyway
  via `triangle.h`'s barycentric-coordinates-as-UV fallback when no real
  UV was ever set; GPU's `TriangleData` has no equivalent fallback
  (`hasUVs` stays false, so UV stays a fixed (0,0) instead), which is most
  visible on a `filename`-textured GPU triangle light: it samples one fixed
  texel instead of the image's real per-point detail.
- `Shape "disk"`/`Shape "cylinder"` are supported on CPU and both GPU
  backends (recursive and wavefront). CPU keeps the CTM unbaked and is
  exactly correct under arbitrary rotation (see `disk_cylinder_hittable.h`);
  both GPU ports carry the same unbaked object↔world transform in
  `DiskData`/`CylinderData` and apply it by hand in the intersection/
  closest-hit programs, so they're exactly correct under rotation too — but
  unlike CPU, a GPU disk/cylinder used as an `AreaLightSource` is not yet
  registered for explicit NEE sampling on either backend (no
  `GpuLightKind::Disk`/`::Cylinder`), so it still emits when directly hit but
  converges noisier than the CPU render of the same scene. Similarly, CPU
  wraps a disk/cylinder in a participating medium when `MediumInterface`
  assigns one (matching Sphere's own handling); GPU does not yet read
  `DiskData`/`CylinderData`'s medium field on either backend, so the same
  directive silently produces a disk/cylinder with no medium on GPU.

- `Shape "curve"` (a cubic Bezier hair/fiber strand) is supported on CPU
  (`src/shared/shapes.h`'s `CurveShape<T>`, wrapped by
  `curve_shape_hittable.h` - real ray-Bezier recursive-subdivision
  intersection) and both GPU backends via tessellation into a tube of
  bilinear patches (`src/shared/curve_tessellate.h`) rather than a native
  curve-intersection program - neither GPU backend has one, matching
  pbrt-v4's own GPU strategy for the same reason (dicing curves is a much
  better fit for the GPU than porting the CPU's recursive-subdivision
  algorithm). This means GPU renders a close but not pixel-identical
  approximation of a curve's exact silhouette, and does not distinguish
  `"type"` (flat/cylinder/ribbon all tessellate to the same round tube) -
  pbrt-v4 has the identical divergence for the identical reason. Only cubic
  (`"integer degree"` 3), Bezier-basis (`"string basis"` `"bezier"`) curves
  are built; a b-spline basis or non-cubic degree falls back to the generic
  "shape not supported" warning. `"integer splitdepth"` is not implemented -
  pbrt-v4 itself forces it to 0 whenever GPU rendering is active, so omitting
  it matches pbrt-v4's own GPU-mode behavior. See `pbrt_scenes/curve-tuft.pbrt`
  for a worked example paired with an ordinary `Material "diffuse"`, and
  `pbrt_scenes/curve-hair-tuft.pbrt` paired with real `Material "hair"`
  fiber shading (see the materials table above) — the latter needs a real
  fiber-tangent axis, which real curve geometry is the one shape here that
  actually has: CPU's `curve_shape_hittable` sets `hit_record::dpdu` to it,
  and both GPU backends' bilinear-patch closest-hit programs (the
  tessellated-curve primitive) compute their own patch dpdu and pass it to
  the Hair branch instead of the shading normal every other Hair-material
  shape (e.g. a plain sphere) still uses as a proxy — see
  `hair_material.h`'s `tangent_is_dpdu` parameter comment for the full
  reasoning and `optix_intersection_bilinear_patch.h`/`wavefront_kernels.cu`'s
  own Hair branches for the GPU mirror.

- `MakeNamedMedium`'s `"type"` parameter supports `"homogeneous"` (the
  default), `"cloud"` (Perlin-FBm density, `src/shared/cloud_medium.h`),
  `"rgbgrid"` (a flat per-voxel `"rgb sigma_a"`/`"rgb sigma_s"` grid,
  `src/shared/rgb_grid_medium.h`) and `"uniformgrid"` (a single-channel flat
  per-voxel `"float density"` grid scaled by a single `"rgb sigma_s"`,
  `src/shared/sampled_grid.h`'s `GridMediumData`) on both backends — CPU via
  `cloud_medium_hittable`/`rgb_grid_medium_hittable`/`grid_medium_hittable`,
  GPU via `MaterialType::CloudMedium`/`::RgbGridMedium`/`::GridMedium`, all
  wired through `pbrt_scene.h`'s `MediumDecl::xform` (the CTM captured at
  declaration time) and `pbrt_flatten.h`'s world-space AABB/world↔medium-
  transform computation. Like the pre-existing homogeneous case, GPU
  dispatch for cloud/rgbgrid/uniformgrid is sphere-hit-triggered only (a
  `MediumInterface` on a disk/cylinder/trianglemesh has no GPU effect — see
  the disk/cylinder gap above). `pbrt_scenes/cloud-medium.pbrt`,
  `pbrt_scenes/rgbgrid-medium.pbrt` and `pbrt_scenes/uniformgrid-medium.pbrt`
  are worked examples of all three. Any other `"type"` value (e.g.
  `"nanovdb"`) still falls back to homogeneous with a warning.

(The `dielectric roughness` and `conductor` routing gaps once listed here were
fixed — see the Materials table above, which is the source of truth for
per-`MaterialKind` behavior.)

- A `Diffuse` material's `"reflectance"` parameter bound to an `"imagemap"`
  `Texture` is decoded and uploaded on both CPU (`mipmap_texture`-backed
  `lambertian`) and GPU (`MaterialData::textureIdx` into the same texture
  table OBJ/MTL `map_Kd` already uses) — see `Material::textureFilename` in
  `pbrt_flatten.h`. Every OTHER material kind's texture-bound parameter (e.g.
  `coateddiffuse`'s `"reflectance"` — pbrt's own `ganesha` example scene uses
  exactly this) and every other `Texture` class (`checkerboard`, `scale`,
  `mix`, ...) still falls back to a flat colour with a warning, unchanged.
  **Known limitation even for the supported Diffuse+imagemap case**: pbrt
  `Shape "trianglemesh"`/`"plymesh"`'s own per-vertex `"uv"`/`"st"` data is not
  threaded through `pbrt_flatten::Triangle` at all (no `u`/`v` fields on that
  struct) — confirmed by hand: a synthetic `Material "diffuse"` +
  `"imagemap"` scene renders a real (if not correctly UV-mapped) texture on
  CPU, but solid black on GPU-recursive, because the two backends' triangle
  code disagrees about what UV to use when none was authored through this
  path. Fixing this needs UV added to `pbrt_flatten::Triangle` and threaded
  through both builders' triangle-construction loops — a separate, real gap,
  not something this texture-upload work fixes on its own.

- A pbrt `Shape`'s own `"alpha"` parameter (an alpha-cutout mask, distinct
  from a Material's own texture-bound parameters above — pbrt authors it
  per-shape, e.g. `barcelona-pavilion`'s foliage: each leaf `Shape "plymesh"`
  gives its own `"texture alpha"`, reusing its colour photo's red channel as
  the mask, matching OBJ/MTL's own `map_d` convention) is now decoded and
  wired into `MaterialData::alphaMaskTexIdx` — the SAME field `map_d` already
  drives on both GPU backends' any-hit/closest-hit alpha tests, and CPU's own
  `triangle::hit()` — see `Material::alphaTextureFilename` in
  `pbrt_flatten.h`. Attached to the Shape's own resolved *material* (not a
  new per-triangle field): every scene in this loader's own corpus gives each
  alpha-masked Shape its own unnamed Material declared immediately before it,
  never a `NamedMaterial` shared by shapes with different alpha masks, so
  this holds in practice though it is not enforced. Shares the reflectance
  case's own UV-threading gap above (pbrt trianglemesh/plymesh has no real
  per-vertex UV) — verified in practice on `barcelona-pavilion`'s own
  foliage anyway: individual leaf/branch silhouettes are visibly cut out
  rather than rendering as solid quads, likely because each leaf's own
  triangles are small enough that the barycentric UV fallback still varies
  usefully across them.
