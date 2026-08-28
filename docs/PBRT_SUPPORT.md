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

Four tiers, used consistently across all five tables below:

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
| `conductor` | Approx | Approx | A named metal spectrum (`"spectrum eta"`/`"spectrum k"` = `"metal-<Name>-eta"`/`"-k"`, e.g. Ag/Au/Al/Cu/Fe) OR an explicit `"rgb eta"`/`"rgb k"` resolves to the real complex-IOR GGX `conductor`/`MaterialType::Conductor` model on both — this codebase's own conductor BxDF is already a plain 3-float-RGB model (matching its own named-spectrum table's shape), so an explicit RGB pair needs no spectral upsampling the way pbrt-v4's own real implementation does. Only an unrecognized spectrum name (or giving just one of `eta`/`k`, not both) still falls back to the fuzz-sphere metal model (roughness fed directly as fuzz, not GGX alpha) on both — symmetric either way. Separate `"uroughness"`/`"vroughness"` (anisotropic GGX) are parsed independently and shade as real anisotropy on both CPU and GPU, on all 3 GPU render paths (recursive, wavefront, SPPM's Conductor/RoughDielectric coverage). A material that authors only one axis (e.g. `"uroughness"` alone, leaving `"vroughness"` to its pbrt-v4 default of 0 - near-mirror in v) is a real, distinct axis value, not collapsed to isotropic. GPU's local tangent frame for this material is now UV/`dpdu`-aligned on the recursive and wavefront backends, matching CPU's `ShadingFrame::from_dpdu` (via `BuildDpduTangentFrame`, `src/shared/microfacet.h` - real per-shape `dpdu`, analytic for sphere/disk/cylinder, UV-gradient-solved for triangle, the shape's own edge/patch tangent for quad/bilinear-patch), so anisotropic highlight orientation now matches CPU almost everywhere. GPU SPPM was deliberately left on the older arbitrary (not `dpdu`-aligned, but still continuous/branchless - `BuildArbitraryTangentFrame`, Duff et al. JCGT 2017) frame - it would need `dpdu` threaded through its own separate camera/photon-pass intersection code and per-pixel persisted state, a materially bigger lift for a backend that already only implements 2 of these 4 material kinds - so SPPM's anisotropic conductor/rough-dielectric highlights still won't generally match CPU's orientation. |
| `dielectric` | Full | Full | Smooth by default; a nonzero `roughness` routes to the real `rough_dielectric`/`MaterialType::RoughDielectric` GGX model on both. Separate `"uroughness"`/`"vroughness"` are parsed independently and shade as real anisotropic GGX on both — see `conductor`'s own note above on GPU's dpdu-aligned tangent frame (recursive/wavefront) and SPPM's remaining orientation caveat, both of which apply here too. `"roughness"` bound to a bare `"imagemap"` Texture also shades as real per-hit texture-sampled roughness on CPU and GPU recursive/wavefront (isotropic only), including correct delta-vs-glossy classification under `--sppm`/`--bdpt`/`--mlt` on CPU. GPU SPPM specifically rejects a texture-bound roughness at scene-load time (its payload carries no UV data to sample the texture with) rather than silently ignoring it — use the default path tracer or a GPU backend other than `--sppm` if the texture matters. |
| `thindielectric` | Full | Full | Both use the correct closed-form un-refracted transmission (`R_eff = R + T²R/(1-R²)`), not a solid-glass approximation. |
| `coateddiffuse` | Full | Full | Same layered rough-coat-over-Lambertian model, same 3 parameters (albedo, ior, roughness), on both. `"reflectance"` bound to a real `"imagemap"` `Texture` (optionally `"scale"`-wrapped) is also decoded on both, including both GPU backends' NEE/MIS evaluation — see "Other known gaps" below. Separate `"uroughness"`/`"vroughness"` for the coat are parsed independently and shade as real anisotropic GGX on both — same GPU tangent-frame situation (dpdu-aligned on recursive/wavefront, arbitrary on SPPM) as `conductor` above. |
| `coatedconductor` | Approx | Approx | A named metal spectrum OR an explicit `"rgb eta"`/`"rgb k"` (same resolution as plain `conductor` above, now shared by both kinds) gets the real complex-IOR coat-over-conductor model. Previously neither was ever parsed for this kind at all — even `"metal-Ag-eta"` was silently ignored. "Nothing given" has no pbrt-v4-documented default for this kind (unlike plain `conductor`'s copper default), so it keeps the pre-existing symmetric approximation: base color reinterpreted as normal-incidence reflectance (eta=1, k solved from it). Separate `"uroughness"`/`"vroughness"` for the coat are parsed independently and shade as real anisotropic GGX on both — same GPU tangent-frame situation (dpdu-aligned on recursive/wavefront, arbitrary on SPPM) as `conductor` above. |
| `diffusetransmission` | Full | Full | Separate reflectance/transmittance colors on both. `"reflectance"`/`"transmittance"` bound to a bare `"imagemap"` `Texture` are also decoded on both (no `"scale"`-wrap support — no bundled scene needs it) — `barcelona-pavilion`'s foliage binds both to the same texture, the motivating case. See `pbrt_scenes/barcelona-pavilion` and "Other known gaps" below. |
| `subsurface` | Full | Full | Real tabulated BSSRDF with device probe-walk on both GPU backends (recursive and wavefront), matching CPU's own tabulated BSSRDF. |
| `measured` (real `.bsdf` file) | Full | Full | Both load and flatten the same tensor tables; both fall back to Lambertian on the same "unresolved filename" gate, so they can't disagree about when the fallback applies. |
| `mix` | Full | Full | Real per-shading-point stochastic two-material blend on all three backends now (`MaterialType::Mix`, `optix_types.h`): each hit deterministically (hashed from the world-space hit point, not a fresh random draw — so a radiance bounce and its shadow ray agree on which sub-material won) resolves to sub-material A or B and shades through that material's own real GPU model — a Mix of e.g. `conductor`+`diffuse` keeps the conductor's real specular highlight on GPU, not an averaged flat color. Falls back to the old flat-Lambertian-averaged-color approximation only for a pathologically deep/cyclic mix-of-mix chain (depth-capped, matching CPU's own `kMaxMixDepth`) — not a case any scene in this loader's corpus has needed. See `pbrt_scenes/mix-material.pbrt`. |
| `hair` | Full | Full | Real Marschner/Chiang fiber scattering (`HairBxDF<T>`) on both — `MaterialType::Hair` was already fully wired for GPU shading before this loader could reach it (see `pbrt_scenes/hair-material.pbrt`). `"sigma_a"` wins if given; else `"reflectance"`/`"color"`; else `"eumelanin"`/`"pheomelanin"`; else the default brown preset — all three resolution formulas (`SigmaAFromConcentration`, `SigmaAFromReflectance`) are pbrt-v4's own closed-form per-channel formulas (neither needs an iterative fit). Real Hair support now also reaches every GPU shape type (sphere, quad, triangle, disk, cylinder, bilinear patch), not just sphere — `Material "hair"` on an ordinary shape used to be unreachable from this loader, so those shapes' own `__trap()` guards were previously dead code; wiring `"hair"` up for real exposed them as a genuine crash until each got its own real (if tangent-proxy) Hair branch. Uses the shading normal as a fiber-tangent proxy on both backends for any non-curve shape (same simplification as this project's own native `build_hair_fibers()` demo) — but paired with real `Shape "curve"` geometry, both backends use the curve's own genuine tangent instead (see that entry above and `pbrt_scenes/curve-hair-tuft.pbrt`). |
| `none` / `""` (interface) | Full | Full | pbrt-v4's real interface-material idiom for a shape that bounds a participating medium with no BSDF response of its own — the ray passes straight through completely unperturbed, only the medium changes. Both backends build it as a real, dedicated pass-through material (CPU's `interface_material`; GPU's `MaterialType::Interface`) rather than a near-invisible `Dielectric` approximation — no Fresnel/refraction math, no critical angle. Every integrator (default path tracer, BDPT/MLT, SPPM, both backends) skips the crossing entirely via a real "medium boundary" classification, rather than treating it as a specular bounce that would otherwise spend a bounce-budget entry and break MIS for a light seen through the boundary. Only supported on sphere/disk/cylinder shapes (the only ones with a `medium` field) — a trianglemesh/plymesh/bilinearmesh boundary warns and drops the medium instead of silently doing so. |
| unrecognized | Fallback | Fallback | Falls back to flat Lambertian using the material's base color; the loader warns by name. |

Cross-cutting: a material parameter bound to a pbrt `texture` (rather than a
constant) is dropped to a constant color on both backends — no `MaterialKind`
here carries a texture through this loader. One exception: a `"reflectance"`
or `"k"` bound to a `Texture` of class `"constant"` (pbrt-v4's literal-value
texture, `"float value"`/`"rgb value"`) resolves to that texture's own real
value at flatten() time, not the generic fallback color.

## Lights

CPU: `pbrt_cpu_builder.h`'s light-building code.
GPU: `pbrt_gpu_builder.h`'s light-building code.

| pbrt light | CPU | GPU | Note |
|---|---|---|---|
| `point` | Full | Full | |
| `spot` | Full | Full | Same cone-angle/falloff semantics on both. |
| `distant` | Full | Full | |
| `goniometric` | Full | Full | Both backends now decode the real `filename` image (PNG/BMP/JPG/HDR via stb_image, plus `.exr` via tinyexr on CPU) instead of synthesizing a uniform distribution, matching pbrt-v4's own approach: it reads the profile through its generic `Image::Read()`, not a raw `.ies`-text parser either — a scene author pre-converts a real IES profile to a square equal-area image first (pbrt-v4's own `imgtool makeequiarea`), same as upstream. A non-square image (the equal-area mapping's own requirement) or a missing/undecodable file falls back to the isotropic uniform distribution, matching the prior behavior for every scene that never named a file at all. GPU decodes at native resolution up to 64×64 (`kGonioImageMaxDim`), nearest-neighbor-downsampling a larger real image to fit rather than cropping or falling back; CPU has no such cap (dynamically sized). See `pbrt_scenes/goniometric-projection.pbrt`. |
| `projection` | Full | Full | Both backends now decode the real `filename` slide image the same way (stb_image on both, plus tinyexr on CPU) instead of a uniform white beam — this is the more purely "reuse" of the two, since projection's own evaluation math needed no format-specific handling to begin with. GPU caps at 64×64 (`kProjImageMaxDim`), downsampled when larger; CPU is uncapped. Still warns when a scene omits `filename` entirely (pbrt-v4 requires one). |
| `infinite` (constant color) | Full | Full | |
| `infinite` (HDRI image) | Full | Full | Same equirectangular importance-sampling distribution (luminance-weighted, sin θ Jacobian) built and used on both. |
| `AreaLightSource "diffuse"` | Full | Approx | Real NEE-samplable geometry (sphere/quad/disk/cylinder/triangle/bilinear patch) on both. Both backends now honor `filename` (spatially-varying image emission, real per-point UV) and `twosided` on every shape kind, for both a direct hit and NEE sampling — previously GPU only did a real texture lookup for triangle lights, and every non-triangle GPU light kind fell back to reading texel (0,0); separately, NEE sampling for every light kind on GPU (including triangle) never checked `mat.twoSided` at all, silently treating every one-sided light as two-sided for next-event estimation while a direct BSDF-sampled hit already correctly gated on it. See `pbrt_scenes/textured-twosided-lights.pbrt` for a scene exercising both fixes on disk/cylinder lights. **Correction**: CPU's `filename` lookup (`mipmap_texture`'s `point_sample=true` path, a real bilinear tap at LOD 0 with no mip/EWA footprint filtering) already matches pbrt-v4's own `DiffuseAreaLight::L()` exactly (`Image::BilerpChannel()` — verified against pbrt-v4's real source, which also does a single bilinear tap with no mip selection, not nearest-neighbor and not EWA) — CPU is `Full` here, not an approximation. GPU's own lookup (`sample_texture()`/`nee_light_texture_emission()`, shared by every GPU texture read, not just this light) is genuine nearest-neighbor (`(int)(u*width)`, no interpolation at all) — LESS accurate than both pbrt-v4's spec and CPU's own implementation, so GPU is the one that's `Approx` here. This is a real, verified CPU/GPU asymmetry (a prior version of this doc entry claimed there wasn't one), but fixing it means giving the ENTIRE GPU texture-sampling path real bilinear/mip filtering (a mip pyramid built and uploaded to GPU memory, EWA math in CUDA) — a large, foundational architecture change on the scale of NanoVDB, not an incremental fix scoped to area lights alone; tracked as a known gap, not attempted here. A `"blackbody L"` colour temperature (Kelvin) is now converted to a real RGB colour on both backends via this codebase's own ported pbrt-v4 spectral pipeline (`BlackbodySpectrum` → `SpectrumToXYZ` → the scene's `ColorSpace` directive, `RGBColorSpace::sRGB()` by default — normalized to ~1 nit before the light's own `"float scale"` is applied — matching pbrt-v4's own light-construction code), not read as a raw number — `barcelona-pavilion`'s night lighting and `contemporary-bathroom` both use this for real (2500K–6500K), and previously rendered as flat colourless white regardless of temperature (`getVec3` silently defaulted to `{1,1,1}` for a 1-number `"blackbody"` param, discarding the temperature entirely). Every other punctual light kind (`point`/`spot`/`distant`/`goniometric`/`infinite`) shares the same fix, via `pbrt_flatten::resolveEmissionColor()` — see that function's own comment. |
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

## Film

CPU: `src/TheRestOfYourLife/camera.h` (`crop_x0`/`crop_x1`/`crop_y0`/`crop_y1`,
resolved by `initialize()`; the render loop's `in_crop` gate) - default path
tracer only, see the Note column. GPU: neither backend reads this yet -
`gpu/optix/scene_builder.cpp` warns at scene-load time instead.

| pbrt param | CPU | GPU | Note |
|---|---|---|---|
| `"float[4] cropwindow"` / `"integer[4] pixelbounds"` | Approx | Unsupported | Restricts rendering to a sub-rectangle of the frame - pbrt-v4 allows both together (cropwindow as an NDC fraction, pixelbounds in pixel space), each independently narrowing the region via intersection; both resolve here to one NDC-fraction rectangle (`pbrt_flatten::FlatScene::cropX0`/`X1`/`Y0`/`Y1`) rather than pixel indices, since `xresolution`/`yresolution` are only advisory in this codebase (a CLI width/height argument wins, same as `maxdepth`/`Sampler` type above) - a pixel-space bound resolved against the wrong resolution would be wrong, where a fraction stays correct. **Approx, not Full, on CPU**: real pbrt-v4 writes a smaller *output image* sized to just the crop rectangle; this codebase instead still writes the full `xresolution`×`yresolution` frame, with every pixel outside the crop rectangle traced as black rather than sampled (the existing per-pixel filter-weight-sum-of-zero path already produces this for free, so no separate blit/composite step was needed) - a real, deliberate simplification chosen to avoid rippling a genuinely different output image size through the PPM/EXR writers, the PNG conversion step, and the Qt GUI's preview, all of which currently assume the output image is `image_width`×`image_height`. **CPU is further scoped to the default path tracer only** - same "one integrator gets a feature, the rest don't" shape as `regularize` above, but unlike `regularize` (whose gap on BDPT/MLT/SPPM happens to coincide with pbrt-v4's own off-by-default semantics), a crop request on `--bdpt`/`--mlt`/`--randomwalk`/`--ao`/`--simplepath`/`--simplevolpath`/`--lightpath`/`--sppm` is real, unimplemented divergence - the full frame renders instead, with a console warning (`cpu_interface_bdpt.cpp`, `cpu_interface.cpp`'s SPPM entry point) rather than silence. GPU (both backends) doesn't read this at all yet - a scene that declares a non-full crop gets a console warning at scene-load time and renders the full frame regardless of `--gpu`/`--wavefront`. |

## Integrator

CPU: `src/TheRestOfYourLife/camera.h` (`ray_color()`/`ray_color_spectral()`).
GPU: `gpu/optix/wavefront_kernels.cu` (`evaluate_materials`/
`evaluate_materials_dielectric`) for `--wavefront`; `gpu/optix/optix_device_helpers.h`
(`shade_material()`) for the recursive backend.

| pbrt param | CPU | GPU (recursive) | GPU (wavefront) | Note |
|---|---|---|---|---|
| `"integer maxdepth"` | Approx | Approx | Approx | Advisory only — the scene's own request has no automatic effect; the `--max_depth` CLI arg always wins, with only a console warning printed on mismatch. Same for `Sampler`'s own type (`--sampler` CLI arg wins) and the top-level `Integrator` type string itself (`--bdpt`/`--sppm`/`--mlt`/default CLI flags win) — none of the three is applied unconditionally from the scene, unlike `PixelFilter` and `regularize` below. |
| `"bool regularize"` | Full | Full | Full | pbrt-v4 defaults this `false` and, when `true`, widens a rough BSDF's GGX alpha (`RegularizeAlpha()`, `src/shared/microfacet.h`) after the first non-specular bounce on a path, reducing caustic fireflies at the cost of some bias — applied unconditionally from the scene's own declaration (same shape as `PixelFilter`, not `maxdepth`'s CLI-overridable shape), since it's a genuine scene-authored rendering-behavior toggle, not a perf knob. CPU gates all 3 real `scatter()` call sites in `camera.h`. GPU-wavefront threads it as an explicit parameter through both material-evaluation kernels (`evaluate_materials`, `evaluate_materials_dielectric`) and their host-side launch chains, since this backend has no accessible global `params` the way the recursive backend does. GPU-recursive reads `params.camera.regularize` directly (real global `__constant__` access, unlike wavefront) at its 4 rough-material call sites in `shade_material()` (`optix_device_helpers.h`: conductor, roughdielectric, coatedconductor, coateddiffuse), gated by a new payload register (p23, `numPayloadValues` 23→24) carrying `anyNonSpecularBounces`-so-far from `optix_raygen.h`'s bounce loop into each closest-hit program — the same "closest-hit reads an INPUT register, never writes it" convention p12 already established for `__miss__ms`'s `prev_brdf_pdf`, just mirrored to a different consumer. `anyNonSpecularBounces` itself reuses the codebase's own existing `scatter_brdf_pdf > 0` specular/non-specular proxy (already relied on for MIS), so no separate `is_specular` payload signal needed threading through. BDPT/MLT/SPPM (all CPU-only) never apply regularization on any backend — a separate, pre-existing characteristic unrelated to this flag, already correctly matching pbrt-v4's off-by-default semantics with no code path to gate. **Behavior change**: before this flag was parsed at all, CPU and GPU-wavefront applied this widening *unconditionally* on every non-specular-then-glossy bounce, for every scene — there was no way to turn it off. Now that the default correctly matches pbrt-v4's real `false`, every bundled scene that doesn't explicitly declare `"bool regularize" [true]` renders *without* the firefly reduction it used to always get; only `pbrt_scenes/sportscar/sportscar-area-lights.pbrt` opts back in. A multi-bounce glossy/rough caustic scene may look visibly noisier than before this change — this is the intended pbrt-v4-conformance fix, not a regression. |

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

- `Shape "trianglemesh"`'s `"point2 uv"` parameter is now parsed
  (`pbrt_flatten::Triangle::uv`/`hasUVs`) and threaded through both CPU
  (`triangle_mesh_data::uvs`, the same field OBJ/MTL `vt` data already
  populates) and both GPU backends (`TriangleData::uv0/1/2`, likewise
  already populated by OBJ/MTL loading - this loader just never fed it from
  a pbrt scene before). `Shape "plymesh"` now threads real per-vertex UV too
  (`pbrt_flatten::MeshResolver` gained a `uvs` out-parameter, filled from
  `ply_mesh.h`'s own existing "u"/"v" (or "s"/"t") vertex-property support -
  the PLY parser already read this data, it was just dropped at the
  resolver-callback boundary before reaching a `Triangle`; no CPU/GPU
  builder changes were needed, since both already consume `Triangle::uv`/
  `hasUVs` generically regardless of which shape produced it). See
  `pbrt_scenes/plymesh-uv.pbrt`. `Shape "loopsubdiv"` deliberately still
  does not thread UV - this is not a gap relative to real pbrt-v4, which
  has no UV support on `loopsubdiv` either (no `"uv"`-equivalent parameter
  in its grammar, and `loopsubdiv.cpp`'s own refinement never touches UV) -
  inventing subdivision-surface UV interpolation here would be a new
  feature beyond pbrt-v4 parity, not a bug fix. When a
  trianglemesh gives no `"uv"` at all, pbrt-v4's own real default (a fixed
  `(0,0)/(1,0)/(1,1)` triple per triangle CORNER, not shared across faces)
  is deliberately NOT synthesized - it would inflate vertex-dedup counts at
  every shared vertex for scenes that never read UV at all. Both backends'
  own barycentric-weights fallback (matching CPU `triangle.h`'s pre-existing
  `rec.u=b1,rec.v=b2`) now covers the no-UV case uniformly instead of GPU's
  previous fixed-`(0,0)` default, which is what actually caused the "solid
  black on GPU-recursive" bug this entry used to describe (a fixed UV
  samples the exact same, often-transparent-border texel across the whole
  mesh) and the "GPU triangle light samples one fixed texel" symptom on the
  materials table's `AreaLightSource` row above - both fixed by the same
  change, since both read the same `uv_u`/`uv_v` computation.
- `Shape "disk"`/`Shape "cylinder"` are supported on CPU and both GPU
  backends (recursive and wavefront). CPU keeps the CTM unbaked and is
  exactly correct under arbitrary rotation (see `disk_cylinder_hittable.h`);
  both GPU ports carry the same unbaked object↔world transform in
  `DiskData`/`CylinderData` and apply it by hand in the intersection/
  closest-hit programs, so they're exactly correct under rotation too. A
  GPU disk/cylinder used as an `AreaLightSource` is now registered for real
  NEE sampling on both backends too (`GpuLightKind::Disk`/`::Cylinder`,
  `optix_disk_cylinder_helpers.h`'s `dc_sample_disk`/`dc_sample_cylinder`/
  `dc_pdf_disk`/`dc_pdf_cylinder` - hand-ported device-safe twins of
  `src/shared/shapes.h`'s `DiskShape<T>`/`CylinderShape<T>`, same reason
  `bilinear_patch.h`'s `blp_*` free functions exist instead of instantiating
  those `std::optional`-returning templates on device), matching every other
  GPU light shape - see `pbrt_scenes/disk-cylinder-light.pbrt`. World-space
  area (needed for the NEE sampling weight and the power-weighted alias
  table) is estimated from a single representative scale factor of the
  object→world transform - exact under a similarity transform (rotation/
  translation/uniform scale), approximate under anisotropic scale, the same
  accepted simplification every other GPU area-light kind already carries.
  Separately, CPU wraps a disk/cylinder in a participating medium when
  `MediumInterface` assigns one (matching Sphere's own handling). GPU now
  does too for **cylinder** (`MaterialType::Medium`, homogeneous only):
  `__closesthit__cylinder`/`__closesthit__wf_cylinder` recompute real
  entry/exit roots as the cylinder's tube quadric intersected with its
  z-slab, in object space - same technique as Sphere's own Medium case, see
  `pbrt_scenes/cylinder-medium.pbrt`. Deliberately does not account for a
  partial `phimax` sweep (a "pie slice" cross-section makes the volume
  bound genuinely harder - documented as a scope limit, not handled) and
  only the plain homogeneous medium type gets this - a `"cloud"`/
  `"rgbgrid"`/`"uniformgrid"` `MediumInterface` on a cylinder still
  correctly traps rather than silently misrendering, matching every other
  still-sphere-only type's behavior on a shape it's never assigned to. Disk
  stays entirely unsupported for `MediumInterface` on GPU, and CPU too in
  practice - a zero-thickness plane has no "inside" volume for a
  homogeneous medium's entry/exit pair to bound, so this is structurally
  not meaningful, not merely unimplemented, and isn't planned.

- `Shape "sphere"`'s `"float zmin"`/`"float zmax"`/`"float phimax"`
  (partial-sphere clipping - caps, wedges, hemispheres, e.g. a domed
  skylight cutout) are now supported on **CPU only**. A full (unclipped)
  sphere is rotation-invariant, so `pbrt_flatten.h` keeps baking it straight
  to a world-space center+radius; a clipped one is orientation-dependent, so
  CPU instead carries the real object-to-world transform and intersects in
  object space against `SphereShape<T>`'s existing clipping math
  (`sphere_clipped_hittable.h`) - exactly `disk_hittable`/`cylinder_hittable`'s
  own technique, including their identical "exact under rotation, approximate
  NEE weighting" character. **GPU still renders a clipped sphere as its
  full, unclipped shape** - GPU spheres use OptiX's hardware sphere
  primitive, which has no clipping support at all (unlike disk/cylinder,
  which already use a custom software intersection program on GPU and could
  in principle grow clipping the same way CPU just did); adding that is a
  real, deliberately deferred follow-up, not an oversight. Solid-angle NEE
  sampling of a clipped sphere used as an `AreaLightSource` (`random()`/
  `pdf_value()`, forwarded to `SphereShape<T>::sample_from()`/`pdf_from()`)
  samples over the FULL sphere's subtended cone, not just the visible cap -
  a pre-existing property of that shared template, not something this
  support adds - so this combination gets extra sampling noise (some
  proposed light directions land on the clipped-away part and contribute
  nothing) rather than bias; a narrow, rare combination in practice.

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

- A `Diffuse`, `CoatedDiffuse`, OR `DiffuseTransmission` material's
  `"reflectance"` parameter bound to a bare `"imagemap"` `Texture` is decoded
  and uploaded on both CPU (`mipmap_texture`-backed `lambertian`/
  `coated_diffuse`/`diffuse_transmission`) and GPU (`MaterialData::
  textureIdx` into the same texture table OBJ/MTL `map_Kd` already uses) —
  see `Material::textureFilename` in `pbrt_flatten.h`. `DiffuseTransmission`
  additionally supports its own `"transmittance"` parameter the same way
  (`MaterialData::transmittanceTextureIdx`, a separate field — see
  `Material::transmittanceTextureFilename`) — `barcelona-pavilion`'s foliage
  binds both `"reflectance"` and `"transmittance"` to the identical bare
  imagemap, the motivating (and only bundled) case. A `"scale"`-wrapped
  `"imagemap"` (`barcelona-pavilion`'s own dominant pattern for both
  `coateddiffuse` and plain `diffuse` surfaces) is supported for `Diffuse`/
  `CoatedDiffuse`/`DiffuseTransmission` alike now — reflectance's scale
  folded into the reused `emissionScale` field on GPU, `scaled_texture` on
  CPU (`Material::textureScale`); `DiffuseTransmission`'s own transmittance
  gets an INDEPENDENT scale (`MaterialData::transmittanceScale`, a
  dedicated GPU field rather than reused, since a scene can wrap
  reflectance and transmittance in two differently-valued `"scale"`
  textures — `Material::transmittanceTextureScale` on CPU). `Diffuse`/
  `CoatedDiffuse` (not `DiffuseTransmission`, still) additionally support
  `"checkerboard"`/`"fbm"`/`"marble"`/`"mix"` procedural textures
  (flat-literal `tex1`/`tex2` only, no nested texture references) — on GPU
  this needed no shading-code changes at all, since `sample_texture()`/
  `wf_sample_texture()` already dispatch purely on the resolved
  `TextureData::kind`, not on the consuming `MaterialType`; only the material
  builder needed a `CoatedDiffuse`-side branch to populate `d.textureIdx`
  with one. pbrt's own `ganesha` example scene (a `coateddiffuse` statue,
  bare imagemap) and `barcelona-pavilion`/`contemporary-bathroom` (many
  `coateddiffuse` surfaces, mostly scale-wrapped) are the motivating cases and
  now render with real per-point texture data instead of a flat fallback
  colour on both backends - see `pbrt_scenes/coateddiffuse-texture.pbrt`. Both
  GPU backends' NEE/MIS evaluation also uses the real per-point value for a
  texture-bound `CoatedDiffuse` (the wavefront backend's shared
  `wf_finish_material_scatter` threads the current hit's own UV through for
  this), matching the scatter path exactly.
  Every OTHER material kind's texture-bound parameter (`conductor`'s
  `"eta"`/`"k"`/`"reflectance"`, `dielectric`'s roughness — neither is
  texture-bound by any bundled scene, unlike `diffusetransmission` above)
  still falls back to a flat colour with a warning, unchanged.
  A `checkerboard`/`mix` `Texture`'s own `tex1`/`tex2` now supports ONE
  level of nesting on both backends: either may independently be a flat
  literal (unchanged) OR a reference to another `Texture` that is itself a
  bare `"imagemap"` (`Material::checkerTex1Filename`/`checkerTex2Filename`/
  `mixTex1Filename`/`mixTex2Filename` in `pbrt_flatten.h`) — GPU's
  `TextureData` gained `tex1ImageIdx`/`tex2ImageIdx` for this (`optix_types.h`),
  read by both `sample_texture()` (recursive backend) and
  `wf_sample_texture()` (wavefront); CPU's `uv_checker_texture` already had
  a polymorphic tex1/tex2 constructor (previously unused by this loader),
  and `mix_texture` gained one to match. `mix`'s own `"amount"` parameter is
  ALSO now supported bound to a bare `"imagemap"` `Texture`, same one-level
  scope as `tex1`/`tex2` — a real per-point spatially-varying blend fraction
  (`Material::mixAmountTextureFilename`, GPU `TextureData::amountImageIdx`,
  CPU `mix_texture`'s own texture-taking amount constructor), not a flat
  scalar; `barcelona-pavilion`'s own `materials.pbrt` has a commented-out
  `"float amount"` override on several `Mix` declarations, hinting the
  original scene author considered exactly this. A SECOND level of nesting
  (any of tex1/tex2/amount naming a Texture that is itself a checkerboard/
  fbm/marble/mix, not a bare imagemap) still falls back to the generic "not
  supported" warning — a documented scope cut, not a new limitation: no
  bundled scene needs it, and going further would need real cycle/
  recursion-depth guarding on GPU that a single level doesn't.
  **Update**: `Shape "trianglemesh"`'s own per-vertex `"point2 uv"` data
  (`"st"` is not a pbrt-v4 alias for it - confirmed against pbrt-v4 source,
  only `"uv"` is read) is now threaded through `pbrt_flatten::Triangle`
  and both builders' triangle-construction loops, fixing the "solid black
  on GPU-recursive" divergence this paragraph used to describe - both
  backends now agree on what UV to use, real or a shared barycentric-weights
  fallback, whether or not the scene ever authors `"uv"`. `"plymesh"`'s own
  per-vertex UV (a PLY file property, read through a different code path)
  is now threaded through too - `MeshResolver` gained a `uvs` out-parameter,
  fed by `ply_mesh.h`'s pre-existing "u"/"v" reader.

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
  this holds in practice though it is not enforced. `barcelona-pavilion`'s
  own foliage (each leaf a `Shape "plymesh"`) now benefits from real
  per-vertex UV on the alpha mask too, once its own `.ply` assets carry
  UV data — individual leaf/branch silhouettes were already visibly cut
  out rather than rendering as solid quads even before this fix, via the
  barycentric UV fallback (small enough triangles that it varied usefully
  across them), so this closes a latent accuracy gap rather than a visible
  regression.
