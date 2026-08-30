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
tracer only, see the Note column. GPU: both backends (recursive and
wavefront) honor it too - `gpu/optix/scene_builder.cpp` resolves the same
NDC-fraction rectangle to pixel bounds at scene-build time, threaded via
`GpuCameraParams::cropX0`/`X1`/`Y0`/`Y1`; `gpu_in_crop()`
(`optix_device_helpers.h`, duplicated in `wavefront_kernels.cu` per that
file's own "separate translation unit" convention) is the device-side gate.

| pbrt param | CPU | GPU | Note |
|---|---|---|---|
| `"float[4] cropwindow"` / `"integer[4] pixelbounds"` | Approx | Approx | Restricts rendering to a sub-rectangle of the frame - pbrt-v4 allows both together (cropwindow as an NDC fraction, pixelbounds in pixel space), each independently narrowing the region via intersection; both resolve here to one NDC-fraction rectangle (`pbrt_flatten::FlatScene::cropX0`/`X1`/`Y0`/`Y1`) rather than pixel indices, since `xresolution`/`yresolution` are only advisory in this codebase (a CLI width/height argument wins, same as `maxdepth`/`Sampler` type above) - a pixel-space bound resolved against the wrong resolution would be wrong, where a fraction stays correct. **Approx, not Full, on every integrator/backend**: real pbrt-v4 writes a smaller *output image* sized to just the crop rectangle; this codebase instead still writes the full `xresolution`×`yresolution` frame, with every pixel outside the crop rectangle left explicit black rather than sampled/traced - CPU via the existing per-pixel filter-weight-sum-of-zero path (default path tracer) or an equivalent per-pixel/per-splat crop gate (BDPT/MLT/RandomWalk/AO/SimplePath/SimpleVolPath/LightPath/SPPM, see below), GPU via an early-return in each backend's own primary-ray-generation kernel (`__raygen__rg`'s explicit black write; `generate_camera_rays`'s skip-the-enqueue; GPU SPPM's `__raygen__sppm_camera_pass` early-return, relying on its own pixel buffer's one-time zero-init) - a real, deliberate simplification everywhere, chosen to avoid rippling a genuinely different output image size through the PPM/EXR writers, the PNG conversion step, and the Qt GUI's preview, all of which currently assume the output image is `image_width`×`image_height`. **Now honored by every CPU integrator and both main GPU backends, plus GPU SPPM** - `--bdpt`/`--mlt`/`--randomwalk`/`--ao`/`--simplepath`/`--simplevolpath` skip the whole per-pixel loop body for an out-of-crop pixel (`bdpt_adapter.h`'s `*_render_with_adapter()` drivers); `--lightpath` and BDPT's own t==1 light-tracing strategy have no per-pixel loop to skip (samples land at essentially arbitrary pixels), so `SplatFilm` itself gates each splat against the crop rect instead; `--mlt`'s own Markov-chain splat lambda does the identical gate inline, since chain mutations aren't pixel-indexed either - none of this needs renormalization, since each accepted splat already carries its own correct weight regardless of how many other splats were dropped. CPU `--sppm` (`sppm_adapter.h`'s `sppm_camera_pass_with_sky()`) skips the camera pass for an out-of-crop pixel every iteration, leaving its visible point permanently invalid (`SPPMFinalImage()` already reconstructs an untouched pixel as black with no divide-by-zero risk, since `radius` stays at its nonzero initial value). GPU SPPM's own `SPPMLaunchParams::camera` is a direct copy of the same `GpuCameraParams` the other two GPU backends already use, so it already carried a resolved crop rectangle with nothing reading it - `__raygen__sppm_camera_pass` now does. The wavefront backend's own `generate_camera_rays` kernel (launched once per SAMPLE, unlike the recursive backend's single whole-render launch) goes further than an early-return: `wf_launch_generate_camera_rays` (`wavefront_launch.cu`) sizes its CUDA launch grid to just the crop rectangle when one is active, so a cropped-out pixel's GPU thread is never scheduled at all on that backend, not merely skipped after the fact. |

## Integrator

CPU: `src/TheRestOfYourLife/camera.h` (`ray_color()`/`ray_color_spectral()`).
GPU: `gpu/optix/wavefront_kernels.cu` (`evaluate_materials`/
`evaluate_materials_dielectric`) for `--wavefront`; `gpu/optix/optix_device_helpers.h`
(`shade_material()`) for the recursive backend.

| pbrt param | CPU | GPU (recursive) | GPU (wavefront) | Note |
|---|---|---|---|---|
| `"integer maxdepth"` | Approx | Approx | Approx | Advisory only — the scene's own request has no automatic effect; the `--max_depth` CLI arg always wins, with only a console warning printed on mismatch. Same for `Sampler`'s own type (`--sampler` CLI arg wins) and the top-level `Integrator` type string itself (`--bdpt`/`--sppm`/`--mlt`/default CLI flags win) — none of the three is applied unconditionally from the scene, unlike `PixelFilter` and `regularize` below. |
| `"string lightsampler"` | Approx | Unsupported | Unsupported | Which light sampler picks the next-event-estimation light — one of `"uniform"`/`"power"`/`"bvh"`; pbrt-v4 defaults this to `"bvh"`. Same "advisory only, CLI decides, warn on mismatch" shape as `maxdepth`/`Sampler` above (not `PixelFilter`/`regularize`'s "applied unconditionally" one) — a light sampler's choice affects convergence/variance, not the converged image, so it's a perf/quality knob rather than a genuine rendering-behavior toggle. CPU: a new `--lightsampler` CLI flag (default `bvh`, matching pbrt-v4's own default and this project's own prior hardcoded choice) selects between this project's 3 already-existing sampler classes (`bvh_light_sampler`, `power_light_list`, a uniform-weight `hittable_list`) at `cpu_render_main()`'s scene-construction step (`cpu_renderer/cpu_interface.cpp`) — the two Cornell-box demo scenes (`A1`/`B2`) that previously hardcoded a hand-tuned BVH light list now also have a matching hand-tuned power-weighted variant (`build_cornell_box_power_lights()`) selected for `"power"`, and fall through to the general `lights_raw`-based construction for `"uniform"`. Same "CPU default path tracer only" scope cut as `--sampler`/`--spectral` — BDPT/MLT/SPPM/the debug integrators each already have their own separate light-sampling code (unrelated to this flag, matching pbrt-v4's own scope: `lightsampler` is a `PathIntegrator`/`VolPathIntegrator` parameter upstream too, not read by BDPT/SPPM there either) and print the same "has no effect under ..." warning `--sampler` does when combined. GPU (both backends) is permanently power-sampler-only (`FEATURE_INVENTORY.md`) with no `--lightsampler` equivalent — `--gpu` is one of the flags in the same "has no effect under ..." warning mentioned above (`launcher/main.cpp`'s `use_gpu` check), same as `--sampler`'s own identical warning already covers `--gpu`. |
| `"bool regularize"` | Full | Full | Full | pbrt-v4 defaults this `false` and, when `true`, widens a rough BSDF's GGX alpha (`RegularizeAlpha()`, `src/shared/microfacet.h`) after the first non-specular bounce on a path, reducing caustic fireflies at the cost of some bias — applied unconditionally from the scene's own declaration (same shape as `PixelFilter`, not `maxdepth`'s CLI-overridable shape), since it's a genuine scene-authored rendering-behavior toggle, not a perf knob. CPU gates all 3 real `scatter()` call sites in `camera.h`. GPU-wavefront threads it as an explicit parameter through both material-evaluation kernels (`evaluate_materials`, `evaluate_materials_dielectric`) and their host-side launch chains, since this backend has no accessible global `params` the way the recursive backend does. GPU-recursive reads `params.camera.regularize` directly (real global `__constant__` access, unlike wavefront) at its 4 rough-material call sites in `shade_material()` (`optix_device_helpers.h`: conductor, roughdielectric, coatedconductor, coateddiffuse), gated by a new payload register (p23, `numPayloadValues` 23→24) carrying `anyNonSpecularBounces`-so-far from `optix_raygen.h`'s bounce loop into each closest-hit program — the same "closest-hit reads an INPUT register, never writes it" convention p12 already established for `__miss__ms`'s `prev_brdf_pdf`, just mirrored to a different consumer. `anyNonSpecularBounces` itself is tracked from a real `is_specular` boolean carried in a spare bit (bit 3) of the existing `p10` scatter-flag register (`pack_scatter_flag()`), not inferred from `scatter_brdf_pdf > 0` — an earlier version of this gate used that pdf-based proxy, but a code-review round found it could misclassify a legitimately non-specular bounce whose pdf underflows to exact `0.0f` at extreme grazing angles, so it was replaced with the same real-boolean approach CPU/GPU-wavefront already used. BDPT/MLT/SPPM (all CPU-only) never apply regularization on any backend — a separate, pre-existing characteristic unrelated to this flag, already correctly matching pbrt-v4's off-by-default semantics with no code path to gate. **Behavior change**: before this flag was parsed at all, CPU and GPU-wavefront applied this widening *unconditionally* on every non-specular-then-glossy bounce, for every scene — there was no way to turn it off. Now that the default correctly matches pbrt-v4's real `false`, every bundled scene that doesn't explicitly declare `"bool regularize" [true]` renders *without* the firefly reduction it used to always get; only `pbrt_scenes/sportscar/sportscar-area-lights.pbrt` opts back in. A multi-bounce glossy/rough caustic scene may look visibly noisier than before this change — this is the intended pbrt-v4-conformance fix, not a regression. |

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
  skylight cutout) are supported on **CPU and both GPU backends**
  (recursive + wavefront). A full (unclipped) sphere is rotation-invariant,
  so `pbrt_flatten.h` keeps baking it straight to a world-space
  center+radius on every backend; a clipped one is orientation-dependent, so
  all three backends instead carry the real object-to-world transform and
  intersect in object space against pbrt-v4's own dual-root z/phi
  clip-rejection algorithm (`SphereShape<T>::intersect()`,
  `src/shared/shapes.h`, wrapped by `sphere_clipped_hittable.h` on CPU;
  hand-duplicated free device functions in `optix_intersection_sphere.h`/
  `wavefront_intersection_sphere.h` on GPU, following the same
  `SphereData::o2w`/`w2o` pattern `DiskData`/`CylinderData` already use) -
  exactly `disk_hittable`/`cylinder_hittable`'s own technique, including
  their identical "exact under rotation, approximate NEE weighting"
  character. GPU intersection is a custom software program (not OptiX's
  hardware sphere primitive) and was already capable of this; the earlier
  belief that GPU spheres couldn't clip was a stale assumption, not a real
  hardware limit. Solid-angle NEE sampling of a clipped sphere used as an
  `AreaLightSource` (`random()`/`pdf_value()` on CPU, forwarded to
  `SphereShape<T>::sample_from()`/`pdf_from()`; `sample_sphere_light()`/
  `wf_sample_sphere_light()` on GPU, using a baked full-sphere center+radius
  populated for exactly this purpose) samples over the FULL sphere's
  subtended cone on every backend, not just the visible cap - a pre-existing
  property of the shared CPU template, deliberately mirrored on GPU rather
  than fixed - so this combination gets extra sampling noise (some proposed
  light directions land on the clipped-away part and contribute nothing)
  rather than bias; a narrow, rare combination in practice. A clipped sphere
  combined with a `MediumInterface` is unsupported on **both** backends now
  (an open shell can't bound a participating medium correctly; previously
  GPU alone got away with it by always rendering every sphere as a full
  closed shape regardless of clipping) - loudly warned, not silently
  dropped. Two further, narrower scope cuts: an **instanced** clipped sphere
  (`ObjectInstance`) still renders as its full, unclipped shape on GPU only
  (the instanced-sphere build loop in `pbrt_gpu_builder.h` wasn't extended -
  CPU has no such gap); and **GPU SPPM** (`sppm_programs.cu`, a third,
  independently-duplicated intersection/closest-hit program neither GPU
  backend above shares) also still renders every sphere as its full,
  unclipped shape, matching this file's own established pattern of scoping
  SPPM out of a GPU feature round when its separate architecture would need
  its own dedicated pass (the `Film "cropwindow"` entry above used to be
  the same kind of deferred-SPPM example, until a later round gave GPU
  SPPM its own dedicated fix) - both are real, deliberately deferred
  follow-ups, not oversights.

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

- `MakeNamedMedium`'s own `"rgb Le"`/`"float Lescale"` (pbrt-v4) — a real
  self-emitting medium (fire/plasma/glowing fog) — is decoded for
  `"homogeneous"` media (`Medium::Le` in `pbrt_flatten.h`) **and now for
  `"rgbgrid"` too, on CPU**: pbrt-v4's own per-voxel `"Le"` array (same
  shape/convention as `"rgb sigma_a"`/`"rgb sigma_s"` — one RGB triple per
  voxel, de-interleaved into `Medium::Le_r`/`Le_g`/`Le_b`) plus a scalar
  `"Lescale"`, both threaded unbaked into `RGBGridMediumData<T>`'s
  pre-existing `Le_grids`/`Le_scale`/`sample_point()` machinery (built for
  exactly this, previously unused — `rgb_grid_medium_hittable.h` used to
  compute and discard the per-point `le[]` value at every scatter event).
  Weighted by sigma_a/sigma_t at the actual scatter point, matching
  homogeneous `Le`'s own collision-probability convention exactly.
  **`"cloud"`/`"uniformgrid"` still drop a nonzero `"Le"` with a warning on
  both CPU and GPU** — pbrt-v4 gives them no equivalent `"Le"` parameter at
  all (only `rgbgrid`'s does), so this isn't a scope gap, just a
  non-feature for those two types. **`rgbgrid`'s own new `"Le"` support is
  CPU-only this round** — GPU's `MaterialType::RgbGridMedium` has no
  emission concept yet (a real, deferred follow-up, same shape as
  homogeneous `Le`'s own earlier CPU-then-GPU staging).
  For `"homogeneous"` media specifically, `"blackbody Le"` (real
  Kelvin-to-RGB, the same conversion already used for every light's own
  `"L"`/`"I"` — see `resolveEmissionColor()`) is supported too, not just a
  literal `"rgb Le"` triple, and `Lescale` is baked into `Le` at flatten
  time (`rgbgrid`'s own per-voxel `"Le"`/`"Lescale"` stay unbaked instead —
  see above — since `RGBGridMediumData::sample_point()` is already the
  real consumer that applies the scale). The homogeneous emission
  contributes via `hg_phase_material::emitted()` (`constant_medium.h`),
  weighted by `sigma_a / sigma_t` — the collision-probability weighting
  pbrt-v4's own
  real volumetric estimator uses, collapsed into this codebase's existing
  "every collision continues, weighted by albedo" simplification for
  scattering.
  Default CPU path tracer only: `camera.h`'s generic `hit_record::mat->
  emitted()` dispatch (the same mechanism surface-area lights use) picks
  this up correctly with no integrator-specific wiring. **BDPT/MLT do
  not** — a review pass found that treating a medium-scatter vertex as a
  real emissive Surface vertex (the same machinery an actual area light
  uses) produces two real bugs: BDPT's own front-face `Le()` gate zeroes
  the contribution for half of all exit directions, since
  `constant_medium::hit()` has no real geometric normal to give it
  (`rec.normal` is an arbitrary placeholder); and BDPT's MIS weight
  computation misapplies its delta-distribution `remap0()` fallback to a
  legitimately-zero (not delta) light-origin pdf, since the medium is never
  a registered light — inflating the MIS denominator and dimming the s=0
  strategy's contribution. Rather than render a biased/dimmed glow, BDPT
  (and MLT, which reuses BDPT's own connection machinery) deliberately
  **suppress** medium emission this round — `material::is_medium_scatter()`
  (`material_base.h`), overridden by `hg_phase_material`, lets
  `bdpt_adapter.h`'s `Intersect()` exclude it before it ever reaches BDPT's
  vertex classification, restoring BDPT's exact pre-this-feature behavior
  for media. A real light-connectable vertex representation for medium
  emission is deferred to a future round.
  **SPPM** partially supports it: the camera pass reads emission
  unconditionally at every hit (same generic dispatch as the default path
  tracer), so DIRECT visibility of a glowing medium renders correctly; but
  the photon pass seeds photons exclusively from the registered light list
  (`SampleLightLe()`), which a `constant_medium` is never added to — so
  INDIRECT/bounce illumination from the medium's own glow is silently
  absent under `--sppm` (nearby surfaces receive no bounce light from it).
  Not fixed this round; a real fix needs media to seed real photons, a
  materially bigger feature.
  GPU (both backends) now implements it too, for `MaterialType::Medium`
  (homogeneous) on both the sphere and cylinder shapes — `MaterialData::
  medium_emission` (`optix_types.h`) carries the same sigma_a/sigma_t-
  weighted value CPU's `constant_medium` constructor computes, baked in at
  build time (`pbrt_gpu_builder.h`'s `mediumMaterialIndex()`). Reading it
  is gated on a genuine sampled collision, not just a ray-medium
  intersection *test* — each backend's own Medium closest-hit case only
  adds it inside the real-scatter branch (the same branch the volume-
  scattering NEE/MIS fix above added `medium_phase_nee_mis()` to), never
  the straight-through no-interaction sub-case, matching CPU's own
  `constant_medium::hit()` (a homogeneous medium's "hit" is by construction
  a real collision, never a null one). Added unconditionally, with no MIS
  weighting — this medium is never a member of either backend's light list,
  so its own `Le` can never be NEE-sampled, matching CPU's unconditional
  `hg_phase_material::emitted()` call and pbrt-v4's own volumetric-emission
  convention. `CloudMedium`/`RgbGridMedium`/`GridMedium` remain unsupported
  on GPU too, matching CPU's identical `"homogeneous"`-only scope —
  `scene_builder.cpp` still warns once at scene-load time if one of those
  three declares a nonzero `"Le"`. Also out of scope on GPU: `DielectricMedium`
  (a fused dielectric-surface-plus-interior-medium material) — the pbrt
  loader never actually builds this `MaterialType` for a `.pbrt` scene at
  all (see `pbrt_gpu_builder.h`'s own comment on why; it's reachable only
  from this codebase's hand-built native scenes, which don't parse `"Le"`),
  so there is no live gap to close there.

- A phase-function scatter event inside a participating medium (any of
  `MaterialType::Medium`/`CloudMedium`/`RgbGridMedium`/`GridMedium`, on
  both GPU backends, both the sphere and cylinder shapes) now does real
  next-event-estimation with MIS, matching CPU's `hg_phase_material`
  (`skip_pdf=false`, routed through `hg_phase_pdf`) exactly — previously
  every one of these GPU scatter events was treated as a specular bounce
  (`is_specular=true`), meaning the only way a scattered ray picked up
  light at all was a lucky Henyey-Greenstein-sampled random walk eventually
  escaping the medium and hitting a light before the path's depth budget
  ran out. Still technically unbiased in the limit, but nowhere near
  converged at any real sample count for a dense or room-filling medium —
  a real, previously undocumented CPU/GPU quality divergence for every
  fog/smoke/cloud/nebula scene rendered on GPU (this was never called out
  in this file; only visible from the `is_specular=true` comments in the
  source itself). Fixed via one shared device function,
  `medium_phase_nee_mis()` (`optix_device_helpers.h`, recursive backend;
  the same NEE/shadow-ray/MIS-weight logic deferred to
  `wf_finish_material_scatter()`'s existing `isPhase` path on the
  wavefront backend, which already had this machinery for
  `MaterialType::DielectricMedium`'s own interior scatter sub-case — the
  fix here was extending that existing gate to the other 4 medium types,
  not building new infrastructure). The ray-passes-straight-through
  ("no interaction this event") sub-case every one of these medium types
  also has stays `is_specular=true` — a free crossing, not a real
  scattering event, matching pbrt-v4's own `SampleLd` semantics. Fixing
  this also surfaced and fixed a real, separate pre-existing bug on the
  wavefront backend only: `wf_sample_henyey_greenstein()`'s `wo` parameter
  (the outgoing direction, i.e. `-`ray direction) was being passed the
  un-negated forward travel direction at every one of its call sites
  (including the pre-existing `DielectricMedium` one), inverting the
  `g>0`/`g<0` forward/back-scatter bias for any anisotropic medium on that
  backend — the recursive backend's own identical call sites already had
  this fixed from an earlier round. NanoVDB heterogeneous media
  (`"nanovdb"`) remain out of scope regardless (still fall back to
  homogeneous, see the `MakeNamedMedium` entry above) — this fix applies
  to every medium type this loader can actually build on GPU today.
  **GPU SPPM** (`sppm_programs.cu`, a third, independently-duplicated
  render loop neither of the two backends above shares — see this same
  file's earlier note on GPU SPPM's own separate architecture) is **not**
  covered by this fix and never was: its material dispatch has no case for
  any of these 5 medium types at all, so a medium's trigger sphere renders
  as an ordinary opaque surface under `--sppm --gpu` rather than
  participating-medium transport of any kind — not "no NEE", genuinely no
  medium scattering. Pre-existing, unaffected either way by this round,
  matching this file's own established pattern of scoping SPPM out of a
  GPU medium/shape feature round rather than silently implying parity.
  A follow-up review pass also caught and fixed two bugs this same round
  introduced: `MaterialType::GridMedium` was missing from both backends'
  shadow-ray non-occluding list (`optix_anyhit_shadow.h`/`wavefront_
  anyhit_shadow.h`), the exact class of bug that already once made
  `DielectricMedium`'s own NEE measure as a no-op — every GridMedium NEE
  shadow ray was dying at the medium's own trigger-sphere boundary before
  reaching a light; and the wavefront backend's punctual-light (point/
  spot/distant) NEE loop never checked the extended `isPhase` gate at all,
  so a punctual light at a phase-scatter event was weighted by a bogus
  `dot(wi, normal)` cull and a flat Lambertian `1/π` BSDF value instead of
  the real HG phase value and the medium's own albedo — both now fixed.

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

- A `Texture "imagemap"`'s own `"string encoding"` / `"string wrap"` /
  `"bool invert"` params (previously never parsed at all — every 8-bit
  texture was silently gamma-2.2-decoded via `stbi_loadf()`'s own process-
  global default, `wrap` was scaffolded in `MipWrapMode` but structurally
  inert, and `invert` didn't exist) are now read and honored on **CPU and
  both GPU backends**, for the primary `"reflectance"` slot
  (`Material::textureFilename`) of `Diffuse`/`CoatedDiffuse`/
  `DiffuseTransmission` — the other four texture-filename slots
  (`transmittanceTextureFilename`/`roughnessTextureFilename`/
  `alphaTextureFilename`/`displacementTextureFilename`) still get this
  codebase's long-standing gamma-2.2/Clamp/no-invert defaults regardless of
  what the scene's own `imagemap` declares, a deliberate scope cut matching
  this loader's own "close the reflectance slot first" precedent (see
  `Material::textureGamma`/`textureWrap`/`textureInvert`'s own comments,
  `pbrt_flatten.h`) — `Material::textureFilename` is never populated for any
  other material kind on either backend (verified directly: both
  `pbrt_cpu_builder.h` and `gpu/optix/pbrt_gpu_builder.h` each have exactly
  3 read sites for it, all 3 of these kinds), so this is the entire gap, not
  a narrowed one. `"encoding"`: `"linear"` → gamma 1.0 (no decode — the
  real use case is a roughness/normal/displacement map bound as reflectance
  on a stand-in material for inspection, or a genuinely-linear photo source);
  `"gamma <value>"` → that exact exponent; `"sRGB"`/absent → 2.2 (this
  codebase's pre-existing default, an approximation of pbrt-v4's real
  piecewise sRGB EOTF, not a change). `"wrap"`: `"repeat"` (pbrt-v4's own
  real default, and now this loader's resolved default too — see below),
  `"clamp"`, or `"black"`. `"invert"`: per-texel `1-c` at load time.
  **Behavior changes**, both deliberate and both gated on the same user
  decision to fix `wrap` for real rather than ship an inert flag a second
  time: (1) UV tiling now actually works on both CPU and GPU —
  `mipmap_texture::value()`/`value_ewa()`/`value_lod()` (`texture.h`) used
  to hard-clamp `u`/`v` to `[0,1]` *before* `MipWrapMode` ever saw them, and
  GPU's own `sampleImage()` (`gpu/optix/optix_device_helpers.h`,
  `gpu/optix/wavefront_kernels.cu`) did the identical hard `[0,1]` clamp, so
  `"wrap" "repeat"` on a UV>1 scene was a structural no-op on both; the
  clamp is now a much wider `wide_clamp([-1024,1024])` on CPU (still
  bounding `bilerp()`'s integer texel math against a pathological UV, just
  no longer defeating real tiling) and the matching `[-1024,1024]` wide
  clamp before wrapping the integer pixel index on GPU (both backends,
  `GpuWrapMode`/`TextureData::wrapMode`). (2) The loader's own resolved
  `wrap` default is now `"repeat"` (pbrt-v4's real one) rather than the
  prior de-facto `Clamp` every image texture got by simply never reaching a
  non-default wrap mode — a scene that authors UV coordinates outside
  `[0,1]` on an image-textured surface with no explicit `"wrap"` param will
  now visibly tile where it previously clamped to the edge pixel, on every
  backend. `MipMapOptions::wrap`'s own C++ struct default (`mipmap.h`)
  deliberately stays `Clamp` — only the pbrt loader path resolves to
  `"repeat"` explicitly, so this doesn't affect this codebase's own native
  (non-pbrt) scenes; `TextureData::wrapMode`'s own C++ struct default
  (`optix_types.h`) deliberately stays `Clamp` for the identical reason —
  every OTHER GPU texture-table entry (checker/mix tex1/tex2/amount,
  roughness/transmittance/displacement) is still built without threading
  these options through at all, so those stay byte-for-byte at GPU's
  original hard-clamp/gamma-2.2/no-invert behavior, unaffected by this
  change. Gamma and invert are baked into the decoded pixel bytes once at
  scene-load time on GPU (`getOrBuildPbrtImageTexture()`,
  `gpu/optix/pbrt_gpu_builder.h`, via `stbi_ldr_to_hdr_gamma()` - the same
  process-global stb_image mechanism `rtw_stb_image.h`'s `rtw_image`
  constructor uses on CPU) rather than at sample time, matching CPU's own
  gamma/invert-at-load vs. wrap-at-sample split exactly; a file shared by
  more than one material with different `encoding`/`wrap`/`invert` requests
  gets a separate, independently-decoded texture-table entry per distinct
  option combination (a small cache-key extension, not a new limitation) so
  the two don't collide. A `"string encoding"` value this loader doesn't
  recognize (not `"linear"`/`"sRGB"`/absent/`"gamma <value>"` — e.g. a real
  pbrt-v4 ICC-profile filename) falls back to gamma 2.2 WITH a warning
  (`resolveTextureGamma()`, `pbrt_flatten.h`), not silently — same for a
  `"gamma <value>"` that parses to something unusable as a decode exponent
  (non-finite, e.g. `"gamma nan"`/`"gamma inf"` — `std::stod` accepts these
  without throwing per `strtod` semantics — or `<= 0`). An unrecognized
  `"string wrap"` value (a typo or wrong case, e.g. `"Clamp"`) similarly
  falls back to `"repeat"` WITH a warning rather than silently. Binding
  `"encoding"`/`"wrap"`/`"invert"` to one of the 4 non-primary texture-
  filename slots (transmittance/roughness/alpha/displacement) also warns
  that the request is ignored there — those slots still only ever get this
  codebase's original gamma-2.2/Clamp/no-invert defaults on either backend,
  regardless of what the scene's own imagemap declares.
