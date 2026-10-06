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
| rough `conductor` / `roughness`-only metal vs pbrt-v4 | Full | Full | Checked against an independent pbrt-v4 path-level reference (the `ConductorBxDF` with the repo's Cu constants, BSDF sampling only), rough copper sphere under a small lamp, sphere-pixel energy at roughness 0.25 / 1.0. (1) Both GPU backends skipped a vertex's NEE whenever the BSDF *continuation* sample was rejected (a reflection landing below the horizon, common at high roughness) - the kernel returned before the NEE step, where pbrt takes the direct-light sample at every vertex regardless: GPU 86% / 75% of the CPU. A rejected sample now carries zero weight and the path ends after its NEE (conductor and rough metal, both backends; the same fix as for rough glass). (2) The CPU took the conductor's Fresnel term once per hit at the view-normal cosine, where pbrt and the GPU evaluate it at the half-vector `|wo.wm|` of the direction being evaluated, so grazing views were too bright (CPU 104% / 108% of the reference). `conductor::scattering_attenuation()` now returns the per-direction colour, and `camera.h` uses it for NEE and for the BSDF-sampled continuation alike (a new `scattering_attenuation(r_in, rec, scattered, srec_attenuation)` overload; every other material keeps the old 3-argument form). Now CPU 99.8 / 101.8%, recursive 100.0 / 99.8% and wavefront 100.2 / 99.9% of that reference/CPU. `rough-metal-lamp.pbrt` guards both. |
| `conductor` | Approx | Approx | A named metal spectrum (`"spectrum eta"`/`"spectrum k"` = `"metal-<Name>-eta"`/`"-k"`, e.g. Ag/Au/Al/Cu/Fe) OR an explicit `"rgb eta"`/`"rgb k"` resolves to the real complex-IOR GGX `conductor`/`MaterialType::Conductor` model on both — this codebase's own conductor BxDF is already a plain 3-float-RGB model (matching its own named-spectrum table's shape), so an explicit RGB pair needs no spectral upsampling the way pbrt-v4's own real implementation does. Only an unrecognized spectrum name (or giving just one of `eta`/`k`, not both) still falls back to the fuzz-sphere metal model (roughness fed directly as fuzz, not GGX alpha) on both — symmetric either way. Separate `"uroughness"`/`"vroughness"` (anisotropic GGX) are parsed independently and shade as real anisotropy on both CPU and GPU, on all 3 GPU render paths (recursive, wavefront, SPPM's Conductor/RoughDielectric coverage). A material that authors only one axis (e.g. `"uroughness"` alone, leaving `"vroughness"` to its pbrt-v4 default of 0 - near-mirror in v) is a real, distinct axis value, not collapsed to isotropic. GPU's local tangent frame for this material is now UV/`dpdu`-aligned on the recursive and wavefront backends, matching CPU's `ShadingFrame::from_dpdu` (via `BuildDpduTangentFrame`, `src/shared/microfacet.h` - real per-shape `dpdu`, analytic for sphere/disk/cylinder, UV-gradient-solved for triangle, the shape's own edge/patch tangent for quad/bilinear-patch), so anisotropic highlight orientation now matches CPU almost everywhere. GPU SPPM was deliberately left on the older arbitrary (not `dpdu`-aligned, but still continuous/branchless - `BuildArbitraryTangentFrame`, Duff et al. JCGT 2017) frame - it would need `dpdu` threaded through its own separate camera/photon-pass intersection code and per-pixel persisted state, a materially bigger lift for a backend that already only implements 2 of these 4 material kinds - so SPPM's anisotropic conductor/rough-dielectric highlights still won't generally match CPU's orientation. |
| `dielectric` | Full | Full | Smooth by default; a nonzero `roughness` routes to the real `rough_dielectric`/`MaterialType::RoughDielectric` GGX model on both. Separate `"uroughness"`/`"vroughness"` are parsed independently and shade as real anisotropic GGX on both — see `conductor`'s own note above on GPU's dpdu-aligned tangent frame (recursive/wavefront) and SPPM's remaining orientation caveat, both of which apply here too. `"roughness"` bound to a bare `"imagemap"` Texture also shades as real per-hit texture-sampled roughness on CPU and GPU recursive/wavefront (isotropic only), including correct delta-vs-glossy classification under `--sppm`/`--bdpt`/`--mlt` on CPU. GPU SPPM specifically rejects a texture-bound roughness at scene-load time (its payload carries no UV data to sample the texture with) rather than silently ignoring it — use the default path tracer or a GPU backend other than `--sppm` if the texture matters. Dispersion (a Cauchy-formula `"eta"`/dispersive material, both smooth and rough) is real on CPU (`--spectral`, true continuous hero-wavelength integration) and both GPU backends: GPU-wavefront matches CPU's continuous `SampledWavelengths` approach; GPU-recursive uses a coarser 3-fixed-representative-wavelength (R/G/B) stochastic scheme instead, chosen once per path. GPU SPPM rejects any dispersive `dielectric`/`rough_dielectric` at scene-load time (no wavelength concept in its own payload) rather than silently flattening it. A named glass spectrum (`"spectrum eta"` = one of pbrt-v4's own 7 real presets — `"glass-BK7"`/`"glass-BAF10"`/`"glass-FK51A"`/`"glass-LASF9"`/`"glass-F5"`/`"glass-F10"`/`"glass-F11"`, see `src/shared/glass_data.h`) resolves to that glass's real refractive index at the 587.6nm d-line on both — a flat IOR, not real per-wavelength dispersion (which needs the separate Cauchy-formula path above); an unrecognized name, an inline numeric `"spectrum eta"` (piecewise-linear wavelength/value data), or a `"texture eta"` binding all warn and fall back rather than misreading the input, symmetric with `conductor`'s own named-spectrum resolution above. |
| rough `dielectric` BSDF (shared `RoughDielectricBxDF`) | Full | Full | Checked against an independent pbrt-v4 path-level reference (`scripts/pbrt_rough_glass_reference.py`: pbrt's `DielectricBxDF` written out from the pbrt source, plain BSDF sampling, calibrated to 0.3% on smooth glass; its sampler agrees with pbrt's own `f()` by quadrature to 0.1%). Three defects, all in the code every backend shares: (1) the transmission `f()` was pbrt's radiance-mode BTDF while this renderer's path throughput does not carry the matching 1/eta'^2 factors (they cancel over a full traversal of a closed glass object), so NEE at every vertex where a ray leaves glass was counted eta^2 = 2.25x too strongly - the BSDF-sampling weights (`G2/G1`, the ratio `f*cos/pdf`) were right and hid it from the white-furnace test; (2) the transmission `pdf()` was eta^2 times the density the sampler draws from, so MIS weighed against the wrong magnitude; (3) `f()`/`pdf()` kept back-facing microfacets that pbrt discards, and the samplers (the shared one and the GPU's inline copies) kept a refracted direction that lands on the same side as the ray, which pbrt rejects (+7% leaving glass at 75 degrees). (4) both GPU backends skipped a vertex's NEE whenever the BSDF *continuation* sample was rejected (a reflection below the horizon, a refraction on the wrong side), because the kernel returned before the NEE step; pbrt and the CPU take the direct-light sample at every vertex regardless, so the GPU lost exactly the rejection probability, which is large leaving glass (a camera inside a rough glass sphere read 1/(1+alpha^2) of the CPU: 80% at roughness 0.25, 63% at 0.6, 50% at 1.0; now 100.0-100.3% at every roughness) - a rejected sample now carries zero weight and the path ends after its NEE. `f()`/`pdf()` now use the importance-transport form that matches the throughput, and the tests check them in absolute terms (`BxDFWhiteFurnace.RoughDielectricPdfIsTheSamplingDensityAndFIntegratesToTheAlbedo`). Rough glass under a small lamp, mean image vs the reference at roughness 0.05 / 0.25 / 0.6: CPU +0.3 / -0.1 / 0.0%, recursive +0.3 / -0.8 / -1.3%, wavefront 0.0 / -1.0 / -1.3% (CPU was +10 / +22 / +28%, GPU +10 / +19 / +21%); under a uniform sky all three within 0.3% (were +2.4 to +3.4%). E12 went from 94 / 91% to 100.5 / 97.7% of CPU (recursive / wavefront), the rest being the spectral uplift of its chromatic fog. Measured in linear float over the glass pixels only (rows below the lamp), because the directly visible lamp holds ~94% of this scene's image energy and hides glass errors from whole-image means (an 8-bit transform also biases dim pixels low): glass-region energy vs the reference at roughness 0.05 / 0.25 / 0.6 is CPU 99.0 / 99.0 / 100.4%, recursive 100.5 / 99.2 / 99.1%, wavefront 99.6 / 98.5 / 99.3% (GPU was 97 / 88 / 80% and 96 / 86 / 79% before defect 4). |
| `dielectric` / `thindielectric` as a shadow occluder | Full | Full | Opaque to NEE shadow rays on every backend, like every pbrt-v4 surface that has a material (`VolPathIntegrator::SampleLd`, `integrators.cpp:1335`); glass is lit through its specular BSDF path (dark shadow plus a caustic under a glass sphere, as pbrt renders it). Previously NEE walked straight through glass, which counted a light seen through it twice and ignored refraction (Cornell glass-sphere scenes were 4-17% too bright). This includes a glass shape that carries a `MediumInterface`: NEE from a scatter vertex inside it is blocked by its own shell, so the fog is lit only along specular chains (a phase-sampled ray refracting back out), exactly as pbrt renders it. It used to stay transparent, which counted every escape from the fog twice (a glass-bounded fog under a sky read +23% at optical radius 0.3, +51% at 3). Now a sky furnace through a glass shell is exact on all three backends (`pbrt_scenes/glass-fog-furnace.pbrt`), and glass-and-fog showcase scenes render 11-33% darker than before (E3 -15%, E11 -29%, E12 -33%, B13 -13%, A9 -11%, all three backends together). A medium boundary that should not block writes `Material "interface"`, which the bundled scenes do. `DiffuseTransmission`, `Subsurface` and `interface` still pass shadow rays. |
| area-light shapes as a shadow occluder | Full | Full | A shadow ray stops 0.002 short of its sampled point and every OTHER surface blocks it, emitters included, as in pbrt-v4 (`SpawnRayTo`, `ray.h:103-107`; the integrator treats any hit as occlusion). The GPU used to ignore every emitter in its shadow any-hits, so a multi-faced lamp (a bulb, a box light, an octahedron) did not shadow its own far side and rendered ~48% too bright (`pbrt_scenes/emissive-octahedron-furnace.pbrt`, checked against the closed-form answer). The shadow ray is also measured from its shifted origin and, on the wavefront backend, re-aimed at the sampled point (a light flush against a ceiling rendered at 22-37% of the CPU before; `pbrt_scenes/flush-ceiling-light.pbrt`). Sphere lights too: they used to be the one exception (a sphere's own light was assumed to be in the way), so an emissive ball hanging in front of a lamp cast no shadow on either GPU backend (116.9% of the CPU, `pbrt_scenes/emissive-sphere-occluder.pbrt`). Sphere lights are cone-sampled, so the target is on the near side and the 0.002 stop-short keeps the ray from reaching it; only a different emitter, or the light's own far side, can block. A scene lit only by a sphere lamp still agrees to 0.1%. |
| BSDF-sampled ray that hits an area light (MIS) | Full | Full | The bounce ray's emission is weighted `pb^2/(pb^2+pl^2)` against the NEE sample taken at the vertex that sampled it, on every backend. The wavefront backend used to add emission only for the camera ray and after specular bounces while still weighting its NEE sample, losing the BSDF strategy's share of a lamp: ~0 for a small or distant light, but 17-19% of Fireplace Room's and a bright closed room's light with a big lamp near the walls (`pbrt_scenes/large-area-light-room.pbrt`). The pdf comes from the light BVH / alias table at the sampling vertex and `wf_reevaluate_light_geometry`, so it covers every light shape; an emitter outside the sampled-light list keeps full weight. |
| homogeneous `MakeNamedMedium` with absorption (`sigma_a > 0`) | Full | Full | A collision scatters with probability `sigma_s/sigma_t` and is absorbed otherwise, on every backend (`collapse_homogeneous_medium` on the CPU; `pbrt_gpu_builder.h` now applies the same factor). The GPU used to scatter every collision at full strength, so an absorbing fog rendered ~14-25% too bright and a pure absorber lit the scene from nowhere (2.2x); `sigma_a = 0` was always right. `pbrt_scenes/absorbing-fog.pbrt`. |
| homogeneous medium with a different extinction per colour channel | Full | Full | `sigma_a`/`sigma_s` that are thicker in one channel than another (`dielectric-medium-showcase`, E11/E12, `cornell-smoke`, `final-scene`, jade/wax slabs) used to be flattened to ONE extinction (the luminance of `sigma_t`) with the colour only in the albedo, on every backend, so a sphere of `sigma_a = (0.1, 0.4, 0.9)` rendered grey 0.47 where exp(-sigma * 2) is (0.82, 0.45, 0.17). Every backend now draws the free flight with `sample_homogeneous_event` (`src/shared/volume_scattering.h`): one colour channel (on the wavefront backend one of the four hero wavelengths) picks the distance and the balance heuristic over the channels weights the result, which is unbiased per channel, keeps the weights at most 3 and reduces to the old model exactly for a grey medium. Shadow rays carry the per-channel transmittance. CPU and recursive read the closed-form absorber to 0.1% and a chromatic scattering furnace to 0.3%; the spectral wavefront backend lands 5-8% low on the strongly chromatic absorber (the uplift of a chromatic `sigma(lambda)` is not the per-channel exp, as in pbrt). `pbrt_scenes/chromatic-absorber.pbrt`, `chromatic-fog-furnace.pbrt`. The camera medium (MediumInterface before Camera) is sampled per channel the same way on all three backends (`pbrt_scenes/chromatic-camera-medium-absorber.pbrt` is a closed-form check, `chromatic-camera-medium.pbrt` a lit room compared per channel). Live Preview needs nothing extra (its volumetric ReSTIR only resamples light candidates; checked per channel, see the Live Preview note below). BDPT, MLT and SPPM (CPU) still use the scalar model - the chromatic absorber reads grey 0.47 there - and print a warning naming the scene (`chromatic_media_integrator_warning`); a per-channel version has to carry a pass-through weight through `constant_medium::hit()`, which a free flight cannot, and re-derive the strategy MIS weights from a per-channel mixture pdf, so it is not a port of the path tracer's estimator. GPU SPPM rejects media scenes outright. The Metal backend also still uses the scalar model. (The heterogeneous `rgbgrid` is the next row.). |
| heterogeneous `rgbgrid` medium, per colour channel with absorption | Full | Full | The grid used a single extinction (the brightest channel of sigma_a + sigma_s) and treated every real collision as a scatter with albedo sigma_s / max(sigma_s): it never absorbed (a grid giving only `sigma_s` has pbrt's default `sigma_a = 1`, ~97% absorption, yet scattered at full strength), and a channel with a smaller sigma_t was made to absorb instead of passing - a non-absorbing chromatic grid lost 40% of its red and 23% of its green in a furnace, on the CPU and both GPU backends. Both GPU backends also dropped `sigma_a` altogether (an absorbing-only grid was invisible) and clamped the grid lookup to the edge voxel where pbrt and the CPU read zero beyond the outermost voxel centres. Every backend now does spectral tracking against the shared scalar majorant (`heterogeneous_tracking_step`, `src/shared/volume_scattering.h`): a real event with probability mean(sigma_t)/majorant carrying sigma_s/mean(sigma_t) (emission sigma_a/mean(sigma_t), as the homogeneous media do), a null event weighting each channel by (majorant - sigma_t_c)/(majorant - mean(sigma_t)), at most 3; shadow rays ratio-track per channel; the GPU majorant covers sigma_a + sigma_s; an absent `sigma_a` grid is 1 as in pbrt. On the wavefront backend the per-voxel RGB is spread over the hero wavelengths with a fitted partition of unity (`wf_rgb_wavelength_basis`) because the shadow any-hit has no uplift tables: a grey grid is exactly flat, a chromatic absorber lands within ~4%. Measured: furnace 1.000 in every channel on all three backends; a uniform 2x2x2 absorber 0.5% (CPU, recursive) / 4% (wavefront) from the analytic exp(-sigma * 0.875 * 2); `rgbgrid-medium` 100.0% / 99.9% (it was 106% against the changed CPU). `pbrt_scenes/chromatic-rgbgrid-absorber.pbrt`, `chromatic-rgbgrid-furnace.pbrt`. The per-voxel `Le` is weighted by the event's absorption share on the GPU now too (it used to be the full `Le` at every collision). BDPT/MLT/SPPM keep the old max-channel `hit()`. |
| shadow-ray transmittance through a medium (CPU) | Full | Full | A sampled medium collision during a shadow ray now blocks it (delta tracking), making the expected visibility exactly `T`. The CPU walk used to also multiply by the medium's own transmittance on the rays that did scatter and attenuate none of the others, i.e. `T*(2-T)`: invisible in thin fog, 24% too bright at optical depth 1. All three backends now match an independent Monte Carlo solution of a point-lit fog sphere to 0.4% (`pbrt_scenes/fog-point-light.pbrt`). Applies to homogeneous, cloud, grid and RGB-grid media alike. |
| punctual lights inside a participating medium | Full | Full | Point/spot/distant lights now light a medium on the recursive GPU backend too (`medium_phase_nee_mis`); it used to sample only area lights and the sky there, so a point-lit fog rendered black. |
| `"texture displacement"` with a grayscale height image (bump mapping) | Full | Approx | Perturbs the shading normal at every non-emissive triangle hit. CPU: `bump_map_material`. GPU (both backends): `gpu/optix/gpu_bump_map.h` applies the same `apply_bump_map` in the triangle closest-hit programs, with a bilinear Repeat sRGB-decoded height lookup, the `scale` texture factor, the CPU's unnormalized `dpdu` (double precision) and its finite-difference step: the pixel footprint (`0.5*(|dudx|+|dudy|)`) for a camera-ray hit on a perspective camera, 0.001 otherwise. The GPU used to ignore it, which rendered Sibenik 12-15% dark and a strongly bumped plane 58% bright (`pbrt_scenes/bump-mapped-plane.pbrt`). Approx: triangles only (a sphere/quad/disk with a bump renders flat), no bump inside a `mix`, a specular-bounce hit gets the footprint of the original camera ray where the CPU uses 0.001, and the cameras the CPU gives no differentials (orthographic, spherical, realistic, animated) use 0.001 on both. An RGB normal map still needs a Lambertian material with no reflectance texture. |
| camera medium (all backends): transmittance, albedo, wavefront | Full | Full | Free flight is sampled against the nearest surface (or infinity) and a collision scatters with `sigma_s/sigma_t`; a ray that passes carries weight 1. The CPU and the recursive GPU backend also multiplied a surviving ray by `T` (a surface behind a pure absorber of optical depth 1 rendered at `exp(-2)` instead of `exp(-1)`); the recursive backend never weighted the continuing path by the scattering albedo and its builder dropped `sigma_s/sigma_t` from it (an absorbing fog 22% bright at the first scatter, 2x by the eighth); the wavefront backend had no camera medium at all (it warned and rendered the scene without fog) and now has one. All three backends match a Monte Carlo solution of the fog to ~3% (`pbrt_scenes/camera-medium-absorbing.pbrt`). A recursive-GPU sphere-light NEE sample also reported the distance to the sphere's *centre*, over-attenuating by `exp(-sigma_t * radius)` in a camera medium (E10 was 5-7% dark); it now reports the distance to the sampled point. |
| path throughput ceiling | Full | Full | A path's running throughput is capped at 50 per channel after each scatter (the CPU's `kMaxPathThroughput`; Russian roulette only acts below 1). The GPU backends had no cap, so a BSDF whose sample weight compounds - `HairBxDF`'s averages ~4, and a closed hair shape bounces a ray inside itself - rendered 2.1x too bright (Hair Fibers B11 2.3-2.5x); `pbrt_scenes/hair-sphere-dim-sky.pbrt`. |
| `cylinder` as an area light (CPU light sampling) | Full | Full | The CPU's NEE samples a direction toward the light and credits whatever emitter the ray hits first, so the direction's density (`CylinderShape::pdf_from`) has to count both crossings of the open tube - pbrt's first-hit-only `PDF()` is only right for an integrator that credits the sampled point instead. With first-hit only the floor beside a cylinder light rendered up to 3x too bright and the scene ~8% above both GPU backends (which evaluate emission at the sampled point, and agree with the same light built from quads); now CPU, recursive and wavefront agree to ~0% (`PbrtBackendAgreementTest.DiskCylinderLightAgreesAcrossBackends`). |
| fuzzed metal (`Material "conductor"` given only a `reflectance`; `metal`) | Full | Full | The reflected direction is perturbed by `fuzz` times a random unit vector, and a ray the perturbation sends below the surface is absorbed (the shared `MetalBxDF`). The recursive GPU backend used to reflect it instead, so a rough fuzzed metal lost no energy: a furnace plane at roughness 1 rendered 0.90 against 0.645 on the CPU and wavefront (`pbrt_scenes/fuzzed-metal-furnace.pbrt`), and Salle de Bain was ~3% bright. |
| Russian roulette (recursive GPU) | Full | Full | The roulette draw now comes from the random stream the hit program advanced. It used to repeat the first variate the hit program had already spent sampling the bounce direction, correlating survival with direction: Breakfast Room's fourth bounce order was ~26% too bright (+2.7% overall). |
| medium boundary crossings keep the MIS state (GPU) | Full | Full | A shape with a `MediumInterface` is a pass-through shell on the GPU whatever surface `Material` it names. Its no-scatter crossing now counts as a free medium-boundary crossing (like `Material "interface"`: no bounce spent, the last real vertex's MIS state survives) instead of resetting it as a specular bounce, which counted every escape after a scatter at full weight on top of the NEE sample. The wavefront loop runs `max_depth + 32` iterations (each ray's own depth enforces the budget) so crossings no longer eat bounces. The recursive backend's flag-4 branch had also lost its `ray_origin` update in the 0.001 offset commit, so a recursive crossing never advanced the ray until the crossing cap killed the path. Sky-lit non-absorbing fog (`pbrt_scenes/fog-furnace.pbrt`, empty sky = 1.0): CPU 1.01, recursive 1.00, wavefront 1.01 at optical radius 3 (recursive 1.51, wavefront 1.52 before). |
| emitter MIS pdf origin (recursive GPU) | Full | Full | A BSDF-sampled hit on an emitter weights against the NEE pdf *from the vertex the sample was taken at* (CPU's `prev_surface_p`, wavefront's `scatterOrigin`). The recursive hit programs used the ray origin, which a free medium-boundary crossing moves, so a bounce off a wall that crossed a fog sphere before reaching the light was weighted against the wrong distance (Homogeneous Medium E1 was 154% of the CPU). The last real vertex now rides in payload p25-p27, used by every emitter shape and by the miss program's sky pdf. |
| `uniformgrid` medium shadow rays (wavefront GPU) | Full | Full | The wavefront launch parameters never received the uniform-grid arrays, so the shadow any-hit's bounds check always failed and NEE shadow rays inside a `uniformgrid` medium were never attenuated: a sky-lit furnace read 4x bright and Uniform Grid Medium E8 +10%. |
| `NormalMappedLambertian` direct light (wavefront GPU) | Full | Full | The wavefront NEE helper took its BSDF colour from `attenuation` for `Lambertian` only, so a normal-mapped surface was lit with the white default instead of its albedo: a blue normal-mapped sphere under a lamp rendered grey (Normal-Mapped Cornell B12, 164-188% of the CPU over the sphere). Now `NormalMappedLambertian` uses its albedo like the sampling case does. |
| bump and normal map texel decoding | Full | Full | pbrt decodes a PNG height image (a float imagemap) as sRGB and interpolates the decoded floats, and reads a tangent-space normal map LINEAR (`scene.cpp`: `Image::Read(..., ColorEncoding::Linear)`). The CPU stored the decoded height requantized to a byte - the first ~12 byte values all round to 0 or 1/255, so a smooth dark ramp became stairs and the finite-difference slope was wrong - and both backends sRGB-decoded normal maps, tilting every texel (a flat 128,128,255 map read as (0.22,0.22,1)). The CPU now interpolates float texels and both backends read normal maps linear. On a linear height ramp (where the finite-difference step cannot matter) the GPU read +1.2% / +4.6% of the CPU at scale 5 / 20 and now 99.9% / 99.7%; `bump-mapped-plane` went from 98.4-99.0% to 99.5-99.9%, `normal-mapped-cornell` from 100.9% to 100.4%. The texture's own `"string encoding"` is still ignored for displacement (a non-PNG default would be linear in pbrt). |
| bump mapping on a mesh without UVs (GPU) | Full | Full | A grayscale `displacement` on a triangle mesh with no UVs was skipped on both GPU backends ("no lookup coordinate"), while the CPU looks it up at the barycentric fallback UV with `dpdu` = first edge and `dpdv` = `n x dpdu`. The GPU now does the same (its dpdu helper's degenerate-UV branch is exactly that), so the tall box in B12 shows its marble relief on every backend. B12 is now 100.0 / 100.3% of CPU (recursive / wavefront) at every depth. |
| heterogeneous medium thickness on camera rays (CPU) | Full | Full | `uniformgrid`/`nanovdb`, `rgbgrid` and `cloud` media sampled free flights along the raw ray parameter, but a camera ray's direction is `pixel_sample - origin`, hundreds of units long, so a primary ray saw a medium `|d|` times too thin (CPU 4-10% bright against both GPU backends on E5/E7/E8, and flat in depth). They now work in world distance like `constant_medium`. E5 and E8 now agree across all three backends to 0.2%, E7 to 2% (the GPU ratio-tracks the brightest channel only). |
| `Material "interface"` by name | Full | Full | pbrt-v4's parser rewrites `none` and `""` to `interface`, so `interface` is the canonical spelling; the loader only knew the aliases and rendered `Material "interface"` as a grey diffuse (an `interface` fog sphere read 0.50 of the sky on all three backends). |
| light leaking at edges of a closed room | Approx | Approx | A ray that starts within its offset of an edge or corner can skip the adjacent surface, so a closed room lit only by the sky shows faint bright lines along its edges. Measured on Fireplace Room with the lamps removed (about 0.1-0.2% of the lit scene's brightness): CPU is the reference, the GPU backends leak ~2.3x as much. The remainder is the GPU shadow-ray offset (`shadowRayEpsilon`, 0.01), which exists to avoid self-intersection crashes on dense custom-primitive scenes; 0.001 brings it to ~1.5x but changes ordinary scenes by only ~1% and was left alone. The recursive backend's bounce-ray offset was 0.01 as well and is now 0.001 like the wavefront backend's (it leaked ~4x CPU before). |
| `dielectric` `"rgb tf"` | Full | Full | **Non-standard.** The OBJ/.mtl `Tf` transmission filter (stained glass, tinted windows): multiplies the transmitted contribution only, leaving the reflection clear. White (the default) is a no-op; a rough or dispersive dielectric ignores it. Used by `pbrt_scenes/environment-*.pbrt`, which a one-off generator ported from the native .mtl loader. |
| smooth `dielectric` Fresnel (wavefront, GPU SPPM) | Full | Full | The wavefront and SPPM kernels chose reflect-vs-refract for smooth glass with Schlick's approximation fed the incident cosine, while the CPU and recursive backends (and pbrt) use the exact dielectric Fresnel. Schlick under-reflects for a ray leaving the glass, so light through two refractions and internal bounces was ~2x too bright: a smooth glass sphere under a small lamp read 105.3% of a pbrt-v4 path-level reference on the wavefront backend (CPU 100.4%, recursive 100.9%). All three now use `FrDielectric` and read 100-101%; `pbrt_scenes/glass-sphere-lamp.pbrt` guards it. |
| `thindielectric` | Full | Full | Both use the correct closed-form un-refracted transmission (`R_eff = R + T²R/(1-R²)`), not a solid-glass approximation. Named glass `"spectrum eta"` resolves the same way as plain `dielectric` above. |
| `coateddiffuse` | Full | Full | pbrt's `LayeredBxDF<DielectricBxDF, DiffuseBxDF>` (same port as `coatedconductor` below), same 3 parameters (albedo, ior, roughness), on both; `thickness`/`albedo`/`g`/`maxdepth`/`nsamples` are not read (a warning). `"reflectance"` bound to a real `"imagemap"` `Texture` (optionally `"scale"`-wrapped) is also decoded on both, including both GPU backends' NEE/MIS evaluation — see "Other known gaps" below. Separate `"uroughness"`/`"vroughness"` for the coat are parsed independently and shade as real anisotropic GGX on both — same GPU tangent-frame situation (dpdu-aligned on recursive/wavefront, arbitrary on SPPM) as `conductor` above. Named glass `"spectrum eta"` (for the coat's own IOR) resolves the same way as plain `dielectric` above. |
| `coatedconductor` | Full | Full | pbrt-v4's own `LayeredBxDF<DielectricBxDF, ConductorBxDF>` (`src/shared/bxdfs_layered.h`, ported from `bxdfs.h`): the coat refracts, `Sample_f` is the random walk (an unbiased weight, `pdfIsProportional`), NEE uses the stochastic `f()`, and `PDF()` is the MIS density - same on the CPU and both GPU backends (CPU, recursive and wavefront agree to ~1% on a lamp-lit sphere for smooth/smooth, smooth coat over rough base, rough coat over smooth base and rough/rough, and the smooth case matches the closed form to 0.1%). Parameter names are pbrt's: `interface.roughness`/`uroughness`/`vroughness` and `interface.eta` (float or named glass) for the coat, `conductor.roughness`/`uroughness`/`vroughness` for the base (both default 0 - a mirror under glass), `conductor.eta`/`conductor.k` (named metal or rgb) or `reflectance`, `thickness`; nothing given is pbrt's copper. As in pbrt the conductor's eta and k are divided by the coat's IOR. The older bare `eta`/`k`/`roughness` spellings still load (one roughness for both interfaces). Not supported (a warning, defaults used): `albedo`, `g`, `maxdepth`, `nsamples`. A smooth coat over a smooth conductor is a delta lobe (no NEE); smooth coat over rough base takes NEE with `f()` and treats the coat's mirror reflection as a specular sample, like pbrt. Metal: the same walk is ported to MSL on the branch `feature/metal-layered-bxdf-pbrt-port` (`gpu/metal/metal_poc_layered_bxdf.metal`; one roughness for both interfaces, no separate `conductor.roughness`); not yet built on a Mac - until it is merged, Metal still has the older simplified model. |
| `diffusetransmission` | Full | Full | Separate reflectance/transmittance colors on both. `"reflectance"`/`"transmittance"` bound to a bare `"imagemap"` `Texture` are also decoded on both (no `"scale"`-wrap support — no bundled scene needs it) — `barcelona-pavilion`'s foliage binds both to the same texture, the motivating case. See `pbrt_scenes/barcelona-pavilion` and "Other known gaps" below. |
| `subsurface` | Full | Full | Real tabulated BSSRDF with device probe-walk on both GPU backends (recursive and wavefront), matching CPU's own tabulated BSSRDF. |
| `measured` (real `.bsdf` file) | Full | Full | Both load and flatten the same tensor tables; both fall back to Lambertian on the same "unresolved filename" gate, so they can't disagree about when the fallback applies. A sample's path weight is pbrt's `f * |cos| / pdf` on all three (it was the bare `f` on all three until 2026-10: a measured sphere in a white furnace read 7.8/8.1/3.5 for a blue table, 12 for a metallic one and 0.34 for white paper, instead of 0.1/0.2/0.4, 0.72 and 1.0; `pbrt_scenes/measured-furnace.pbrt` guards it, against an albedo integrated independently from `f()`). `sample_f` also evaluates the spectra at pbrt's luminance-warped point rather than at `u_wm` (it used to differ from `f()` by up to 26% on the synthetic table; `MeasuredBxdfConsistency.SampleFAlbedoMatchesEvalFAlbedo` checks they agree). Like the CPU, both GPU backends treat it as a no-NEE, no-MIS bounce, and query the table at three fixed wavelengths (612/549/465 nm) rather than pbrt's sampled ones. Metal does not implement `measured`. |
| `mix` | Full | Full | Real per-shading-point stochastic two-material blend on all three backends now (`MaterialType::Mix`, `optix_types.h`): each hit deterministically (hashed from the world-space hit point, not a fresh random draw — so a radiance bounce and its shadow ray agree on which sub-material won) resolves to sub-material A or B and shades through that material's own real GPU model — a Mix of e.g. `conductor`+`diffuse` keeps the conductor's real specular highlight on GPU, not an averaged flat color. Falls back to the old flat-Lambertian-averaged-color approximation only for a pathologically deep/cyclic mix-of-mix chain (depth-capped, matching CPU's own `kMaxMixDepth`) — not a case any scene in this loader's corpus has needed. See `pbrt_scenes/mix-material.pbrt`. |
| `hair` | Full | Full | Real Marschner/Chiang fiber scattering (`HairBxDF<T>`) on both — `MaterialType::Hair` was already fully wired for GPU shading before this loader could reach it (see `pbrt_scenes/hair-material.pbrt`). `"sigma_a"` wins if given; else `"reflectance"`/`"color"`; else `"eumelanin"`/`"pheomelanin"`; else the default brown preset — all three resolution formulas (`SigmaAFromConcentration`, `SigmaAFromReflectance`) are pbrt-v4's own closed-form per-channel formulas (neither needs an iterative fit). Real Hair support now also reaches every GPU shape type (sphere, quad, triangle, disk, cylinder, bilinear patch), not just sphere — `Material "hair"` on an ordinary shape used to be unreachable from this loader, so those shapes' own `__trap()` guards were previously dead code; wiring `"hair"` up for real exposed them as a genuine crash until each got its own real (if tangent-proxy) Hair branch. Uses the shading normal as a fiber-tangent proxy on both backends for any non-curve shape (same simplification as this project's own native `build_hair_fibers()` demo) — but paired with real `Shape "curve"` geometry, both backends use the curve's own genuine tangent instead (see that entry above and `pbrt_scenes/curve-hair-tuft.pbrt`). |
| `interface` / `none` / `""` | Full | Full | pbrt-v4's real interface-material idiom for a shape that bounds a participating medium with no BSDF response of its own — the ray passes straight through completely unperturbed, only the medium changes. Both backends build it as a real, dedicated pass-through material (CPU's `interface_material`; GPU's `MaterialType::Interface`) rather than a near-invisible `Dielectric` approximation — no Fresnel/refraction math, no critical angle. Every integrator (default path tracer, BDPT/MLT, SPPM, both backends) skips the crossing entirely via a real "medium boundary" classification, rather than treating it as a specular bounce that would otherwise spend a bounce-budget entry and break MIS for a light seen through the boundary. Only supported on sphere/disk/cylinder shapes (the only ones with a `medium` field) — a trianglemesh/plymesh/bilinearmesh boundary warns and drops the medium instead of silently doing so. |
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
| `AreaLightSource "diffuse"` | Full | Approx | Real NEE-samplable geometry (sphere/quad/disk/cylinder/triangle/bilinear patch) on both. Both backends now honor `filename` (spatially-varying image emission, real per-point UV) and `twosided` on every shape kind, for both a direct hit and NEE sampling — previously GPU only did a real texture lookup for triangle lights, and every non-triangle GPU light kind fell back to reading texel (0,0); separately, NEE sampling for every light kind on GPU (including triangle) never checked `mat.twoSided` at all, silently treating every one-sided light as two-sided for next-event estimation while a direct BSDF-sampled hit already correctly gated on it. See `pbrt_scenes/textured-twosided-lights.pbrt` for a scene exercising both fixes on disk/cylinder lights. **Correction**: CPU's `filename` lookup (`mipmap_texture`'s `point_sample=true` path, a real bilinear tap at LOD 0 with no mip/EWA footprint filtering) already matches pbrt-v4's own `DiffuseAreaLight::L()` exactly (`Image::BilerpChannel()` — verified against pbrt-v4's real source, which also does a single bilinear tap with no mip selection, not nearest-neighbor and not EWA) — CPU is `Full` here, not an approximation. GPU's own lookup (`sample_texture()`/`nee_light_texture_emission()`, shared by every GPU texture read) is a bilinear tap at mip level 0 with the texture's own wrap mode (Repeat/Clamp/Black) - the same interpolation as CPU and pbrt-v4, but with no mip pyramid, so a texture minified below screen resolution aliases where pbrt's filtering would not (an earlier version of this entry called the GPU lookup nearest-neighbour; it is not). Adding a GPU mip pyramid is a larger, foundational change, tracked as a known gap. A `"blackbody L"` colour temperature (Kelvin) is now converted to a real RGB colour on both backends via this codebase's own ported pbrt-v4 spectral pipeline (`BlackbodySpectrum` → `SpectrumToXYZ` → the scene's `ColorSpace` directive, `RGBColorSpace::sRGB()` by default — normalized to ~1 nit before the light's own `"float scale"` is applied — matching pbrt-v4's own light-construction code), not read as a raw number — `barcelona-pavilion`'s night lighting and `contemporary-bathroom` both use this for real (2500K–6500K), and previously rendered as flat colourless white regardless of temperature (`getVec3` silently defaulted to `{1,1,1}` for a 1-number `"blackbody"` param, discarding the temperature entirely). Every other punctual light kind (`point`/`spot`/`distant`/`goniometric`/`infinite`) shares the same fix, via `pbrt_flatten::resolveEmissionColor()` — see that function's own comment. |
| anything else | Unsupported | Unsupported | Dropped with a warning; not visible on either backend. |

## Cameras (`Camera::type`)

CPU: `src/TheRestOfYourLife/scene_registry.h`'s `setup_camera` lambda for
loaded pbrt scenes. GPU: `gpu/optix/scene_builder.cpp`'s
`build_loaded_pbrt_scene()`.

| pbrt camera | CPU | GPU | Note |
|---|---|---|---|
| `perspective` (pinhole) | Full | Full | Both honor an explicit `screenwindow` too (anamorphic/off-center framing), falling back to the normal aspect-scaled viewport otherwise - CPU via `camera::has_screen_window`/`screen_window` (`camera.h`), GPU via `build_pinhole_camera_params()`'s own `screen_window` param (`scene_builder.cpp`), the identical viewport-width/height/center-shift formula on both. A code-review pass found GPU initially had no equivalent at all here (silently ignored, unlike this same round's other CPU-only gaps, which all warn) - closed by wiring the real math into GPU's camera builder rather than adding a disclosure warning, since (unlike animated curves/PixelFilter's real cross-pixel radius) the underlying math was cheap to mirror exactly. |
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
(`optix_device_helpers.h`, duplicated in `wavefront_kernels_materials.cu` per that
file's own "separate translation unit" convention) is the device-side gate.

| pbrt param | CPU | GPU | Note |
|---|---|---|---|
| `"float[4] cropwindow"` / `"integer[4] pixelbounds"` | Approx | Approx | Restricts rendering to a sub-rectangle of the frame - pbrt-v4 allows both together (cropwindow as an NDC fraction, pixelbounds in pixel space), each independently narrowing the region via intersection; both resolve here to one NDC-fraction rectangle (`pbrt_flatten::FlatScene::cropX0`/`X1`/`Y0`/`Y1`) rather than pixel indices, since `xresolution`/`yresolution` are only advisory in this codebase (a CLI width/height argument wins, same as `maxdepth`/`Sampler` type above) - a pixel-space bound resolved against the wrong resolution would be wrong, where a fraction stays correct. **Approx, not Full, on every integrator/backend**: real pbrt-v4 writes a smaller *output image* sized to just the crop rectangle; this codebase instead still writes the full `xresolution`×`yresolution` frame, with every pixel outside the crop rectangle left explicit black rather than sampled/traced - CPU via the existing per-pixel filter-weight-sum-of-zero path (default path tracer) or an equivalent per-pixel/per-splat crop gate (BDPT/MLT/RandomWalk/AO/SimplePath/SimpleVolPath/LightPath/SPPM, see below), GPU via an early-return in each backend's own primary-ray-generation kernel (`__raygen__rg`'s explicit black write; `generate_camera_rays`'s skip-the-enqueue; GPU SPPM's `__raygen__sppm_camera_pass` early-return, relying on its own pixel buffer's one-time zero-init) - a real, deliberate simplification everywhere, chosen to avoid rippling a genuinely different output image size through the PPM/EXR writers, the PNG conversion step, and the Qt GUI's preview, all of which currently assume the output image is `image_width`×`image_height`. **Now honored by every CPU integrator and both main GPU backends, plus GPU SPPM** - `--bdpt`/`--mlt`/`--randomwalk`/`--ao`/`--simplepath`/`--simplevolpath` skip the whole per-pixel loop body for an out-of-crop pixel (`bdpt_adapter.h`'s `*_render_with_adapter()` drivers); `--lightpath` and BDPT's own t==1 light-tracing strategy have no per-pixel loop to skip (samples land at essentially arbitrary pixels), so `SplatFilm` itself gates each splat against the crop rect instead; `--mlt`'s own Markov-chain splat lambda does the identical gate inline, since chain mutations aren't pixel-indexed either - none of this needs renormalization, since each accepted splat already carries its own correct weight regardless of how many other splats were dropped. CPU `--sppm` (`sppm_adapter.h`'s `sppm_camera_pass_with_sky()`) skips the camera pass for an out-of-crop pixel every iteration, leaving its visible point permanently invalid (`SPPMFinalImage()` already reconstructs an untouched pixel as black with no divide-by-zero risk, since `radius` stays at its nonzero initial value). GPU SPPM's own `SPPMLaunchParams::camera` is a direct copy of the same `GpuCameraParams` the other two GPU backends already use, so it already carried a resolved crop rectangle with nothing reading it - `__raygen__sppm_camera_pass` now does. The wavefront backend's own `generate_camera_rays` kernel (launched once per SAMPLE, unlike the recursive backend's single whole-render launch) goes further than an early-return: `wf_launch_generate_camera_rays` (`wavefront_launch.cu`) sizes its CUDA launch grid to just the crop rectangle when one is active, so a cropped-out pixel's GPU thread is never scheduled at all on that backend, not merely skipped after the fact. |

| `"float maxcomponentvalue"` | Full | Approx | Per-sample firefly clamp: if the largest of a sample's r/g/b exceeds this, all three are scaled down so the max component lands exactly at the threshold (pbrt-v4's own real default is effectively unbounded, `1e9`). CPU: `pbrt_scene.h` parses it into `Scene::maxComponentValue`, `pbrt_flatten.h` carries it through to `FlatScene::maxComponentValue`, and `scene_registry.h` wires it onto `camera::max_component_value`, applied unconditionally in `camera.h`'s per-sample loop via `src/shared/film.h`'s `clamp_sensor_rgb()` - a pre-existing, independently-tested helper that had no caller anywhere in the codebase until this. Default CPU path tracer only (same scope cut as `PixelFilter`/`regularize`) - `--spectral` accumulates in CIE XYZ at the point in the loop this clamp needs to run in RGB, so it's skipped there; BDPT/MLT/SPPM/the debug integrators have no equivalent clamp either. **Now real on GPU too, split by backend** (`GpuCameraParams::maxComponentValue`, `optix_types.h`; wired from both the CLI `--maxcomponentvalue` flag - `optix_interface.cpp` - and a loaded scene's own directive - `scene_builder.cpp`): the **recursive backend is Full** - one sample's whole radiance is already known as a single local (`radiance`, `optix_raygen.h`) before it's ever added to the pixel accumulator, so the clamp there is an exact, byte-for-byte port of CPU's own semantics. The **wavefront backend is only Approx** - this backend's framebuffer is a single running total shared across every sample AND bounce of the entire render, fed by 7 independent `atomicAdd` call sites (NEE shadow hits across 4 material-evaluation kernels, escaped/miss rays, BSSRDF probe-exit success and failure, deferred shadow-ray accumulation) with no single point where "this one sample's total radiance" is ever known as one value - a true per-sample-total clamp would need deferring every one of those adds into a per-ray accumulator until definitive path termination, a materially bigger architectural change (the same class of tradeoff already accepted for GPU's area-light texture filtering, ~30 lines up). Instead, each of the 7 sites clamps its own individual contribution independently before adding (`wavefront_kernels_materials.cu`) - real firefly suppression, but not pbrt-v4's exact semantics: a sample whose total exceeds the threshold via several individually-under-threshold contributions isn't caught. |

## Integrator

CPU: `src/TheRestOfYourLife/camera.h` (`ray_color()`/`ray_color_spectral()`).
GPU: `gpu/optix/wavefront_kernels_materials.cu` (`evaluate_materials`/
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

- **Layered BSDFs: rough coat over rough base.** pbrt's own `LayeredBxDF::f()` weighs its two connection strategies with a density
  that is not the competing strategy's (`exitInterface.PDF(-w, wi)` is a density over the outside direction, the other strategy's is
  over the inside one), so the weights do not sum to one and `f()` runs 5-9% above the random walk in `Sample_f` when both the coat
  and the base are rough (albedo 0.594 against 0.547 at alpha 0.3; with the competing density it is 0.546). The port keeps pbrt's
  formula, so NEE (which uses `f()`) carries that bias exactly as pbrt-v4's does; every other combination (smooth coat, smooth base)
  has no such term and agrees to ~1%. On the GPU backends the specular first-bounce sample of a smooth-coat/rough-base BSDF is
  reported to the wavefront tail as a non-specular bounce with a huge MIS pdf (so the light samples still run), which only matters
  for `bool regularize`.

- **Small measured residuals (linear float means, `expectBackendsAgree`).** `camera-medium-absorbing` is noise-limited, not biased: a point light inside fog gives the CPU a heavy tail (four runs of the same render: 1.568, 1.580, 1.980, 1.540) and the GPU backends are bit-reproducible, so each GPU number is one draw; at 2048 spp the CPU is 1.573-1.575, recursive 1.609, wavefront 1.581.

- **Live Preview (ReSTIR) in fog.** `LivePreviewRestirLightBvhTest.ChromaticMediaRestirOnAndOffAgreePerChannel` compares ReSTIR DI and DI+GI with classic NEE per colour channel: E3, E11, A8 and E1 agree within ~3% in every channel (volumetric ReSTIR only resamples light candidates, so a chromatic extinction does not move one channel against another). E1 - a fog sphere nearly filling the Cornell box - used to read 96-97% (DI) and 95% (DI+GI) in every channel, grey fog included: a medium scatter point's reservoir was carried across frames, but the scatter point is redrawn every frame, so the reused sample's weight belonged to a different target. That history reuse for volumes (`kVolumeRestirHistoryReuse`, `wavefront_path_tracer.cpp`) is now off - single-frame noise is the same either way - and E1 reads 99.8% / 98.7-99.1%. The within-frame resampling at the scatter point is unchanged.

- A chromatic homogeneous fog (E11/E12's `sigma_s`) used to read ~2-3% darker on the wavefront backend than on recursive and CPU, which was put down to the spectral uplift of the scattering tint. The real cause was the media model: all three backends used one extinction for every channel, so the three agreed with each other only as far as they shared that error (and E11 was 35-50% off per channel against the per-channel answer). With per-channel extinction (see the table row above) the backends agree on E1 and E11 and the spectral residual that is left is the one the strongly chromatic absorber shows (wavefront 5-8% low).

- `Shape "plymesh"` also reads Wavefront `.obj` (`src/shared/ply_mesh.h`), and a
  filename written `file.obj#name` loads only the faces under `usemtl name`
  (`file.obj#` is the faces before the first `usemtl`). The environment scenes
  (`pbrt_scenes/environment-*.pbrt`) use this to give each .mtl material its own
  `Material` + `Shape` without splitting a gigabyte OBJ on disk; the whole file is
  parsed once per scene load and each `#name` Shape copies its group out of a
  process-wide cache that `pbrt_load::loadFile()` clears when it finishes.

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

- **`Shape "cone"`/`Shape "paraboloid"`/`Shape "hyperboloid"` are not pbrt-v4
  shapes at all.** pbrt-v4 itself removed all three from pbrt-v3's shape set
  ("the rarely-used and occasionally-buggy 'cone', 'hyperboloid', and
  'paraboloid' Shapes have been removed" - pbrt-v4's own "Differences from
  pbrt-v3" notes; confirmed against the real upstream `mmp/pbrt-v4` source,
  whose `shapes.h`/`shapes.cpp` has no Cone/Paraboloid/Hyperboloid class at
  all - real pbrt-v4's only shapes are sphere/disk/cylinder/trianglemesh/
  curve/bilinearmesh). Cone and Paraboloid are kept here anyway as a
  deliberate pbrt-v3-compatibility extension, not a pbrt-v4 gap being
  closed - every "matches pbrt-v4" phrase below refers to shared parameter-
  naming/clipping conventions pbrt-v4's own surviving shapes still use the
  same way (radius/zmin/zmax/phimax), not a claim that pbrt-v4 has these
  shapes. Hyperboloid was never added here and, since it isn't a real
  pbrt-v4 gap either, isn't planned.

  `Shape "cone"`/`Shape "paraboloid"` are supported, **CPU only**
  (`ConeShape<T>`/`ParaboloidShape<T>`, `src/shared/shapes.h`, wrapped by
  `cone_hittable`/`paraboloid_hittable`,
  `src/TheRestOfYourLife/cone_paraboloid_hittable.h` - same object-space-plus-
  unbaked-CTM technique as `disk_hittable`/`cylinder_hittable`, transforming
  the ray into object space at intersection time rather than baking to
  world-space, so both are exactly correct under arbitrary rotation). Real
  `"radius"`/`"height"`/`"phimax"` (cone) and `"radius"`/`"zmin"`/`"zmax"`/
  `"phimax"` (paraboloid) parameters, matching pbrt-v3's own defaults and
  clipping semantics (the last version of pbrt to have these shapes). Real `AreaLightSource` support (both shapes gained
  `random()`/`pdf_value()` overrides in `cone_paraboloid_hittable.h`, backed
  by `ConeShape<T>`/`ParaboloidShape<T>`'s own new `sample()`/`sample_from()`/
  `pdf_from()` in `shapes.h` - a deliberately simpler-than-pbrt-v3 uniform-in-z
  sampling technique with a real, position-dependent `dA/dz` pdf rather than
  pbrt-v3's own exact closed-form inverse-CDF, still statistically unbiased)
  and real `MediumInterface` support (wrapped in a `constant_medium` via the
  same shape-agnostic `addMediumIfPresent()` helper Sphere/Disk/Cylinder
  already use). **AreaLightSource reach, precisely**: `random()`/
  `pdf_value()` are enough for the default CPU path tracer's own NEE (and
  CPU SPPM's direct-lighting pass), but neither shape overrides
  `hittable::sample_area()` - so an emissive cone/paraboloid is invisible to
  BDPT/MLT's own light-sampling (`bdpt_adapter.h` builds its light
  distribution strictly from `sample_area()`, with no NEE-only fallback)
  and to SPPM's photon-emission pass (`sppm_adapter.h`'s identical filter) -
  a --bdpt/--mlt render of such a scene shows zero contribution from that
  light, and SPPM gets direct lighting from it but no caustic/photon
  contribution. This is not a new limitation specific to Cone/Paraboloid -
  `disk_hittable`/`cylinder_hittable` have had the identical gap since
  their own AreaLightSource support landed - but it now applies to two more
  shapes. **GPU (both backends) does not support either shape at all** - a scene
  using one warns at load time and the shape is silently absent from the
  GPU render (`scene_builder.cpp`), matching how this loader already handles
  every other CPU-only shape gap. One ray direction exactly on the
  paraboloid's own symmetry axis (`dx=dy=0`) degenerates the intersection's
  quadratic to a linear equation and is not handled - an accepted limitation
  matching `CylinderShape<T>`'s own identical precedent for a ray exactly
  parallel to its axis. `Shape "hyperboloid"` was never added here (see this
  bullet's opening note - not a real pbrt-v4 gap, and a meaningfully harder
  shape than cone/paraboloid to begin with, no simple implicit quadric the
  way those and cylinder have) - a scene using it falls through to the
  generic "shape not supported" warning like any other unimplemented type.

- pbrt-v4's own **"camera medium"**: a `MediumInterface` declared before the
  `Camera` directive (its "outside" name, real pbrt-v4's own convention -
  `pbrt_scene.h`'s `Scene::cameraMediumIndex`) makes every primary ray start
  already immersed in that medium, no bounding shape needed at all - the
  medium simply fills all of space, for a whole-scene fog/haze/underwater
  effect. Previously silently dropped entirely (no field anywhere captured
  this). Now real on **CPU's default path tracer** (`ray_color()`,
  `camera::camera_medium`, `ambient_medium` - `constant_medium.h`) **and
  both GPU backends** - recursive (`optix_raygen.h`'s own call site,
  `sample_camera_medium()` - `optix_device_helpers.h`, `GpuCameraParams::
  cameraMediumSigmaT` - `optix_types.h`) and wavefront (`__raygen__wf_trace` samples the free flight and turns a scatter into a hit on a synthetic Medium material, `SceneData::cameraMediumMaterialIdx`; `__raygen__wf_shadow` attenuates every shadow ray by `exp(-sigma_t * length)`) - homogeneous only, and not
  combined with a real per-shape medium in the same scene (`pbrt_flatten.h`
  warns and drops the camera medium for either combination, rather than
  modeling the interaction - the same resolved `FlatScene::
  cameraMediumIndex` both backends consume, so this scope cut is enforced
  once, not per-backend). Implemented as an explicit step AFTER the
  per-bounce ray-scene intersection already ran (CPU: `world.hit()`; GPU:
  the primary `optixTrace` call, using its own `t_hit`/miss result as the
  clip distance - every closest-hit program's hit_light/absorbed branches
  now pack `t_hit` too, not just the "scattered" branch, so this is reliable
  regardless of what the ray hit), rather than one more entry in the
  scene's own BVH/scene-traversal structure - an unbounded medium's own
  free-path sample needs to be clipped to whatever real surface (or
  infinity, for an escaped ray) is already known to be nearest, which only
  works reliably once that's already been determined; see `ambient_medium`'s
  own comment (CPU) for the full reasoning, including why the real blocker
  isn't traversal order (a plain hittable/BVH entry already resolves nearest-
  hit correctly) but the lack of a side-channel for a non-winning entry to
  still attenuate the path's throughput. Applies on every bounce of a path,
  not just the primary ray - once inside a medium filling all of space,
  every ray segment is inside it too; this doesn't model a real "exit" via
  some other shape's own `MediumInterface` (the excluded combination above).
  NEE/shadow-ray attenuation through it is also applied, on both backends
  that support this feature (a light behind the fog IS dimmed by it on the
  way to a shadow-ray target now, not just primary/bounce-ray transmission)
  - CPU: each of `ray_color()`'s NEE strategies (area/portal/sky/punctual)
  multiplies by `ambient_medium::transmittance_over()` at the shadow ray's
  own real distance (`camera_medium_trans`, `camera.h`); GPU recursive:
  every NEE call site across all 7 material-shading blocks in
  `optix_device_helpers.h` (plus the shared punctual-light helper,
  `optix_device_helpers_lighting.h`) multiplies by
  `camera_medium_shadow_trans()`, the same Beer-Lambert formula against
  `GpuCameraParams::cameraMediumSigmaT`. Both treat an infinite-distance
  light (sky, or the portal light's own window) as fully extinguished by
  any positive extinction - physically exact for an unbounded medium, not
  an approximation. `ray_color_spectral()`, BDPT/MLT, CPU SPPM, and
  GPU SPPM still don't consume this field at all - each warns explicitly
  (`cpu_interface.cpp`/`cpu_interface_bdpt.cpp`/`optix_renderer_render.cpp`/
  `optix_interface.cpp`'s `sppm_gpu_unsupported_reason` - the last of these
  rejects the render outright rather than warning-and-continuing, matching
  that function's own established convention for every other GPU-SPPM gap)
  rather than silently rendering without the requested fog.

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
  pbrt-v4 has the identical divergence for the identical reason. Cubic
  (`"integer degree"` 3), Bezier-basis (`"string basis"` `"bezier"`) curves
  are the default and most common case; quadratic (`"integer degree"` 2)
  Bezier and cubic `"string basis"` `"bspline"` curves are ALSO now real,
  on both backends - both reduce EXACTLY (not approximately) to the same
  cubic-Bezier-per-segment representation `CurveShape<T>`'s intersection
  math already needs (a quadratic Bezier's exact degree-elevation to cubic;
  a uniform cubic B-spline's exact per-segment change-of-basis to Bezier -
  `pbrt_flatten.h`'s `curveDegreeElevateQuadratic()`/
  `curveBsplineSegmentToBezierCubic()`), so the conversion lives entirely in
  the loader with no changes needed to either backend's own curve code. The
  one remaining unsupported combination is a quadratic (degree 2) B-spline -
  real pbrt-v4 scenes essentially never combine the two, and closing it
  needs a second, quadratic-specific B-spline conversion matrix for
  marginal real-world value; falls back to the generic "shape not
  supported" warning, same as any other genuinely unsupported degree/basis.
  `"integer splitdepth"` is not implemented -
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
  reasoning and `optix_intersection_bilinear_patch.h`/`wavefront_kernels_materials.cu`'s
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
  are worked examples of all three. `"nanovdb"` is ALSO real now, **CPU
  only** — see this file's own entry below for the full scope; any other
  `"type"` value still falls back to homogeneous with a warning.

- `MakeNamedMedium "nanovdb"` (pbrt-v4's real NanoVDB-format sparse
  density grid, read from an external `.nvdb` file) is real on **CPU**:
  `pbrt_cpu_builder.h` reads the named `float` grid (`"gridname"`,
  default `"density"`) via a vendored, header-only NanoVDB reader
  (`src/external/nanovdb/` — NVIDIA's own `NanoVDB.h`/`io/IO.h`,
  Apache-2.0, no third-party dependencies beyond the C++ standard library
  for uncompressed `.nvdb` files) and bakes its active index region into a
  dense flat array, reusing `GridMediumData<double>`/
  `grid_medium_hittable.h` (the SAME machinery `"uniformgrid"` above
  already uses) completely unchanged - so it gets that machinery's real
  DDA-majorant delta tracking for free. The world-space placement/
  transform can't be resolved at `flatten()` time the way cloud/rgbgrid/
  uniformgrid's `"p0"`/`"p1"` can (the grid's own extent isn't known until
  the file is read), so the scene's CTM is carried through unbaked
  (`Medium::nanovdbXform`) and composed with the grid's own baked
  index-to-world transform in `pbrt_cpu_builder.h` itself, by sampling 4
  points through the composed map rather than hand-deriving NanoVDB's own
  matrix-layout convention (any affine map is fully determined by 4
  points). See `pbrt_scenes/nanovdb-medium.pbrt` (scene `E9`) for a worked
  example, using a small synthetic test grid
  (`pbrt_scenes/nanovdb-sphere.nvdb`) authored directly via NanoVDB's own
  header-only grid-construction tools (`tools/CreatePrimitives.h`) - no
  external asset download, no OpenVDB dependency anywhere in this loader.
  Real blackbody emission is supported via a second named `"string
  temperaturename"` grid (per-voxel Kelvin, converted to RGB and weighted
  by `sigma_a/sigma_t` at each scatter event, same convention as
  `"rgbgrid"`'s own per-voxel `"Le"`) - `sigma_a` is only forced to 0 (pure
  scattering) when no temperature grid is given; a real `sigma_a` is
  required alongside `temperaturename` for the emission to be visible.
  Real, disclosed scope cuts: only a single named `float` density grid (no
  other NanoVDB build types - `Vec3f`/`Mask`/`Int32`/index-grids); no
  animated/sequence grids; the sparse grid is densified into a flat array at load time
  rather than sampled natively sparse (a real memory/scope tradeoff for
  reusing the existing dense-grid machinery unchanged, not a NanoVDB
  limitation) - capped at 512 voxels per axis (`pbrt_cpu_builder.h`'s own
  `kMaxVoxelsPerAxis`), both to bound worst-case memory for a real, sparse-
  but-large-bbox asset and to close off a real integer-overflow-into-heap-
  overflow risk a code-review pass found in the pre-cap version (a
  corrupt/malicious file claiming an extreme bbox could silently
  under-allocate the dense array while the bake loop still wrote the true,
  huge extent). A grid whose active region genuinely needs more than 512³
  voxels falls back to an invisible medium rather than attempting the
  allocation. Same "sigma_a forced to 0" scope as `"uniformgrid"` above
  (`grid_medium_hittable.h`'s own limitation, not new here).
  **GPU has no NanoVDB support at all** - `pbrt_gpu_builder.h` has no
  `"nanovdb"` branch, so a nanovdb medium falls through to the generic
  homogeneous-medium path there, rendering as flat fog filling the WHOLE
  boundary shape (using the scene's own `sigma_a`/`sigma_s`) rather than
  the real sparse density field - `scene_builder.cpp` warns explicitly by
  name, since this is a visibly *wrong* render on GPU, not merely an
  absent one (unlike, say, Cone/Paraboloid, which just don't appear).
  NanoVDB's own format is explicitly designed to need no deserialization
  on GPU (the raw file bytes already ARE the traversable structure, just
  `cudaMemcpy` + `reinterpret_cast`), which could make GPU support cheaper
  than a typical CPU-to-CUDA port if attempted later - but this codebase
  has two prior unresolved GPU device-crash precedents on non-trivial
  device call graphs (`CloudMedium::compute_density()`'s member-call
  stall, worked around by hand-duplicating a free function; the light-BVH
  device-consumption CUDA 700 crash, never root-caused despite 5
  ruled-out theories), so GPU NanoVDB is scoped as a genuinely separate
  follow-up round with its own isolated spike, not attempted here.

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
  non-feature for those two types. **`rgbgrid`'s own `"Le"` support is now
  real on GPU too**, not CPU-only (`GpuRgbGridMedium::leDataOffset`/
  `Le_scale`, `optix_intersection_sphere.h`/`wavefront_kernels_materials.cu`'s own
  RgbGridMedium closest-hit cases, `gpu/optix/pbrt_gpu_builder.h`'s scene
  builder). The GPU weights it by the event's absorption share like the CPU does (`Le_c * w_c * sigma_a_c / mean(sigma_t)`) now that its grid carries
  `sigma_a` (`GpuRgbGridMedium::saDataOffset`); it used to emit the FULL `Le` at every collision because the GPU grid had no `sigma_a`.
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
  three declares a nonzero `"Le"`. `DielectricMedium` (a fused dielectric-
  surface-plus-interior-medium material) now also gets `"Le"` for the one
  case the pbrt loader actually builds it for (see immediately below) —
  `MaterialData::medium_emission` lives in the same union slot as plain
  `Medium`'s, and `medium_phase_nee_mis()` already read it unconditionally
  for every medium-interior scatter case including `DielectricMedium`
  before this round, so populating it at build time was a one-line addition
  once the loader started reaching that `MaterialType` at all.

  **`DielectricMedium` fusion, now live for pbrt-authored scenes**: earlier
  revisions of this doc described `DielectricMedium` as reachable only from
  this codebase's hand-built native scenes, since `pbrt_gpu_builder.h`
  unconditionally built a plain `Medium` (dropping the shape's own surface
  Material entirely) whenever `MediumInterface` was present — a scene
  pairing fog with a real dielectric shell ("jade"/"wax"/mist-in-glass)
  rendered as a plain fog sphere on GPU, with no visible glass surface at
  all. Fixed for the specific, common case this codebase's own
  `subsurface-slab.pbrt`/`dielectric-medium-showcase.pbrt` scenes use: a
  **sphere** whose own `Material` is a real, **smooth** dielectric (no
  roughness, no roughness texture) now builds the pre-existing fused
  `MaterialType::DielectricMedium` instead — the exact material
  `scene_builder.cpp`'s `add_dielectric_medium()` already built for the
  native scenes these migrated from, now reachable from the generic loader
  too (`pbrt_gpu_builder.h`'s `mediumMaterialIndex()`). Verified via direct
  before/after GPU render comparison on B13 (Subsurface Slab): the fix adds
  visible specular highlights on both spheres that were completely absent
  before (a flat, matte, fog-only look), now matching CPU's reference
  render closely.

  **Follow-up round: cylinder fusion, and the opaque-surface case closed
  outright.** Two more real gaps from the paragraph above are now fixed:

  - **Cylinder + smooth dielectric + medium** now fuses exactly like
    sphere. `__closesthit__cylinder` (`optix_intersection_disk_cylinder.h`)
    and `__closesthit__wf_cylinder`/`evaluate_materials()`'s
    `MaterialType::DielectricMedium` case (`wavefront_intersection_disk_
    cylinder.h`/`wavefront_kernels_materials.cu`) both gained the same
    entry-surface-refracts / exit-surface-or-interior-phase-scatter
    structure sphere's own `DielectricMedium` case already had, reusing
    each backend's existing cylinder near/far chord math (tube quadric
    clipped to a z-slab). Both backends' cylinder shadow any-hit
    (`__anyhit__shadow_cylinder`/`__anyhit__wf_shadow_cylinder`) gained
    real Beer-Lambert attenuation for it too, matching sphere's identical
    fix. `pbrt_scenes/cylinder-medium.pbrt` (scene E6, eta=1.001 near-
    invisible shell) is this project's own pre-existing bundled example of
    exactly this combination — verified via direct before/after GPU render
    comparison against CPU: GPU averaged **0.243** brightness before this
    fix (CPU: 0.112, a huge, clearly-visible over-brightening — the old
    plain-`Medium` cylinder path rendered a nearly flat, overexposed-
    looking fog block) and **0.093** after (within normal Monte-Carlo noise
    of CPU's 0.112), with the post-fix image's brightness gradient now
    visually matching CPU's closely. Not previously caught by any test:
    the existing unit test only checked which `MaterialType` a cylinder
    resolved to, never its actual rendered brightness.
  - **Opaque surface + medium** (diffuse+medium, metal+medium,
    coateddiffuse/coatedconductor+medium, subsurface+medium, measured+
    medium, hair+medium — on **either** sphere or cylinder) is a real,
    separate bug now fixed without needing CPU's dual-hittable machinery
    at all: `pbrt_gpu_builder.h`'s `isOpaqueSurfaceMaterial()` recognizes
    that CPU's own two-hittable composition always resolves this
    combination identically to the surface material alone anyway (an
    opaque surface's own entry-point hit can never lose a nearest-hit
    comparison to the medium's own stochastically-sampled interior hit, so
    the medium is never actually visible on CPU either), so GPU now simply
    builds the real surface material and skips the medium path entirely —
    the exact same visual result CPU already produces, far more cheaply
    than a true dual-primitive composition. Before this fix, GPU built a
    plain fog-only `Medium` instead, silently discarding the real, always-
    winning opaque material — arguably a worse divergence than the
    dielectric case, since the rendered material was wrong in both
    transparency and response, not just missing a refractive highlight. No
    bundled scene exercised this combination before now (every existing
    `MediumInterface` scene in `pbrt_scenes/` already used a dielectric
    surface), so this was a latent, never-triggered bug rather than a
    visible regression in any shipped scene — closed proactively, with new
    unit test coverage (`pbrt_gpu_disk_cylinder_tests.cpp`'s
    `CylinderMediumInterfaceWithOpaqueSurfaceKeepsRealMaterial`).

  **Follow-up round: THIN dielectric fusion closed too.** `Material
  "thindielectric"` combined with a medium now fuses into
  `MaterialType::DielectricMedium` exactly like smooth `Dielectric`
  does — `thin_dielectric_scatter()`/`wf_thin_dielectric_scatter()`
  (`optix_device_helpers.h`/`wavefront_device_helpers.h`, pbrt-v4's own
  `ThinDielectricBxDF` reflect-or-straight-through coin-flip, no bending)
  needed no tangent frame, texture, or NEE setup to drop into the entry/
  exit boundary alongside the smooth case, unlike rough (see below) — a
  cheap, mechanical extension, not a new architecture. `mediumMaterialIndex()`
  (`pbrt_gpu_builder.h`) now also recognizes `MaterialKind::ThinDielectric`
  and sets `dielectric_medium_extra.surfaceKind` (0=smooth/1=thin/2=rough,
  see the rough entry below for the third value) both backends' entry/
  exit branches key off. New example scene `pbrt_scenes/thin-dielectric-
  medium.pbrt` (E11) demonstrates it — unlike the smooth-dielectric "near-
  invisible shell" convention (`eta≈1.001`, needed there to avoid visibly
  bending the view through real refraction), a thin shell never bends
  anything, so this scene uses a real, visible `eta=1.5` glass shell with
  the fog clearly visible straight through it, plus a genuine thin-film-
  like Fresnel glint. Verified via direct GPU/CPU visual + numeric
  comparison (`MaterialCpuGpuParityTest` passes at the standard 30%
  tolerance, no special-case override needed) — both images match closely.

  **A real, independent, pre-existing bug found and fixed while scoping
  this**: the wavefront backend's own PLAIN (non-fused)
  `MaterialType::ThinDielectric` case (`wavefront_kernels_materials.cu`)
  used a different, incorrect reflectance formula (`Fr/(Fr+T²)`) that does
  not match pbrt-v4's `ThinDielectricBxDF` — both CPU
  (`src/shared/bxdfs_simple.h`'s shared `ThinDielectricBxDF`, used by
  `material_pbrt.h`'s `thin_dielectric`) and the recursive GPU backend's
  own `MaterialType::ThinDielectric` case already used the correct
  multi-bounce geometric series (`R_eff = R + T²R/(1-R²)`); only this one
  wavefront case had independently drifted. Diverges substantially at
  moderate incidence (e.g. R=0.1: old gives ≈0.110, correct gives ≈0.182).
  Fixed by routing it through the same corrected `wf_thin_dielectric_
  scatter()` helper the new `DielectricMedium`-fused case needed anyway —
  this was not something the fusion work introduced, a latent bug this
  project's own authoritative shared BxDF reference made easy to catch by
  comparison.

  **Follow-up round: ROUGH dielectric fusion closed too — the real
  engineering case.** A frosted/rough `Material "dielectric"` (nonzero
  roughness, or a roughness texture) combined with a medium now fuses into
  `MaterialType::DielectricMedium`'s entry/exit boundary with a genuine GGX
  microfacet BSDF and real glossy NEE/MIS, not just a cheap coin-flip or
  refraction call. `mediumMaterialIndex()` (`pbrt_gpu_builder.h`) now
  classifies rough the same way the plain (non-fused) `RoughDielectric`
  build path already does (`roughness_u`/`roughness_v` > 0, or a roughness
  texture), setting `surfaceKind=2` and storing the flat roughness in
  `dielectric_medium_extra.roughness` — NOT the top-level `fuzz`/`roughness`
  field `RoughDielectric` itself uses, since that union slot is already
  `g` (the medium's own Henyey-Greenstein asymmetry) for this fused type;
  `roughnessV`/`remapRoughness`/`textureIdx` are reused directly (genuinely
  free for `DielectricMedium` otherwise).

  **Recursive backend**: `rough_dielectric_scatter_and_nee()`
  (`optix_device_helpers.h`) factors the GGX VNDF sample/reflect/refract +
  inline NEE/MIS (against area, punctual, and sky lights) straight out of
  `shade_material()`'s own `MaterialType::RoughDielectric` case — both the
  plain material and `DielectricMedium`'s fused entry/exit sub-cases now
  call the identical function (parameterized by which union slot the flat
  roughness lives in, and whether dispersion is even safe to read — see the
  function's own header comment on why `mat.dispersive_extra` and `mat.
  dielectric_medium_extra` alias the same union slot, making dispersion
  support for the fused case actively unsafe, not just unimplemented).
  `material_needs_dpdu()` now includes `DielectricMedium` unconditionally
  (every fused instance pays a small tangent-frame cost, even smooth/thin
  ones that don't use it, rather than threading a finer per-instance gate
  through that dispatch).

  **Wavefront backend — the real decision point**: `wf_finish_material_
  scatter()`'s `isPhase`/`glossy_isType` event classification is driven
  entirely by a `matType` function ARGUMENT (not by re-reading the
  material's own stored type), and unconditionally routes every
  `DielectricMedium` event into its medium-interior-phase-scatter handling
  — there is no "fused rough dielectric surface" case in that dispatch to
  route into instead, and reworking that ~1100-line shared function's event
  classification (used by every other glossy and medium material type) was
  explicitly out of scope. A true hand-duplicated inline NEE block was
  scoped and found to have a real correctness problem of its own, not just
  more code: `wf_finish_material_scatter()`'s own NEE gate and the next
  bounce's `specular_bounce` flag both read the SAME `is_specular` value,
  so suppressing the shared function's own (wrong, phase-based) NEE
  attempt by forcing `is_specular=true` would also mislabel a genuinely
  glossy bounce as specular for the NEXT vertex's own MIS decision — a real
  bias (that bounce's BSDF-sampled continuation ray, if it later hits a
  light directly, would skip MIS weighting and add full unweighted
  radiance instead of its correct fractional share). Avoiding that bias
  with a true duplicate would mean also taking over the shared function's
  next-ray-setup/Russian-roulette/throughput logic - the exact scope
  explosion avoiding a rework was meant to prevent.

  The chosen fix instead: `evaluate_materials()`'s `DielectricMedium` case
  (`wavefront_kernels_materials.cu`) does its own real GGX VNDF scatter
  sampling (mirroring `wavefront_kernels_materials_dielectric.cu`'s own
  `RoughDielectric` case, the same shared `TrowbridgeReitz<float>`/
  `RoughDielectricBxDF<float>` templates), then — only for the genuinely
  glossy (non-`EffectivelySmooth`) sub-case — passes `MaterialType::
  RoughDielectric` instead of the real `mat.type` as the `matType`
  ARGUMENT to the shared `wf_finish_material_scatter()` tail call every
  case already falls through to (a new `effectiveMatType` local, `mat.type`
  by default, overridden only there). Every other argument (`h.materialIdx`,
  `glossyAlphaForNEE`/`glossyAlphaVForNEE`, `matEta`) still correctly points
  at/derives from the real fused material, so this reuses the EXISTING,
  already-correct, already-tested `RoughDielectric` glossy-NEE path
  verbatim — zero edits to the shared function, correct `is_specular`/MIS/
  next-bounce behavior, and ReSTIR DI/GI/probe-cache/NRC-training all keep
  working for this combination (better than the disclosed trade-off an
  earlier draft of this fix accepted). `wf_material_needs_dpdu()` gained
  the identical `DielectricMedium` addition as the recursive backend's own
  `material_needs_dpdu()` (cylinder's own `objDpdu` was already
  unconditional, no change needed there).

  **A real, independent bug found while reading this shared function
  closely**: none found this round beyond the thin-dielectric one above -
  the matType-substitution approach was chosen specifically because it
  reuses already-verified code rather than writing new NEE math that could
  hide a fresh one.

  New example scene `pbrt_scenes/rough-dielectric-medium.pbrt` (E12,
  "frosted jade" — a real `eta=1.5`, `roughness=0.25` glass sphere wrapping
  a jade-green scattering medium, comparable directly against
  `subsurface-slab.pbrt`'s smooth jade sphere) verified this is correct, not
  just compiling: `MaterialCpuGpuParityTest` passed at the standard 30%
  tolerance on the first attempt (no special-case override needed, itself
  a good sign for an implementation this involved), and a direct three-way
  visual/numeric render comparison showed all three backends producing the
  same soft, frosted-glass look with the jade glow diffusing through it —
  CPU 0.1425 avg brightness, GPU-recursive 0.1381 (1.3% mean-abs-diff from
  CPU), GPU-wavefront 0.1174 (2.7%/2.3% mean-abs-diff from CPU/recursive
  respectively) — all comfortably within normal Monte-Carlo noise at this
  scene's 128spp, not a bias signature.

  Still **not** fixed, and the real remaining GPU limitation:
  `DiffuseTransmission`/`Interface` combined with a medium (on either
  shape), and ANY surface material combined with a medium on a disk (never
  meaningful - no "inside" for a zero-thickness plane) or combined with
  CloudMedium/RgbGridMedium/GridMedium specifically (the fused material
  only exists for homogeneous media). CPU's real dual-primitive
  composition (an actual second, coincident-geometry primitive per medium
  shape) remains the only way to close the gap for these remaining
  combinations uniformly - a materially different, bigger architecture
  change than anything this or the prior rounds made, and no bundled scene
  currently needs it.

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
  (`"nanovdb"`, real on CPU now — see the `MakeNamedMedium` entry above)
  remain out of scope on GPU regardless (GPU has no `"nanovdb"` branch at
  all, so it falls through to the generic homogeneous-medium path) — this
  fix applies to every medium type this loader can actually build on GPU
  today.
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
  original scene author considered exactly this. `checkerboard`/`mix`'s own
  `tex1`/`tex2` now ALSO support a SECOND level of nesting, **CPU only**:
  either may name a further `Texture` that is itself `checkerboard`/`mix`
  (not just a bare imagemap or flat literal), real recursive evaluation via
  `pbrt_flatten::NestedProceduralTexture` (`pbrt_flatten.h`) and
  `bvh_aggregate_hittable.h`-style CPU-only wiring — `checkerOrMixSlot()`/
  `buildNestedProceduralTexture()` (`pbrt_cpu_builder.h`) build a real
  nested `uv_checker_texture`/`mix_texture` object for that slot. GPU has
  no representation for this (`TextureData::tex1ImageIdx`/`tex2ImageIdx`
  are image-only, `optix_types.h`) — a scene using it renders correctly on
  CPU and falls back to a flat colour on GPU: not the previous default
  white/black, but a real average of the nested pattern's own two colours
  (computed once at `flatten()` time, so GPU still gets a reasonable
  approximation instead of an arbitrary placeholder), with an explicit
  warning (`scene_builder.cpp`) naming the material count affected. `mix`'s
  own `"amount"` parameter now ALSO reaches this same second level (it used
  to stay capped at one, bare-imagemap-only) — a texture-driven blend mask
  itself nested in a further `checkerboard`/`mix` (e.g. a checker-driven
  dirt/wear mask, or a mix-of-mixes weight) resolves the same way tex1/tex2
  already did, via the same `checkerOrMixSlot()` reused for the amount slot
  too; GPU's flat-colour fallback for this case reduces the nested pattern's
  own average colour to a single representative scalar
  (`nestedProceduralAverageScalar()`) rather than leaving `mixAmount` at its
  meaningless struct default. A THIRD level of nesting (that second-level
  texture's own tex1/tex2/amount naming yet another procedural texture, on
  any of the three slots) still falls back to the generic "not supported"
  warning on both backends — a documented, deliberately bounded cap, not a
  new limitation: no bundled scene needs it, and unbounded recursion would
  need real cycle-detection this loader has never needed before, for a
  feature real pbrt-v4 scenes essentially never exercise past 2 levels.

- `Diffuse`/`CoatedDiffuse` `"reflectance"` also now supports 4 more pbrt-v4
  procedural texture kinds on both backends: `"windy"` (two FBm calls
  combined, parameterless in real pbrt-v4), `"wrinkled"` (raw Turbulence,
  not FBm - same `"octaves"`/`"roughness"` params as `fbm`), `"dots"` (a
  per-UV-cell polka-dot pattern, `"inside"`/`"outside"` each supporting the
  same one-level-nested-bare-imagemap scope as `checkerboard`'s own
  `tex1`/`tex2`), and `"bilerp"` (plain bilinear blend of 4 corner colours
  by (u,v), flat-literal `v00`/`v01`/`v10`/`v11` only - no nested-imagemap
  support, since a real scene binding anything but a flat colour to a
  bilerp corner is vanishingly rare). CPU: `windy_texture`/
  `wrinkled_texture`/`dots_texture`/`bilerp_texture` (`texture.h`), reusing
  `fbm_simple`/`turbulence_simple`/`perlin_noise` (`noise.h`) - `dots`'s
  per-cell hash reuses `perlin_noise` at a fixed z=0.5, matching real
  pbrt-v4's own `Noise(x,y)` 2-arg overload exactly (the same 3-arg
  gradient noise FBm/Turbulence already use, not a separate hash). GPU:
  `TextureKind::Windy`/`Wrinkled`/`Dots`/`Bilerp` (`optix_types.h`), with
  `Wrinkled` reusing `omega`/`octaves` and `Dots` reusing
  `color1`/`color2`/`tex1ImageIdx`/`tex2ImageIdx` (the same fields
  `FBm`/`Mix` already use), `Bilerp` adding two new corner fields
  (`bilerpV10`/`bilerpV11` - `color1`/`color2` carry `v00`/`v01`). `"ptex"`
  remains unsupported and falls back to the generic "texture not
  supported" warning like any other unrecognized class, unchanged.
  **This is a genuinely larger gap than "needs an external library"
  suggests** - investigated properly once (vendored Disney's Ptex
  v2.5.4 source and built its libdeflate dependency via vcpkg to check;
  both since removed/left uninstalled again, no trace in the build).
  Real per-face Ptex lookup needs to know WHICH MESH FACE a shading
  point belongs to, and that index does not survive to texture-
  evaluation time anywhere in this codebase today - closing that gap
  needs four separate changes, not one: (1) a new field on
  `hit_record` (`src/TheRestOfYourLife/hittable.h`), (2) `triangle::hit()`
  (`src/TheRestOfYourLife/triangle.h`) actually writing it in - it already
  computes `tri_idx` internally but never copies it out, (3) extending
  `texture::value()`/`value_diff()`'s signature
  (`src/TheRestOfYourLife/texture.h`) and every call site across
  `material_simple.h`/`material_pbrt.h`/`principled_material.h`/
  `normal_map_materials.h`, and (4) fixing `pbrt_flatten.h`/
  `pbrt_cpu_builder.h` itself - every `Shape "trianglemesh"`/`"plymesh"`/
  `"loopsubdiv"` in a scene currently gets merged into ONE global,
  vertex-deduplicated triangle list (`emitGeometry`,
  `pbrt_cpu_builder.h`), discarding per-shape face-order boundaries, so
  even a correctly-plumbed index would point into the wrong (merged)
  numbering for any multi-mesh scene - nearly all real ones. This is a
  real architectural change to the core intersection/shading pipeline
  touching every material and texture in the renderer, not a contained
  texture-kind addition - tracked here as a deliberately deferred gap,
  not a quick follow-up.
  **Historical note**: an earlier, separate procedural-texture port
  (`src/shared/textures.h`, plus most of `src/shared/procedural_textures.h`'s
  original content) duplicated `windy`/`wrinkled`/`dots`/`bilerp`/`marble`/
  `fbm` with no live consumer anywhere in the actual rendering pipeline -
  discovered while implementing this round, flagged for investigation into
  whether it was worth finishing/integrating or removing. It's since been
  removed as dead code (`dcdf15cf`); `src/shared/textures.h` no longer
  exists, and `procedural_textures.h` now keeps only the one piece that
  wasn't redundant (real anti-aliased checkerboard filtering, used by
  `uv_checker_texture::value_diff()` - see that file's own header comment).
  The actually-wired `windy_texture`/`wrinkled_texture`/`dots_texture`/
  `bilerp_texture` implementations live in `src/TheRestOfYourLife/texture.h`
  alone.

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
  `DiffuseTransmission`, and (a follow-up round) the `transmittance`
  (`DiffuseTransmission`) and `roughness` (`Dielectric`) slots too, via each
  slot's own `TextureDecodeOptions` field
  (`transmittanceTextureOptions`/`roughnessTextureOptions`,
  `resolveTextureDecodeOptions()`, `pbrt_flatten.h`) — the remaining two
  texture-filename slots (`alphaTextureFilename`/`displacementTextureFilename`)
  still get this codebase's long-standing gamma-2.2/Clamp/no-invert defaults
  regardless of what the scene's own `imagemap` declares: alpha is a
  coverage MASK, not colour, so `"encoding"` (gamma) isn't meaningful there
  by this codebase's own established design
  (`getOrBuildPbrtAlphaMaskTexture()`'s own comment,
  `gpu/optix/pbrt_gpu_builder.h`); displacement goes through a materially
  different CPU pipeline (`rtw_image`/`image_texture`, not
  `mipmap_texture`/`MipMapOptions`) with no wrap-mode concept at all today —
  a deliberate, narrower scope cut than before. `"encoding"`: `"linear"` → gamma 1.0 (no decode — the
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
  `gpu/optix/wavefront_kernels_materials.cu`) did the identical hard `[0,1]` clamp, so
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
  displacement) is still built without threading these options through at
  all, so those stay byte-for-byte at GPU's original hard-clamp/gamma-2.2/
  no-invert behavior, unaffected by this change (roughness/transmittance now
  thread their own resolved options through, per the follow-up round noted
  above). Gamma and invert are baked into the decoded pixel bytes once at
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
  `"encoding"`/`"wrap"`/`"invert"` to one of the 2 remaining non-resolving
  texture-filename slots (alpha/displacement) also warns that the request is
  ignored there — those slots still only ever get this codebase's original
  gamma-2.2/Clamp/no-invert defaults on either backend, regardless of what
  the scene's own imagemap declares.
