# Multiple Importance Sampling (MIS)

## TL;DR

MIS combines two sampling strategies - sampling the BRDF and sampling the
lights directly (next-event estimation, NEE) - and weights their
contributions so neither strategy's weaknesses dominate. It is implemented
and in active use on the CPU renderer and both GPU (OptiX) backends.

## The problem: neither strategy alone works everywhere

**BRDF sampling** scatters rays according to the surface's own scattering
distribution (e.g. cosine-weighted for Lambertian). It works well for
glossy/specular surfaces, where only rays near the reflection direction
carry any BRDF value anyway - but for a small light source, the odds of a
randomly-scattered ray happening to hit it are tiny, so most samples
contribute nothing and the image is noisy.

**Light sampling (NEE)** samples a point on a light directly and connects
to it with a shadow ray. It converges quickly for diffuse surfaces and
small/bright lights - but for a glossy or specular surface, the BRDF value
in the direction of that sampled light point is usually near zero (the
surface only reflects strongly near the mirror direction), so most of
those samples contribute nothing either.

Neither strategy dominates the other in general; which one is better
depends on the surface and the light at that specific hit point.

## The solution: weight both strategies by how well each fits

At a shading point, MIS draws one sample from the BRDF's own distribution
and one sample from the light's distribution, evaluates both, and combines
them with weights computed from the **power heuristic** (Veach & Guibas,
1995):

```
weight(pdf_a, pdf_b) = pdf_a^2 / (pdf_a^2 + pdf_b^2)   // beta = 2
```

For a BRDF-strategy sample, `pdf_a` is that direction's BRDF pdf and
`pdf_b` is what the light strategy's pdf *would have been* for that same
direction (and vice versa for a light-strategy sample). The result:
whichever strategy would have been more likely to generate that particular
direction gets most of the weight, and the combined estimator stays
unbiased regardless. Concretely: for a diffuse surface under a tiny light,
the light-sampled ray's light-pdf is high and its would-be BRDF-pdf is
low, so it gets weighted heavily; for a glossy surface, the BRDF-sampled
ray's BRDF-pdf dominates instead. Neither strategy's weak case drags the
result down, because the other strategy's sample is already covering it.

## Current implementation

The power heuristic itself lives in one place, shared by every backend:
`src/shared/mis_sampling.h`'s `PowerHeuristic(nf, fPdf, ng, gPdf)` (a
`CPU_GPU`-tagged template, compiled by both the host compiler and nvcc -
see that file's own comment). Both the CPU renderer and both GPU backends
wrap it in a local `mis_power_heuristic()` helper and use it at every NEE
(light-sampling) site:

- **CPU**: `src/TheRestOfYourLife/camera.h`'s `ray_color()` (and its
  spectral counterpart `ray_color_spectral()`) - one-sample MIS combining
  a BSDF-sampled bounce with an NEE light sample each iteration, weighted
  via `mis_power_heuristic()`.
- **GPU-recursive**: `gpu/optix/optix_device_helpers.h`'s `shade_material()`
  - real NEE + MIS at every non-delta material hit, including against the
  sky/infinite lights, not just area lights.
- **GPU-wavefront**: the equivalent NEE-gated path in
  `wavefront_kernels.cu`'s material-evaluation kernel.

Delta-distribution materials (mirror, perfect dielectric) skip NEE/MIS
entirely on all three backends, since a delta BSDF has zero probability of
matching any light-sampled direction - there is nothing for MIS to weight
in that case, and specular reflection/refraction is sampled directly
instead. See [`FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md) for the full
per-material, per-backend NEE/MIS support matrix (a handful of specific
material/backend combinations use a documented, deliberately simpler
unbiased-but-noisier estimator instead of full MIS - e.g. participating
media phase-function scattering and `DiffuseTransmission`'s second
hemisphere on GPU - each with its own comment explaining why).

## Further reading

- Veach & Guibas, 1995: "Optimally Combining Sampling Techniques for Monte
  Carlo Rendering" - the original MIS paper.
- *Physically Based Rendering* (pbr-book.org), the "Monte Carlo
  Integration" chapter's MIS section - this codebase's `PowerHeuristic`
  signature mirrors pbrt-v4's own directly.
