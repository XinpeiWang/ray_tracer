// optix_raygen.h -- Ray generation program
// Included by optix_programs.cu

extern "C" __global__ void __raygen__rg() {
	// Get pixel coordinates
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	const unsigned int px = idx.x;
	const unsigned int py = idx.y;

	if (px >= params.width || py >= params.height) return;

	// Film "cropwindow"/"pixelbounds" (pbrt-v4) - a pixel outside the crop
	// rectangle is written explicit black (matching CPU camera.h's own
	// in_crop-gated normalization) rather than left unwritten: the
	// framebuffer is cudaMalloc'd, not zero-initialized, so skipping the
	// write here without an explicit black store would leave stale/garbage
	// device memory in the final image outside the crop instead of black.
	if (!gpu_in_crop(params.camera, static_cast<int>(px), static_cast<int>(py))) {
		const unsigned int idx_flat = py * params.width + px;
		params.framebuffer[idx_flat] = make_float3(0.0f, 0.0f, 0.0f);
		if (params.albedoBuffer) params.albedoBuffer[idx_flat] = make_float3(0.0f, 0.0f, 0.0f);
		if (params.normalBuffer) params.normalBuffer[idx_flat] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}

	//Accumulate samples
	float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
	// Sum of this pixel's per-sample filter weights (see gpu_filter_evaluate()'s
	// own comment) - the final pixel value divides by this instead of
	// samplesPerPixel, matching CPU camera.h's `weighted_color / weight_sum`
	// pbrt-v4 film-reconstruction formula.
	float  weight_sum = 0.0f;
	// Denoiser guide-layer AOVs (see PathTracingPayload::albedo/normal's own
	// comment) - accumulated across samples exactly like pixel_color, only
	// from each sample's depth==0 (primary-ray) hit. Zero-cost when nobody
	// consumes them: still just a few extra float3 adds per sample, and the
	// final write below is skipped entirely when albedoBuffer/normalBuffer
	// are null (denoising not requested for this render).
	float3 albedo_sum = make_float3(0.0f, 0.0f, 0.0f);
	float3 normal_sum = make_float3(0.0f, 0.0f, 0.0f);

	for (unsigned int s = 0; s < params.samplesPerPixel; ++s) {
		// Fresh, independently-hashed seed per sample (pixel, sample index, and
		// frame all mixed in), rather than a single seed mutated in place and
		// carried over from one sample to the next across the whole pixel loop.
		// The latter was a real, confirmed bug: for a high-samples-per-pixel
		// render, chaining pcg_hash() across hundreds of bounces' worth of
		// random_float() calls develops a statistical bias in its low-order
		// bits that most materials don't visibly react to (Lambertian's
		// smoothly-varying cosine-hemisphere sample tolerates it fine), but
		// which systematically skews MaterialType::Principled's threshold-
		// based lobe selection (`u1 < p_diff` / `u1 < p_diff+p_spec`) toward
		// the wrong lobe more and more often as the chain got longer -
		// reproduced directly: at 5 spp a render matched the CPU reference's
		// brightness closely, but at 100+ spp it converged to a much darker,
		// wrong image, and the darkening tracked spp alone (not width or
		// max_depth). Re-deriving the seed here removes the long-chain
		// dependency entirely - each sample's bounce sequence starts from its
		// own independent hash instead of continuing the previous sample's.
		unsigned int seed = pcg_hash((py * params.width + px) * 9781u + s * 6271u + params.frameNumber * 719393u);

		// Halton low-discrepancy pixel offset (pbrt-v4 HaltonSampler pattern)
		// base-2 for x, base-3 for y. Pixel coords (px,py) are mixed into the
		// sample index for per-pixel decorrelation — adjacent pixels use different
		// sub-sequences, avoiding a structured grid artifact across the image.
		// Bounce RNG (seed) keeps using PCG32 for scatter/light directions.
		float hx = halton2(s, px, py);
		float hy = halton3(s, px, py);
		float u = (float(px) + hx) / float(params.width - 1);
		float v = (float(params.height - 1 - py) + hy) / float(params.height - 1);  // Flip Y

		// Sub-pixel offset in [-0.5, 0.5] for the reconstruction filter -
		// same underlying Halton values as u/v above (u/v's own pixel-
		// center-relative math is unaffected by this re-centering: see
		// gpu_filter_evaluate()'s own comment; every filter shape here is
		// an even function in each axis, so the offset's sign convention
		// relative to CPU's own doesn't matter).
		float ox = hx - 0.5f;
		float oy = hy - 0.5f;
		float filter_w = gpu_filter_evaluate(params.camera.filterKind,
			params.camera.filterB, params.camera.filterC,
			params.camera.filterSigma, params.camera.filterTau, ox, oy);
		weight_sum += filter_w;

		// NOTE for future CameraKind additions: this `v` is Y-flipped to a
		// lower-left-origin convention (py=0/top row -> v=1) to match
		// Perspective/Orthographic's `lower_left_corner + u*horizontal +
		// v*vertical` construction. Any new camera whose REFERENCE model
		// (e.g. a pbrt-v4 GenerateRay) assumes raw raster order (v=0 at the
		// top row) needs to locally undo this flip - `float v_x = 1.0f - v;`
		// - before feeding it to that reference formula, or the image comes
		// out vertically mirrored. generate_primary_ray()'s Spherical case
		// (`v_sph`) and sample_realistic_camera_ray()'s Realistic case
		// (`v_raw`), both in optix_device_helpers.h, are two prior instances
		// of this same trap being discovered and fixed independently -
		// check whether it applies before wiring up a new CameraKind rather
		// than rediscovering it a third time. wavefront_kernels.cu's
		// wf_generate_primary_rays kernel applies the identical flip for the
		// same reason.

		// Generate camera ray (perspective/orthographic/spherical/realistic,
		// see generate_primary_ray in optix_device_helpers.h). `cam_weight`
		// is the Realistic lens' cos^4(theta)/pdf vignetting term (1.0 for
		// every other CameraKind) folded into the initial throughput.
		float3 ray_origin, ray_direction;
		float cam_weight;
		generate_primary_ray(u, v, seed, ray_origin, ray_direction, cam_weight);

		// Ray time for motion blur (RTIOW shutter convention: uniform in
		// [0,1), sampled once per pixel-sample and reused for every bounce
		// of this sample) - matches src/TheRestOfYourLife/camera.h's
		// `auto ray_time = random_double();`. Scenes without motion always
		// use 0.0f, which optix_intersection_sphere.h's lerp(center,
		// center1, 0.0f) resolves to `center` exactly regardless of
		// center1's (possibly garbage) value.
		float ray_time = params.motionBlurEnabled ? random_float(seed) : 0.0f;

		// Path tracing loop
		// filter_w is deliberately NOT folded in here (unlike cam_weight,
		// which IS real physical lens throughput) - it's a pure
		// reconstruction/blending weight, unrelated to how much light this
		// path carries, and must never influence a stochastic transport
		// decision. It's applied once, separately, when this sample's
		// radiance is added to pixel_color below - exactly mirroring CPU
		// camera.h's own `sample = sample * camera_weight; ... weighted_color
		// += w * sample;` split (camera_weight early, filter weight late).
		// Folding filter_w in here was a real, confirmed bug: Gaussian's own
		// evaluate() is tiny in absolute magnitude (peaks around 0.06 per
		// axis, ~0.004 for the 2D product, well under 1 even at dead
		// center - see gpu_filter_evaluate()'s own comment) - contaminating
		// throughput with it made Russian Roulette see a near-zero beta from
		// depth 0 onward and kill the overwhelming majority of paths almost
		// immediately. Still mathematically unbiased (RR's 1/(1-q) reweight
		// compensates in expectation), but the variance explosion made any
		// render with real bounce depth (e.g. a heterogeneous medium needing
		// several bounces to random-walk out) converge to near-black at
		// ordinary sample counts.
		float3 throughput = make_float3(cam_weight, cam_weight, cam_weight);
		float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
		// pbrt-v4 dispersion (Cauchy formula), recursive backend's simplified
		// 3-representative-wavelength scheme - see shade_material()'s own
		// inout_rgb_channel parameter comment (optix_device_helpers.h) for
		// the full rationale. Fresh per SAMPLE (like `seed` above - each
		// sample independently rolls its own channel if/when it hits a
		// dispersive Dielectric), persists across every BOUNCE of this one
		// sample via payload register p24 (packed/unpacked each trace call
		// below, same convention as `throughput`/`radiance` themselves).
		unsigned int rgbChannel = kRgbChannelUnset;
		float  prev_brdf_pdf = 0.0f;  // BRDF PDF of the ray that arrived at this bounce (0 = primary)
		// pbrt-v4 etaScale: product of eta^2 over every transmission event
		// so far - see PathTracingPayload::eta's own comment.
		float  eta_scale = 1.0f;
		// pbrt-v4 anyNonSpecularBounces: true once any bounce so far in this
		// path was NOT a specular/delta event - tracked unconditionally
		// regardless of Integrator "bool regularize"'s value (only the
		// alpha-WIDENING at shade_material()'s 4 rough-material call sites
		// is gated on regularize && this flag - see that function's own
		// do_regularize parameter comment), matching CPU camera.h's
		// any_nonspecular and GPU-wavefront's h.any_nonspecular exactly.
		// scatter_brdf_pdf > 0.0f (below) is this codebase's own existing
		// proxy for "was the arriving bounce non-specular" - the same
		// convention CPU's prev_bsdf_pdf==0 <=> specular already relies on,
		// so no new is_specular signal needs threading out of the closest-
		// hit programs just for this.
		bool   any_nonspecular = false;
		// A MaterialType::Interface crossing (flag==4 below) is free - it
		// doesn't consume the depth loop or an RR trial - so nothing else
		// bounds how many a single path can take. kMaxMediumBoundaryCrossings
		// (src/shared/cpu_gpu.h) is the one shared bound every integrator
		// that supports this uses.
		int mediumBoundaryCrossings = 0;

		for (unsigned int depth = 0; depth < params.maxDepth; ++depth) {
			// --stats: one traced ray per iteration (primary on depth==0, a
			// bounce continuation after) - see optix_types.h's
			// LaunchParams::statsBounceRays own comment. Null unless --stats
			// was requested (optix_renderer_render.cpp), so this is a single
			// pointer-null check, not an atomic, on every other render.
			if (params.statsBounceRays) atomicAdd(params.statsBounceRays, 1ull);

			// Initialize payload
			PathTracingPayload payload;
			payload.attenuation = throughput;
			payload.emission = make_float3(0.0f, 0.0f, 0.0f);
			payload.seed = seed;
			payload.depth = depth;
			payload.scattered = false;

			// Trace ray - pack 16 payload registers (13 original + p13-p15,
			// added for MaterialType::Subsurface's explicit next-ray-origin
			// override - see optix_types.h's RAY_TYPE_PROBE comment and
			// shade_material()'s out_bssrdf_exit/out_bssrdf_exit_pos).
			unsigned int p0 = __float_as_uint(payload.attenuation.x);
			unsigned int p1 = __float_as_uint(payload.attenuation.y);
			unsigned int p2 = __float_as_uint(payload.attenuation.z);
			unsigned int p3 = 0;  // emission (will be set by hit/miss)
			unsigned int p4 = 0;
			unsigned int p5 = 0;
			unsigned int p6 = 0;  // scatter direction
			unsigned int p7 = 0;
			unsigned int p8 = 0;
			unsigned int p9 = payload.seed;
			unsigned int p10 = 0;  // scattered flag
			unsigned int p11 = 0;  // hit distance 't'
			// INPUT for __miss__ms (see its own comment): prev_brdf_pdf, the BRDF
			// PDF of the ray that arrives at this trace - 0 for the primary ray or
			// a specular bounce (CPU's prev_bsdf_pdf convention exactly), non-zero
			// otherwise. Every closest-hit program only ever WRITES p12 (never
			// reads it - see their own optixSetPayload_12 calls), so this incoming
			// value is only ever consumed by the miss program; hit programs then
			// overwrite it with their own OUTPUT meaning (brdf_pdf of the new
			// scatter direction, or the NEE light pdf for a hit_light result).
			unsigned int p12 = __float_as_uint(prev_brdf_pdf);
			// p13-p15: explicit next-ray origin, only written (and only
			// meaningful) when flag==3 below.
			unsigned int p13 = 0, p14 = 0, p15 = 0;
			// p16-p21: denoiser guide-layer AOVs (albedo.xyz, normal.xyz) -
			// see PathTracingPayload::albedo/normal's own comment. Every
			// closest-hit/miss program writes these unconditionally
			// (regardless of depth - a hit program has no way to know which
			// bounce it's shading), so init to 0 here is just a safety
			// default, not something any program is expected to leave
			// untouched.
			unsigned int p16 = 0, p17 = 0, p18 = 0, p19 = 0, p20 = 0, p21 = 0;
			// p22: eta - see PathTracingPayload::eta's own comment. Initialized
			// to 1.0f (a no-op multiplier) here, the same "input survives if no
			// program writes it" convention p12 already relies on for
			// __miss__ms - only the closest-hit branches that produce a real
			// transmission event ever call optixSetPayload_22.
			unsigned int p22 = __float_as_uint(1.0f);
			// p23: anyNonSpecularBounces-so-far, an INPUT to the closest-hit
			// programs (read via optixGetPayload_23() to compute
			// do_regularize for shade_material() - see that function's own
			// parameter comment), same "closest-hit only WRITES p12"-style
			// convention as prev_brdf_pdf but mirrored: this register is
			// only ever READ by closest-hit programs, never written by
			// them, so it isn't part of the "unpack payload" section below.
			unsigned int p23 = any_nonspecular ? 1u : 0u;
			// p24: rgbChannel - see this function's own rgbChannel comment
			// above and shade_material()'s inout_rgb_channel parameter
			// comment. Genuinely IN/OUT (unlike p22/p23): every closest-hit
			// program reads the incoming value via optixGetPayload_24()
			// AND writes it back (unchanged if this bounce wasn't the
			// path's first dispersive hit), so it persists correctly.
			unsigned int p24 = rgbChannel;

			optixTrace(
				params.traversable,     // Acceleration structure
				ray_origin,             // Ray origin
				ray_direction,          // Ray direction
				0.001f,                 // tmin
				1e16f,                  // tmax
				ray_time,               // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				RAY_TYPE_RADIANCE,      // SBT offset
				RAY_TYPE_COUNT,         // SBT stride
				RAY_TYPE_RADIANCE,      // missSBTIndex
				p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
				p16, p17, p18, p19, p20, p21, p22, p23, p24
			);

			// Unpack payload (16 registers)
			payload.attenuation.x = __uint_as_float(p0);
			payload.attenuation.y = __uint_as_float(p1);
			payload.attenuation.z = __uint_as_float(p2);
			payload.emission.x = __uint_as_float(p3);  // Emission from this hit
			payload.emission.y = __uint_as_float(p4);
			payload.emission.z = __uint_as_float(p5);
			payload.scatterDir.x = __uint_as_float(p6);
			payload.scatterDir.y = __uint_as_float(p7);
			payload.scatterDir.z = __uint_as_float(p8);
			payload.seed = p9;
			unsigned int flag = p10;
			// bit 3 (value 8) of p10 carries shade_material()'s real
			// is_specular for this bounce - see pack_scatter_flag()'s own
			// comment (optix_device_helpers.h) for why this rides along in
			// the same register rather than being re-derived from
			// scatter_brdf_pdf==0.0f below (a proxy that a legitimately
			// non-specular but numerically-underflowed-to-zero pdf could
			// misclassify - matches CPU camera.h/GPU-wavefront's own use of
			// a real is_specular boolean rather than a derived one). Masked
			// back off immediately so every existing `flag == N` comparison
			// below is unaffected - base values (0/1/2/3/4) are all < 8.
			bool bounce_is_specular = (flag & 8u) != 0u;
			flag &= 7u;
			float t_hit = __uint_as_float(p11);
			float scatter_brdf_pdf = __uint_as_float(p12);  // BRDF PDF of the new scatter direction
			// Explicit next-ray origin (flag==3 only - MaterialType::
			// Subsurface's probe-walk exit point, found off to the side of
			// this ray, which `ray_origin + t_hit*ray_direction` below
			// cannot represent).
			float3 explicit_origin = make_float3(__uint_as_float(p13), __uint_as_float(p14), __uint_as_float(p15));
			payload.albedo.x = __uint_as_float(p16);
			payload.albedo.y = __uint_as_float(p17);
			payload.albedo.z = __uint_as_float(p18);
			payload.normal.x = __uint_as_float(p19);
			payload.normal.y = __uint_as_float(p20);
			payload.normal.z = __uint_as_float(p21);
			payload.eta = __uint_as_float(p22);

			// Dispersion (see this function's own rgbChannel comment above):
			// if THIS bounce is the path's first-ever dispersive hit (was
			// unset going in, a closest-hit program just chose one), confine
			// `throughput` to that one RGB channel with a compensating 3x
			// weight - a standard stochastic-channel-selection estimator
			// (unbiased: each of the 3 equally-likely channels, averaged
			// over many samples, reconstructs the full-RGB expectation).
			// Applying this to `throughput` itself (rather than each
			// `radiance +=` site separately) is sufficient and correct: it's
			// a ONE-TIME multiply that every later radiance contribution
			// this sample naturally inherits through throughput's own
			// ongoing multiplication chain, and a path that never hits a
			// dispersive material (rgbChannel stays kRgbChannelUnset for
			// the whole sample) never pays this at all - zero extra
			// variance for every non-dispersive scene/path.
			unsigned int newRgbChannel = p24;
			if (rgbChannel == kRgbChannelUnset && newRgbChannel != kRgbChannelUnset) {
				float3 channelMask = make_float3(newRgbChannel == 0u ? 3.0f : 0.0f,
				                                  newRgbChannel == 1u ? 3.0f : 0.0f,
				                                  newRgbChannel == 2u ? 3.0f : 0.0f);
				throughput = throughput * channelMask;
			}
			rgbChannel = newRgbChannel;

			// Denoiser guide-layer AOVs only matter on the primary ray -
			// capture this sample's depth==0 hit regardless of which branch
			// (scattered/hit_light/absorbed) it takes below, same "primary
			// ray only" scope as the CPU/wavefront ray-differential feature.
			// Gated on params.albedoBuffer (matches pack_aov_payload()'s own
			// gate, optix_device_helpers.h) so the common non-denoised path
			// skips this add - payload.albedo/normal are whatever p16-p21
			// happened to default to (0, per their init above) when the hit/
			// miss program skipped packing them, so summing would be a
			// harmless no-op anyway, but skipping it avoids the wasted work.
			if (depth == 0 && params.albedoBuffer) {
				albedo_sum = albedo_sum + payload.albedo;
				normal_sum = normal_sum + payload.normal;
			}

			// Decode flag: 0=absorbed, 1=scattered, 2=hit_light, 3=scattered
			// w/ explicit origin (Subsurface probe exit), 4=medium boundary
			// (MaterialType::Interface - real pass-through, see its own
			// comment in optix_types.h).
			if (flag == 4) {
				// True pass-through - nothing actually scattered here.
				// prev_brdf_pdf/eta_scale are left exactly as they were (the
				// NEXT emitter hit's MIS weight should reflect whatever the
				// last REAL vertex was, not this crossing), and this doesn't
				// consume the depth loop or an RR trial either - a free
				// crossing, matching CPU camera.h's own is_medium_boundary
				// branch exactly.
				throughput = throughput * payload.attenuation;
				float3 hit_point = ray_origin + t_hit * ray_direction;
				// 0.01f, not 0.001f: matches the normal-scatter continuation
				// ray's own offset below - a smaller epsilon here previously
				// caused reproducible self-intersection/illegal-memory-access
				// crashes on dense geometry elsewhere in this codebase (see
				// wavefront_kernels.cu's shadow-ray epsilon comment), so this
				// new pass-through ray uses the same, already-fixed value
				// rather than reintroducing a smaller one.
				ray_origin = hit_point + 0.01f * ray_direction;
				// ray_direction is left unchanged - real pass-through.
				seed = payload.seed;
				if (++mediumBoundaryCrossings > kMaxMediumBoundaryCrossings) break;
				--depth;
				continue;
			} else if (flag == 2) {
				// Hit an emissive surface via a BRDF-sampled bounce.
				// Apply MIS weight (pbrt-v4 PathIntegrator pattern):
				//   w_b = PowerHeuristic(p_b, p_l)  where:
				//     p_b = prev_brdf_pdf  (BRDF PDF of the direction that arrived here)
				//     p_l = p12            (NEE selection*geometry PDF for this light+direction)
				// Special cases: depth==0 (primary ray) or prev_brdf_pdf==0 (specular bounce)
				//   -> add full Le, no MIS.
				float3 Le = payload.emission;
				if (depth > 0 && prev_brdf_pdf > 0.0f &&
					(Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f)) {
					float p_l = scatter_brdf_pdf;  // hit program writes light NEE pdf into p12 for flag==2
					if (p_l > 0.0f) {
						float w_b = mis_power_heuristic(prev_brdf_pdf, p_l);
						radiance = radiance + throughput * w_b * Le;
					} else {
						radiance = radiance + throughput * Le;
					}
				} else {
					radiance = radiance + throughput * Le;
				}
				break;
			} else if (flag == 1 || flag == 3) {
				// Scattered - compute scatter origin and update for next bounce.
				// flag==3: MaterialType::Subsurface's probe walk found an
				// off-ray exit point - the next ray must start THERE, not
				// at `ray_origin + t_hit*ray_direction` (which only ever
				// names a point along the CURRENT ray, exactly like every
				// other material's t_hit convention, including the Medium
				// family's own re-intersection override - see optix_types.h's
				// RAY_TYPE_PROBE comment for why this needed new payload
				// registers rather than reusing that mechanism).

				// Add NEE direct-light emission from this surface hit (already MIS-weighted inside hit program)
				radiance = radiance + throughput * payload.emission;

				float3 hit_point = (flag == 3) ? explicit_origin : (ray_origin + t_hit * ray_direction);
				float3 scatter_origin = hit_point + 0.01f * normalize(payload.scatterDir);

				// Multiply throughput by surface BRDF (attenuation from hit program)
				throughput = throughput * payload.attenuation;

				// pbrt-v4 etaScale: accumulated every bounce (payload.eta is
				// 1.0f, a no-op, unless this hit was a real transmission
				// event - see PathTracingPayload::eta's own comment), not
				// just when RR actually runs below - matches CPU's camera.h
				// (`if (srec.is_transmission) eta_scale *= srec.eta * srec.eta;`
				// runs unconditionally on every bounce, independent of depth).
				eta_scale *= payload.eta * payload.eta;

				// Russian Roulette (pbrt-v4 PathIntegrator pattern)
				// Start after depth > 1 so the primary ray and first bounce
				// always survive. rrBeta = throughput * etaScale (pbrt-v4:
				// keeps RR from killing transmission-heavy/glass paths too
				// aggressively - a path deep inside a chain of refractions
				// has real throughput that raw beta alone underestimates).
				// q = max(0, 1 - MaxComponent(rrBeta)); terminate if rand < q,
				// else reweight (the raw throughput, matching CPU/pbrt-v4:
				// only the SURVIVAL test uses rrBeta, the actual reweight
				// stays in throughput's own units).
				if (depth > 1) {
					float3 rr_beta = throughput * eta_scale;
					float rr_max = fmaxf(rr_beta.x, fmaxf(rr_beta.y, rr_beta.z));
					if (rr_max < 1.0f) {
						float q = fmaxf(0.0f, 1.0f - rr_max);
						if (random_float(seed) < q) break;   // terminate path
						throughput = throughput / (1.0f - q); // unbiased reweight
					}
				}

				// Carry BRDF PDF of the new scatter direction for MIS at the next bounce
				prev_brdf_pdf = scatter_brdf_pdf;
				// pbrt-v4 anyNonSpecularBounces |= !bs->IsSpecular() - uses
				// bounce_is_specular (the real signal unpacked from p10's
				// bit 3 above), not a scatter_brdf_pdf>0 proxy: a
				// legitimately non-specular bounce can still pack an exact
				// 0.0f brdf_pdf at extreme grazing angles (TrowbridgeReitz::D()'s
				// own cos^2(theta)<1e-16 clamp, microfacet.h), which a
				// pdf-based proxy would misclassify as specular - matching
				// CPU camera.h/GPU-wavefront's own use of a real is_specular
				// boolean rather than one derived from pdf. Never reset to
				// false once true, matching CPU/wavefront exactly.
				any_nonspecular = any_nonspecular || !bounce_is_specular;

				ray_origin = scatter_origin;
				ray_direction = normalize(payload.scatterDir);  // MUST normalize!
				seed = payload.seed;
			} else {
				// Absorbed — add any surface emission (e.g. background hit) then stop
				radiance = radiance + throughput * payload.emission;
				break;
			}
		}  // end depth loop

		// filter_w applied here, not folded into throughput - see this
		// sample's own throughput-declaration comment for why.
		pixel_color = pixel_color + filter_w * radiance;
	}  // end sample loop

	// Reconstruction filter: weighted_sum / weight_sum (pbrt-v4 film
	// formula), matching CPU camera.h exactly - see this loop's own
	// weight_sum comment. Guards weight_sum<=0 (a pathological filter
	// parameterization giving every sample zero weight) rather than
	// dividing by zero.
	pixel_color = (weight_sum > 0.0f) ? pixel_color / weight_sum : make_float3(0.0f, 0.0f, 0.0f);

	// Write to framebuffer
	const unsigned int idx_flat = py * params.width + px;
	params.framebuffer[idx_flat] = pixel_color;

	// Denoiser guide-layer AOVs - only written when the caller actually
	// allocated these buffers (denoising requested for this render; see
	// OptiXRenderer::render()'s own alloc site). Skipping the write entirely
	// when null avoids wasted bandwidth on the (default) non-denoise path.
	if (params.albedoBuffer) {
		params.albedoBuffer[idx_flat] = albedo_sum / float(params.samplesPerPixel);
	}
	if (params.normalBuffer) {
		params.normalBuffer[idx_flat] = normal_sum / float(params.samplesPerPixel);
	}
}
