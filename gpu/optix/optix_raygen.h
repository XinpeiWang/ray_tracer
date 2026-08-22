// optix_raygen.h -- Ray generation program
// Included by optix_programs.cu

extern "C" __global__ void __raygen__rg() {
	// Get pixel coordinates
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	const unsigned int px = idx.x;
	const unsigned int py = idx.y;

	if (px >= params.width || py >= params.height) return;

	//Accumulate samples
	float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
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
		float u = (float(px) + halton2(s, px, py)) / float(params.width - 1);
		float v = (float(params.height - 1 - py) + halton3(s, px, py)) / float(params.height - 1);  // Flip Y

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
		float3 throughput = make_float3(cam_weight, cam_weight, cam_weight);
		float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
		float  prev_brdf_pdf = 0.0f;  // BRDF PDF of the ray that arrived at this bounce (0 = primary)

		for (unsigned int depth = 0; depth < params.maxDepth; ++depth) {
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
				p16, p17, p18, p19, p20, p21
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

			// Decode flag: 0=absorbed, 1=scattered, 2=hit_light
			if (flag == 2) {
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

				// Russian Roulette (pbrt-v4 PathIntegrator pattern)
				// Start after depth > 1 so the primary ray and first bounce always survive.
				// q = max(0, 1 - MaxComponent(throughput)); terminate if rand < q, else reweight.
				// Missing pbrt-v4's etaScale correction (CPU's camera.h has
				// it; see that file's own comment) - a per-refraction term
				// that keeps RR from killing transmission-heavy paths too
				// aggressively. Real, disclosed gap (both GPU backends lack
				// it, wavefront_kernels.cu's own RR block has a matching
				// note) - would need new payload registers threaded through
				// every closest-hit program to track, not fixed here.
				if (depth > 1) {
					float rr_max = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
					if (rr_max < 1.0f) {
						float q = fmaxf(0.0f, 1.0f - rr_max);
						if (random_float(seed) < q) break;   // terminate path
						throughput = throughput / (1.0f - q); // unbiased reweight
					}
				}

				// Carry BRDF PDF of the new scatter direction for MIS at the next bounce
				prev_brdf_pdf = scatter_brdf_pdf;

				ray_origin = scatter_origin;
				ray_direction = normalize(payload.scatterDir);  // MUST normalize!
				seed = payload.seed;
			} else {
				// Absorbed — add any surface emission (e.g. background hit) then stop
				radiance = radiance + throughput * payload.emission;
				break;
			}
		}  // end depth loop

		pixel_color = pixel_color + radiance;
	}  // end sample loop

	// Average samples
	pixel_color = pixel_color / float(params.samplesPerPixel);

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
