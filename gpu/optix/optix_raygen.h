// optix_raygen.h -- Ray generation program
// Included by optix_programs.cu

extern "C" __global__ void __raygen__rg() {
	// Get pixel coordinates
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	const unsigned int px = idx.x;
	const unsigned int py = idx.y;

	if (px >= params.width || py >= params.height) return;

	// Initialize random seed from pixel + frame
	unsigned int seed = (py * params.width + px) + params.frameNumber * 719393;

	//Accumulate samples
	float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

	for (unsigned int s = 0; s < params.samplesPerPixel; ++s) {
		// Halton low-discrepancy pixel offset (pbrt-v4 HaltonSampler pattern)
		// base-2 for x, base-3 for y. Pixel coords (px,py) are mixed into the
		// sample index for per-pixel decorrelation — adjacent pixels use different
		// sub-sequences, avoiding a structured grid artifact across the image.
		// Bounce RNG (seed) keeps using PCG32 for scatter/light directions.
		float u = (float(px) + halton2(s, px, py)) / float(params.width - 1);
		float v = (float(params.height - 1 - py) + halton3(s, px, py)) / float(params.height - 1);  // Flip Y

		// Generate camera ray (perspective/orthographic/spherical, see
		// generate_primary_ray in optix_device_helpers.h)
		float3 ray_origin, ray_direction;
		generate_primary_ray(u, v, seed, ray_origin, ray_direction);

		// Path tracing loop
		float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
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

			// Trace ray - pack 13 payload registers
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
			unsigned int p12 = 0;  // brdf_pdf of scattered direction (for MIS on next bounce)

			optixTrace(
				params.traversable,     // Acceleration structure
				ray_origin,             // Ray origin
				ray_direction,          // Ray direction
				0.001f,                 // tmin
				1e16f,                  // tmax
				0.0f,                   // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				RAY_TYPE_RADIANCE,      // SBT offset
				RAY_TYPE_COUNT,         // SBT stride
				RAY_TYPE_RADIANCE,      // missSBTIndex
				p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12
			);

			// Unpack payload (13 registers)
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
			} else if (flag == 1) {
				// Scattered - compute scatter origin and update for next bounce

				// Add NEE direct-light emission from this surface hit (already MIS-weighted inside hit program)
				radiance = radiance + throughput * payload.emission;

				float3 hit_point = ray_origin + t_hit * ray_direction;
				float3 scatter_origin = hit_point + 0.01f * normalize(payload.scatterDir);

				// Multiply throughput by surface BRDF (attenuation from hit program)
				throughput = throughput * payload.attenuation;

				// Russian Roulette (pbrt-v4 PathIntegrator pattern)
				// Start after depth > 1 so the primary ray and first bounce always survive.
				// q = max(0, 1 - MaxComponent(throughput)); terminate if rand < q, else reweight.
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
}
