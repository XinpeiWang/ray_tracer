// optix_intersection_sphere.h -- Sphere intersection + closest-hit programs
// Included by optix_programs.cu

extern "C" __global__ void __intersection__sphere() {
	// Get primitive index from OptiX
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch sphere data from device array
	const SphereData& sphere = params.spheres[primIdx];

	// Motion blur: interpolate to this ray's time. lerp(center, center1, 0)
	// == center exactly, so this is a provable no-op for every sphere in
	// every scene that doesn't set motionBlurEnabled (optix_raygen.h always
	// passes rayTime=0.0f in that case) - center1 can hold garbage there and
	// it's still safe. Matches src/TheRestOfYourLife/sphere.h's
	// `current_center = center.at(r.time())` linear interpolation exactly.
	const float ray_time = optixGetRayTime();
	const float3 center = make_float3(
		sphere.center.x + ray_time * (sphere.center1.x - sphere.center.x),
		sphere.center.y + ray_time * (sphere.center1.y - sphere.center.y),
		sphere.center.z + ray_time * (sphere.center1.z - sphere.center.z)
	);

	// Get ray parameters
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	// Sphere intersection
	const float3 oc = ray_orig - center;
	const float a = dot(ray_dir, ray_dir);
	const float half_b = dot(oc, ray_dir);
	const float c = dot(oc, oc) - sphere.radius * sphere.radius;
	const float discriminant = half_b * half_b - a * c;

	if (discriminant < 0.0f) return;  // No hit

	const float sqrtd = sqrtf(discriminant);

	// Find nearest root in valid range
	float root = (-half_b - sqrtd) / a;
	if (root < ray_tmin || root > ray_tmax) {
		root = (-half_b + sqrtd) / a;
		if (root < ray_tmin || root > ray_tmax)
			return;  // No valid hit
	}

	// Report intersection
	optixReportIntersection(
		root,                      // t value
		0,                         // hit kind
		__float_as_int(center.x),  // attribute 0 (time-interpolated center, for normal calc)
		__float_as_int(center.y),  // attribute 1
		__float_as_int(center.z),  // attribute 2
		__float_as_int(sphere.radius)     // attribute 3
	);
}

//==============================================================================
// Sphere Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__sphere() {
	// Get primitive index
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch sphere data from device array
	const SphereData& sphere = params.spheres[primIdx];
	const int matIdx = sphere.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	// Reconstruct sphere data from attributes
	const float3 sphere_center = make_float3(
		__int_as_float(optixGetAttribute_0()),
		__int_as_float(optixGetAttribute_1()),
		__int_as_float(optixGetAttribute_2())
	);
	const float sphere_radius = __int_as_float(optixGetAttribute_3());

	// Get hit point
	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Compute normal
	float3 outward_normal = (hit_point - sphere_center) / sphere_radius;
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	// Sphere UV (only meaningful for a Lambertian material with
	// textureIdx >= 0 - see shade_material()'s uv_u/uv_v params). Matches
	// CPU's get_sphere_uv() (sphere.h:131-144) exactly, using the raw
	// outward_normal (not the front-face-corrected one) since UV mapping
	// is a property of the surface point, independent of which side the
	// ray hit from.
	const float sphere_theta = acosf(-outward_normal.y);
	const float sphere_phi = atan2f(-outward_normal.z, outward_normal.x) + 3.14159265358979323846f;
	const float sphere_uv_u = sphere_phi / (2.0f * 3.14159265358979323846f);
	const float sphere_uv_v = sphere_theta / 3.14159265358979323846f;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material (all materials can emit, most have emission=0)
	float3 emission = mat.emission;

	// Material scattering
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing
	bool is_medium = false;    // MaterialType::Medium: t_hit below must use medium_t_hit,
	float medium_t_hit = 0.0f; // not optixGetRayTmax() (which is just the sphere's entry surface)

	if (mat.type == MaterialType::Medium) {
			// Homogeneous participating medium - see MaterialType::Medium's
			// comment in optix_types.h. Recompute both sphere-intersection
			// roots (t here is only the near/entry root optixGetRayTmax()
			// already reported) to find the exit distance, then sample a
			// free-path distance via Beer-Lambert inversion; if it lands
			// before the exit, scatter via the HG phase function, else pass
			// straight through and continue from the exit point.
			float3 unit_dir = normalize(ray_dir);
			float3 oc2 = ray_orig - sphere_center;
			float half_b2 = dot(oc2, unit_dir);
			float c2 = dot(oc2, oc2) - sphere_radius * sphere_radius;
			float disc2 = fmaxf(0.0f, half_b2 * half_b2 - c2);
			float sq2 = sqrtf(disc2);
			float t_near = fmaxf(0.0f, -half_b2 - sq2);
			float t_far  = -half_b2 + sq2;
			float dist_inside = fmaxf(0.0f, t_far - t_near);

			float sigma_t = mat.ior;
			float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - random_float(seed))) / sigma_t) : 1e30f;

			if (free_path < dist_inside) {
				medium_t_hit = t_near + free_path;
				scattered_dir = sample_henyey_greenstein(unit_dir, mat.fuzz, seed);
				attenuation = mat.albedo;
			} else {
				medium_t_hit = t_far;
				scattered_dir = unit_dir;  // straight through, no interaction
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
			}
			scattered    = true;
			is_specular  = true;  // no NEE/MIS for volume scattering (not yet implemented)
			is_medium    = true;
	} else if (mat.type == MaterialType::Hair) {
			// Marschner/Chiang fiber scattering - see sample_hair_material's
			// comment in optix_device_helpers.h. Matches hair_material.h's
			// skip_pdf=true: no NEE/MIS, res.r/g/b already divides by the
			// sample pdf (same convention as this codebase's other
			// is_specular=true cases).
			scattered   = sample_hair_material(ray_dir, normal, mat, seed, scattered_dir, attenuation);
			is_specular = true;
	} else if (mat.type == MaterialType::DielectricMedium) {
			// Combined dielectric surface + internal medium - see this
			// type's comment in optix_types.h. On entry (front_face) the
			// direct dielectric surface always wins the bounce (matches the
			// CPU's two-hittable trick: the medium's sampled hit distance
			// can never be closer than the entry surface), so just refract/
			// reflect normally. On the following bounce, now travelling
			// inside toward this sphere's exit surface (front_face false),
			// recompute both roots exactly like the Medium branch above to
			// get the remaining distance to the exit, sample a free path,
			// and either scatter via the HG phase function or fall through
			// to a normal exit refraction/reflection at the far surface.
			if (front_face) {
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
				scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
			} else {
				float3 unit_dir = normalize(ray_dir);
				float3 oc2 = ray_orig - sphere_center;
				float half_b2 = dot(oc2, unit_dir);
				float c2 = dot(oc2, oc2) - sphere_radius * sphere_radius;
				float disc2 = fmaxf(0.0f, half_b2 * half_b2 - c2);
				float sq2 = sqrtf(disc2);
				float t_near = fmaxf(0.0f, -half_b2 - sq2);
				float t_far  = -half_b2 + sq2;
				float dist_inside = fmaxf(0.0f, t_far - t_near);

				float sigma_t = mat.eta_c.x;
				float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - random_float(seed))) / sigma_t) : 1e30f;

				if (free_path < dist_inside) {
					medium_t_hit  = t_near + free_path;
					scattered_dir = sample_henyey_greenstein(unit_dir, mat.fuzz, seed);
					attenuation   = mat.albedo;
					is_medium     = true;
				} else {
					attenuation = make_float3(1.0f, 1.0f, 1.0f);
					scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
				}
			}
			scattered   = true;
			is_specular = true;  // specular bounce / no NEE-MIS for volume scattering
	} else if (mat.type == MaterialType::NormalMappedLambertian) {
			// Tangent (dpdu): CPU's sphere.h computes this from the RAW
			// outward_normal (matching sphere_uv_u/v's own convention
			// above), not the front-face-corrected one - cross with world
			// up, fallback to (1,0,0) at the poles where that cross
			// product degenerates.
			float3 dpdu;
			{
				const float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
				const float3 t = cross(world_up, outward_normal);
				const float tlen = length(t);
				dpdu = (tlen > 1e-6f) ? (t / tlen) : make_float3(1.0f, 0.0f, 0.0f);
			}

			// Decode the tangent-space normal from the map texture exactly
			// like CPU's normal_map_material::apply(): 2*RGB-1, normalize
			// (fallback (0,0,1) i.e. "no perturbation" if degenerate).
			const float3 packed = sample_texture(mat.textureIdx, sphere_uv_u, sphere_uv_v, hit_point);
			float ns_x = 2.0f * packed.x - 1.0f;
			float ns_y = 2.0f * packed.y - 1.0f;
			float ns_z = 2.0f * packed.z - 1.0f;
			const float ns_len = sqrtf(ns_x * ns_x + ns_y * ns_y + ns_z * ns_z);
			if (ns_len > 1e-8f) { ns_x /= ns_len; ns_y /= ns_len; ns_z /= ns_len; }
			else                { ns_x = 0.0f; ns_y = 0.0f; ns_z = 1.0f; }

			float out_nx, out_ny, out_nz;
			apply_normal_map(ns_x, ns_y, ns_z, normal.x, normal.y, normal.z,
				dpdu.x, dpdu.y, dpdu.z, out_nx, out_ny, out_nz);
			const float3 perturbed_normal = make_float3(out_nx, out_ny, out_nz);

			// Shade as plain Lambertian (mat.albedo) using the perturbed
			// normal - reuses shade_material()'s existing Lambertian NEE/
			// MIS logic verbatim rather than duplicating it. textureIdx is
			// cleared since it means "normal map" for this material type,
			// not "albedo texture" the way Lambertian itself reads it.
			MaterialData effective = mat;
			effective.type = MaterialType::Lambertian;
			effective.textureIdx = -1;
			shade_material(effective, perturbed_normal, ray_dir, hit_point, front_face, sphere_uv_u, sphere_uv_v, seed,
				attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission);
	} else {
		shade_material(mat, normal, ray_dir, hit_point, front_face, sphere_uv_u, sphere_uv_v, seed,
			attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission);
	}

	// Pack updated payload back into registers

	// p3-p5: emission from this surface hit
	// p6-p8: scatter direction (if scattered)
	// p9: updated seed
	// p10: scattered flag (0=absorbed, 1=scattered, 2=hit_light)
	// p11: hit distance 't'
	// p12: brdf_pdf of scattered dir (flag==1) OR light NEE pdf of incoming dir (flag==2) for MIS

	// Always set emission (for all material types - non-lights have emission=0)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		// Return surface attenuation ONLY (raygen will multiply with throughput)
		// Medium: the scatter/exit point is not the sphere's entry surface
		// (optixGetRayTmax()) - use the distance computed in the Medium
		// case above instead.
		float t_hit = is_medium ? medium_t_hit : optixGetRayTmax();  // Hit distance
		// p12: BRDF PDF for MIS on the next bounce.
		// Specular (Metal/Dielectric): 0 = no MIS (pbrt-v4 specularBounce pattern).
		// Lambertian: cosine_pdf of the sampled direction.
		// NormalizedFresnel: (1-Fr)*cos/(c*pi) stored in brdf_pdf_override.
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));  // Scatter direction
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// p12: NEE PDF for the incoming ray direction reaching this sphere light.
		// This is the solid-angle PDF used by the NEE sampler, enabling MIS in raygen.
		float light_pdf_for_incoming = 0.0f;
		if (params.aliasTable && params.numLights > 0) {
			// Find selection PMF for this sphere in the alias table
			int prim_idx = (int)primIdx;
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_idx && params.isLightSphere[li]) {
					sel_pdf = params.aliasTable[li].pdf;
					break;
				}
			}
			// Solid-angle PDF: 1 / (2*pi*(1 - cos_theta_max))
			float3 to_center = sphere_center - ray_orig;
			float dist_sq = dot(to_center, to_center);
			if (dist_sq > sphere_radius * sphere_radius) {
				float cos_theta_max = sqrtf(1.0f - sphere_radius * sphere_radius / dist_sq);
				float solid_angle = 2.0f * 3.14159265f * (1.0f - cos_theta_max);
				light_pdf_for_incoming = sel_pdf / solid_angle;
			}
		}
		optixSetPayload_10(2);  // hit_light
		optixSetPayload_12(__float_as_uint(light_pdf_for_incoming));
	} else {
		optixSetPayload_10(0);  // absorbed
		optixSetPayload_12(0);
	}
}

//==============================================================================
// Quad Intersection Program
//==============================================================================

