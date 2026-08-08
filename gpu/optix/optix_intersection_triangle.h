// optix_intersection_triangle.h -- Triangle intersection + closest-hit programs
// Included by optix_programs.cu
//
// Watertight Woop/Moller-Trumbore intersection (pbrt-v4 IntersectTriangle),
// ported from src/TheRestOfYourLife/triangle.h's CPU reference (same
// permute-shear-edge-function algorithm, float here instead of double).
// Flat shading only - the single geometric normal cross(e1,e2), matching
// TriangleData (no per-vertex normal array; scene 37's procedural mesh never
// builds one on the CPU side either, so triangle.h's own per-vertex-normal
// interpolation path is unexercised there too).

extern "C" __global__ void __intersection__triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const TriangleData& tri = params.triangles[primIdx];

	const float3 ro = optixGetWorldRayOrigin();
	const float3 rd = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	// Translate vertices to ray origin
	float3 p0t = tri.p0 - ro;
	float3 p1t = tri.p1 - ro;
	float3 p2t = tri.p2 - ro;

	// Permute: kz = largest abs component of direction
	const float ax = fabsf(rd.x), ay = fabsf(rd.y), az = fabsf(rd.z);
	int kz, kx, ky;
	if (ax >= ay && ax >= az)      { kz = 0; kx = 1; ky = 2; }
	else if (ay >= ax && ay >= az) { kz = 1; kx = 2; ky = 0; }
	else                            { kz = 2; kx = 0; ky = 1; }

	auto comp = [](const float3& v, int k) -> float {
		return k == 0 ? v.x : (k == 1 ? v.y : v.z);
	};
	const float Dx = comp(rd, kx), Dy = comp(rd, ky), Dz = comp(rd, kz);
	float p0x = comp(p0t, kx), p0y = comp(p0t, ky), p0z = comp(p0t, kz);
	float p1x = comp(p1t, kx), p1y = comp(p1t, ky), p1z = comp(p1t, kz);
	float p2x = comp(p2t, kx), p2y = comp(p2t, ky), p2z = comp(p2t, kz);

	// Shear
	const float Sx = -Dx / Dz, Sy = -Dy / Dz, Sz = 1.0f / Dz;
	p0x += Sx * p0z; p0y += Sy * p0z;
	p1x += Sx * p1z; p1y += Sy * p1z;
	p2x += Sx * p2z; p2y += Sy * p2z;

	// Edge function coefficients
	const float e0 = p1x*p2y - p1y*p2x;
	const float e1 = p2x*p0y - p2y*p0x;
	const float e2 = p0x*p1y - p0y*p1x;

	if ((e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) && (e0 > 0.0f || e1 > 0.0f || e2 > 0.0f))
		return;
	const float det = e0 + e1 + e2;
	if (det == 0.0f) return;

	p0z *= Sz; p1z *= Sz; p2z *= Sz;
	const float tScaled = e0*p0z + e1*p1z + e2*p2z;
	if (det < 0.0f && (tScaled >= 0.0f || tScaled < ray_tmax * det)) return;
	if (det > 0.0f && (tScaled <= 0.0f || tScaled > ray_tmax * det)) return;

	const float invDet = 1.0f / det;
	const float t = tScaled * invDet;
	if (t < ray_tmin || t > ray_tmax) return;

	optixReportIntersection(t, 0, 0, 0, 0, 0);
}

//==============================================================================
// Triangle Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const TriangleData& tri = params.triangles[primIdx];
	const MaterialData& mat = params.materials[tri.materialIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	const float3 geom_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
	const bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	const float3 final_normal = front_face ? geom_normal : -geom_normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	float3 emission = mat.emission;

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	float brdf_pdf_override = -1.0f;

	switch (mat.type) {
		case MaterialType::Lambertian: {
			scattered_dir = final_normal + random_unit_vector(seed);
			if (near_zero(scattered_dir)) {
				scattered_dir = final_normal;
			}
			scattered_dir = normalize(scattered_dir);
			attenuation = mat.albedo;
			scattered = true;

			if (params.numLights > 0) {
				int light_idx;
				float selection_pdf;
				if (params.aliasTable) {
					int slot = int(random_float(seed) * float(params.numLights));
					if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
					const GpuAliasEntry& entry = params.aliasTable[slot];
					light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
					selection_pdf = params.aliasTable[light_idx].pdf;
				} else {
					light_idx = int(random_float(seed) * float(params.numLights));
					if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
					selection_pdf = 1.0f / float(params.numLights);
				}

				int prim_idx = params.lightIndices[light_idx];
				bool is_sphere = params.isLightSphere[light_idx];

				float3 to_light;
				float geom_pdf = 0.0f;
				float max_dist = 0.0f;

				if (is_sphere) {
					const SphereData& light_sphere = params.spheres[prim_idx];
					to_light = sample_sphere_light(light_sphere, hit_point, seed, geom_pdf);
					float3 to_center = light_sphere.center - hit_point;
					max_dist = length(to_center);
				} else {
					const QuadData& light_quad = params.quads[prim_idx];
					to_light = sample_quad_light(light_quad, hit_point, seed, geom_pdf, max_dist);
				}

				float light_pdf = selection_pdf * geom_pdf;

				if (light_pdf > 1e-6f) {
					bool visible = trace_shadow_ray(hit_point, to_light, max_dist);

					if (visible) {
						float brdf_pdf = cosine_pdf(to_light, final_normal);
						float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);

						float3 light_emission = make_float3(0.0f, 0.0f, 0.0f);
						if (is_sphere) {
							const MaterialData& light_mat = params.materials[params.spheres[prim_idx].materialIdx];
							light_emission = light_mat.emission;
						} else {
							const MaterialData& light_mat = params.materials[params.quads[prim_idx].materialIdx];
							light_emission = light_mat.emission;
						}

						float cos_theta = fmaxf(0.0f, dot(to_light, final_normal));
						float3 brdf = mat.albedo / 3.14159265358979323846f;
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						emission = emission + direct_light;
					}
				}
			}

			add_punctual_lights_lambertian(hit_point, final_normal, mat.albedo, emission);

			break;
		}

		case MaterialType::Metal: {
			float3 reflected = reflect(normalize(ray_dir), final_normal);
			scattered_dir = normalize(reflected) + mat.fuzz * random_in_unit_sphere(seed);
			if (dot(scattered_dir, final_normal) <= 0.0f)
				scattered_dir = reflected;
			attenuation = mat.albedo;
			scattered = true;
			is_specular = true;
			break;
		}

		case MaterialType::Dielectric: {
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			float ri = front_face ? (1.0f / mat.ior) : mat.ior;
			float3 unit_direction = normalize(ray_dir);
			float cos_theta = fminf(dot(-unit_direction, final_normal), 1.0f);
			float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

			bool cannot_refract = ri * sin_theta > 1.0f;

			if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > random_float(seed)) {
				scattered_dir = reflect(unit_direction, final_normal);
			} else {
				scattered_dir = refract(unit_direction, final_normal, ri);
			}
			scattered = true;
			is_specular = true;
			break;
		}

		case MaterialType::Conductor: {
			float c_alpha = sqrtf(mat.fuzz);
			float3 cn = final_normal;
			float3 cup = (fabsf(cn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 ctan  = normalize(cross(cup, cn));
			float3 cbitan = cross(cn, ctan);

			float3 cwi = normalize(-ray_dir);
			float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
			if (cwi_z <= 0.0f) { scattered = false; break; }

			TrowbridgeReitz<float> c_dist(c_alpha, c_alpha);
			float cwm_x, cwm_y, cwm_z;
			c_dist.Sample_wm(cwi_x, cwi_y, cwi_z,
							 random_float(seed), random_float(seed),
							 cwm_x, cwm_y, cwm_z);

			float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
			float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
			float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
			float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
			if (cwo_z <= 0.0f) { scattered = false; break; }

			float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
			float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
			float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;

			float3 c_F = FrConductorRGB(c_dot,
									   mat.eta_c.x, mat.eta_c.y, mat.eta_c.z,
									   mat.k_c.x,   mat.k_c.y,   mat.k_c.z);
			attenuation = make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight);

			scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::DiffuseLight: {
			scattered = false;
			break;
		}

		default: {
			// Unknown/unported material for triangles - absorb. Only
			// Lambertian/Metal/Dielectric/Conductor/DiffuseLight are needed
			// for scene 37; extend here if a future scene puts another
			// material on a triangle mesh.
			scattered = false;
			break;
		}
	}

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		float t_hit = optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, final_normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// Triangles are never registered as lights by scene_builder.cpp (no
		// scene emits light from a triangle mesh), so this is unreachable in
		// practice - pdf=0 is safe, matching bilinear patch's same note.
		optixSetPayload_10(2);
		optixSetPayload_12(0);
	} else {
		optixSetPayload_10(0);
		optixSetPayload_12(0);
	}
}
