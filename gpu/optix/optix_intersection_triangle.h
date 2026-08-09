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

	shade_material(mat, final_normal, ray_dir, hit_point, front_face, seed,
		attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission);

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
