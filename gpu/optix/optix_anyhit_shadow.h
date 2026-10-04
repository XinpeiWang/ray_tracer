// optix_anyhit_shadow.h -- Shadow any-hit programs
// Included by optix_programs.cu

// Stochastic ratio tracking of a shadow ray through a heterogeneous medium
// (CloudMedium / RgbGridMedium / GridMedium), multiplying the running
// transmittance in the ShadowRayState. Ports wavefront_anyhit_shadow.h's
// identical branches (same single GLOBAL majorant, same achromatic max-channel
// simplification for RgbGrid, same clamped 1 - sigma_t_local/sigma_maj update,
// same 128-step cap), so both GPU backends - and the primary-ray free-path
// sampling they already share - stay mutually consistent. Until this existed
// the recursive backend let every surface-NEE shadow ray through these media
// untouched ("hasn't been ported to this shadow-ray path yet"), so a cloud or
// nebula cast no shadow on anything behind it (E4/E5: recursive 20-28% brighter
// than wavefront in the blocks behind the medium).
__device__ __forceinline__ void shadow_ratio_track_heterogeneous(
	const MaterialData& mat, ShadowRayState* st
) {
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 unit_dir = normalize(optixGetWorldRayDirection());
	const float tMax = st->maxDistance;

	if (mat.type == MaterialType::CloudMedium) {
		const int cloudIdx = (int)mat.cloud_medium_extra.cloudMediumIdx;
		// Bounds check: an unsupported volume kind can leave a material tagged
		// with this type but no uploaded entry (see the wavefront copy).
		if (cloudIdx < 0 || (unsigned int)cloudIdx >= params.numCloudMediums) return;
		const CloudMedium<float>& cloud = params.cloudMediums[cloudIdx];
		float ray_o3[3] = { ray_orig.x, ray_orig.y, ray_orig.z };
		float ray_d3[3] = { unit_dir.x, unit_dir.y, unit_dir.z };
		auto maj_it = cloud.sample_ray(ray_o3, ray_d3, tMax);
		float segMin, segMax, sigma_maj;
		if (maj_it.next(segMin, segMax, sigma_maj) && sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float t = segMin;
			for (int i = 0; i < 128 && st->transmittance > 0.0f; ++i) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - random_float(st->seed))) / sigma_maj;
				t += dt;
				if (t >= segMax) break;
				float3 p = ray_orig + t * unit_dir;
				float mx, my, mz;
				cloud.world_to_medium_pt(p.x, p.y, p.z, mx, my, mz);
				float d = gpu_cloud_density(cloud, mx, my, mz);
				// Total extinction (absorption + out-scattering), unlike the
				// primary path's accept/reject on sigma_s alone.
				float sigma_t_local = d * (cloud.sigma_a + cloud.sigma_s);
				st->transmittance *= fmaxf(0.0f, 1.0f - sigma_t_local / sigma_maj);
			}
		}
		return;
	}

	// RgbGridMedium / GridMedium
	const bool isRgb = (mat.type == MaterialType::RgbGridMedium);
	const int idx = isRgb ? (int)mat.rgb_grid_medium_extra.rgbGridMediumIdx
	                      : (int)mat.grid_medium_extra.gridMediumIdx;
	const unsigned int count = isRgb ? params.numRgbGridMediums : params.numGridMediums;
	if (idx < 0 || (unsigned int)idx >= count) return;
	const float* mat9; const float* translate3;
	int nx, ny, nz, dataOffset; float sigma_maj, sigma_scale;
	if (isRgb) {
		const GpuRgbGridMedium& g = params.rgbGridMediums[idx];
		mat9 = g.mat; translate3 = g.translate;
		nx = g.nx; ny = g.ny; nz = g.nz; dataOffset = g.dataOffset;
		sigma_maj = g.sigma_maj; sigma_scale = g.sigma_scale;
	} else {
		const GpuGridMedium& g = params.gridMediums[idx];
		mat9 = g.mat; translate3 = g.translate;
		nx = g.nx; ny = g.ny; nz = g.nz; dataOffset = g.dataOffset;
		sigma_maj = g.sigma_maj; sigma_scale = g.sigma_scale;
	}
	// Ray into the medium's normalized [0,1]^3 space, then a box-slab test -
	// identical to the recursive primary path's own setup.
	const float mox = mat9[0]*ray_orig.x + mat9[1]*ray_orig.y + mat9[2]*ray_orig.z + translate3[0];
	const float moy = mat9[3]*ray_orig.x + mat9[4]*ray_orig.y + mat9[5]*ray_orig.z + translate3[1];
	const float moz = mat9[6]*ray_orig.x + mat9[7]*ray_orig.y + mat9[8]*ray_orig.z + translate3[2];
	const float mdx = mat9[0]*unit_dir.x + mat9[1]*unit_dir.y + mat9[2]*unit_dir.z;
	const float mdy = mat9[3]*unit_dir.x + mat9[4]*unit_dir.y + mat9[5]*unit_dir.z;
	const float mdz = mat9[6]*unit_dir.x + mat9[7]*unit_dir.y + mat9[8]*unit_dir.z;
	float segMin = 0.0f, segMax = tMax;
	bool has_seg = true;
	{
		float invd, s0, s1;
		invd = (mdx != 0.0f) ? 1.0f/mdx : 1e30f;
		s0 = (0.0f - mox)*invd; s1 = (1.0f - mox)*invd;
		if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
		segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
		if (segMin > segMax) has_seg = false;
		invd = (mdy != 0.0f) ? 1.0f/mdy : 1e30f;
		s0 = (0.0f - moy)*invd; s1 = (1.0f - moy)*invd;
		if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
		segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
		if (segMin > segMax) has_seg = false;
		invd = (mdz != 0.0f) ? 1.0f/mdz : 1e30f;
		s0 = (0.0f - moz)*invd; s1 = (1.0f - moz)*invd;
		if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
		segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
		if (segMin > segMax) has_seg = false;
	}
	if (!has_seg || sigma_maj <= 0.0f) return;
	if (segMin < 0.0f) segMin = 0.0f;
	float tt = segMin;
	const int voxelCount = nx * ny * nz;
	if (isRgb) {
		const float* rData = params.rgbGridData + dataOffset;
		const float* gData = rData + voxelCount;
		const float* bData = gData + voxelCount;
		for (int iter = 0; iter < 128 && st->transmittance > 0.0f; ++iter) {
			float dt = -logf(fmaxf(1e-8f, 1.0f - random_float(st->seed))) / sigma_maj;
			tt += dt;
			if (tt >= segMax) break;
			float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
			float dr = gpu_rgb_grid_trilinear(rData, nx, ny, nz, px, py, pz);
			float dg = gpu_rgb_grid_trilinear(gData, nx, ny, nz, px, py, pz);
			float db = gpu_rgb_grid_trilinear(bData, nx, ny, nz, px, py, pz);
			float sigma_t_local = fmaxf(dr * sigma_scale, fmaxf(dg * sigma_scale, db * sigma_scale));
			st->transmittance *= fmaxf(0.0f, 1.0f - sigma_t_local / sigma_maj);
		}
	} else {
		const float* data = params.gridData + dataOffset;
		for (int iter = 0; iter < 128 && st->transmittance > 0.0f; ++iter) {
			float dt = -logf(fmaxf(1e-8f, 1.0f - random_float(st->seed))) / sigma_maj;
			tt += dt;
			if (tt >= segMax) break;
			float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
			float d = gpu_rgb_grid_trilinear(data, nx, ny, nz, px, py, pz);
			st->transmittance *= fmaxf(0.0f, 1.0f - (d * sigma_scale) / sigma_maj);
		}
	}
}

extern "C" __global__ void __anyhit__shadow_sphere() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	// See optix_intersection_sphere.h: a primitive index is local to its GAS,
	// and the table entry is SIGNED (-1 = not instanced) - casting -1 straight
	// to unsigned would wrap to UINT_MAX instead of falling back to base 0.
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int sphBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const SphereData& sphere = params.spheres[sphBase + primIdx];
	// Resolved to a real, non-Mix material before any mat.type check below -
	// see MaterialType::Mix's own comment (optix_types.h). Matches CPU's
	// mix_material::is_shadow_transmissive(), which delegates to the same
	// deterministic (hashed-hit-point) sub-material pick as scatter().
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = sphere.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// IMPORTANT: When hitting light source, set NOT occluded and terminate
	// This allows the shadow ray to "see" the light
	if (mat.type == MaterialType::DiffuseLight) {
		// Emitter: not an occluder, but must not end traversal as "visible" either
		// (any-hit order is unspecified - see D4 note above __anyhit__shadow_sphere).
		optixIgnoreIntersection();
		return;
	}

	// Homogeneous Medium/DielectricMedium: real Beer-Lambert attenuation of
	// the running shadow-ray transmittance (payload 1), NOT unconditional
	// pass-through - matches CPU's hg_phase_material::shadow_transmittance_
	// impl (constant_medium.h) and GPU-wavefront's identical branch
	// (__anyhit__wf_shadow_sphere, wavefront_anyhit_shadow.h) exactly, using
	// the same analytic near/far chord math. This replaced an earlier
	// unconditional optixIgnoreIntersection() (below, still used for
	// CloudMedium/RgbGridMedium/GridMedium, and for the dielectric-family
	// surface types) that treated these two types as perfectly, losslessly
	// transparent to every shadow ray - confirmed as the actual cause of a
	// real CPU/GPU-recursive/GPU-wavefront 3-way brightness mismatch on B13
	// (Subsurface Slab, MaterialCpuGpuParityTest.
	// BrightnessAndChannelsConsistentAcrossBackends/Scene21_Subsurface_Slab):
	// CPU and GPU-wavefront both already dim a shadow ray by the medium's
	// own extinction; GPU-recursive alone let 100% of a light's contribution
	// straight through the wax slab/jade sphere's own fill, every time
	// medium_phase_nee_mis()'s interior phase-scatter NEE (optix_device_
	// helpers_lighting.h) sampled a shadow ray back out through this same
	// boundary toward a light - measured ~36-42% too bright vs. both other
	// backends (which agreed with each other) before this fix.
	// DielectricMedium (a real glass surface that bounds a medium) is deliberately NOT here: like every pbrt-v4
	// surface with a material it is opaque to NEE shadow rays (VolPathIntegrator::SampleLd), so it falls through to
	// the occluder case below. Light reaches the fog inside only along specular chains, as in pbrt.
	if (mat.type == MaterialType::Medium) {
		const float3 ray_orig = optixGetWorldRayOrigin();
		const float3 ray_dir = optixGetWorldRayDirection();  // unit length by construction (trace_shadow_ray's own callers)
		const bool is_box = (sphere.shapeKind == GpuMediumShapeKind::Box);
		float t_near, t_far;
		if (is_box) {
			float bn, bf;
			box_slab_intersect(ray_orig, ray_dir, sphere.boxMin, sphere.boxMax, bn, bf);
			t_near = fmaxf(0.0f, bn);
			t_far = bf;
		} else {
			// Object-space center/radius via the intersection program's own
			// reported attributes (time-interpolated for a moving sphere) -
			// see __closesthit__sphere's identical read just above, in this
			// same translation unit's own optix_intersection_sphere.h.
			const float3 sphere_center = make_float3(
				__int_as_float(optixGetAttribute_0()),
				__int_as_float(optixGetAttribute_1()),
				__int_as_float(optixGetAttribute_2()));
			const float sphere_radius = __int_as_float(optixGetAttribute_3());
			const float3 oc = ray_orig - sphere_center;
			const float half_b = dot(oc, ray_dir);
			const float c = dot(oc, oc) - sphere_radius * sphere_radius;
			const float disc = fmaxf(0.0f, half_b * half_b - c);
			const float sq = sqrtf(disc);
			t_near = fmaxf(0.0f, -half_b - sq);
			t_far = -half_b + sq;
		}
		const float sigma_t = mat.sigma_t;
		// NOT optixGetRayTmax(): inside an any-hit program that is the CANDIDATE
		// hit's own t (OptiX shrinks tmax to it while the program runs), not the
		// ray's original max_distance. Using it (as this did, wrongly believing
		// it was the original) clipped the chord to [t_near, t_hit] - zero length
		// for a ray entering the sphere from outside, where the reported hit IS
		// the near root - so any shadow ray crossing a medium sphere from outside
		// read T=1 and the sphere cast no shadow at all (B13/E11-with-floor: rec
		// 2.5x too bright under the spheres; only interior-origin rays, where the
		// reported hit is the far root, got a correct chord). The original
		// distance is cached in the ShadowRayState explicitly, the same quantity
		// wavefront's WfShadowPayload::tMax holds.
		const float segFar = fminf(t_far, shadow_state_from_payload()->maxDistance);
		const float segLen = fmaxf(0.0f, segFar - fmaxf(0.0f, t_near));
		float transmittance = shadow_state_from_payload()->transmittance;
		transmittance *= expf(-sigma_t * segLen);
		shadow_state_from_payload()->transmittance = transmittance;  // memory write: survives optixIgnoreIntersection (a payload-register write would not)
		if (transmittance <= 0.0f) {
			shadow_state_from_payload()->occluded = 1;  // fully attenuated - treat as occluded
			optixTerminateRay();
			return;
		}
		optixIgnoreIntersection();  // continue traversal (attenuated, not occluding)
		return;
	}

	// Heterogeneous media (Cloud/RgbGrid/Grid): stochastic ratio tracking of the
	// running transmittance, exactly like the wavefront backend - see
	// shadow_ratio_track_heterogeneous() above. (These used to be an
	// unconditional pass-through, i.e. they cast no shadow at all.)
	if (mat.type == MaterialType::CloudMedium ||
		mat.type == MaterialType::RgbGridMedium ||
		mat.type == MaterialType::GridMedium) {
		ShadowRayState* st = shadow_state_from_payload();
		shadow_ratio_track_heterogeneous(mat, st);
		if (st->transmittance <= 0.0f) {
			st->occluded = 1;  // fully attenuated - treat as occluded
			optixTerminateRay();
			return;
		}
		optixIgnoreIntersection();  // continue traversal (attenuated, not occluding)
		return;
	}

	// Transmissive materials let light through -- ignore them in shadow rays.
	if (mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();  // continue traversal (not an occluder)
		return;
	}

	// For opaque materials, treat as occluder
	shadow_state_from_payload()->occluded = 1;  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

// Shadow any-hit for quads
// For opaque geometry, any hit means occlusion - terminate immediately
extern "C" __global__ void __anyhit__shadow_quad() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = params.quads[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = quad.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// Transmissive materials let light through -- ignore them in shadow rays
	if (mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();  // continue traversal (not an occluder)
		return;
	}

	// For opaque materials, treat as occluder
	shadow_state_from_payload()->occluded = 1;  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

// Shadow any-hit for bilinear patches
extern "C" __global__ void __anyhit__shadow_bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = patch.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	if (mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	shadow_state_from_payload()->occluded = 1;  // occluded = true
	optixTerminateRay();
}

// Shadow any-hit for disks. Same simple (emitters occluding like any other surface,
// dielectric-family ignored, everything else opaque) list as quad/bilinear
// patch above - the pbrt loader never assigns a Medium/DielectricMedium
// material to a disk (see pbrt_gpu_builder.h's disk loop), so no CloudMedium/
// RgbGridMedium/Medium/DielectricMedium entries are needed here.
extern "C" __global__ void __anyhit__shadow_disk() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = params.disks[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = disk.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	if (mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	shadow_state_from_payload()->occluded = 1;  // occluded = true
	optixTerminateRay();
}

// Shadow any-hit for cylinders - same list as disk above.
extern "C" __global__ void __anyhit__shadow_cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = params.cylinders[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = cyl.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// Homogeneous Medium/DielectricMedium: real Beer-Lambert attenuation of
	// the running shadow-ray transmittance (payload 1), same fix and same
	// reasoning as __anyhit__shadow_sphere's identical branch (this file,
	// above) - see that comment for the full B13/Subsurface-Slab root-cause
	// derivation. Cylinder's own near/far chord (tube quadric clipped to a
	// z-slab, ignoring a partial phi sweep - see __closesthit__cylinder's
	// cylinderMediumNearFar() for the full derivation, duplicated here since
	// shadow any-hit and closest-hit are different OptiX programs with no
	// shared local-lambda scope) replaces sphere's quadric-root math; the
	// Beer-Lambert integral itself is identical. MaterialType::Medium was
	// previously ignored unconditionally here too, same gap sphere's own
	// comment describes; DielectricMedium reaching a cylinder at all is new
	// this round (pbrt_gpu_builder.h's cylinder loop now resolves
	// MediumInterface + a smooth dielectric surface via mediumMaterialIndex()).
	// DielectricMedium (a real glass surface that bounds a medium) is deliberately NOT here: like every pbrt-v4
	// surface with a material it is opaque to NEE shadow rays (VolPathIntegrator::SampleLd), so it falls through to
	// the occluder case below. Light reaches the fog inside only along specular chains, as in pbrt.
	if (mat.type == MaterialType::Medium) {
		const float3 ray_orig = optixGetWorldRayOrigin();
		const float3 ray_dir = optixGetWorldRayDirection();  // unit length by construction (trace_shadow_ray's own callers)
		const float3 ro = dc_apply_point(cyl.w2o, ray_orig);
		const float3 rd = dc_apply_vector(cyl.w2o, ray_dir);  // NOT normalised - see optix_disk_cylinder_helpers.h's own convention

		float tube_t0 = -1e30f, tube_t1 = 1e30f;
		bool hasTube;
		if (rd.x == 0.0f && rd.y == 0.0f) {
			hasTube = (double)ro.x * ro.x + (double)ro.y * ro.y <= (double)cyl.radius * (double)cyl.radius;
		} else {
			hasTube = dc_solve_tube_quadratic(ro, rd, cyl.radius, tube_t0, tube_t1);
		}
		float z_t0 = -1e30f, z_t1 = 1e30f;
		bool hasZSlab = true;
		if (rd.z == 0.0f) {
			hasZSlab = (ro.z >= cyl.zMin && ro.z <= cyl.zMax);
		} else {
			float za = (cyl.zMin - ro.z) / rd.z;
			float zb = (cyl.zMax - ro.z) / rd.z;
			z_t0 = fminf(za, zb);
			z_t1 = fmaxf(za, zb);
		}
		float t_near = fmaxf(0.0f, fmaxf(tube_t0, z_t0));
		float t_far  = fminf(tube_t1, z_t1);
		if (!hasTube || !hasZSlab || t_far < t_near) { t_near = 0.0f; t_far = 0.0f; }

		const float sigma_t = mat.sigma_t;
		// Original max_distance, NOT optixGetRayTmax() (the candidate hit's own
		// t inside an any-hit) - see __anyhit__shadow_sphere's identical comment.
		const float segFar = fminf(t_far, shadow_state_from_payload()->maxDistance);
		const float segLen = fmaxf(0.0f, segFar - fmaxf(0.0f, t_near));
		float transmittance = shadow_state_from_payload()->transmittance;
		transmittance *= expf(-sigma_t * segLen);
		shadow_state_from_payload()->transmittance = transmittance;  // memory write: survives optixIgnoreIntersection (a payload-register write would not)
		if (transmittance <= 0.0f) {
			shadow_state_from_payload()->occluded = 1;  // fully attenuated - treat as occluded
			optixTerminateRay();
			return;
		}
		optixIgnoreIntersection();  // continue traversal (attenuated, not occluding)
		return;
	}

	if (mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	shadow_state_from_payload()->occluded = 1;  // occluded = true
	optixTerminateRay();
}

// Shadow any-hit for triangles
extern "C" __global__ void __anyhit__shadow_triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	// See optix_intersection_triangle.h: a primitive index is local to its GAS,
	// and the table entry is SIGNED (-1 = not instanced) - casting -1 straight
	// to unsigned would wrap to UINT_MAX instead of falling back to base 0.
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int triBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const TriangleData& tri = params.triangles[triBase + primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here
	// (before even the alpha-cutout check just below - a Mix wrapper's own
	// alphaMaskTexIdx is always -1, so checking it unresolved would silently
	// skip a sub-material's real alpha cutout).
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = tri.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// Alpha-cutout: a transparent pixel of a leaf/foliage material casts no
	// shadow there, same as any other "not actually here" any-hit case
	// below - checked first since it applies regardless of the material's
	// type otherwise (an alpha-masked material here is always ordinary
	// lambertian, never a light or dielectric, but there's no reason to
	// assume that will always hold). No-op (skips straight to the existing
	// checks) for the overwhelming majority of triangles, whose material
	// has no alpha mask.
	if (mat.alphaMaskTexIdx >= 0) {
		float uv_u = 0.0f, uv_v = 0.0f;
		if (tri.hasUVs) {
			const float2 bary = optixGetTriangleBarycentrics();
			const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;
			uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
			uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
		}
		if (!passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v, shadow_hit_point)) {
			optixIgnoreIntersection();
			return;
		}
	}

	// MaterialType::Subsurface belongs in this list: its entry interface IS a
	// plain dielectric surface (see MaterialType::Subsurface's own comment,
	// shade_material() in optix_device_helpers.h - "the exact same smooth
	// DielectricBxDF sample as MaterialType::Dielectric"), and CPU's
	// equivalent `class subsurface::is_shadow_transmissive()`
	// (material_pbrt.h) returns true with a comment explicitly stating it
	// matches this list - Subsurface's own omission here was a real bug
	// (the CPU code was written assuming GPU parity that didn't actually
	// exist): any NEE shadow ray crossing a Subsurface-shaded mesh (skin,
	// wax, marble - Subsurface is triangle-only in this backend, which is
	// why only this triangle any-hit needs it, not the sphere one above)
	// was wrongly reported fully occluded instead of passing through.
	if (mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Subsurface ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	shadow_state_from_payload()->occluded = 1;  // occluded = true
	optixTerminateRay();
}

//==============================================================================
// Miss Program
//==============================================================================

