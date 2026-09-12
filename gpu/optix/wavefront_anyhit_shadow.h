// wavefront_anyhit_shadow.h -- Shadow-ray any-hit programs, one per geometry type
// Included by wavefront_programs.cu, after wavefront_common.h.

// __anyhit__wf_shadow_*: one per geometry type (sphere/quad/bilinear_patch/
// triangle), mirroring optix_anyhit_shadow.h's __anyhit__shadow_* exactly -
// see that file's own comments for why DiffuseLight and the transmissive
// materials need special handling, not just "any hit = occluded":
//
//   - MaterialType::DiffuseLight is NOT an occluder - a shadow ray sampled
//     toward a light is EXPECTED to end inside/on that light's own surface,
//     so treating the hit as occlusion would make every area light shadow
//     itself. This one previously used a single generic __anyhit__wf_shadow
//     with no material lookup at all (unconditional occluded=true regardless
//     of what was hit), so any shadow ray whose sampled direction actually
//     reached the light within its tMax read as occluded - which for a
//     sphere light is any point with a large-enough cosine to the light
//     (i.e. exactly the well-lit directions NEE most needs), and for a quad
//     light happened to not manifest because the tMax/origin-offset epsilon
//     interaction (see wavefront_kernels.cu's shadow-ray-setup comment)
//     pushed the effective light-plane distance the other way for a
//     camera-facing quad normal. Confirmed via scene B14 (Measured BRDF
//     showroom, sphere light only): floor and sphere-tops facing the light
//     directly went black under --wavefront while the recursive path
//     rendered them correctly lit.
//   - Dielectric/RoughDielectric/ThinDielectric/DiffuseTransmission let
//     light through unattenuated rather than blocking NEE outright (matches
//     optix_anyhit_shadow.h's own comment).
//   - Medium/CloudMedium/RgbGridMedium/GridMedium/DielectricMedium (sphere/
//     cylinder only - the pbrt loader never assigns these to quad/bilinear-
//     patch/triangle/disk) let light through too, but ATTENUATED: real
//     Beer-Lambert (homogeneous Medium/DielectricMedium) or ratio tracking
//     (heterogeneous Cloud/RgbGrid/Grid) multiplies WfShadowPayload::
//     transmittance down for the chord this shadow ray crosses through the
//     medium, rather than treating it as fully non-occluding - see that
//     field's own comment (wavefront_common.h) for why this replaced the
//     old unconditional optixIgnoreIntersection() for these five types.
extern "C" __global__ void __anyhit__wf_shadow_sphere() {
	const int instBase = wf_instance_base();
	const SphereData& sph = wf_params.spheres[wf_prim_base(instBase) + optixGetPrimitiveIndex()];
	// Resolved to a real, non-Mix material before any mat.type check below -
	// see MaterialType::Mix's own comment (optix_types.h). Matches CPU's
	// mix_material::is_shadow_transmissive(), which delegates to the same
	// deterministic (hashed-hit-point) sub-material pick as scatter().
	const float shadow_t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float3 shadow_hit_point = ray_orig + shadow_t * ray_dir;
	int matIdx = sph.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}

	// Participating media: attenuate the running transmittance instead of
	// the old unconditional pass-through - see WfShadowPayload::
	// transmittance's own comment (wavefront_common.h). Homogeneous Medium/
	// DielectricMedium get deterministic Beer-Lambert over the sphere's own
	// analytic near/far chord; the three heterogeneous kinds get stochastic
	// ratio tracking against their existing GLOBAL majorant (sigma_maj) -
	// same simplification the primary-ray free-path sampling already makes
	// (wavefront_kernels_materials.cu), so shadow and primary-ray results
	// stay mutually consistent.
	if (mat.type == MaterialType::Medium || mat.type == MaterialType::DielectricMedium) {
		const float3 unit_dir = normalize(ray_dir);
		const bool is_box = (sph.shapeKind == GpuMediumShapeKind::Box);
		const float3 sphere_center = wf_read_hit_sphere_center();
		float t_near, t_far;
		wf_medium_sphere_near_far(ray_orig, unit_dir, sph, sphere_center, is_box, t_near, t_far);
		const float sigma_t = (mat.type == MaterialType::Medium)
			? mat.sigma_t : mat.dielectric_medium_extra.sigma_t;
		const float segFar = fminf(t_far, sp->tMax);
		const float segLen = fmaxf(0.0f, segFar - fmaxf(0.0f, t_near));
		sp->transmittance *= expf(-sigma_t * segLen);
		if (sp->transmittance <= 0.0f) { optixTerminateRay(); return; }
		optixIgnoreIntersection();
		return;
	}
	if (mat.type == MaterialType::CloudMedium) {
		const int cloudIdx = (int)mat.cloud_medium_extra.cloudMediumIdx;
		// Bounds check before dereferencing: an unsupported PBRT volume kind
		// (e.g. NanoVDB, which GPU doesn't implement at all) can leave a
		// material tagged with this type but no real backing entry ever
		// uploaded - cloudIdx then stays at its default/sentinel value
		// instead of a valid array index. The OLD unconditional
		// optixIgnoreIntersection() never dereferenced this at all, so this
		// was latent and harmless before; treat it the same way here (no
		// attenuation) rather than reading wf_params.cloudMediums out of
		// bounds (confirmed real: without this guard, RgbGridMedium/
		// GridMedium's identical case crashed CUDA 700 "illegal memory
		// access" on exactly this kind of scene).
		if (cloudIdx < 0 || (unsigned int)cloudIdx >= wf_params.numCloudMediums) {
			optixIgnoreIntersection();
			return;
		}
		const float3 unit_dir = normalize(ray_dir);
		const CloudMedium<float>& cloud = wf_params.cloudMediums[cloudIdx];
		float ray_o3[3] = { ray_orig.x, ray_orig.y, ray_orig.z };
		float ray_d3[3] = { unit_dir.x, unit_dir.y, unit_dir.z };
		auto maj_it = cloud.sample_ray(ray_o3, ray_d3, sp->tMax);
		float segMin, segMax, sigma_maj;
		if (maj_it.next(segMin, segMax, sigma_maj) && sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float t = segMin;
			// Cap matches wavefront_kernels_materials.cu's primary-path
			// CloudMedium branch.
			for (int i = 0; i < 128 && sp->transmittance > 0.0f; ++i) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(sp->seed))) / sigma_maj;
				t += dt;
				if (t >= segMax) break;
				float3 p = ray_orig + t * unit_dir;
				float mx, my, mz;
				cloud.world_to_medium_pt(p.x, p.y, p.z, mx, my, mz);
				float d = gpu_cloud_density(cloud, mx, my, mz);
				// Total extinction, not just scattering (cloud.sample_ray's own
				// sigma_maj = sigma_a+sigma_s too - see cloud_medium.h) - shadow-
				// ray transmittance needs absorption AND out-scattering, unlike
				// the primary path's accept/reject test just above sigma_s alone.
				float sigma_t_local = d * (cloud.sigma_a + cloud.sigma_s);
				sp->transmittance *= 1.0f - sigma_t_local / sigma_maj;
			}
		}
		if (sp->transmittance <= 0.0f) { optixTerminateRay(); return; }
		optixIgnoreIntersection();
		return;
	}
	if (mat.type == MaterialType::RgbGridMedium || mat.type == MaterialType::GridMedium) {
		const bool isRgb = (mat.type == MaterialType::RgbGridMedium);
		const int idx = isRgb ? (int)mat.rgb_grid_medium_extra.rgbGridMediumIdx
							   : (int)mat.grid_medium_extra.gridMediumIdx;
		// Bounds check before dereferencing - see the identical CloudMedium
		// guard above for why this is needed (an unsupported PBRT volume
		// kind can leave this index at a sentinel/default instead of a real
		// one): confirmed real by this exact case crashing CUDA 700 "illegal
		// memory access" before this guard existed, on a scene requesting an
		// unsupported grid-medium variant (NanoVDB) that never actually
		// uploaded a GpuRgbGridMedium/GpuGridMedium entry.
		const unsigned int count = isRgb ? wf_params.numRgbGridMediums : wf_params.numGridMediums;
		if (idx < 0 || (unsigned int)idx >= count) {
			optixIgnoreIntersection();
			return;
		}
		const float3 unit_dir = normalize(ray_dir);
		const float* mat9; const float* translate3; int nx, ny, nz, dataOffset; float sigma_maj, sigma_scale;
		if (isRgb) {
			const GpuRgbGridMedium& g = wf_params.rgbGridMediums[idx];
			mat9 = g.mat; translate3 = g.translate;
			nx = g.nx; ny = g.ny; nz = g.nz; dataOffset = g.dataOffset;
			sigma_maj = g.sigma_maj; sigma_scale = g.sigma_scale;
		} else {
			const GpuGridMedium& g = wf_params.gridMediums[idx];
			mat9 = g.mat; translate3 = g.translate;
			nx = g.nx; ny = g.ny; nz = g.nz; dataOffset = g.dataOffset;
			sigma_maj = g.sigma_maj; sigma_scale = g.sigma_scale;
		}
		// Ray transformed into the medium's own normalized [0,1]^3 space, then
		// a plain box-slab test - mirrors wavefront_kernels_materials.cu's
		// identical RgbGridMedium/GridMedium primary-path setup exactly (same
		// mat/translate transform, same single-GLOBAL-majorant simplification,
		// not CPU's real per-voxel DDA majorant grid).
		float mox = mat9[0]*ray_orig.x + mat9[1]*ray_orig.y + mat9[2]*ray_orig.z + translate3[0];
		float moy = mat9[3]*ray_orig.x + mat9[4]*ray_orig.y + mat9[5]*ray_orig.z + translate3[1];
		float moz = mat9[6]*ray_orig.x + mat9[7]*ray_orig.y + mat9[8]*ray_orig.z + translate3[2];
		float mdx = mat9[0]*unit_dir.x + mat9[1]*unit_dir.y + mat9[2]*unit_dir.z;
		float mdy = mat9[3]*unit_dir.x + mat9[4]*unit_dir.y + mat9[5]*unit_dir.z;
		float mdz = mat9[6]*unit_dir.x + mat9[7]*unit_dir.y + mat9[8]*unit_dir.z;

		float segMin = 0.0f, segMax = sp->tMax;
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

		if (has_seg && sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float tt = segMin;
			const int voxelCount = nx * ny * nz;
			if (isRgb) {
				const float* rData = wf_params.rgbGridData + dataOffset;
				const float* gData = rData + voxelCount;
				const float* bData = gData + voxelCount;
				for (int iter = 0; iter < 128 && sp->transmittance > 0.0f; ++iter) {
					float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(sp->seed))) / sigma_maj;
					tt += dt;
					if (tt >= segMax) break;
					float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
					float dr = gpu_rgb_grid_trilinear(rData, nx, ny, nz, px, py, pz);
					float dg = gpu_rgb_grid_trilinear(gData, nx, ny, nz, px, py, pz);
					float db = gpu_rgb_grid_trilinear(bData, nx, ny, nz, px, py, pz);
					float sr = dr * sigma_scale, sg = dg * sigma_scale, sb = db * sigma_scale;
					// Achromatic max-channel simplification, matching the
					// primary path's own accept/reject probability exactly
					// (wavefront_kernels_materials.cu) - keeps transmittance a
					// single scalar and shadow/primary-ray results consistent.
					float sigma_t_local = fmaxf(sr, fmaxf(sg, sb));
					sp->transmittance *= 1.0f - sigma_t_local / sigma_maj;
				}
			} else {
				const float* data = wf_params.gridData + dataOffset;
				for (int iter = 0; iter < 128 && sp->transmittance > 0.0f; ++iter) {
					float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(sp->seed))) / sigma_maj;
					tt += dt;
					if (tt >= segMax) break;
					float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
					float d = gpu_rgb_grid_trilinear(data, nx, ny, nz, px, py, pz);
					float sigma_t_local = d * sigma_scale;
					sp->transmittance *= 1.0f - sigma_t_local / sigma_maj;
				}
			}
		}
		if (sp->transmittance <= 0.0f) { optixTerminateRay(); return; }
		optixIgnoreIntersection();
		return;
	}

	sp->transmittance = 0.0f;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_quad() {
	const QuadData& quad = wf_params.quads[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = quad.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->transmittance = 0.0f;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_bilinear_patch() {
	const BilinearPatchData& patch = wf_params.bilinearPatches[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = patch.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->transmittance = 0.0f;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_triangle() {
	const int instBase = wf_instance_base();
	const TriangleData& tri = wf_params.triangles[wf_prim_base(instBase) + optixGetPrimitiveIndex()];
	// Mix must resolve before even the alpha-cutout check just below - a Mix
	// wrapper's own alphaMaskTexIdx is always -1, so checking it unresolved
	// would silently skip a sub-material's real alpha cutout. See
	// __anyhit__wf_shadow_sphere's own comment.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = tri.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// Alpha-cutout: a transparent pixel of a leaf/foliage material casts no
	// shadow there - see __anyhit__wf_triangle's own comment for why
	// optixGetAttribute_0/1() (this hit group's own custom-intersection
	// barycentrics), not optixGetTriangleBarycentrics(). No-op for the
	// overwhelming majority of triangles, whose material has no alpha mask.
	if (mat.alphaMaskTexIdx >= 0) {
		// Barycentric UV fallback - see __anyhit__wf_triangle's own comment.
		const float b1 = __int_as_float(optixGetAttribute_0());
		const float b2 = __int_as_float(optixGetAttribute_1());
		const float b0 = 1.0f - b1 - b2;
		float uv_u = b1, uv_v = b2;
		if (tri.hasUVs) {
			uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
			uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
		}
		if (!wf_passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v)) {
			optixIgnoreIntersection();
			return;
		}
	}

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->transmittance = 0.0f;
	optixTerminateRay();
}

// Disk/Cylinder (Phase 4c) - same simple DiffuseLight/dielectric-family/
// opaque list as quad/bilinear-patch above, matching the recursive backend's
// __anyhit__shadow_disk/__anyhit__shadow_cylinder (optix_anyhit_shadow.h)
// exactly - the pbrt loader never assigns Medium/DielectricMedium to a
// disk/cylinder (see pbrt_gpu_builder.h's disk/cylinder loop), so no
// CloudMedium/RgbGridMedium/Medium/DielectricMedium entries are needed here.
extern "C" __global__ void __anyhit__wf_shadow_disk() {
	const DiskData& disk = wf_params.disks[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = disk.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->transmittance = 0.0f;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_cylinder() {
	const CylinderData& cyl = wf_params.cylinders[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float3 shadow_hit_point = ray_orig + shadow_t * ray_dir;
	int matIdx = cyl.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	// MaterialType::Medium (homogeneous only - the pbrt loader never assigns
	// CloudMedium/RgbGridMedium/GridMedium/DielectricMedium to a cylinder):
	// Beer-Lambert over the cylinder's own analytic near/far chord, same
	// treatment as __anyhit__wf_shadow_sphere's identical Medium case.
	if (mat.type == MaterialType::Medium) {
		// ray_dir (raw, not normalize()'d) - matches __closesthit__wf_cylinder's
		// own call into this same helper exactly (wavefront_intersection_
		// disk_cylinder.h); world ray directions reaching this point are
		// already unit length by construction, same invariant that code
		// already relies on for its own t_near/t_far-as-real-distances use.
		float t_near, t_far;
		wf_medium_cylinder_near_far(ray_orig, ray_dir, cyl, t_near, t_far);
		const float segFar = fminf(t_far, sp->tMax);
		const float segLen = fmaxf(0.0f, segFar - fmaxf(0.0f, t_near));
		sp->transmittance *= expf(-mat.sigma_t * segLen);
		if (sp->transmittance <= 0.0f) { optixTerminateRay(); return; }
		optixIgnoreIntersection();
		return;
	}
	sp->transmittance = 0.0f;
	optixTerminateRay();
}

