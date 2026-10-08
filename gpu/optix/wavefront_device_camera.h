#pragma once
// wavefront_device_camera.h -- part 2 of 3 of wavefront_device_helpers.h (included by it, in order; not meant to be included on its own).

// ============================================================================
// Kernel 1 — generate_camera_rays
//   Fills rayQueue with one primary ray per pixel for sample index `sampleIdx`.
// ============================================================================

// Realistic multi-element lens camera, wavefront duplicate of
// optix_device_helpers.h's sample_realistic_camera_ray (see that function's
// doc comment for the full algorithm derivation). generate_camera_rays
// below now flips v the same way optix_raygen.h does (py=0/top row -> v=1),
// so - exactly like the recursive path - that flip must be undone here to
// get back to the raw top-to-bottom raster convention
// RealisticCamera::raster_to_film expects.
__device__ __forceinline__ bool wf_sample_realistic_camera_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& out_origin, float3& out_direction, float& out_weight
) {
	out_weight = 0.0f;
	if (cam.numLensElements <= 0 || cam.numExitPupilBounds <= 0) return false;

	float v_raw = 1.0f - v;  // undo generate_camera_rays' lower-left-origin flip
	float pfx = -((2.0f*u - 1.0f) * cam.film_half_x);  // pbrt-v4 negates x
	float pfy =   (2.0f*v_raw - 1.0f) * cam.film_half_y;

	float rFilm = sqrtf(pfx*pfx + pfy*pfy);
	float film_diag = 2.0f * sqrtf(cam.film_half_x*cam.film_half_x + cam.film_half_y*cam.film_half_y);
	int sz = cam.numExitPupilBounds;
	int rIndex = (int)(rFilm / (film_diag*0.5f) * (float)sz);
	if (rIndex >= sz) rIndex = sz - 1;
	if (rIndex < 0) rIndex = 0;
	const GpuExitPupilBounds& b = cam.exitPupilBounds[rIndex];
	if (b.degenerate != 0) return false;

	float area = (b.xMax - b.xMin) * (b.yMax - b.yMin);
	if (area <= 0.0f) return false;
	float ppdf = 1.0f / area;

	float u0 = wf_rand(seed), u1 = wf_rand(seed);
	float lx = b.xMin + u0*(b.xMax - b.xMin);
	float ly = b.yMin + u1*(b.yMax - b.yMin);

	float sinTheta = (rFilm > 0.0f) ? pfy/rFilm : 0.0f;
	float cosTheta0 = (rFilm > 0.0f) ? pfx/rFilm : 1.0f;
	float ppx = cosTheta0*lx - sinTheta*ly;
	float ppy = sinTheta*lx + cosTheta0*ly;
	float ppz = cam.lens_rear_z;

	float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
	float rLen = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);

	float lox = pfx, loy = pfy, loz = 0.0f;
	float ldx = rdx, ldy = rdy, ldz = -rdz;
	float elementZ = 0.0f;

	for (int i = cam.numLensElements - 1; i >= 0; --i) {
		const GpuLensElement& el = cam.lensElements[i];
		elementZ -= el.thickness;
		bool isStop = (el.curvatureRadius == 0.0f);
		float t, nx = 0.0f, ny = 0.0f, nz = 0.0f;

		if (isStop) {
			if (ldz == 0.0f) return false;
			t = (elementZ - loz) / ldz;
			if (t < 0.0f) return false;
		} else {
			float zCenter = elementZ + el.curvatureRadius;
			float cox = lox, coy = loy, coz = loz - zCenter;
			float A = ldx*ldx + ldy*ldy + ldz*ldz;
			float B = 2.0f*(ldx*cox + ldy*coy + ldz*coz);
			float C = cox*cox + coy*coy + coz*coz - el.curvatureRadius*el.curvatureRadius;
			float disc = B*B - 4.0f*A*C;
			if (disc < 0.0f) return false;
			float sq = sqrtf(disc);
			float q = (B < 0.0f) ? -0.5f*(B - sq) : -0.5f*(B + sq);
			float t0 = q / A;
			float t1 = C / q;
			if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
			bool useCloserT = (ldz > 0.0f) != (el.curvatureRadius < 0.0f);
			t = useCloserT ? fminf(t0, t1) : fmaxf(t0, t1);
			if (t < 0.0f) return false;
			float hx0 = lox+t*ldx, hy0 = loy+t*ldy, hz0 = loz+t*ldz;
			nx = hx0; ny = hy0; nz = hz0 - zCenter;
			float nlen = sqrtf(nx*nx+ny*ny+nz*nz);
			if (nlen == 0.0f) return false;
			nx/=nlen; ny/=nlen; nz/=nlen;
			if (ldx*nx+ldy*ny+ldz*nz > 0.0f) { nx=-nx; ny=-ny; nz=-nz; }
		}

		float hx = lox+t*ldx, hy = loy+t*ldy, hz = loz+t*ldz;
		if (hx*hx + hy*hy > el.apertureRadius*el.apertureRadius) return false;
		lox = hx; loy = hy; loz = hz;

		if (!isStop) {
			float eta_i = (el.eta == 0.0f) ? 1.0f : el.eta;
			float eta_t = (i > 0 && cam.lensElements[i-1].eta != 0.0f) ? cam.lensElements[i-1].eta : 1.0f;
			float len = sqrtf(ldx*ldx+ldy*ldy+ldz*ldz);
			float dxn = ldx/len, dyn = ldy/len, dzn = ldz/len;
			float eta = eta_i/eta_t;
			float cosI = -(dxn*nx+dyn*ny+dzn*nz);
			float sin2T = eta*eta * fmaxf(0.0f, 1.0f - cosI*cosI);
			if (sin2T >= 1.0f) return false;
			float cosT = sqrtf(1.0f - sin2T);
			ldx = eta*dxn + (eta*cosI - cosT)*nx;
			ldy = eta*dyn + (eta*cosI - cosT)*ny;
			ldz = eta*dzn + (eta*cosI - cosT)*nz;
		}
	}

	float lensOutOx = lox, lensOutOy = loy, lensOutOz = -loz;
	float lensOutDx = ldx, lensOutDy = ldy, lensOutDz = -ldz;

	float cosThetaW = (rLen > 0.0f) ? fabsf(rdz/rLen) : 0.0f;
	float lrz = cam.lens_rear_z;
	if (lrz <= 0.0f) return false;
	float w = (cosThetaW*cosThetaW*cosThetaW*cosThetaW) / (ppdf * lrz * lrz);

	out_origin = cam.origin + lensOutOx*cam.su + lensOutOy*cam.sv + lensOutOz*cam.sw;
	out_direction = normalize(lensOutDx*cam.su + lensOutDy*cam.sv + lensOutDz*cam.sw);
	out_weight = w;
	return true;
}

// Generate a primary camera ray, dispatching on camera.kind. Duplicated from
// optix_device_helpers.h's generate_primary_ray (with the wf_ prefix)
// rather than shared, matching this file's existing pattern of not sharing
// device helpers with the recursive path (this kernel has no access to the
// recursive path's __constant__ params global). `weight` applies to
// Realistic only (1.0 for every other CameraKind).
__device__ __forceinline__ void wf_generate_primary_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& origin, float3& direction, float& weight
) {
	weight = 1.0f;

	if (cam.animated) {
		// Real per-ray shutter-time camera motion blur - see
		// optix_device_helpers.h's generate_primary_ray for the derivation
		// this mirrors.
		float3 local_pixel_sample = cam.localLowerLeftCorner + u * cam.localHorizontal + v * cam.localVertical;
		float3 local_origin = make_float3(0.0f, 0.0f, 0.0f);
		bool hasDOF = (cam.localDefocusDiskU.x != 0.0f || cam.localDefocusDiskU.y != 0.0f || cam.localDefocusDiskU.z != 0.0f ||
					   cam.localDefocusDiskV.x != 0.0f || cam.localDefocusDiskV.y != 0.0f || cam.localDefocusDiskV.z != 0.0f);
		if (hasDOF) {
			float rx = 2.0f * wf_rand(seed) - 1.0f, ry = 2.0f * wf_rand(seed) - 1.0f;
			while (rx*rx + ry*ry >= 1.0f) { rx = 2.0f * wf_rand(seed) - 1.0f; ry = 2.0f * wf_rand(seed) - 1.0f; }
			local_origin = rx * cam.localDefocusDiskU + ry * cam.localDefocusDiskV;
		}
		float3 local_direction = local_pixel_sample - local_origin;
		float dt = wf_rand(seed);
		// Slerp + quaternion-to-matrix built ONCE (gpu_camera_anim_rotation,
		// camera_motion_blur_device.h - shared with optix_device_helpers.h,
		// since neither it nor gpu_camera_anim_apply touch this kernel's
		// own state), then applied to both origin and direction.
		GpuAnimRotMat rot = gpu_camera_anim_rotation(cam.animR0, cam.animR1, dt);
		origin = gpu_camera_anim_apply(rot, local_origin, true, cam.animT0, cam.animT1, dt);
		// local_direction is not unit length and rotation preserves length,
		// so this needs an explicit normalize - every other camera branch
		// in this function returns a unit direction too.
		direction = normalize(gpu_camera_anim_apply(rot, local_direction, false, cam.animT0, cam.animT1, dt));
		return;
	}

	switch (cam.kind) {
		case CameraKind::Orthographic: {
			origin = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			direction = cam.w;
			break;
		}
		case CameraKind::Spherical: {
			// Both mappings finish with a swap(dir.y, dir.z) folded directly
			// into which raw component feeds ly vs lz below - see
			// optix_device_helpers.h's generate_primary_ray for the same
			// pattern (including why v_sph below undoes optix_raygen.h's
			// shared Y-flip) and src/shared/cameras.h for the CPU reference.
			const float v_sph = 1.0f - v;
			float lx, ly, lz;
			if (cam.sphericalMapping == 1) {  // EqualArea
				double ud = (double)u, vd = (double)v_sph;
				wf_wrap_equal_area_square(ud, vd);
				double ewx, ewy, ewz;
				wf_equal_area_square_to_sphere(ud, vd, ewx, ewy, ewz);
				lx = (float)ewx;
				ly = (float)ewz;  // swap(wy,wz): final y = raw z
				lz = (float)ewy;  // swap(wy,wz): final z = raw y
			} else {  // EquiRectangular
				float theta = 3.14159265358979323846f * v_sph;
				float phi   = 2.0f * 3.14159265358979323846f * u;
				float sin_t = sinf(theta), cos_t = cosf(theta);
				lx = sin_t * cosf(phi);
				ly = cos_t;
				lz = sin_t * sinf(phi);
			}
			origin = cam.origin;
			direction = normalize(lx * cam.su + ly * cam.sv + lz * cam.sw);
			break;
		}
		case CameraKind::Realistic: {
			if (!wf_sample_realistic_camera_ray(cam, u, v, seed, origin, direction, weight)) {
				origin = cam.origin;
				direction = cam.sw;
				weight = 0.0f;
			}
			break;
		}
		default: { // Perspective, optionally thin-lens DOF
			float3 pixel_sample = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			bool hasDOF = (cam.defocus_disk_u.x != 0.0f || cam.defocus_disk_u.y != 0.0f || cam.defocus_disk_u.z != 0.0f ||
						   cam.defocus_disk_v.x != 0.0f || cam.defocus_disk_v.y != 0.0f || cam.defocus_disk_v.z != 0.0f);
			if (hasDOF) {
				float rx = 2.0f * wf_rand(seed) - 1.0f, ry = 2.0f * wf_rand(seed) - 1.0f;
				while (rx*rx + ry*ry >= 1.0f) { rx = 2.0f * wf_rand(seed) - 1.0f; ry = 2.0f * wf_rand(seed) - 1.0f; }
				origin = cam.origin + rx * cam.defocus_disk_u + ry * cam.defocus_disk_v;
			} else {
				origin = cam.origin;
			}
			direction = normalize(pixel_sample - origin);
			break;
		}
	}
}

// Per-shading-point glossy-frame/path-guiding context, shared by
// wf_finish_material_scatter()'s evalGlossyF lambda and its own
// glossy_isType-gated matType switch further down - see
// wf_setup_glossy_context()'s own comment for why this is computed once
// per shading point rather than once per (up to 3) NEE query direction.
struct GlossyCtx {
	bool valid = false;
	float3 tan = make_float3(0.0f, 0.0f, 0.0f);
	float3 bit = make_float3(0.0f, 0.0f, 0.0f);
	float wi_x = 0.0f, wi_y = 0.0f, wi_z = 0.0f;
	float alpha = 0.0f;
	float alpha_v = 0.0f;
	float pGuide = 0.0f;
	int guideProbeIdx = -1;
};

// Extracted out of wf_finish_material_scatter() (a code-health pass - that
// function had grown to ~1500 lines) as the first, lowest-risk of several
// planned extractions: no early returns, no lambda captures to convert,
// pure input->output. Computed ONCE per shading point instead of on each
// of the (up to) 3 calls evalGlossyF gets (area-light NEE, sky NEE,
// punctual-light NEE) - normal/phaseWo/matType are all invariant across
// those calls within one wf_finish_material_scatter() invocation, so
// re-deriving the local frame/wi/guiding lookup on every call would redo
// identical work up to 3x for no reason. `valid` folds both of
// evalGlossyF's own early-outs (wrong matType, wi_z<=0) into one check the
// caller no longer needs to redo itself.
CPU_GPU GlossyCtx wf_setup_glossy_context(
	MaterialType matType, const float3& normal, const float3& dpdu, const float3& phaseWo,
	// See wf_finish_material_scatter()'s own glossyAlpha/glossyAlphaV
	// parameter comments - already regularized/resolved by the caller, not
	// re-derived here. alpha_v <0 (RoughMetal, or any non-glossy matType)
	// falls back to alpha, matching MaterialData::roughnessV's own
	// isotropic sentinel.
	float glossyAlpha, float glossyAlphaV,
	GpuProbeGridMeta guidingGridMeta, const GpuGuidingHistogram* guidingHistograms, const GpuProbe* guidingProbes,
	const float3& hit_point) {
	GlossyCtx ctx;
	ctx.alpha = glossyAlpha;
	ctx.alpha_v = (glossyAlphaV >= 0.0f) ? glossyAlphaV : ctx.alpha;
	const bool glossy_isType = (matType == MaterialType::Conductor || matType == MaterialType::RoughDielectric ||
		matType == MaterialType::CoatedDiffuse || matType == MaterialType::CoatedConductor ||
		matType == MaterialType::RoughMetal || matType == MaterialType::Measured);
	if (glossy_isType) {
		// RoughMetal stays on the arbitrary frame (isotropic-only, no
		// anisotropic variant exists - matches optix_device_helpers.h's
		// identical RoughMetal exclusion); the other 4 glossy kinds get the
		// real, UV-aligned frame.
		if (matType == MaterialType::RoughMetal) {
			BuildArbitraryTangentFrame(normal.x, normal.y, normal.z,
			                            ctx.tan.x, ctx.tan.y, ctx.tan.z,
			                            ctx.bit.x, ctx.bit.y, ctx.bit.z);
		} else {
			BuildDpduTangentFrame(normal.x, normal.y, normal.z, dpdu.x, dpdu.y, dpdu.z,
			                       ctx.tan.x, ctx.tan.y, ctx.tan.z,
			                       ctx.bit.x, ctx.bit.y, ctx.bit.z);
		}
		ctx.wi_x = dot(phaseWo, ctx.tan);
		ctx.wi_y = dot(phaseWo, ctx.bit);
		ctx.wi_z = dot(phaseWo, normal);
		if (ctx.wi_z > 0.0f) ctx.valid = true;
	}

	// Real-time path guiding (Live Preview only) - v1 scope is Conductor/
	// RoughMetal only (see guidingHistograms's own parameter comment,
	// wf_finish_material_scatter()), computed ONCE here rather than inside
	// evalGlossyF since the nearest probe/pGuide depend only on hit_point,
	// not on which of the (up to 3) query directions evalGlossyF is asked
	// about. Reaching here with is_specular==false already means
	// evaluate_materials() itself found this Conductor/RoughMetal hit
	// non-EffectivelySmooth (a smooth one sets is_specular=true and never
	// reaches this whole NEE block) - so no separate EffectivelySmooth
	// re-check is needed, matching that switch-arm's own guiding gate
	// exactly.
	if (ctx.valid && guidingHistograms != nullptr &&
		(matType == MaterialType::Conductor || matType == MaterialType::RoughMetal)) {
		ctx.guideProbeIdx = wf_guiding_nearest_probe(guidingGridMeta, guidingProbes, hit_point);
		if (ctx.guideProbeIdx >= 0) {
			ctx.pGuide = wf_guiding_probability(guidingHistograms[ctx.guideProbeIdx]);
		}
	}
	return ctx;
}

// Result of wf_nee_pick_light() below - the ReSTIR-or-classic light draw
// wf_finish_material_scatter()'s area-light NEE block consumes. Field names/
// defaults mirror exactly what used to be plain locals inline in that
// function: haveSample/to_light/max_dist/light_pdf/nee_norm/
// light_emission_spec, initialized the same "no sample yet" way either path
// left them in before either filled them in (or didn't, on a failed draw).
struct NeeLightSample {
	bool haveSample = false;
	float3 to_light = make_float3(0.0f, 0.0f, 0.0f);
	float max_dist = 0.0f;
	float light_pdf = 0.0f;
	float nee_norm = 0.0f;
	SampledSpectrum<kWFNWavelengths> light_emission_spec = SampledSpectrum<kWFNWavelengths>(0.f);
};

// Extracted out of wf_finish_material_scatter() (item 4 sub-step 3 of this
// project's own code-health plan) - the ReSTIR-or-classic direct-light draw
// that used to sit inline between wf_setup_glossy_context's own call site
// and the area-light shading block (wf_local_light_bsdf's own call site,
// unchanged by this extraction - that block still lives in
// wf_finish_material_scatter itself and still consumes this function's
// return value exactly as it consumed the equivalent inline locals before).
//
// Unlike wf_local_light_bsdf, this has NO dependency on any of
// wf_finish_material_scatter's other local lambdas (evalGlossyF,
// liftUnboundedRGB, addToFramebuffer) - the only lambda it needs
// (liftEmission) is self-contained (only reads `swl`, defined fresh below),
// so - like wf_setup_glossy_context above - this becomes a real function
// with an explicit parameter list instead of a local lambda. NOT tagged
// CPU_GPU like wf_setup_glossy_context, though: several callees this
// function makes (wf_generate_restir_candidate, wf_rand,
// wf_restir_temporal_combine, wf_restir_volume_temporal_combine,
// wf_reevaluate_light_geometry, wf_light_bvh_pmf, wf_light_raw_emission) are
// themselves plain `__device__ __forceinline__` (device-only, no host
// overload) - a CPU_GPU tag here would fail to compile for the host side of
// this translation unit. __device__-only, matching wf_finish_material_scatter
// itself exactly.
//
// Side effects (same ones the inline code had, now scoped to this function,
// still firing at the exact same relative point in the overall NEE sequence
// since this is called synchronously, once, from the same call site):
//   - `restirReservoirs[pixelIndex] = res` / `restirCtx.normalOut[pixelIndex]`
//     (surface ReSTIR persistence, useRestir && !isPhase only)
//   - `restirVolumeReservoirs[pixelIndex] = volRes` and the
//     volumeMatIdxOut/volumePhaseWoGOut/volumeEntryPointOut writes
//     (volumetric ReSTIR persistence, useRestir && isPhase && non-null
//     restirVolumeReservoirs only)
//   - `seed` mutated by reference throughout (wf_generate_restir_candidate's
//     own draws in the RIS loop, wf_rand() per candidate,
//     wf_restir_temporal_combine/wf_restir_volume_temporal_combine's own
//     internal draws, and the classic single-draw path's own
//     wf_generate_restir_candidate call) - preserved bit-for-bit in the same
//     order as before, since none of this moved relative to anything else.
__device__ __forceinline__ NeeLightSample wf_nee_pick_light(
	const float3& hit_point, unsigned int& seed, float time,
	const SphereData* spheres, const QuadData* quads,
	const TriangleData* triangles, const BilinearPatchData* bilinearPatches,
	const DiskData* disks, const CylinderData* cylinders,
	const MaterialData* materials,
	const int* lightIndices, const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable, unsigned int numLights,
	const TextureData* textures, const unsigned char* texturePixels,
	WfLightBvhContext lightBvh,
	const SampledWavelengths<kWFNWavelengths>& swl,
	int pixelIndex, int depth, int matIdx, bool isPhase,
	const float3& normal, const float3& phaseWo, float phaseG,
	GpuReservoir* restirReservoirs, const GpuRestirTemporalContext& restirCtx,
	const float3& mediumEntryPoint, float mediumMeanFreePath,
	GpuVolumeReservoir* restirVolumeReservoirs,
	const GpuVolumeRestirTemporalContext& restirVolumeCtx,
	int* volumeMatIdxOut, float4* volumePhaseWoGOut, float4* volumeEntryPointOut)
{
	using SS = SampledSpectrum<kWFNWavelengths>;

	// ReSTIR DI (Live Preview only, restirReservoirs non-null - see this
	// function's own parameter comment) replaces the single alias-table draw
	// below with weighted resampling over kRestirCandidateCount candidates,
	// for primary hits only (depth==0 - Bitterli 2020's canonical scope; an
	// indirect bounce's NEE keeps the classic single-draw path even under
	// real-time preview). haveSample/to_light/max_dist/light_pdf/nee_norm are
	// filled by EITHER this block or the classic single-draw block just below
	// it, then consumed identically by the caller's shared BSDF-evaluation/
	// shadow-ray code - light_pdf remains the drawn/winning candidate's own
	// selection_pdf*geom_pdf (used only for the MIS weight against BSDF
	// sampling), while nee_norm is what actually normalizes the radiance
	// contribution: 1/light_pdf classically, or the reservoir's own unbiased
	// contribution weight W under ReSTIR (W already IS an unbiased estimator
	// of that reciprocal - see restir_reservoir_add's own comment) - so the
	// Ld formula the caller computes multiplies by nee_norm instead of
	// dividing by light_pdf.
	bool   haveSample = false;
	float3 to_light = make_float3(0.0f, 0.0f, 0.0f);
	float  max_dist = 0.0f;
	float  light_pdf = 0.0f;
	float  nee_norm = 0.0f;
	SS light_emission_spec(0.f);

	// A light's RGB colour is an ILLUMINANT (pbrt-v4 RGBIlluminantSpectrum:
	// scale * rsp(lambda) * D65(lambda)), not a bare RGBUnboundedSpectrum
	// (scale * rsp(lambda)) -- without the D65 factor, a grey light
	// uplifts to a flat/equal-energy spectrum (chromaticity (0.333,
	// 0.333)) instead of D65-white (0.3127,0.3290), which then
	// reconstructs as a non-neutral RGB through wf_xyz_to_linear_rgb's
	// D65-targeted matrix (R inflated ~20%, G/B suppressed ~5-11%) -- see
	// dev_sample_d65()'s own comment in spectral_device.h for the full
	// derivation.
	auto liftEmission = [&](float3 le) -> SS {
		float m = le.x > le.y ? (le.x > le.z ? le.x : le.z)
							  : (le.y > le.z ? le.y : le.z);
		float sc = 2.f * m;
		if (sc <= 0.f) return SS(0.f);
		float c0, c1, c2;
		dev_srgb_to_coeffs(le.x/sc, le.y/sc, le.z/sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = sc * poly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
		return s;
	};

	// Gates on restirReservoirs (surface) alone, not restirVolumeReservoirs -
	// safe ONLY because WavefrontPathTracer::render() always allocates/
	// passes both together under the single restirEnabled_ toggle (no
	// separate UI toggle for the volumetric feature - see
	// restirVolumeReservoirs's own parameter comment below). The isPhase
	// branches further down re-check restirVolumeReservoirs explicitly
	// before touching it, so a null restirVolumeReservoirs with a non-null
	// restirReservoirs degrades gracefully (volume RIS/persistence simply
	// skipped) - but the reverse (restirVolumeReservoirs non-null,
	// restirReservoirs null) would silently skip volumetric ReSTIR entirely
	// despite a caller believing it was enabled. If this coupling is ever
	// loosened (e.g. an independent volumetric-ReSTIR toggle), this gate
	// needs to test both pointers - not a bare OR, though, since the surface
	// persistence write below (`restirReservoirs[pixelIndex] = res`) still
	// assumes restirReservoirs itself is non-null whenever useRestir is true.
	// Not for diffuse transmission: the resampling target below is a one-sided cosine proxy, which would give a light on the far side of the surface zero
	// weight and so never pick it, though the material's BSDF takes light from both sides. Classic light sampling handles it.
	const bool useRestir = (restirReservoirs != nullptr && depth == 0 && materials[matIdx].type != MaterialType::DiffuseTransmission);
	if (useRestir) {
		GpuReservoir res;
		for (int i = 0; i < kRestirCandidateCount; ++i) {
			GpuLightSample cand; float3 candDir; float candMaxDist = 0.0f, candPdf = 0.0f; float3 candRaw;
			if (!wf_generate_restir_candidate(hit_point, seed, time,
					spheres, quads, triangles, bilinearPatches, disks, cylinders,
					materials, lightIndices, lightKinds, aliasTable, numLights,
					textures, texturePixels, cand, candDir, candMaxDist, candPdf, candRaw,
					lightBvh))
				// numLights==0/no aliasTable is a loop-invariant condition (every
				// remaining draw would fail identically, so `break` was correct
				// for that case alone) - but a light BVH's per-draw zero-
				// importance reject (wf_generate_restir_candidate's own comment)
				// is NOT loop-invariant: a DIFFERENT random draw can easily land
				// in a light's cone of influence even when this one didn't.
				// `continue`, not `break`, so a BVH-active scene still spends
				// its full kRestirCandidateCount budget instead of aborting the
				// reservoir after the first unlucky draw.
				continue;
			// 1e-6f, not a looser 1e-9f: matches the classic single-draw NEE
			// path's own `light_pdf > 1e-6f` gate exactly (a few lines below)
			// - that threshold is what already bounds the classic path's own
			// worst-case 1/light_pdf to ~1e6, battle-tested by every existing
			// non-ReSTIR render. A looser floor here let candidates with a
			// near-degenerate geometric pdf (a shading point landing extremely
			// close to a randomly sampled point on an area light) through,
			// producing a risWeight up to 1000x larger than the classic path
			// would ever accept - RIS then reservoir-selects that single
			// candidate outright, giving the whole reservoir a correspondingly
			// huge W that saturates the pixel white and, via spatial/temporal
			// reuse, spreads into a visible blocky artifact across nearby
			// pixels and following frames.
			if (candPdf <= 1e-6f) continue;
			// Resampling-only target proxy: a plain Lambertian-cosine-weighted
			// luminance-like magnitude of the (already twoSided-gated) raw
			// emission - NOT the exact per-material BSDF value (that's only
			// evaluated once, below, for the FINAL winning sample). An
			// approximate p_hat still yields an unbiased ReSTIR estimator (it
			// only changes variance, not correctness - see restir_reservoir_
			// ucw's own comment and wavefront_restir_helpers.h's header
			// comment); evaluating the real per-material BSDF (glossy
			// evalGlossyF included) for all kRestirCandidateCount draws every
			// pixel every frame would be far more expensive for a resampling
			// decision that only needs a reasonable importance proxy.
			// isPhase: `normal` here is the medium BOUNDARY's entry-surface
			// normal (h.normal, wavefront_kernels_materials.cu), unrelated to
			// the actual interior scatter point - wf_restir_target_proxy's
			// cosine term would silently zero/bias every candidate against
			// it. wf_restir_target_proxy_phase replaces that cosine with the
			// phase value between phaseWo (this vertex's own incoming
			// direction) and the candidate light direction instead - see that
			// function's own comment (wavefront_restir_math.h).
			float pHat = isPhase
				? wf_restir_target_proxy_phase(candRaw, candDir, phaseWo, phaseG)
				: wf_restir_target_proxy(candRaw, candDir, normal);
			float risWeight = pHat / candPdf;
			restir_reservoir_add(res, cand, risWeight, 1, pHat, wf_rand(seed));
		}
		// Temporal reuse - folds in the reprojected previous-frame reservoir
		// (if any) BEFORE finalizing, so W/pHat reflect the combined M, not
		// just this frame's kRestirCandidateCount fresh draws. A no-op
		// (restirCtx.historyValid false, the default) for every call site
		// that doesn't pass a real context - see wf_restir_temporal_combine's
		// own comment. Skipped entirely for isPhase: restirCtx is the
		// SURFACE reservoir's history (indexed by pixel, populated by surface
		// hits) - reprojecting it into a phase vertex's own RIS result would
		// combine two incompatible sample kinds at the same pixel index.
		if (!isPhase) {
			wf_restir_temporal_combine(res, hit_point, normal, restirCtx, seed,
				spheres, quads, triangles, bilinearPatches, disks, cylinders,
				materials, textures, texturePixels);
		} else if (restirVolumeReservoirs != nullptr) {
			// Volumetric analog of the surface combine just above - keyed by
			// mediumEntryPoint (stable) rather than hit_point (redrawn fresh
			// every frame for a phase vertex), phaseWo/phaseG standing in for
			// normal, and gated additionally on matIdx equality (see
			// wf_restir_volume_temporal_combine's own comment,
			// wavefront_restir_helpers.h). A no-op when
			// restirVolumeCtx.historyValid is false (the default), same shape
			// as wf_restir_temporal_combine above.
			//
			// wf_restir_volume_temporal_combine takes a GpuVolumeReservoir&,
			// not `res` (GpuReservoir) - the two share the exact same core
			// RIS fields (restir_reservoir_add's own template only ever
			// touches those), so this bridges via a temporary that copies
			// `res`'s own within-frame RIS result in, then copies the
			// (possibly temporally-combined) result back out - `res` stays
			// the single source of truth restir_finalize/the shading step
			// below both read, regardless of isPhase.
			GpuVolumeReservoir volRes;
			volRes.sample = res.sample;
			volRes.weightSum = res.weightSum;
			volRes.M = res.M;
			volRes.W = res.W;
			volRes.pHat = res.pHat;
			wf_restir_volume_temporal_combine(volRes, mediumEntryPoint, phaseWo, phaseG, matIdx,
				restirVolumeCtx, seed,
				spheres, quads, triangles, bilinearPatches, disks, cylinders,
				materials, textures, texturePixels);
			res.sample = volRes.sample;
			res.weightSum = volRes.weightSum;
			res.M = volRes.M;
			res.W = volRes.W;
			res.pHat = volRes.pHat;
		}
		restir_finalize(res);
		// isPhase deliberately does NOT persist `res` into restirReservoirs/
		// restirCtx.normalOut - see wf_restir_target_proxy_phase's own call
		// site above for why a phase vertex's reservoir is incompatible with
		// the surface buffer's Lambertian-cosine convention: this frame's
		// depth==0 worldPos/normal AOV writes (evaluate_materials(), fired
		// unconditionally for every depth==0 hit including a medium's own
		// entry-surface point) already give a phase-vertex pixel a "valid"
		// worldPos.w/normal from the SURFACE spatial-reuse pass's point of
		// view; writing a phase-derived reservoir into the same buffer would
		// let that pass and next frame's temporal reuse silently blend
		// surface and volumetric samples together. `res` still drives THIS
		// frame's own shading immediately below regardless - only cross-
		// frame/cross-pixel persistence is skipped, so isPhase still gets the
		// full within-frame RIS resampling benefit over classic single-draw
		// NEE, just without carrying forward across frames or pixels (a
		// deliberately narrower scope than surface DI's full temporal+
		// spatial reuse - see this project's own plan for why).
		if (!isPhase) {
			// Written unconditionally (even an invalid/empty reservoir) - this
			// is the CURRENT frame's own buffer, which the spatial-reuse pass
			// (wavefront_kernels_restir.cu) reads next, and which next
			// frame's temporal reuse ultimately reads via that pass's own
			// output - must reflect this pixel's real outcome (including "no
			// light reached this pixel this frame"), not be left stale from a
			// reused allocation.
			restirReservoirs[pixelIndex] = res;
			if (restirCtx.normalOut) restirCtx.normalOut[pixelIndex] = normal;
		} else if (restirVolumeReservoirs != nullptr) {
			// Volumetric analog of the surface persistence write just above,
			// into the SEPARATE d_volumeReservoirs_/d_volumeMatIdx_ buffers -
			// never the surface restirReservoirs buffer above, which the
			// surface spatial-reuse/temporal-combine code already assumes is
			// exclusively Lambertian-cosine-convention samples (see this
			// project's own plan for why the two are kept apart). Written
			// unconditionally, same "reflect this pixel's real outcome, don't
			// leave it stale" reasoning as the surface write. volumeMatIdxOut
			// is written whenever a phase vertex is reached at all (not
			// gated on res.valid()) - the spatial-reuse pass's own neighbor
			// gate (wf_restir_volume_spatial_valid) needs this pixel's medium
			// identity regardless of whether THIS frame's own RIS loop found
			// a usable light candidate.
			GpuVolumeReservoir volRes;
			volRes.sample = res.sample;
			volRes.weightSum = res.weightSum;
			volRes.M = res.M;
			volRes.W = res.W;
			volRes.pHat = res.pHat;
			volRes.mediumMatIdx = matIdx;
			restirVolumeReservoirs[pixelIndex] = volRes;
			if (volumeMatIdxOut) volumeMatIdxOut[pixelIndex] = matIdx;
			if (volumePhaseWoGOut) volumePhaseWoGOut[pixelIndex] = make_float4(phaseWo.x, phaseWo.y, phaseWo.z, phaseG);
			if (volumeEntryPointOut) volumeEntryPointOut[pixelIndex] = make_float4(mediumEntryPoint.x, mediumEntryPoint.y, mediumEntryPoint.z, mediumMeanFreePath);
		}

		if (res.valid() && res.W > 0.0f) {
			float geomPdfAtHit = 0.0f;
			// Re-derive from hit_point - exact, not an approximation, even
			// when res.sample won by temporal reuse (a DIFFERENT pixel/
			// frame's own origin, not hit_point) rather than this frame's own
			// RIS loop: wf_reevaluate_light_geometry only ever needs the
			// sample's already-fixed point/normal/time (GpuLightSample::time's
			// own comment) plus the NEW query origin - no search/Jacobian
			// required regardless of whether the origin actually changed (see
			// that function's own header comment).
			if (wf_reevaluate_light_geometry(res.sample, hit_point, spheres, quads, triangles,
					bilinearPatches, disks, cylinders, to_light, max_dist, geomPdfAtHit) &&
				geomPdfAtHit > 0.0f) {
				// The winning reservoir sample may have been drawn (this frame,
				// or via temporal/spatial reuse) using EITHER selection method -
				// re-deriving its selection_pdf from the alias table unconditionally
				// would be wrong whenever it was actually drawn via the light BVH's
				// position-dependent pmf (a fixed, power-only alias pdf can differ
				// from the BVH pmf by an order of magnitude or more at this exact
				// hit_point), corrupting the MIS weight below (wf_mis(light_pdf,
				// brdf_pdf_l)). wf_light_bvh_pmf() replays the bit-trail to recover
				// the SAME pmf the BVH draw would have produced at this point,
				// exactly like gpu_light_bvh_pmf() does for the recursive backend's
				// own BSDF-hit MIS case (optix_device_helpers_lighting.h).
				const float selection_pdf = (lightBvh.nodeCount > 0)
					? wf_light_bvh_pmf(hit_point.x, hit_point.y, hit_point.z, res.sample.lightIdx, numLights,
						lightBvh.nodes, lightBvh.bitTrail, lightBvh.nodeCount,
						lightBvh.allBMinX, lightBvh.allBMinY, lightBvh.allBMinZ,
						lightBvh.allBMaxX, lightBvh.allBMaxY, lightBvh.allBMaxZ)
					: aliasTable[res.sample.lightIdx].pdf;
				light_pdf = selection_pdf * geomPdfAtHit;
				float3 raw = wf_light_raw_emission(res.sample, to_light, materials, spheres, quads, triangles,
													bilinearPatches, disks, cylinders, textures, texturePixels);
				light_emission_spec = liftEmission(raw);
				nee_norm = res.W;
				haveSample = true;
			}
		}
	} else
	if (numLights > 0 && aliasTable) {
		// Same alias-table-draw + per-shape dispatch + twoSided-gated
		// emission lookup the ReSTIR candidate loop above uses - shared via
		// wf_generate_restir_candidate/wf_light_raw_emission (wavefront_
		// restir_helpers.h) rather than a second hand-duplicated copy, so a
		// future light-kind addition or texture-lookup fix only has one
		// place to change instead of two that can silently drift apart.
		// liftEmission() is the shared one declared above (this ReSTIR-aware
		// function's own scope, used by both the ReSTIR and classic paths).
		GpuLightSample cand;
		float3 raw;
		if (wf_generate_restir_candidate(hit_point, seed, time,
				spheres, quads, triangles, bilinearPatches, disks, cylinders,
				materials, lightIndices, lightKinds, aliasTable, numLights,
				textures, texturePixels, cand, to_light, max_dist, light_pdf, raw,
				lightBvh)) {
			light_emission_spec = liftEmission(raw);
			nee_norm = (light_pdf > 1e-6f) ? (1.0f / light_pdf) : 0.0f;
			haveSample = true;
		}
	}

	NeeLightSample result;
	result.haveSample = haveSample;
	result.to_light = to_light;
	result.max_dist = max_dist;
	result.light_pdf = light_pdf;
	result.nee_norm = nee_norm;
	result.light_emission_spec = light_emission_spec;
	return result;
}

// MIS weight for a BSDF-sampled ray that arrives at an emitter: w = pb^2 / (pb^2 + pl^2), where pb is the
// BSDF pdf the ray was sampled with and pl the solid-angle pdf next-event estimation at the vertex that sampled
// it (HitWorkItem::scatterOrigin - not the ray's own origin, which an interface pass-through moves) would have
// had for choosing this very point (light-selection pmf x area-to-solid-angle). The NEE contribution at
// that vertex is weighted by the complementary pl^2/(pb^2+pl^2) (wf_finish_material_scatter), so dropping this
// half - as the emissive early-exit used to for every non-specular bounce - lost the share of a lamp's light that
// NEE gives to the BSDF strategy: ~0 for a small or distant light, 10-20% of each indirect bounce near a large
// one. Returns 1 when NEE could not have sampled the point (an emitter outside the light list, pl == 0).
__device__ __forceinline__ float wf_emitter_hit_mis_weight(
	const HitWorkItem& h, const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
	const int* lightIndices, const GpuLightKind* lightKinds, const GpuAliasEntry* aliasTable,
	unsigned int numLights, const WfLightBvhContext& lightBvh)
{
	if (!(h.brdf_pdf > 0.0f) || numLights == 0 || aliasTable == nullptr) return 1.0f;

	// HitWorkItem::geomType and GpuLightKind number the shapes differently.
	GpuLightKind kind;
	switch (h.geomType) {
		case 0: kind = GpuLightKind::Sphere; break;
		case 1: kind = GpuLightKind::Quad; break;
		case 2: kind = GpuLightKind::BilinearPatch; break;
		case 3: kind = GpuLightKind::Triangle; break;
		case 4: kind = GpuLightKind::Disk; break;
		default: kind = GpuLightKind::Cylinder; break;
	}
	const int li = gpu_find_light(lightIndices, lightKinds, numLights, kind, h.primIdx);
	if (li < 0) return 1.0f;

	GpuLightSample s;
	s.lightIdx = li;
	s.kind     = kind;
	s.primIdx  = h.primIdx;
	s.sampleU  = h.uv_u;
	s.sampleV  = h.uv_v;
	s.point    = h.hitPoint;
	s.normal   = h.normal;
	s.time     = h.time;
	float3 toLight; float dist, geomPdf;
	if (!wf_reevaluate_light_geometry(s, h.scatterOrigin, spheres, quads, triangles, bilinearPatches, disks,
									  cylinders, toLight, dist, geomPdf) || !(geomPdf > 0.0f))
		return 1.0f;

	const float3& o = h.scatterOrigin;
	const float selection = (lightBvh.nodeCount > 0)
		? wf_light_bvh_pmf(o.x, o.y, o.z, li, numLights, lightBvh.nodes, lightBvh.bitTrail, lightBvh.nodeCount,
			lightBvh.allBMinX, lightBvh.allBMinY, lightBvh.allBMinZ,
			lightBvh.allBMaxX, lightBvh.allBMaxY, lightBvh.allBMaxZ)
		: aliasTable[li].pdf;
	const float pl = selection * geomPdf;
	if (!(pl > 0.0f)) return 1.0f;
	return 1.0f - wf_mis(pl, h.brdf_pdf);   // wf_mis(a, b) = a^2/(a^2+b^2), so this is pb^2/(pb^2+pl^2)
}

// Extracted out of wf_finish_material_scatter() (item 4 sub-step 4 of this
// project's own code-health plan) - the shadow-ray build+push logic that was
// near-identically repeated across the area-light, sky-light, and punctual-
// light NEE blocks below. Like wf_setup_glossy_context/wf_nee_pick_light
// above, this becomes a real function with an explicit parameter list rather
// than a local lambda: it has no dependency on any of wf_finish_material_
// scatter's OTHER local lambdas (evalGlossyF, wf_local_light_bsdf,
// addToFramebuffer, liftEmission, ...), only on values already available as
// wf_finish_material_scatter's own parameters or as plain locals computed
// identically for all 3 call sites within one invocation (hit_point/normal/
// isPhase/shadow_eps/filterWeight/swl/pixelIndex/time/shadowQueue).
//
// raw_cos is passed in rather than recomputed as dot(lightDir, normal) here:
// every call site already derives its own raw_cos earlier (for its own
// cos_l/cull-gate logic before ever reaching this function), so passing it
// through avoids a redundant dot product and guarantees a bit-identical
// result to what that site already used.
//
// tMax and shadowSeed are pre-computed BY THE CALLER rather than derived in
// here, since they are the two genuinely call-site-varying-IN-KIND (not just
// in value) pieces of the 3 sites' shadow-ray setup:
//   - tMax: the area-light and punctual-light sites subtract a 0.002f
//     near-light epsilon (max_dist - 0.002f / t_max - 0.002f) so the shadow
//     ray doesn't self-intersect the light's own surface; the sky site
//     passes the unadjusted 1e30f sentinel instead (an infinite light has no
//     near-surface to avoid).
//   - shadowSeed: the area-light and sky-light sites pass wf_pcg(seed)
//     (each fires at most once per wf_finish_material_scatter call); the
//     punctual-light loop instead mixes in `pli` (wf_pcg(seed ^ (pli *
//     0x9E3779B9u))) since that loop can push several shadow rays per call,
//     one per light, with no decorrelating wf_rand(seed) draw between
//     iterations - see that call site's own comment.
// Both are ordinary values by the time they reach here, so this function
// itself stays free of any site-specific branching.
__device__ __forceinline__ void wf_push_nee_shadow_ray(
	WorkQueue<ShadowRayWorkItem>& shadowQueue,
	const float3& hit_point, const float3& normal, bool isPhase, float shadow_eps,
	float raw_cos, const float3& lightDir, float tMax,
	const SampledSpectrum<kWFNWavelengths>& Ld, float filterWeight,
	const SampledWavelengths<kWFNWavelengths>& swl,
	int pixelIndex, float time, bool isGiCandidate, unsigned int shadowSeed)
{
	// See the area-light NEE block's own "0.01, not the original 0.001"
	// comment (wf_finish_material_scatter, below) for the full story on why
	// this is a combined normal+direction offset (not the normal alone), why
	// isPhase skips the normal term entirely, and why the normal term uses
	// copysignf(shadow_eps, raw_cos) rather than always +shadow_eps (only
	// RoughDielectric's transmission side, raw_cos<0, ever nudges the origin
	// to the far side of the surface instead of back into the same
	// hemisphere the ray isn't going toward).
	ShadowRayWorkItem shadow;
	shadow.origin    = hit_point + (isPhase ? make_float3(0.0f, 0.0f, 0.0f) : copysignf(shadow_eps, raw_cos) * normal)
		+ shadow_eps * normalize(lightDir);
	// Aimed at the sampled target from the shifted origin - see shadow_ray_toward().
	shadow_ray_toward(hit_point, shadow.origin, normalize(lightDir), tMax, shadow.direction, shadow.tMax);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		shadow.Ld[i]              = Ld[i] * filterWeight;  // see RayWorkItem::filterWeight's own comment
		shadow.wavelengths[i]     = swl.lambda[i];
		shadow.wavelength_pdfs[i] = swl.pdf[i];
	}
	shadow.pixelIndex = pixelIndex;
	shadow.time = time;
	// See giCandidateEligible's own comment (wf_finish_material_scatter,
	// below).
	shadow.isGiCandidate = isGiCandidate;
	// See ShadowRayWorkItem::seed's own comment - shadowSeed is already the
	// derived value (wf_pcg(seed), or wf_pcg(seed ^ (pli * 0x9E3779B9u)) for
	// the punctual-light loop), not a consuming wf_rand(seed) draw itself, so
	// this doesn't perturb the caller's own subsequent sampling.
	shadow.seed = shadowSeed;
	shadowQueue.push(shadow);
}
