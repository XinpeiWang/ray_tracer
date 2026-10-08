#pragma once
// optix_device_mix_camera.h -- part 5 of 5 of optix_device_helpers.h (included by it, in order; not meant to be included on its own).

// True for the 7 MaterialTypes that need sphere/box-specific handling -
// Medium/DielectricMedium/CloudMedium/RgbGridMedium/GridMedium (volume
// ray-marching against a sphere's own two intersection roots, or a box's
// slab test - see optix_intersection_sphere.h's shape-specific branches)
// and Hair/Principled (sampled directly via a sibling dispatch alongside
// shade_material(), never through it - see shade_material()'s own default:
// comment above). NormalMappedLambertian is deliberately NOT included here:
// unlike these 6, it has a real, working implementation on triangles too
// (optix_intersection_triangle.h's own tangent-frame branch), just not on
// quads or bilinear patches - callers decide separately whether to also
// reject that type.
//
// A primitive type that does not implement one of these 6 has no
// physically-sensible fallback to construct (there is no "inside" to a
// flat quad/triangle/bilinear-patch to ray-march a medium through), so
// scene authors assigning one of these types to an unsupported primitive
// is a real authoring bug, not a recoverable render state. Callers should
// trap on this BEFORE ever reaching shade_material(), with a message
// identifying the actual type/primitive mismatch - shade_material()'s own
// default: trap would still catch it, but with a generic "unhandled
// MaterialType" message that doesn't reveal it was actually a valid type
// used on the wrong geometry.
// Device-side port of CPU's branch_hash01() (src/TheRestOfYourLife/
// material_pbrt.h) - deterministic [0,1) value from a world-space point, NOT
// a fresh random_float()/seed draw. Mix resolution needs this determinism:
// a single scattering event's radiance closest-hit, its shadow any-hit, and
// (on re-intersection) any later ray touching the same point must all agree
// on which sub-material a Mix resolved to - a per-call RNG draw would let
// them silently disagree, exactly the bug branch_hash01's own CPU comment
// describes for scatter()/scattering_pdf()/is_shadow_transmissive().
__device__ __forceinline__ float mix_branch_hash01(const float3& p) {
	float h = sinf(p.x * 127.1f + p.y * 311.7f + p.z * 74.7f) * 43758.5453f;
	return h - floorf(h);
}

// Resolves a (possibly Mix) MaterialData to a real, non-Mix MaterialData by
// hashing `hit_point` and iteratively picking sub-material A or B -
// LOOPING, not recursing (a Mix's own sub-material can itself be another
// Mix - pbrt-v4 allows mix-of-mix, matches pbrt_gpu_builder.h's build-time
// resolveMixColor()/makeMaterial() recursion), so this never adds to
// shade_material()'s call graph. Must run before ANY other mat.type branch
// in a shape's closest-hit/intersection program or shadow any-hit program
// (see MaterialType::Mix's own comment) - in particular before
// material_requires_sphere_only_handling() below, so a Mix resolving to a
// sphere-only type on the wrong geometry still traps correctly, and before
// shade_material() itself, since Hair/Subsurface/Medium-family are
// dispatched via sibling paths this function never touches.
//
// `outMatIdx` receives the resolved material's own index into
// params.materials (updated even when `mat` was never Mix, so callers can
// use it unconditionally) - needed by callers that separately look up
// per-material auxiliary data by index (e.g. Subsurface's bssrdfTables,
// Measured's measuredTables) after resolution.
__device__ __forceinline__ MaterialData resolve_mix_material(MaterialData mat, int matIdx,
															  const float3& hit_point, int& outMatIdx) {
	constexpr int kMaxMixDepth = 8;
	for (int depth = 0; mat.type == MaterialType::Mix && depth < kMaxMixDepth; ++depth) {
		const float w = mat.mix_extra.mixWeight;
		const float h = mix_branch_hash01(hit_point);
		const int subIdx = (h >= w) ? static_cast<int>(mat.mix_extra.mixMaterialAIdx)
									 : static_cast<int>(mat.mix_extra.mixMaterialBIdx);
		matIdx = subIdx;
		mat = params.materials[subIdx];
	}
	outMatIdx = matIdx;
	return mat;
}

__device__ __forceinline__ bool material_requires_sphere_only_handling(MaterialType type) {
	switch (type) {
		case MaterialType::Medium:
		case MaterialType::DielectricMedium:
		case MaterialType::CloudMedium:
		case MaterialType::RgbGridMedium:
		case MaterialType::GridMedium:
		case MaterialType::Hair:
		case MaterialType::Principled:
			return true;
		default:
			return false;
	}
}

// Whether shade_material()'s dpdu parameter is ever actually read for this
// material type - NormalMappedLambertian (its normal-map tangent basis), the
// 4 anisotropy-capable kinds (their UV-aligned shading frame, see
// BuildDpduTangentFrame's own comment, microfacet.h), and DielectricMedium
// (rough_dielectric_scatter_and_nee()'s own tangent frame, needed whenever
// THIS material's own surfaceKind turns out to be rough - checked per-hit
// inside that function, not here, since this dispatch only has the
// MaterialType to go on, not the fused sub-flavor; a smooth/thin
// DielectricMedium pays a small, unconditional dpdu-computation cost it
// doesn't actually use, the same trade-off material_requires_sphere_only_
// handling()'s own switch-based dispatch style already accepts elsewhere
// for simplicity over per-instance precision). Every OTHER material kind
// (Lambertian, Metal, Dielectric, DiffuseLight, RoughMetal, the rest of the
// Medium family, etc.) never touches dpdu at all, so each intersection
// file's own dpdu computation is gated on this - rather than paying for
// trig/solve/transform work on every hit regardless of whether the
// material can ever use the result.
__device__ __forceinline__ bool material_needs_dpdu(MaterialType type) {
	switch (type) {
		case MaterialType::NormalMappedLambertian:
		case MaterialType::Conductor:
		case MaterialType::RoughDielectric:
		case MaterialType::CoatedDiffuse:
		case MaterialType::CoatedConductor:
		case MaterialType::DielectricMedium:
			return true;
		default:
			return false;
	}
}

// Realistic multi-element lens camera (pbrt-v4 RealisticCamera, src/shared/
// cameras.h). Host-side precompute (focus-adjusted lens table + exit-pupil
// bounds table) happens once in scene_builder.cpp by directly constructing a
// RealisticCamera<float> and reading its accessors - this device function
// only ports the per-ray hot path: film-plane mapping, exit-pupil sampling,
// and the per-element Snell's-law trace (Woop et al. watertight lens
// intersection, mirrors cameras.h's trace_lenses_from_film/sample_exit_pupil/
// generate_ray exactly, including the eta_i/eta_t Refract convention fix).
// u,v are in [0,1]^2 with v already flipped by the raygen caller (matching
// the other CameraKinds' "lower-left-origin" viewport convention) - unlike
// those, RealisticCamera's film coordinates increase top-to-bottom matching
// raw raster row order (see raster_to_film's comment in cameras.h), so v is
// un-flipped back here first.
// Returns false (weight left at 0) if the ray is fully vignetted.
__device__ __forceinline__ bool sample_realistic_camera_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& out_origin, float3& out_direction, float& out_weight
) {
	out_weight = 0.0f;
	if (cam.numLensElements <= 0 || cam.numExitPupilBounds <= 0) return false;

	float v_raw = 1.0f - v;  // undo raygen's lower-left-origin flip
	float pfx = -((2.0f*u - 1.0f) * cam.film_half_x);  // pbrt-v4 negates x
	float pfy =   (2.0f*v_raw - 1.0f) * cam.film_half_y;

	// sample_exit_pupil
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

	float u0 = random_float(seed), u1 = random_float(seed);
	float lx = b.xMin + u0*(b.xMax - b.xMin);
	float ly = b.yMin + u1*(b.yMax - b.yMin);

	float sinTheta = (rFilm > 0.0f) ? pfy/rFilm : 0.0f;
	float cosTheta0 = (rFilm > 0.0f) ? pfx/rFilm : 1.0f;
	float ppx = cosTheta0*lx - sinTheta*ly;
	float ppy = sinTheta*lx + cosTheta0*ly;
	float ppz = cam.lens_rear_z;

	float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
	float rLen = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);

	// trace_lenses_from_film: camera space (film z=0, +z toward scene) ->
	// lens space (z flipped): loz=-oz, ldz=-dz.
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
			// intersect_spherical
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
			float eta = eta_i/eta_t;  // matches cameras.h Refract's eta=eta_i/eta_t contract
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

	// Camera space -> world space via su(right)/sv(up)/sw(forward)/origin.
	out_origin = cam.origin + lensOutOx*cam.su + lensOutOy*cam.sv + lensOutOz*cam.sw;
	out_direction = normalize(lensOutDx*cam.su + lensOutDy*cam.sv + lensOutDz*cam.sw);
	out_weight = w;
	return true;
}

// Generate a primary camera ray for raster coordinates (u,v) in [0,1]^2,
// dispatching on params.camera.kind. Mirrors the CPU camera models in
// src/shared/cameras.h (OrthographicCamera/SphericalCamera/RealisticCamera::
// generate_ray) and src/TheRestOfYourLife/camera.h's book-style
// defocus_angle/focus_dist thin-lens DOF, which the Perspective case folds
// in via camera.defocus_disk_u/v (both zero = DOF disabled, matching how
// scene_builder.cpp always zero-initializes GpuCameraParams). `weight`
// applies to Realistic only (cos^4(theta)/pdf vignetting term, 1.0 for every
// other CameraKind - fold into the caller's initial path throughput).
__device__ __forceinline__ void generate_primary_ray(
	float u, float v, unsigned int& seed, float3& origin, float3& direction, float& weight
) {
	const GpuCameraParams& cam = params.camera;
	weight = 1.0f;

	if (cam.animated) {
		// Real per-ray shutter-time camera motion blur - mirrors
		// src/TheRestOfYourLife/camera.h's camera_is_animated branch of
		// get_ray(): build a LOCAL-space ray exactly as the static
		// Perspective case below does, then place it in world space via a
		// per-sample-time-interpolated camera-to-world transform instead
		// of a single static basis. Perspective-only, like CPU's own
		// camera_is_animated (mutually exclusive with an alt camera model
		// there too) - scene_builder.cpp never sets `animated` alongside a
		// non-Perspective `kind`.
		float3 local_pixel_sample = cam.localLowerLeftCorner + u * cam.localHorizontal + v * cam.localVertical;
		float3 local_origin = make_float3(0.0f, 0.0f, 0.0f);
		bool hasDOF = (cam.localDefocusDiskU.x != 0.0f || cam.localDefocusDiskU.y != 0.0f || cam.localDefocusDiskU.z != 0.0f ||
					   cam.localDefocusDiskV.x != 0.0f || cam.localDefocusDiskV.y != 0.0f || cam.localDefocusDiskV.z != 0.0f);
		if (hasDOF) {
			float3 p = random_in_unit_disk(seed);
			local_origin = p.x * cam.localDefocusDiskU + p.y * cam.localDefocusDiskV;
		}
		float3 local_direction = local_pixel_sample - local_origin;
		// Shutter-time interpolation fraction, uniform on [0,1]:
		// AnimatedTransform::Interpolate always normalizes by (endTime -
		// startTime) before use, so for a time drawn uniformly across the
		// shutter window, dt is uniform on [0,1] regardless of the
		// window's actual numeric bounds - GpuCameraParams doesn't need to
		// store shutterOpen/Close at all.
		float dt = random_float(seed);
		// Slerp + quaternion-to-matrix built ONCE, then applied to both
		// origin and direction (gpu_camera_anim_apply, camera_motion_blur_
		// device.h) - the matrix is identical for both, only the vector and
		// isPoint differ.
		GpuAnimRotMat rot = gpu_camera_anim_rotation(cam.animR0, cam.animR1, dt);
		origin = gpu_camera_anim_apply(rot, local_origin, true, cam.animT0, cam.animT1, dt);
		// local_direction is not unit length (it's pixel_sample - lens
		// origin, same as the static Perspective case just below) and the
		// rotation preserves length, so this needs an explicit normalize -
		// every other camera branch in this function returns a unit
		// direction too.
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
			// pbrt-v4 SphericalCamera::GenerateRay - see src/shared/cameras.h
			// for the reference this mirrors, both mappings finish with a
			// swap(dir.y, dir.z) folded directly into which raw component
			// feeds ly vs lz below (rather than an actual runtime swap).
			// `v` here already carries the Y-flip optix_raygen.h applies for
			// every CameraKind (v=1 at the top framebuffer row), matching
			// Perspective/Orthographic's `vertical` basis vector, which
			// points world-up - so v=1 (top row) adds the full +up vector,
			// correctly landing "up" at the top of the frame for those two.
			// CPU's SphericalCamera::generate_ray, by contrast, uses raw
			// pFilm_y/res_y with NO such flip (v=0 at its own top row) - its
			// theta=v*pi formula relies on THAT convention to put dir.y=+1
			// (up) at the top row. Feeding it this shared, already-flipped
			// `v` directly would put "down" at the top of the frame instead
			// - undo the flip locally so both mappings match CPU exactly.
			const float v_sph = 1.0f - v;
			float lx, ly, lz;
			if (cam.sphericalMapping == 1) {  // EqualArea
				double ud = (double)u, vd = (double)v_sph;
				dev_wrap_equal_area_square(ud, vd);
				double ewx, ewy, ewz;
				dev_equal_area_square_to_sphere(ud, vd, ewx, ewy, ewz);
				lx = (float)ewx;
				ly = (float)ewz;  // swap(wy,wz): final y = raw z
				lz = (float)ewy;  // swap(wy,wz): final z = raw y
			} else {  // EquiRectangular: theta in [0,pi], phi in [0,2pi]
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
			if (!sample_realistic_camera_ray(cam, u, v, seed, origin, direction, weight)) {
				origin = cam.origin;
				direction = cam.sw;  // arbitrary valid direction; weight=0 zeroes its contribution
				weight = 0.0f;
			}
			break;
		}
		default: { // Perspective, optionally thin-lens DOF
			float3 pixel_sample = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			bool hasDOF = (cam.defocus_disk_u.x != 0.0f || cam.defocus_disk_u.y != 0.0f || cam.defocus_disk_u.z != 0.0f ||
						   cam.defocus_disk_v.x != 0.0f || cam.defocus_disk_v.y != 0.0f || cam.defocus_disk_v.z != 0.0f);
			if (hasDOF) {
				float3 p = random_in_unit_disk(seed);
				origin = cam.origin + p.x * cam.defocus_disk_u + p.y * cam.defocus_disk_v;
			} else {
				origin = cam.origin;
			}
			direction = normalize(pixel_sample - origin);
			break;
		}
	}
}

// Denoiser guide-layer AOV payload packing (albedo/normal, p16-p21) - see
// PathTracingPayload::albedo/normal's own comment (optix_types.h) and
// raygen's own comment for why every closest-hit/miss program packs these
// in every branch (scattered/DiffuseLight-hit/absorbed/miss) - raygen is
// the one that decides whether to accumulate them, not the hit/miss
// program. Shared by all 4 closest-hit programs (sphere/quad/triangle/
// bilinear-patch) and the miss program instead of each hand-rolling the
// same 6 optixSetPayload_* calls. Gated on params.albedoBuffer (set only
// when this render will actually be denoised - see OptiXRenderer::render()'s
// own alloc site) so the common non-denoised path doesn't pay 6 extra
// register writes on every ray for a feature it isn't using; raygen's own
// accumulation is gated the same way (see its depth==0 check), and the
// payload registers left unwritten here are never read when albedoBuffer
// is null, since raygen only accumulates/writes through that same guard.
__device__ __forceinline__ void pack_aov_payload(float3 albedo, float3 normal) {
	if (!params.albedoBuffer) return;
	optixSetPayload_16(__float_as_uint(albedo.x));
	optixSetPayload_17(__float_as_uint(albedo.y));
	optixSetPayload_18(__float_as_uint(albedo.z));
	optixSetPayload_19(__float_as_uint(normal.x));
	optixSetPayload_20(__float_as_uint(normal.y));
	optixSetPayload_21(__float_as_uint(normal.z));
}

//==============================================================================
// Sphere Intersection Program
//==============================================================================
