// wavefront_kernels.cu
// CUDA compute kernels for the wavefront GPU path tracer.
//
// These kernels run between OptiX trace calls and handle the parts that
// don't need ray traversal:
//   1. generate_camera_rays  - fill the initial RayQueue for a given sample
//   2. evaluate_materials    - shade hit results, write shadow + next-ray items
//   3. accumulate_miss       - add background radiance for escaped rays
//   4. accumulate_shadow     - add pending Ld contributions that passed shadow test
//   5. reset_queue_counter   - zero an atomic counter before a new phase
//
// The evaluate_materials kernel intentionally mirrors the material evaluation
// logic in optix_programs.cu (closesthit) but operates on the pre-filled
// HitQueue instead of within an OptiX program.  This removes the recursion
// and allows all material evaluations to proceed in one wide CUDA kernel.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_types.h"
#include "optix_types.h"
#include "spectral_device.h"
#include "sampled_spectrum.h"
#include "spectrum_types.h"
#include "color_utils.h"
#include "optix_math_helpers.h"
#include "fresnel.h"
#include "microfacet.h"
#include "math_utils.h"
#include "bxdfs.h"  // HairBxDF<T> - see MaterialType::Hair

// ============================================================================
// Device helpers (shared with optix_programs.cu logic)
// ============================================================================

__device__ __forceinline__ unsigned int wf_pcg(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __forceinline__ float wf_rand(unsigned int& seed) {
	seed = wf_pcg(seed);
	return float(seed) / 4294967296.0f;
}

__device__ __forceinline__ float3 wf_rand3(unsigned int& seed) {
	return make_float3(wf_rand(seed), wf_rand(seed), wf_rand(seed));
}

__device__ __forceinline__ float3 wf_rand_unit(unsigned int& seed) {
	while (true) {
		float3 p = 2.0f * wf_rand3(seed) - make_float3(1.0f, 1.0f, 1.0f);
		float  l = dot(p, p);
		if (l > 1e-8f && l < 1.0f) return p / sqrtf(l);
	}
}

__device__ __forceinline__ bool wf_near_zero(const float3& v) {
	const float s = 1e-8f;
	return fabsf(v.x) < s && fabsf(v.y) < s && fabsf(v.z) < s;
}

// Henyey-Greenstein phase function direction sample. Duplicated from
// optix_device_helpers.h's sample_henyey_greenstein (with the wf_ prefix)
// rather than shared, matching this file's existing pattern of not sharing
// device helpers with the recursive path.
__device__ __forceinline__ float3 wf_sample_henyey_greenstein(const float3& wo, float g, unsigned int& seed) {
	float u1 = wf_rand(seed), u2 = wf_rand(seed);
	float cos_theta;
	if (fabsf(g) < 1e-3f) {
		cos_theta = 1.0f - 2.0f * u1;
	} else {
		float sqr = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
		cos_theta = -(1.0f + g * g - sqr * sqr) / (2.0f * g);
	}
	float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
	float phi = 2.0f * 3.14159265358979323846f * u2;
	float3 t1 = (fabsf(wo.x) > 0.9f) ? normalize(cross(make_float3(0, 1, 0), wo))
									  : normalize(cross(make_float3(1, 0, 0), wo));
	float3 t2 = cross(wo, t1);
	return normalize(sin_theta * cosf(phi) * t1 + sin_theta * sinf(phi) * t2 + cos_theta * wo);
}

// MaterialType::Hair: Marschner/Chiang fiber scattering, duplicated from
// optix_device_helpers.h's sample_hair_material (with the wf_ prefix)
// rather than shared, matching this file's existing pattern of not sharing
// device helpers with the recursive path.
__device__ __forceinline__ bool wf_sample_hair_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	HairBxDF<float> bxdf(
		wf_rand(seed) * 2.0f - 1.0f,
		mat.ior,
		mat.albedo.x, mat.albedo.y, mat.albedo.z,
		mat.fuzz,
		mat.eta_c.x,
		mat.eta_c.y);

	float3 unit_dir = normalize(ray_dir);
	float u1 = wf_rand(seed), u2 = wf_rand(seed);
	float u3 = wf_rand(seed), u4 = wf_rand(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3, u4);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
}

// reflect/refract wrappers
__device__ __forceinline__ float3 wf_reflect(const float3& v, const float3& n) {
	return cpu_gpu_reflect(v, n);
}
__device__ __forceinline__ float3 wf_refract(const float3& v, const float3& n, float e) {
	return cpu_gpu_refract<float3, float>(v, n, e);
}

// MIS power heuristic (balance if both 0)
__device__ __forceinline__ float wf_mis(float a, float b) {
	float a2 = a * a, b2 = b * b;
	return (a2 + b2 < 1e-10f) ? 1.0f : a2 / (a2 + b2);
}

// XYZ -> LINEAR sRGB (matrix only, no gamma/OETF curve).
//
// sampled_spectrum.h's XYZToSRGB() is a full linear-XYZ-to-display-sRGB
// conversion, gamma-encoding included by design (see its GammaEncodingApplied
// unit test) - correct for a caller that writes gamma-encoded pixels
// directly. This codebase's actual output path doesn't: both GPU renderers
// write into a shared linear-RGB framebuffer, and optix_interface.cpp
// applies gamma exactly once, itself, when converting that framebuffer to
// 8-bit PPM output (see its "Write pixels with gamma correction" step) -
// correct for the recursive path, whose device code (optix_intersection_
// {sphere,quad}.h) writes genuinely linear RGB. Using XYZToSRGB() here
// applied gamma a second time on top of that shared step, compounding into
// severe overexposure for anything bright enough to matter (barely visible
// for area lights' modest emission values, blown fully to white for point/
// spot lights' much larger radiance) - this stays in the same linear space
// the rest of the pipeline expects.
// No negative clamp here deliberately: this runs once per PARTIAL contribution
// (one bounce's emission, one light's NEE term, ...), and each partial XYZ->RGB
// step can legitimately dip slightly negative for saturated colors even though
// every individual spectral sample is non-negative (a standard artifact of the
// sRGB gamut being narrower than the visible-color gamut some spectra round-trip
// through). Clamping per-contribution before the caller's atomicAdd sums them
// into the framebuffer is a one-directional bias - each clamp can only push the
// running total up, never down - and it compounds with every extra bounce/light,
// which is why this was far more visible for point lights (evaluated at every
// diffuse bounce) than area lights. The single already-correct clamp on the
// fully summed per-pixel value lives downstream in optix_interface.cpp, right
// before the gamma curve; let it do the clamping.
__device__ __forceinline__ void wf_xyz_to_linear_rgb(float X, float Y, float Z,
													   float& r, float& g, float& b) {
	r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
	g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
	b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
}

// Sample a point on a sphere light; returns direction, sets geom_pdf, and
// sets maxDist to the distance to the ACTUAL sampled point on the sphere's
// surface (via ray-sphere intersection along the sampled direction) - not
// the distance to the sphere's center. Matches wf_sample_quad_light's
// contract (maxDist = distance to the sampled surface point), needed so the
// caller's shadow ray stops at the light's surface instead of continuing
// past it toward the center, where it would self-intersect the light
// sphere and register as occluded on every sample. This bug was never
// caught before because no scene had ever registered a genuinely emissive
// sphere as a light in wavefront mode until the spherical-camera scene.
__device__ float3 wf_sample_sphere_light(const SphereData& sph, const float3& hit,
										  unsigned int& seed, float& pdf, float& maxDist) {
	float3 to_c = sph.center - hit;
	float dist  = length(to_c);
	float r     = sph.radius;
	float3 dir;
	if (dist <= r) {
		pdf = 1.0f / (4.0f * 3.14159265f * r * r);
		dir = normalize(wf_rand_unit(seed));
	} else {
		float cos_max = sqrtf(fmaxf(0.0f, 1.0f - (r * r) / (dist * dist)));
		float phi     = 2.0f * 3.14159265f * wf_rand(seed);
		float cos_t   = 1.0f - wf_rand(seed) * (1.0f - cos_max);
		float sin_t   = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
		float3 w      = normalize(to_c);
		float3 u, v;
		if (fabsf(w.x) > 0.9f) u = normalize(cross(make_float3(0,1,0), w));
		else                    u = normalize(cross(make_float3(1,0,0), w));
		v = cross(w, u);
		dir = normalize(sin_t * cosf(phi) * u + sin_t * sinf(phi) * v + cos_t * w);
		float solid = 2.0f * 3.14159265f * (1.0f - cos_max);
		pdf = (solid > 1e-10f) ? 1.0f / solid : 1.0f;
	}
	// Ray-sphere intersection along `dir` from `hit` to find the true
	// surface distance (dir is constructed to be guaranteed to hit the
	// sphere, so the discriminant is never negative in practice). Take the
	// near root normally; when `hit` is inside the sphere (dist<=r above)
	// the near root is behind the origin (negative), so fall back to the
	// far root in that case.
	float3 oc = hit - sph.center;
	float b = dot(oc, dir);
	float c = dot(oc, oc) - r * r;
	float disc = fmaxf(0.0f, b * b - c);
	float sq = sqrtf(disc);
	float tNear = -b - sq;
	maxDist = (tNear > 1e-6f) ? tNear : fmaxf(0.0f, -b + sq);
	return dir;
}

// Sample a point on a quad light.
__device__ float3 wf_sample_quad_light(const QuadData& q, const float3& hit,
										unsigned int& seed, float& geom_pdf, float& maxDist) {
	float s = wf_rand(seed), t = wf_rand(seed);
	float3 p = q.Q + s * q.u + t * q.v;
	float3 dir = p - hit;
	maxDist     = length(dir);
	dir         = normalize(dir);
	float area  = length(cross(q.u, q.v));
	float cos_l = fabsf(dot(q.normal, -dir));
	geom_pdf = (cos_l > 1e-6f && maxDist > 1e-6f) ? (maxDist * maxDist) / (cos_l * area) : 0.0f;
	return dir;
}

// Equal-area sphere->square mapping for the goniometric light's image
// lookup. Duplicated from optix_device_helpers.h's dev_equal_area_sphere_to_square
// (itself a local copy of src/shared/sampling_patched.h's EqualAreaSphereToSquare -
// see that comment for why it's not shared via #include).
__device__ __forceinline__ void wf_equal_area_sphere_to_square(
	double wx, double wy, double wz, double& u, double& v
) {
	double x = fabs(wx), y = fabs(wy), z = fabs(wz);
	double r = sqrt(fmax(0.0, 1.0 - z));
	double a = (x > y) ? x : y;
	double b = (x > y) ? y : x;
	b = (a == 0.0) ? 0.0 : b / a;

	const double t1 =  0.406758566246788489601959989e-5;
	const double t2 =  0.636226545274016134946890922156;
	const double t3 =  0.61572017898280213493197203466e-2;
	const double t4 = -0.247333733281268944196501420480;
	const double t5 =  0.881770664775316294736387951347e-1;
	const double t6 =  0.419038818029165735901852432784e-1;
	const double t7 = -0.251390972343483509333252996350e-1;
	double phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))));
	if (x < y) phi = 1.0 - phi;

	double vv = phi * r;
	double uu = r - vv;
	if (wz < 0.0) { double tmp = uu; uu = 1.0 - vv; vv = 1.0 - tmp; }
	uu = copysign(uu, wx);
	vv = copysign(vv, wy);
	u = 0.5*(uu + 1.0);
	v = 0.5*(vv + 1.0);
}

// Evaluate one punctual (point/spot/distant/goniometric/projection) light at
// shading point p: same dispatch as optix_device_helpers.h's
// eval_punctual_light(), duplicated here (with the wf_ prefix) rather than
// shared, matching this file's existing pattern of not sharing NEE helpers
// with the recursive path (this kernel has no access to the recursive
// path's __constant__ params global, and the two strategies' NEE
// application differs - RGB float3 here vs this file's spectral radiance).
// PDF is always 1 (delta light) - see punctual_lights.h.
__device__ __forceinline__ bool wf_eval_punctual_light(
	const PunctualLightGPU& light, const float3& p,
	float3& wi, float3& Li, float& t_max)
{
	float Lr = 0.0f, Lg = 0.0f, Lb = 0.0f, wx = 0.0f, wy = 0.0f, wz = 0.0f;
	switch (light.kind) {
		case PunctualLightKind::Point: {
			light.point.sample_wi(p.x, p.y, p.z, wx, wy, wz);
			light.point.eval_Li(p.x, p.y, p.z, Lr, Lg, Lb);
			float dx = light.point.pos_x - p.x, dy = light.point.pos_y - p.y, dz = light.point.pos_z - p.z;
			t_max = sqrtf(dx * dx + dy * dy + dz * dz);
			break;
		}
		case PunctualLightKind::Spot: {
			light.spot.sample_wi(p.x, p.y, p.z, wx, wy, wz);
			light.spot.eval_Li(p.x, p.y, p.z, Lr, Lg, Lb);
			float dx = light.spot.pos_x - p.x, dy = light.spot.pos_y - p.y, dz = light.spot.pos_z - p.z;
			t_max = sqrtf(dx * dx + dy * dy + dz * dz);
			break;
		}
		case PunctualLightKind::Distant: {
			light.distant.sample_wi(wx, wy, wz);
			light.distant.eval_Li(Lr, Lg, Lb);
			t_max = 1e30f;
			break;
		}
		case PunctualLightKind::Goniometric: {
			const GoniometricLightGPU& g = light.gonio;
			float dx = g.pos_x - p.x, dy = g.pos_y - p.y, dz = g.pos_z - p.z;
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < 1e-20f) return false;
			t_max = sqrtf(r2);
			float inv_r = 1.0f / t_max;
			wx = dx * inv_r; wy = dy * inv_r; wz = dz * inv_r;
			float lx = g.world_to_light[0]*(-wx) + g.world_to_light[1]*(-wy) + g.world_to_light[2]*(-wz);
			float ly = g.world_to_light[3]*(-wx) + g.world_to_light[4]*(-wy) + g.world_to_light[5]*(-wz);
			float lz = g.world_to_light[6]*(-wx) + g.world_to_light[7]*(-wy) + g.world_to_light[8]*(-wz);
			double u, v;
			wf_equal_area_sphere_to_square((double)lx, (double)ly, (double)lz, u, v);
			int iu = (int)(u * g.nu); iu = iu < 0 ? 0 : (iu >= g.nu ? g.nu - 1 : iu);
			int iv = (int)(v * g.nv); iv = iv < 0 ? 0 : (iv >= g.nv ? g.nv - 1 : iv);
			float gonio = g.image[iv * g.nu + iu];
			float weight = g.scale * gonio / r2;
			Lr = g.ir * weight; Lg = g.ig * weight; Lb = g.ib * weight;
			break;
		}
		case PunctualLightKind::Projection: {
			const ProjectionLightGPU& pr = light.proj;
			float dx = pr.pos_x - p.x, dy = pr.pos_y - p.y, dz = pr.pos_z - p.z;
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < 1e-20f) return false;
			t_max = sqrtf(r2);
			float inv_r = 1.0f / t_max;
			wx = dx * inv_r; wy = dy * inv_r; wz = dz * inv_r;
			float lx = pr.world_to_light[0]*(-wx) + pr.world_to_light[1]*(-wy) + pr.world_to_light[2]*(-wz);
			float ly = pr.world_to_light[3]*(-wx) + pr.world_to_light[4]*(-wy) + pr.world_to_light[5]*(-wz);
			float lz = pr.world_to_light[6]*(-wx) + pr.world_to_light[7]*(-wy) + pr.world_to_light[8]*(-wz);
			if (lz < pr.hither) return false;
			float sx = pr.inv_tan * lx / lz;
			float sy = pr.inv_tan * ly / lz;
			if (sx < pr.sb_xmin || sx > pr.sb_xmax || sy < pr.sb_ymin || sy > pr.sb_ymax) return false;
			float u = (sx - pr.sb_xmin) / (pr.sb_xmax - pr.sb_xmin);
			float v = (sy - pr.sb_ymin) / (pr.sb_ymax - pr.sb_ymin);
			int iu = (int)(u * pr.nx); iu = iu < 0 ? 0 : (iu >= pr.nx ? pr.nx - 1 : iu);
			int iv = (int)(v * pr.ny); iv = iv < 0 ? 0 : (iv >= pr.ny ? pr.ny - 1 : iv);
			int idx = (iv * pr.nx + iu) * 3;
			float rC = fmaxf(0.0f, pr.image_rgb[idx + 0]);
			float gC = fmaxf(0.0f, pr.image_rgb[idx + 1]);
			float bC = fmaxf(0.0f, pr.image_rgb[idx + 2]);
			float inv_r2 = 1.0f / r2;
			Lr = pr.scale * rC * inv_r2; Lg = pr.scale * gC * inv_r2; Lb = pr.scale * bC * inv_r2;
			break;
		}
		default:
			return false;
	}
	if (Lr <= 0.0f && Lg <= 0.0f && Lb <= 0.0f) return false;
	wi = make_float3(wx, wy, wz);
	Li = make_float3(Lr, Lg, Lb);
	return true;
}

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
	switch (cam.kind) {
		case CameraKind::Orthographic: {
			origin = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			direction = cam.w;
			break;
		}
		case CameraKind::Spherical: {
			float theta = 3.14159265358979323846f * v;
			float phi   = 2.0f * 3.14159265358979323846f * u;
			float sin_t = sinf(theta), cos_t = cosf(theta);
			float lx = sin_t * cosf(phi);
			float ly = cos_t;
			float lz = sin_t * sinf(phi);
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

extern "C" __global__ void generate_camera_rays(
	WorkQueue<RayWorkItem> rayQueue,
	unsigned int width,
	unsigned int height,
	GpuCameraParams camera,
	unsigned int sampleIdx,
	unsigned int frameNumber
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= (int)width || py >= (int)height) return;

	int pixelIdx = py * (int)width + px;
	unsigned int seed = wf_pcg(wf_pcg(pixelIdx + sampleIdx * width * height) ^ frameNumber);

	float u = (float(px) + wf_rand(seed)) / float(width  - 1);
	// Flip Y to match optix_raygen.h's lower-left-origin viewport convention
	// (py=0/top row -> v=1, matching how lower_left_corner+u*horizontal+
	// v*vertical is constructed for Perspective/Orthographic, and how
	// Spherical's theta=pi*v expects v=0 at the bottom). Without this flip
	// every wavefront-mode render using those camera kinds came out
	// vertically mirrored relative to the recursive path.
	float v = (float(height - 1 - py) + wf_rand(seed)) / float(height - 1);

	RayWorkItem item;
	float cam_weight;
	wf_generate_primary_ray(camera, u, v, seed, item.origin, item.direction, cam_weight);
	// Sample hero wavelengths for spectral rendering (pbrt-v4: SampledWavelengths::SampleVisible)
	float lambda_u = wf_rand(seed);
	SampledWavelengths<kWFNWavelengths> swl = SampledWavelengths<kWFNWavelengths>::SampleVisible(lambda_u);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		item.throughput[i]     = cam_weight;
		item.radiance[i]       = 0.0f;
		item.wavelengths[i]    = swl.lambda[i];
		item.wavelength_pdfs[i] = swl.pdf[i];
	}
	item.seed       = seed;
	item.pixelIndex = pixelIdx;
	item.depth      = 0;
	item.specular_bounce = 1;  // primary ray: always allow emissive hit
	item.tMin       = 0.001f;
	item.tMax       = 1e30f;

	rayQueue.push(item);
}

// ============================================================================
// Kernel 2 — evaluate_materials
//   Processes HitWorkItems.  For each hit:
//     - Evaluates the BSDF (same logic as closesthit in optix_programs.cu)
//     - Pushes a ShadowRayWorkItem for direct-light NEE
//     - Pushes a RayWorkItem into nextRayQueue for the next bounce
//     - Emissive surfaces write directly to the framebuffer
// ============================================================================

extern "C" __global__ void evaluate_materials(
	WorkQueue<HitWorkItem>       hitQueue,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      framebuffer,
	// Scene data
	const SphereData*   spheres,
	const QuadData*     quads,
	const MaterialData* materials,
	const int*          lightIndices,
	const bool*         isLightSphere,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	int maxDepth
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];

	float3 normal    = h.normal;
	float3 hit_point = h.hitPoint;
	unsigned int seed = h.seed;

	// Reconstruct spectral path state from work item
	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SS throughput(h.throughput);
	SS radiance(h.radiance);
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = h.wavelengths[i];
		swl.pdf[i]    = h.wavelength_pdfs[i];
	}

	// Helper: convert SampledSpectrum to sRGB and atomicAdd to framebuffer
	auto addToFramebuffer = [&](int pixIdx, const SS& L) {
		auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		atomicAdd(&framebuffer[pixIdx].x, r);
		atomicAdd(&framebuffer[pixIdx].y, g);
		atomicAdd(&framebuffer[pixIdx].z, b);
	};

	// -------------------------------------------------------------------------
	// Emissive: add emission term, path terminates (no scatter).
	// pbrt-v4 alignment: only add emissive if depth==0 or specular_bounce==1
	// to avoid double-counting with NEE shadow rays at non-specular bounces.
	// -------------------------------------------------------------------------
	if (mat.type == MaterialType::DiffuseLight) {
		if (h.specular_bounce || h.depth == 0) {
			float cos_face = dot(-h.rayDir, normal);
			if (cos_face > 0.0f) {
				// Uplift RGB emission to spectrum via device sRGB table
				// (deferred: emissionSpectrum helper is declared below; use inline here)
				float3 le = mat.emission;
				float m_le = le.x > le.y ? (le.x > le.z ? le.x : le.z)
										  : (le.y > le.z ? le.y : le.z);
				float sc = 2.f * m_le;
				if (sc > 0.f) {
					float c0, c1, c2;
					dev_srgb_to_coeffs(le.x/sc, le.y/sc, le.z/sc, c0, c1, c2);
					RGBSigmoidPolynomial emitPoly(c0, c1, c2);
					SS emitS(0.f);
					for (int i = 0; i < kWFNWavelengths; ++i)
						emitS[i] = sc * emitPoly(swl.lambda[i]);
					radiance = radiance + throughput * emitS;
				}
			}
		}
		addToFramebuffer(h.pixelIndex, radiance);
		return; // path ends at emissive surface
	}

	if (h.depth >= maxDepth) {
		// Max depth reached — just accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

	// -------------------------------------------------------------------------
	// Scatter — evaluate BSDF (mirrors optix_programs.cu closesthit logic)
	// -------------------------------------------------------------------------
	float3 scattered_dir    = make_float3(0, 0, 0);
	SS     attenuation(0.f);   // spectral BSDF * cos / pdf
	bool   scattered        = false;
	bool   is_specular      = false;
	float  brdf_pdf_override = -1.0f;

	// Helper: uplift RGB albedo to SampledSpectrum via device sigmoid polynomial
	auto albedoSpectrum = [&](float3 rgb) -> SS {
		float c0, c1, c2;
		float r = rgb.x < 0.f ? 0.f : (rgb.x > 1.f ? 1.f : rgb.x);
		float g = rgb.y < 0.f ? 0.f : (rgb.y > 1.f ? 1.f : rgb.y);
		float b = rgb.z < 0.f ? 0.f : (rgb.z > 1.f ? 1.f : rgb.z);
		dev_srgb_to_coeffs(r, g, b, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = poly(swl.lambda[i]);
		return s;
	};
	auto emissionSpectrum = [&](float3 /*rgb*/) -> SS { return SS(0.f); };  // placeholder, liftEmission used inline
	(void)emissionSpectrum;

	// Uplift an unbounded-positive RGB weight (can exceed 1, unlike a plain
	// reflectance) to spectral: same normalize-by-2*max/uplift/rescale
	// technique as the NEE block's liftEmission lambda below - needed for
	// MaterialType::Hair, whose HairBxDF::sample() result already divides by
	// the sample pdf (a proper importance-sampling weight, not a [0,1]
	// albedo), so naively clamping it into albedoSpectrum's range would bias
	// bright/peaky fiber lobes toward black.
	auto unboundedSpectrum = [&](float3 rgb) -> SS {
		float m = rgb.x > rgb.y ? (rgb.x > rgb.z ? rgb.x : rgb.z) : (rgb.y > rgb.z ? rgb.y : rgb.z);
		float sc = 2.f * m;
		if (sc <= 0.f) return SS(0.f);
		float c0, c1, c2;
		dev_srgb_to_coeffs(rgb.x/sc, rgb.y/sc, rgb.z/sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = sc * poly(swl.lambda[i]);
		return s;
	};

	switch (mat.type) {
	case MaterialType::Lambertian: {
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		attenuation = albedoSpectrum(mat.albedo);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Metal: {
		float3 reflected = wf_reflect(normalize(h.rayDir), normal);
		scattered_dir    = normalize(reflected + mat.fuzz * wf_rand_unit(seed));
		attenuation      = albedoSpectrum(mat.albedo);
		scattered        = (dot(scattered_dir, normal) > 0.0f);
		is_specular      = true;
		break;
	}
	case MaterialType::Dielectric: {
		float eta = dot(h.rayDir, normal) < 0.0f ? (1.0f / mat.ior) : mat.ior;
		float3 unit_dir = normalize(h.rayDir);
		float  cos_t = fminf(dot(-unit_dir, normal), 1.0f);
		float  sin_t = sqrtf(1.0f - cos_t * cos_t);
		bool   cannot_refract = eta * sin_t > 1.0f;
		float  r0 = (1.0f - mat.ior) / (1.0f + mat.ior);
		r0 = r0 * r0;
		float schlick = r0 + (1.0f - r0) * powf(1.0f - cos_t, 5.0f);
		if (cannot_refract || schlick > wf_rand(seed))
			scattered_dir = wf_reflect(unit_dir, normal);
		else
			scattered_dir = wf_refract(unit_dir, normal, eta);
		attenuation = SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::RoughDielectric: {
		float rd_alpha = sqrtf(mat.fuzz);
		float3 n = normal;
		float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 tan_v  = normalize(cross(up_v, n));
		float3 bitan = cross(n, tan_v);
		float3 wi_w = -normalize(h.rayDir);
		float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
		if (wi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> rd_dist(rd_alpha, rd_alpha);
		float wm_x, wm_y, wm_z;
		rd_dist.Sample_wm(wi_x, wi_y, wi_z, wf_rand(seed), wf_rand(seed), wm_x, wm_y, wm_z);
		float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		float Fr = FrDielectric(rd_dot, mat.ior);
		if (wf_rand(seed) < Fr) {
			// Reflect
			float wo_x = 2.0f*rd_dot*wm_x - wi_x;
			float wo_y = 2.0f*rd_dot*wm_y - wi_y;
			float wo_z = 2.0f*rd_dot*wm_z - wi_z;
			scattered_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
		} else {
			// Refract
			float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
			float eta = dot(h.rayDir, normal) < 0.0f ? (1.0f / mat.ior) : mat.ior;
			scattered_dir = wf_refract(normalize(h.rayDir), wm_world, eta);
		}
		attenuation = albedoSpectrum(mat.albedo);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::Conductor: {
		float c_alpha = sqrtf(mat.fuzz);
		float3 cn = normal;
		float3 cup = (fabsf(cn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 ctan   = normalize(cross(cup, cn));
		float3 cbitan = cross(cn, ctan);
		float3 cwi = -normalize(h.rayDir);
		float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
		if (cwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> c_dist(c_alpha, c_alpha);
		float cwm_x, cwm_y, cwm_z;
		c_dist.Sample_wm(cwi_x, cwi_y, cwi_z, wf_rand(seed), wf_rand(seed), cwm_x, cwm_y, cwm_z);
		float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
		float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
		float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
		float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
		if (cwo_z <= 0.0f) { scattered = false; break; }
		float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
		float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
		float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;
		float3 c_F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
		// Use average Fresnel weight as scalar (conductor is specular, color from albedo)
		attenuation = albedoSpectrum(make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight));
		scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::CoatedDiffuse: {
		float cd_alpha = sqrtf(mat.fuzz);
		float3 cdn = normal;
		float3 cdup = (fabsf(cdn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 cdtan   = normalize(cross(cdup, cdn));
		float3 cdbitan = cross(cdn, cdtan);
		float3 cdwi = -normalize(h.rayDir);
		float cdwi_x = dot(cdwi, cdtan), cdwi_y = dot(cdwi, cdbitan), cdwi_z = dot(cdwi, cdn);
		TrowbridgeReitz<float> cd_dist(cd_alpha, cd_alpha);
		float cdwm_x, cdwm_y, cdwm_z;
		cd_dist.Sample_wm(cdwi_x, cdwi_y, cdwi_z, wf_rand(seed), wf_rand(seed), cdwm_x, cdwm_y, cdwm_z);
		float cd_dot = cdwi_x*cdwm_x + cdwi_y*cdwm_y + cdwi_z*cdwm_z;
		float Fr = FrDielectric(fmaxf(cd_dot, 0.0f), mat.ior);
		if (wf_rand(seed) < Fr) {
			float cdwo_x = 2.0f*cd_dot*cdwm_x - cdwi_x;
			float cdwo_y = 2.0f*cd_dot*cdwm_y - cdwi_y;
			float cdwo_z = 2.0f*cd_dot*cdwm_z - cdwi_z;
			scattered_dir = normalize(cdwo_x*cdtan + cdwo_y*cdbitan + cdwo_z*cdn);
			attenuation   = SS(1.f);
			is_specular   = true;
		} else {
			scattered_dir = normalize(normal + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = normal;
			attenuation = albedoSpectrum(mat.albedo);
			is_specular = false;
		}
		scattered = (dot(scattered_dir, normal) > 0.0f);
		break;
	}
	case MaterialType::ThinDielectric: {
		float3 V = -normalize(h.rayDir);
		float cos_i = fabsf(dot(V, normal));
		float Fr = FrDielectric(cos_i, mat.ior);
		// Account for double transmission (glass slab): T^2
		float T2 = (1.0f - Fr) * (1.0f - Fr);
		if (wf_rand(seed) < Fr / (Fr + T2)) {
			scattered_dir = wf_reflect(-V, normal);
		} else {
			scattered_dir = normalize(h.rayDir); // straight through
		}
		attenuation = SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::CoatedConductor: {
		// Rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF).
		// Matches optix_intersection_sphere.h's recursive-path handling: path B
		// (transmit into the coat, bounce off the conductor, exit back through
		// the coat) must weight the conductor's Fresnel by T_in*T_out (how
		// much light actually gets through the dielectric coat both ways) -
		// the previous version here resampled a microfacet from the ORIGINAL
		// viewing direction and skipped both the refraction-into-the-coat
		// step and the T_in/T_out weighting entirely, making the conductor
		// visible at full strength as if the coat weren't there.
		float cc_alpha = sqrtf(mat.fuzz);
		float3 ccn = normal;
		float3 ccup = (fabsf(ccn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 cctan   = normalize(cross(ccup, ccn));
		float3 ccbitan = cross(ccn, cctan);
		float3 ccwi = -normalize(h.rayDir);
		float ccwi_x = dot(ccwi, cctan), ccwi_y = dot(ccwi, ccbitan), ccwi_z = dot(ccwi, ccn);
		if (ccwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> cc_dist(cc_alpha, cc_alpha);
		float ccwm_x, ccwm_y, ccwm_z;
		cc_dist.Sample_wm(ccwi_x, ccwi_y, ccwi_z, wf_rand(seed), wf_rand(seed), ccwm_x, ccwm_y, ccwm_z);
		float cc_dot = ccwi_x*ccwm_x + ccwi_y*ccwm_y + ccwi_z*ccwm_z;
		float F_in = FrDielectric(fmaxf(cc_dot, 0.0f), mat.ior);
		float3 ccwo;
		if (wf_rand(seed) < F_in) {
			// Path A: coat specular reflection
			float wo_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float wo_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float wo_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			if (wo_z <= 0.0f) { scattered = false; break; }
			float G1 = cc_dist.G1(ccwi_x, ccwi_y, ccwi_z);
			float G  = cc_dist.G(wo_x, wo_y, wo_z, ccwi_x, ccwi_y, ccwi_z);
			float w  = (G1 > 1e-8f) ? G / G1 : 0.0f;
			float fv = F_in * w;
			attenuation = SS(fv);
			ccwo = make_float3(wo_x, wo_y, wo_z);
		} else {
			// Path B: transmit into layer -> conductor bounce -> exit coat
			float w_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float w_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float w_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			if (w_z > 0.0f) w_z = -w_z;   // ensure pointing downward into layer
			if (w_z == 0.0f) { scattered = false; break; }

			// Flip to conductor frame: "incoming from above" (fw_z > 0)
			float fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;

			float bwm_x, bwm_y, bwm_z;
			cc_dist.Sample_wm(fw_x, fw_y, fw_z, wf_rand(seed), wf_rand(seed), bwm_x, bwm_y, bwm_z);
			float cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
			if (cos_c <= 0.0f) { scattered = false; break; }

			float rwo_x = 2.0f*cos_c*bwm_x - fw_x;
			float rwo_y = 2.0f*cos_c*bwm_y - fw_y;
			float rwo_z = 2.0f*cos_c*bwm_z - fw_z;
			if (rwo_z <= 0.0f) { scattered = false; break; }

			float G1_c = cc_dist.G1(fw_x, fw_y, fw_z);
			float G_c  = cc_dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
			float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

			float3 c_F  = FrConductorRGB(cos_c, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);

			float F_out = FrDielectric(rwo_z, 1.0f / mat.ior);  // inside -> outside
			float T_out = 1.0f - F_out;
			float T_in  = 1.0f - F_in;

			attenuation = albedoSpectrum(make_float3(
				c_F.x * wt_c * T_in * T_out,
				c_F.y * wt_c * T_in * T_out,
				c_F.z * wt_c * T_in * T_out));
			ccwo = make_float3(rwo_x, rwo_y, rwo_z);
		}
		scattered_dir = normalize(ccwo.x*cctan + ccwo.y*ccbitan + ccwo.z*ccn);
		scattered   = (dot(scattered_dir, normal) > 0.0f);
		is_specular = true;
		break;
	}
	case MaterialType::DiffuseTransmission: {
		// pbrt-v4 DiffuseTransmissionBxDF - matches optix_intersection_sphere.h's
		// recursive-path handling exactly: albedo = reflectance R (same
		// hemisphere), emission = transmittance T (reused field, not
		// derived as 1-R), max-component probability weighting.
		float3 R = mat.albedo;
		float3 T_col = mat.emission;
		float pr = fmaxf(R.x, fmaxf(R.y, R.z));
		float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
		if (pr + pt <= 0.0f) { scattered = false; break; }
		if (wf_rand(seed) < pr / (pr + pt)) {
			scattered_dir = normalize(normal + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = normal;
			attenuation = albedoSpectrum(R);
		} else {
			float3 neg_n = -normal;
			scattered_dir = normalize(neg_n + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = neg_n;
			attenuation = albedoSpectrum(T_col);
		}
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::NormalizedFresnel: {
		float nf_eta = mat.ior;
		float inv_eta = 1.0f / nf_eta;
		float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
		if (nf_c <= 0.0f) nf_c = 1e-6f;
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		float cos_wi = fmaxf(dot(scattered_dir, normal), 1e-6f);
		float fr = FrDielectric(cos_wi, nf_eta);
		float weight = (1.0f - fr) / nf_c;
		attenuation = SS(weight);
		scattered   = true;
		is_specular = false;
		brdf_pdf_override = (1.0f - fr) * cos_wi / (nf_c * 3.14159265f);
		break;
	}
	case MaterialType::Medium: {
		// Homogeneous participating medium - mirrors optix_intersection_sphere.h's
		// closesthit Medium case. Geometry (entry/exit roots) was already
		// recomputed in __closesthit__wf_sphere and handed over via h.t (near)
		// and h.mediumTFar (far); here we sample the free-path distance and
		// either scatter inside via the HG phase function or pass straight
		// through to the exit point, matching the recursive path's
		// is_specular=true (no NEE/MIS for volume scattering, not yet implemented).
		float t_near = h.t;
		float t_far  = h.mediumTFar;
		float dist_inside = fmaxf(0.0f, t_far - t_near);
		float sigma_t = mat.ior;
		float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;
		float3 unit_dir = normalize(h.rayDir);
		if (free_path < dist_inside) {
			float medium_t = t_near + free_path;
			hit_point     = h.rayOrigin + medium_t * unit_dir;
			scattered_dir = wf_sample_henyey_greenstein(unit_dir, mat.fuzz, seed);
			attenuation   = albedoSpectrum(mat.albedo);
		} else {
			hit_point     = h.rayOrigin + t_far * unit_dir;
			scattered_dir = unit_dir;  // straight through, no interaction
			attenuation   = SS(1.f);
		}
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::Hair: {
		// Marschner/Chiang fiber scattering - see wf_sample_hair_material's
		// comment above. Matches hair_material.h's skip_pdf=true: no NEE/MIS
		// (is_specular=true), and the result needs unboundedSpectrum (not
		// albedoSpectrum) since it's already divided by the sample pdf.
		float3 sdir, atten;
		if (wf_sample_hair_material(h.rayDir, normal, mat, seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
		break;
	}
	default:
		scattered = false;
		break;
	}

	if (!scattered) {
		// Path absorbed — accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

	// -------------------------------------------------------------------------
	// NEE: direct-light shadow ray (non-specular materials only)
	// -------------------------------------------------------------------------
	if (!is_specular) {
	if (numLights > 0 && aliasTable) {
		// Pick a light via alias table
		int   slot = int(wf_rand(seed) * float(numLights));
		if (slot >= (int)numLights) slot = (int)numLights - 1;
		const GpuAliasEntry& entry = aliasTable[slot];
		int light_idx = (wf_rand(seed) < entry.q) ? slot : entry.alias;
		float selection_pdf = aliasTable[light_idx].pdf;

		int   prim_idx        = lightIndices[light_idx];
		bool  is_sphere_light = isLightSphere[light_idx];

		float  geom_pdf = 0.0f, max_dist = 0.0f;
		float3 to_light;
		SS light_emission_spec(0.f);

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
				s[i] = sc * poly(swl.lambda[i]);
			return s;
		};

		if (is_sphere_light) {
			const SphereData& s = spheres[prim_idx];
			to_light = wf_sample_sphere_light(s, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[s.materialIdx].emission);
		} else {
			const QuadData& q = quads[prim_idx];
			to_light = wf_sample_quad_light(q, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[q.materialIdx].emission);
		}

		float light_pdf = selection_pdf * geom_pdf;
		if (light_pdf > 1e-6f && dot(to_light, normal) > 0.0f) {
			float cos_l = fmaxf(dot(to_light, normal), 0.0f);
			float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
			// Lambertian's BRDF is albedo/pi, direction-independent - `attenuation`
			// already equals albedoSpectrum(mat.albedo) from the switch above, so
			// reuse it here instead of leaving the NEE contribution achromatic
			// (was previously missing entirely: every Lambertian surface
			// contributed as if albedo were white, regardless of actual color).
			// NormalizedFresnel's bsdf_val below is already a complete,
			// achromatic BRDF value (no color/texture involved), so it needs
			// no equivalent multiply.
			SS bsdf_color(1.f);
			if (mat.type == MaterialType::Lambertian) {
				bsdf_color = attenuation;
			} else if (mat.type == MaterialType::NormalizedFresnel) {
				float nf_eta = mat.ior;
				float inv_eta = 1.0f / nf_eta;
				float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
				if (nf_c <= 0.0f) nf_c = 1e-6f;
				float fr_l = FrDielectric(cos_l, nf_eta);
				bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
			}
			float brdf_pdf_l = (brdf_pdf_override > 0.0f) ? brdf_pdf_override : (bsdf_val * cos_l);
			float mis_w = wf_mis(light_pdf, brdf_pdf_l);

			// Spectral direct-light contribution
			SS Ld = (mis_w * bsdf_val * cos_l / light_pdf) * throughput * bsdf_color * light_emission_spec;

			ShadowRayWorkItem shadow;
			shadow.origin    = hit_point + 0.001f * normal;
			shadow.direction = to_light;
			shadow.tMax      = max_dist - 0.002f;
			for (int i = 0; i < kWFNWavelengths; ++i) {
				shadow.Ld[i]             = Ld[i];
				shadow.wavelengths[i]    = swl.lambda[i];
				shadow.wavelength_pdfs[i] = swl.pdf[i];
			}
			shadow.pixelIndex = h.pixelIndex;
			shadowQueue.push(shadow);
		}
	}

	// -------------------------------------------------------------------------
	// NEE: punctual (point/spot/distant) delta lights. Unlike the area-light
	// block above (one stochastic pick via alias table), every contributing
	// punctual light gets its own shadow ray - matches recursive path
	// (optix_device_helpers.h add_punctual_lights_lambertian) and the CPU
	// reference (camera.h punct_lights loop): pdf=1 by construction, so no
	// MIS weight or pdf division, just beta * BRDF * cos_theta * Li per light.
	// -------------------------------------------------------------------------
	for (unsigned int pli = 0; pli < numPunctualLights; ++pli) {
		float3 wi, Li; float t_max;
		if (!wf_eval_punctual_light(punctualLights[pli], hit_point, wi, Li, t_max)) continue;
		float cos_l = dot(wi, normal);
		if (cos_l <= 0.0f) continue;

		float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
		// See the area-light block above: attenuation == albedoSpectrum(mat.albedo)
		// for Lambertian (direction-independent BRDF, safe to reuse here);
		// NormalizedFresnel's bsdf_val is already a complete achromatic value.
		SS bsdf_color(1.f);
		if (mat.type == MaterialType::Lambertian) {
			bsdf_color = attenuation;
		} else if (mat.type == MaterialType::NormalizedFresnel) {
			float nf_eta = mat.ior;
			float inv_eta = 1.0f / nf_eta;
			float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
			if (nf_c <= 0.0f) nf_c = 1e-6f;
			float fr_l = FrDielectric(cos_l, nf_eta);
			bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
		}

		// Uplift RGB Li to spectrum (same pattern as liftEmission above)
		float m = Li.x > Li.y ? (Li.x > Li.z ? Li.x : Li.z) : (Li.y > Li.z ? Li.y : Li.z);
		float sc = 2.f * m;
		SS Li_spec(0.f);
		if (sc > 0.f) {
			float c0, c1, c2;
			dev_srgb_to_coeffs(Li.x / sc, Li.y / sc, Li.z / sc, c0, c1, c2);
			RGBSigmoidPolynomial poly(c0, c1, c2);
			for (int i = 0; i < kWFNWavelengths; ++i)
				Li_spec[i] = sc * poly(swl.lambda[i]);
		}

		SS Ld = (bsdf_val * cos_l) * throughput * bsdf_color * Li_spec;

		ShadowRayWorkItem shadow;
		shadow.origin    = hit_point + 0.001f * normal;
		shadow.direction = wi;
		shadow.tMax      = t_max - 0.002f;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			shadow.Ld[i]              = Ld[i];
			shadow.wavelengths[i]     = swl.lambda[i];
			shadow.wavelength_pdfs[i] = swl.pdf[i];
		}
		shadow.pixelIndex = h.pixelIndex;
		shadowQueue.push(shadow);
	}

	// Flush accumulated prior-bounce radiance (once, regardless of whether
	// any area/punctual lights actually contributed above).
	if ((bool)radiance) addToFramebuffer(h.pixelIndex, radiance);
	} // if (!is_specular) - NEE (area + punctual lights)

	// -------------------------------------------------------------------------
	// Bounce: push next ray
	// -------------------------------------------------------------------------
	SS new_throughput = throughput * attenuation;

	// Russian roulette after depth 3
	if (h.depth >= 3) {
		float p = new_throughput.MaxComponentValue();
		if (wf_rand(seed) >= p) {
			if (is_specular) addToFramebuffer(h.pixelIndex, radiance);
			return;
		}
		new_throughput = new_throughput / p;
	}

	RayWorkItem next;
	next.origin     = hit_point + 0.001f * scattered_dir;
	next.direction  = normalize(scattered_dir);
	next.seed            = seed;
	next.pixelIndex      = h.pixelIndex;
	next.depth           = h.depth + 1;
	next.specular_bounce = is_specular ? 1 : 0;
	next.tMin       = 0.001f;
	next.tMax       = 1e30f;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		next.throughput[i]      = new_throughput[i];
		next.radiance[i]        = is_specular ? radiance[i] : 0.0f;
		next.wavelengths[i]     = swl.lambda[i];
		next.wavelength_pdfs[i] = swl.pdf[i];
	}
	nextRayQueue.push(next);
}

// Uplift a flat RGB color to a spectral sample at the given hero
// wavelengths. Standalone copy of the `liftEmission` lambda used inside
// evaluate_materials below (that one captures its `swl` by reference from
// its own hit context; accumulate_miss needs the same math against a
// MissWorkItem's own wavelengths instead).
__device__ __forceinline__ SampledSpectrum<kWFNWavelengths> wf_lift_rgb_to_spectrum(
	float3 rgb, const SampledWavelengths<kWFNWavelengths>& swl
) {
	using SS = SampledSpectrum<kWFNWavelengths>;
	float m = rgb.x > rgb.y ? (rgb.x > rgb.z ? rgb.x : rgb.z) : (rgb.y > rgb.z ? rgb.y : rgb.z);
	float sc = 2.f * m;
	if (sc <= 0.f) return SS(0.f);
	float c0, c1, c2;
	dev_srgb_to_coeffs(rgb.x / sc, rgb.y / sc, rgb.z / sc, c0, c1, c2);
	RGBSigmoidPolynomial poly(c0, c1, c2);
	SS s(0.f);
	for (int i = 0; i < kWFNWavelengths; ++i)
		s[i] = sc * poly(swl.lambda[i]);
	return s;
}

// ============================================================================
// Kernel 3 — accumulate_miss
//   For rays that escaped the scene, add the flat background color (black
//   by default - see GpuCameraParams::backgroundColor in optix_types.h for
//   why a flat color, not an environment map, matches what the CPU
//   renderer actually does for every scene).
// ============================================================================

extern "C" __global__ void accumulate_miss(
	WorkQueue<MissWorkItem> missQueue,
	int                     numMiss,
	float3*                 framebuffer,
	float3                  backgroundColor
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numMiss) return;
	if (backgroundColor.x <= 0.0f && backgroundColor.y <= 0.0f && backgroundColor.z <= 0.0f) return;

	const MissWorkItem& m = missQueue.items[idx];
	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = m.wavelengths[i];
		swl.pdf[i]    = m.wavelength_pdfs[i];
	}
	SS throughput(m.throughput);
	SS L = throughput * wf_lift_rgb_to_spectrum(backgroundColor, swl);

	auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z, kDevCIEMin, kDevCIENSamples);
	float r, g, b;
	wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
	atomicAdd(&framebuffer[m.pixelIndex].x, r);
	atomicAdd(&framebuffer[m.pixelIndex].y, g);
	atomicAdd(&framebuffer[m.pixelIndex].z, b);
}

// ============================================================================
// Kernel 4 — accumulate_shadow
//   Runs after the OptiX shadow-trace pass.  ShadowRayWorkItems that were NOT
//   occluded have their `occluded` flag cleared by the shadow miss program;
//   we accumulate their Ld into the framebuffer.
//   (The `occluded` flag is stored in a separate bool array passed alongside
//    the shadow queue items — see wavefront_path_tracer.cpp.)
// ============================================================================

extern "C" __global__ void accumulate_shadow(
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	int                          numShadow,
	const bool*                  occluded,   // per-item occlusion result
	float3*                      framebuffer
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numShadow) return;

	if (!occluded[idx]) {
		const ShadowRayWorkItem& s = shadowQueue.items[idx];
		// Reconstruct spectral wavelengths for XYZ conversion
		using SS  = SampledSpectrum<kWFNWavelengths>;
		using SWL = SampledWavelengths<kWFNWavelengths>;
		SS Ld(s.Ld);
		SWL swl;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			swl.lambda[i] = s.wavelengths[i];
			swl.pdf[i]    = s.wavelength_pdfs[i];
		}
		auto xyz = SampledSpectrumToXYZ(Ld, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		atomicAdd(&framebuffer[s.pixelIndex].x, r);
		atomicAdd(&framebuffer[s.pixelIndex].y, g);
		atomicAdd(&framebuffer[s.pixelIndex].z, b);
	}
}

// ============================================================================
// Kernel 5 — reset_queue_counter  (single-thread helper)
// ============================================================================

extern "C" __global__ void reset_queue_counter(int* counter) {
	*counter = 0;
}

// ============================================================================
// Kernel 6 — normalize_framebuffer
//   Divides accumulated radiance by samplesPerPixel for the final image.
// ============================================================================

extern "C" __global__ void normalize_framebuffer(
	float3*      framebuffer,
	unsigned int numPixels,
	float        invSPP
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	framebuffer[idx] = framebuffer[idx] * invSPP;
}
