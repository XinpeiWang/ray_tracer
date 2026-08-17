// optix_sky_light.h -- Device-side real importance-sampled HDR sky
// (LightSource "infinite" with an image, recursive GPU backend only - see
// wavefront_sky_light.h for the wavefront backend's own copy). Included by
// optix_device_helpers.h, AFTER the `extern "C" { __constant__ LaunchParams
// params; }` declaration and after random_float()/random_unit_vector() are
// defined (these functions read params.camera.skyDist directly and call
// random_float() for their own samples), same inclusion-order requirement as
// optix_measured_bxdf.h just above it.
//
// A float, raw-pointer/array reimplementation of:
//   - src/shared/piecewise_dist.h's PiecewiseConstant1D::sample()/pdf() and
//     PiecewiseConstant2D::sample()/pdf(),
//   - src/TheRestOfYourLife/sky_light.h's dir_to_uv()/sample_Le()/pdf_Li()/
//     Le() (the equirectangular mapping + Jacobian correction - see that
//     file's own header comment for the derivation, NOT re-derived here).
// NOT a call to those CPU functions, which use std::vector and double
// precision throughout. Every call site that used to always take the
// uniform-sphere + flat-colour path (this codebase's own earlier sky-NEE
// work, commit 7883ad9) now dispatches here FIRST when params.camera.
// skyDist.height > 0 (a real image was uploaded - see GpuSkyDistribution's
// own comment in optix_types.h), falling back to that exact same uniform-
// sphere + flat-colour code for a constant-colour sky or no sky at all.

// ---------------------------------------------------------------------------
// pc1d_sample() / pc1d_pdf() -- PiecewiseConstant1D::sample()/pdf() ports.
// `cdf` has n+1 entries, `func` has n entries, matching piecewise_dist.h's
// own layout exactly (cdf[0]=0, cdf[n]=1, func_int already folded out of the
// upload - see pbrt_gpu_builder.h). func_int is guaranteed > 0 here: a
// degenerate (all-zero-weight) row/marginal is normalized to a uniform
// distribution CPU-side, at construction time, before upload - see
// PiecewiseConstant1D's own constructor comment - so device code never needs
// to special-case it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pc1d_sample(const float* cdf, const float* func, float func_int,
											   int n, float u, float& pdf_out) {
	// Binary search: largest o such that cdf[o] <= u (mirrors std::upper_
	// bound(u)-1, clamped to a valid bin).
	int lo = 0, hi = n;
	while (lo < hi) {
		int mid = (lo + hi) >> 1;
		if (cdf[mid] <= u) lo = mid + 1; else hi = mid;
	}
	int o = lo - 1;
	if (o < 0) o = 0;
	if (o >= n) o = n - 1;

	float du = u - cdf[o];
	float dcdf = cdf[o + 1] - cdf[o];
	if (dcdf > 0.0f) du /= dcdf;

	pdf_out = func[o] / func_int;
	return (o + du) / (float)n;
}

__device__ __forceinline__ float pc1d_pdf(const float* func, float func_int, int n, float x) {
	int o = (int)(x * (float)n);
	if (o < 0) o = 0;
	if (o >= n) o = n - 1;
	return func[o] / func_int;
}

// ---------------------------------------------------------------------------
// pc2d_sample() / pc2d_pdf() -- PiecewiseConstant2D::sample()/pdf() ports.
// Row r's conditional cdf/func slices sit at fixed strides (r*(width+1) /
// r*width) since every row shares the same width - unlike GpuPL2DTable's
// per-axis variable offsets, this needs no per-row offset table.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pc2d_sample(const GpuSkyDistribution& d, float ru, float rv,
											  float& u_out, float& v_out, float& pdf_out) {
	float pdf_v;
	float v = pc1d_sample(d.marginalCdf, d.marginalFunc, d.marginalFuncInt, d.height, rv, pdf_v);

	int row = (int)(v * (float)d.height);
	if (row < 0) row = 0;
	if (row >= d.height) row = d.height - 1;

	const float* rowCdf  = d.conditionalCdf  + row * (d.width + 1);
	const float* rowFunc = d.conditionalFunc + row * d.width;
	const float  rowFuncInt = d.conditionalFuncInt[row];

	float pdf_u;
	float u = pc1d_sample(rowCdf, rowFunc, rowFuncInt, d.width, ru, pdf_u);

	u_out = u; v_out = v; pdf_out = pdf_u * pdf_v;
}

__device__ __forceinline__ float pc2d_pdf(const GpuSkyDistribution& d, float u, float v) {
	int row = (int)(v * (float)d.height);
	if (row < 0) row = 0;
	if (row >= d.height) row = d.height - 1;

	const float* rowFunc = d.conditionalFunc + row * d.width;
	const float  rowFuncInt = d.conditionalFuncInt[row];

	const float pdf_u = pc1d_pdf(rowFunc, rowFuncInt, d.width, u);
	const float pdf_v = pc1d_pdf(d.marginalFunc, d.marginalFuncInt, d.height, v);
	return pdf_u * pdf_v;
}

// ---------------------------------------------------------------------------
// sky_dir_to_uv() / sky_uv_to_dir() -- the equirectangular mapping, mirrors
// sky_light.h's private dir_to_uv() and sample_Le()'s inverse exactly
// (theta=acos(-dir.y), phi=atan2(-dir.z,dir.x)+pi -- and back).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void sky_dir_to_uv(const float3& dir, float& u, float& v) {
	const float kPi = 3.14159265358979323846f;
	float ct = -dir.y;
	ct = fminf(1.0f, fmaxf(-1.0f, ct));
	const float theta = acosf(ct);
	const float phi = atan2f(-dir.z, dir.x) + kPi;
	u = phi / (2.0f * kPi);
	v = theta / kPi;
}

__device__ __forceinline__ float3 sky_uv_to_dir(float u, float v, float& sin_theta_out) {
	const float kPi = 3.14159265358979323846f;
	const float phi = u * 2.0f * kPi;
	const float theta = v * kPi;
	const float sin_theta = sinf(theta);
	const float cos_theta = cosf(theta);
	sin_theta_out = sin_theta;
	return normalize(make_float3(sin_theta * cosf(phi), cos_theta, -sin_theta * sinf(phi)));
}

// ---------------------------------------------------------------------------
// sky_Le() -- per-direction radiance lookup. Mirrors sky_light::Le(), which
// samples env_tex (hdr_image_texture::value()) at dir_to_uv(dir): that
// function flips v (v_image = 1-v, image origin top-left) and nearest-
// samples, clamped [0,size) - both replicated exactly here, including the
// v-flip, which is NOT the same convention the distribution's own row index
// uses (see PiecewiseConstant2D's own construction in pbrt_gpu_builder.h,
// which indexes image row v directly, unflipped) - a real but harmless CPU
// quirk (affects sampling efficiency, not correctness - see this codebase's
// own notes) that must be mirrored bit-for-bit for GPU/CPU visual parity,
// not "fixed" here.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 sky_Le(const GpuSkyDistribution& d, const float3& dir) {
	float u, v;
	sky_dir_to_uv(dir, u, v);
	u = fminf(1.0f, fmaxf(0.0f, u));
	v = fminf(1.0f, fmaxf(0.0f, v));
	const float v_img = 1.0f - v; // mirrors hdr_image_texture::value()'s v-flip

	int i = (int)(u * (float)d.width);
	int j = (int)(v_img * (float)d.height);
	if (i < 0) i = 0; if (i >= d.width)  i = d.width  - 1;
	if (j < 0) j = 0; if (j >= d.height) j = d.height - 1;

	const float* px = d.imagePixels + (j * d.width + i) * 3;
	return d.scale * make_float3(px[0], px[1], px[2]);
}

// ---------------------------------------------------------------------------
// sky_sample_Li() -- direction + solid-angle pdf for one NEE sample. Mirrors
// sky_light::sample_Le()'s image branch (the Jacobian pdf_omega = pdf_image /
// (2*pi^2*sin_theta)). Returns false at the pole-singularity guard (mirrors
// CPU's own `if (sin_theta < 1e-10)` fallback) - the caller should then take
// the SAME uniform-sphere + flat-colour path it already has for a constant-
// colour sky, exactly as sky_light::sample_Le() itself falls back to
// random_unit_vector()/1/(4*pi) there.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool sky_sample_Li(const GpuSkyDistribution& d, unsigned int& seed,
												float3& dir_out, float& pdf_out) {
	const float ru = random_float(seed), rv = random_float(seed);
	float us, vs, pdf_img;
	pc2d_sample(d, ru, rv, us, vs, pdf_img);

	float sin_theta;
	const float3 dir = sky_uv_to_dir(us, vs, sin_theta);
	if (sin_theta < 1e-6f) return false;

	const float kPi = 3.14159265358979323846f;
	dir_out = dir;
	pdf_out = pdf_img / (2.0f * kPi * kPi * sin_theta);
	return true;
}

// pdf_Li() -- solid-angle PDF for a GIVEN direction (MIS weight in the miss
// case + NEE). Mirrors sky_light::pdf_Li(vec3)'s image branch exactly,
// including its own pole guard (returns 0, matching CPU).
__device__ __forceinline__ float sky_pdf_Li(const GpuSkyDistribution& d, const float3& dir) {
	float u, v;
	sky_dir_to_uv(dir, u, v);
	const float kPi = 3.14159265358979323846f;
	const float sin_theta = sinf(v * kPi);
	if (sin_theta < 1e-6f) return 0.0f;
	return pc2d_pdf(d, u, v) / (2.0f * kPi * kPi * sin_theta);
}

// ---------------------------------------------------------------------------
// High-level wrappers used by shade_material()'s NEE blocks and
// __miss__ms() (optix_miss.h). Dispatch between the real image-based
// machinery above (params.camera.skyDist.height > 0) and the existing
// uniform-sphere + flat-colour path (unchanged), exactly mirroring sky_
// light's own dual-mode behavior (has_dist true/false).
// ---------------------------------------------------------------------------

// One sky NEE sample: direction, its solid-angle pdf, and the radiance
// arriving from it (Le at that exact direction for an image sky, or the flat
// `flatColor` otherwise).
__device__ __forceinline__ void sample_sky_nee(unsigned int& seed, const float3& flatColor,
												  float3& dir_out, float& pdf_out, float3& Le_out) {
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0 && sky_sample_Li(d, seed, dir_out, pdf_out)) {
		Le_out = sky_Le(d, dir_out);
		return;
	}
	const float kPi = 3.14159265358979323846f;
	dir_out = random_unit_vector(seed);
	pdf_out = 1.0f / (4.0f * kPi);
	Le_out = flatColor;
}

// Solid-angle pdf of the sky strategy at a SPECIFIC direction (for MIS
// against a BSDF-sampled escape - __miss__ms()). Uniform-sphere constant when
// no image was uploaded, matching the existing behavior exactly.
__device__ __forceinline__ float sky_pdf_for_mis(const float3& dir) {
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) return sky_pdf_Li(d, dir);
	const float kPi = 3.14159265358979323846f;
	return 1.0f / (4.0f * kPi);
}

// Radiance arriving from a specific (escaped-ray) direction - real per-
// direction Le() for an image sky, else the flat `flatColor`.
__device__ __forceinline__ float3 sky_radiance(const float3& dir, const float3& flatColor) {
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) return sky_Le(d, dir);
	return flatColor;
}
