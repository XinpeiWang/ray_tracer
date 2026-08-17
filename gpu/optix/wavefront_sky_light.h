// wavefront_sky_light.h -- Device-side real importance-sampled HDR sky for
// the wavefront GPU backend (mirrors optix_sky_light.h function-for-function
// with a wf_ prefix, matching this backend's own established "duplicate,
// don't share, device helpers with the recursive path" convention - see
// wavefront_measured_bxdf.h's own header comment for the full rationale).
// Included by wavefront_kernels.cu, after wf_rand()/wf_rand_unit() are
// defined (wf_sky_sample_Li() calls wf_rand()).
//
// Unlike optix_sky_light.h, these functions take an explicit
// `const GpuSkyDistribution&` parameter rather than reading a `params`
// global - evaluate_materials/accumulate_miss are plain CUDA kernels, not
// OptiX programs, so every array/struct they need arrives as an explicit
// parameter (same reasoning as wavefront_measured_bxdf.h's own functions
// taking explicit table/array parameters instead of a launch-params global).

// ---------------------------------------------------------------------------
// wf_pc1d_sample() / wf_pc1d_pdf() -- see optix_sky_light.h's pc1d_sample()/
// pc1d_pdf() for the full algorithm comment.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wf_pc1d_sample(const float* cdf, const float* func, float func_int,
												   int n, float u, float& pdf_out) {
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

__device__ __forceinline__ float wf_pc1d_pdf(const float* func, float func_int, int n, float x) {
	int o = (int)(x * (float)n);
	if (o < 0) o = 0;
	if (o >= n) o = n - 1;
	return func[o] / func_int;
}

// wf_pc2d_sample() / wf_pc2d_pdf() -- see optix_sky_light.h's pc2d_sample()/
// pc2d_pdf().
__device__ __forceinline__ void wf_pc2d_sample(const GpuSkyDistribution& d, float ru, float rv,
												  float& u_out, float& v_out, float& pdf_out) {
	float pdf_v;
	float v = wf_pc1d_sample(d.marginalCdf, d.marginalFunc, d.marginalFuncInt, d.height, rv, pdf_v);

	int row = (int)(v * (float)d.height);
	if (row < 0) row = 0;
	if (row >= d.height) row = d.height - 1;

	const float* rowCdf  = d.conditionalCdf  + row * (d.width + 1);
	const float* rowFunc = d.conditionalFunc + row * d.width;
	const float  rowFuncInt = d.conditionalFuncInt[row];

	float pdf_u;
	float u = wf_pc1d_sample(rowCdf, rowFunc, rowFuncInt, d.width, ru, pdf_u);

	u_out = u; v_out = v; pdf_out = pdf_u * pdf_v;
}

__device__ __forceinline__ float wf_pc2d_pdf(const GpuSkyDistribution& d, float u, float v) {
	int row = (int)(v * (float)d.height);
	if (row < 0) row = 0;
	if (row >= d.height) row = d.height - 1;

	const float* rowFunc = d.conditionalFunc + row * d.width;
	const float  rowFuncInt = d.conditionalFuncInt[row];

	const float pdf_u = wf_pc1d_pdf(rowFunc, rowFuncInt, d.width, u);
	const float pdf_v = wf_pc1d_pdf(d.marginalFunc, d.marginalFuncInt, d.height, v);
	return pdf_u * pdf_v;
}

// wf_sky_dir_to_uv() / wf_sky_uv_to_dir() -- see optix_sky_light.h's
// sky_dir_to_uv()/sky_uv_to_dir().
__device__ __forceinline__ void wf_sky_dir_to_uv(const float3& dir, float& u, float& v) {
	const float kPi = 3.14159265358979323846f;
	float ct = -dir.y;
	ct = fminf(1.0f, fmaxf(-1.0f, ct));
	const float theta = acosf(ct);
	const float phi = atan2f(-dir.z, dir.x) + kPi;
	u = phi / (2.0f * kPi);
	v = theta / kPi;
}

__device__ __forceinline__ float3 wf_sky_uv_to_dir(float u, float v, float& sin_theta_out) {
	const float kPi = 3.14159265358979323846f;
	const float phi = u * 2.0f * kPi;
	const float theta = v * kPi;
	const float sin_theta = sinf(theta);
	const float cos_theta = cosf(theta);
	sin_theta_out = sin_theta;
	return normalize(make_float3(sin_theta * cosf(phi), cos_theta, -sin_theta * sinf(phi)));
}

// wf_sky_Le() -- see optix_sky_light.h's sky_Le() (same v-flip/nearest-
// sample quirk mirrored from hdr_image_texture::value(), same reasoning).
__device__ __forceinline__ float3 wf_sky_Le(const GpuSkyDistribution& d, const float3& dir) {
	float u, v;
	wf_sky_dir_to_uv(dir, u, v);
	u = fminf(1.0f, fmaxf(0.0f, u));
	v = fminf(1.0f, fmaxf(0.0f, v));
	const float v_img = 1.0f - v;

	int i = (int)(u * (float)d.width);
	int j = (int)(v_img * (float)d.height);
	if (i < 0) i = 0; if (i >= d.width)  i = d.width  - 1;
	if (j < 0) j = 0; if (j >= d.height) j = d.height - 1;

	const float* px = d.imagePixels + (j * d.width + i) * 3;
	return d.scale * make_float3(px[0], px[1], px[2]);
}

// wf_sky_sample_Li() / wf_sky_pdf_Li() -- see optix_sky_light.h's
// sky_sample_Li()/sky_pdf_Li().
__device__ __forceinline__ bool wf_sky_sample_Li(const GpuSkyDistribution& d, unsigned int& seed,
													float3& dir_out, float& pdf_out) {
	const float ru = wf_rand(seed), rv = wf_rand(seed);
	float us, vs, pdf_img;
	wf_pc2d_sample(d, ru, rv, us, vs, pdf_img);

	float sin_theta;
	const float3 dir = wf_sky_uv_to_dir(us, vs, sin_theta);
	if (sin_theta < 1e-6f) return false;

	const float kPi = 3.14159265358979323846f;
	dir_out = dir;
	pdf_out = pdf_img / (2.0f * kPi * kPi * sin_theta);
	return true;
}

__device__ __forceinline__ float wf_sky_pdf_Li(const GpuSkyDistribution& d, const float3& dir) {
	float u, v;
	wf_sky_dir_to_uv(dir, u, v);
	const float kPi = 3.14159265358979323846f;
	const float sin_theta = sinf(v * kPi);
	if (sin_theta < 1e-6f) return 0.0f;
	return wf_pc2d_pdf(d, u, v) / (2.0f * kPi * kPi * sin_theta);
}

// ---------------------------------------------------------------------------
// High-level wrappers used by wf_finish_material_scatter()'s sky-NEE block
// and accumulate_miss(). See optix_sky_light.h's own equivalents for the
// full comment - identical dispatch, just taking `d` as an explicit
// parameter instead of reading params.camera.skyDist.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wf_sample_sky_nee(const GpuSkyDistribution& d, unsigned int& seed,
													 const float3& flatColor,
													 float3& dir_out, float& pdf_out, float3& Le_out) {
	if (d.height > 0 && d.width > 0 && wf_sky_sample_Li(d, seed, dir_out, pdf_out)) {
		Le_out = wf_sky_Le(d, dir_out);
		return;
	}
	dir_out = wf_rand_unit(seed);
	pdf_out = 1.0f / (4.0f * 3.14159265f);
	Le_out = flatColor;
}

__device__ __forceinline__ float wf_sky_pdf_for_mis(const GpuSkyDistribution& d, const float3& dir) {
	if (d.height > 0 && d.width > 0) return wf_sky_pdf_Li(d, dir);
	return 1.0f / (4.0f * 3.14159265f);
}

__device__ __forceinline__ float3 wf_sky_radiance(const GpuSkyDistribution& d, const float3& dir,
													 const float3& flatColor) {
	if (d.height > 0 && d.width > 0) return wf_sky_Le(d, dir);
	return flatColor;
}
