// optix_bssrdf.h -- Device-side tabulated-BSSRDF evaluation (recursive GPU
// backend only, Phase 1). Included by optix_device_helpers.h, AFTER the
// `extern "C" { __constant__ LaunchParams params; }` declaration (these
// functions read the flat bssrdfRhoSamples/bssrdfRadiusSamples/bssrdfProfile/
// bssrdfProfileCdf arrays directly off `params`).
//
// A float, raw-pointer/array reimplementation of src/shared/bssrdf.h's
// TabulatedBSSRDF::sr()/pdf_sr()/sample_sr() and the Catmull-Rom spline
// machinery they depend on (src/shared/sampling_2d.h's catmull_rom_weights/
// SampleCatmullRom2D) - NOT a call to those CPU functions, which use
// std::vector/std::pair/lambdas and double precision throughout, none of
// which this codebase's other device code relies on (no nvcc
// --extended-lambda flag is set for this backend's compile - see
// build_optix.targets' OptixFlags - so this file deliberately avoids
// lambdas, matching every other device header here).
//
// GpuBssrdfTable (optix_types.h) holds a table's shape (n_rho/n_radius) and
// offsets into the four shared flat device buffers this file reads through
// `params`; see that struct's own comment for why rho_eff is derived from
// bssrdfProfileCdf's last column rather than uploaded separately.

// ---------------------------------------------------------------------------
// gpu_find_interval_le -- binary search for the largest i in [0, n-2] such
// that arr[i] <= x. Mirrors src/shared/sampling_2d.h's
// catmullrom_find_interval(n, pred) specialised to pred(i) = arr[i] <= x,
// the only predicate this file's callers need (raw-array node/CDF lookups).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int gpu_find_interval_le(const float* arr, int n, float x) {
	int size = n - 2, first = 1;
	while (size > 0) {
		int half = size >> 1;
		int middle = first + half;
		if (arr[middle] <= x) { first = middle + 1; size -= half + 1; }
		else                  { size = half; }
	}
	int r = first - 1;
	if (r < 0)     r = 0;
	if (r > n - 2) r = n - 2;
	return r;
}

// Catmull-Rom interpolation weights for value x among `nodes` (n of them).
// Returns false (weights untouched) if x is out of [nodes[0], nodes[n-1]].
// *offset is the index of the FIRST of 4 contributing nodes (idx-1); a
// weight is exactly 0 for any of the 4 whose node index would fall outside
// [0,n) - callers still guard the array read anyway (GPU OOB reads are a
// fatal illegal-memory-access, unlike CPU UB that often silently "worked").
// Direct float port of src/shared/sampling_2d.h's catmull_rom_weights().
__device__ __forceinline__ bool gpu_catmull_rom_weights(const float* nodes, int n, float x,
														  int* offset, float weights[4]) {
	if (!(x >= nodes[0] && x <= nodes[n - 1])) return false;
	const int idx = gpu_find_interval_le(nodes, n, x);
	*offset = idx - 1;
	const float x0 = nodes[idx], x1 = nodes[idx + 1];
	const float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
	weights[1] =  2.0f * t3 - 3.0f * t2 + 1.0f;
	weights[2] = -2.0f * t3 + 3.0f * t2;
	if (idx > 0) {
		const float w0 = (t3 - 2.0f * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
		weights[0] = -w0;
		weights[2] += w0;
	} else {
		const float w0 = t3 - 2.0f * t2 + t;
		weights[0] = 0.0f;
		weights[1] -= w0;
		weights[2] += w0;
	}
	if (idx + 2 < n) {
		const float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
		weights[1] -= w3;
		weights[3] = w3;
	} else {
		const float w3 = t3 - t2;
		weights[1] -= w3;
		weights[2] += w3;
		weights[3] = 0.0f;
	}
	return true;
}

// sr()/pdf_sr() -- weighted 4x4 (rho x radius) table lookups. Both share
// this same interpolation shape; sr() divides by sigma_t^2's inverse
// (converting from optical-radius space back to rendering space) while
// pdf_sr() additionally normalises by the row's effective albedo.
// Mirrors src/shared/bssrdf.h's TabulatedBSSRDF::sr()/pdf_sr() exactly.
__device__ __forceinline__ float gpu_bssrdf_sr(const GpuBssrdfTable& tab, float sigma_t, float rho, float r) {
	const float kPi = 3.14159265358979323846f;
	const float r_opt = r * sigma_t;
	const float* rho_samples    = params.bssrdfRhoSamples + tab.rho_offset;
	const float* radius_samples = params.bssrdfRadiusSamples + tab.radius_offset;
	const float* profile        = params.bssrdfProfile + tab.profile_offset;

	int rho_off, rad_off;
	float rho_w[4], rad_w[4];
	if (!gpu_catmull_rom_weights(rho_samples, tab.n_rho, rho, &rho_off, rho_w)) return 0.0f;
	if (!gpu_catmull_rom_weights(radius_samples, tab.n_radius, r_opt, &rad_off, rad_w)) return 0.0f;

	float val = 0.0f;
	for (int j = 0; j < 4; ++j) {
		if (rho_w[j] == 0.0f) continue;
		const int ri = rho_off + j;
		if (ri < 0 || ri >= tab.n_rho) continue;
		for (int k = 0; k < 4; ++k) {
			if (rad_w[k] == 0.0f) continue;
			const int rj = rad_off + k;
			if (rj < 0 || rj >= tab.n_radius) continue;
			val += rho_w[j] * rad_w[k] * profile[ri * tab.n_radius + rj];
		}
	}
	if (r_opt != 0.0f) val /= (2.0f * kPi * r_opt);
	val = fmaxf(0.0f, val);
	return val * sigma_t * sigma_t;
}

__device__ __forceinline__ float gpu_bssrdf_pdf_sr(const GpuBssrdfTable& tab, float sigma_t, float rho, float r) {
	const float kPi = 3.14159265358979323846f;
	const float r_opt = r * sigma_t;
	const float* rho_samples    = params.bssrdfRhoSamples + tab.rho_offset;
	const float* radius_samples = params.bssrdfRadiusSamples + tab.radius_offset;
	const float* profile        = params.bssrdfProfile + tab.profile_offset;
	// profile_cdf's last column IS rho_eff (integrate_catmull_rom's returned
	// total equals its own cdf[n-1] - see GpuBssrdfTable's comment), so no
	// separate rho_eff array is needed.
	const float* profile_cdf    = params.bssrdfProfileCdf + tab.profile_offset;

	int rho_off, rad_off;
	float rho_w[4], rad_w[4];
	if (!gpu_catmull_rom_weights(rho_samples, tab.n_rho, rho, &rho_off, rho_w)) return 0.0f;
	if (!gpu_catmull_rom_weights(radius_samples, tab.n_radius, r_opt, &rad_off, rad_w)) return 0.0f;

	float val = 0.0f, rho_eff_interp = 0.0f;
	for (int j = 0; j < 4; ++j) {
		if (rho_w[j] == 0.0f) continue;
		const int ri = rho_off + j;
		if (ri < 0 || ri >= tab.n_rho) continue;
		rho_eff_interp += rho_w[j] * profile_cdf[ri * tab.n_radius + (tab.n_radius - 1)];
		for (int k = 0; k < 4; ++k) {
			if (rad_w[k] == 0.0f) continue;
			const int rj = rad_off + k;
			if (rj < 0 || rj >= tab.n_radius) continue;
			val += rho_w[j] * rad_w[k] * profile[ri * tab.n_radius + rj];
		}
	}
	if (r_opt != 0.0f) val /= (2.0f * kPi * r_opt);
	if (rho_eff_interp <= 0.0f) return 0.0f;
	return fmaxf(0.0f, val * sigma_t * sigma_t / rho_eff_interp);
}

// ---------------------------------------------------------------------------
// sample_sr() -- importance-sample a radius from the tabulated profile.
// Ports src/shared/sampling_2d.h's SampleCatmullRom2D() to device float,
// without lambdas: gpu_cr2d_interp() plays the role of that function's
// captured `interpolate` closure, taking (offset, weights, n1, n2) as
// explicit parameters instead of capturing them.
// ---------------------------------------------------------------------------

// Evaluate the rho-weighted interpolation of one column `col` of a n1*n2
// row-major table (`arr`), using precomputed 4 Catmull-Rom weights over the
// rho (row) axis starting at `offset`. Mirrors SampleCatmullRom2D's local
// `interpolate` lambda exactly.
__device__ __forceinline__ float gpu_cr2d_interp(const float* arr, int col, int offset,
												   const float weights[4], int n1, int n2) {
	float v = 0.0f;
	for (int k = 0; k < 4; ++k) {
		const int ri = offset + k;
		if (weights[k] != 0.0f && ri >= 0 && ri < n1)
			v += arr[ri * n2 + col] * weights[k];
	}
	return v;
}

// Cubic-Hermite CDF (pbrt-v4's Fhat) and its derivative (fhat) at t in
// [0,1], for the segment [f0,d0] -> [f1,d1] found by SampleCatmullRom2D,
// offset so a root of (Fhat(t)-u) is what the caller wants. Mirrors
// src/shared/sampling_2d.h's evaluate_polynomial()-based Fhat/fhat
// construction inside SampleCatmullRom2D's own `eval` lambda.
__device__ __forceinline__ void gpu_hermite_cdf_eval(float t, float f0, float d0, float f1, float d1,
													   float u, float& Fhat_minus_u, float& fhat) {
	const float t2 = t * t, t3 = t2 * t, t4 = t3 * t;
	const float Fhat = f0 * t + 0.5f * d0 * t2
		+ ((-2.0f * d0 - d1) * (1.0f / 3.0f) + f1 - f0) * t3
		+ (0.25f * (d0 + d1) + 0.5f * (f0 - f1)) * t4;
	fhat = f0 + d0 * t
		+ (-2.0f * d0 - d1 + 3.0f * (f1 - f0)) * t2
		+ (d0 + d1 + 2.0f * (f0 - f1)) * t3;
	Fhat_minus_u = Fhat - u;
}

// Hybrid Newton-bisection root find of (Fhat(t) - u) == 0 for t in [0,1].
// Direct port of src/shared/sampling_2d.h's catmullrom_newton_bisection(),
// specialised to the cubic-Hermite-CDF `f` above (no lambda parameter -
// this file's one and only caller needs exactly this f).
__device__ __forceinline__ float gpu_catmullrom_newton_bisection(float f0, float d0, float f1, float d1, float u) {
	float x0 = 0.0f, x1 = 1.0f;
	float Fmu0, fp0, Fmu1, fp1;
	gpu_hermite_cdf_eval(x0, f0, d0, f1, d1, u, Fmu0, fp0);
	gpu_hermite_cdf_eval(x1, f0, d0, f1, d1, u, Fmu1, fp1);
	if (fabsf(Fmu0) < 1e-6f) return x0;
	if (fabsf(Fmu1) < 1e-6f) return x1;
	const bool neg_start = (Fmu0 < 0.0f);
	float xMid = x0 + (x1 - x0) * (-Fmu0) / (Fmu1 - Fmu0);
	for (int iter = 0; iter < 64; ++iter) {
		if (!(x0 < xMid && xMid < x1)) xMid = 0.5f * (x0 + x1);
		float Fmid, dfMid;
		gpu_hermite_cdf_eval(xMid, f0, d0, f1, d1, u, Fmid, dfMid);
		if (fabsf(Fmid) < 1e-6f || x1 - x0 < 1e-6f) break;
		if ((Fmid < 0.0f) == neg_start) x0 = xMid; else x1 = xMid;
		xMid -= Fmid / dfMid;
	}
	return xMid;
}

// SampleCatmullRom2D device port: samples a radius (in OPTICAL space, still
// needing division by sigma_t - see gpu_bssrdf_sample_sr() below) from the
// 2D profile table, conditioned on `rho` (the row/albedo axis) and a
// uniform sample `u` (the column/radius axis to invert).
__device__ __forceinline__ float gpu_sample_catmull_rom_2d(
	const float* nodes1, int n1, const float* nodes2, int n2,
	const float* values, const float* cdf, float rho, float u)
{
	int offset;
	float weights[4];
	if (!gpu_catmull_rom_weights(nodes1, n1, rho, &offset, weights)) return 0.0f;

	const float maximum = gpu_cr2d_interp(cdf, n2 - 1, offset, weights, n1, n2);
	if (maximum <= 0.0f) return 0.0f;
	u *= maximum;

	// Binary search largest idx in [0,n2-2] with interp(cdf,idx) <= u -
	// same shape as gpu_find_interval_le() but reading through
	// gpu_cr2d_interp() (a function of the rho-weighted row combination)
	// rather than a raw array, so it can't reuse that helper directly.
	int size = n2 - 2, first = 1;
	while (size > 0) {
		const int half = size >> 1;
		const int middle = first + half;
		if (gpu_cr2d_interp(cdf, middle, offset, weights, n1, n2) <= u) { first = middle + 1; size -= half + 1; }
		else                                                            { size = half; }
	}
	int idx = first - 1;
	if (idx < 0)      idx = 0;
	if (idx > n2 - 2) idx = n2 - 2;

	const float f0 = gpu_cr2d_interp(values, idx, offset, weights, n1, n2);
	const float f1 = gpu_cr2d_interp(values, idx + 1, offset, weights, n1, n2);
	const float x0 = nodes2[idx], x1 = nodes2[idx + 1];
	const float w = x1 - x0;
	const float d0 = (idx > 0)
		? w * (f1 - gpu_cr2d_interp(values, idx - 1, offset, weights, n1, n2)) / (x1 - nodes2[idx - 1])
		: (f1 - f0);
	const float d1 = (idx + 2 < n2)
		? w * (gpu_cr2d_interp(values, idx + 2, offset, weights, n1, n2) - f0) / (nodes2[idx + 2] - x0)
		: (f1 - f0);
	const float uu = (u - gpu_cr2d_interp(cdf, idx, offset, weights, n1, n2)) / w;

	const float t = gpu_catmullrom_newton_bisection(f0, d0, f1, d1, uu);
	return x0 + w * t;
}

// TabulatedBSSRDF::sample_sr() device port: samples a radius in RENDERING
// space (already divided by sigma_t), or -1 if sigma_t <= 0 (matches CPU's
// own "sigma_t==0" bailout).
__device__ __forceinline__ float gpu_bssrdf_sample_sr(const GpuBssrdfTable& tab, float sigma_t, float rho, float u) {
	if (sigma_t <= 0.0f) return -1.0f;
	const float* rho_samples    = params.bssrdfRhoSamples + tab.rho_offset;
	const float* radius_samples = params.bssrdfRadiusSamples + tab.radius_offset;
	const float* profile        = params.bssrdfProfile + tab.profile_offset;
	const float* profile_cdf    = params.bssrdfProfileCdf + tab.profile_offset;
	const float r_opt = gpu_sample_catmull_rom_2d(rho_samples, tab.n_rho, radius_samples, tab.n_radius,
												   profile, profile_cdf, rho, u);
	return r_opt / sigma_t;
}
