// optix_measured_bxdf.h -- Device-side real tabulated-measured-BRDF
// evaluation (MaterialType::Measured, recursive GPU backend only - see
// wavefront_measured_bxdf.h for the wavefront backend's own copy). Included
// by optix_device_helpers.h, AFTER the `extern "C" { __constant__
// LaunchParams params; }` declaration (these functions read the flat
// measuredParamValues/measuredData/measuredMcdf/measuredCcdf arrays directly
// off `params`), same inclusion-order requirement as optix_bssrdf.h just
// above it.
//
// A float, raw-pointer/array reimplementation of:
//   - src/shared/piecewise_linear_2d.h's PiecewiseLinear2D<Dimension>::
//     Sample()/Eval() (Dimension 0, 2, and 3 - GpuPL2DTable::dim carries
//     which, since it is a RUNTIME field here, not a compile-time template
//     parameter the way CPU's class is templated - see GpuPL2DTable's own
//     comment in optix_types.h),
//   - src/shared/measured_bxdf.h's MeasuredBxDF<T>::sample_f().
// NOT a call to those CPU functions, which use std::vector, a compile-time
// Dimension template parameter, and double precision throughout, none of
// which this codebase's other device code relies on (no nvcc
// --extended-lambda flag is set for this backend's compile - see
// build_optix.targets' OptixFlags - so this file deliberately avoids
// lambdas, matching optix_bssrdf.h's own established style exactly).
//
// Only Sample()/Eval() are ported here (NOT Invert()), and only
// MeasuredBxDF::sample_f() (NOT f()/pdf()): this material is sampled
// exactly ONCE per hit, the same shape as Metal/Conductor/Hair/Principled
// (skip_pdf-equivalent, no NEE - see MaterialType::Measured's own comment
// in optix_types.h and shade_material()'s Measured case below). CPU's own
// `class measured::scatter()` (material_pbrt.h) only ever calls sample_f()
// too, for the identical reason - its scattering_pdf() (which does need
// Invert(), via MeasuredBxDF::pdf()) exists there for API parity only, per
// that class's own comment, and is never invoked while skip_pdf is set.
//
// LookImpl<Dim>'s CPU recursion (piecewise_linear_2d.h ~lines 313-323)
// pairwise-folds 2^Dim corner values, weighted by the Dim FillPW-computed
// interpolation weight pairs. gpu_pl2d_look() below computes the
// mathematically identical sum by iterating explicitly over all 2^dim
// corner combinations instead of recursing - algebraically the same value
// (a sum of products, just reassociated/reordered), not a different
// formula. This is a deliberate departure from a literal recursive-template
// port: `dim` is a runtime GpuPL2DTable field on device (0, 2, or 3 - a
// single GpuMeasuredTable mixes all three sub-table shapes), so the
// compile-time recursion CPU's class uses does not translate directly
// without either instantiating three template variants or writing three
// near-duplicate functions; an explicit corner-mask loop handles all three
// with one function and stays in the same no-template, no-lambda,
// explicit-loop spirit optix_bssrdf.h already established for this kind of
// device math.

// ---------------------------------------------------------------------------
// gpu_pl2d_look() -- combines up to 2^dim (dim in {0,2,3}) corner values of
// one PiecewiseLinear2D sub-table, given the base flat index i0, the
// per-slice element count sz (nx*ny for data/ccdf lookups, ny for mcdf
// lookups - see each caller below, matching CPU's own varying `sz` argument
// to LookImpl), and the FillPW-computed interpolation weight pairs
// pw[2*k]/pw[2*k+1] and per-axis slice strides stride[k]. Mirrors
// LookImpl<Dim> exactly (see this file's own header comment for why this is
// an explicit corner-mask loop rather than a literal recursive-template port).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gpu_pl2d_look(const float* data, uint32_t i0, uint32_t sz,
												  const float* pw, int dim, const int stride[3]) {
	float result = 0.0f;
	const int combos = 1 << dim;
	for (int mask = 0; mask < combos; ++mask) {
		uint32_t idx = i0;
		float w = 1.0f;
		for (int k = 0; k < dim; ++k) {
			const int bit = (mask >> k) & 1;
			if (bit) idx += (uint32_t)stride[k] * sz;
			w *= pw[2 * k + bit];
		}
		result += w * data[idx];
	}
	return result;
}

// gpu_pl2d_fill_pw() -- mirrors PiecewiseLinear2D<Dim>::FillPW(): for each of
// tab.dim conditioning axes, finds its bracketing pair in that axis's
// parameter-value array (binary search: largest idx with values[idx] <= p,
// clamped so idx+1 stays valid) and records the interpolation weight pair,
// accumulating the slice offset `soff`. p[] holds tab.dim parameter values,
// in the SAME order the table was built with (phi_i, theta_i[, lambda] -
// matches measured_bxdf_loader.h's paramRes/paramVals construction, and
// therefore gpu_measured_sample_f()'s own call order below).
__device__ __forceinline__ void gpu_pl2d_fill_pw(const GpuPL2DTable& tab, const float* paramValues,
													const float p[3], uint32_t& soff, float pw[6]) {
	soff = 0;
	for (int dim = 0; dim < tab.dim; ++dim) {
		const int n = tab.param_res[dim];
		if (n <= 1) { pw[2 * dim] = 1.0f; pw[2 * dim + 1] = 0.0f; continue; }
		const float* axis = paramValues + tab.param_value_offset[dim];
		const float pv = p[dim];
		uint32_t lo = 0, hi = (uint32_t)n;
		while (lo + 1 < hi) {
			const uint32_t mid = (lo + hi) >> 1;
			if (axis[mid] <= pv) lo = mid; else hi = mid;
		}
		int idx = (int)lo;
		if (idx + 1 >= n) idx = n - 2;
		const float p0 = axis[idx], p1 = axis[idx + 1];
		float t = (pv - p0) / (p1 - p0);
		t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
		pw[2 * dim + 1] = t;
		pw[2 * dim]     = 1.0f - t;
		soff += (uint32_t)tab.param_stride[dim] * (uint32_t)idx;
	}
}

// gpu_pl2d_eval() -- PiecewiseLinear2D<Dim>::Eval() port (bilinear
// interpolation of the normalized density at (px,py), blended across
// conditioning axes via gpu_pl2d_look()). Used for the ndf/sigma (dim=0) and
// spectra (dim=3) sub-tables - see gpu_measured_sample_f()'s own calls.
// `p` is unused (may be nullptr) when tab.dim == 0.
__device__ __forceinline__ float gpu_pl2d_eval(const GpuPL2DTable& tab, const float* data,
												  const float* paramValues,
												  float px, float py, const float p[3]) {
	float pw[6];
	uint32_t soff;
	gpu_pl2d_fill_pw(tab, paramValues, p, soff, pw);

	const float invx = (float)(tab.nx - 1), invy = (float)(tab.ny - 1);
	const float x = px * invx, y = py * invy;
	int ix = (int)x; if (ix > tab.nx - 2) ix = tab.nx - 2;
	int iy = (int)y; if (iy > tab.ny - 2) iy = tab.ny - 2;
	const float wx1 = x - (float)ix, wx0 = 1.0f - wx1;
	const float wy1 = y - (float)iy, wy0 = 1.0f - wy1;

	uint32_t idx = (uint32_t)(ix + iy * tab.nx);
	const uint32_t sz = (uint32_t)(tab.nx * tab.ny);
	if (tab.dim != 0) idx += soff * sz;

	const float* base = data + tab.data_offset;
	const float v00 = gpu_pl2d_look(base,              idx, sz, pw, tab.dim, tab.param_stride);
	const float v10 = gpu_pl2d_look(base + 1,          idx, sz, pw, tab.dim, tab.param_stride);
	const float v01 = gpu_pl2d_look(base + tab.nx,     idx, sz, pw, tab.dim, tab.param_stride);
	const float v11 = gpu_pl2d_look(base + tab.nx + 1, idx, sz, pw, tab.dim, tab.param_stride);

	return (wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11)) * invx * invy;
}

// gpu_pl2d_sample() -- PiecewiseLinear2D<Dim>::Sample() port (2D bilinear-
// patch warp sampling with an analytic per-patch inverse: marginal row via
// mcdf, conditional column via ccdf, exact quadratic solve within the chosen
// patch). Used for the luminance/vndf (dim=2, real CDFs) sub-tables only -
// see gpu_measured_sample_f()'s own two calls.
__device__ __forceinline__ void gpu_pl2d_sample(const GpuPL2DTable& tab,
												   const float* data, const float* mcdf, const float* ccdf,
												   const float* paramValues,
												   float u0, float u1, const float p[3],
												   float& out_px, float& out_py, float& out_pdf) {
	const float kOme = 1.0f - 1.19209e-7f;
	u0 = u0 < (1.0f - kOme) ? (1.0f - kOme) : (u0 > kOme ? kOme : u0);
	u1 = u1 < (1.0f - kOme) ? (1.0f - kOme) : (u1 > kOme ? kOme : u1);

	float pw[6];
	uint32_t soff;
	gpu_pl2d_fill_pw(tab, paramValues, p, soff, pw);

	const int nx = tab.nx, ny = tab.ny;
	const uint32_t sz = (uint32_t)(nx * ny);
	const float* dataBase = data + tab.data_offset;
	const float* mcdfBase = mcdf + tab.mcdf_offset;
	const float* ccdfBase = ccdf + tab.ccdf_offset;

	// Sample the marginal row: FindInterval over ny with predicate
	// Look(mcdf, mb+i, ny, pw) < u1 (note: `ny`, not nx*ny, is mcdf's own
	// per-slice stride - mcdf holds one CDF value per row per slice, unlike
	// ccdf/data which hold nx*ny).
	const uint32_t mb = (tab.dim != 0) ? (soff * (uint32_t)ny) : 0u;
	uint32_t rlo = 0, rhi = (uint32_t)ny;
	while (rlo + 1 < rhi) {
		const uint32_t mid = (rlo + rhi) >> 1;
		const float fm = gpu_pl2d_look(mcdfBase, mb + mid, (uint32_t)ny, pw, tab.dim, tab.param_stride);
		if (fm < u1) rlo = mid; else rhi = mid;
	}
	const uint32_t row = rlo;
	u1 -= gpu_pl2d_look(mcdfBase, mb + row, (uint32_t)ny, pw, tab.dim, tab.param_stride);

	uint32_t off = row * (uint32_t)nx + ((tab.dim != 0) ? soff * sz : 0u);
	const float r0 = gpu_pl2d_look(ccdfBase, off + (uint32_t)nx - 1,       sz, pw, tab.dim, tab.param_stride);
	const float r1 = gpu_pl2d_look(ccdfBase, off + (uint32_t)(nx * 2 - 1), sz, pw, tab.dim, tab.param_stride);

	const bool ky = fabsf(r0 - r1) < 1e-4f * (r0 + r1);
	u1 = ky ? (2.0f * u1) : (r0 - sqrtf(fmaxf(0.0f, r0 * r0 - 2.0f * u1 * (r0 - r1))));
	u1 /= ky ? (r0 + r1) : (r0 - r1);

	// Sample the conditional column.
	u0 *= (1.0f - u1) * r0 + u1 * r1;
	uint32_t clo = 0, chi = (uint32_t)nx;
	while (clo + 1 < chi) {
		const uint32_t mid = (clo + chi) >> 1;
		const float v0 = gpu_pl2d_look(ccdfBase,      off + mid, sz, pw, tab.dim, tab.param_stride);
		const float v1 = gpu_pl2d_look(ccdfBase + nx, off + mid, sz, pw, tab.dim, tab.param_stride);
		const float fc = (1.0f - u1) * v0 + u1 * v1;
		if (fc < u0) clo = mid; else chi = mid;
	}
	const uint32_t col = clo;
	{
		const float v0 = gpu_pl2d_look(ccdfBase,      off + col, sz, pw, tab.dim, tab.param_stride);
		const float v1 = gpu_pl2d_look(ccdfBase + nx, off + col, sz, pw, tab.dim, tab.param_stride);
		u0 -= (1.0f - u1) * v0 + u1 * v1;
	}
	off += col;

	const float v00 = gpu_pl2d_look(dataBase,          off, sz, pw, tab.dim, tab.param_stride);
	const float v10 = gpu_pl2d_look(dataBase + 1,      off, sz, pw, tab.dim, tab.param_stride);
	const float v01 = gpu_pl2d_look(dataBase + nx,     off, sz, pw, tab.dim, tab.param_stride);
	const float v11 = gpu_pl2d_look(dataBase + nx + 1, off, sz, pw, tab.dim, tab.param_stride);

	const float c0 = (1.0f - u1) * v00 + u1 * v01;
	const float c1 = (1.0f - u1) * v10 + u1 * v11;

	const bool kx = fabsf(c0 - c1) < 1e-4f * (c0 + c1);
	u0 = kx ? (2.0f * u0) : (c0 - sqrtf(fmaxf(0.0f, c0 * c0 - 2.0f * u0 * (c0 - c1))));
	u0 /= kx ? (c0 + c1) : (c0 - c1);

	const float px_scale = 1.0f / (float)(nx - 1);
	const float py_scale = 1.0f / (float)(ny - 1);
	out_px  = ((float)col + u0) * px_scale;
	out_py  = ((float)row + u1) * py_scale;
	out_pdf = ((1.0f - u0) * c0 + u0 * c1) * (float)(nx - 1) * (float)(ny - 1);
}

// ---------------------------------------------------------------------------
// gpu_measured_sample_f() -- MeasuredBxDF<T>::sample_f() port (see
// src/shared/measured_bxdf.h's own extensive comment for the algorithm this
// mirrors step for step). `wo` is in the LOCAL SHADING FRAME (z = shading
// normal, matching ShadingFrame<float>::to_local() - see
// sample_measured_material() below, this file's own caller, which builds
// that frame using this codebase's existing shared CPU/GPU shading-frame
// helper, src/shared/shading_frame.h - the exact same one material_pbrt.h's
// `class measured` already uses with T=double). `lambda` holds the fixed
// R/G/B query wavelengths (matches `class measured`'s kLambdaR/G/B exactly -
// see sample_measured_material()'s own comment for the values).
//
// Returns false (outputs untouched) on any of the same rejection paths CPU
// has: grazing/degenerate wo, zero luminance/vndf sampling pdf, a reflected
// half-vector that puts wi below the horizon, zero denom/jacobian. Mirrors
// CPU's own `if (!ok) return false;` bailout in `class measured::scatter()`.
//
// On success, wi/fr/fg/fb are exactly what CPU's sample_f() returns -
// fr/fg/fb are the material's full skip_pdf throughput weight as-is (no
// further division by pdf_out here, matching what CPU's own `class
// measured::scatter()` does: `srec.attenuation = color(fr,fg,fb)` directly,
// discarding the pdf output pbrt-v4's own Sample_f contract would otherwise
// want the caller to divide by - see MaterialType::Measured's own comment
// in optix_types.h and shade_material()'s Measured case for why this is
// exactly what "matching Metal's skip_pdf pattern" means here).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool gpu_measured_sample_f(
	const GpuMeasuredTable& tab, bool isotropic,
	const float* paramValues, const float* data, const float* mcdf, const float* ccdf,
	float wox, float woy, float woz,
	float u0, float u1, const float lambda[3],
	float& wix, float& wiy, float& wiz,
	float& fr, float& fg, float& fb)
{
	const float kPi = 3.14159265358979323846f;
	fr = fg = fb = 0.0f;

	const bool flip = (woz < 0.0f);
	if (flip) { wox = -wox; woy = -woy; woz = -woz; }
	if (woz <= 0.0f) return false;

	const float theta_o = acosf(fmaxf(-1.0f, fminf(1.0f, woz)));
	const float phi_o    = atan2f(woy, wox);

	float lum_p[3] = { phi_o, theta_o, 0.0f };
	float lum_px, lum_py, lum_pdf;
	gpu_pl2d_sample(tab.luminance, data, mcdf, ccdf, paramValues, u0, u1, lum_p, lum_px, lum_py, lum_pdf);
	if (lum_pdf == 0.0f) return false;

	float vndf_p[3] = { phi_o, theta_o, 0.0f };
	float u_wm_x, u_wm_y, vndf_pdf;
	gpu_pl2d_sample(tab.vndf, data, mcdf, ccdf, paramValues, lum_px, lum_py, vndf_p, u_wm_x, u_wm_y, vndf_pdf);
	if (vndf_pdf == 0.0f) return false;

	float phi_m = (2.0f * u_wm_y - 1.0f) * kPi;
	const float theta_m = u_wm_x * u_wm_x * (kPi * 0.5f);
	if (isotropic) phi_m += phi_o;
	const float sinTheta_m = sinf(theta_m), cosTheta_m = cosf(theta_m);
	const float wmx = sinTheta_m * cosf(phi_m);
	const float wmy = sinTheta_m * sinf(phi_m);
	const float wmz = cosTheta_m;

	const float dot_wo_wm = wox * wmx + woy * wmy + woz * wmz;
	const float wi_x = 2.0f * dot_wo_wm * wmx - wox;
	const float wi_y = 2.0f * dot_wo_wm * wmy - woy;
	const float wi_z = 2.0f * dot_wo_wm * wmz - woz;
	if (wi_z <= 0.0f) return false;

	float spec_p[3] = { phi_o, theta_o, 0.0f };
	float val[3];
	for (int c = 0; c < 3; ++c) {
		spec_p[2] = lambda[c];
		val[c] = fmaxf(0.0f, gpu_pl2d_eval(tab.spectra, data, paramValues, u_wm_x, u_wm_y, spec_p));
	}

	const float u_wo_x = sqrtf(theta_o * (2.0f / kPi));
	const float u_wo_y = phi_o * (1.0f / (2.0f * kPi)) + 0.5f;
	const float ndf_val   = gpu_pl2d_eval(tab.ndf,   data, paramValues, u_wm_x, u_wm_y, nullptr);
	const float sigma_val = gpu_pl2d_eval(tab.sigma, data, paramValues, u_wo_x, u_wo_y, nullptr);
	const float abs_cos_wi = fabsf(wi_z);
	const float denom = 4.0f * sigma_val * abs_cos_wi;
	if (denom == 0.0f) return false;
	const float scale = ndf_val / denom;

	const float jacobian = 4.0f * dot_wo_wm *
		fmaxf(2.0f * kPi * kPi * u_wm_x * sinTheta_m, 1e-6f);
	if (jacobian == 0.0f) return false;

	const float final_pdf = vndf_pdf * lum_pdf / jacobian;
	if (final_pdf == 0.0f) return false;

	float ox = wi_x, oy = wi_y, oz = wi_z;
	if (flip) { ox = -ox; oy = -oy; oz = -oz; }
	wix = ox; wiy = oy; wiz = oz;

	fr = val[0] * scale;
	fg = val[1] * scale;
	fb = val[2] * scale;
	return true;
}

// sample_measured_material() -- world-space wrapper matching this file's
// sibling helpers (sample_hair_material()/sample_principled_material(),
// optix_device_helpers.h): builds the local shading frame via
// ShadingFrame<float>::from_normal() (src/shared/shading_frame.h - CPU_GPU-
// tagged, no std::vector, so it compiles directly for device code with
// T=float, exactly like HairBxDF<float>/PrincipledBxDF<T> already do
// elsewhere in this file - this is the "existing world<->local shading-
// frame helper" this material reuses rather than writing a new one),
// converts wo to local space, samples once via gpu_measured_sample_f(), and
// converts the sampled wi back to world. Returns false (matching the
// Metal-family scatter-rejection pattern) if sample_f() itself rejected the
// sample or `mat.textureIdx` doesn't name a valid table.
__device__ __forceinline__ bool sample_measured_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	if (mat.textureIdx < 0 || (unsigned int)mat.textureIdx >= params.numMeasuredTables)
		return false;
	const GpuMeasuredTable& tab = params.measuredTables[mat.textureIdx];

	ShadingFrame<float> frame = ShadingFrame<float>::from_normal(normal.x, normal.y, normal.z);
	float3 wo_world = normalize(-ray_dir);
	float wox, woy, woz;
	frame.to_local(wo_world.x, wo_world.y, wo_world.z, wox, woy, woz);

	// sRGB-primary approximations - matches material_pbrt.h `class
	// measured`'s kLambdaR/G/B exactly (this material queries the tensor's
	// spectral interpolant at 3 fixed wavelengths rather than pbrt-v4's
	// stochastic hero-wavelength spectral sampling - see that class's own
	// comment for why).
	const float lambda[3] = { 612.0f, 549.0f, 465.0f };

	float wix, wiy, wiz, fr, fg, fb;
	if (!gpu_measured_sample_f(tab, tab.isotropic != 0,
			params.measuredParamValues, params.measuredData, params.measuredMcdf, params.measuredCcdf,
			wox, woy, woz, random_float(seed), random_float(seed), lambda,
			wix, wiy, wiz, fr, fg, fb))
		return false;

	float wdx, wdy, wdz;
	frame.to_world(wix, wiy, wiz, wdx, wdy, wdz);
	scattered_dir = normalize(make_float3(wdx, wdy, wdz));
	attenuation   = make_float3(fr, fg, fb);
	return true;
}
