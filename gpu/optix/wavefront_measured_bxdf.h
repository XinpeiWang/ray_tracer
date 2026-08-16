// wavefront_measured_bxdf.h -- Device-side real tabulated-measured-BRDF
// evaluation for the wavefront GPU backend (MaterialType::Measured).
// Included by wavefront_kernels.cu.
//
// Wavefront-native duplicate of optix_measured_bxdf.h's math (see that
// file's own extensive comment for the algorithm and design rationale -
// this file mirrors it function-for-function with a wf_ prefix) - matching
// this backend's own established "duplicate, don't share, device helpers
// with the recursive path" convention (every other wf_-prefixed function in
// wavefront_kernels.cu says the same thing: wf_sample_hair_material,
// wf_sample_principled_material, wf_dielectric_scatter, ...). Unlike
// wavefront_probe.h's BSSRDF duplicates, this is NOT about two separate
// OptiX modules each needing their own `__constant__` launch-params symbol
// (evaluate_materials is a plain CUDA kernel, not an OptiX program - it has
// no launch-params global at all, recursive or wavefront; every array this
// file's functions need arrives as an explicit pointer parameter, exactly
// like MaterialType::CloudMedium/RgbGridMedium's own cloudMediums/
// rgbGridMediums arguments already do). The duplication here is purely
// "this is a separate translation unit from optix_device_helpers.h, so it
// gets its own copy," the same reason wf_reflect/wf_refract/wf_pcg/wf_rand
// exist alongside their recursive-path equivalents.
//
// Only Sample()/Eval() (not Invert()) and only sample_f() (not f()/pdf())
// are ported, for the same reason as the recursive backend's copy: this
// material samples exactly once per hit (skip_pdf-equivalent, no NEE - same
// shape as MaterialType::Metal/Hair/Principled's existing wavefront cases).

// ---------------------------------------------------------------------------
// wf_gpu_pl2d_look() -- see optix_measured_bxdf.h's gpu_pl2d_look() for the
// full algorithm comment (explicit 2^dim corner-mask sum, algebraically
// identical to PiecewiseLinear2D<Dim>::LookImpl<Dim>'s recursive fold).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wf_gpu_pl2d_look(const float* data, uint32_t i0, uint32_t sz,
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

// wf_gpu_pl2d_fill_pw() -- see optix_measured_bxdf.h's gpu_pl2d_fill_pw().
__device__ __forceinline__ void wf_gpu_pl2d_fill_pw(const GpuPL2DTable& tab, const float* paramValues,
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

// wf_gpu_pl2d_eval() -- see optix_measured_bxdf.h's gpu_pl2d_eval(). Used
// for the ndf/sigma (dim=0) and spectra (dim=3) sub-tables. `p` may be
// nullptr when tab.dim == 0.
__device__ __forceinline__ float wf_gpu_pl2d_eval(const GpuPL2DTable& tab, const float* data,
													 const float* paramValues,
													 float px, float py, const float p[3]) {
	float pw[6];
	uint32_t soff;
	wf_gpu_pl2d_fill_pw(tab, paramValues, p, soff, pw);

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
	const float v00 = wf_gpu_pl2d_look(base,              idx, sz, pw, tab.dim, tab.param_stride);
	const float v10 = wf_gpu_pl2d_look(base + 1,          idx, sz, pw, tab.dim, tab.param_stride);
	const float v01 = wf_gpu_pl2d_look(base + tab.nx,     idx, sz, pw, tab.dim, tab.param_stride);
	const float v11 = wf_gpu_pl2d_look(base + tab.nx + 1, idx, sz, pw, tab.dim, tab.param_stride);

	return (wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11)) * invx * invy;
}

// wf_gpu_pl2d_sample() -- see optix_measured_bxdf.h's gpu_pl2d_sample().
// Used for the luminance/vndf (dim=2, real CDFs) sub-tables only.
__device__ __forceinline__ void wf_gpu_pl2d_sample(const GpuPL2DTable& tab,
													  const float* data, const float* mcdf, const float* ccdf,
													  const float* paramValues,
													  float u0, float u1, const float p[3],
													  float& out_px, float& out_py, float& out_pdf) {
	const float kOme = 1.0f - 1.19209e-7f;
	u0 = u0 < (1.0f - kOme) ? (1.0f - kOme) : (u0 > kOme ? kOme : u0);
	u1 = u1 < (1.0f - kOme) ? (1.0f - kOme) : (u1 > kOme ? kOme : u1);

	float pw[6];
	uint32_t soff;
	wf_gpu_pl2d_fill_pw(tab, paramValues, p, soff, pw);

	const int nx = tab.nx, ny = tab.ny;
	const uint32_t sz = (uint32_t)(nx * ny);
	const float* dataBase = data + tab.data_offset;
	const float* mcdfBase = mcdf + tab.mcdf_offset;
	const float* ccdfBase = ccdf + tab.ccdf_offset;

	const uint32_t mb = (tab.dim != 0) ? (soff * (uint32_t)ny) : 0u;
	uint32_t rlo = 0, rhi = (uint32_t)ny;
	while (rlo + 1 < rhi) {
		const uint32_t mid = (rlo + rhi) >> 1;
		const float fm = wf_gpu_pl2d_look(mcdfBase, mb + mid, (uint32_t)ny, pw, tab.dim, tab.param_stride);
		if (fm < u1) rlo = mid; else rhi = mid;
	}
	const uint32_t row = rlo;
	u1 -= wf_gpu_pl2d_look(mcdfBase, mb + row, (uint32_t)ny, pw, tab.dim, tab.param_stride);

	uint32_t off = row * (uint32_t)nx + ((tab.dim != 0) ? soff * sz : 0u);
	const float r0 = wf_gpu_pl2d_look(ccdfBase, off + (uint32_t)nx - 1,       sz, pw, tab.dim, tab.param_stride);
	const float r1 = wf_gpu_pl2d_look(ccdfBase, off + (uint32_t)(nx * 2 - 1), sz, pw, tab.dim, tab.param_stride);

	const bool ky = fabsf(r0 - r1) < 1e-4f * (r0 + r1);
	u1 = ky ? (2.0f * u1) : (r0 - sqrtf(fmaxf(0.0f, r0 * r0 - 2.0f * u1 * (r0 - r1))));
	u1 /= ky ? (r0 + r1) : (r0 - r1);

	u0 *= (1.0f - u1) * r0 + u1 * r1;
	uint32_t clo = 0, chi = (uint32_t)nx;
	while (clo + 1 < chi) {
		const uint32_t mid = (clo + chi) >> 1;
		const float v0 = wf_gpu_pl2d_look(ccdfBase,      off + mid, sz, pw, tab.dim, tab.param_stride);
		const float v1 = wf_gpu_pl2d_look(ccdfBase + nx, off + mid, sz, pw, tab.dim, tab.param_stride);
		const float fc = (1.0f - u1) * v0 + u1 * v1;
		if (fc < u0) clo = mid; else chi = mid;
	}
	const uint32_t col = clo;
	{
		const float v0 = wf_gpu_pl2d_look(ccdfBase,      off + col, sz, pw, tab.dim, tab.param_stride);
		const float v1 = wf_gpu_pl2d_look(ccdfBase + nx, off + col, sz, pw, tab.dim, tab.param_stride);
		u0 -= (1.0f - u1) * v0 + u1 * v1;
	}
	off += col;

	const float v00 = wf_gpu_pl2d_look(dataBase,          off, sz, pw, tab.dim, tab.param_stride);
	const float v10 = wf_gpu_pl2d_look(dataBase + 1,      off, sz, pw, tab.dim, tab.param_stride);
	const float v01 = wf_gpu_pl2d_look(dataBase + nx,     off, sz, pw, tab.dim, tab.param_stride);
	const float v11 = wf_gpu_pl2d_look(dataBase + nx + 1, off, sz, pw, tab.dim, tab.param_stride);

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

// wf_gpu_measured_sample_f() -- see optix_measured_bxdf.h's
// gpu_measured_sample_f() for the full algorithm/return-value comment
// (identical here). `wo` is in the local shading frame.
__device__ __forceinline__ bool wf_gpu_measured_sample_f(
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
	wf_gpu_pl2d_sample(tab.luminance, data, mcdf, ccdf, paramValues, u0, u1, lum_p, lum_px, lum_py, lum_pdf);
	if (lum_pdf == 0.0f) return false;

	float vndf_p[3] = { phi_o, theta_o, 0.0f };
	float u_wm_x, u_wm_y, vndf_pdf;
	wf_gpu_pl2d_sample(tab.vndf, data, mcdf, ccdf, paramValues, lum_px, lum_py, vndf_p, u_wm_x, u_wm_y, vndf_pdf);
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
		val[c] = fmaxf(0.0f, wf_gpu_pl2d_eval(tab.spectra, data, paramValues, u_wm_x, u_wm_y, spec_p));
	}

	const float u_wo_x = sqrtf(theta_o * (2.0f / kPi));
	const float u_wo_y = phi_o * (1.0f / (2.0f * kPi)) + 0.5f;
	const float ndf_val   = wf_gpu_pl2d_eval(tab.ndf,   data, paramValues, u_wm_x, u_wm_y, nullptr);
	const float sigma_val = wf_gpu_pl2d_eval(tab.sigma, data, paramValues, u_wo_x, u_wo_y, nullptr);
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

// wf_sample_measured_material() -- world-space wrapper, duplicated from
// optix_measured_bxdf.h's sample_measured_material() (with the wf_ prefix
// and explicit table-array parameters instead of a `params` global -
// matching this file's own established pattern, see the header comment
// above). Builds the local shading frame via ShadingFrame<float>::
// from_normal() (src/shared/shading_frame.h - shared, not duplicated: it's
// a plain CPU_GPU-tagged value type with no global state, the same class
// wf_sample_hair_material's sibling HairBxDF<float> instantiation pattern
// already relies on being safe to use directly on this backend too).
__device__ __forceinline__ bool wf_sample_measured_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	const GpuMeasuredTable* measuredTables, unsigned int numMeasuredTables,
	const float* measuredParamValues, const float* measuredData,
	const float* measuredMcdf, const float* measuredCcdf,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	if (mat.textureIdx < 0 || (unsigned int)mat.textureIdx >= numMeasuredTables)
		return false;
	const GpuMeasuredTable& tab = measuredTables[mat.textureIdx];

	ShadingFrame<float> frame = ShadingFrame<float>::from_normal(normal.x, normal.y, normal.z);
	float3 wo_world = normalize(-ray_dir);
	float wox, woy, woz;
	frame.to_local(wo_world.x, wo_world.y, wo_world.z, wox, woy, woz);

	// sRGB-primary approximations - matches material_pbrt.h `class
	// measured`'s kLambdaR/G/B exactly (same fixed-query-wavelength
	// convention as optix_measured_bxdf.h's own copy).
	const float lambda[3] = { 612.0f, 549.0f, 465.0f };

	float wix, wiy, wiz, fr, fg, fb;
	if (!wf_gpu_measured_sample_f(tab, tab.isotropic != 0,
			measuredParamValues, measuredData, measuredMcdf, measuredCcdf,
			wox, woy, woz, wf_rand(seed), wf_rand(seed), lambda,
			wix, wiy, wiz, fr, fg, fb))
		return false;

	float wdx, wdy, wdz;
	frame.to_world(wix, wiy, wiz, wdx, wdy, wdz);
	scattered_dir = normalize(make_float3(wdx, wdy, wdz));
	attenuation   = make_float3(fr, fg, fb);
	return true;
}
