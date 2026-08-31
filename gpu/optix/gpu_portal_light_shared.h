// gpu_portal_light_shared.h -- Device-side real windowed-portal infinite
// light math (LightSource "infinite" with a "portal[4]" quad), shared
// between the GPU-recursive backend (optix_portal_light.h) and the GPU-
// wavefront backend (wavefront_portal_light.h). Same split rationale as
// gpu_sky_light_shared.h (see that file's own header comment): every
// function here takes its data as plain explicit parameters, no OptiX
// intrinsic and no `params`/`wf_params` global read, so both backends
// include and call it directly.
//
// A float, raw-pointer reimplementation of:
//   - src/shared/portal_image_infinite_light.h's PortalImageInfiniteLightData<T>
//     (ImageFromRender/RenderFromImage_uv/ImageBounds/eval_Le_rgb/sample_li/
//     pdf_li - the double-precision, std::vector-backed CPU class this
//     mirrors),
//   - src/shared/summed_area_table.h's SummedAreaTable::Integral()/Lookup()/
//     LookupInt() and WindowedPiecewiseConstant2D::Sample()/PDF()/Eval()/
//     SampleBisection().
// NOT a call to those CPU functions. The three flat buffers this reads
// (GpuPortalLight::rectifiedImage/distFunc/satSum) are NOT rebuilt here -
// they're uploaded verbatim from a real host-side PortalImageInfiniteLightData
// instance (see GpuPortalLight's own comment, optix_types.h).
//
// satSum stays double (not narrowed to float like every other GPU buffer in
// this codebase) - it's a bilinearly-interpolated prefix-sum table, and the
// windowed bisection search (gpu_portal_sample below) repeatedly subtracts
// nearly-equal large sums to get small differences; SummedAreaTable itself
// stores it as double for the identical reason (that class's own comment).

#pragma once

// ---------------------------------------------------------------------------
// Frame math -- GpuPortalLight::frameX/Y/Z ToLocal()/FromLocal(), mirrors
// src/shared/vec3_frame.h's Frame<T>::to_local()/from_local() exactly.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 gpu_portal_to_local(const GpuPortalLight& d, const float3& v) {
	return make_float3(dot(v, d.frameX), dot(v, d.frameY), dot(v, d.frameZ));
}

__device__ __forceinline__ float3 gpu_portal_from_local(const GpuPortalLight& d, const float3& v) {
	return d.frameX * v.x + d.frameY * v.y + d.frameZ * v.z;
}

// ---------------------------------------------------------------------------
// gpu_portal_image_from_render() -- render-space direction -> image uv in
// [0,1]^2, plus optional Jacobian duv_dw. Mirrors PortalImageInfiniteLightData::
// ImageFromRender() exactly. Returns false if the direction is behind the
// portal (w.z <= 0 in portal-frame local space).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool gpu_portal_image_from_render(const GpuPortalLight& d, const float3& wRender,
															   float& u_out, float& v_out, float* duv_dw = nullptr) {
	const float3 w = gpu_portal_to_local(d, wRender);
	if (w.z <= 0.0f) return false;
	if (duv_dw) {
		const float pi_sq = 3.14159265358979323846f * 3.14159265358979323846f;
		*duv_dw = pi_sq * (1.0f - w.x * w.x) * (1.0f - w.y * w.y) / w.z;
	}
	const float alpha = atan2f(w.x, w.z);
	const float beta  = atan2f(w.y, w.z);
	const float kHalfPi = 1.57079632679489661923f;
	const float kPi = 3.14159265358979323846f;
	u_out = fminf(1.0f, fmaxf(0.0f, (alpha + kHalfPi) / kPi));
	v_out = fminf(1.0f, fmaxf(0.0f, (beta  + kHalfPi) / kPi));
	return true;
}

// ---------------------------------------------------------------------------
// gpu_portal_render_from_image() -- image uv -> render-space direction, plus
// optional Jacobian duv_dw. Mirrors RenderFromImage_uv() exactly.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 gpu_portal_render_from_image(const GpuPortalLight& d, float u, float v,
																 float* duv_dw = nullptr) {
	const float kHalfPi = 1.57079632679489661923f;
	const float kPi = 3.14159265358979323846f;
	const float alpha = -kHalfPi + u * kPi;
	const float beta  = -kHalfPi + v * kPi;
	const float x = tanf(alpha), y = tanf(beta);
	const float3 w = normalize(make_float3(x, y, 1.0f));
	if (duv_dw) {
		const float pi_sq = kPi * kPi;
		*duv_dw = pi_sq * (1.0f - w.x * w.x) * (1.0f - w.y * w.y) / w.z;
	}
	return gpu_portal_from_local(d, w);
}

// ---------------------------------------------------------------------------
// gpu_portal_image_bounds() -- [uMin,uMax]x[vMin,vMax] of the portal quad as
// seen from shading point p, using the diagonal corners p0/p2. Mirrors
// ImageBounds() exactly. Returns false if either corner is behind the portal
// from p (same ImageFromRender() failure the CPU class propagates).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool gpu_portal_image_bounds(const GpuPortalLight& d, const float3& p,
														  float& uMin, float& vMin, float& uMax, float& vMax) {
	const float3 d0 = normalize(d.p0 - p);
	const float3 d2 = normalize(d.p2 - p);
	float u0, v0, u2, v2;
	if (!gpu_portal_image_from_render(d, d0, u0, v0)) return false;
	if (!gpu_portal_image_from_render(d, d2, u2, v2)) return false;
	uMin = fminf(u0, u2); uMax = fmaxf(u0, u2);
	vMin = fminf(v0, v2); vMax = fmaxf(v0, v2);
	return true;
}

// ---------------------------------------------------------------------------
// gpu_portal_sat_lookup() -- bilinearly-interpolated prefix-sum lookup.
// Mirrors SummedAreaTable::Lookup()/LookupInt() exactly (including the
// lower-boundary-is-zero semantics at x==0/y==0).
// ---------------------------------------------------------------------------
__device__ __forceinline__ double gpu_portal_sat_lookup_int(const double* satSum, int width, int height, int x, int y) {
	if (x == 0 || y == 0) return 0.0;
	x = min(x - 1, width - 1);
	y = min(y - 1, height - 1);
	return satSum[y * width + x];
}

__device__ __forceinline__ double gpu_portal_sat_lookup(const double* satSum, int width, int height, float x, float y) {
	x *= (float)width;
	y *= (float)height;
	int x0 = (int)x, y0 = (int)y;
	const double v00 = gpu_portal_sat_lookup_int(satSum, width, height, x0,     y0);
	const double v10 = gpu_portal_sat_lookup_int(satSum, width, height, x0 + 1, y0);
	const double v01 = gpu_portal_sat_lookup_int(satSum, width, height, x0,     y0 + 1);
	const double v11 = gpu_portal_sat_lookup_int(satSum, width, height, x0 + 1, y0 + 1);
	const float dx = x - (float)(int)x, dy = y - (float)(int)y;
	return (1.0 - dx) * (1.0 - dy) * v00 + (1.0 - dx) * dy * v01
		 + dx * (1.0 - dy) * v10 + dx * dy * v11;
}

// gpu_portal_sat_integral() -- Mirrors SummedAreaTable::Integral() exactly.
__device__ __forceinline__ float gpu_portal_sat_integral(const double* satSum, int width, int height,
														   float uMin, float vMin, float uMax, float vMax) {
	const double s = (gpu_portal_sat_lookup(satSum, width, height, uMax, vMax) -
					   gpu_portal_sat_lookup(satSum, width, height, uMin, vMax)) +
					  (gpu_portal_sat_lookup(satSum, width, height, uMin, vMin) -
					   gpu_portal_sat_lookup(satSum, width, height, uMax, vMin));
	return (float)fmax(s / (double)(width * height), 0.0);
}

// gpu_portal_dist_eval() -- nearest-neighbour lookup of the raw distribution
// value at (x,y). Mirrors WindowedPiecewiseConstant2D::Eval() exactly.
__device__ __forceinline__ float gpu_portal_dist_eval(const float* distFunc, int width, int height, float x, float y) {
	int ix = min((int)(x * (float)width),  width  - 1);
	int iy = min((int)(y * (float)height), height - 1);
	return distFunc[iy * width + ix];
}

// gpu_portal_bisection() -- Mirrors WindowedPiecewiseConstant2D::SampleBisection()
// exactly, with CDF evaluated via the caller-supplied SAT integral (P below).
// P takes a single float (the trial x/y) and returns the normalized CDF value.
template <typename CDF>
__device__ __forceinline__ float gpu_portal_bisection(CDF P, float u, float lo, float hi, int n) {
	while (ceilf((float)n * hi) - floorf((float)n * lo) > 1.0f) {
		const float mid = (lo + hi) * 0.5f;
		if (P(mid) > u) hi = mid; else lo = mid;
	}
	const float Plo = P(lo), Phi = P(hi);
	const float t = (Phi - Plo > 0.0f) ? (u - Plo) / (Phi - Plo) : 0.5f;
	return fminf(hi, fmaxf(lo, lo + t * (hi - lo)));
}

// ---------------------------------------------------------------------------
// gpu_portal_Le() -- per-direction radiance from origin (ox,oy,oz) looking
// along dir. Mirrors PortalImageInfiniteLightData::eval_Le_rgb() exactly
// (nearest lookup into the rectified image, NOT the bilinear sample the
// image's own construction used - see that method's own comment).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 gpu_portal_Le(const GpuPortalLight& d, const float3& origin, const float3& dir) {
	const float3 wRender = normalize(dir);
	float u, v;
	if (!gpu_portal_image_from_render(d, wRender, u, v)) return make_float3(0.0f, 0.0f, 0.0f);
	float bMinU, bMinV, bMaxU, bMaxV;
	if (!gpu_portal_image_bounds(d, origin, bMinU, bMinV, bMaxU, bMaxV)) return make_float3(0.0f, 0.0f, 0.0f);
	if (u < bMinU || u > bMaxU || v < bMinV || v > bMaxV) return make_float3(0.0f, 0.0f, 0.0f);

	int ix = min((int)(u * (float)d.width),  d.width  - 1);
	int iy = min((int)(v * (float)d.height), d.height - 1);
	const float* px = d.rectifiedImage + (iy * d.width + ix) * 3;
	return d.scale * make_float3(px[0], px[1], px[2]);
}

// ---------------------------------------------------------------------------
// gpu_portal_pdf_Li() -- solid-angle PDF for a KNOWN direction from a known
// shading point. Mirrors pdf_li() exactly.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gpu_portal_pdf_Li(const GpuPortalLight& d, const float3& p, const float3& dir) {
	const float3 wRender = normalize(dir);
	float u, v, duv_dw;
	if (!gpu_portal_image_from_render(d, wRender, u, v, &duv_dw)) return 0.0f;
	if (duv_dw == 0.0f) return 0.0f;

	float bMinU, bMinV, bMaxU, bMaxV;
	if (!gpu_portal_image_bounds(d, p, bMinU, bMinV, bMaxU, bMaxV)) return 0.0f;

	const float funcInt = gpu_portal_sat_integral(d.satSum, d.width, d.height, bMinU, bMinV, bMaxU, bMaxV);
	if (funcInt == 0.0f) return 0.0f;
	const float mapPDF = gpu_portal_dist_eval(d.distFunc, d.width, d.height, u, v) / funcInt;
	return mapPDF / duv_dw;
}

// ---------------------------------------------------------------------------
// gpu_portal_sample_Li() -- importance-sample a render-space direction from
// shading point p, given 2 already-generated uniform random numbers (same
// "caller passes randoms in" convention as gpu_sky_light_shared.h's own
// gpu_sky_sample_Li() - see that file's header comment). Mirrors sample_li()
// exactly (WindowedPiecewiseConstant2D::Sample() inlined: marginal-in-x then
// conditional-in-y bisection search against the SAT integral).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool gpu_portal_sample_Li(const GpuPortalLight& d, const float3& p,
													   float ru, float rv,
													   float3& dir_out, float& pdf_out) {
	float bMinU, bMinV, bMaxU, bMaxV;
	if (!gpu_portal_image_bounds(d, p, bMinU, bMinV, bMaxU, bMaxV)) return false;

	const float bInt = gpu_portal_sat_integral(d.satSum, d.width, d.height, bMinU, bMinV, bMaxU, bMaxV);
	if (bInt == 0.0f) return false;

	// Marginal CDF in x over the full window height.
	auto Px = [&](float x) -> float {
		return gpu_portal_sat_integral(d.satSum, d.width, d.height, bMinU, bMinV, x, bMaxV) / bInt;
	};
	const float sx = gpu_portal_bisection(Px, ru, bMinU, bMaxU, d.width);

	// Conditional bounds: one column-strip wide, matching Sample()'s own
	// colMin/colMax cell-snapping (floor/ceil of sx*nx, back to [0,1]).
	const int nx = d.width;
	float colMin = floorf(sx * (float)nx) / (float)nx;
	float colMax = ceilf (sx * (float)nx) / (float)nx;
	if (colMin == colMax) colMax += 1.0f / (float)nx;

	const float condInt = gpu_portal_sat_integral(d.satSum, d.width, d.height, colMin, bMinV, colMax, bMaxV);
	if (condInt == 0.0f) return false;

	auto Py = [&](float y) -> float {
		return gpu_portal_sat_integral(d.satSum, d.width, d.height, colMin, bMinV, colMax, y) / condInt;
	};
	const float sy = gpu_portal_bisection(Py, rv, bMinV, bMaxV, d.height);

	const float mapPDF = gpu_portal_dist_eval(d.distFunc, d.width, d.height, sx, sy) / bInt;

	float duv_dw;
	const float3 wRender = gpu_portal_render_from_image(d, sx, sy, &duv_dw);
	if (duv_dw == 0.0f) return false;

	dir_out = wRender;
	pdf_out = mapPDF / duv_dw;
	return true;
}
