#pragma once
// gpu_bump_map.h -- pbrt-v4 BumpMap on the GPU (both backends' triangle closest-hit programs).
//
// The CPU wraps a material in bump_map_material (normal_map_materials.h) when its pbrt "texture displacement"
// resolves to a grayscale image: displacement = scale * red(texture(u, v)) read with a bilinear, Repeat lookup,
// finite differences at +step along u and v (0.001 where there is no ray footprint, i.e. every bounce hit), then
// the shared apply_bump_map(). This reproduces that for any triangle hit, so the GPU stops rendering a bumped
// surface flat (Sibenik Cathedral was 12-15% darker than the CPU for this reason alone).
//
// The finite-difference step is the CPU's too: 0.001 for a bounce hit, and for a primary camera-ray hit
// 0.5 * (|dudx| + |dudy|) from the pixel's footprint on the surface (gpu_bump_footprint_step). That footprint
// matters: on a distant or heavily tiled surface (Sibenik's stone) it spans many texels, so the CPU's primary
// bump is a smoothed one, while a fixed 0.001 step reads the texture at full detail and renders it ~25% too
// bright.

#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/normal_map.h"

// Bilinear, wrap-aware read of the red channel, the convention of sample_texture() and of the CPU's
// bilinear_wrap_texture (texel centres at (i + 0.5) / size, image row 0 at the top).
inline __host__ __device__ float gpu_sample_displacement(const TextureData* textures, const unsigned char* pixels,
														 int texIdx, float u, float v) {
	const TextureData& t = textures[texIdx];
	if (t.width <= 0 || t.height <= 0) return 0.0f;
	const float uw = fminf(fmaxf(u, -1024.0f), 1024.0f);
	const float vw = fminf(fmaxf(1.0f - v, -1024.0f), 1024.0f);
	const double x = (double)uw * t.width - 0.5;
	const double y = (double)vw * t.height - 0.5;
	const int x0 = static_cast<int>(floor(x)), y0 = static_cast<int>(floor(y));
	const float fx = (float)(x - x0), fy = (float)(y - y0);
	auto texel = [&](int xi, int yi) -> float {
		switch (t.wrapMode) {
			case GpuWrapMode::Repeat:
				xi = ((xi % t.width) + t.width) % t.width;
				yi = ((yi % t.height) + t.height) % t.height;
				break;
			case GpuWrapMode::Black:
				if (xi < 0 || xi >= t.width || yi < 0 || yi >= t.height) return 0.0f;
				break;
			case GpuWrapMode::Clamp:
				xi = xi < 0 ? 0 : (xi >= t.width ? t.width - 1 : xi);
				yi = yi < 0 ? 0 : (yi >= t.height ? t.height - 1 : yi);
				break;
		}
		return texel_rgb(&pixels[t.pixelOffset + (yi * t.width + xi) * 3], t.srgb).x;
	};
	return (1.0f - fy) * ((1.0f - fx) * texel(x0, y0) + fx * texel(x0 + 1, y0)) +
				  fy  * ((1.0f - fx) * texel(x0, y0 + 1) + fx * texel(x0 + 1, y0 + 1));
}

// `n` is the unit shading normal facing the ray, `dpdu` the UNNORMALIZED surface tangent (world units per unit
// u, as the CPU's rec.dpdu): the finite-difference slope is per unit u, so its length sets how steep the relief
// reads. dpdv = cross(n, dpdu) as bump_map_material::apply() does.
inline __host__ __device__ float3 gpu_bump_shading_normal(const TextureData* textures, const unsigned char* pixels,
														  int texIdx, float scale, float u, float v,
														  const float3& n, const float3& dpdu, float footprintStep) {
	const float step = footprintStep > 1e-8f ? footprintStep : 0.001f;   // 0.001: the CPU's no-footprint fallback
	const float disp   = scale * gpu_sample_displacement(textures, pixels, texIdx, u,        v);
	const float disp_u = scale * gpu_sample_displacement(textures, pixels, texIdx, u + step, v);
	const float disp_v = scale * gpu_sample_displacement(textures, pixels, texIdx, u,        v + step);
	float3 dpdv = cross(n, dpdu);
	if (dot(dpdv, dpdv) < 1e-12f) dpdv = make_float3(0.0f, 1.0f, 0.0f);
	float ox, oy, oz;
	apply_bump_map<float>(disp, disp_u, disp_v, step, step, n.x, n.y, n.z, dpdu.x, dpdu.y, dpdu.z,
						  dpdv.x, dpdv.y, dpdv.z, ox, oy, oz);
	return make_float3(ox, oy, oz);
}

// The CPU's rec.dpdu and rec.dpdv for a triangle: the 2x2 UV-edge Jacobian solve, UNNORMALIZED, in double. The
// solve cancels badly in float on a small triangle with tiny UV deltas, so it is done in double as on the CPU.
// A degenerate UV mapping falls back to a unit edge for dpdu and cross(n, dpdu) for dpdv, as the CPU does.
inline __host__ __device__ void gpu_triangle_dpdu_dpdv(const float3& p0, const float3& p1, const float3& p2,
													   const float2& uv0, const float2& uv1, const float2& uv2,
													   const float3& n, float3& dpdu, float3& dpdv) {
	const double e1x = (double)p1.x - p0.x, e1y = (double)p1.y - p0.y, e1z = (double)p1.z - p0.z;
	const double e2x = (double)p2.x - p0.x, e2y = (double)p2.y - p0.y, e2z = (double)p2.z - p0.z;
	const double du1 = (double)uv1.x - uv0.x, dv1 = (double)uv1.y - uv0.y;
	const double du2 = (double)uv2.x - uv0.x, dv2 = (double)uv2.y - uv0.y;
	const double det = du1 * dv2 - dv1 * du2;
	if (fabs(det) > 1e-12) {
		const double inv = 1.0 / det;
		dpdu = make_float3((float)(inv * (dv2 * e1x - dv1 * e2x)),
						   (float)(inv * (dv2 * e1y - dv1 * e2y)),
						   (float)(inv * (dv2 * e1z - dv1 * e2z)));
		dpdv = make_float3((float)(inv * (-du2 * e1x + du1 * e2x)),
						   (float)(inv * (-du2 * e1y + du1 * e2y)),
						   (float)(inv * (-du2 * e1z + du1 * e2z)));
		return;
	}
	const double len2 = e1x * e1x + e1y * e1y + e1z * e1z;
	if (len2 > 1e-14) {
		const double inv = 1.0 / sqrt(len2);
		dpdu = make_float3((float)(e1x * inv), (float)(e1y * inv), (float)(e1z * inv));
	} else {
		dpdu = make_float3(1.0f, 0.0f, 0.0f);
	}
	dpdv = cross(n, dpdu);
}

// The viewport of a perspective pinhole/thin-lens camera, enough to rebuild the two one-pixel-offset rays the
// CPU's get_ray() attaches to a primary ray. `enabled` is false for every other camera (orthographic,
// spherical, realistic, animated), which the CPU gives no differentials either.
struct GpuBumpFootprint {
	int    enabled;
	float3 lowerLeft;
	float3 horizontal;   // the whole viewport's extent, so one pixel is horizontal / width
	float3 vertical;
	int    width, height;
};

inline __host__ __device__ GpuBumpFootprint gpu_make_bump_footprint(const GpuCameraParams& cam, int width, int height) {
	GpuBumpFootprint f;
	f.enabled    = (cam.kind == CameraKind::Perspective && !cam.animated && width > 0 && height > 0) ? 1 : 0;
	f.lowerLeft  = cam.lower_left_corner;
	f.horizontal = cam.horizontal;
	f.vertical   = cam.vertical;
	f.width      = width;
	f.height     = height;
	return f;
}

// 0.5 * (|dudx| + |dudy|) for a primary ray (origin O, unit direction d) that hit the surface at p: the two
// neighbouring-pixel rays, offset by one pixel on the viewport and sharing O, are intersected with the tangent
// plane (n, p) and the offsets projected onto (dpdu, dpdv) by least squares - the CPU's
// SurfaceInteraction::compute_differentials(). Returns 0 where the CPU would have no footprint (not a
// perspective camera, a ray parallel to the plane, a degenerate mapping), which callers read as "use 0.001".
inline __host__ __device__ float gpu_bump_footprint_step(const GpuBumpFootprint& f, const float3& O, const float3& d,
														 const float3& p, const float3& n,
														 const float3& dpdu, const float3& dpdv) {
	if (!f.enabled) return 0.0f;
	float3 pn = cross(f.horizontal, f.vertical);
	const float pnLen = sqrtf(dot(pn, pn));
	if (pnLen < 1e-12f) return 0.0f;
	pn = pn * (1.0f / pnLen);
	const float den = dot(pn, d);
	if (fabsf(den) < 1e-8f) return 0.0f;
	// Length of the UNnormalized primary direction (pixel sample - ray origin): it lands on the viewport plane.
	const float L = dot(pn, f.lowerLeft - O) / den;
	const float3 D  = d * L;
	const float3 Dx = D + f.horizontal * (1.0f / (float)f.width);
	const float3 Dy = D + f.vertical   * (1.0f / (float)f.height);
	const float dnx = dot(n, Dx), dny = dot(n, Dy);
	if (dnx == 0.0f || dny == 0.0f) return 0.0f;
	const float3 pop = p - O;
	const float3 dpdx = Dx * (dot(n, pop) / dnx) - pop;     // (O + tx * Dx) - p
	const float3 dpdy = Dy * (dot(n, pop) / dny) - pop;
	const double a00 = dot(dpdu, dpdu), a01 = dot(dpdu, dpdv), a11 = dot(dpdv, dpdv);
	const double det = a00 * a11 - a01 * a01;
	if (!(fabs(det) > 0.0)) return 0.0f;
	const double inv = 1.0 / det;
	const double bx0 = dot(dpdu, dpdx), bx1 = dot(dpdv, dpdx);
	const double by0 = dot(dpdu, dpdy), by1 = dot(dpdv, dpdy);
	const double dudx = (a11 * bx0 - a01 * bx1) * inv;
	const double dudy = (a11 * by0 - a01 * by1) * inv;
	const double step = 0.5 * (fabs(dudx) + fabs(dudy));
	return (step == step && step < 1e8) ? (float)step : 0.0f;
}
