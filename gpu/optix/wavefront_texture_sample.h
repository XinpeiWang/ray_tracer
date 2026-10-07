#pragma once
// wavefront_texture_sample.h -- the GPU texture lookup (image, checker, marble, windy, wrinkled, dots, bilerp, mix, noise) taking its texture table and
// pixel buffer as explicit arguments rather than reading a launch-params global, so the wavefront kernels and the SPPM kernels (which have different
// launch-params structs) share one implementation. Moved verbatim out of wavefront_device_helpers.h.

#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/noise.h"   // fbm_simple/turbulence_simple/perlin_noise

// De Casteljau cubic Bezier evaluation - duplicated from optix_device_
// helpers.h's marble_cubic_bezier4 (with the wf_ prefix), matching this
// file's existing pattern of not sharing device helpers with the recursive
// path.
__device__ __forceinline__ float3 wf_marble_cubic_bezier4(
		const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t) {
	const float s = 1.0f - t;
	const float w0 = s*s*s, w1 = 3.0f*s*s*t, w2 = 3.0f*s*t*t, w3 = t*t*t;
	return make_float3(
		w0*p0.x + w1*p1.x + w2*p2.x + w3*p3.x,
		w0*p0.y + w1*p1.y + w2*p2.y + w3*p3.y,
		w0*p0.z + w1*p1.z + w2*p2.z + w3*p3.z);
}

// Matches marble_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_marble_texture (with the wf_ prefix),
// same no-shared-device-helpers convention as wf_marble_cubic_bezier4 above.
__device__ __forceinline__ float3 wf_sample_marble_texture(const TextureData& tex, const float3& p) {
	const float px = p.x * tex.marbleScale, py = p.y * tex.marbleScale, pz = p.z * tex.marbleScale;
	const float fbm_val = fbm_simple<float>(px, py, pz, tex.omega, tex.octaves);
	const float marble = py + tex.marbleVariation * fbm_val;
	float t = 0.5f + 0.5f * sinf(marble);

	constexpr int kN = 9;
	const float3 knots[kN] = {
		make_float3(.58f,.58f,.60f), make_float3(.58f,.58f,.60f), make_float3(.58f,.58f,.60f),
		make_float3(.50f,.50f,.50f), make_float3(.60f,.59f,.58f), make_float3(.58f,.58f,.60f),
		make_float3(.58f,.58f,.60f), make_float3(.20f,.20f,.33f), make_float3(.58f,.58f,.60f)
	};
	constexpr int nSeg = kN - 3;
	int first = static_cast<int>(t * nSeg);
	if (first >= nSeg) first = nSeg - 1;
	const float lt = t * nSeg - first;

	float3 rgb = wf_marble_cubic_bezier4(knots[first], knots[first+1], knots[first+2], knots[first+3], lt);
	return make_float3(fminf(rgb.x * 1.5f, 1.0f), fminf(rgb.y * 1.5f, 1.0f), fminf(rgb.z * 1.5f, 1.0f));
}

// Matches windy_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_windy_texture (with the wf_ prefix), same
// no-shared-device-helpers convention as wf_marble_cubic_bezier4 above.
__device__ __forceinline__ float3 wf_sample_windy_texture(const float3& p) {
	const float windStrength = fbm_simple<float>(0.1f*p.x, 0.1f*p.y, 0.1f*p.z, 0.5f, 3);
	const float waveHeight   = fbm_simple<float>(p.x, p.y, p.z, 0.5f, 6);
	const float v = fabsf(windStrength) * waveHeight;
	float t = 0.5f + 0.5f * v;
	t = fminf(fmaxf(t, 0.0f), 1.0f);
	return make_float3(t, t, t);
}

// Matches wrinkled_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_wrinkled_texture (with the wf_ prefix).
__device__ __forceinline__ float3 wf_sample_wrinkled_texture(const TextureData& tex, const float3& p) {
	const float v = turbulence_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
	const float t = fminf(fmaxf(v, 0.0f), 1.0f);
	return make_float3(t, t, t);
}

// Matches dots_texture::is_inside_dot() (texture.h) exactly - duplicated
// from optix_device_helpers.h's is_inside_dot (with the wf_ prefix).
__device__ __forceinline__ bool wf_is_inside_dot(float s, float t) {
	const float sCell = floorf(s + 0.5f);
	const float tCell = floorf(t + 0.5f);
	if (perlin_noise<float>(sCell + 0.5f, tCell + 0.5f, 0.5f) <= 0.0f) return false;
	constexpr float radius = 0.35f;
	constexpr float maxShift = 0.5f - radius;
	const float sCenter = sCell + maxShift * perlin_noise<float>(sCell + 1.5f, tCell + 2.8f, 0.5f);
	const float tCenter = tCell + maxShift * perlin_noise<float>(sCell + 4.5f, tCell + 9.8f, 0.5f);
	const float ds = s - sCenter, dt = t - tCenter;
	return ds*ds + dt*dt < radius*radius;
}

// Matches bilerp_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_bilerp_texture (with the wf_ prefix).
// color1/color2 carry v00/v01; the other two corners are packed into
// uScale/vScale/omega and marbleScale/marbleVariation/mixAmount (see
// TextureKind::Bilerp's own comment, optix_types.h).
__device__ __forceinline__ float3 wf_sample_bilerp_texture(const TextureData& tex, float u, float v) {
	const float3& v00 = tex.color1; const float3& v01 = tex.color2;
	const float3 v10 = make_float3(tex.uScale, tex.vScale, tex.omega);
	const float3 v11 = make_float3(tex.marbleScale, tex.marbleVariation, tex.mixAmount);
	const float a = (1.0f-u)*(1.0f-v), b = u*(1.0f-v), c = (1.0f-u)*v, d = u*v;
	return make_float3(
		a*v00.x + b*v10.x + c*v01.x + d*v11.x,
		a*v00.y + b*v10.y + c*v01.y + d*v11.y,
		a*v00.z + b*v10.z + c*v01.z + d*v11.z);
}

// Samples a texture by index - duplicated from optix_device_helpers.h's
// sample_texture (with the wf_ prefix), matching this file's existing
// pattern of not sharing device helpers with the recursive path. Only
// called when MaterialData::textureIdx >= 0 (Lambertian/NormalMappedLambertian).
__device__ __forceinline__ float3 wf_sample_texture(
	const TextureData* textures, const unsigned char* texturePixels,
	int textureIdx, float u, float v, const float3& p)
{
	const TextureData& tex = textures[textureIdx];
	// Shared by the Image case below and by UVChecker/Mix's own
	// tex1ImageIdx/tex2ImageIdx (a one-level-nested bare imagemap - see
	// TextureData's own comment and optix_device_helpers.h's identical
	// sampleImage lambda) - factored out so both call sites in THIS
	// duplicated copy share one instance too. See optix_device_helpers.h's
	// own sampleImage comment for the full wide-clamp/wrap-mode/BILINEAR
	// rationale - mirrored here verbatim (same cross-module duplication
	// reason every other wf_ helper in this file is duplicated rather than
	// shared). BILINEAR, not nearest-neighbor, matching CPU's own
	// mipmap_texture::value() (texture.h)'s LOD-0 bilerp() exactly - a
	// prior version of this function (and the comment it carried) was
	// genuinely wrong about matching CPU's nearest-neighbor behavior; CPU
	// has never been nearest-neighbor here. See optix_device_helpers.h's
	// own comment for the full investigation (scene J2's regional-diff
	// finding) this fixes.
	auto sampleImage = [&](const TextureData& t) -> float3 {
		if (t.width <= 0 || t.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uw = fminf(fmaxf(u, -1024.0f), 1024.0f);
		const float vw = fminf(fmaxf(1.0f - v, -1024.0f), 1024.0f);
		// double, not float - see optix_device_helpers.h's identical
		// sampleImage comment for why (matches CPU's own bilerp() promotion,
		// mipmap.h - guards against float's 2^24 exact-integer limit for a
		// wide-tiled UV times a large texture width).
		const double x = (double)uw * t.width - 0.5;
		const double y = (double)vw * t.height - 0.5;
		const int x0 = static_cast<int>(floor(x)), x1 = x0 + 1;
		const int y0 = static_cast<int>(floor(y)), y1 = y0 + 1;
		const float fx = (float)(x - x0), fy = (float)(y - y0);

		// No `default:` case, deliberately - see optix_device_helpers.h's
		// identical comment for why (lets the compiler flag a future
		// unhandled GpuWrapMode enumerator here too).
		auto wrapTexel = [&](int xi, int yi) -> float3 {
			switch (t.wrapMode) {
			case GpuWrapMode::Repeat:
				xi = ((xi % t.width) + t.width) % t.width;
				yi = ((yi % t.height) + t.height) % t.height;
				break;
			case GpuWrapMode::Black:
				if (xi < 0 || xi >= t.width || yi < 0 || yi >= t.height) return make_float3(0.0f, 0.0f, 0.0f);
				break;
			case GpuWrapMode::Clamp:
				xi = min(max(xi, 0), t.width - 1);
				yi = min(max(yi, 0), t.height - 1);
				break;
			}
			const unsigned char* px = texturePixels + t.pixelOffset + (yi * t.width + xi) * 3;
			return texel_rgb(px, t.srgb);
		};

		const float3 c00 = wrapTexel(x0, y0);
		const float3 c10 = wrapTexel(x1, y0);
		const float3 c01 = wrapTexel(x0, y1);
		const float3 c11 = wrapTexel(x1, y1);
		return (1.0f - fy) * ((1.0f - fx) * c00 + fx * c10)
		     +         fy  * ((1.0f - fx) * c01 + fx * c11);
	};
	if (tex.kind == TextureKind::Image) {
		return sampleImage(tex);
	} else if (tex.kind == TextureKind::Checker) {
		const int xi = static_cast<int>(floorf(tex.noiseScale * p.x));
		const int yi = static_cast<int>(floorf(tex.noiseScale * p.y));
		const int zi = static_cast<int>(floorf(tex.noiseScale * p.z));
		const bool is_even = ((xi + yi + zi) % 2) == 0;
		return is_even ? tex.color1 : tex.color2;
	} else if (tex.kind == TextureKind::UVChecker) {
		// Matches optix_device_helpers.h's sample_texture() UVChecker branch
		// (and uv_checker_texture::value(), texture.h) exactly, including
		// tex1ImageIdx/tex2ImageIdx's one-level-nested bare imagemap support.
		const int ui = static_cast<int>(floorf(u * tex.uScale));
		const int vi = static_cast<int>(floorf(v * tex.vScale));
		const bool is_even = ((ui + vi) % 2) == 0;
		// Only the winning cell's slot is sampled - see
		// optix_device_helpers.h's identical UVChecker branch comment.
		return is_even
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::FBm) {
		// Matches optix_device_helpers.h's sample_texture() FBm branch (and
		// fbm_texture::value(), texture.h) exactly.
		const float v = fbm_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
		float t = 0.5f + 0.5f * v;
		t = fminf(fmaxf(t, 0.0f), 1.0f);
		return make_float3(t, t, t);
	} else if (tex.kind == TextureKind::Marble) {
		return wf_sample_marble_texture(tex, p);
	} else if (tex.kind == TextureKind::Windy) {
		return wf_sample_windy_texture(p);
	} else if (tex.kind == TextureKind::Wrinkled) {
		return wf_sample_wrinkled_texture(tex, p);
	} else if (tex.kind == TextureKind::Dots) {
		const bool inside = wf_is_inside_dot(u, v);
		return inside
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::Bilerp) {
		return wf_sample_bilerp_texture(tex, u, v);
	} else if (tex.kind == TextureKind::Mix) {
		// Matches optix_device_helpers.h's sample_texture() Mix branch (and
		// mix_texture::value(), texture.h) exactly, including
		// amountImageIdx's one-level-nested bare imagemap support for
		// "amount" itself.
		const float3 c1 = (tex.tex1ImageIdx >= 0) ? sampleImage(textures[tex.tex1ImageIdx]) : tex.color1;
		const float3 c2 = (tex.tex2ImageIdx >= 0) ? sampleImage(textures[tex.tex2ImageIdx]) : tex.color2;
		const float amt = (tex.amountImageIdx >= 0) ? sampleImage(textures[tex.amountImageIdx]).x : tex.mixAmount;
		return (1.0f - amt) * c1 + amt * c2;
	} else {
		const float turb = turbulence_simple<float>(p.x, p.y, p.z, 0.5f, 7);
		const float s = 0.5f * (1.0f + sinf(tex.noiseScale * p.z + 10.0f * turb));
		return make_float3(s, s, s);
	}
}

