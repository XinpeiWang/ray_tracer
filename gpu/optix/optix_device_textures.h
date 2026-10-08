#pragma once
// optix_device_textures.h -- part 2 of 5 of optix_device_helpers.h (included by it, in order; not meant to be included on its own).

// Samples a texture by index (see TextureData in optix_types.h), matching
// CPU's texture::value(u, v, p) dispatch (src/TheRestOfYourLife/texture.h)
// for the two kinds ported so far - only called when
// MaterialData::textureIdx >= 0 (Lambertian only for now, see
// shade_material() below). u/v are used for Image; p (world-space hit
// point) is used for Noise - CPU's own base-class interface takes all
// three regardless of which the concrete texture subclass actually needs,
// same here.
__device__ __forceinline__ float3 sample_texture(int textureIdx, float u, float v, const float3& p) {
	const TextureData& tex = params.textures[textureIdx];
	// Shared by the Image case below and by UVChecker/Mix's own
	// tex1ImageIdx/tex2ImageIdx (a one-level-nested bare imagemap bound to
	// tex1/tex2 instead of a flat literal - see TextureData's own comment)
	// - factored out so both call sites share one copy of the pixel-lookup
	// math instead of duplicating it. wide-clamp uv to [-1024,1024] (a
	// safety rail against a pathological UV, NOT [0,1] - see wide_clamp()'s
	// own comment), flip v (stored image rows are top-to-bottom, v=0 is the
	// bottom of the [0,1] texture-coordinate convention), then bilinearly
	// interpolate the 4 texels around (u,v), each wrapped independently per
	// t.wrapMode (GpuWrapMode's own comment) - Repeat/Black/Clamp. A failed
	// image load (width/height <= 0) matches CPU's own solid-cyan debugging
	// fallback (texture.h) exactly.
	//
	// BILINEAR, not nearest-neighbor: this now matches CPU's own
	// mipmap_texture::value() (texture.h) exactly - its "zero derivatives"
	// path (used whenever no real screen-space UV footprint is available,
	// e.g. every non-primary-ray hit) is NOT a nearest-neighbor lookup, it's
	// MIPMap::filter()'s own LOD-0 `bilerp()` (mipmap.h) - a real 4-tap
	// bilinear blend with texel CENTERS at (i+0.5)/width, hence the `-0.5f`
	// before floor() below (mirrors bilerp()'s own `s*w - 0.5`). A prior
	// version of this function did pure nearest-neighbor and claimed (in
	// this now-corrected comment) that this matched CPU's value() - it did
	// not; CPU has never been nearest-neighbor here, even without ray
	// differentials. Confirmed as a real, non-noise, texture-filtering
	// discrepancy (not just the already-known MaterialType::
	// DiffuseTransmission NEE-strategy noise) by a CPU/GPU parity sweep's
	// regional-diff check on scene J2 (a 4x4-pixel texture, where
	// nearest-vs-bilinear snapping is maximally visible) - see tests/
	// integration/material_cpu_gpu_parity_tests.cpp's own J2 comment. This
	// still does NOT implement full EWA/mipmap minification filtering (no
	// GPU mip pyramid exists - TextureData holds one resolution level only,
	// a real, larger, and still-open architectural gap for a texture
	// MINIFIED below screen resolution) - bilinear only fixes MAGNIFICATION
	// (texture resolution below screen resolution, J2's exact case and the
	// common one for small/debug textures), which needs no mip chain at all.
	auto sampleImage = [&](const TextureData& t) -> float3 {
		if (t.width <= 0 || t.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uw = fminf(fmaxf(u, -1024.0f), 1024.0f);
		const float vw = fminf(fmaxf(1.0f - v, -1024.0f), 1024.0f);
		// double, not float, for the multiply - matches CPU's own bilerp()
		// (src/shared/mipmap.h), which promotes to double for this exact
		// computation: a wide-tiled UV (now possible here too, via the
		// +-1024 clamp above) times a large texture width can exceed
		// float's exact-integer range (2^24), losing sub-texel precision
		// right before floor() picks the pixel index.
		const double x = (double)uw * t.width - 0.5;
		const double y = (double)vw * t.height - 0.5;
		const int x0 = static_cast<int>(floor(x)), x1 = x0 + 1;
		const int y0 = static_cast<int>(floor(y)), y1 = y0 + 1;
		const float fx = (float)(x - x0), fy = (float)(y - y0);

		// One corner's wrapped texel fetch - No `default:` case in the
		// switch, deliberately - this is what lets the compiler
		// (-Wswitch/MSVC's equivalent) flag a future 4th GpuWrapMode
		// enumerator left unhandled here, in BOTH this copy and
		// wavefront_device_helpers.h's own duplicate, instead of both
		// silently falling through to Clamp behavior with no diagnostic
		// anywhere.
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
			const unsigned char* px = params.texturePixels + t.pixelOffset + (yi * t.width + xi) * 3;
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
		// Matches checker_texture::value() (texture.h:53-61) exactly: floor
		// each world-space coordinate scaled by 1/scale, sum the three
		// integers, and pick a color by parity - equality-to-zero on `%2`
		// is sign-agnostic, so this is correct for negative coordinates too
		// (just like the CPU version, which relies on the same C++ rule).
		const int xi = static_cast<int>(floorf(tex.noiseScale * p.x));
		const int yi = static_cast<int>(floorf(tex.noiseScale * p.y));
		const int zi = static_cast<int>(floorf(tex.noiseScale * p.z));
		const bool is_even = ((xi + yi + zi) % 2) == 0;
		return is_even ? tex.color1 : tex.color2;
	} else if (tex.kind == TextureKind::UVChecker) {
		// Matches uv_checker_texture::value() (texture.h) exactly: parity of
		// floor(u*uscale)+floor(v*vscale) - pbrt-v4's own UV-tiled
		// checkerboard convention, deliberately NOT the world-space Checker
		// case above (see TextureKind::UVChecker's own comment). tex1ImageIdx/
		// tex2ImageIdx (-1 by default) let either cell colour instead be a
		// one-level-nested bare imagemap - see TextureData's own comment.
		const int ui = static_cast<int>(floorf(u * tex.uScale));
		const int vi = static_cast<int>(floorf(v * tex.vScale));
		const bool is_even = ((ui + vi) % 2) == 0;
		// Only the winning cell's slot is sampled - is_even already fully
		// determines which of tex1/tex2 is used, so evaluating BOTH
		// (including a global-memory image fetch when either is nested)
		// would be pure waste on this per-ray hot path.
		return is_even
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(params.textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(params.textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::FBm) {
		// Matches fbm_texture::value() (texture.h) exactly: fbm_simple(p,
		// omega, octaves) mapped from [-~1,~1] to a clamped [0,1] greyscale.
		const float v = fbm_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
		float t = 0.5f + 0.5f * v;
		t = fminf(fmaxf(t, 0.0f), 1.0f);
		return make_float3(t, t, t);
	} else if (tex.kind == TextureKind::Marble) {
		return sample_marble_texture(tex, p);
	} else if (tex.kind == TextureKind::Windy) {
		return sample_windy_texture(p);
	} else if (tex.kind == TextureKind::Wrinkled) {
		return sample_wrinkled_texture(tex, p);
	} else if (tex.kind == TextureKind::Dots) {
		// Same one-level-nested-imagemap convention as UVChecker/Mix above -
		// tex1ImageIdx/tex2ImageIdx (-1 by default) let inside/outside each
		// independently be a bare imagemap instead of the flat color1/color2.
		const bool inside = is_inside_dot(u, v);
		return inside
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(params.textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(params.textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::Bilerp) {
		return sample_bilerp_texture(tex, u, v);
	} else if (tex.kind == TextureKind::Mix) {
		// Matches mix_texture::value() (texture.h) exactly: lerp, no
		// footprint/UV dependence. tex1ImageIdx/tex2ImageIdx - see
		// UVChecker's own identical comment just above. amountImageIdx (-1
		// by default) lets "amount" itself be a one-level-nested bare
		// imagemap instead of the flat tex.mixAmount scalar - only its .x
		// channel is read, matching mix_texture's own amount_tex convention.
		const float3 c1 = (tex.tex1ImageIdx >= 0) ? sampleImage(params.textures[tex.tex1ImageIdx]) : tex.color1;
		const float3 c2 = (tex.tex2ImageIdx >= 0) ? sampleImage(params.textures[tex.tex2ImageIdx]) : tex.color2;
		const float amt = (tex.amountImageIdx >= 0) ? sampleImage(params.textures[tex.amountImageIdx]).x : tex.mixAmount;
		return (1.0f - amt) * c1 + amt * c2;
	} else {
		// Matches noise_texture::value() (texture.h:127-129) exactly:
		// color(.5,.5,.5) * (1 + sin(scale*p.z + 10*turb(p,7))), where
		// turb(p,7) is perlin::turb's own default (depth=7, omega=0.5,
		// perlin.h:37) delegating to turbulence_simple<T> (noise.h:252).
		const float turb = turbulence_simple<float>(p.x, p.y, p.z, 0.5f, 7);
		const float s = 0.5f * (1.0f + sinf(tex.noiseScale * p.z + 10.0f * turb));
		return make_float3(s, s, s);
	}
}

// Emission the caller's local `emission` variable should be initialized to
// BEFORE calling shade_material() - see shade_material()'s own comment on
// that in/out convention (NEE contributions get added on top of whatever
// this returns). Guards the mat.emission union-slot read behind
// mat.type == DiffuseLight, so a material that reuses that slot for
// something else entirely (DiffuseTransmission's transmittance,
// Subsurface's sigma_a, ...) never depends on shade_material()'s own switch
// remembering to reset it back to zero - previously TWO separate cases each
// carried a near-identical "don't forget to zero this" comment doing that
// reset by hand; a third material reusing this slot in the future would
// silently need a fourth. One-sided per front_face UNLESS mat.twoSided
// (matches CPU's diffuse_light::emitted()/is_two_sided()). textureIdx>=0
// samples a real texture at the given UV - map_Ke on triangles (see
// add_diffuse_light()'s textureIdx comment) or a pbrt AreaLightSource
// "filename" image (any geometry the caller passes real UV for) - scaled by
// emissionScale (a no-op 1.0 multiply when the light wasn't built with a
// "scale" param). Any future geometry type gets both for free by going
// through this accessor instead of reading mat.emission raw.
__device__ __forceinline__ float3 material_emission(
		const MaterialData& mat, bool front_face,
		float uv_u = 0.0f, float uv_v = 0.0f,
		float3 hit_point = make_float3(0.0f, 0.0f, 0.0f)) {
	if (mat.type != MaterialType::DiffuseLight || (!mat.twoSided && !front_face))
		return make_float3(0.0f, 0.0f, 0.0f);
	if (mat.textureIdx >= 0) {
		const float3 texel = sample_texture(mat.textureIdx, uv_u, uv_v, hit_point);
		return make_float3(texel.x * mat.emissionScale, texel.y * mat.emissionScale, texel.z * mat.emissionScale);
	}
	return mat.emission;
}

// Alpha-cutout test (OBJ/.mtl map_d): returns true if the hit should be
// kept, false if it should be treated as fully transparent - a caller at an
// any-hit call site (radiance or shadow) should then call
// optixIgnoreIntersection() so the ray continues past this point as if the
// geometry weren't there. A no-op (always true) for alphaMaskTexIdx < 0,
// i.e. the overwhelming majority of materials. pbrt-v4's stochastic alpha test
// (cpu/primitive.cpp:57-71): a hit is kept with probability `alpha`, decided by a
// hash of the ray itself (ray_hash.h), so shadow rays and radiance rays make
// independent, state-free choices and a partial-alpha material is partly
// transparent instead of snapping to a threshold. Must be called from an any-hit
// program: it reads the current ray.
__device__ __forceinline__ bool passes_alpha_cutout(int alphaMaskTexIdx, float u, float v, const float3& p) {
	if (alphaMaskTexIdx < 0) return true;
	const float a = sample_texture(alphaMaskTexIdx, u, v, p).x;
	const float3 o = optixGetWorldRayOrigin();
	const float3 d = optixGetWorldRayDirection();
	return ray_hash::alphaPasses(a, o.x, o.y, o.z, d.x, d.y, d.z);
}
