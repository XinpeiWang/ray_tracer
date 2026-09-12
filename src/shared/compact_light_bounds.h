// compact_light_bounds.h
// Quantised, cache-friendly light bound for BVH nodes, ported from pbrt-v4.
//
// CompactLightBounds stores a LightBounds in 20 bytes by quantising:
//   - emission axis w        -> OctahedralVector (2 x uint16_t = 4 bytes)
//   - phi                    -> float            (4 bytes)
//   - cosTheta_o, cosTheta_e -> 15-bit integers  (packed bitfield)
//   - twoSided               -> 1-bit            (packed bitfield)
//   - AABB corners           -> 2 x 3 x uint16_t (12 bytes) relative to scene AABB
//
// API (mirrors pbrt-v4 lightsamplers.h):
//   CompactLightBounds(lb, allBoundsMin, allBoundsMax)
//   Importance(px,py,pz, nx,ny,nz, allBoundsMin, allBoundsMax)
//   Bounds(allBoundsMin, allBoundsMax, outMin[3], outMax[3])
//   CosTheta_o(), CosTheta_e(), TwoSided()
//
// Dependencies:
//   light_bounds.h         -- LightBounds
//   octahedral_variance.h  -- OctahedralVector
//   scalar_math.h          -- SafeSqrt, Sqr, Clamp, Lerp
//   direction_cone.h       -- BoundSubtendedDirections
//
// References: pbrt-v4 src/pbrt/lightsamplers.h  (Apache-2.0)

#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "light_bounds.h"          // LightBounds, Importance free-fn, Union
#include "octahedral_variance.h"   // OctahedralVector
#include "scalar_math.h"           // SafeSqrt, Sqr, Clamp, Lerp
#include "direction_cone.h"        // BoundSubtendedDirections

#include "cpu_gpu.h"

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4141 4293 4244)
#endif

// ===========================================================================
// CompactLightBounds
// ===========================================================================

struct CompactLightBounds {
	// -----------------------------------------------------------------------
	// Construction
	// -----------------------------------------------------------------------
	CompactLightBounds() = default;

	// Construct from a full LightBounds + the scene-wide AABB extents.
	// allBMin/allBMax are the scene-wide bounding box used for quantisation.
	CPU_GPU CompactLightBounds(const LightBounds& lb,
								float allBMinX, float allBMinY, float allBMinZ,
								float allBMaxX, float allBMaxY, float allBMaxZ)
	{
		// Normalise and store w via OctahedralVector
		w = OctahedralVector(lb.wx, lb.wy, lb.wz);

		phi = lb.phi;

		// Quantise cosines: [-1,1] -> [0, 32767]
		const unsigned int qco = QuantizeCos(lb.cosTheta_o);
		const unsigned int qce = QuantizeCos(lb.cosTheta_e);
		packedCosThetaAndSide = (qco & 0x7FFFu) | ((qce & 0x7FFFu) << 15) | (lb.twoSided ? (1u << 30) : 0u);

		// Quantise AABB corners into [0, 65535] relative to the scene AABB.
		// Corner 0 = pMin (floor), corner 1 = pMax (ceil) to be conservative.
		float bMin[3] = { lb.bMinX, lb.bMinY, lb.bMinZ };
		float bMax[3] = { lb.bMaxX, lb.bMaxY, lb.bMaxZ };
		float aMin[3] = { allBMinX, allBMinY, allBMinZ };
		float aMax[3] = { allBMaxX, allBMaxY, allBMaxZ };

		for (int c = 0; c < 3; ++c) {
			qb[0][c] = static_cast<uint16_t>(
				std::floor(QuantizeBounds(bMin[c], aMin[c], aMax[c])));
			qb[1][c] = static_cast<uint16_t>(
				std::ceil( QuantizeBounds(bMax[c], aMin[c], aMax[c])));
		}
	}

	// -----------------------------------------------------------------------
	// Accessors (dequantise)
	// -----------------------------------------------------------------------
	CPU_GPU bool  TwoSided()   const { return (packedCosThetaAndSide & (1u << 30)) != 0; }
	CPU_GPU float CosTheta_o() const { return 2.f * ((packedCosThetaAndSide & 0x7FFFu) / 32767.f) - 1.f; }
	CPU_GPU float CosTheta_e() const { return 2.f * (((packedCosThetaAndSide >> 15) & 0x7FFFu) / 32767.f) - 1.f; }

	// Reconstruct the float AABB from quantised corners + scene AABB.
	CPU_GPU void Bounds(float allBMinX, float allBMinY, float allBMinZ,
						float allBMaxX, float allBMaxY, float allBMaxZ,
						float outMin[3], float outMax[3]) const
	{
		float aMin[3] = { allBMinX, allBMinY, allBMinZ };
		float aMax[3] = { allBMaxX, allBMaxY, allBMaxZ };
		for (int c = 0; c < 3; ++c) {
			outMin[c] = Lerp(qb[0][c] / 65535.f, aMin[c], aMax[c]);
			outMax[c] = Lerp(qb[1][c] / 65535.f, aMin[c], aMax[c]);
		}
	}

	// -----------------------------------------------------------------------
	// Importance(p, n, allBounds)
	//
	// Mirrors pbrt-v4 CompactLightBounds::Importance(Point3f, Normal3f, Bounds3f).
	// Dequantises on the fly then runs the same cone-angle arithmetic as
	// LightBounds::Importance.
	//
	// Hand-flattened into one self-contained function (no BoundSubtendedDirections()
	// call, no cosSubClamped/sinSubClamped lambdas) - this was an ATTEMPTED fix
	// for GPU-recursive's own KNOWN UNRESOLVED BUG (see gpu_light_bvh_sample_
	// index()'s own comment, optix_device_helpers_lighting.h, for the full
	// diagnosis and this attempt's own negative result): this exact function
	// returns 0.0 for both children at the light BVH's root on effectively
	// 100% of calls inside that backend's one-thread-per-pixel recursive
	// megakernel, while the byte-identical uploaded data computes healthy
	// nonzero results on the host AND on wavefront's separately-compiled,
	// shallower kernels - a genuine NVCC floating-point divergence specific
	// to this megakernel, not a data/logic bug. This flatten matches the fix
	// that resolved the ONE other confirmed instance of this exact bug class
	// in this codebase (gpu_cloud_density()'s own dnoise() history,
	// optix_intersection_sphere.h) - but re-instrumented testing against a
	// fresh 23-node/12-light tree AFTER this flatten still showed 100%
	// failure (2530301/2530301 calls), unchanged from before it. Kept anyway
	// as a real, if smaller, improvement independent of that bug: no lambda,
	// and reuses the diagonal-length/distance-squared values Importance()
	// already computed instead of recomputing them under BoundSubtendedDirections()'s
	// own local names (that function is otherwise unaffected - still used
	// by LightBounds::Importance() in light_bounds.h, the CPU-only sibling
	// this struct's own Importance() doesn't call). Functionally identical
	// to the previous BoundSubtendedDirections()-calling version - see git
	// history for that version if this ever needs to be cross-checked
	// line-by-line again.
	// -----------------------------------------------------------------------
	CPU_GPU float Importance(float px, float py, float pz,
							  float nx, float ny, float nz,
							  float allBMinX, float allBMinY, float allBMinZ,
							  float allBMaxX, float allBMaxY, float allBMaxZ) const
	{
		// Dequantise AABB
		float bMin[3], bMax[3];
		Bounds(allBMinX, allBMinY, allBMinZ, allBMaxX, allBMaxY, allBMaxZ, bMin, bMax);

		// Dequantise cone angles
		float cosTheta_o = CosTheta_o();
		float cosTheta_e = CosTheta_e();

		// Decode emission axis
		float wox, woy, woz;
		w.ToVec3(wox, woy, woz);

		// Centroid and clamped d2
		float cx = (bMin[0] + bMax[0]) * 0.5f;
		float cy = (bMin[1] + bMax[1]) * 0.5f;
		float cz = (bMin[2] + bMax[2]) * 0.5f;
		float dx = bMax[0] - bMin[0], dy = bMax[1] - bMin[1], dz = bMax[2] - bMin[2];
		float diagLen = std::sqrt(dx*dx + dy*dy + dz*dz);
		float d2raw = (cx-px)*(cx-px) + (cy-py)*(cy-py) + (cz-pz)*(cz-pz);
#if defined(__CUDACC__)
		float d2 = fmaxf(d2raw, diagLen * 0.5f);
#else
		float d2 = std::max(d2raw, diagLen * 0.5f);
#endif

		// Direction toward p
		float wix = px - cx, wiy = py - cy, wiz = pz - cz;
		float wiLen = std::sqrt(wix*wix + wiy*wiy + wiz*wiz);
		if (wiLen == 0.f) return 0.f;
		wix /= wiLen; wiy /= wiLen; wiz /= wiLen;

		// cos(theta_w)
		float cosTheta_w = wox*wix + woy*wiy + woz*wiz;
		if (TwoSided()) cosTheta_w = std::abs(cosTheta_w);
		float sinTheta_w = SafeSqrt(1.f - Sqr(cosTheta_w));

		// cos(theta_b) - BoundSubtendedDirections(bMin,bMax,p).cosTheta,
		// inlined directly (same centroid/diagonal already computed above as
		// cx/cy/cz/dx/dy/dz - the box is the SAME box, so this reuses them
		// rather than recomputing under different names the way the separate-
		// function version's own local cx/cy/cz shadowed these). p inside the
		// box's bounding sphere (bd2 < radius^2) is DirectionCone::
		// EntireSphere()'s own cosTheta = -1, matching that factory exactly;
		// otherwise SafeSqrt(1 - sin^2ThetaMax) matches DirectionCone's
		// constructor storing cosThetaMax verbatim (this function only ever
		// reads .cosTheta from the result, never the cone's axis, so the
		// axis itself - cx-px,cy-py,cz-pz - was never needed here).
		float radius = 0.5f * diagLen;
		float cosTheta_b;
		if (d2raw < radius * radius) {
			cosTheta_b = -1.f;
		} else {
			float sin2ThetaMax = (radius * radius) / d2raw;
			cosTheta_b = SafeSqrt(1.f - sin2ThetaMax);
		}
		float sinTheta_b = SafeSqrt(1.f - Sqr(cosTheta_b));

		// cos(theta') - cosSubClamped/sinSubClamped inlined directly at each
		// use site (no lambda - see this function's own header comment).
		float sinTheta_o = SafeSqrt(1.f - Sqr(cosTheta_o));
		float cosTheta_x = (cosTheta_w > cosTheta_o) ? 1.f : (cosTheta_w * cosTheta_o + sinTheta_w * sinTheta_o);
		float sinTheta_x = (cosTheta_w > cosTheta_o) ? 0.f : (sinTheta_w * cosTheta_o - cosTheta_w * sinTheta_o);
		float cosThetap  = (cosTheta_x > cosTheta_b) ? 1.f : (cosTheta_x * cosTheta_b + sinTheta_x * sinTheta_b);
		if (cosThetap <= cosTheta_e) return 0.f;

		float importance = phi * cosThetap / d2;

		// Surface normal weighting
		float nLen2 = nx*nx + ny*ny + nz*nz;
		if (nLen2 > 0.f) {
			float cosTheta_i = std::abs(nx*wix + ny*wiy + nz*wiz);
			float sinTheta_i = SafeSqrt(1.f - Sqr(cosTheta_i));
			float cosThetap_i = (cosTheta_i > cosTheta_b) ? 1.f : (cosTheta_i * cosTheta_b + sinTheta_i * sinTheta_b);
			importance *= cosThetap_i;
		}

#if defined(__CUDACC__)
		return fmaxf(importance, 0.f);
#else
		return std::max(importance, 0.f);
#endif
	}

	// -----------------------------------------------------------------------
	// Storage (mirrors pbrt-v4 private layout exactly)
	// -----------------------------------------------------------------------
	OctahedralVector w;          // 4 bytes
	float phi = 0.f;             // 4 bytes
	// Manually packed (NOT a C++ bitfield) qCosTheta_o (bits 0-14) |
	// qCosTheta_e (bits 15-29) | twoSided (bit 30) - see LightBVHNode's own
	// comment (light_bvh_node.h) for why: this struct is written host-side
	// (BVHLightSampler2::buildBVH(), MSVC) and read device-side
	// (LightBVHNode::lightBounds.Importance(), NVCC) as raw bytes, and a
	// real C++ bitfield's packing order/allocation-unit is compiler/ABI-
	// defined - manual shift/mask has one unambiguous layout on every
	// compiler. Same 4-byte footprint as the bitfield it replaces (unlike
	// LightBVHNode's own fix, which widened to two plain fields since it had
	// alignas(32) slack to spend and only 2 sub-fields to begin with - this
	// one has 3 sub-fields that would double CompactLightBounds's size as
	// plain fields, and this struct is read on every BVH traversal step, not
	// just at a leaf, so keeping it compact matters more here).
	unsigned int packedCosThetaAndSide = 0;
	uint16_t qb[2][3];           // 12 bytes — [corner][axis]

private:
	// Quantise cosine in [-1, 1] to [0, 32767]
	CPU_GPU static unsigned int QuantizeCos(float c) {
		c = Clamp(c, -1.f, 1.f);
		return static_cast<unsigned int>(std::floor(32767.f * ((c + 1.f) / 2.f)));
	}

	// Quantise a coordinate in [min, max] to [0, 65535]
	CPU_GPU static float QuantizeBounds(float c, float minV, float maxV) {
		if (minV == maxV) return 0.f;
		return 65535.f * Clamp((c - minV) / (maxV - minV), 0.f, 1.f);
	}
};

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
