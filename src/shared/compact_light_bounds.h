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

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU inline
#  endif
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4293 4244)
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
		qCosTheta_o = QuantizeCos(lb.cosTheta_o);
		qCosTheta_e = QuantizeCos(lb.cosTheta_e);
		twoSided    = lb.twoSided ? 1u : 0u;

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
	CPU_GPU bool  TwoSided()   const { return twoSided != 0; }
	CPU_GPU float CosTheta_o() const { return 2.f * (qCosTheta_o / 32767.f) - 1.f; }
	CPU_GPU float CosTheta_e() const { return 2.f * (qCosTheta_e / 32767.f) - 1.f; }

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

		// Helpers identical to pbrt-v4
		auto cosSubClamped = [](float sinA, float cosA, float sinB, float cosB) -> float {
			if (cosA > cosB) return 1.f;
			return cosA * cosB + sinA * sinB;
		};
		auto sinSubClamped = [](float sinA, float cosA, float sinB, float cosB) -> float {
			if (cosA > cosB) return 0.f;
			return sinA * cosB - cosA * sinB;
		};

		// Centroid and clamped d2
		float cx = (bMin[0] + bMax[0]) * 0.5f;
		float cy = (bMin[1] + bMax[1]) * 0.5f;
		float cz = (bMin[2] + bMax[2]) * 0.5f;
		float dx = bMax[0] - bMin[0], dy = bMax[1] - bMin[1], dz = bMax[2] - bMin[2];
		float diagLen = std::sqrt(dx*dx + dy*dy + dz*dz);
		float d2raw = (cx-px)*(cx-px) + (cy-py)*(cy-py) + (cz-pz)*(cz-pz);
		float d2 = std::max(d2raw, diagLen * 0.5f);

		// Direction toward p
		float wix = px - cx, wiy = py - cy, wiz = pz - cz;
		float wiLen = std::sqrt(wix*wix + wiy*wiy + wiz*wiz);
		if (wiLen == 0.f) return 0.f;
		wix /= wiLen; wiy /= wiLen; wiz /= wiLen;

		// cos(theta_w)
		float cosTheta_w = wox*wix + woy*wiy + woz*wiz;
		if (twoSided) cosTheta_w = std::abs(cosTheta_w);
		float sinTheta_w = SafeSqrt(1.f - Sqr(cosTheta_w));

		// cos(theta_b)
		float cosTheta_b = BoundSubtendedDirections(
			bMin[0], bMin[1], bMin[2], bMax[0], bMax[1], bMax[2], px, py, pz).cosTheta;
		float sinTheta_b = SafeSqrt(1.f - Sqr(cosTheta_b));

		// cos(theta')
		float sinTheta_o = SafeSqrt(1.f - Sqr(cosTheta_o));
		float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
		float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
		float cosThetap  = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
		if (cosThetap <= cosTheta_e) return 0.f;

		float importance = phi * cosThetap / d2;

		// Surface normal weighting
		float nLen2 = nx*nx + ny*ny + nz*nz;
		if (nLen2 > 0.f) {
			float cosTheta_i = std::abs(nx*wix + ny*wiy + nz*wiz);
			float sinTheta_i = SafeSqrt(1.f - Sqr(cosTheta_i));
			float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
			importance *= cosThetap_i;
		}

		return std::max(importance, 0.f);
	}

	// -----------------------------------------------------------------------
	// Storage (mirrors pbrt-v4 private layout exactly)
	// -----------------------------------------------------------------------
	OctahedralVector w;          // 4 bytes
	float phi = 0.f;             // 4 bytes
	// Anonymous struct ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â matches pbrt-v4 CompactLightBounds private members
	struct {
		unsigned int qCosTheta_o : 15;
		unsigned int qCosTheta_e : 15;
		unsigned int twoSided    :  1;
	};
	uint16_t qb[2][3];           // 12 bytes ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â [corner][axis]

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
