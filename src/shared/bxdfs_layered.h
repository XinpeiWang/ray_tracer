#pragma once
#include "bxdfs_base.h"

// ===========================================================================
// 8. CoatedDiffuseBxDF  (rough dielectric coat over Lambertian base)
//    Mirrors pbrt-v4 CoatedDiffuseBxDF = LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>
//
//    Full random-walk evaluation (pbrt-v4 LayeredBxDF::Sample_f):
//      - Top interface: GGX dielectric (coat_ior)
//      - Bottom interface: Lambertian (albedo)
//      - Optional medium albedo and HG phase (set albedo_medium > 0)
//      - maxDepth bounces; terminates via Russian roulette after depth 3
//
//    sample_local(wi_x,wi_y,wi_z, seed0, seed1):
//      - Returns the sampled exit direction + throughput weight (no pdf division)
//      - seed0/seed1 are 64-bit values from the caller's RNG state
//    All in local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedDiffuseBxDF {
	T albedo_r, albedo_g, albedo_b;   // Lambertian base color
	T coat_ior;                        // dielectric coat IOR (>1, e.g. 1.5)
	T alpha_x;                         // GGX roughness u-direction
	T alpha_y;                         // GGX roughness v-direction (equal = isotropic)
	T thickness = T(0.01);             // layer thickness (pbrt-v4 default 0.01)
	T g         = T(0);               // HG phase function asymmetry (0=none)
	T medium_albedo = T(0);           // scattering albedo of medium (0=vacuum)
	int maxDepth  = 10;               // max random-walk bounces
	int nSamples  = 1;                // samples per call (averaged)

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		uint64_t seed0, uint64_t seed1 = 0) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		layered_detail::PCG32 rng(seed0, seed1 ^ 0xdeadbeefull);

		// pbrt-v4: Sample entrance (top) interface
		T u1 = (T)rng.Uniform<float>(), u2 = (T)rng.Uniform<float>(), uc = (T)rng.Uniform<float>();

		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);
		T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T F_in  = FrDielectric(cos_i, coat_ior);

		if (uc < F_in) {
			// Reflect off the coat top -- specular (same as single-bounce)
			T wo_x = T(2)*cos_i*wm_x - wi_x;
			T wo_y = T(2)*cos_i*wm_y - wi_y;
			T wo_z = T(2)*cos_i*wm_z - wi_z;
			if (wo_z <= T(0)) { res.valid = false; return res; }

			T G1 = dist.G1(wi_x, wi_y, wi_z);
			T G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
			T w  = (G1 > T(1e-8)) ? G / G1 : T(0);
			T fv = F_in * w;

			res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
			res.r = fv; res.g = fv; res.b = fv;
			res.is_specular = true;
			res.valid = true;
			return res;
		}

		// Transmitted into the layer -- start random walk toward bottom
		// w is the current direction (travels downward after transmission)
		T w_x = T(2)*cos_i*wm_x - wi_x;
		T w_y = T(2)*cos_i*wm_y - wi_y;
		T w_z = -(T(2)*cos_i*wm_z - wi_z);   // flip z: now going downward
		if (w_z == T(0)) { res.valid = false; return res; }
		// Ensure traveling downward (-z in layer)
		if (w_z > T(0)) w_z = -w_z;

		// pbrt-v4: beta starts with transmittance through top interface (1-F_in)
		T beta_r = T(1) - F_in, beta_g = T(1) - F_in, beta_b = T(1) - F_in;
		T z = thickness;   // started at top (z=thickness), going to bottom (z=0)

		for (int depth = 0; depth < maxDepth; ++depth) {
			// Russian roulette after depth 3 -- mirrors pbrt-v4: rrBeta = maxBeta < 0.25
			if (depth > 3) {
#if defined(__CUDACC__)
				T rrBeta = fmaxf(beta_r, fmaxf(beta_g, beta_b));
#else
				T rrBeta = std::max(beta_r, std::max(beta_g, beta_b));
#endif
				if (rrBeta < T(0.25)) {
					T q = std::max(T(0), T(1) - rrBeta);
					if ((T)rng.Uniform<float>() < q) { res.valid = false; return res; }
					beta_r /= T(1) - q; beta_g /= T(1) - q; beta_b /= T(1) - q;
				}
			}

			// Medium scattering or direct boundary hit
			if (medium_albedo > T(0)) {
				T dz = layered_detail::SampleExponential((T)rng.Uniform<float>(), T(1) / (w_z < T(0) ? -w_z : w_z));
				T zp = (w_z > T(0)) ? z + dz : z - dz;
				if (zp > T(0) && zp < thickness) {
					// pbrt-v4: beta *= albedo * ps->p / ps->pdf; for HG p/pdf=1
					// sample phase direction
					T cos_theta = layered_detail::hg_sample_cos(g, (T)rng.Uniform<float>());
					beta_r *= medium_albedo;
					beta_g *= medium_albedo;
					beta_b *= medium_albedo;
					// Update direction (simplified: only wz updated, wxy randomised)
					T phi_s = T(6.28318530717958647692) * (T)rng.Uniform<float>();
					T sin_theta = layered_detail::safe_sqrt(T(1) - cos_theta*cos_theta);
#if defined(__CUDACC__)
					w_x = sin_theta * cosf(phi_s);
					w_y = sin_theta * sinf(phi_s);
#else
					w_x = sin_theta * std::cos(phi_s);
					w_y = sin_theta * std::sin(phi_s);
#endif
					w_z = cos_theta;
					z   = zp;
					continue;
				}
#if defined(__CUDACC__)
				z = fmaxf(T(0), fminf(zp, thickness));
#else
				z = std::max(T(0), std::min(zp, thickness));
#endif
			} else {
				// No medium: advance directly to the other interface
				beta_r *= layered_detail::Tr(thickness, w_z);
				beta_g *= layered_detail::Tr(thickness, w_z);
				beta_b *= layered_detail::Tr(thickness, w_z);
				z = (w_z < T(0)) ? T(0) : thickness;
			}

			if (z <= T(0)) {
				// Hit bottom: Lambertian bounce
				layered_detail::cosine_sample((T)rng.Uniform<float>(), (T)rng.Uniform<float>(), w_x, w_y, w_z);
				// w_z is positive (going up toward top)
				beta_r *= albedo_r;
				beta_g *= albedo_g;
				beta_b *= albedo_b;
				z = T(0);
				// Next iteration will hit top interface
			} else {
				// Hit top interface from inside: try to exit
				T wm2_x, wm2_y, wm2_z;
				dist.Sample_wm(w_x, w_y, w_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), wm2_x, wm2_y, wm2_z);
				T cos2 = w_x*wm2_x + w_y*wm2_y + w_z*wm2_z;
				T F_out = FrDielectric(cos2, coat_ior);
				if ((T)rng.Uniform<float>() < F_out) {
					// Internal reflection: continue walk downward
					T rx = T(2)*cos2*wm2_x - w_x;
					T ry = T(2)*cos2*wm2_y - w_y;
					T rz = T(2)*cos2*wm2_z - w_z;
					w_x = rx; w_y = ry; w_z = rz;
					if (w_z > T(0)) w_z = -w_z; // keep traveling downward
					z = thickness;
					continue;
				}
				// Exit through coat top
				T out_x = T(2)*cos2*wm2_x - w_x;
				T out_y = T(2)*cos2*wm2_y - w_y;
				T out_z = T(2)*cos2*wm2_z - w_z;
				if (out_z <= T(0)) { res.valid = false; return res; }

				res.wo_x = out_x; res.wo_y = out_y; res.wo_z = out_z;
				res.r    = beta_r * (T(1) - F_out);
				res.g = beta_g * (T(1) - F_out);
				res.b    = beta_b * (T(1) - F_out);
				res.is_specular = false;
				res.valid = true;
				return res;
			}
		}
		res.valid = false;
		return res;
	}

	// Backward-compat 5-float overload: derives a seed from the 5 input values
	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2, T u3, T u4, T u5) const
	{
		// Combine random inputs into a seed pair for the random walk
		uint32_t s0, s1, s2, s3, s4;
		s0 = (uint32_t)(u1 * 16777216.0f);
		s1 = (uint32_t)(u2 * 16777216.0f);
		s2 = (uint32_t)(u3 * 16777216.0f);
		s3 = (uint32_t)(u4 * 16777216.0f);
		s4 = (uint32_t)(u5 * 16777216.0f);
		uint64_t seed0 = ((uint64_t)s0 << 32) | (uint64_t)s1;
		uint64_t seed1 = ((uint64_t)s2 << 32) | ((uint64_t)s3 * 0x9e3779b97f4a7c15ULL + s4);
		return sample_local(wi_x, wi_y, wi_z, seed0, seed1);
	}

	// f(wi, wo, seed) -- stochastic BSDF value at an arbitrary queried
	// direction wo, for real NEE/MIS (pbrt-v4's LayeredBxDF::f() is
	// genuinely stochastic -- there is no closed form for a random-walk
	// layered BSDF -- see this codebase's TASK #228 comment history).
	//
	// This is a SIMPLIFIED single-connection-per-bounce estimator, not a
	// byte-for-byte port of pbrt-v4's full bidirectional-MIS LayeredBxDF::f()
	// (which additionally connects at the diffuse bottom interface and
	// power-heuristic-weights between phase-function and interface
	// sampling for variance reduction). This version only connects toward
	// wo each time the walk reaches the TOP interface from inside (the
	// only exit point for this reflective BSDF), and advances the
	// remaining walk deterministically (weighted by the interface
	// reflectance, not a stochastic accept/reject) rather than needing a
	// second re-sampled "shadow" path through the exit interface. It is
	// still an unbiased Monte Carlo estimator of the same integral, just
	// higher-variance for deep multi-bounce paths (rare -- Russian
	// roulette after depth 3 already kills most of those) in exchange for
	// being far simpler to derive and verify correctly under this
	// session's time constraints -- verified via a dedicated white-furnace
	// energy-conservation test (tests/unit/bsdf_chi2_tests.cpp) rather
	// than trusted from derivation alone, matching this file's own
	// RoughDielectricBxDF::f() precedent.
	//
	// Every interface crossing here (entrance transmission, internal
	// reflection, exit) mirrors sample_local()'s own model EXACTLY: each
	// is a standard specular reflection about a VNDF-sampled half-vector,
	// with the OUTGOING direction's z-component then negated/mirrored to
	// cross to the other side of the layer -- NOT a true Snell refraction
	// (unlike RoughDielectricBxDF's f(), which models a real single
	// dielectric interface). This is a real, pre-existing simplification
	// already baked into sample_local()'s tested behavior (see this
	// struct's own header comment), not something introduced here -- f()
	// mirrors it for consistency rather than using a physically truer but
	// sample_local()-inconsistent model.
	CPU_GPU void f(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z,
					uint64_t seed0, uint64_t seed1,
					T& fr, T& fg, T& fb) const {
		fr = fg = fb = T(0);
		if (wi_z <= T(0) || wo_z <= T(0)) return;

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);

		// Zero-bounce term: direct reflection off the coat's own top
		// surface (no penetration at all) -- matches sample_local()'s own
		// entrance-reflect branch exactly (standard GGX reflection shape *
		// FrDielectric at the wi/wo half-vector). Deterministic, not part
		// of the stochastic average below.
		{
			T hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
#if defined(__CUDACC__)
			T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
			T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
			if (hlen > T(1e-8)) {
				hx /= hlen; hy /= hlen; hz /= hlen;
				T D = dist.D(hx, hy, hz);
				T G = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
				T cos_wi_h = wi_x*hx + wi_y*hy + wi_z*hz;
				T F0 = FrDielectric(cos_wi_h, coat_ior);
				T val = D * G * F0 / (T(4) * wi_z * wo_z);
				fr = val; fg = val; fb = val;
			}
		}

		layered_detail::PCG32 rng(seed0, seed1 ^ 0x9e3779b9ull);
		int n = nSamples > 0 ? nSamples : 1;
		T accum_r = T(0), accum_g = T(0), accum_b = T(0);

		for (int s = 0; s < n; ++s) {
			T u1 = (T)rng.Uniform<float>(), u2 = (T)rng.Uniform<float>();
			T wm_x, wm_y, wm_z;
			dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);
			T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
			T F_in = FrDielectric(cos_i, coat_ior);

			T w_x = T(2)*cos_i*wm_x - wi_x;
			T w_y = T(2)*cos_i*wm_y - wi_y;
			T w_z = -(T(2)*cos_i*wm_z - wi_z);
			if (w_z == T(0)) continue;
			if (w_z > T(0)) w_z = -w_z;

			T beta_r = T(1) - F_in, beta_g = T(1) - F_in, beta_b = T(1) - F_in;
			T z = thickness;

			for (int depth = 0; depth < maxDepth; ++depth) {
				if (depth > 3) {
#if defined(__CUDACC__)
					T rrBeta = fmaxf(beta_r, fmaxf(beta_g, beta_b));
#else
					T rrBeta = std::max(beta_r, std::max(beta_g, beta_b));
#endif
					if (rrBeta < T(0.25)) {
						T q = std::max(T(0), T(1) - rrBeta);
						if ((T)rng.Uniform<float>() < q) break;
						beta_r /= T(1) - q; beta_g /= T(1) - q; beta_b /= T(1) - q;
					}
				}

				if (medium_albedo > T(0)) {
					T dz = layered_detail::SampleExponential((T)rng.Uniform<float>(), T(1) / (w_z < T(0) ? -w_z : w_z));
					T zp = (w_z > T(0)) ? z + dz : z - dz;
					if (zp > T(0) && zp < thickness) {
						T cos_theta = layered_detail::hg_sample_cos(g, (T)rng.Uniform<float>());
						beta_r *= medium_albedo; beta_g *= medium_albedo; beta_b *= medium_albedo;
						T phi_s = T(6.28318530717958647692) * (T)rng.Uniform<float>();
						T sin_theta = layered_detail::safe_sqrt(T(1) - cos_theta*cos_theta);
#if defined(__CUDACC__)
						w_x = sin_theta * cosf(phi_s);
						w_y = sin_theta * sinf(phi_s);
#else
						w_x = sin_theta * std::cos(phi_s);
						w_y = sin_theta * std::sin(phi_s);
#endif
						w_z = cos_theta;
						z   = zp;
						continue;
					}
#if defined(__CUDACC__)
					z = fmaxf(T(0), fminf(zp, thickness));
#else
					z = std::max(T(0), std::min(zp, thickness));
#endif
				} else {
					beta_r *= layered_detail::Tr(thickness, w_z);
					beta_g *= layered_detail::Tr(thickness, w_z);
					beta_b *= layered_detail::Tr(thickness, w_z);
					z = (w_z < T(0)) ? T(0) : thickness;
				}

				if (z <= T(0)) {
					// Lambertian bottom bounce (no connection here -- see
					// this method's own header comment on the simplified,
					// top-exit-only connection strategy).
					layered_detail::cosine_sample((T)rng.Uniform<float>(), (T)rng.Uniform<float>(), w_x, w_y, w_z);
					beta_r *= albedo_r; beta_g *= albedo_g; beta_b *= albedo_b;
					z = T(0);
				} else {
					// At top interface from inside: connect toward wo using
					// a plain reflection half-vector, matching
					// sample_local()'s own exit formula (out=reflect(w,wm2))
					// exactly. w arrives here with w_z > 0 (upward) --
					// every path into this branch comes right after a
					// bottom bounce (cosine_sample()'s own comment: "w_z is
					// positive, going up toward top") or an internal
					// reflection whose subsequent advance step sends it
					// back to z=0 first -- so w is ALREADY on the same side
					// as wo (wo_z > 0 too), needing no mirroring (an
					// earlier version of this code incorrectly assumed
					// w_z < 0 here, which only holds for the very first
					// step right after entrance, a case this loop structure
					// never actually reaches this branch on -- caught by
					// this test file's own energy-conservation check
					// returning ~0 instead of a plausible reflectance).
					T hx = w_x + wo_x, hy = w_y + wo_y, hz = w_z + wo_z;
#if defined(__CUDACC__)
					T hlen2 = sqrtf(hx*hx + hy*hy + hz*hz);
#else
					T hlen2 = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
					if (hlen2 > T(1e-8)) {
						hx /= hlen2; hy /= hlen2; hz /= hlen2;
						T D = dist.D(hx, hy, hz);
						T G = dist.G(wo_x, wo_y, wo_z, w_x, w_y, w_z);
						T cos_w_h = w_x*hx + w_y*hy + w_z*hz;
						T F_exit = FrDielectric(cos_w_h, coat_ior);
						T shape = D * G / (T(4) * w_z * wo_z);
						T connect_val = shape * (T(1) - F_exit);
						accum_r += beta_r * connect_val * wo_z;
						accum_g += beta_g * connect_val * wo_z;
						accum_b += beta_b * connect_val * wo_z;
					}

					// Continue the walk: deterministic internal reflection,
					// weighted by the ACTUAL reflectance F_out (the
					// transmission possibility was already accounted for
					// by the connection above, so no stochastic branch
					// here -- see this method's own header comment).
					T wm2_x, wm2_y, wm2_z;
					dist.Sample_wm(w_x, w_y, w_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), wm2_x, wm2_y, wm2_z);
					T cos2 = w_x*wm2_x + w_y*wm2_y + w_z*wm2_z;
					T F_out = FrDielectric(cos2, coat_ior);
					T rx = T(2)*cos2*wm2_x - w_x;
					T ry = T(2)*cos2*wm2_y - w_y;
					T rz = T(2)*cos2*wm2_z - w_z;
					w_x = rx; w_y = ry; w_z = rz;
					if (w_z > T(0)) w_z = -w_z;
					beta_r *= F_out; beta_g *= F_out; beta_b *= F_out;
					z = thickness;
				}
			}
		}

		fr += accum_r / (T)n;
		fg += accum_g / (T)n;
		fb += accum_b / (T)n;
	}
};

// ===========================================================================
// 9. CoatedConductorBxDF  (rough dielectric coat over GGX conductor)
//    Mirrors pbrt-v4 CoatedConductorBxDF = LayeredBxDF<DielectricBxDF, ConductorBxDF, true>
//
//    Full random-walk evaluation (pbrt-v4 LayeredBxDF::Sample_f):
//      - Top interface: GGX dielectric (coat_ior)
//      - Bottom interface: GGX conductor (complex Fresnel per RGB channel)
//      - Optional medium scattering (medium_albedo + HG phase g)
//      - maxDepth bounces; Russian roulette after depth 3
//
//    sample_local(wi_x,wi_y,wi_z, seed0, seed1):
//      - Returns sampled exit direction + throughput (no pdf division)
//    All in local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedConductorBxDF {
	T eta_r, eta_g, eta_b;   // conductor real IOR
	T k_r,   k_g,   k_b;    // conductor extinction
	T coat_ior;              // dielectric coat IOR
	T alpha_x;               // GGX roughness u-direction
	T alpha_y;               // GGX roughness v-direction (equal = isotropic)
	T thickness = T(0.01);
	T g         = T(0);
	T medium_albedo = T(0);
	int maxDepth  = 10;
	int nSamples  = 1;

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		uint64_t seed0, uint64_t seed1 = 0) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		layered_detail::PCG32 rng(seed0, seed1 ^ 0xdeadbeefull);

		T u1 = (T)rng.Uniform<float>(), u2 = (T)rng.Uniform<float>(), uc = (T)rng.Uniform<float>();

		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);
		T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T F_in  = FrDielectric(cos_i, coat_ior);

		if (uc < F_in) {
			// Reflect off the coat top (achromatic)
			T wo_x = T(2)*cos_i*wm_x - wi_x;
			T wo_y = T(2)*cos_i*wm_y - wi_y;
			T wo_z = T(2)*cos_i*wm_z - wi_z;
			if (wo_z <= T(0)) { res.valid = false; return res; }

			T G1 = dist.G1(wi_x, wi_y, wi_z);
			T G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
			T w  = (G1 > T(1e-8)) ? G / G1 : T(0);
			T fv = F_in * w;

			res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
			res.r = fv; res.g = fv; res.b = fv;
			res.is_specular = true;
			res.valid = true;
			return res;
		}

		// Transmitted into the layer: random walk
		T w_x = T(2)*cos_i*wm_x - wi_x;
		T w_y = T(2)*cos_i*wm_y - wi_y;
		T w_z = -(T(2)*cos_i*wm_z - wi_z);
		if (w_z == T(0)) { res.valid = false; return res; }
		if (w_z > T(0)) w_z = -w_z;

		// pbrt-v4: beta starts with transmittance through top interface (1-F_in)
		T beta_r = T(1) - F_in;
		T beta_g = T(1) - F_in;
		T beta_b = T(1) - F_in;
		T z = thickness;

		for (int depth = 0; depth < maxDepth; ++depth) {
			// Russian roulette after depth 3 -- mirrors pbrt-v4: rrBeta = maxBeta < 0.25
			if (depth > 3) {
#if defined(__CUDACC__)
				T rrBeta = fmaxf(beta_r, fmaxf(beta_g, beta_b));
#else
				T rrBeta = std::max(beta_r, std::max(beta_g, beta_b));
#endif
				if (rrBeta < T(0.25)) {
					T q = std::max(T(0), T(1) - rrBeta);
					if ((T)rng.Uniform<float>() < q) { res.valid = false; return res; }
					beta_r /= T(1) - q; beta_g /= T(1) - q; beta_b /= T(1) - q;
				}
			}

			if (medium_albedo > T(0)) {
				T dz = layered_detail::SampleExponential((T)rng.Uniform<float>(), T(1) / (w_z < T(0) ? -w_z : w_z));
				T zp = (w_z > T(0)) ? z + dz : z - dz;
				if (zp > T(0) && zp < thickness) {
					// pbrt-v4: beta *= albedo * ps->p / ps->pdf; for HG p/pdf=1
					T cos_theta = layered_detail::hg_sample_cos(g, (T)rng.Uniform<float>());
					beta_r *= medium_albedo; beta_g *= medium_albedo; beta_b *= medium_albedo;
					T phi_s = T(6.28318530717958647692) * (T)rng.Uniform<float>();
					T sin_theta = layered_detail::safe_sqrt(T(1) - cos_theta*cos_theta);
#if defined(__CUDACC__)
					w_x = sin_theta * cosf(phi_s);
					w_y = sin_theta * sinf(phi_s);
#else
					w_x = sin_theta * std::cos(phi_s);
					w_y = sin_theta * std::sin(phi_s);
#endif
					w_z = cos_theta;
					z   = zp;
					continue;
				}
#if defined(__CUDACC__)
				z = fmaxf(T(0), fminf(zp, thickness));
#else
				z = std::max(T(0), std::min(zp, thickness));
#endif
			} else {
				beta_r *= layered_detail::Tr(thickness, w_z);
				beta_g *= layered_detail::Tr(thickness, w_z);
				beta_b *= layered_detail::Tr(thickness, w_z);
				z = (w_z < T(0)) ? T(0) : thickness;
			}

			if (z <= T(0)) {
				// Hit conductor bottom: GGX VNDF + complex Fresnel
				// Flip to local frame of bottom interface (z points upward into layer)
				T fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;  // now fw_z > 0 (incoming from top)
				T bwm_x, bwm_y, bwm_z;
				dist.Sample_wm(fw_x, fw_y, fw_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), bwm_x, bwm_y, bwm_z);
				T cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
				T rwo_x = T(2)*cos_c*bwm_x - fw_x;
				T rwo_y = T(2)*cos_c*bwm_y - fw_y;
				T rwo_z = T(2)*cos_c*bwm_z - fw_z;

				T G1_c = dist.G1(fw_x, fw_y, fw_z);
				T G_c  = dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
				T wt_c = (G1_c > T(1e-8)) ? G_c / G1_c : T(0);

				beta_r *= FrComplex(cos_c, eta_r, k_r) * wt_c;
				beta_g *= FrComplex(cos_c, eta_g, k_g) * wt_c;
				beta_b *= FrComplex(cos_c, eta_b, k_b) * wt_c;

				// Reflected direction goes upward (+z in layer frame -> -bottom frame -> +layer)
				w_x = rwo_x; w_y = rwo_y;
				w_z = (rwo_z < T(0)) ? -rwo_z : rwo_z; // going upward toward top
				z = T(0);
			} else {
				// Hit top interface from inside: attempt exit
				T wm2_x, wm2_y, wm2_z;
				dist.Sample_wm(w_x, w_y, w_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), wm2_x, wm2_y, wm2_z);
				T cos2  = w_x*wm2_x + w_y*wm2_y + w_z*wm2_z;
				T F_out = FrDielectric(cos2, coat_ior);
				if ((T)rng.Uniform<float>() < F_out) {
					// Internal reflection
					T rx = T(2)*cos2*wm2_x - w_x;
					T ry = T(2)*cos2*wm2_y - w_y;
					T rz = T(2)*cos2*wm2_z - w_z;
					w_x = rx; w_y = ry; w_z = rz;
					if (w_z > T(0)) w_z = -w_z;
					z = thickness;
					continue;
				}
				// Exit through coat
				T out_x = T(2)*cos2*wm2_x - w_x;
				T out_y = T(2)*cos2*wm2_y - w_y;
				T out_z = T(2)*cos2*wm2_z - w_z;
				if (out_z <= T(0)) { res.valid = false; return res; }

				res.wo_x = out_x; res.wo_y = out_y; res.wo_z = out_z;
				res.r    = beta_r * (T(1) - F_out);
				res.g = beta_g * (T(1) - F_out);
				res.b    = beta_b * (T(1) - F_out);
				res.is_specular = false;
				res.valid = true;
				return res;
			}
		}
		res.valid = false;
		return res;
	}

	// Backward-compat 5-float overload
	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2, T u3, T u4, T u5) const
	{
		uint32_t s0 = (uint32_t)(u1 * 16777216.0f);
		uint32_t s1 = (uint32_t)(u2 * 16777216.0f);
		uint32_t s2 = (uint32_t)(u3 * 16777216.0f);
		uint32_t s3 = (uint32_t)(u4 * 16777216.0f);
		uint32_t s4 = (uint32_t)(u5 * 16777216.0f);
		uint64_t seed0 = ((uint64_t)s0 << 32) | (uint64_t)s1;
		uint64_t seed1 = ((uint64_t)s2 << 32) | ((uint64_t)s3 * 0x9e3779b97f4a7c15ULL + s4);
		return sample_local(wi_x, wi_y, wi_z, seed0, seed1);
	}

	// f(wi, wo, seed) -- stochastic BSDF value for real NEE/MIS. Identical
	// structure and simplifications to CoatedDiffuseBxDF::f() (see that
	// method's own header comment for the full rationale) -- only the
	// bottom-interface bounce differs (GGX conductor reflection, matching
	// this struct's own sample_local() bottom-bounce code, instead of a
	// Lambertian cosine bounce).
	CPU_GPU void f(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z,
					uint64_t seed0, uint64_t seed1,
					T& fr, T& fg, T& fb) const {
		fr = fg = fb = T(0);
		if (wi_z <= T(0) || wo_z <= T(0)) return;

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);

		// Zero-bounce term: direct reflection off the coat's own top
		// surface -- see CoatedDiffuseBxDF::f()'s identical term.
		{
			T hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
#if defined(__CUDACC__)
			T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
			T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
			if (hlen > T(1e-8)) {
				hx /= hlen; hy /= hlen; hz /= hlen;
				T D = dist.D(hx, hy, hz);
				T G = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
				T cos_wi_h = wi_x*hx + wi_y*hy + wi_z*hz;
				T F0 = FrDielectric(cos_wi_h, coat_ior);
				T val = D * G * F0 / (T(4) * wi_z * wo_z);
				fr = val; fg = val; fb = val;
			}
		}

		layered_detail::PCG32 rng(seed0, seed1 ^ 0x9e3779b9ull);
		int n = nSamples > 0 ? nSamples : 1;
		T accum_r = T(0), accum_g = T(0), accum_b = T(0);

		for (int s = 0; s < n; ++s) {
			T u1 = (T)rng.Uniform<float>(), u2 = (T)rng.Uniform<float>();
			T wm_x, wm_y, wm_z;
			dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);
			T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
			T F_in = FrDielectric(cos_i, coat_ior);

			T w_x = T(2)*cos_i*wm_x - wi_x;
			T w_y = T(2)*cos_i*wm_y - wi_y;
			T w_z = -(T(2)*cos_i*wm_z - wi_z);
			if (w_z == T(0)) continue;
			if (w_z > T(0)) w_z = -w_z;

			T beta_r = T(1) - F_in, beta_g = T(1) - F_in, beta_b = T(1) - F_in;
			T z = thickness;

			for (int depth = 0; depth < maxDepth; ++depth) {
				if (depth > 3) {
#if defined(__CUDACC__)
					T rrBeta = fmaxf(beta_r, fmaxf(beta_g, beta_b));
#else
					T rrBeta = std::max(beta_r, std::max(beta_g, beta_b));
#endif
					if (rrBeta < T(0.25)) {
						T q = std::max(T(0), T(1) - rrBeta);
						if ((T)rng.Uniform<float>() < q) break;
						beta_r /= T(1) - q; beta_g /= T(1) - q; beta_b /= T(1) - q;
					}
				}

				if (medium_albedo > T(0)) {
					T dz = layered_detail::SampleExponential((T)rng.Uniform<float>(), T(1) / (w_z < T(0) ? -w_z : w_z));
					T zp = (w_z > T(0)) ? z + dz : z - dz;
					if (zp > T(0) && zp < thickness) {
						T cos_theta = layered_detail::hg_sample_cos(g, (T)rng.Uniform<float>());
						beta_r *= medium_albedo; beta_g *= medium_albedo; beta_b *= medium_albedo;
						T phi_s = T(6.28318530717958647692) * (T)rng.Uniform<float>();
						T sin_theta = layered_detail::safe_sqrt(T(1) - cos_theta*cos_theta);
#if defined(__CUDACC__)
						w_x = sin_theta * cosf(phi_s);
						w_y = sin_theta * sinf(phi_s);
#else
						w_x = sin_theta * std::cos(phi_s);
						w_y = sin_theta * std::sin(phi_s);
#endif
						w_z = cos_theta;
						z   = zp;
						continue;
					}
#if defined(__CUDACC__)
					z = fmaxf(T(0), fminf(zp, thickness));
#else
					z = std::max(T(0), std::min(zp, thickness));
#endif
				} else {
					beta_r *= layered_detail::Tr(thickness, w_z);
					beta_g *= layered_detail::Tr(thickness, w_z);
					beta_b *= layered_detail::Tr(thickness, w_z);
					z = (w_z < T(0)) ? T(0) : thickness;
				}

				if (z <= T(0)) {
					// Hit conductor bottom: GGX VNDF + complex Fresnel,
					// matching sample_local()'s own bottom-bounce code
					// exactly (flip to the bottom interface's own local
					// frame, where the incoming direction has fw_z > 0).
					T fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;
					T bwm_x, bwm_y, bwm_z;
					dist.Sample_wm(fw_x, fw_y, fw_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), bwm_x, bwm_y, bwm_z);
					T cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
					T rwo_x = T(2)*cos_c*bwm_x - fw_x;
					T rwo_y = T(2)*cos_c*bwm_y - fw_y;
					T rwo_z = T(2)*cos_c*bwm_z - fw_z;

					T G1_c = dist.G1(fw_x, fw_y, fw_z);
					T G_c  = dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
					T wt_c = (G1_c > T(1e-8)) ? G_c / G1_c : T(0);

					beta_r *= FrComplex(cos_c, eta_r, k_r) * wt_c;
					beta_g *= FrComplex(cos_c, eta_g, k_g) * wt_c;
					beta_b *= FrComplex(cos_c, eta_b, k_b) * wt_c;

					w_x = rwo_x; w_y = rwo_y;
					w_z = (rwo_z < T(0)) ? -rwo_z : rwo_z;
					z = T(0);
				} else {
					// At top interface from inside: connect toward wo --
					// see CoatedDiffuseBxDF::f()'s identical connection
					// step for the full derivation/rationale (w arrives
					// here with w_z > 0, needing no mirroring).
					T hx = w_x + wo_x, hy = w_y + wo_y, hz = w_z + wo_z;
#if defined(__CUDACC__)
					T hlen2 = sqrtf(hx*hx + hy*hy + hz*hz);
#else
					T hlen2 = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
					if (hlen2 > T(1e-8)) {
						hx /= hlen2; hy /= hlen2; hz /= hlen2;
						T D = dist.D(hx, hy, hz);
						T G = dist.G(wo_x, wo_y, wo_z, w_x, w_y, w_z);
						T cos_w_h = w_x*hx + w_y*hy + w_z*hz;
						T F_exit = FrDielectric(cos_w_h, coat_ior);
						T shape = D * G / (T(4) * w_z * wo_z);
						T connect_val = shape * (T(1) - F_exit);
						accum_r += beta_r * connect_val * wo_z;
						accum_g += beta_g * connect_val * wo_z;
						accum_b += beta_b * connect_val * wo_z;
					}

					T wm2_x, wm2_y, wm2_z;
					dist.Sample_wm(w_x, w_y, w_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), wm2_x, wm2_y, wm2_z);
					T cos2 = w_x*wm2_x + w_y*wm2_y + w_z*wm2_z;
					T F_out = FrDielectric(cos2, coat_ior);
					T rx = T(2)*cos2*wm2_x - w_x;
					T ry = T(2)*cos2*wm2_y - w_y;
					T rz = T(2)*cos2*wm2_z - w_z;
					w_x = rx; w_y = ry; w_z = rz;
					if (w_z > T(0)) w_z = -w_z;
					beta_r *= F_out; beta_g *= F_out; beta_b *= F_out;
					z = thickness;
				}
			}
		}

		fr += accum_r / (T)n;
		fg += accum_g / (T)n;
		fb += accum_b / (T)n;
	}
};

// ===========================================================================
// 10. DiffuseTransmissionBxDF  (cosine-weighted diffuse reflect + transmit)
//     Mirrors pbrt-v4 DiffuseTransmissionBxDF
//
//     Stochastically chooses:
//       reflect   (u1 < pr/(pr+pt)) -> cosine-weighted wo in upper hemisphere
//       transmit  (u1 >= pr/(pr+pt)) -> cosine-weighted wo in lower hemisphere
//
//     u1: reflect/transmit decision
//     u2, u3: cosine-hemisphere sample
//     pr, pt: max-component of R and T (caller provides)
//     All in world space; caller provides the surface normal.
// ===========================================================================
template<typename T>
struct DiffuseTransmissionBxDF {
	T R_r, R_g, R_b;  // reflectance color
	T T_r, T_g, T_b;  // transmittance color

	// nx,ny,nz: outward surface normal (world space)
	// u1: selection; u2,u3: hemisphere sample
	CPU_GPU BxDFSampleResult<T> sample(
		T nx, T ny, T nz,
		T u1, T u2, T u3) const
	{
		BxDFSampleResult<T> res{};
#if defined(__CUDACC__)
		T pr = fmaxf(R_r, fmaxf(R_g, R_b));
		T pt = fmaxf(T_r, fmaxf(T_g, T_b));
#else
		T pr = std::fmax(R_r, std::fmax(R_g, R_b));
		T pt = std::fmax(T_r, std::fmax(T_g, T_b));
#endif
		if (pr + pt <= T(0)) { res.valid = false; return res; }

		T tx, ty, tz, bx, by, bz;

		bool reflect = (u1 < pr / (pr + pt));
		if (reflect) {
			make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);
			cosine_hemisphere_sample(nx,ny,nz, tx,ty,tz, bx,by,bz, u2, u3,
									 res.wo_x, res.wo_y, res.wo_z);
			res.r = R_r; res.g = R_g; res.b = R_b;
		} else {
			// Transmit: cosine-weighted in -normal hemisphere
			make_onb(-nx, -ny, -nz, tx, ty, tz, bx, by, bz);
			cosine_hemisphere_sample(-nx,-ny,-nz, tx,ty,tz, bx,by,bz, u2, u3,
									 res.wo_x, res.wo_y, res.wo_z);
			res.r = T_r; res.g = T_g; res.b = T_b;
		}

		res.is_specular = false;
		res.valid = true;
		return res;
	}

	CPU_GPU T scattering_pdf(T nx, T ny, T nz,
							  T wox, T woy, T woz) const
	{
#if defined(__CUDACC__)
		T pr = fmaxf(R_r, fmaxf(R_g, R_b));
		T pt = fmaxf(T_r, fmaxf(T_g, T_b));
#else
		T pr = std::fmax(R_r, std::fmax(R_g, R_b));
		T pt = std::fmax(T_r, std::fmax(T_g, T_b));
#endif
		if (pr + pt <= T(0)) return T(0);
		T cos_theta = wox*nx + woy*ny + woz*nz;
		T inv_pi = T(1) / T(3.14159265358979323846);
		if (cos_theta > T(0))
			return (pr / (pr + pt)) * (cos_theta * inv_pi);   // reflection
		else
			return (pt / (pr + pt)) * (-cos_theta * inv_pi);  // transmission
	}
};

// ===========================================================================
// 11. NormalizedFresnelBxDF  (Fresnel-weighted diffuse reflection)
//     Mirrors pbrt-v4 NormalizedFresnelBxDF
//
//     f(wi) = (1 - FrDielectric(cos_wi, eta)) / (c * pi)
//     c     = 1 - 2 * FresnelMoment1(1/eta)
//     PDF   = cos_wi / pi  (cosine-weighted sampling)
//     Weight (sample / pdf) = (1 - Fr(cos_wi)) / c
//
//     u1, u2: cosine-hemisphere sample
// ===========================================================================
template<typename T>
struct NormalizedFresnelBxDF {
	T eta;  // surface IOR
	T c;    // 1 - 2 * FresnelMoment1(1/eta)  -- precomputed by caller

	CPU_GPU BxDFSampleResult<T> sample(
		T nx, T ny, T nz,
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		T tx, ty, tz, bx, by, bz;
		make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);
		cosine_hemisphere_sample(nx,ny,nz, tx,ty,tz, bx,by,bz, u1, u2,
								 res.wo_x, res.wo_y, res.wo_z);
		// attenuation = white; weight encoded in scattering_pdf via MIS path
		res.r = T(1); res.g = T(1); res.b = T(1);
		res.is_specular = false;
		res.valid = true;
		return res;
	}

	// scattering_pdf = BSDF * cos = (1 - Fr) * cos / (c * pi)
	CPU_GPU T scattering_pdf(T nx, T ny, T nz,
							  T wox, T woy, T woz) const
	{
		T cos_wi = wox*nx + woy*ny + woz*nz;
		if (cos_wi <= T(0)) return T(0);
		T fr = FrDielectric(cos_wi, eta);
		T cv = (c > T(1e-6)) ? c : T(1e-6);
		return (T(1) - fr) * cos_wi / (cv * T(3.14159265358979323846));
	}
};

