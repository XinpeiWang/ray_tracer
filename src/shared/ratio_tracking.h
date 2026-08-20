#pragma once
// ---------------------------------------------------------------------------
// ratio_tracking.h -- Null-scattering / ratio-tracking volume integration
//
// Mirrors pbrt-v4 SampleT_maj (media.h) and the delta-tracking loops inside
// VolPathIntegrator::Li and VolPathIntegrator::SampleLd (cpu/integrators.cpp).
//
// What this provides
// ==================
//   1. RayMajorantSegment<T>          -- [tMin, tMax] interval with sigma_maj
//   2. HomogeneousMajorantIterator<T> -- single-segment iterator for a
//                                        HomogeneousMediumData (sigma_t = sigma_maj)
//   3. MediumSample<T>                -- result of sampling one medium interaction
//   4. SampleMediumInteraction<T>(...)-- delta-tracking free-path sampler
//                                        (returns absorption, scatter, or null event)
//   5. RatioTrackingTr<T>(...)        -- unbiased ratio-tracking transmittance
//                                        estimator (for shadow/visibility queries)
//
// pbrt-v4 alignment
// =================
//   - SampleMediumInteraction implements SampleT_maj + the delta-tracking
//     callback used in VolPathIntegrator (media.h lines 735-800,
//     integrators.cpp lines 730-800).
//   - RatioTrackingTr implements the ratio-tracking path in
//     VolPathIntegrator::SampleLd (integrators.cpp lines 330-365).
//   - SampleExponential: -log(1-u)/a  (pbrt-v4 util/sampling.h SampleExponential)
//   - Event probabilities: pAbsorb = sigma_a/sigma_maj, pScatter = sigma_s/sigma_maj,
//     pNull = max(0, 1 - pAbsorb - pScatter)
//
// Design rules
// ============
//   - Header-only, CPU_GPU tagged, templated on scalar type T
//   - No virtual functions, no heap allocation
//   - Depends on: volume_scattering.h, rng.h, scalar_math.h
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include <cmath>
#include <algorithm>
#include <optional>
#include "volume_scattering.h"  // HomogeneousMediumData, HenyeyGreensteinPhaseFunction
#include "rng.h"                // RNG::Uniform<float>()
#include "scalar_math.h"        // SafeSqrt, Sqr, etc.
#include "grid_medium.h"        // RayMajorantSegment<T> (canonical definition)

// ===========================================================================
// 1. RayMajorantSegment -- defined in grid_medium.h
// ===========================================================================

// ===========================================================================
// 2. HomogeneousMajorantIterator
template <typename T>
struct HomogeneousMajorantIterator {
	RayMajorantSegment<T> seg;
	bool called = false;

	CPU_GPU HomogeneousMajorantIterator() : called(true) {}

	CPU_GPU HomogeneousMajorantIterator(T tMin, T tMax,
										const HomogeneousMediumData<T>& medium)
	: called(false) {
		seg.tMin       = tMin;
		seg.tMax       = tMax;
		seg.sigma_maj_r = medium.sigma_ar + medium.sigma_sr;
		seg.sigma_maj_g = medium.sigma_ag + medium.sigma_sg;
		seg.sigma_maj_b = medium.sigma_ab + medium.sigma_sb;
		seg.sigma_maj   = seg.sigma_maj_r;  // use R-channel as representative
	}

	CPU_GPU std::optional<RayMajorantSegment<T>> Next() {
		if (called) return std::nullopt;
		called = true;
		return seg;
	}
};

// ===========================================================================
// Helper: SampleExponential
// Returns the exponentially distributed sample -ln(1-u)/a.
// Mirrors pbrt-v4 SampleExponential (util/sampling.h).
// ===========================================================================
template <typename T>
CPU_GPU T SampleExponential(T u, T a) {
	return -std::log(T(1) - u) / a;
}

// ===========================================================================
// 3. MediumEventType -- outcome of one delta-tracking step
//    Mirrors the three-way split in VolPathIntegrator's callback:
//      kAbsorption -- path terminated (absorbed)
//      kScatter    -- real scatter event; update direction and continue
//      kNull       -- null scatter; continue without changing direction
// ===========================================================================
enum class MediumEventType { kAbsorption, kScatter, kNull };

// ===========================================================================
// 4. MediumSample
//    The result returned by SampleMediumInteraction.
//    - event_type:  which event occurred
//    - t:           parametric distance to the event along the ray
//    - sigma_s_{r,g,b}: scattering coefficient at the sample point (real scatter only)
//    - phase_g:     HG anisotropy at the sample point
//    - T_maj_{r,g,b}: accumulated majorant transmittance weight up to t
//      (needed for MIS-weighted unbiased estimators in VolPathIntegrator::SampleLd)
// ===========================================================================
template <typename T>
struct MediumSample {
	MediumEventType event_type = MediumEventType::kNull;
	T t        = T(0);
	T sigma_s_r = T(0), sigma_s_g = T(0), sigma_s_b = T(0);
	T sigma_a_r = T(0), sigma_a_g = T(0), sigma_a_b = T(0);
	T phase_g  = T(0);
	T T_maj_r  = T(1), T_maj_g = T(1), T_maj_b = T(1);
};

// ===========================================================================
// 5. SampleMediumInteraction
//
//    Delta-tracking free-path sampler through a HomogeneousMedium.
//    Mirrors SampleT_maj + the VolPathIntegrator callback (integrators.cpp
//    lines 730-800).
//
//    Parameters
//    ----------
//    medium    : HomogeneousMediumData<T>  (constant coefficients)
//    tMin, tMax: parametric ray interval
//    rng       : RNG&  (consumed in-place for reproducibility)
//
//    Returns
//    -------
//    MediumSample<T> with event_type set to:
//      kAbsorption -- absorbed at t; caller should terminate the path
//      kScatter    -- real scatter at t; caller should sample phase function
//      kNull       -- no real event in [tMin, tMax]; caller continues path
//
//    The T_maj fields carry the "unbiased weight" up to the sampled t so
//    that SampleLd-style MIS estimators can use them directly.
//
//    pbrt-v4 reference: SampleT_maj<HomogeneousMedium> + VolPathIntegrator
//    callback (absorption / scatter / null branches).
// ===========================================================================
template <typename T>
CPU_GPU MediumSample<T> SampleMediumInteraction(
	const HomogeneousMediumData<T>& medium,
	T tMin, T tMax,
	RNG& rng)
{
	MediumSample<T> result;

	// Build the single majorant segment for a homogeneous medium
	T sigma_maj_r = medium.sigma_ar + medium.sigma_sr;
	T sigma_maj_g = medium.sigma_ag + medium.sigma_sg;
	T sigma_maj_b = medium.sigma_ab + medium.sigma_sb;

	// Use the R-channel majorant for exponential free-path sampling (pbrt-v4
	// uses sigma_maj[0], which corresponds to the first spectral channel).
	T sigma_maj = sigma_maj_r;
	if (sigma_maj <= T(0)) {
		// Vacuum / zero-density medium: no interaction possible
		result.event_type = MediumEventType::kNull;
		return result;
	}

	// Accumulated majorant transmittance (reset to 1 per segment in pbrt-v4)
	T T_maj_r = T(1), T_maj_g = T(1), T_maj_b = T(1);

	T t_cur = tMin;
	while (true) {
		// Sample free path in [t_cur, tMax] under the majorant
		T u = rng.Uniform<float>();
		T dt = SampleExponential(u, sigma_maj);
		T t  = t_cur + dt;

		if (t >= tMax) {
			// No event in remaining interval: apply full transmittance and exit
			T remaining = tMax - t_cur;
			T_maj_r *= std::exp(-sigma_maj_r * remaining);
			T_maj_g *= std::exp(-sigma_maj_g * remaining);
			T_maj_b *= std::exp(-sigma_maj_b * remaining);
			result.event_type = MediumEventType::kNull;
			result.T_maj_r = T_maj_r;
			result.T_maj_g = T_maj_g;
			result.T_maj_b = T_maj_b;
			return result;
		}

		// Accumulate transmittance up to t
		T_maj_r *= std::exp(-sigma_maj_r * dt);
		T_maj_g *= std::exp(-sigma_maj_g * dt);
		T_maj_b *= std::exp(-sigma_maj_b * dt);

		// Compute null-scatter probability (pbrt-v4 pNull = max(0, 1-pA-pS))
		T pAbsorb  = medium.sigma_ar / sigma_maj_r;  // use R-channel probabilities
		T pScatter = medium.sigma_sr / sigma_maj_r;
		T pNull    = std::max(T(0), T(1) - pAbsorb - pScatter);

		// Randomly select event type
		T um = rng.Uniform<float>();
		if (um < pAbsorb) {
			// Absorption event: path terminated
			result.event_type = MediumEventType::kAbsorption;
			result.t          = t;
			result.sigma_a_r  = medium.sigma_ar;
			result.sigma_a_g  = medium.sigma_ag;
			result.sigma_a_b  = medium.sigma_ab;
			result.phase_g    = medium.g;
			result.T_maj_r    = T_maj_r;
			result.T_maj_g    = T_maj_g;
			result.T_maj_b    = T_maj_b;
			return result;
		} else if (um < pAbsorb + pScatter) {
			// Real scatter event: caller should sample phase function
			result.event_type = MediumEventType::kScatter;
			result.t          = t;
			result.sigma_s_r  = medium.sigma_sr;
			result.sigma_s_g  = medium.sigma_sg;
			result.sigma_s_b  = medium.sigma_sb;
			result.sigma_a_r  = medium.sigma_ar;
			result.sigma_a_g  = medium.sigma_ag;
			result.sigma_a_b  = medium.sigma_ab;
			result.phase_g    = medium.g;
			result.T_maj_r    = T_maj_r;
			result.T_maj_g    = T_maj_g;
			result.T_maj_b    = T_maj_b;
			return result;
		} else {
			// Null event: reset T_maj and continue from t (pbrt-v4: T_maj = 1, tMin = t)
			T_maj_r = T(1); T_maj_g = T(1); T_maj_b = T(1);
			t_cur   = t;
			// Refresh the random variate (pbrt-v4: uMode = rng.Uniform<Float>())
			// (already consumed um above; next loop iteration draws fresh u)
		}
	}
}

// ===========================================================================
// 6. RatioTrackingTr
//
//    Transmittance estimator for HomogeneousMediumData.
//
//    For a homogeneous medium sigma_n = max(0, sigma_maj - sigma_a - sigma_s)
//    is identically 0 because sigma_maj = sigma_a + sigma_s = sigma_t.
//    In this case the stochastic ratio-tracking estimator from pbrt-v4
//    (VolPathIntegrator::SampleLd, integrators.cpp lines 330-365) degenerates:
//    every sampled point has null-scattering weight sigma_n/sigma_maj = 0,
//    collapsing Tr to 0 -- which is not Beer-Lambert.
//
//    pbrt-v4 handles this correctly: HomogeneousMedium::SampleRay returns a
//    HomogeneousMajorantIterator where sigma_maj == sigma_t, and in
//    SampleT_maj the zero-dt accumulated T_maj path yields Beer-Lambert
//    automatically.  For a homogeneous medium the transmittance is
//    deterministic: Tr = exp(-sigma_t * d).
//
//    This function therefore computes Beer-Lambert directly.
//    The stochastic loop is only needed for *heterogeneous* media (e.g.
//    DDAMajorantIterator) where sigma_n > 0.  A heterogeneous overload can
//    be added when a HeterogeneousMediumData type is introduced.
//
//    Returns per-channel transmittance (Tr_r, Tr_g, Tr_b) in [0, 1].
// ===========================================================================
template <typename T>
CPU_GPU void RatioTrackingTr(
	const HomogeneousMediumData<T>& medium,
	T tMin, T tMax,
	RNG& /*rng*/,  // unused for homogeneous media; kept for API symmetry
	T& Tr_r, T& Tr_g, T& Tr_b,
	int /*max_steps*/ = 1024)
{
	// Beer-Lambert: Tr = exp(-sigma_t * d), sigma_t = sigma_a + sigma_s
	T d = tMax - tMin;
	Tr_r = std::exp(-(medium.sigma_ar + medium.sigma_sr) * d);
	Tr_g = std::exp(-(medium.sigma_ag + medium.sigma_sg) * d);
	Tr_b = std::exp(-(medium.sigma_ab + medium.sigma_sb) * d);
}

// ===========================================================================
// 7. RatioTrackingTrHeterogeneous
//
//    Single-channel stochastic ratio-tracking transmittance estimator for a
//    HETEROGENEOUS medium - the extension RatioTrackingTr's comment above
//    anticipated ("can be added when a HeterogeneousMediumData type is
//    introduced"). Mirrors pbrt-v4 VolPathIntegrator::SampleLd's
//    ratio-tracking loop (integrators.cpp lines 330-365): march at the
//    majorant rate, multiply in (1 - sigma_t(point)/sigma_maj) at every
//    null collision.
//
//    Templated on caller-supplied callables rather than tied to one
//    medium's density representation or one RNG convention: this project
//    has more than one of each (CloudMedium<T>'s procedural FBm density vs.
//    RGBGridMediumData<T>'s per-voxel grid; this file's own RNG class vs.
//    TheRestOfYourLife's global random_double()).
//
//    uniform():      () -> T, a uniform sample in [0,1)
//    sigma_t_at(t):   T -> T, local sigma_t at parametric distance t
// ===========================================================================
template <typename T, typename UniformFn, typename SigmaTFn>
CPU_GPU T RatioTrackingTrHeterogeneous(
	T tMin, T tMax, T sigma_maj,
	UniformFn&& uniform, SigmaTFn&& sigma_t_at,
	int max_steps = 100000)
{
	if (sigma_maj <= T(0)) return T(1);
	T Tr = T(1);
	T t  = tMin;
	for (int i = 0; i < max_steps; ++i) {
		T dt = SampleExponential(uniform(), sigma_maj);
		t += dt;
		if (t >= tMax) break;  // exited the medium: done
		T sigma_t = sigma_t_at(t);
		Tr *= T(1) - sigma_t / sigma_maj;
		if (Tr <= T(0)) return T(0);
	}
	return Tr;
}
