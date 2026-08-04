#pragma once
// ---------------------------------------------------------------------------
// homogeneous_medium.h
//
// Port of pbrt-v4 HomogeneousMedium (src/pbrt/media.h lines 217-263) and
// HomogeneousMajorantIterator (media.h lines 80-100).
//
// A homogeneous participating medium has spatially constant absorption
// (sigma_a), scattering (sigma_s), and emission (Le) coefficients plus an
// anisotropy parameter g for the Henyey-Greenstein phase function.
//
// The majorant for a homogeneous medium is just sigma_t = sigma_a + sigma_s
// across the entire ray interval [0, tMax], returned as a single segment.
// This mirrors pbrt-v4's HomogeneousMajorantIterator which yields one
// RayMajorantSegment then exhausts immediately.
//
// Design rules (matching grid_medium.h / rgb_grid_medium.h):
//   - Plain template structs tagged CPU_GPU
//   - T = double on CPU, float on GPU
//   - Scalars for sigma_a, sigma_s, Le (monochromatic port; pbrt-v4 uses
//     SampledSpectrum but our shared headers are wavelength-agnostic)
//   - No heap allocation in hot paths
//
// Depends on:
//   grid_medium.h     -- RayMajorantSegment<T>
//   volume_scattering.h -- HenyeyGreensteinPhaseFunction<T>
//
// Usage:
//   HomogeneousMedium<double> med(sigma_a, sigma_s, Le, g);
//   // Query properties at any point (uniform):
//   auto props = med.sample_point();
//   // Iterate majorant segments along a ray [0, tMax]:
//   auto it = med.sample_ray(tMax);
//   while (auto seg = it.next()) { /* use seg->tMin, tMax, sigma_maj */ }
//
// pbrt-v4 reference: src/pbrt/media.h lines 80-100, 217-263.
// ---------------------------------------------------------------------------

#include "grid_medium.h"       // RayMajorantSegment<T>
#include "volume_scattering.h" // HenyeyGreensteinPhaseFunction<T>
#include <algorithm>
#include <cmath>
#include <optional>

// ---------------------------------------------------------------------------
// HomogeneousMajorantIterator<T>
//
// Mirrors pbrt-v4 HomogeneousMajorantIterator (media.h lines 80-100).
// Yields a single RayMajorantSegment covering [tMin, tMax] with
// sigma_maj = sigma_a + sigma_s, then returns nullopt on the next call.
// ---------------------------------------------------------------------------
template<typename T>
class HomogeneousMajorantIterator {
public:
	// Default-constructed iterator is already exhausted (pbrt-v4: called=true)
	CPU_GPU HomogeneousMajorantIterator() : called_(true) {}

	CPU_GPU HomogeneousMajorantIterator(T tMin, T tMax, T sigma_maj)
		: seg_{tMin, tMax, sigma_maj}, called_(false) {}

	// next() mirrors pbrt-v4 HomogeneousMajorantIterator::Next().
	// Returns the single segment on the first call, nullopt thereafter.
	CPU_GPU std::optional<RayMajorantSegment<T>> next() {
		if (called_) return std::nullopt;
		called_ = true;
		return seg_;
	}

private:
	RayMajorantSegment<T> seg_;
	bool called_;
};

// ---------------------------------------------------------------------------
// MediumPoint<T>
//
// Mirrors pbrt-v4 MediumProperties (media.h lines 73-78).
// Holds the scattering coefficients and emission at a query point.
// We use a scalar phase function g value instead of a PhaseFunction variant.
// ---------------------------------------------------------------------------
template<typename T>
struct MediumPoint {
	T sigma_a;   // absorption coefficient
	T sigma_s;   // scattering coefficient
	T Le;        // emission radiance (monochromatic)
	T g;         // HG anisotropy parameter in [-1, 1]

	CPU_GPU T sigma_t() const { return sigma_a + sigma_s; }
};

// ---------------------------------------------------------------------------
// HomogeneousMedium<T>
//
// Mirrors pbrt-v4 HomogeneousMedium (media.h lines 217-263).
//
// Members:
//   sigma_a_  -- absorption coefficient (>= 0)
//   sigma_s_  -- scattering coefficient (>= 0)
//   Le_       -- emission coefficient   (>= 0)
//   g_        -- HG anisotropy parameter in (-1, 1)
//   phase_    -- HenyeyGreensteinPhaseFunction<T> for importance sampling
// ---------------------------------------------------------------------------
template<typename T>
struct HomogeneousMedium {
	// MajorantIterator type alias — mirrors pbrt-v4:
	//   using MajorantIterator = HomogeneousMajorantIterator;
	using MajorantIterator = HomogeneousMajorantIterator<T>;

	// Constructors
	CPU_GPU HomogeneousMedium() : sigma_a_(T(0)), sigma_s_(T(0)), Le_(T(0)), g_(T(0)) {}

	// Primary constructor
	// sigmaScale multiplies both sigma_a and sigma_s (matches pbrt-v4 Scale calls).
	// LeScale multiplies Le.
	CPU_GPU HomogeneousMedium(T sigma_a, T sigma_s, T Le, T g,
							   T sigmaScale = T(1), T LeScale = T(1))
		: sigma_a_(sigma_a * sigmaScale)
		, sigma_s_(sigma_s * sigmaScale)
		, Le_(Le * LeScale)
		, g_(g)
		, phase_(g)
	{}

	// ---------------------------------------------------------------------------
	// IsEmissive — pbrt-v4 HomogeneousMedium::IsEmissive
	// ---------------------------------------------------------------------------
	CPU_GPU bool is_emissive() const { return Le_ > T(0); }

	// ---------------------------------------------------------------------------
	// sample_point — mirrors pbrt-v4 HomogeneousMedium::SamplePoint.
	// Returns MediumPoint with constant coefficients (uniform medium).
	// ---------------------------------------------------------------------------
	CPU_GPU MediumPoint<T> sample_point() const {
		return MediumPoint<T>{ sigma_a_, sigma_s_, Le_, g_ };
	}

	// ---------------------------------------------------------------------------
	// sample_ray — mirrors pbrt-v4 HomogeneousMedium::SampleRay.
	// Returns an iterator that yields one segment [0, tMax] with
	//   sigma_maj = sigma_a + sigma_s  (the exact majorant for a uniform medium).
	// ---------------------------------------------------------------------------
	CPU_GPU HomogeneousMajorantIterator<T> sample_ray(T tMax) const {
		T sigma_maj = sigma_a_ + sigma_s_;
		return HomogeneousMajorantIterator<T>(T(0), tMax, sigma_maj);
	}

	// Convenience accessors
	CPU_GPU T sigma_a()  const { return sigma_a_; }
	CPU_GPU T sigma_s()  const { return sigma_s_; }
	CPU_GPU T sigma_t()  const { return sigma_a_ + sigma_s_; }
	CPU_GPU T Le()       const { return Le_; }
	CPU_GPU T g()        const { return g_; }

	// Phase function for importance sampling
	CPU_GPU const HenyeyGreensteinPhaseFunction<T>& phase() const { return phase_; }

private:
	T sigma_a_;
	T sigma_s_;
	T Le_;
	T g_;
	HenyeyGreensteinPhaseFunction<T> phase_;
};
