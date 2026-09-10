#pragma once
// wavefront_restir_gi_math.h -- ReSTIR GI's own pure math: exactly the
// piece DI's own wavefront_restir_math.h does NOT already cover, split out
// the same way and for the same reason (see that header's own comment) -
// no scene data, no RNG draw beyond an already-generated [0,1) float, so
// this is unit-testable from a plain host build. Included by
// wavefront_restir_gi_helpers.h (the __device__-only candidate-generation/
// origin-context glue, which can only be included from within
// wavefront_device_helpers.h - mirrors wavefront_restir_helpers.h's own
// include-order comment) and from the GI finalize/spatial-reuse kernel
// (wavefront_kernels_restir.cu).
//
// GI reuses DI's RIS/reservoir-combine core UNCHANGED - restir_reservoir_add,
// restir_reservoir_combine, restir_finalize, and wf_restir_target_proxy
// (wavefront_restir_math.h) are payload-agnostic and apply verbatim to a
// GpuGiSample's cached RGB radiance in place of a light's raw emission. The
// ONE piece GI needs beyond that is wf_restir_gi_jacobian below.
//
// Why DI's reuse needs no separate Jacobian but GI's does (see this
// project's own ReSTIR GI plan for the full derivation): a DI light sample's
// pdf is an AREA-domain quantity intrinsic to the light's own geometry -
// origin-independent, so recomputing a fresh solid-angle pdf at a NEW
// shading origin (wf_reevaluate_light_geometry: area_pdf * dist^2 / cosine)
// is already the complete, exact correction. A GI candidate's pdf
// (GpuGiSample::pdfAtX0) is fundamentally different: it's a SOLID-ANGLE pdf
// at the ORIGINATING primary hit x0, produced by x0's own BSDF-sampling
// distribution - that density does not transport to a different origin x0'
// by a simple area-domain round trip (an "convert to an area pdf at x1 and
// reuse it the way DI reuses area pdfs" shortcut was checked algebraically
// against the correct shift-mapping formula below and is NOT equivalent -
// it silently drops a cosine/distance-squared factor). The standard fix
// (Ouyang et al. 2021 / gradient-domain rendering's own shift-mapping
// Jacobian) is the explicit multiplicative correction wf_restir_gi_jacobian
// computes.
//
// Callers fold this Jacobian into the `otherPHatAtDstContext` value passed
// to DI's own restir_reservoir_combine (wavefront_restir_math.h) - e.g.
// `restir_reservoir_combine(dst, other, freshPHat * jacobian, rand01)` -
// rather than needing a GI-specific combine function. This is exact, not an
// approximation: that product only ever feeds `weightSum` (via
// restir_reservoir_add's `risWeight`) and, if this candidate wins, the
// reservoir's own stored `pHat` - both remain internally consistent as long
// as the SAME `freshPHat * jacobian` product is used throughout this one
// candidate's own weight computation, exactly the way restir_finalize's
// W = weightSum / (M * pHat) expects.

#include "optix_types.h"
#include "wavefront_types.h"       // GpuGiSample/GpuGiReservoir
#include "optix_math_helpers.h"    // dot()/fabsf()/length() via <cmath> already pulled in transitively
#include "wavefront_restir_math.h" // restir_reservoir_add/combine/finalize, wf_restir_target_proxy - reused verbatim

// Shift-mapping Jacobian for reconnecting a stored GI sample's secondary
// point x1 to a DIFFERENT primary-hit origin than the one it was generated
// from. `s.x0Point`/`s.x1Point`/`s.x1Normal` are the ORIGINAL generation
// context (GpuGiSample's own fields); `newOrigin` is the CURRENT context's
// x0 (the pixel doing the reusing - itself, one frame later for temporal
// reuse, or a screen-space neighbor for spatial reuse).
//
// |J| = (cosTheta(x1->newOrigin) / cosTheta(x1->x0)) *
//       (dist(x0,x1)^2 / dist(newOrigin,x1)^2)
//
// both cosines measured against the secondary point's OWN shading normal
// (s.x1Normal), matching Ouyang et al. 2021's eq. 8/11 shift-mapping
// Jacobian (equivalently the reconnection Jacobian used throughout
// gradient-domain rendering for a "half-vector"/vertex-reconnection shift).
// Returns 0 (not NaN/Inf) for a degenerate configuration (x0 or newOrigin
// coincident with x1, or a grazing view of x1 from either origin) -
// mirrors DI's own wf_restir_jacobian's return-0-on-degenerate convention
// (wavefront_restir_math.h), which callers already treat as "reject this
// candidate", not "crash".
CPU_GPU inline float wf_restir_gi_jacobian(const float3& newOrigin, const GpuGiSample& s) {
	const float3 toOrigOrigin = s.x0Point - s.x1Point;
	const float  distOrigSq   = dot(toOrigOrigin, toOrigOrigin);
	if (distOrigSq < 1e-12f) return 0.0f;
	const float  distOrig     = sqrtf(distOrigSq);
	const float  cosOrig      = fabsf(dot(toOrigOrigin / distOrig, s.x1Normal));

	const float3 toNewOrigin = newOrigin - s.x1Point;
	const float  distNewSq   = dot(toNewOrigin, toNewOrigin);
	if (distNewSq < 1e-12f) return 0.0f;
	const float  distNew     = sqrtf(distNewSq);
	const float  cosNew      = fabsf(dot(toNewOrigin / distNew, s.x1Normal));

	if (cosOrig < 1e-6f) return 0.0f;
	return (cosNew / cosOrig) * (distOrigSq / distNewSq);
}
