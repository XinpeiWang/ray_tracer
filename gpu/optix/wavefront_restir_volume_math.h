#pragma once
// wavefront_restir_volume_math.h -- ReSTIR-for-volumetric-media's own pure,
// no-scene-data math: the spatial-neighbor validity gate and its supporting
// mean-free-path helper. See this project's own plan for the full design.
//
// Deliberately thin - unlike ReSTIR GI's own wavefront_restir_gi_math.h,
// no Jacobian/shift-mapping derivation lives here. A volumetric ReSTIR
// candidate is a GpuLightSample - the SAME area-domain, origin-independent
// sample type surface DI already resamples - so the existing
// wf_restir_jacobian (wavefront_restir_math.h) already provides the exact
// robustness guard this feature needs, unmodified: it only ever compares
// the CANDIDATE LIGHT's own geometry as seen from two querying points,
// with no dependency on what kind of vertex (surface or phase-scatter) is
// doing the querying.
//
// Split out into its own header purely so it can be unit-tested from a
// plain host build with no scene-data/OptiX-module include-order
// constraint, the same reason wavefront_restir_math.h/wavefront_restir_gi_math.h
// are their own headers - see wavefront_restir_helpers.h's own comment.

#include "optix_types.h"
#include "optix_math_helpers.h"  // dot()/fabsf()

// Mean free path (1/extinction) for a medium - the volumetric equivalent of
// "how far apart is too far apart" scale a surface's own geometric-edge
// discontinuity gives a normal-cosine test for free. For a heterogeneous
// medium, pass its majorant (an UPPER bound on true local density) rather
// than a locally-varying real density - 1/majorant UNDER-estimates the true
// mean free path almost everywhere in the volume, which errs the caller's
// own distance threshold toward TIGHTER, not looser, the safe direction for
// a variance-control gate (see wf_restir_volume_spatial_valid's own
// comment).
CPU_GPU inline float wf_restir_volume_mean_free_path(float sigmaTOrMajorant) {
	return 1.0f / fmaxf(sigmaTOrMajorant, 1e-6f);
}

// Spatial-neighbor validity gate, replacing DI's own normal-cosine
// threshold (a phase-scatter vertex has no shading normal). Two gates,
// both must pass:
//  - mediumMatIdx equality (hard reject) - the volumetric analog of "don't
//    blend across a geometric edge," stricter than a cosine test since two
//    different media can have wildly different sigma_t/g/albedo with no
//    partial-credit case.
//  - a mean-free-path-scaled distance threshold (soft, tunable via `scale`)
//    between the two pixels' own medium ENTRY points - catches two
//    disjoint volumes of the same material asset (e.g. a repeated "Fog"
//    prefab) landing near each other on screen, which mediumMatIdx
//    equality alone would not.
// A `phaseWo`-cosine gate (the direction-based analog of a normal-cosine
// test) was considered and deliberately NOT included: within the existing
// kRestirSpatialRadiusPixels=20px neighbor radius, two pinhole-camera
// pixels' view directions differ by a fraction of a degree - such a gate
// would pass almost unconditionally there, providing no real discriminative
// power (unlike a surface normal, which genuinely can differ sharply
// between adjacent pixels straddling a geometric edge).
CPU_GPU inline bool wf_restir_volume_spatial_valid(
		float3 curEntry, float3 neighborEntry,
		int curMediumMatIdx, int neighborMediumMatIdx,
		float meanFreePath, float scale) {
	if (curMediumMatIdx < 0 || neighborMediumMatIdx < 0) return false;
	if (curMediumMatIdx != neighborMediumMatIdx) return false;
	const float3 delta = make_float3(curEntry.x - neighborEntry.x, curEntry.y - neighborEntry.y, curEntry.z - neighborEntry.z);
	const float distSq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
	const float threshold = scale * meanFreePath;
	return distSq <= threshold * threshold;
}
