#pragma once
// probe_grid_types.h -- World-space irradiance probe cache (DDGI-style),
// wavefront GPU backend, Live Preview only.
//
// Fills the gap between ReSTIR GI (single-bounce, depth 0->1 only - see
// wavefront_kernels_restir.cu's own "MVP scope: Lambertian x0 only" comment)
// and every deeper bounce, which today just traces on with classic
// single-draw NEE at full cost forever. A persistent grid of probes, built
// once per scene and updated incrementally over many frames, gives a cheap
// depth>=2 diffuse lookup instead of tracing further - see this project's
// own plan for the full design rationale.
//
// Kept in its own header rather than optix_types.h (scoped to ReSTIR-DI/
// light structs) or wavefront_types.h (scoped to spectral-width-coupled GI
// structs, kWFNWavelengths) - a probe grid has neither coupling. Included
// from both the host side (wavefront_path_tracer.h/.cpp, optix_renderer.h/
// optix_renderer_scene.cpp - plain floats/ints, no CUDA-only syntax) and the
// device side (wavefront_probe_cache.h, wavefront_device_helpers.h).

#include <cuda_runtime.h>
// Relative path, not a bare quoted include - see optix_types.h's own
// identical comment for why (resolves under both the plain MSBuild .cpp
// compile and nvcc's OptiX compile, whose -I flags differ).
#include "../../src/shared/cpu_gpu.h"

// Per-probe persistent state: a low-order (L1) real spherical-harmonics
// irradiance estimate per color channel, plus a mean hit distance/variance
// used as a coarse light-leak guard at lookup time. Deliberately NOT full
// DDGI octahedral irradiance+depth textures - no per-probe texture-atlas
// infrastructure exists anywhere in this GPU pipeline today, and this
// codebase's own established pattern (ReSTIR GI's Lambertian-only MVP,
// BSSRDF's single dedicated raygen instead of a general subsurface
// framework) is smallest-correct-thing-first. SH L1 is plain POD, evaluates
// via a closed-form dot product (wf_probe_sh_irradiance() below), and
// matches the "plain data struct, no textures" convention GpuReservoir/
// GpuGiSample already set. Cost: SH L1 is a low-pass filter over the
// sphere - it can't represent sharp lighting gradients - but this feeds an
// already-blurry diffuse cosine lobe at depth>=2, so the mismatch is
// second-order.
struct GpuProbe {
	// Real SH L1 basis coefficients {Y00, Y1-1, Y10, Y11}, one triple per
	// color channel - see wf_probe_sh_project()/wf_probe_sh_irradiance()
	// (this file) for the exact basis functions and the closed-form
	// irradiance-from-SH-L1 formula (Ramamoorthi & Hanrahan 2001).
	float shR[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float shG[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float shB[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	// EMA of hit distance / hit-distance^2 along traced probe-update rays -
	// variance = meanDistSq - meanDist^2. Used at lookup time as a coarse
	// stand-in for DDGI's own per-texel depth map: a shading point farther
	// from a probe than that probe's own average unoccluded reach is
	// treated as occluded from that probe (see wf_query_probe_grid()'s own
	// comment for the exact test and its known thin-wall-leak limitation).
	float meanDist = 0.0f;
	float meanDistSq = 0.0f;
	// 0 => this probe has never been updated (e.g. still mid-way through its
	// first round-robin cycle after a scene load) - wf_query_probe_grid()
	// skips it entirely and renormalizes the remaining trilinear corners.
	// This IS this v1's entire "uninitialized probe" handling - no DDGI-style
	// probe relocation/classification.
	int numRaysEverTraced = 0;
};

// Grid placement metadata - one instance, uploaded once per scene build
// (OptiXRenderer::probeGridMeta_, see optix_renderer_scene.cpp's own
// buildProbeGrid()). Per-probe STATE (GpuProbe, above) lives in a separate
// flat device array (OptiXRenderer::d_probeGrid_) indexed by
// probeIndex(probeCoord(...)) - this struct only carries the fixed
// placement, never re-uploaded after scene build (mutated device-side only
// from then on, same "owned and mutated device-side across frames" pattern
// as WavefrontPathTracer::d_reservoirsHistory_).
struct GpuProbeGridMeta {
	float3 gridMin = {0.0f, 0.0f, 0.0f};   // world-space position of probe (0,0,0)
	float3 cellSize = {1.0f, 1.0f, 1.0f};  // world-space spacing between adjacent probes, per axis
	int3   dims = {0, 0, 0};               // probe counts per axis; dims.x*dims.y*dims.z == totalProbes
	int    totalProbes = 0;                // 0 => no probe grid built for this scene (empty/degenerate scene bounds)

	// Continuous probe-space coordinate for a world position (NOT clamped or
	// rounded - callers floor/ceil as needed for trilinear interpolation).
	CPU_GPU float3 probeSpaceCoord(float3 worldPos) const {
		return make_float3((worldPos.x - gridMin.x) / cellSize.x,
							(worldPos.y - gridMin.y) / cellSize.y,
							(worldPos.z - gridMin.z) / cellSize.z);
	}
	// Flat array index for an integer probe coordinate - caller's
	// responsibility to clamp coord into [0,dims) first (see
	// wf_query_probe_grid()'s own corner-clamping).
	CPU_GPU int probeIndex(int3 coord) const {
		return (coord.z * dims.y + coord.y) * dims.x + coord.x;
	}
	// World-space position of a given integer probe coordinate - used both
	// by the update-scheduling host code (building ProbeCacheRayWorkItem::
	// origin) and by the lookup's own per-corner leak-distance test.
	CPU_GPU float3 probeWorldPos(int3 coord) const {
		return make_float3(gridMin.x + coord.x * cellSize.x,
							gridMin.y + coord.y * cellSize.y,
							gridMin.z + coord.z * cellSize.z);
	}
};

// Real SH-L1 basis functions (Ramamoorthi & Hanrahan 2001 eq. 3 / pbrt-v4
// 4.5.2's own real-SH convention), evaluated at a UNIT direction d. Used
// both to PROJECT a single incoming-radiance sample onto the basis at
// probe-update time (wavefront_probe_cache.h's own accumulate step) and, via
// wf_probe_sh_irradiance() below, to EVALUATE the cosine-convolved
// irradiance at a shading normal.
CPU_GPU void wf_probe_sh_basis(float3 d, float basis[4]) {
	basis[0] = 0.282095f;          // Y00 = 1/(2*sqrt(pi))
	basis[1] = 0.488603f * d.y;    // Y1,-1
	basis[2] = 0.488603f * d.z;    // Y1,0
	basis[3] = 0.488603f * d.x;    // Y1,1
}

// Closed-form irradiance (cosine-convolved) from SH-L1 coefficients at unit
// normal n - Ramamoorthi & Hanrahan 2001's own A0/A1 convolution constants
// collapse the general "integrate SH * clamped-cosine over the hemisphere"
// formula to this for L1-only. Returns un-clamped (caller fmaxf(0,...) as
// needed - a small negative value is possible for a nearly-empty probe with
// numerical noise in its own EMA, not a sign of a real negative irradiance).
CPU_GPU float wf_probe_sh_irradiance(const float sh[4], float3 n) {
	return sh[0] * 0.886227f + (sh[1] * n.y + sh[2] * n.z + sh[3] * n.x) * 1.023328f;
}

// Trilinear 8-probe irradiance lookup at a shading point, with a coarse
// per-corner light-leak test substituting for DDGI's own per-texel depth
// map (see this project's own plan for the accepted thin-wall-leak
// trade-off). Returns {0,0,0} (a cache MISS, never a false-dark answer) when
// every one of the 8 surrounding probes is either out of grid bounds,
// uninitialized (numRaysEverTraced==0 - see GpuProbe's own comment), or
// fails the leak test - the caller (wf_finish_material_scatter) falls
// through to a normal continuation ray on a miss, exactly like a reservoir
// with W==0 does elsewhere in this codebase.
//
// kLeakSlack=1.5: a corner is trusted out to 2*kLeakSlack = 3 standard
// deviations past its own mean unoccluded reach (this project's own plan's
// own formula - the leading 2.0f below is a fixed baseline multiplier, not
// itself part of kLeakSlack) - loose enough that a probe sitting in an
// open room doesn't spuriously reject nearby geometry from ordinary sampling
// noise in its own EMA, tight enough to reject a shading point that is
// obviously on the far side of a wall from a probe whose rays never reach
// that far.
CPU_GPU float3 wf_query_probe_grid(const GpuProbeGridMeta& meta, const GpuProbe* probes,
										   float3 worldPos, float3 normal) {
	if (meta.totalProbes <= 0 || probes == nullptr) return make_float3(0.0f, 0.0f, 0.0f);

	const float3 pc = meta.probeSpaceCoord(worldPos);
	const int3 base = {(int)floorf(pc.x), (int)floorf(pc.y), (int)floorf(pc.z)};
	const float3 frac = make_float3(pc.x - base.x, pc.y - base.y, pc.z - base.z);
	const float kLeakSlack = 1.5f;

	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float weightSum = 0.0f;
	for (int i = 0; i < 8; ++i) {
		const int dx = i & 1, dy = (i >> 1) & 1, dz = (i >> 2) & 1;
		const int3 coord = {base.x + dx, base.y + dy, base.z + dz};
		if (coord.x < 0 || coord.x >= meta.dims.x ||
			coord.y < 0 || coord.y >= meta.dims.y ||
			coord.z < 0 || coord.z >= meta.dims.z) continue;

		const int idx = meta.probeIndex(coord);
		const GpuProbe& p = probes[idx];
		if (p.numRaysEverTraced == 0) continue;

		// Component-wise, not `worldPos - probePos` - float3 has no
		// operator- visible to a plain (non-nvcc) host compile, and this
		// header is included from both sides (see this file's own header
		// comment) - matches optix_renderer_scene.cpp's own float3 math
		// convention for the same reason.
		const float3 probePos = meta.probeWorldPos(coord);
		const float ddx = worldPos.x - probePos.x, ddy = worldPos.y - probePos.y, ddz = worldPos.z - probePos.z;
		const float dist = sqrtf(ddx * ddx + ddy * ddy + ddz * ddz);
		const float variance = fmaxf(0.0f, p.meanDistSq - p.meanDist * p.meanDist);
		const float maxReach = p.meanDist + 2.0f * sqrtf(variance) * kLeakSlack;
		if (dist > maxReach) continue;

		const float wx = dx ? frac.x : (1.0f - frac.x);
		const float wy = dy ? frac.y : (1.0f - frac.y);
		const float wz = dz ? frac.z : (1.0f - frac.z);
		const float w = wx * wy * wz;
		if (w <= 0.0f) continue;

		sum.x += w * fmaxf(0.0f, wf_probe_sh_irradiance(p.shR, normal));
		sum.y += w * fmaxf(0.0f, wf_probe_sh_irradiance(p.shG, normal));
		sum.z += w * fmaxf(0.0f, wf_probe_sh_irradiance(p.shB, normal));
		weightSum += w;
	}

	if (weightSum <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
	return make_float3(sum.x / weightSum, sum.y / weightSum, sum.z / weightSum);
}
