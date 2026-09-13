#pragma once
// wavefront_guiding.h -- real-time path guiding (Live Preview only, wavefront
// GPU backend). See this project's own plan for the full design rationale.
//
// Reuses GpuProbeGridMeta's existing world-space grid (probe_grid_types.h) -
// no new spatial structure, no new update rays. Adds one small per-probe
// directional histogram (a coarse 4x4 octahedral grid, kGuidingCells=16),
// fed by the SAME already-traced probe-cache-update rays
// (wavefront_kernels_restir.cu's own probe_cache_accumulate), at zero extra
// ray-tracing cost. Used to bias glossy/rough materials' own BSDF-sampled
// bounce direction toward a coarse estimate of where incident radiance
// actually is - MaterialType::Conductor/RoughMetal only in v1 (see
// wavefront_kernels_materials.cu's own call sites) - via a defensive mixture
// pdf combining this histogram's own pdf with the BSDF's own VNDF pdf,
// exactly the "multiple sampling techniques, one combined pdf" pattern
// (Veach/pbrt), NOT a 3rd wf_mis() participant - NEE-vs-BSDF MIS for direct
// lighting is completely untouched by this file.
//
// Deliberately dependency-free: every function below takes its own random
// numbers as explicit float parameters (the same convention TrowbridgeReitz::
// Sample_wm(), src/shared/microfacet.h, already uses) rather than calling
// wf_rand()/wf_pcg() itself - those are defined partway through
// wavefront_device_helpers.h, so a direct dependency would impose an
// awkward "must be included after them" ordering. This header has none of
// that: it can be included from anywhere probe_grid_types.h already is,
// host or device side, in any order.
//
// GpuGuidingHistogram's own capacity/resolution and the pGuide ramp-up
// policy (wf_guiding_probability(), below) are the two central v1 scope
// choices - see this project's own plan for the reasoning on why 16 cells
// and a sub-1.0 max probability, not a larger/adaptive histogram or a
// guiding-only sampler.

#include <cuda_runtime.h>
// Relative path, not a bare quoted include - see optix_types.h's/
// probe_grid_types.h's own identical comment for why (resolves under both
// the plain MSBuild .cpp compile and nvcc's OptiX compile, whose -I flags
// differ).
#include "../../src/shared/cpu_gpu.h"
#include "probe_grid_types.h"  // GpuProbeGridMeta - wf_guiding_nearest_probe() below

static constexpr int kGuidingCellsPerAxis = 4;
static constexpr int kGuidingCells = kGuidingCellsPerAxis * kGuidingCellsPerAxis;  // 16

// Per-probe directional histogram - one per entry of the SAME array
// GpuProbe (probe_grid_types.h) is indexed by (OptiXRenderer::d_guidingHistograms_,
// sized probeGridMeta_.totalProbes, allocated/freed alongside d_probeGrid_).
// numSamplesEverAdded==0 means "never updated" - wf_guiding_probability()
// returns 0 for that case, the same graceful-miss convention GpuProbe::
// numRaysEverTraced==0 already uses for the SH-L1 cache.
struct GpuGuidingHistogram {
	float cellWeight[kGuidingCells] = {};  // per-cell EMA of incident-radiance luminance
	int   numSamplesEverAdded = 0;
};

// Direction -> continuous octahedral (u,v) in [-1,1]^2. Reuses
// OctahedralVector's own pre-Encode() projection math (src/shared/
// octahedral_variance.h: normalize by the L1 norm, then fold the lower
// hemisphere) without its 16-bit lossless quantization - a coarse NxN
// bucket id is all guiding needs, not a compression-grade encoding.
CPU_GPU void wf_guiding_dir_to_uv(float3 d, float& u, float& v) {
	const float len1 = fabsf(d.x) + fabsf(d.y) + fabsf(d.z);
	if (len1 < 1e-8f) { u = 0.0f; v = 0.0f; return; }
	float vx = d.x / len1, vy = d.y / len1;
	const float vz = d.z / len1;
	if (vz < 0.0f) {
		const float ax = fabsf(vx), ay = fabsf(vy);
		const float ox = (1.0f - ay) * (vx >= 0.0f ? 1.0f : -1.0f);
		const float oy = (1.0f - ax) * (vy >= 0.0f ? 1.0f : -1.0f);
		vx = ox; vy = oy;
	}
	u = vx; v = vy;
}

// Inverse of wf_guiding_dir_to_uv() - matches OctahedralVector::ToVec3()'s
// own unfold+normalize.
CPU_GPU float3 wf_guiding_uv_to_dir(float u, float v) {
	float ox = u, oy = v;
	float oz = 1.0f - (fabsf(ox) + fabsf(oy));
	if (oz < 0.0f) {
		const float xo = ox;
		ox = (1.0f - fabsf(oy)) * (xo >= 0.0f ? 1.0f : -1.0f);
		oy = (1.0f - fabsf(xo)) * (oy >= 0.0f ? 1.0f : -1.0f);
	}
	const float len = sqrtf(ox * ox + oy * oy + oz * oz);
	if (len < 1e-8f) return make_float3(0.0f, 0.0f, 1.0f);
	return make_float3(ox / len, oy / len, oz / len);
}

// Direction -> flat cell index in [0, kGuidingCells). No bare min/max on
// ints here (this header compiles under plain host MSVC too, where
// NOMINMAX already keeps Windows.h's own min/max macros out of the way,
// but this project's own convention - see probe_grid_types.h's build
// history - is to avoid relying on either <algorithm> or a stray macro
// collision in a small header like this one; a plain clamp is just as
// clear).
CPU_GPU int wf_guiding_cell_index(float3 dir) {
	float u, v;
	wf_guiding_dir_to_uv(dir, u, v);
	int cx = (int)((u * 0.5f + 0.5f) * kGuidingCellsPerAxis);
	int cy = (int)((v * 0.5f + 0.5f) * kGuidingCellsPerAxis);
	if (cx < 0) cx = 0; if (cx > kGuidingCellsPerAxis - 1) cx = kGuidingCellsPerAxis - 1;
	if (cy < 0) cy = 0; if (cy > kGuidingCellsPerAxis - 1) cy = kGuidingCellsPerAxis - 1;
	return cy * kGuidingCellsPerAxis + cx;
}

// Cell index -> a direction stratified within that cell (uJitterU/V in
// [0,1) place it inside the cell rather than always returning the cell
// center).
CPU_GPU float3 wf_guiding_cell_to_dir(int cellIndex, float uJitterU, float uJitterV) {
	const int cx = cellIndex % kGuidingCellsPerAxis;
	const int cy = cellIndex / kGuidingCellsPerAxis;
	const float cellSize = 2.0f / (float)kGuidingCellsPerAxis;
	const float u = -1.0f + cellSize * ((float)cx + uJitterU);
	const float v = -1.0f + cellSize * ((float)cy + uJitterV);
	return wf_guiding_uv_to_dir(u, v);
}

// Solid angle of one cell - octahedral equal-area cells split the sphere's
// 4*pi steradians evenly, so every cell has the SAME solid angle. This is
// what makes wf_guided_pdf() below an O(1) lookup instead of needing a
// per-cell Jacobian.
CPU_GPU float wf_guiding_cell_solid_angle() {
	return (4.0f * 3.14159265358979323846f) / (float)kGuidingCells;
}

// Weighted linear scan over the histogram's kGuidingCells (small and fixed -
// deliberately NOT a precomputed CDF array the way GpuSkyDistribution
// (optix_types.h) builds one for its own, much larger, built-ONCE
// distribution - see this project's own plan for why that machinery is the
// wrong template for a small, per-frame-updated-in-place histogram).
// uPick selects the cell (proportional to weight); uJitterU/V place the
// returned direction within it. Callers should gate on
// wf_guiding_probability(hist.numSamplesEverAdded) > 0 first (an
// unpopulated histogram, all-zero weights, falls back to cell 0's own
// center here rather than being a meaningful sample).
CPU_GPU float3 wf_guided_sample_direction(const GpuGuidingHistogram& hist,
												  float uPick, float uJitterU, float uJitterV) {
	float total = 0.0f;
	for (int i = 0; i < kGuidingCells; ++i) total += fmaxf(0.0f, hist.cellWeight[i]);
	if (total <= 0.0f) return wf_guiding_cell_to_dir(0, uJitterU, uJitterV);
	const float target = uPick * total;
	float running = 0.0f;
	int chosen = kGuidingCells - 1;
	for (int i = 0; i < kGuidingCells; ++i) {
		running += fmaxf(0.0f, hist.cellWeight[i]);
		if (target <= running) { chosen = i; break; }
	}
	return wf_guiding_cell_to_dir(chosen, uJitterU, uJitterV);
}

// PDF of wf_guided_sample_direction() at an arbitrary direction (needed to
// weight a BSDF-sampled direction against the guided distribution too, the
// same "evaluate the OTHER technique's pdf at MY sampled direction" duty
// every MIS combination in this codebase already performs) - O(1): map
// `dir` to its cell, normalize that cell's weight by the histogram total,
// divide by the cell's own solid angle. Returns 0 for an unpopulated
// histogram, matching wf_guided_sample_direction()'s own degenerate case.
CPU_GPU float wf_guided_pdf(const GpuGuidingHistogram& hist, float3 dir) {
	float total = 0.0f;
	for (int i = 0; i < kGuidingCells; ++i) total += fmaxf(0.0f, hist.cellWeight[i]);
	if (total <= 0.0f) return 0.0f;
	const int cell = wf_guiding_cell_index(dir);
	const float w = fmaxf(0.0f, hist.cellWeight[cell]);
	return (w / total) / wf_guiding_cell_solid_angle();
}

// EMA-blends one new (direction, luminance) sample into the matching cell -
// called from probe_cache_accumulate (wavefront_kernels_restir.cu) with the
// SAME direction/resolved-radiance a probe-cache-update ray already
// produced for the SH-L1 projection, at zero extra ray-tracing cost.
// kGuidingHistoryCap (32, smaller than the SH-L1 cache's own
// kProbeHistoryCap=64) keeps a single CELL responsive despite it only ever
// getting a fraction of one probe's total ray budget (a probe visited once
// every ~totalProbes/kProbesPerFrame_ frames splits that one sample across
// only ONE of kGuidingCells cells, unlike SH-L1's 4 shared coefficients
// which get a contribution from every visit).
CPU_GPU void wf_guiding_accumulate(GpuGuidingHistogram& hist, float3 dir, float luminance) {
	const int kGuidingHistoryCap = 32;
	const int cell = wf_guiding_cell_index(dir);
	int cap = hist.numSamplesEverAdded + 1;
	if (cap > kGuidingHistoryCap) cap = kGuidingHistoryCap;
	const float alpha = 1.0f / (float)cap;
	const float clamped = fmaxf(0.0f, luminance);
	hist.cellWeight[cell] = hist.cellWeight[cell] + alpha * (clamped - hist.cellWeight[cell]);
	hist.numSamplesEverAdded += 1;
}

// Nearest probe index for a world position, clamped into [0,dims) per axis -
// unlike wf_query_probe_grid()'s own trilinear 8-corner blend of a smooth
// SH-L1 signal, guiding uses nearest-probe only (blending directional
// HISTOGRAMS across probes isn't a simple scalar lerp - see this project's
// own plan for why). Returns -1 if the grid has no probes at all.
CPU_GPU int wf_guiding_nearest_probe(const GpuProbeGridMeta& meta, float3 worldPos) {
	if (meta.totalProbes <= 0) return -1;
	const float3 pc = meta.probeSpaceCoord(worldPos);
	int cx = (int)(pc.x + 0.5f);
	int cy = (int)(pc.y + 0.5f);
	int cz = (int)(pc.z + 0.5f);
	if (cx < 0) cx = 0; if (cx > meta.dims.x - 1) cx = meta.dims.x - 1;
	if (cy < 0) cy = 0; if (cy > meta.dims.y - 1) cy = meta.dims.y - 1;
	if (cz < 0) cz = 0; if (cz > meta.dims.z - 1) cz = meta.dims.z - 1;
	const int3 coord = {cx, cy, cz};
	return meta.probeIndex(coord);
}

// pGuide ramp-up policy - 0 until the nearest probe's histogram has real
// samples (graceful-miss, same convention as GpuProbe::numRaysEverTraced==0
// skipping the SH-L1 cache query entirely), then rises linearly to
// kGuideMaxProb over kGuideRampUpSamples accumulate() calls. Capped well
// below 1.0 so BSDF/VNDF sampling always keeps real weight in the mixture -
// a defensive/mixture pdf's variance-reduction guarantee needs BOTH
// techniques to keep nonzero probability mass everywhere the other one can
// reach, and BSDF sampling stays excellent for the lobe's own specular-ish
// core regardless of how well-populated guiding's own histogram gets.
CPU_GPU float wf_guiding_probability(int numSamplesEverAdded) {
	if (numSamplesEverAdded <= 0) return 0.0f;
	const float kGuideMaxProb = 0.5f;
	const float kGuideRampUpSamples = 16.0f;
	float t = (float)numSamplesEverAdded / kGuideRampUpSamples;
	if (t > 1.0f) t = 1.0f;
	return kGuideMaxProb * t;
}
