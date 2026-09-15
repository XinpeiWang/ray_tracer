// light_bvh_node.h
// BVH node for the light hierarchy, ported from pbrt-v4.
//
// LightBVHNode is the fundamental storage unit of the BVH light tree.
// Each node is aligned to 32 bytes (one cache line) and stores:
//   - lightBounds : CompactLightBounds  -- quantised power/direction bound
//   - childOrLightIndex : 31-bit uint   -- interior: index of child1 node
//                                          leaf: index into the lights array
//   - isLeaf : 1-bit flag               -- 0 = interior, 1 = leaf
//
// Static factories (mirrors pbrt-v4):
//   MakeLeaf(lightIndex, cb)      -- create a leaf node
//   MakeInterior(child1Index, cb) -- create an interior node
//
// Usage in the BVH traversal:
//   if (!node.isLeaf)  -> recurse into child (nodeIndex+1) and node.childOrLightIndex
//   else               -> emit lights[node.childOrLightIndex] with accumulated pmf
//
// Dependencies:
//   compact_light_bounds.h  -- CompactLightBounds
//
// References: pbrt-v4 src/pbrt/lightsamplers.h  (Apache-2.0)

#pragma once
#include "compact_light_bounds.h"

#include "cpu_gpu.h"
#include <cstddef>       // offsetof, for the layout static_asserts below
#include <type_traits>   // is_standard_layout, for the layout static_asserts below

// ===========================================================================
// LightBVHNode
// ===========================================================================

struct alignas(32) LightBVHNode {
	// Public members — matches pbrt-v4 layout exactly, EXCEPT
	// childOrLightIndex/isLeaf are two plain fields here, not a packed
	// bitfield. A bitfield's bit-packing order/allocation-unit rules are
	// compiler/ABI-defined, not specified by the C++ standard - this struct
	// is written host-side (MSVC, via src/shared/bvh_light_sampler2.h's
	// BVHLightSampler2::buildBVH()) and read device-side (NVCC, via
	// gpu_light_bvh_sample_index()/gpu_light_bvh_pmf(), optix_device_
	// helpers.h) as raw bytes (a host cudaMemcpy, not a re-parse), so the two
	// compilers packing `unsigned childOrLightIndex:31; unsigned isLeaf:1;`
	// differently would make the device read back a garbage childOrLightIndex
	// - confirmed as the real, reproducible root cause of a CUDA 700 illegal
	// memory access the first time a multi-light scene (needing a real
	// interior node, not just a single leaf) exercised this path: nodeIndex
	// decoded from the corrupted bits indexed miles past the uploaded nodes_
	// array. Plain fields have no such ambiguity - every compiler lays out
	// two ordinary struct members identically.
	CompactLightBounds lightBounds;
	unsigned int childOrLightIndex = 0;
	unsigned int isLeaf = 0;  // 1 = leaf, 0 = interior (kept as unsigned int, not bool, to match this struct's own pre-existing on-the-wire convention)

	// Default constructor
	LightBVHNode() = default;

	// ---------------------------------------------------------------------------
	// MakeLeaf: create a leaf node storing a single light.
	// lightIndex  -- index into the flat lights array
	// cb          -- compact bound for this light
	// ---------------------------------------------------------------------------
	CPU_GPU static LightBVHNode MakeLeaf(unsigned int lightIndex,
										  const CompactLightBounds& cb)
	{
		return LightBVHNode{cb, lightIndex, 1u};
	}

	// ---------------------------------------------------------------------------
	// MakeInterior: create an interior node.
	// child1Index -- index of the second child in the nodes array
	//               (child0 is implicitly nodeIndex + 1)
	// cb          -- compact bound covering both children
	// ---------------------------------------------------------------------------
	CPU_GPU static LightBVHNode MakeInterior(unsigned int child1Index,
											  const CompactLightBounds& cb)
	{
		return LightBVHNode{cb, child1Index, 0u};
	}
};

// This struct is written host-side (MSVC) and read device-side (NVCC) as raw
// uploaded bytes, not re-parsed - see the struct's own comment on the real,
// reproduced MSVC/NVCC bitfield-packing divergence that previously caused a
// CUDA 700 illegal memory access. These asserts pin the exact regression
// that already happened once (a reintroduced `childOrLightIndex:31` /
// `isLeaf:1` bitfield, or a reordering, would change these offsets) as a
// compile-time check in BOTH the host and device translation units that
// include this shared header, so a future recurrence fails the build
// instead of needing another multi-day debugging session to diagnose.
static_assert(std::is_standard_layout<LightBVHNode>::value,
			  "LightBVHNode must stay standard-layout (plain data members, "
			  "no virtuals) for offsetof()/a raw host-to-device byte copy "
			  "to be well-defined");
static_assert(offsetof(LightBVHNode, childOrLightIndex) == sizeof(CompactLightBounds),
			  "LightBVHNode::childOrLightIndex moved - verify MSVC and NVCC "
			  "still agree on this struct's layout (see struct comment)");
static_assert(offsetof(LightBVHNode, isLeaf) ==
				  sizeof(CompactLightBounds) + sizeof(unsigned int),
			  "LightBVHNode::isLeaf moved - verify MSVC and NVCC still "
			  "agree on this struct's layout (see struct comment)");

// Return type for a light-BVH traversal query - shared by both GPU backends'
// own traversal code (gpu_light_bvh_sample_index(), optix_device_helpers_
// lighting.h, GPU-recursive; wf_light_bvh_sample_index(), wavefront_restir_
// helpers.h, wavefront) since it's plain data with no backend-specific
// dependency, unlike the traversal functions themselves (each reads its own
// backend's __constant__ params/wf_params global, so THOSE stay hand-
// duplicated per this codebase's own established convention - see either
// function's own header comment).
struct GpuLightBvhSample {
	int lightIndex;  // -1 = no light BVH built, or zero importance everywhere
	float pmf;
};
