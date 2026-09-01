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
