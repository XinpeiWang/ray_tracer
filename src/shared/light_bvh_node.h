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

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU inline
#  endif
#endif

// ===========================================================================
// LightBVHNode
// ===========================================================================

struct alignas(32) LightBVHNode {
	// Public members ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â matches pbrt-v4 layout exactly
	CompactLightBounds lightBounds;
	struct {
		unsigned int childOrLightIndex : 31;
		unsigned int isLeaf            :  1;
	};

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
		return LightBVHNode{cb, {lightIndex, 1}};
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
		return LightBVHNode{cb, {child1Index, 0}};
	}
};
