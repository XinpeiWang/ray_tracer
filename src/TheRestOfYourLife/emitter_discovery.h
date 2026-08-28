#pragma once
//==============================================================================================
// emitter_discovery.h -- finds real, sample_area()-capable leaf shapes inside
// a possibly-wrapped hittable graph, for BDPTSceneAdapter's and
// SPPMSceneAdapter's constructor-time emitter scan.
//
// Both classes' own constructors used to scan `world.objects` directly,
// calling obj->sample_area(...) on each top-level entry - correct only when
// `world` is flat (every hand-authored native scene's build_world()). A
// pbrt-loaded scene's world is NOT flat: pbrt_cpu::build() (pbrt_cpu_
// builder.h) wraps its returned world in a single bvh_node for real-ray-
// tracing performance, so world.objects there holds exactly one bvh_node,
// not a flat list of spheres/quads/triangles - the naive scan then finds
// nothing (bvh_node has no sample_area() override, it's an aggregate, not a
// light), silently rendering every pbrt-loaded scene's area lights invisible
// under BDPT/MLT/RandomWalk/AO/SPPM.
//
// collectEmitterCandidates() below fixes this by recursively unwrapping
// PURELY STRUCTURAL/organizational containers - hittable_list, bvh_node,
// bvh_leaf, triangle_mesh, triangle_mesh_mtl - to find the real leaf shapes
// underneath. `world` itself is never touched (this only reads through it
// into a separate output vector), so it keeps the real BVH's O(log n) ray-
// intersection performance for actual rendering; this is purely a
// discovery-time helper.
//
// Deliberately narrower than cpu_interface.cpp's own spectral_scan_
// hittable() (which this was originally modeled on): that walker ALSO
// recurses through translate/rotate_y/transform_instance, which is safe for
// ITS purpose (finding a material for a --spectral whitelist check - it
// doesn't care about world-space geometry) but would be WRONG here. Those
// three types apply a REAL runtime coordinate transform inside their own
// hit() (translate offsets rec.p, rotate_y rotates it, transform_instance
// applies a full object-to-world matrix) that the wrapped object's own
// sample_area() knows nothing about - unwrapping through get_object() and
// calling sample_area() on the raw inner shape would return a position/
// normal in the WRONG (pre-transform, shared-object-local) frame, a correct-
// ness bug, not just a discovery gap. triangle_mesh/triangle_mesh_mtl are
// safe to unwrap because they apply no transform of their own - hit() just
// delegates to an inner BVH whose triangles already store final world-space
// vertex positions (baked in at OBJ/pbrt load time, not at hit() time).
// A light wrapped in translate/rotate_y/transform_instance is therefore
// still not found by this walker (same as before this fix existed) -
// narrower than full recursion, but correct rather than silently wrong.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "bvh.h"
#include "mesh.h"

#include <memory>
#include <vector>

inline void collectEmitterCandidates(const shared_ptr<hittable>& h,
									  std::vector<shared_ptr<hittable>>& out) {
	if (!h) return;
	if (auto list = std::dynamic_pointer_cast<hittable_list>(h)) {
		for (const auto& obj : list->objects) collectEmitterCandidates(obj, out);
		return;
	}
	if (auto node = std::dynamic_pointer_cast<bvh_node>(h)) {
		collectEmitterCandidates(node->get_left(), out);
		collectEmitterCandidates(node->get_right(), out);
		return;
	}
	if (auto leaf = std::dynamic_pointer_cast<bvh_leaf>(h)) {
		for (const auto& obj : leaf->get_prims()) collectEmitterCandidates(obj, out);
		return;
	}
	if (auto m = std::dynamic_pointer_cast<triangle_mesh>(h)) {
		collectEmitterCandidates(m->get_object(), out);
		return;
	}
	if (auto mm = std::dynamic_pointer_cast<triangle_mesh_mtl>(h)) {
		collectEmitterCandidates(mm->get_object(), out);
		return;
	}
	out.push_back(h);
}
