#pragma once
// scene_size.h -- how big a scene is, as one number in the scene's own world units.
//
// Interactive controls need a length to scale by: a free-fly key press that moves 20 units is a comfortable step in the 555-unit Cornell box and a leap out of the
// scene in a 6-unit room. The size is the largest side of the scene's bounding box, measured the way the GPU scene frame is (gpu_scene_frame.h: triangles and
// spheres after cones, paraboloids, patches and curves are tessellated, a huge ground-plane-as-a-sphere left out), so a scene with a 2000-unit ground sphere
// still reports the size of what is on it. Disks and cylinders are not counted (they are not part of that frame); a scene made only of those reports 0, which
// means "unknown" and makes a caller fall back to something else (the camera's distance to its target).
//
// A scene file carries it in its own header as "# @rt-size 555" (see pbrt_discover.h's readHeaderTags), written by `ray_tracer --print-scene-size` and
// scripts/stamp_scene_sizes.py for the bundled scenes and by the Scene Builder for its own. Standard library plus the pbrt loader.

#include <cmath>
#include <string>

#include "gpu_scene_frame.h"
#include "gpu_tessellate.h"
#include "pbrt_flatten.h"
#include "pbrt_load.h"

namespace scene_size {

// The size of an already flattened scene (0 when it has no triangles or spheres).
inline double of(const pbrt_flatten::FlatScene &scene) {
	pbrt_flatten::FlatScene copy = scene;   // tessellation rewrites the lists, and the caller's scene is still needed
	gpu_tessellate::tessellateForBackend(copy, gpu_tessellate::TessellationCaps::all(), nullptr);
	const double extent = gpu_scene_frame::computeSceneFrame(copy).extent;
	return extent > 1e-6 ? extent : 0.0;
}

// The size of the scene in a .pbrt file, or a negative number (with `error` set) when the file cannot be loaded (a missing mesh file, say).
inline double ofFile(const std::string &path, std::string *error = nullptr) {
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path);
	if (!r.ok) {
		if (error) *error = r.error;
		return -1.0;
	}
	return of(r.scene);
}

// Three significant digits: a size is a hint for a step length, and "555" reads better in a header than "555.0000000001".
inline double rounded(double size) {
	if (size <= 0.0) return 0.0;
	const double magnitude = std::pow(10.0, std::floor(std::log10(size)) - 2.0);
	return std::round(size / magnitude) * magnitude;
}

}  // namespace scene_size
