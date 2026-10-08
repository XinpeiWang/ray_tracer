// mesh_preview.h - what the Scene Builder's 3D view needs to draw a mesh file: its bounding box and a sample of its vertices (the whole mesh would
// be far too many polygons to redraw while an object is being dragged). Qt-free; reads .ply and .obj through ply_mesh.h.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "ply_mesh.h"

namespace mesh_preview {

struct MeshPreview {
	bool ok = false;
	std::string error;
	double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};  // the bounding box, in the file's own units (before the object's scale)
	std::vector<float> samples;                   // 3 per sampled vertex, evenly spread through the file's vertices
	size_t vertexCount = 0, triangleCount = 0;
};

// Loads `path` and keeps its bounds and at most `maxSamples` vertices. A file larger than `maxBytes` is not loaded (a preview must not stall the
// window): ok is false and `error` says so.
inline MeshPreview load(const std::string& path, size_t maxSamples = 2500, std::uintmax_t maxBytes = 80u * 1000u * 1000u) {
	MeshPreview out;
	{
		std::ifstream f(path, std::ios::binary | std::ios::ate);
		if (!f) { out.error = "the file could not be opened"; return out; }
		const std::streamoff size = f.tellg();
		if (size > 0 && static_cast<std::uintmax_t>(size) > maxBytes) { out.error = "the file is too big to preview"; return out; }
	}
	const ply_mesh::LoadResult r = ply_mesh::loadFile(path);
	if (!r.ok) { out.error = r.error; return out; }
	const size_t n = r.mesh.vertexCount();
	if (n == 0) { out.error = "the file has no vertices"; return out; }
	// A corrupt vertex (NaN or infinite) must not poison the box, or the 3D view's camera would be framed on NaN and go blank: only finite vertices count.
	auto finite = [&r](size_t i) {
		for (size_t a = 0; a < 3; ++a)
			if (!std::isfinite(r.mesh.positions[3 * i + a])) return false;
		return true;
	};
	size_t good = 0;
	for (size_t i = 0; i < n; ++i) {
		if (!finite(i)) continue;
		for (size_t a = 0; a < 3; ++a) {
			const double v = r.mesh.positions[3 * i + a];
			if (good == 0) out.lo[a] = out.hi[a] = v;
			out.lo[a] = std::min(out.lo[a], v);
			out.hi[a] = std::max(out.hi[a], v);
		}
		++good;
	}
	if (good == 0) { out.error = "the file has no usable vertices"; return out; }
	const size_t keep = std::min(good, std::max<size_t>(1, maxSamples));
	size_t seen = 0, next = 0;  // the next sample is the (next * good / keep)-th usable vertex
	for (size_t i = 0; i < n && next < keep; ++i) {
		if (!finite(i)) continue;
		if (seen == next * good / keep) {
			for (size_t a = 0; a < 3; ++a) out.samples.push_back(r.mesh.positions[3 * i + a]);
			++next;
		}
		++seen;
	}
	out.vertexCount = n;
	out.triangleCount = r.mesh.triangleCount();
	out.ok = true;
	return out;
}

}  // namespace mesh_preview
