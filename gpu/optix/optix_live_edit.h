#pragma once
// optix_live_edit.h -- the OptiX half of the Live Preview's object editing (src/shared/realtime_api.h: realtime_pick_object, realtime_set_object_offset,
// realtime_reset_objects, realtime_export_arrangement). The shared pieces are live_object_edit.h (which objects a scene has, moving one, finding which one a point
// belongs to) and pbrt_arrangement.h (writing the moved scene back out); this file keeps the state between calls:
//
//   * the offsets: one world-space {dx, dy, dz} per object of the current scene, absolute from where the file puts it. They outlive a rebuild and end with a
//     different scene or a reset.
//   * the scene as the last frame drew it (`Built`): the objects, their ranges in the file's text, the offsets it was built with and its pick index. A move only
//     bumps `version`; the next frame sees that and rebuilds, so until then a click is still matched against the picture that is on screen.
//
// scene_builder.cpp applies the offsets while it builds a scene from a pbrt file (build_loaded_pbrt_scene) and records the Built; optix_interface.cpp owns the
// extern "C" entry points. OptiX does not rescale or recentre a scene, so first-hit positions are already in the scene's own (pbrt) coordinates.
//
// Header-only; thread-safe (the GUI thread moves and picks while the render thread builds).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// optix_interface.cpp reaches here after Windows.h (through the CUDA headers) with its min/max macros still defined, which the shared headers below cannot take.
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#include "../../src/shared/live_object_edit.h"
#include "../../src/shared/pbrt_arrangement.h"
#pragma pop_macro("max")
#pragma pop_macro("min")

namespace optix_live_edit {

using Offsets = std::vector<std::array<double, 3>>;

// True while this thread is inside rt_realtime_render_frame(): only then does a scene build keep what picking needs (the pick index is memory a batch render does not).
inline thread_local bool g_liveFrame = false;
struct LiveFrameScope {
	bool previous;
	LiveFrameScope() : previous(g_liveFrame) { g_liveFrame = true; }
	~LiveFrameScope() { g_liveFrame = previous; }
};

// A scene as one build drew it.
struct Built {
	std::string key;                                 // what was built: the path plus the offsets it was built with
	live_objects::PickIndex pick;
	live_objects::ObjectList objects;
	std::vector<pbrt_flatten::ShapeRange> ranges;    // where each shape sits in the file's text, for "Save arrangement"
	Offsets offsets;                                 // the offsets this build had
	double extent = 0.0;                             // the largest side of the box around everything pickable: picking tolerance is a fraction of it
};

inline bool anyMoved(const Offsets& o) {
	for (const auto& d : o)
		if (d[0] != 0.0 || d[1] != 0.0 || d[2] != 0.0) return true;
	return false;
}

// A key for a build: the unmoved scene is just its path (the same key as before editing existed); a moved one adds every nonzero offset exactly.
inline std::string keyFor(const std::string& path, const Offsets& offsets) {
	if (!anyMoved(offsets)) return path;
	std::string key = path + "|moved";
	char buf[100];
	for (std::size_t i = 0; i < offsets.size(); ++i) {
		const auto& d = offsets[i];
		if (d[0] == 0.0 && d[1] == 0.0 && d[2] == 0.0) continue;
		std::snprintf(buf, sizeof buf, "|%zu:%.17g,%.17g,%.17g", i, d[0], d[1], d[2]);
		key += buf;
	}
	return key;
}

// The pick information for a scene that has been built from `moved` (the flattened scene with the offsets already applied, before any tessellation).
inline std::shared_ptr<const Built> makeBuilt(const pbrt_flatten::FlatScene& original, const pbrt_flatten::FlatScene& moved, live_objects::ObjectList objects,
                                              Offsets offsets, std::string key) {
	auto b = std::make_shared<Built>();
	b->key = std::move(key);
	offsets.resize(objects.size(), {0.0, 0.0, 0.0});
	b->offsets = std::move(offsets);
	b->pick = live_objects::buildPickIndex(moved, objects);
	b->objects = std::move(objects);
	b->ranges = original.shapeRanges;
	double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
	for (const live_objects::PickShape& s : b->pick.shapes)
		for (int a = 0; a < 3; ++a) {
			lo[a] = (std::min)(lo[a], s.lo[a]);
			hi[a] = (std::max)(hi[a], s.hi[a]);
		}
	b->extent = b->pick.shapes.empty() ? 0.0 : (std::max)({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
	return b;
}

// The same from the unmoved scene and the offsets alone, for a build that was made without it.
inline std::shared_ptr<const Built> makeBuiltFrom(const pbrt_flatten::FlatScene& original, const Offsets& offsets, std::string key) {
	pbrt_flatten::FlatScene moved = original;
	live_objects::ObjectList objects = live_objects::objectsOf(original);
	for (std::size_t i = 0; i < objects.size() && i < offsets.size(); ++i) live_objects::translateObject(moved, objects, i, offsets[i].data());
	return makeBuilt(original, moved, std::move(objects), offsets, std::move(key));
}

class Edits {
public:
	static Edits& get() {
		static Edits instance;
		return instance;
	}

	// The offsets to build `path` with: empty unless this scene has been built for a Live Preview and an object has been moved since.
	Offsets offsetsFor(const std::string& path) {
		std::lock_guard<std::mutex> lock(mutex_);
		return path == path_ ? offsets_ : Offsets();
	}
	// A bumped number means "the offsets changed since you last built": the next frame must rebuild the scene.
	std::uint64_t version() {
		std::lock_guard<std::mutex> lock(mutex_);
		return version_;
	}

	// A frame has just been built from `path`. A different scene ends the edits of the one before.
	void note(const std::string& path, std::shared_ptr<const Built> built) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (path != path_) {
			path_ = path;
			offsets_.clear();
		}
		if (offsets_.size() != built->objects.size()) offsets_.resize(built->objects.size(), {0.0, 0.0, 0.0});
		built_ = std::move(built);
	}

	bool setOffset(const std::string& path, int object, double dx, double dy, double dz) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (path != path_ || object < 0 || static_cast<std::size_t>(object) >= offsets_.size()) return false;
		std::array<double, 3>& o = offsets_[static_cast<std::size_t>(object)];
		if (o[0] == dx && o[1] == dy && o[2] == dz) return true;
		o = {dx, dy, dz};
		++version_;
		return true;
	}

	void reset(const std::string& path) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (path != path_) return;
		bool any = false;
		for (auto& o : offsets_) {
			any = any || o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0;
			o = {0.0, 0.0, 0.0};
		}
		if (any) ++version_;
	}

	// The object whose surface holds the world point, or -1 (also before any frame has been drawn). The box is where the object will be once the move that is
	// waiting is drawn: the scene the pick looks at is still the one on screen, so its box is shifted by (offset now - offset it was built with).
	int pick(const std::string& path, double x, double y, double z, double* outLo, double* outHi, double* outOffset, char* outLabel, int labelSize) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (path != path_ || !built_) return -1;
		// A first-hit position lies on the surface up to the picture's rounding; half a percent of the scene's size covers that.
		const double tolerance = 0.005 * (built_->extent > 0.0 ? built_->extent : 1.0);
		const double p[3] = {x, y, z};
		const live_objects::PickResult hit = live_objects::pick(built_->pick, p, tolerance);
		if (hit.object < 0) return -1;
		const live_objects::PickShape& shape = built_->pick.shapes[static_cast<std::size_t>(hit.slot)];
		const std::size_t o = static_cast<std::size_t>(hit.object);
		for (int a = 0; a < 3; ++a) {
			const double now = o < offsets_.size() ? offsets_[o][static_cast<std::size_t>(a)] : 0.0;
			const double built = o < built_->offsets.size() ? built_->offsets[o][static_cast<std::size_t>(a)] : 0.0;
			if (outLo) outLo[a] = shape.lo[a] + (now - built);
			if (outHi) outHi[a] = shape.hi[a] + (now - built);
			if (outOffset) outOffset[a] = now;
		}
		if (outLabel && labelSize > 0) std::snprintf(outLabel, static_cast<std::size_t>(labelSize), "%s", shape.label.c_str());
		return hit.object;
	}

	// Writes the scene's file with the moved objects where they now are (pbrt_arrangement.h). False with the reason in `message`.
	bool exportArrangement(const std::string& path, const char* outPath, char* message, int messageSize) {
		const auto say = [&](const std::string& text) { if (message && messageSize > 0) std::snprintf(message, static_cast<std::size_t>(messageSize), "%s", text.c_str()); };
		std::lock_guard<std::mutex> lock(mutex_);
		if (path != path_ || !built_) { say("nothing has been drawn yet, so there is nothing to save"); return false; }
		namespace fs = std::filesystem;
		std::ifstream in(path_, std::ios::binary);
		if (!in) { say("cannot read the scene file " + path_); return false; }
		std::ostringstream text;
		text << in.rdbuf();
		const fs::path original = fs::absolute(path_);
		const std::string dir = original.parent_path().string() + "/";
		int objectsMoved = 0;
		for (const auto& o : offsets_) objectsMoved += (o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0) ? 1 : 0;
		if (objectsMoved == 0) { say("no object has been moved"); return false; }
		const pbrt_arrangement::Result result = pbrt_arrangement::write(text.str(), built_->ranges, built_->objects, offsets_, dir,
			[](const std::string& file) { std::error_code ec; return fs::exists(file, ec); },
			"# " + original.filename().string() + " with " + std::to_string(objectsMoved) + " object(s) moved, saved from Live Preview.\n");
		std::ofstream out(outPath, std::ios::binary);
		if (!out) { say(std::string("cannot write ") + outPath); return false; }
		out << result.text;
		out.close();
		if (!out) { say(std::string("cannot write ") + outPath); return false; }
		std::string summary = std::to_string(objectsMoved) + " object(s) moved";
		if (result.skippedShapes > 0) summary += "; " + std::to_string(result.skippedShapes) + " shape(s) sit in included files and stayed where they were";
		say(summary);
		return true;
	}

private:
	std::mutex mutex_;
	std::string path_;                  // the scene file the offsets belong to
	Offsets offsets_;
	std::shared_ptr<const Built> built_;
	std::uint64_t version_ = 0;
};

}  // namespace optix_live_edit
