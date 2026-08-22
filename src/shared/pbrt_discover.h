#pragma once
// pbrt_discover.h -- finds .pbrt scenes on disk and reads just enough of each
// one to describe it, without loading its geometry.
//
// WHY A HEADER PARSE RATHER THAN A FULL LOAD
// ------------------------------------------
// The registry is built during static initialization, before anything knows
// which scene the user will pick. pbrt_load::loadFile() reads every Include
// and every .ply a scene references, which for a real scene is hundreds of
// megabytes and several seconds - paid at startup, for every scene, to
// populate a dropdown the user might never open. That is not a tolerable cost
// for metadata.
//
// pbrt's own file layout makes the cheap path exact rather than a guess: the
// camera, film and sampler directives are all required to appear BEFORE
// WorldBegin, and only geometry may appear after it. So truncating the text at
// WorldBegin yields a document that still contains every field this header
// needs and none of the ones that cost money to resolve. Geometry is loaded
// later, on demand, by whoever actually renders the scene.
//
// Deliberately Qt-free and registry-free: it returns plain data, so it can be
// tested without a GUI and without linking the CPU renderer's class hierarchy.
// scene_registry.h turns that data into SceneDescriptors; that is the only
// place the two ideas meet.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pbrt_flatten.h"
#include "pbrt_scene.h"

namespace pbrt_discover {

// What a .pbrt file on disk claims about itself, without its geometry.
struct Discovered {
	std::string path;                 // full path, what to load later
	std::string name;                 // filename stem, shown in the UI
	pbrt_flatten::Camera camera;      // derived from LookAt/Camera
	// The file's own PixelFilter directive, already resolved to numeric
	// params by flatten() - unlike samplerType/integrator/maxDepth below,
	// this one IS applied at render time (see PixelFilter's own comment,
	// pbrt_flatten.h), not just advisory.
	pbrt_flatten::PixelFilter filter;
	int samplesPerPixel = 16;
	int xResolution = 1280;
	int yResolution = 720;
	// The file's own Integrator directive - "integer maxdepth" and the
	// integrator name (e.g. "volpath", "bdpt") - matching pbrt_scene::
	// Scene's own defaults exactly, so a file with no Integrator directive
	// at all reports the same values a real pbrt would fall back to.
	// scene_registry.h's wire_pbrt_backed_scene() copies these onto
	// SceneDescriptor so a render can warn when what it's actually doing
	// diverges from what the scene asked for (see cpu_interface.cpp).
	int maxDepth = 5;
	std::string integrator = "volpath";
	// The file's own Sampler directive type name - same "matches
	// pbrt_scene::Scene's own default, informational only" shape as
	// integrator above; see SceneDescriptor::recommended_sampler's comment.
	std::string samplerType = "sobol";
	bool ok = false;
	std::string error;
	// Whether the file itself declared a Camera directive (see
	// pbrt_scene::Scene::cameraDeclared's comment) - scanDirectory() uses
	// this to tell a real top-level scene apart from a fragment that just
	// happens to parse cleanly.
	bool declaresCamera = false;
	// True when this file was found one level down, inside a subdirectory
	// of the scanned directory, rather than directly in it. scanDirectory()'s
	// own comment documents why: published collections (e.g.
	// github.com/mmp/pbrt-v4-scenes) package each scene as its own folder of
	// a .pbrt plus that scene's geometry/textures/bsdfs, which is exactly
	// what the one-level descent is there to find. A hand-authored scene
	// meant to be self-contained and checked into this repo has no reason to
	// need that folder-of-assets shape, so it sits directly in the scanned
	// directory instead. scene_registry.h uses this - not a per-file guess -
	// to set SceneDescriptor::requires_files.
	bool nested = false;
};

namespace detail {

// pbrt requires WorldBegin to separate scene-wide options from geometry, so
// everything before it is exactly the part we want and all of the part we can
// afford.
//
// WorldBegin ITSELF IS KEPT, and that is not cosmetic. The camera transform is
// not recorded when the Camera directive is read - it is the CTM *at
// WorldBegin* that becomes worldToCamera (see the field's comment in
// pbrt_scene.h). Cutting the directive away therefore leaves the transform at
// identity, and every scene reports a camera sitting at the origin looking
// down +Z regardless of what its LookAt said. The renders would still be
// correct, because those go through the full load - but the camera values the
// GUI shows and lets the user edit would all be wrong, which is a far quieter
// failure. A test pins this.
//
// Falls back to the whole text when absent: a file that is only options is
// unusual but legal, and truncating nothing is the right answer.
inline std::string headerOf(const std::string &text) {
	const std::size_t at = text.find("WorldBegin");
	if (at == std::string::npos) return text;
	return text.substr(0, at) + "WorldBegin\n";
}

inline std::string stemOf(const std::string &path) {
	const std::size_t slash = path.find_last_of("/\\");
	std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
	const std::size_t dot = base.find_last_of('.');
	if (dot != std::string::npos && dot > 0) base = base.substr(0, dot);
	return base;
}

inline std::string basenameOf(const std::string &path) {
	const std::size_t slash = path.find_last_of("/\\");
	return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

// The bare filenames (extension kept, directory component stripped) that
// this file's own Include/Import directives name - a raw text scan, not a
// real parse, so it still works on a file pbrt_scene::parse() rejects
// outright. That matters here specifically: a geometry/materials library
// with no WorldBegin of its own (see headerOf()'s "falls back to the whole
// text" comment) gets its ENTIRE body read as header content, which for a
// file that is nothing but geometry and NamedMaterial references can fail
// to parse even though it is perfectly valid pbrt once actually spliced
// into the real scene file that Includes it - see scanDirectory()'s own
// comment for why that case needs this list rather than a parse-success
// check to identify.
inline void collectIncludedNames(const std::string &text, std::set<std::string> &out) {
	for (const char *keyword : {"Include", "Import"}) {
		const std::string kw = keyword;
		std::size_t pos = 0;
		while ((pos = text.find(kw, pos)) != std::string::npos) {
			const std::size_t after = pos + kw.size();
			pos = after;
			// Reject a match that is part of a longer identifier (e.g. some
			// future "IncludeOnce" directive), not a whole-word "Include".
			if (after < text.size() &&
				(std::isalnum(static_cast<unsigned char>(text[after])) || text[after] == '_'))
				continue;
			const std::size_t q1 = text.find('"', after);
			if (q1 == std::string::npos) continue;
			const std::size_t q2 = text.find('"', q1 + 1);
			if (q2 == std::string::npos) continue;
			out.insert(basenameOf(text.substr(q1 + 1, q2 - q1 - 1)));
		}
	}
}

} // namespace detail

// Describes a single file. `text` is supplied by the caller so this stays
// testable without touching a filesystem, matching every other layer in the
// pbrt chain.
inline Discovered describe(const std::string &path, const std::string &text) {
	Discovered d;
	d.path = path;
	d.name = detail::stemOf(path);

	// No file resolver: an Include before WorldBegin is legal but rare, and
	// resolving it here would reintroduce exactly the disk cost this avoids.
	// The full load later sees the complete file, so nothing is lost.
	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(detail::headerOf(text));
	if (!parsed.ok) {
		d.error = parsed.error;
		return d;
	}

	d.samplesPerPixel = parsed.scene.samplesPerPixel;
	d.xResolution = parsed.scene.xResolution;
	d.yResolution = parsed.scene.yResolution;
	d.declaresCamera = parsed.scene.cameraDeclared;
	d.maxDepth = parsed.scene.maxDepth;
	d.integrator = parsed.scene.integrator;
	d.samplerType = parsed.scene.samplerType;

	// flatten() with no mesh resolver still derives the camera - that comes
	// from the world-to-camera matrix and the fov, neither of which needs a
	// single triangle to exist. Same for the PixelFilter - flatten() has
	// already resolved it to numeric params, no geometry needed either.
	const pbrt_flatten::FlatScene flat = pbrt_flatten::flatten(parsed.scene);
	d.camera = flat.camera;
	d.filter = flat.filter;
	d.ok = true;
	return d;
}

inline Discovered describeFile(const std::string &path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		Discovered d;
		d.path = path;
		d.name = detail::stemOf(path);
		d.error = "cannot open " + path;
		return d;
	}
	std::ostringstream ss;
	ss << in.rdbuf();
	return describe(path, ss.str());
}

namespace detail {

inline void collectPbrtFiles(const std::filesystem::path &from, std::vector<std::string> &out) {
	std::error_code ec;
	for (const auto &entry : std::filesystem::directory_iterator(from, ec)) {
		if (ec) break;
		if (!entry.is_regular_file(ec)) continue;
		std::string ext = entry.path().extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(),
					   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (ext == ".pbrt") out.push_back(entry.path().string());
	}
}

} // namespace detail

// Every .pbrt inside `dir`, plus every .pbrt one level down inside each of
// `dir`'s own subdirectories, sorted by path so the scene ids a user sees are
// stable between runs - directory order is not, and an id that moves when an
// unrelated file is added would break saved settings and any script that
// passes a scene number.
//
// One level of subdirectory descent, not zero and not unlimited: published
// scene collections (e.g. github.com/mmp/pbrt-v4-scenes) package each scene
// as its own folder holding both the renderable .pbrt file(s) and that
// scene's own geometry/, textures/, bsdfs/ - "one folder per scene" needs
// exactly one level to find those. Descending further would also list a
// fragment's OWN nested asset folders as if they were scenes.
//
// A file that parses cleanly but never declares its own Camera directive is
// excluded - it is a fragment (a materials/geometry library meant only to be
// Include'd from a real scene file), not something renderable on its own.
// Some collections keep these alongside the real scene file(s) rather than
// tucked into a subdirectory, so location alone cannot tell them apart; see
// pbrt_scene::Scene::cameraDeclared.
//
// A file some sibling's own Include/Import directive names is excluded too,
// unconditionally - even when it FAILS to parse standalone. That second
// case is not hypothetical: mmp/pbrt-v4-scenes' barcelona-pavilion and
// contemporary-bathroom each ship a geometry.pbrt with no WorldBegin of its
// own, referencing NamedMaterials declared in a sibling materials.pbrt that
// the real scene file Includes first. Fed to this parser alone, that
// reference is unresolved and the header scan reports a parse error - but
// that is not a gap in this parser relative to real pbrt-v4: pbrt-v4 also
// requires MakeNamedMaterial before the NamedMaterial that names it, in
// Include-expanded processing order (there is no forward-reference
// tolerance in either implementation), so `pbrt geometry.pbrt` would fail
// exactly the same way in real pbrt. The file was never meant to be loaded
// on its own, and provably so - collectIncludedNames() reads that proof
// straight from whichever real scene file names it - so it is excluded
// before its own parse result is even consulted.
//
// Any other file that FAILS to parse still surfaces regardless - hiding a
// parse error in a file nothing else claims is worse than a spurious entry.
inline std::vector<Discovered> scanDirectory(const std::string &dir) {
	std::vector<Discovered> found;
	std::error_code ec;
	if (!std::filesystem::is_directory(dir, ec)) return found;

	// Collected separately, not into one list, so each path can be tagged
	// with which pass found it before everything is sorted back together -
	// see Discovered::nested's comment for what that tag drives.
	std::vector<std::string> topLevelPaths, nestedPaths;
	detail::collectPbrtFiles(dir, topLevelPaths);
	for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
		if (ec) break;
		if (entry.is_directory(ec)) detail::collectPbrtFiles(entry.path(), nestedPaths);
	}

	std::vector<std::pair<std::string, bool>> paths;
	paths.reserve(topLevelPaths.size() + nestedPaths.size());
	for (std::string &p : topLevelPaths) paths.emplace_back(std::move(p), false);
	for (std::string &p : nestedPaths) paths.emplace_back(std::move(p), true);
	std::sort(paths.begin(), paths.end(),
			  [](const auto &a, const auto &b) { return a.first < b.first; });

	// A second raw-text read per file, on top of the one describeFile() does
	// below - acceptable here the same way the rest of this header accepts
	// it: these are small text files, not the geometry/texture payloads this
	// design goes out of its way to avoid touching (see the file's own "WHY
	// A HEADER PARSE RATHER THAN A FULL LOAD" comment).
	std::set<std::string> includedBySomeone;
	for (const auto &[p, isNested] : paths) {
		std::ifstream in(p, std::ios::binary);
		if (!in) continue;
		std::ostringstream ss;
		ss << in.rdbuf();
		detail::collectIncludedNames(ss.str(), includedBySomeone);
	}

	for (const auto &[p, isNested] : paths) {
		if (includedBySomeone.count(detail::basenameOf(p))) continue;
		Discovered d = describeFile(p);
		d.nested = isNested;
		if (d.ok && !d.declaresCamera) continue;
		found.push_back(d);
	}
	return found;
}

// Where to look, in order, for a directory of user scenes.
//
// RAY_TRACER_PBRT_DIR wins so a user can point at a scene collection living
// anywhere without copying it into the install. The relative fallbacks cover
// the two ways this program actually runs: from the repository root during
// development, and from RayTracer_Package/ when launched by the GUI, whose
// working directory is the package rather than the repo.
inline std::vector<std::string> defaultSearchPaths() {
	std::vector<std::string> dirs;
	if (const char *env = std::getenv("RAY_TRACER_PBRT_DIR")) {
		if (*env) dirs.emplace_back(env);
	}
	dirs.emplace_back("pbrt_scenes");
	dirs.emplace_back("../pbrt_scenes");
	dirs.emplace_back("../../pbrt_scenes");
	return dirs;
}

// The first search path that exists and holds at least one .pbrt wins. An
// empty directory is skipped rather than accepted, so a stray empty
// pbrt_scenes/ beside the executable cannot mask a real collection further up.
inline std::vector<Discovered> scanDefaultPaths() {
	for (const std::string &dir : defaultSearchPaths()) {
		std::vector<Discovered> found = scanDirectory(dir);
		if (!found.empty()) return found;
	}
	return {};
}

} // namespace pbrt_discover
