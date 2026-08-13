#pragma once
// pbrt_load.h -- loads a .pbrt file from disk into flattened geometry.
//
// The layers below this are pure by design: pbrt_scene.h takes text and a
// FileResolver, pbrt_flatten.h takes a description and a MeshResolver, and
// ply_mesh.h takes bytes. That made them testable without a filesystem, and
// left exactly one thing unimplemented - actually finding the files.
//
// PATHS ARE RELATIVE TO THE SCENE, NOT THE WORKING DIRECTORY
// ----------------------------------------------------------
// This is the whole reason the resolvers were callbacks. A pbrt scene says
// `Include "geometry/killeroo.pbrt"` and `"string filename" "textures/x.png"`
// meaning relative to the scene file itself, so the same scene loads whether
// it is opened from its own directory or from anywhere else. Resolving against
// the process's working directory instead works exactly once - when you happen
// to have cd'd into the scene folder - and fails for every other caller,
// including the GUI, which runs from the package directory.
//
// Nested includes resolve against the file that named them, which matters
// because a scene's geometry/ subdirectory routinely includes its own
// neighbours by bare filename.

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pbrt_flatten.h"
#include "pbrt_scene.h"
#include "ply_mesh.h"

namespace pbrt_load {

struct LoadResult {
	bool ok = false;
	std::string error;
	pbrt_flatten::FlatScene scene;
};

namespace detail {

// Everything up to and including the final separator, or empty for a bare
// filename. Both separators are accepted: pbrt scenes are written on every
// platform and a Windows-authored one carries backslashes.
inline std::string directoryOf(const std::string &path) {
	const std::size_t slash = path.find_last_of("/\\");
	return (slash == std::string::npos) ? std::string() : path.substr(0, slash + 1);
}

inline bool isAbsolute(const std::string &path) {
	if (path.empty()) return false;
	if (path[0] == '/' || path[0] == '\\') return true;
	return path.size() > 1 && path[1] == ':';        // C:\...
}

inline std::string join(const std::string &dir, const std::string &path) {
	return (dir.empty() || isAbsolute(path)) ? path : dir + path;
}

inline bool readFile(const std::string &path, std::string &out) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return false;
	std::ostringstream ss;
	ss << in.rdbuf();
	out = ss.str();
	return true;
}

} // namespace detail

// Reads `path`, resolving Include and plymesh references relative to it.
inline LoadResult loadFile(const std::string &path) {
	using namespace detail;
	LoadResult r;

	std::string text;
	if (!readFile(path, text)) {
		r.error = "cannot open scene file: " + path;
		return r;
	}
	const std::string sceneDir = directoryOf(path);

	// An included file's own includes resolve against ITS directory, not the
	// top-level scene's - tracked by remembering the directory each resolved
	// path came from. Without this, geometry/a.pbrt including "b.pbrt" would
	// be looked for beside the scene rather than beside a.pbrt.
	std::vector<std::pair<std::string, std::string>> resolvedDirs;
	const pbrt_scene::FileResolver files =
		[&](const std::string &want, std::string &contents) {
			// Try the directory of whichever file most recently resolved,
			// then the scene directory, then the path as given.
			for (auto it = resolvedDirs.rbegin(); it != resolvedDirs.rend(); ++it) {
				if (readFile(join(it->second, want), contents)) {
					resolvedDirs.emplace_back(want, directoryOf(join(it->second, want)));
					return true;
				}
			}
			if (readFile(join(sceneDir, want), contents)) {
				resolvedDirs.emplace_back(want, directoryOf(join(sceneDir, want)));
				return true;
			}
			if (readFile(want, contents)) {
				resolvedDirs.emplace_back(want, directoryOf(want));
				return true;
			}
			return false;
		};

	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(text, files);
	if (!parsed.ok) {
		r.error = path + ": " + parsed.error;
		return r;
	}

	const pbrt_flatten::MeshResolver meshes =
		[&sceneDir](const std::string &want, std::vector<float> &positions,
					std::vector<int> &indices) {
			ply_mesh::LoadResult m = ply_mesh::loadFile(join(sceneDir, want));
			if (!m.ok) m = ply_mesh::loadFile(want);   // already absolute, or cwd-relative
			// A scene naming "x.ply" when the folder ships "x.ply.gz" is common
			// enough to be worth one extra open. Without this the failure is a
			// missing-file message about a file that is plainly right there.
			if (!m.ok) m = ply_mesh::loadFile(join(sceneDir, want) + ".gz");
			if (!m.ok) m = ply_mesh::loadFile(want + ".gz");
			if (!m.ok) return false;
			positions = std::move(m.mesh.positions);
			indices = std::move(m.mesh.indices);
			return true;
		};

	r.scene = pbrt_flatten::flatten(parsed.scene, meshes);
	r.ok = true;
	return r;
}

// Reads a file a scene references outside the Include/plymesh graphs this
// header already resolves - currently just a "realistic" camera's lensfile,
// which pbrt_flatten.h can name but has no file access of its own to read.
// Same relative-to-the-scene-first, then as-given convention as everything
// else in this file, so a lens file living beside its scene is found without
// the caller having to know the scene's own directory.
inline bool loadFileNear(const std::string &scenePath, const std::string &relPath,
						 std::string &outContents) {
	using namespace detail;
	const std::string sceneDir = directoryOf(scenePath);
	if (readFile(join(sceneDir, relPath), outContents)) return true;
	return readFile(relPath, outContents);
}

// Parses a pbrt-format lens description table: whitespace-separated
// "radius thickness eta aperture-diameter" rows in millimetres, '#' comments
// to end of line, blank lines ignored - the format pbrt-v4 ships its own
// sample lens files in (e.g. dgauss.22deg.dat), matching what
// RealisticCamera's own lens-table constructor argument already expects, so
// the result can be passed straight through with no further conversion.
// Returns however many complete 4-number rows were found; a malformed file
// that does not divide evenly drops its trailing partial row rather than
// misreading every row after it out of phase.
inline std::vector<double> parseLensFile(const std::string &text) {
	std::vector<double> row;
	std::istringstream in(text);
	std::string line;
	while (std::getline(in, line)) {
		const std::size_t hash = line.find('#');
		if (hash != std::string::npos) line.resize(hash);
		std::istringstream ls(line);
		double v;
		while (ls >> v) row.push_back(v);
	}
	row.resize((row.size() / 4) * 4);
	return row;
}

} // namespace pbrt_load
