#pragma once
// pbrt_asset_check.h -- which files a .pbrt scene refers to that are not on disk
//
// The GUI uses this to warn, before anything is rendered, that a scene flagged "requires
// external files" is missing them and to say where they should be. It is deliberately a
// cheap text scan of the scene file (and the files it Includes), not a parse: selecting a
// scene in the GUI must stay instant even for a multi-megabyte scene description, and all
// that is needed here is the list of paths.
//
// A reference counts as present under the same lookups pbrt_load.h uses: next to the scene
// file, then as given (relative to the working directory), and for a mesh also with ".gz"
// appended to either. Header-only and Qt-free so the renderer's scene-metadata library can
// use it.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace pbrt_asset_check {

struct Result {
	// Referenced paths (as written in the scene) that could not be found. Deduplicated, in
	// the order first met. When the scene file itself is missing this holds just that path.
	std::vector<std::string> missing;
	// The same files as absolute, normalised paths - where each one has to be placed. Parallel to `missing`.
	std::vector<std::string> missingPaths;
	// Distinct files the scene refers to, found or not (the scene file itself not counted).
	int referenced = 0;
	// Absolute, normalised directory of the first missing file - where the user should put
	// the assets. Empty when nothing is missing.
	std::string folder;
};

namespace detail {

inline bool readText(const std::filesystem::path &p, std::string &out) {
	std::ifstream f(p, std::ios::binary);
	if (!f) return false;
	std::ostringstream ss;
	ss << f.rdbuf();
	out = ss.str();
	return true;
}

inline bool existsFile(const std::filesystem::path &p) {
	std::error_code ec;
	return std::filesystem::is_regular_file(p, ec);
}

// Scans `text` for the string values of "string filename" / "string lensfile" parameters
// and for Include/Import directives. Both read as quoted strings, so one pass over the
// quotes finds them: a quoted string that names the parameter, then the quoted value after
// it (possibly inside brackets).
inline void scanText(const std::string &text, std::vector<std::string> &files,
					 std::vector<std::string> &includes) {
	const std::size_t n = text.size();
	std::size_t i = 0;
	auto skipSpace = [&](std::size_t &k) {
		while (k < n && (std::isspace(static_cast<unsigned char>(text[k])) || text[k] == '[')) ++k;
	};
	auto readQuoted = [&](std::size_t &k, std::string &out) {
		if (k >= n || text[k] != '"') return false;
		const std::size_t start = ++k;
		while (k < n && text[k] != '"') ++k;
		if (k >= n) return false;
		out = text.substr(start, k - start);
		++k;
		return true;
	};
	while (i < n) {
		if (text[i] == '#') {   // comment to end of line
			while (i < n && text[i] != '\n') ++i;
			continue;
		}
		if (text[i] != '"') {
			// Include "x" / Import "x": the directive word precedes its quoted path.
			if ((text.compare(i, 7, "Include") == 0 || text.compare(i, 6, "Import") == 0) &&
				(i == 0 || std::isspace(static_cast<unsigned char>(text[i - 1])))) {
				std::size_t k = i + (text[i] == 'I' && text[i + 1] == 'n' ? 7 : 6);
				while (k < n && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
				std::string value;
				if (readQuoted(k, value)) includes.push_back(value);
				i = k;
				continue;
			}
			++i;
			continue;
		}
		std::string token;
		if (!readQuoted(i, token)) break;
		// "string filename" with arbitrary spacing between the two words.
		std::istringstream words(token);
		std::string type, name, extra;
		words >> type >> name >> extra;
		if (type == "string" && extra.empty() && (name == "filename" || name == "lensfile")) {
			std::size_t k = i;
			skipSpace(k);
			std::string value;
			if (readQuoted(k, value)) {
				files.push_back(value);
				i = k;
			}
		}
	}
}

// "model.obj#group" (a Shape naming one group of an OBJ) refers to the file before the "#".
inline std::string fileOf(const std::string &want) { return want.substr(0, want.find('#')); }

inline bool resolves(const std::filesystem::path &sceneDir, const std::string &want, bool mayBeGz) {
	// Not named `near`: that is a macro in the Windows headers (windef.h), which turns this declaration into a syntax error under MSVC.
	const std::filesystem::path inSceneDir = sceneDir / want;
	if (existsFile(inSceneDir) || existsFile(want)) return true;
	if (mayBeGz) return existsFile(inSceneDir.string() + ".gz") || existsFile(want + ".gz");
	return false;
}

// The deepest directory that contains every one of `dirs` - "models" when the missing files are
// models/a.obj and models/textures/b.png, so the warning points at the folder to install
// rather than at whichever subfolder happened to be scanned first.
inline std::filesystem::path commonDirectory(const std::vector<std::filesystem::path> &dirs) {
	std::filesystem::path common = dirs.front();
	for (const std::filesystem::path &d : dirs) {
		std::filesystem::path next;
		auto a = common.begin(), b = d.begin();
		for (; a != common.end() && b != d.end() && *a == *b; ++a, ++b) next /= *a;
		common = next;
	}
	return common;
}

}  // namespace detail

inline Result check(const std::string &scenePath) {
	Result r;
	const std::filesystem::path scene(scenePath);
	const std::filesystem::path sceneDir = scene.parent_path();
	std::error_code ec;

	std::string text;
	if (!detail::readText(scene, text)) {
		r.missing.push_back(scenePath);
		r.missingPaths.push_back(std::filesystem::absolute(scene, ec).lexically_normal().string());
		r.folder = std::filesystem::absolute(sceneDir, ec).lexically_normal().string();
		return r;
	}

	std::set<std::string> seen;
	std::vector<std::filesystem::path> missingDirs;
	const auto noteMissing = [&](const std::string &name) {
		r.missing.push_back(name);
		const std::filesystem::path full = std::filesystem::absolute(sceneDir / name, ec).lexically_normal();
		r.missingPaths.push_back(full.string());
		missingDirs.push_back(full.parent_path());
	};
	std::vector<std::string> pendingTexts{std::move(text)};
	// Includes are followed a few levels (a scene commonly splits geometry into
	// per-object files); the cap also stops an include cycle.
	for (int depth = 0; depth < 4 && !pendingTexts.empty(); ++depth) {
		std::vector<std::string> nextTexts;
		for (const std::string &t : pendingTexts) {
			std::vector<std::string> files, includes;
			detail::scanText(t, files, includes);
			for (const std::string &raw : files) {
				const std::string f = detail::fileOf(raw);
				if (f.empty() || !seen.insert(f).second) continue;
				++r.referenced;
				if (!detail::resolves(sceneDir, f, true)) noteMissing(f);
			}
			for (const std::string &inc : includes) {
				if (!seen.insert("include:" + inc).second) continue;
				std::string incText;
				if (detail::readText(sceneDir / inc, incText) || detail::readText(inc, incText)) {
					nextTexts.push_back(std::move(incText));
				} else {
					++r.referenced;
					noteMissing(inc);
				}
			}
		}
		pendingTexts = std::move(nextTexts);
	}
	if (!missingDirs.empty()) r.folder = detail::commonDirectory(missingDirs).string();
	return r;
}

}  // namespace pbrt_asset_check
