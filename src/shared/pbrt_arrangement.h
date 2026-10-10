#pragma once
// pbrt_arrangement.h -- write a pbrt scene file with some of its objects moved: the Live Preview's "Save arrangement". The original text is kept as it is; each
// Shape of a moved object is wrapped as
//     AttributeBegin  Translate <t>  <the Shape directive>  AttributeEnd
// so everything else (materials, lights, other shapes, comments) stays byte for byte, and nothing is re-written from the flattened scene (which has lost the
// parameters). A Translate placed first in the new block is applied in the frame the Shape's own transform is in, so to move the shape by a WORLD offset d the
// translation written is t = A^-1 d, A being the linear part of the transform in force at the Shape (live_objects::translateObject moves the flattened scene
// by d in world space; the unit tests flatten both and compare). Standard library only.
//
// A scene made by the Scene Builder carries its document in a "# @rt-builder-doc" comment, which the Builder reads when it opens the file (the directives are ignored). The
// moves are written into that document too (each moved object's position changes by the world offset: the Builder writes Translate <position> first in every object's block,
// so that is exactly the same move), so the Builder opens the arrangement, not the original. That needs the document's objects and the scene's objects to be the same list
// (one block per object, in order); when they are not, the marker is taken out of the saved file so the Builder does not open a stale copy.
//
// A scene saved somewhere else would lose the files it names by relative path, so any "string ..." parameter or Include whose value is a relative path to an existing
// file is rewritten to the absolute path.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "live_object_edit.h"
#include "pbrt_scene.h"
#include "scene_document.h"   // scene_doc::Document, fromPbrt/toPbrt (a Scene Builder scene carries its document in a comment)

namespace pbrt_arrangement {

struct Result {
	std::string text;
	int movedShapes = 0;     // shapes wrapped in a Translate
	int skippedShapes = 0;   // shapes of moved objects that sit in an Include'd file, which stays as it is
	int rewrittenPaths = 0;  // relative file names made absolute
	bool builderDocumentUpdated = false;   // a Scene Builder scene: its embedded document now has the moved positions
	bool builderDocumentDropped = false;   // a Scene Builder scene whose document does not line up with the scene's objects: the marker was removed
};

namespace detail {

// Solves A t = d for the linear part of a row-major 4x4 transform. False for a singular one.
inline bool solveLinear(const double* m, const double d[3], double t[3]) {
	const double a = m[0], b = m[1], c = m[2], e = m[4], f = m[5], g = m[6], h = m[8], i = m[9], j = m[10];
	const double det = a * (f * j - g * i) - b * (e * j - g * h) + c * (e * i - f * h);
	if (std::fabs(det) < 1e-12) return false;
	t[0] = (d[0] * (f * j - g * i) - b * (d[1] * j - g * d[2]) + c * (d[1] * i - f * d[2])) / det;
	t[1] = (a * (d[1] * j - g * d[2]) - d[0] * (e * j - g * h) + c * (e * d[2] - d[1] * h)) / det;
	t[2] = (a * (f * d[2] - d[1] * i) - b * (e * d[2] - d[1] * h) + d[0] * (e * i - f * h)) / det;
	return true;
}

struct Event {
	std::size_t pos = 0, resume = 0;   // text before `pos` is copied, then `text`, then copying resumes at `resume` (> pos for a replacement)
	int order = 0;                     // at one position: a closing insertion, then a replacement, then an opening insertion
	std::string text;
};

inline std::string number(double v) {
	char buf[40];
	std::snprintf(buf, sizeof buf, "%.17g", v);
	return buf;
}

inline bool isAbsolutePath(const std::string& p) {
	return !p.empty() && (p[0] == '/' || p[0] == '\\' || (p.size() > 1 && p[1] == ':'));
}

}  // namespace detail

// `ranges` and `objects` are the scene's (live_objects::objectsOf), `offsets[o]` is how far object o moves in world space. `originalDir` is the folder of the original
// file with a trailing separator; `fileExists` says whether a path names a readable file; `header` is a comment put at the top.
inline Result write(const std::string& text, const std::vector<pbrt_flatten::ShapeRange>& ranges, const live_objects::ObjectList& objects,
                    const std::vector<std::array<double, 3>>& offsets, const std::string& originalDir,
                    const std::function<bool(const std::string&)>& fileExists, const std::string& header) {
	Result result;
	std::vector<detail::Event> events;
	for (std::size_t o = 0; o < objects.size() && o < offsets.size(); ++o) {
		const std::array<double, 3>& d = offsets[o];
		if (d[0] == 0.0 && d[1] == 0.0 && d[2] == 0.0) continue;
		for (std::size_t ri : objects[o]) {
			if (ri >= ranges.size()) continue;
			const pbrt_flatten::ShapeRange& r = ranges[ri];
			if (r.srcFile != 0 || r.srcEnd <= r.srcBegin || r.srcEnd > text.size()) { ++result.skippedShapes; continue; }
			double t[3];
			if (!detail::solveLinear(r.ctm, d.data(), t)) { t[0] = d[0]; t[1] = d[1]; t[2] = d[2]; }
			detail::Event open{r.srcBegin, r.srcBegin, 2, "\nAttributeBegin\nTranslate " + detail::number(t[0]) + " " + detail::number(t[1]) + " " + detail::number(t[2]) + "\n"};
			detail::Event close{r.srcEnd, r.srcEnd, 0, "\nAttributeEnd\n"};
			events.push_back(open);
			events.push_back(close);
			++result.movedShapes;
		}
	}
	// A Scene Builder scene: keep its embedded document in step with the directives.
	if (result.movedShapes > 0) {
		const std::string marker = "# @rt-builder-doc ";
		std::size_t at = text.compare(0, marker.size(), marker) == 0 ? 0 : text.find("\n" + marker);
		if (at != std::string::npos && at != 0) ++at;
		if (at != std::string::npos) {
			const std::size_t jsonBegin = at + marker.size();
			std::size_t jsonEnd = text.find_first_of("\r\n", jsonBegin);
			if (jsonEnd == std::string::npos) jsonEnd = text.size();
			scene_doc::Document doc;
			std::string err;
			if (scene_doc::fromJson(text.substr(jsonBegin, jsonEnd - jsonBegin), doc, err) && doc.objects.size() == objects.size()) {
				for (std::size_t o = 0; o < objects.size() && o < offsets.size(); ++o) {
					doc.objects[o].position.x += offsets[o][0];
					doc.objects[o].position.y += offsets[o][1];
					doc.objects[o].position.z += offsets[o][2];
				}
				events.push_back({jsonBegin, jsonEnd, 1, scene_doc::toJson(doc)});
				result.builderDocumentUpdated = true;
			} else {
				events.push_back({at, jsonBegin, 1, "# (arranged in Live Preview: the Scene Builder cannot edit this copy) "});
				result.builderDocumentDropped = true;
			}
		}
	}
	// Relative file names: where a "string <name>" parameter or an Include names an existing file next to the original, name it absolutely.
	std::vector<pbrt_scene::detail::Token> tokens;
	pbrt_scene::detail::tokenizeInto(text, 0, tokens);
	for (std::size_t i = 1; i < tokens.size(); ++i) {
		const pbrt_scene::detail::Token& value = tokens[i];
		const pbrt_scene::detail::Token& before = tokens[i - 1];
		const bool isFileValue = value.quoted && ((before.quoted && before.text.compare(0, 7, "string ") == 0) || (!before.quoted && (before.text == "Include" || before.text == "Import")));
		if (!isFileValue || value.text.empty() || detail::isAbsolutePath(value.text)) continue;
		const std::string absolute = originalDir + value.text;
		if (!fileExists(absolute)) continue;
		events.push_back({value.begin, value.end, 1, "\"" + absolute + "\""});
		++result.rewrittenPaths;
	}
	std::stable_sort(events.begin(), events.end(), [](const detail::Event& a, const detail::Event& b) { return a.pos != b.pos ? a.pos < b.pos : a.order < b.order; });
	std::string out = header;
	std::size_t cursor = 0;
	for (const detail::Event& e : events) {
		out.append(text, cursor, e.pos - cursor);
		out += e.text;
		cursor = e.resume;
	}
	out.append(text, cursor, std::string::npos);
	result.text = std::move(out);
	return result;
}

}  // namespace pbrt_arrangement
