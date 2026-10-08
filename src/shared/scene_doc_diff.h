#pragma once
// scene_doc_diff.h -- a one-line description of what changed between two versions of a Scene Builder document, for the GUI's log ("object 'Ball': position,
// radius; +1 object (Table top)"). std-only.

#include "scene_document.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace scene_doc {

namespace diff_detail {

using Fields = std::vector<std::pair<std::string, std::string>>;

inline std::string v(double x) { std::ostringstream s; s.imbue(std::locale::classic()); s << x; return s.str(); }
inline std::string v(const Float3& p) { return v(p.x) + "," + v(p.y) + "," + v(p.z); }
inline std::string v(const Rgb& c) { return v(c.r) + "," + v(c.g) + "," + v(c.b); }

inline Fields fieldsOf(const Object& o) {
	return {{"name", o.name}, {"shape", toString(o.shape)}, {"position", v(o.position)}, {"rotation", v(o.rotation)}, {"radius", v(o.radius)},
	        {"radius2", v(o.radius2)}, {"height", v(o.height)}, {"size", v(o.size)}, {"steps", std::to_string(o.steps)}, {"meshScale", v(o.meshScale)},
	        {"meshFile", o.meshFile}, {"material", toString(o.material.kind)}, {"color", v(o.material.color)}, {"checker", o.material.checker ? "1" : "0"},
	        {"color2", v(o.material.color2)}, {"checkerCount", v(o.material.checkerCount)}, {"roughness", v(o.material.roughness)}, {"ior", v(o.material.ior)},
	        {"transmittance", v(o.material.transmittance)}, {"picture", o.material.imageFile}, {"emissive", o.emissive ? "1" : "0"}, {"emission", v(o.emission)},
	        {"strength", v(o.emissionStrength)}, {"twoSided", o.twoSided ? "1" : "0"}};
}
inline Fields fieldsOf(const Light& l) {
	return {{"name", l.name}, {"kind", toString(l.kind)}, {"position", v(l.position)}, {"target", v(l.target)}, {"color", v(l.color)},
	        {"intensity", v(l.intensity)}, {"coneAngle", v(l.coneAngle)}, {"coneDelta", v(l.coneDelta)}, {"image", l.imageFile}};
}
inline Fields fieldsOf(const Camera& c) {
	return {{"position", v(c.position)}, {"target", v(c.target)}, {"up", v(c.up)}, {"fov", v(c.fov)}, {"lensRadius", v(c.lensRadius)}, {"focusDistance", v(c.focusDistance)}};
}
inline Fields fieldsOf(const RenderSettings& r) {
	return {{"width", std::to_string(r.width)}, {"height", std::to_string(r.height)}, {"samples", std::to_string(r.samples)}, {"maxDepth", std::to_string(r.maxDepth)}};
}

// "a,b,c" naming the fields that differ ("position 0,1,0 -> 2,1,0" when there is just one, which is the common drag/spin-box case).
inline std::string changedFields(const Fields& a, const Fields& b) {
	std::string names;
	std::size_t count = 0;
	std::string last;
	for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
		if (a[i].second == b[i].second) continue;
		names += (names.empty() ? "" : ",") + a[i].first;
		last = a[i].first + " " + a[i].second + " -> " + b[i].second;
		++count;
	}
	if (count == 1 && last.size() < 120) return last;
	return names;
}

template <typename T>
void listChanges(std::vector<std::string>& parts, const char* what, const std::vector<T>& a, const std::vector<T>& b) {
	if (a.size() != b.size()) {
		std::string names;
		const std::vector<T>& bigger = a.size() < b.size() ? b : a;
		const std::vector<T>& smaller = a.size() < b.size() ? a : b;
		const std::size_t count = bigger.size() - smaller.size();
		// The items that came or went are the first ones where the two lists stop agreeing (a deleted middle item is not the last one).
		std::size_t from = 0;
		while (from < smaller.size() && fieldsOf(smaller[from]) == fieldsOf(bigger[from])) ++from;
		for (std::size_t i = from; i < from + count && i < bigger.size(); ++i) names += (names.empty() ? "" : ", ") + bigger[i].name;
		parts.push_back(std::string(a.size() < b.size() ? "+" : "-") + std::to_string(count) + " " + what + (count == 1 ? "" : "s") + " (" + names + ")");
		return;
	}
	for (std::size_t i = 0; i < a.size(); ++i) {
		const std::string changes = changedFields(fieldsOf(a[i]), fieldsOf(b[i]));
		if (!changes.empty()) parts.push_back(std::string(what) + " '" + b[i].name + "': " + changes);
	}
}

}  // namespace diff_detail

// Empty when nothing a user can see differs. Long lists of changes are cut after a few.
inline std::string describeChange(const Document& a, const Document& b) {
	using namespace diff_detail;
	std::vector<std::string> parts;
	if (a.title != b.title) parts.push_back("title '" + a.title + "' -> '" + b.title + "'");
	if (const std::string c = changedFields(fieldsOf(a.camera), fieldsOf(b.camera)); !c.empty()) parts.push_back("camera: " + c);
	if (const std::string c = changedFields(fieldsOf(a.render), fieldsOf(b.render)); !c.empty()) parts.push_back("render: " + c);
	listChanges(parts, "object", a.objects, b.objects);
	listChanges(parts, "light", a.lights, b.lights);
	std::string out;
	for (std::size_t i = 0; i < parts.size() && i < 6; ++i) out += (out.empty() ? "" : "; ") + parts[i];
	if (parts.size() > 6) out += "; ... (+" + std::to_string(parts.size() - 6) + " more)";
	return out;
}

}  // namespace scene_doc
