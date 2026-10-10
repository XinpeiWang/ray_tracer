#pragma once
// scene_camera_views.h -- saved cameras for the Scene Builder: name where the camera is, and come back to it later (a front view, a close-up, a bird's-eye
// shot), like Blender's saved camera markers. A view is a whole camera (position, what it looks at, field of view, lens radius, focus distance), kept in the
// document's JSON copy and nowhere else: the .pbrt text has just the scene's one camera.
//
// Every function changes the Document it is given and does nothing else (no selection, no undo): the GUI wraps one call in one undo step. std-only, like
// scene_document.h.

#include "scene_document.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace scene_doc {

constexpr std::size_t kMaxCameraViews = 100;

namespace camera_views_detail {

inline std::string trimmed(const std::string& s) {
	const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
	std::size_t a = 0, b = s.size();
	while (a < b && isSpace(static_cast<unsigned char>(s[a]))) ++a;
	while (b > a && isSpace(static_cast<unsigned char>(s[b - 1]))) --b;
	return s.substr(a, b - a);
}

inline bool nameTaken(const Document& d, const std::string& name, int except) {
	for (std::size_t i = 0; i < d.cameraViews.size(); ++i)
		if (static_cast<int>(i) != except && d.cameraViews[i].name == name) return true;
	return false;
}

}  // namespace camera_views_detail

// `wanted` (or "View" if it is blank) when no other view has that name, else "View 2", "View 3"... `except` is a view being renamed (its own name does not count).
inline std::string uniqueViewName(const Document& d, const std::string& wanted, int except = -1) {
	std::string base = camera_views_detail::trimmed(wanted);
	if (base.empty()) base = "View";
	if (!camera_views_detail::nameTaken(d, base, except)) return base;
	// "Front 2" asks for the next free number after "Front", not "Front 2 2".
	std::string stem = base;
	std::size_t end = stem.size();
	while (end > 0 && std::isdigit(static_cast<unsigned char>(stem[end - 1]))) --end;
	if (end < stem.size() && end > 1 && stem[end - 1] == ' ') stem.erase(end - 1);
	for (int n = 2;; ++n) {
		const std::string candidate = stem + " " + std::to_string(n);
		if (!camera_views_detail::nameTaken(d, candidate, except)) return candidate;
	}
}

// Saves `camera` as a new view named from `wanted`; returns its index, or -1 when there are already kMaxCameraViews.
inline int saveCameraView(Document& d, const std::string& wanted, const Camera& camera) {
	if (d.cameraViews.size() >= kMaxCameraViews) return -1;
	d.cameraViews.push_back(CameraView{uniqueViewName(d, wanted), camera});
	return static_cast<int>(d.cameraViews.size()) - 1;
}

// The scene's camera becomes the saved one. False for a bad index.
inline bool applyCameraView(Document& d, int index) {
	if (index < 0 || index >= static_cast<int>(d.cameraViews.size())) return false;
	d.camera = d.cameraViews[static_cast<std::size_t>(index)].camera;
	return true;
}

// The saved view takes `camera` (its name stays). False for a bad index.
inline bool updateCameraView(Document& d, int index, const Camera& camera) {
	if (index < 0 || index >= static_cast<int>(d.cameraViews.size())) return false;
	d.cameraViews[static_cast<std::size_t>(index)].camera = camera;
	return true;
}

inline bool removeCameraView(Document& d, int index) {
	if (index < 0 || index >= static_cast<int>(d.cameraViews.size())) return false;
	d.cameraViews.erase(d.cameraViews.begin() + index);
	return true;
}

// Renames a view; the name is trimmed, and numbered if another view has it. A blank name is refused (false), and so is a bad index.
inline bool renameCameraView(Document& d, int index, const std::string& wanted) {
	if (index < 0 || index >= static_cast<int>(d.cameraViews.size()) || camera_views_detail::trimmed(wanted).empty()) return false;
	d.cameraViews[static_cast<std::size_t>(index)].name = uniqueViewName(d, wanted, index);
	return true;
}

// `base` moved to look from `eye` at `target`: the lens, field of view and up stay. The focus distance follows what is looked at while the camera has no depth of
// field (lens radius 0) and is left alone otherwise (the user set it on purpose).
inline Camera cameraLookingFrom(const Camera& base, const Float3& eye, const Float3& target) {
	Camera c = base;
	c.position = eye;
	c.target = target;
	if (c.lensRadius <= 0.0) {
		const double dx = target.x - eye.x, dy = target.y - eye.y, dz = target.z - eye.z;
		const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (dist > 1e-6) c.focusDistance = dist;
	}
	return c;
}

}  // namespace scene_doc
