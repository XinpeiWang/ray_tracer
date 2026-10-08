#pragma once
// scene_validate.h -- the checks the Scene Builder shows before rendering: what makes a scene unrenderable, and what is only worth a note.
//
// Part of the Scene Builder's document (scene_document.h includes all of it); header-only, standard library only.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "scene_model.h"
#include "scene_shapes.h"

namespace scene_doc {

namespace detail {
// An absolute path to a file that is not there. (A relative path is relative to wherever the .pbrt file ends up, which validate() cannot know;
// the Scene Builder's file dialogs only ever store absolute ones.)
inline bool absoluteFileMissing(const std::string& path) {
	if (path.empty()) return false;
	std::error_code ec;
	const std::filesystem::path p(path);
	return p.is_absolute() && !std::filesystem::exists(p, ec);
}
}  // namespace detail

inline std::vector<Problem> validate(const Document& d) {
	std::vector<Problem> out;
	auto err = [&](const std::string& m) { out.push_back({Problem::Severity::Error, m}); };
	auto warn = [&](const std::string& m) { out.push_back({Problem::Severity::Warning, m}); };

	const Float3 dc{d.camera.target.x - d.camera.position.x, d.camera.target.y - d.camera.position.y, d.camera.target.z - d.camera.position.z};
	const double len = std::sqrt(dc.x * dc.x + dc.y * dc.y + dc.z * dc.z);
	if (len < 1e-6) err("The camera is at its target: move one of them.");
	else {
		const double upLen = std::sqrt(d.camera.up.x * d.camera.up.x + d.camera.up.y * d.camera.up.y + d.camera.up.z * d.camera.up.z);
		if (upLen < 1e-9) err("The camera's up direction is zero.");
		else {
			const double cx = dc.y * d.camera.up.z - dc.z * d.camera.up.y, cy = dc.z * d.camera.up.x - dc.x * d.camera.up.z,
			             cz = dc.x * d.camera.up.y - dc.y * d.camera.up.x;
			if (std::sqrt(cx * cx + cy * cy + cz * cz) < 1e-6 * len * upLen) err("The camera looks straight along its up direction: tilt it a little.");
		}
	}
	if (!(d.camera.fov >= 1.0 && d.camera.fov <= 170.0)) err("The field of view must be between 1 and 170 degrees.");
	if (d.camera.lensRadius < 0.0) err("The lens radius cannot be negative.");
	if (d.camera.lensRadius > 0.0 && !(d.camera.focusDistance > 0.0)) err("A camera with depth of field needs a focus distance above zero.");
	if (d.render.width < 1 || d.render.height < 1 || d.render.width > 16384 || d.render.height > 16384) err("The image size must be between 1 and 16384 pixels each way.");
	if (d.render.samples < 1) err("At least one sample per pixel is needed.");
	if (d.render.maxDepth < 1 || d.render.maxDepth > 100) err("The maximum depth must be between 1 and 100.");

	bool anyLight = !d.lights.empty();
	for (const Object& o : d.objects) {
		const std::string who = "'" + o.name + "': ";
		if (o.emissive) anyLight = true;
		switch (o.shape) {
			case ShapeKind::Sphere: if (!(o.radius > 0.0)) err(who + "the radius must be above zero."); break;
			case ShapeKind::Disk: if (!(o.radius > 0.0)) err(who + "the radius must be above zero."); break;
			case ShapeKind::Cylinder:
			case ShapeKind::Cone:
				if (!(o.radius > 0.0) || !(o.height > 0.0)) err(who + "the radius and height must be above zero.");
				break;
			case ShapeKind::Box: if (!(o.size.x > 0.0 && o.size.y > 0.0 && o.size.z > 0.0)) err(who + "every box size must be above zero."); break;
			case ShapeKind::Wedge:
			case ShapeKind::Stairs:
				if (!(o.size.x > 0.0 && o.size.y > 0.0 && o.size.z > 0.0)) err(who + "every size must be above zero.");
				if (o.shape == ShapeKind::Stairs && (o.steps < 1 || o.steps > kMaxStairSteps)) err(who + "the number of steps must be between 1 and " + std::to_string(kMaxStairSteps) + ".");
				break;
			case ShapeKind::Pyramid:
				if (!(o.size.x > 0.0 && o.size.z > 0.0 && o.height > 0.0)) err(who + "the pyramid's base and height must be above zero.");
				break;
			case ShapeKind::Dome: if (!(o.radius > 0.0)) err(who + "the radius must be above zero."); break;
			case ShapeKind::Capsule:
				if (!(o.radius > 0.0) || !(o.height > 0.0)) err(who + "the radius and height must be above zero.");
				else if (o.height < 2 * o.radius) warn(who + "the capsule is shorter than its two rounded ends, so it is as tall as they need (twice the radius).");
				break;
			case ShapeKind::Torus:
				if (!(o.radius > 0.0) || !(o.radius2 > 0.0)) err(who + "the ring and tube radii must be above zero.");
				else if (o.radius2 >= o.radius) err(who + "the tube must be thinner than the ring's radius, or the ring folds into itself.");
				break;
			case ShapeKind::Tube:
				if (!(o.radius > 0.0) || !(o.height > 0.0) || !(o.radius2 > 0.0)) err(who + "the radii and height must be above zero.");
				else if (o.radius2 >= o.radius) err(who + "the hole must be narrower than the outside.");
				break;
			case ShapeKind::Quad: if (!(o.size.x > 0.0 && o.size.z > 0.0)) err(who + "the quad's width and depth must be above zero."); break;
			case ShapeKind::Mesh:
				if (o.meshFile.empty()) err(who + "choose a mesh file (.ply or .obj).");
				if (!(o.meshScale > 0.0)) err(who + "the mesh scale must be above zero.");
				break;
		}
		if (o.emissive && o.emissionStrength <= 0.0) warn(who + "it is an area light with no strength, so it gives no light.");
		if (!o.material.imageFile.empty()) {
			if (o.material.kind != MaterialKind::Diffuse && o.material.kind != MaterialKind::CoatedDiffuse)
				warn(who + "a picture only applies to diffuse and glossy paint materials.");
			else if (o.material.checker) warn(who + "the picture replaces the checker pattern.");
			if (detail::absoluteFileMissing(o.material.imageFile))
				warn(who + "the picture file was not found (" + o.material.imageFile + "), so it will render without it. Choose it again.");
		}
		if (o.shape == ShapeKind::Mesh && detail::absoluteFileMissing(o.meshFile))
			warn(who + "the mesh file was not found (" + o.meshFile + "). Choose it again.");
		if (o.material.checker && o.material.kind != MaterialKind::Diffuse) warn(who + "the checker pattern only applies to diffuse materials.");
		if (o.material.checker && o.shape == ShapeKind::Box) warn(who + "the checker pattern is not applied to boxes.");
		if ((o.material.kind == MaterialKind::Dielectric || o.material.kind == MaterialKind::CoatedDiffuse) && !(o.material.ior >= 1.0 && o.material.ior <= 3.0))
			warn(who + "an index of refraction outside 1 to 3 is unusual.");
		if (o.material.roughness < 0.0 || o.material.roughness > 1.0) err(who + "the roughness must be between 0 and 1.");
		if (o.material.kind == MaterialKind::Diffuse || o.material.kind == MaterialKind::Conductor || o.material.kind == MaterialKind::CoatedDiffuse ||
		    o.material.kind == MaterialKind::DiffuseTransmission) {
			auto bad = [](const Rgb& c) { return c.r < 0 || c.g < 0 || c.b < 0 || c.r > 1 || c.g > 1 || c.b > 1; };
			if (bad(o.material.color)) warn(who + "a reflectance colour outside 0 to 1 gives more light out than in.");
		}
	}
	for (const Light& l : d.lights) {
		const std::string who = "'" + l.name + "': ";
		if (l.intensity < 0.0) err(who + "the light strength cannot be negative.");
		if (l.kind == LightKind::Infinite && detail::absoluteFileMissing(l.imageFile)) warn(who + "the sky image was not found (" + l.imageFile + "). Choose it again.");
		if (l.kind == LightKind::Spot && !(l.coneAngle > 0.0 && l.coneAngle < 90.0)) err(who + "the spot cone angle must be between 0 and 90 degrees.");
		if (l.kind == LightKind::Spot && (l.coneDelta < 0.0 || l.coneDelta > l.coneAngle)) err(who + "the spot soft edge must be between 0 and the cone angle.");
		if ((l.kind == LightKind::Spot || l.kind == LightKind::Distant) && l.position.x == l.target.x && l.position.y == l.target.y && l.position.z == l.target.z)
			err(who + "the light is at the point it aims at.");
	}
	if (!anyLight) warn("The scene has no light, so the picture will be black. Add a light or make an object emissive.");
	return out;
}

inline bool hasErrors(const std::vector<Problem>& ps) {
	return std::any_of(ps.begin(), ps.end(), [](const Problem& p) { return p.severity == Problem::Severity::Error; });
}

}  // namespace scene_doc
