#pragma once
// scene_props.h -- the Scene Builder's ready-made props: a table, a chair, a tree and so on, each a handful of ordinary objects (boxes, cylinders, spheres, cones)
// placed relative to the prop's own origin, which is the middle of its footprint on the floor. Adding one adds those objects to the document; from then on they
// are plain objects that can be moved, recoloured or deleted one by one, and nothing in the saved file says they came from a prop.
//
// std-only, like scene_document.h.

#include "scene_document.h"

#include <string>
#include <vector>

namespace scene_doc {

enum class PropKind { Table, Chair, Tree, Snowman, Column, StreetLamp };

inline const std::vector<PropKind>& allPropKinds() {
	static const std::vector<PropKind> all = {PropKind::Table, PropKind::Chair, PropKind::Tree, PropKind::Snowman, PropKind::Column, PropKind::StreetLamp};
	return all;
}

inline const char* toString(PropKind k) {
	switch (k) {
		case PropKind::Table: return "table";
		case PropKind::Chair: return "chair";
		case PropKind::Tree: return "tree";
		case PropKind::Snowman: return "snowman";
		case PropKind::Column: return "column";
		case PropKind::StreetLamp: return "street lamp";
	}
	return "table";
}

namespace props_detail {

inline Object part(const std::string& name, ShapeKind shape, Float3 position, Rgb color) {
	Object o = makeObject(shape, name);
	o.position = position;
	o.material.color = color;
	return o;
}
inline Object box(const std::string& name, Float3 position, Float3 size, Rgb color) {
	Object o = part(name, ShapeKind::Box, position, color);
	o.size = size;
	return o;
}
inline Object cylinder(const std::string& name, Float3 position, double radius, double height, Rgb color) {
	Object o = part(name, ShapeKind::Cylinder, position, color);
	o.radius = radius;
	o.height = height;
	return o;
}
inline Object cone(const std::string& name, Float3 position, double radius, double height, Rgb color) {
	Object o = part(name, ShapeKind::Cone, position, color);
	o.radius = radius;
	o.height = height;
	return o;
}
inline Object sphere(const std::string& name, Float3 position, double radius, Rgb color) {
	Object o = part(name, ShapeKind::Sphere, position, color);
	o.radius = radius;
	return o;
}

}  // namespace props_detail

// The objects of one prop, named "<Prop> <part>", standing on the floor (y = 0) with the middle of the footprint at the origin.
inline std::vector<Object> makeProp(PropKind kind) {
	using namespace props_detail;
	const Rgb wood{0.55, 0.36, 0.2};
	std::vector<Object> parts;
	switch (kind) {
		case PropKind::Table: {
			parts.push_back(box("Table top", {0, 0.86, 0}, {2.0, 0.12, 1.2}, wood));
			int n = 1;
			for (double sx : {-1.0, 1.0})
				for (double sz : {-1.0, 1.0}) parts.push_back(cylinder("Table leg " + std::to_string(n++), {sx * 0.88, 0.4, sz * 0.48}, 0.06, 0.8, wood));
			break;
		}
		case PropKind::Chair: {
			parts.push_back(box("Chair seat", {0, 0.47, 0}, {0.6, 0.08, 0.6}, wood));
			parts.push_back(box("Chair back", {0, 0.91, -0.26}, {0.6, 0.8, 0.08}, wood));
			int n = 1;
			for (double sx : {-1.0, 1.0})
				for (double sz : {-1.0, 1.0}) parts.push_back(cylinder("Chair leg " + std::to_string(n++), {sx * 0.25, 0.215, sz * 0.25}, 0.04, 0.43, wood));
			break;
		}
		case PropKind::Tree: {
			parts.push_back(cylinder("Tree trunk", {0, 0.6, 0}, 0.2, 1.2, Rgb{0.4, 0.26, 0.14}));
			parts.push_back(cone("Tree foliage low", {0, 2.0, 0}, 1.1, 1.6, Rgb{0.12, 0.45, 0.15}));
			parts.push_back(cone("Tree foliage high", {0, 2.9, 0}, 0.8, 1.4, Rgb{0.14, 0.5, 0.18}));
			break;
		}
		case PropKind::Snowman: {
			const Rgb snow{0.92, 0.94, 0.97};
			parts.push_back(sphere("Snowman body", {0, 0.7, 0}, 0.7, snow));
			parts.push_back(sphere("Snowman middle", {0, 1.7, 0}, 0.5, snow));
			parts.push_back(sphere("Snowman head", {0, 2.4, 0}, 0.33, snow));
			Object nose = cone("Snowman nose", {0, 2.4, 0.4}, 0.06, 0.3, Rgb{0.9, 0.4, 0.05});
			nose.rotation = {90, 0, 0};   // points the cone along +Z, out of the face
			parts.push_back(nose);
			parts.push_back(sphere("Snowman left eye", {-0.11, 2.5, 0.29}, 0.04, Rgb{0.02, 0.02, 0.02}));
			parts.push_back(sphere("Snowman right eye", {0.11, 2.5, 0.29}, 0.04, Rgb{0.02, 0.02, 0.02}));
			break;
		}
		case PropKind::Column: {
			const Rgb stone{0.78, 0.76, 0.72};
			parts.push_back(box("Column base", {0, 0.1, 0}, {1.1, 0.2, 1.1}, stone));
			parts.push_back(cylinder("Column shaft", {0, 1.7, 0}, 0.4, 3.0, stone));
			parts.push_back(box("Column top", {0, 3.3, 0}, {1.1, 0.2, 1.1}, stone));
			break;
		}
		case PropKind::StreetLamp: {
			const Rgb iron{0.12, 0.12, 0.14};
			parts.push_back(cylinder("Lamp base", {0, 0.1, 0}, 0.2, 0.2, iron));
			parts.push_back(cylinder("Lamp pole", {0, 1.6, 0}, 0.05, 3.0, iron));
			Object bulb = sphere("Lamp bulb", {0, 3.25, 0}, 0.22, Rgb{1.0, 0.9, 0.7});
			bulb.emissive = true;
			bulb.emission = {1.0, 0.85, 0.6};
			bulb.emissionStrength = 12.0;
			parts.push_back(bulb);
			break;
		}
	}
	return parts;
}

}  // namespace scene_doc
