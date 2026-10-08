#pragma once
// scene_document.h -- the document model behind the GUI's Scene Builder: a small, editable description of a scene (a camera, shapes with
// materials, lights) that is written out as a pbrt-v4 scene the renderer already loads, and read back for editing.
//
// Header-only and dependency-free (standard library only), so the Qt GUI, the unit tests and the command line all use the same code.
//
// This header includes the parts, so including it gives all of them:
//   scene_model.h        the data types and their names in files      scene_json.h          the document <-> JSON copy a builder file carries
//   scene_validate.h     the checks shown before rendering            scene_pbrt_writer.h   document -> pbrt text (toPbrt) and back (fromPbrt)
// and below, the ready-made starting points (makeObject, makeStarterScene).
//
// What a builder file is. toPbrt() writes an ordinary .pbrt file whose first lines are comments, one of them a single-line JSON copy of the whole
// document (`# @rt-builder-doc {...}`). Any renderer reads the file as pbrt (the comment is ignored); the Scene Builder reads the JSON back, so the
// scene stays editable. A .pbrt file without that line is not a builder scene (fromPbrt() says so) and is not parsed.
//
// Conventions of the builder's world (so a user never has to think about pbrt's):
//   * +Y is up, units are free (the starter scene is a few units across).
//   * Rotation is three angles in degrees about the world X, Y and Z axes, applied in the order X, then Y, then Z.
//   * A Sphere is centred at its position. Box, Quad, Cylinder, Cone and the generated shapes (Pyramid, Wedge, Stairs, Torus, Capsule, Dome, Tube; see
//     scene_shapes.h) are centred on their position too (their bounding box centre).
//   * A Quad is a flat rectangle in the XZ plane (size.x by size.z) whose front side faces +Y; a Disk faces +Y; a Cylinder and a Cone stand along +Y.
//     An emissive object radiates from its front side (a Box, a Sphere and the generated shapes outward; a Quad or Disk towards +Y, so a ceiling light is rotated 180 degrees about X).
//   * Colours are linear RGB in 0..1 (an emission colour is multiplied by its strength).
//
// pbrt-v4 reference for every directive written here: https://pbrt.org/fileformat-v4

#include <string>

#include "scene_json.h"
#include "scene_model.h"
#include "scene_pbrt_writer.h"
#include "scene_validate.h"

namespace scene_doc {

// ---------------------------------------------------------------------------------------------------------------------------------
// Starting points
// ---------------------------------------------------------------------------------------------------------------------------------
inline Object makeObject(ShapeKind shape, const std::string& name) {
	Object o;
	o.shape = shape;
	o.name = name;
	switch (shape) {
		case ShapeKind::Sphere: o.radius = 1.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Box: o.size = {1.5, 1.5, 1.5}; o.position = {0, 0.75, 0}; break;
		case ShapeKind::Quad: o.size = {6, 1, 6}; o.position = {0, 0, 0}; break;
		case ShapeKind::Disk: o.radius = 1.5; o.position = {0, 0, 0}; break;
		case ShapeKind::Cylinder: o.radius = 0.6; o.height = 2.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Cone: o.radius = 0.8; o.height = 2.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Mesh: o.meshScale = 1.0; break;
		case ShapeKind::Pyramid: o.size = {1.6, 1.6, 1.6}; o.height = 1.8; o.position = {0, 0.9, 0}; break;
		case ShapeKind::Wedge: o.size = {2.0, 1.0, 2.0}; o.position = {0, 0.5, 0}; break;
		case ShapeKind::Stairs: o.size = {2.0, 1.5, 2.5}; o.steps = 5; o.position = {0, 0.75, 0}; break;
		case ShapeKind::Torus: o.radius = 1.0; o.radius2 = 0.3; o.position = {0, 0.3, 0}; break;
		case ShapeKind::Capsule: o.radius = 0.5; o.height = 2.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Dome: o.radius = 1.2; o.position = {0, 0.6, 0}; break;
		case ShapeKind::Tube: o.radius = 0.7; o.radius2 = 0.5; o.height = 1.6; o.position = {0, 0.8, 0}; break;
	}
	return o;
}

// A light panel: a quad that faces down, to hang above a scene.
inline Object makeAreaLightPanel(const std::string& name) {
	Object o = makeObject(ShapeKind::Quad, name);
	o.size = {2.0, 1.0, 2.0};
	o.position = {0, 5.0, 0};
	o.rotation = {180.0, 0.0, 0.0};
	o.emissive = true;
	o.emission = {1.0, 1.0, 1.0};
	o.emissionStrength = 12.0;
	o.material.color = {0.0, 0.0, 0.0};
	return o;
}

// What "New scene" opens: a checkered floor, a glass sphere, a gold sphere, a red box, a panel light and a faint sky: it renders to something nice at once.
inline Document makeStarterScene() {
	Document d;
	d.title = "My scene";

	Object floorObj = makeObject(ShapeKind::Quad, "Floor");
	floorObj.size = {14, 1, 14};
	floorObj.material.checker = true;
	floorObj.material.color = {0.82, 0.82, 0.82};
	floorObj.material.color2 = {0.25, 0.27, 0.30};
	floorObj.material.checkerCount = 10;
	d.objects.push_back(floorObj);

	Object glass = makeObject(ShapeKind::Sphere, "Glass ball");
	glass.position = {-1.3, 1.0, 0.4};
	glass.material.kind = MaterialKind::Dielectric;
	d.objects.push_back(glass);

	Object gold = makeObject(ShapeKind::Sphere, "Gold ball");
	gold.radius = 0.8;
	gold.position = {1.4, 0.8, -0.3};
	gold.material.kind = MaterialKind::Conductor;
	gold.material.color = {0.95, 0.72, 0.30};
	gold.material.roughness = 0.12;
	d.objects.push_back(gold);

	Object box = makeObject(ShapeKind::Box, "Red box");
	box.size = {1.0, 1.0, 1.0};
	box.position = {0.1, 0.5, -1.8};
	box.rotation = {0, 28, 0};
	box.material.color = {0.75, 0.12, 0.10};
	d.objects.push_back(box);

	d.objects.push_back(makeAreaLightPanel("Ceiling light"));

	Light sky;
	sky.name = "Sky";
	sky.kind = LightKind::Infinite;
	sky.color = {0.55, 0.70, 1.0};
	sky.intensity = 0.35;
	d.lights.push_back(sky);
	return d;
}

}  // namespace scene_doc
