#pragma once
// scene_model.h -- the document's data types (camera, materials, objects, lights, render settings) and their names in files.
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

#include "scene_shapes.h"

namespace scene_doc {

struct Float3 {
	double x = 0, y = 0, z = 0;
};

struct Rgb {
	double r = 0.8, g = 0.8, b = 0.8;
};

enum class MaterialKind { Diffuse, Conductor, Dielectric, CoatedDiffuse, DiffuseTransmission };

struct Material {
	MaterialKind kind = MaterialKind::Diffuse;
	Rgb color{0.8, 0.8, 0.8};          // diffuse / coated reflectance, conductor reflectance; unused by Dielectric
	bool checker = false;              // Diffuse only: a checkerboard of `color` and `color2`
	Rgb color2{0.2, 0.2, 0.2};
	double checkerCount = 8.0;         // checks across the object (its width), for Quad / Disk / Sphere
	double roughness = 0.0;            // Conductor, Dielectric, CoatedDiffuse: 0 = smooth
	double ior = 1.5;                  // Dielectric and CoatedDiffuse
	Rgb transmittance{0.5, 0.5, 0.5};  // DiffuseTransmission (its reflectance is `color`)
	std::string imageFile;             // Diffuse / CoatedDiffuse: a picture used as the surface colour (replaces `color` and the checker)
};

enum class ShapeKind { Sphere, Box, Quad, Disk, Cylinder, Cone, Mesh, Pyramid, Wedge, Stairs, Torus, Capsule, Dome, Tube };

// Every shape, in the order the GUI lists them.
inline const std::vector<ShapeKind>& allShapeKinds() {
	static const std::vector<ShapeKind> all = {ShapeKind::Sphere,  ShapeKind::Box,   ShapeKind::Quad,    ShapeKind::Disk,  ShapeKind::Cylinder,
	                                           ShapeKind::Cone,    ShapeKind::Pyramid, ShapeKind::Wedge, ShapeKind::Stairs, ShapeKind::Torus,
	                                           ShapeKind::Capsule, ShapeKind::Dome,  ShapeKind::Tube,    ShapeKind::Mesh};
	return all;
}
// A shape that is written as a generated triangle mesh (scene_shapes.h) rather than one of pbrt's own shapes.
inline bool isGeneratedShape(ShapeKind k) {
	return k == ShapeKind::Pyramid || k == ShapeKind::Wedge || k == ShapeKind::Stairs || k == ShapeKind::Torus || k == ShapeKind::Capsule ||
	       k == ShapeKind::Dome || k == ShapeKind::Tube;
}

struct Object {
	std::string name = "Object";
	ShapeKind shape = ShapeKind::Sphere;
	Float3 position{};
	Float3 rotation{};                   // degrees, about world X, Y, Z (applied in that order)
	double radius = 1.0;               // Sphere, Disk, Cylinder, Cone, Capsule, Dome; Torus: the ring; Tube: outside
	double radius2 = 0.25;             // Torus: the tube's radius; Tube: the hole's radius
	double height = 1.0;               // Cylinder, Cone, Pyramid, Capsule, Tube
	Float3 size{1.0, 1.0, 1.0};          // Box, Wedge, Stairs (x, y, z); Quad (x and z); Pyramid (x and z, the base)
	int steps = 5;                     // Stairs
	double meshScale = 1.0;            // Mesh
	std::string meshFile;              // Mesh: a .ply file
	Material material;
	bool emissive = false;             // an area light
	Rgb emission{1.0, 1.0, 1.0};
	double emissionStrength = 10.0;    // radiance = emission * strength
	bool twoSided = false;             // emits from both sides
};

// The radius of a sphere around the object's position that holds all of it, whatever its rotation (a mesh, whose size the document does not know, counts as its
// scale). Used for the scene's size (scene_pbrt_writer.h documentSize) and for spacing copies (scene_array.h).
inline double boundingRadius(const Object& o) {
	double r = 0.5;
	switch (o.shape) {
		case ShapeKind::Sphere: r = o.radius; break;
		case ShapeKind::Box:
		case ShapeKind::Wedge:
		case ShapeKind::Stairs: r = 0.5 * std::sqrt(o.size.x * o.size.x + o.size.y * o.size.y + o.size.z * o.size.z); break;
		case ShapeKind::Quad: r = 0.5 * std::sqrt(o.size.x * o.size.x + o.size.z * o.size.z); break;
		case ShapeKind::Pyramid: r = 0.5 * std::sqrt(o.size.x * o.size.x + o.height * o.height + o.size.z * o.size.z); break;
		case ShapeKind::Mesh: r = std::fabs(o.meshScale); break;
		default: r = std::sqrt((o.radius + o.radius2) * (o.radius + o.radius2) + o.height * o.height); break;   // disk, cylinder, cone, capsule, dome, torus, tube
	}
	return std::fabs(r);
}

enum class LightKind { Point, Spot, Distant, Infinite };

// The Sun & sky option of an Infinite light (scene_sky.h): when `physicalSky` is on, the light's picture is a generated clear sky for this sun, and the scene's
// Sun (a Distant light) follows it. Angles in degrees; azimuth 0 is +X, 90 is -Z, 180 is -X, 270 is +Z.
struct SkyParams {
	double sunElevation = 35.0;   // above the horizon
	double sunAzimuth = 250.0;
	double turbidity = 3.0;       // haze: 1.7 very clear, 3 ordinary, 6+ hazy
	double groundAlbedo = 0.3;    // the ground the sky light bounces off
};

struct Light {
	std::string name = "Light";
	LightKind kind = LightKind::Point;
	Float3 position{0.0, 5.0, 0.0};      // Point, Spot (and Distant: where it shines from)
	Float3 target{0.0, 0.0, 0.0};        // Spot, Distant: what it points at
	Rgb color{1.0, 1.0, 1.0};
	double intensity = 50.0;           // Point/Spot: radiant intensity; Distant/Infinite: radiance
	double coneAngle = 30.0;           // Spot, degrees
	double coneDelta = 5.0;            // Spot, degrees of soft edge
	std::string imageFile;             // Infinite: an equirectangular image (empty: a constant colour)
	bool physicalSky = false;          // Infinite: imageFile is the generated sky of `sky` (scene_sky.h)
	SkyParams sky;
};

struct Camera {
	Float3 position{0.0, 2.5, 8.0};
	Float3 target{0.0, 1.0, 0.0};
	Float3 up{0.0, 1.0, 0.0};
	double fov = 40.0;                 // vertical field of view, degrees
	double lensRadius = 0.0;           // 0 = a pinhole; > 0 blurs what is not at focusDistance
	double focusDistance = 8.0;
};

struct RenderSettings {
	int width = 800;
	int height = 600;
	int samples = 64;
	int maxDepth = 8;
};

struct Document {
	std::string title = "Untitled scene";
	Camera camera;
	RenderSettings render;
	std::vector<Object> objects;
	std::vector<Light> lights;
};

struct Problem {
	enum class Severity { Warning, Error } severity = Severity::Warning;
	std::string message;
};

// ---------------------------------------------------------------------------------------------------------------------------------
// Names <-> enums (also what the JSON stores)
// ---------------------------------------------------------------------------------------------------------------------------------
inline const char* toString(MaterialKind k) {
	switch (k) {
		case MaterialKind::Diffuse: return "diffuse";
		case MaterialKind::Conductor: return "conductor";
		case MaterialKind::Dielectric: return "dielectric";
		case MaterialKind::CoatedDiffuse: return "coateddiffuse";
		case MaterialKind::DiffuseTransmission: return "diffusetransmission";
	}
	return "diffuse";
}
inline bool materialKindFromString(const std::string& s, MaterialKind& out) {
	for (MaterialKind k : {MaterialKind::Diffuse, MaterialKind::Conductor, MaterialKind::Dielectric, MaterialKind::CoatedDiffuse,
	                       MaterialKind::DiffuseTransmission})
		if (s == toString(k)) { out = k; return true; }
	return false;
}
inline const char* toString(ShapeKind k) {
	switch (k) {
		case ShapeKind::Sphere: return "sphere";
		case ShapeKind::Box: return "box";
		case ShapeKind::Quad: return "quad";
		case ShapeKind::Disk: return "disk";
		case ShapeKind::Cylinder: return "cylinder";
		case ShapeKind::Cone: return "cone";
		case ShapeKind::Mesh: return "mesh";
		case ShapeKind::Pyramid: return "pyramid";
		case ShapeKind::Wedge: return "wedge";
		case ShapeKind::Stairs: return "stairs";
		case ShapeKind::Torus: return "torus";
		case ShapeKind::Capsule: return "capsule";
		case ShapeKind::Dome: return "dome";
		case ShapeKind::Tube: return "tube";
	}
	return "sphere";
}
inline bool shapeKindFromString(const std::string& s, ShapeKind& out) {
	for (ShapeKind k : allShapeKinds())
		if (s == toString(k)) { out = k; return true; }
	return false;
}
inline const char* toString(LightKind k) {
	switch (k) {
		case LightKind::Point: return "point";
		case LightKind::Spot: return "spot";
		case LightKind::Distant: return "distant";
		case LightKind::Infinite: return "infinite";
	}
	return "point";
}
inline bool lightKindFromString(const std::string& s, LightKind& out) {
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite})
		if (s == toString(k)) { out = k; return true; }
	return false;
}

}  // namespace scene_doc
