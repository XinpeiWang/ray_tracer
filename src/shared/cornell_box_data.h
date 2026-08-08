#pragma once
// cornell_box_data.h - shared geometry/material data for the classic Cornell
// box scene (scene 0: 5 walls, 2 lights, a rotated white box, a glass
// sphere), consumed by BOTH:
//   - src/TheRestOfYourLife/scenes_book.h::build_cornell_box() (CPU)
//   - gpu/optix/scene_builder.cpp::build_cornell_box() (GPU)
//
// These were previously two independent, hand-copied definitions that
// silently drifted apart - GPU was missing the dim warm accent light CPU
// had (see the CpuGpuLightParityTest that caught it). Editing a wall,
// light, the box, or the glass sphere here now changes both renderers at
// once instead of requiring the same edit twice.
//
// Deliberately NOT a shared builder function: CPU's hittable_list (a tree
// of shared_ptr<hittable> with virtual dispatch) and GPU's SceneData (flat
// POD arrays for CUDA upload) are fundamentally different representations,
// so a single function can't produce both. Plain constexpr data is the
// smallest thing both sides can consume without forcing one renderer's
// object model onto the other.

namespace cornell_box_data {

struct Vec3Spec { double x, y, z; };
struct ColorSpec { double r, g, b; };

// One planar quad: Q = corner, u/v = edge vectors - matches this codebase's
// quad(Q, u, v, material) convention on both CPU and GPU exactly, so each
// side can build its own quad type directly from these fields with no
// reinterpretation. `color` is diffuse albedo for a wall, or emission for
// a light (is_light selects which).
struct QuadSpec {
	Vec3Spec Q, u, v;
	ColorSpec color;
	bool is_light;
};

struct BoxSpec {
	Vec3Spec corner_min, corner_max;
	ColorSpec color;
	double rotate_y_degrees;
	Vec3Spec translate;
};

struct SphereSpec {
	Vec3Spec center;
	double radius;
	double glass_ior;
};

inline constexpr QuadSpec kQuads[] = {
	// Right wall (green), +X face at x=555
	{ {555,0,0}, {0,0,555}, {0,555,0}, {0.12,0.45,0.15}, false },
	// Left wall (red), origin face at x=0
	{ {0,0,555}, {0,0,-555}, {0,555,0}, {0.65,0.05,0.05}, false },
	// Ceiling (white), +Y face at y=555
	{ {0,555,0}, {555,0,0}, {0,0,555}, {0.73,0.73,0.73}, false },
	// Floor (white), XZ plane at y=0
	{ {0,0,555}, {555,0,0}, {0,0,-555}, {0.73,0.73,0.73}, false },
	// Back wall (white), +Z face at z=555
	{ {555,0,555}, {-555,0,0}, {0,555,0}, {0.73,0.73,0.73}, false },
	// Main ceiling light (bright white)
	{ {213,554,227}, {130,0,0}, {0,0,105}, {15,15,15}, true },
	// Secondary accent light embedded in the green wall (dim warm)
	{ {554,100,200}, {0,0,150}, {0,200,0}, {4,2,1}, true },
};
inline constexpr int kNumQuads = sizeof(kQuads) / sizeof(kQuads[0]);

// White box, rotated 15 degrees about Y then translated - matches the book's
// classic Cornell box box() call.
inline constexpr BoxSpec kBox = {
	{0,0,0}, {165,330,165},   // corner_min, corner_max (pre-rotation/translation)
	{0.73,0.73,0.73},         // white, same albedo as the walls
	15.0,                     // rotate_y degrees
	{265,0,295},              // translate
};

inline constexpr SphereSpec kGlassSphere = {
	{190,90,190}, 90.0, 1.5,  // center, radius, dielectric IOR
};

} // namespace cornell_box_data
