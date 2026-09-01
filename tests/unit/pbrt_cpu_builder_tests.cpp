/**
 * @file pbrt_cpu_builder_tests.cpp
 * @brief Unit tests for building CPU hittables from a flattened pbrt scene
 *
 * These fire actual rays at the built world rather than inspecting the object
 * graph. Counting primitives proves objects were created; hitting them proves
 * they were created in the right place, which is what every earlier stage in
 * the chain exists to get right.
 */

#include <gtest/gtest.h>

#include "pbrt_cpu_builder.h"
#include "pbrt_flatten.h"
#include "pbrt_load.h"
#include "pbrt_scene.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

pbrt_cpu::BuildResult buildFrom(const std::string &text) {
	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(text);
	EXPECT_TRUE(parsed.ok) << parsed.error;
	return pbrt_cpu::build(pbrt_flatten::flatten(parsed.scene));
}

// Casts a ray and reports whether it hit, and how far along.
bool castRay(const pbrt_cpu::BuildResult &b, const point3 &origin,
			 const vec3 &direction, double &tOut) {
	hit_record rec;
	const ray r(origin, direction);
	if (!b.world->hit(r, interval(0.001, infinity), rec)) return false;
	tOut = rec.t;
	return true;
}

// A unit quad in the z=0 plane spanning (0,0)..(1,1).
const char *kQuad =
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
	"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n";

// Hand-authored minimal 1x1 24bpp BMP - same technique as pbrt_alpha_cutout_
// tests.cpp's own solidBmp1x1() (duplicated, not shared - small self-
// contained test-file helpers). A uniform-gray pixel decodes as a genuine
// grayscale bump/displacement map (is_grayscale_image()'s own convention).
std::string solidGrayBmp1x1() {
	std::string bytes;
	const auto u16 = [&](unsigned short v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
	};
	const auto u32 = [&](unsigned int v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 16) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 24) & 0xFF));
	};
	bytes.push_back('B'); bytes.push_back('M');
	u32(14 + 40 + 4);
	u32(0);
	u32(14 + 40);
	u32(40);
	u32(1);
	u32(1);
	u16(1);
	u16(24);
	u32(0);
	u32(4);
	u32(0); u32(0);
	u32(0); u32(0);
	bytes.push_back(static_cast<char>(128));
	bytes.push_back(static_cast<char>(128));
	bytes.push_back(static_cast<char>(128));
	bytes.push_back('\0');
	return bytes;
}

// Hand-authored minimal 2x2 24bpp BMP, two-tone (white/black by row) rather
// than solidGrayBmp1x1()'s single uniform pixel - needed to prove a
// goniometric/projection light's real decoded image (not the flat
// kUniformImage/kUniformSlide fallback) is what reached eval_I/eval_I_rgb:
// a uniform image gives identical intensity at every direction by
// construction, so seeing ANY variance across sampled directions is only
// possible with real, non-uniform pixel data actually decoded and used.
std::string twoToneBmp2x2() {
	std::string bytes;
	const auto u16 = [&](unsigned short v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
	};
	const auto u32 = [&](unsigned int v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 16) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 24) & 0xFF));
	};
	const std::size_t rowBytes = 8;  // 2 px * 3 bytes, padded to a 4-byte boundary
	const std::size_t dataSize = rowBytes * 2;
	bytes.push_back('B'); bytes.push_back('M');
	u32(static_cast<unsigned int>(14 + 40 + dataSize));
	u32(0);
	u32(14 + 40);
	u32(40);
	u32(2);
	u32(2);
	u16(1);
	u16(24);
	u32(0);
	u32(static_cast<unsigned int>(dataSize));
	u32(0); u32(0);
	u32(0); u32(0);
	// Bottom row first (BMP storage order) - black; top row - white.
	// (B,G,R) byte order per pixel, but black/white are channel-symmetric so
	// this doesn't matter here.
	for (int row = 0; row < 2; ++row) {
		const unsigned char v = (row == 0) ? 0 : 255;
		for (int px = 0; px < 2; ++px) {
			bytes.push_back(static_cast<char>(v));
			bytes.push_back(static_cast<char>(v));
			bytes.push_back(static_cast<char>(v));
		}
		bytes.push_back('\0'); bytes.push_back('\0');  // row padding to 8 bytes
	}
	return bytes;
}

class CpuBuilderTempTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_cpu_builder_tests/";
		std::string cmd = "if not exist \"" + root_ + "\" mkdir \"" + root_ + "\" >nul 2>&1";
		for (char &c : cmd) if (c == '/') c = '\\';
		std::system(cmd.c_str());
	}
	void TearDown() override {
		for (const std::string &f : written_) std::remove(f.c_str());
	}
	void write(const std::string &relative, const std::string &contents) {
		const std::string full = root_ + relative;
		std::ofstream out(full, std::ios::binary);
		out << contents;
		out.close();
		written_.push_back(full);
	}
	std::string path(const std::string &relative) const { return root_ + relative; }
private:
	std::string root_;
	std::vector<std::string> written_;
};

} // namespace

TEST(PbrtCpuBuildTest, GeometryIsWhereTheSceneSaysItIs) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_EQ(b.triangleCount, 2u);

	double t = 0.0;
	ASSERT_TRUE(castRay(b, point3(0.5, 0.5, -5), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 5.0, 1e-6);

	EXPECT_FALSE(castRay(b, point3(9, 9, -5), vec3(0, 0, 1), t))
		<< "a ray well outside the quad must miss";
}

TEST(PbrtCpuBuildTest, TranslationInTheSceneMovesTheGeometry) {
	const pbrt_cpu::BuildResult b = buildFrom(std::string("Translate 0 0 10\n") + kQuad);
	double t = 0.0;
	ASSERT_TRUE(castRay(b, point3(0.5, 0.5, -5), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 15.0, 1e-6) << "the quad moved 10 further away";
}

TEST(PbrtCpuBuildTest, SphereIsBuiltAtItsTransformedCentreAndRadius) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 0 10\n"
		"Scale 2 2 2\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	// Centre at z=10, radius 2, so the near surface sits at z=8.
	ASSERT_TRUE(castRay(b, point3(0, 0, 0), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 8.0, 1e-6);
}

TEST(PbrtCpuBuildTest, SphereZMaxClipsAwayTheTopCap) {
	// A horizontal ray through the removed cap band (z=0.9, above zmax=0.5)
	// crosses where the full sphere's surface would be (both roots land at
	// z=0.9 since the ray never changes z) but the clipped sphere has no
	// surface left there at all.
	const pbrt_cpu::BuildResult full = buildFrom(
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	double t = 0.0;
	ASSERT_TRUE(castRay(full, point3(-5, 0, 0.9), vec3(1, 0, 0), t))
		<< "sanity check: the full sphere's belt at z=0.9 is really there";

	const pbrt_cpu::BuildResult clipped = buildFrom(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float zmax\" [ 0.5 ]\n");
	EXPECT_FALSE(castRay(clipped, point3(-5, 0, 0.9), vec3(1, 0, 0), t))
		<< "zmax=0.5 removes the whole z=0.9 band, so this ray should miss entirely";
}

TEST(PbrtCpuBuildTest, SphereZMinFallsThroughToTheFarRootWhenTheNearOneIsClipped) {
	// zmin=0 keeps only the upper hemisphere. A ray straight up through the
	// centre hits the (clipped-away) bottom pole first on a full sphere, but
	// must fall through to the (kept) top pole on the clipped one.
	const pbrt_cpu::BuildResult full = buildFrom(
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	double tFull = 0.0;
	ASSERT_TRUE(castRay(full, point3(0, 0, -5), vec3(0, 0, 1), tFull));
	EXPECT_NEAR(tFull, 4.0, 1e-6) << "hits the near (bottom) pole at z=-1";

	const pbrt_cpu::BuildResult clipped = buildFrom(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ]\n");
	double tClipped = 0.0;
	ASSERT_TRUE(castRay(clipped, point3(0, 0, -5), vec3(0, 0, 1), tClipped));
	EXPECT_NEAR(tClipped, 6.0, 1e-6)
		<< "bottom pole is clipped away, so this must fall through to the top pole at z=1";
}

TEST(PbrtCpuBuildTest, SpherePhiMaxClipsAnAzimuthalWedgeFallingThroughToTheFarRoot) {
	// phimax=90 keeps only phi in [0,90] degrees (pbrt-v4's atan2(y,x)
	// convention). A ray through the centre along X hits phi=180 (rejected)
	// on the near side and phi=0 (kept, boundary-inclusive) on the far side.
	const pbrt_cpu::BuildResult clipped = buildFrom(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float phimax\" [ 90 ]\n");
	double t = 0.0;
	ASSERT_TRUE(castRay(clipped, point3(-5, 0, 0), vec3(1, 0, 0), t));
	EXPECT_NEAR(t, 6.0, 1e-6)
		<< "phi=180 near root clipped away, falls through to phi=0 at x=1";
}

TEST(PbrtCpuBuildTest, ClippedSphereUnderRotationHitsAtTheExactTransformedPosition) {
	// Mirrors DiskUnderRotationHitsAtTheExactTransformedPosition below: a
	// clipped sphere is no longer rotation-invariant (see pbrt_flatten::
	// Sphere's own comment), so this proves the real object-to-world
	// transform - not just the plain baked centre/radius approximation - is
	// what the clipped path actually uses.
	//
	// zmin=0 keeps local z>=0 (the "north" hemisphere). Rotate 90 degrees
	// about X maps local (0,0,1) (kept) to world (0,-1,0), and local
	// (0,0,-1) (clipped away) to world (0,1,0).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Rotate 90 1 0 0\n"
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ]\n");
	double t = 0.0;
	// A ray from world (0,5,0) toward -Y hits world (0,1,0) first - the
	// image of the CLIPPED-away south pole - so it must fall through to the
	// far root at world (0,-1,0), the image of the kept north pole.
	ASSERT_TRUE(castRay(b, point3(0, 5, 0), vec3(0, -1, 0), t));
	EXPECT_NEAR(t, 6.0, 1e-6)
		<< "if rotation weren't applied to the clip test, this would either "
		   "hit at t=4 (clip ignored) or miss entirely (wrong hemisphere clipped)";
}

TEST(PbrtCpuBuildTest, ClippedSphereDpdvDoesNotVanishAtThePole) {
	// phimax=90 clips azimuthally only, leaving both poles reachable - a ray
	// straight down onto the north pole exercises sphere_clipped_hittable's
	// degenerate-pole fallback. Only dpdu has a genuine coordinate
	// singularity there (like longitude lines converging at a globe's
	// pole); dpdv (a line of latitude/theta) has a well-defined, nonzero
	// length-rate right at the pole and must not collapse toward zero.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float phimax\" [ 90 ]\n");
	const ray r(point3(0, 0, 5), vec3(0, 0, -1));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(r, interval(0.001, infinity), rec));
	EXPECT_NEAR(rec.t, 4.0, 1e-6) << "sanity check: this ray really lands on the pole";
	EXPECT_GT(rec.dpdv.length(), 0.5)
		<< "dpdv should have magnitude ~(thetaZMax-thetaZMin)*radius, not vanish";
}

TEST(PbrtCpuBuildTest, ClippedSphereSampleAreaLandsOnlyOnTheVisibleCap) {
	// zmin=0 keeps only the upper (z>=0) hemisphere. sample_area() is used
	// directly as a light EMISSION point (not re-intersected), so a sample
	// landing on the clipped-away z<0 region would be a real bias under
	// --bdpt/--sppm, not just wasted noise.
	//
	// b.lights (unlike b.world, which build() wraps in a bvh_node - see
	// pbrt_cpu_builder.h's own comment on that) stays a flat, unaccelerated
	// list, exactly matching how bdpt_adapter.h/sppm_adapter.h's own
	// real emitter-scan loops reach individual shapes' sample_area()
	// directly - so this needs a real AreaLightSource to land there at all.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"diffuse\" \"rgb reflectance\" [ 0 0 0 ]\n"
		"AreaLightSource \"diffuse\" \"rgb L\" [ 1 1 1 ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ]\n");
	ASSERT_EQ(b.lights->objects.size(), 1u);
	for (double u1 = 0.0; u1 <= 1.0; u1 += 0.1) {
		for (double u2 = 0.0; u2 <= 1.0; u2 += 0.1) {
			AreaLightSample s;
			ASSERT_TRUE(b.lights->objects[0]->sample_area(u1, u2, s));
			EXPECT_GE(s.p.z(), -1e-9)
				<< "u1=" << u1 << " u2=" << u2 << " landed outside the kept z>=0 hemisphere";
		}
	}
	AreaLightSample lo, hi;
	ASSERT_TRUE(b.lights->objects[0]->sample_area(0.0, 0.0, lo));
	ASSERT_TRUE(b.lights->objects[0]->sample_area(1.0, 0.0, hi));
	EXPECT_NEAR(lo.p.z(), 0.0, 1e-6) << "u1=0 should land at zmin";
	EXPECT_NEAR(hi.p.z(), 1.0, 1e-6) << "u1=1 should land at zmax";
}

TEST(PbrtCpuBuildTest, ClippedSphereWithMediumInterfaceSkipsTheMediumWrapperButStaysReachable) {
	// Like MediumInterfaceWrapsTheSphereInAParticipatingMedium below, the
	// precise structural claim (s.medium stays the real index for GPU;
	// s.cpuMediumUnsupported gates CPU's own wrapper) lives in
	// pbrt_flatten_tests.cpp - this only confirms pbrt_cpu_builder.h's
	// cpuMediumUnsupported check doesn't skip building/registering the
	// clipped sphere itself, just its constant_medium wrapper.
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.1 0.1 0.1 ] \"rgb sigma_s\" [ 2 2 2 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t))
		<< "the clipped sphere itself must still be there, medium wrapper or not";
}

TEST(PbrtCpuBuildTest, DiskIsBuiltAtItsTransformedPositionAndRadius) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 0 10\n"
		"Scale 2 2 2\n"
		"Shape \"disk\" \"float radius\" [ 1 ]\n");
	EXPECT_EQ(b.diskCount, 1u);
	double t = 0.0;
	// A disk at object-space z=0, radius 1, scaled by 2 and moved to z=10:
	// world-space disk of radius 2 centred at (0,0,10), still facing +Z.
	ASSERT_TRUE(castRay(b, point3(0, 0, 0), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 10.0, 1e-6);
}

TEST(PbrtCpuBuildTest, CylinderIsBuiltAtItsTransformedPositionAndRadius) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 10 0 0\n"
		"Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ -1 ] "
		"\"float zmax\" [ 1 ]\n");
	EXPECT_EQ(b.cylinderCount, 1u);
	double t = 0.0;
	// The cylinder's axis (object-space Z) moves to run through x=10, y=0.
	// A ray fired along +X hits the near wall at x=10-radius=9.
	ASSERT_TRUE(castRay(b, point3(0, 0, 0), vec3(1, 0, 0), t));
	EXPECT_NEAR(t, 9.0, 1e-6);
}

TEST(PbrtCpuBuildTest, CylinderHasNoEndCaps) {
	// pbrt's cylinder (and this project's CylinderShape<T> port) is an open
	// tube, not a capped can - matching DiskShape/CylinderShape's own
	// intersect() which only solves for the lateral surface. A ray travelling
	// straight down the axis must pass through untouched.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ -1 ] "
		"\"float zmax\" [ 1 ]\n");
	double t = 0.0;
	EXPECT_FALSE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t))
		<< "a ray down the cylinder's own axis should exit through the open "
		   "end, not bounce off a cap that doesn't exist";
}

TEST(PbrtCpuBuildTest, DiskUnderRotationHitsAtTheExactTransformedPosition) {
	// Disk/Cylinder deliberately keep their CTM unbaked (see
	// disk_cylinder_hittable.h's own header comment) instead of following
	// Sphere's "bake to world-space centre+radius, warn under anisotropic
	// scale" approximation - Sphere can get away with that because it's
	// rotation-invariant, but a disk's orientation is exactly what a rotation
	// changes. This proves a rotated disk is hit exactly, not approximated.
	//
	// Rotating 90 degrees about X leaves the local X axis fixed and swaps
	// local Y and Z, so a disk originally spanning the object-space XY plane
	// (its outward normal along +/-Z) ends up spanning the world-space XZ
	// plane instead - regardless of the handedness convention for the
	// rotation's sign, since X is untouched either way.
	const pbrt_cpu::BuildResult b =
		buildFrom("Rotate 90 1 0 0\nShape \"disk\" \"float radius\" [ 1 ]\n");
	EXPECT_EQ(b.diskCount, 1u);
	double t = 0.0;
	// Local point (x=0.5, z=0) is within the unit disk (radius 0.5 < 1) and
	// keeps x=0.5 under the rotation, landing in the world-space y=0 plane.
	ASSERT_TRUE(castRay(b, point3(0.5, -5, 0), vec3(0, 1, 0), t));
	EXPECT_NEAR(t, 5.0, 1e-6);
}

TEST(PbrtCpuBuildTest, CurveIsBuiltAtItsTransformedPositionAndWidth) {
	// A straight cylinder-type curve along world-space Z, width0=width1=0.2
	// (a "thickened 1D curve" - like pbrt-v4's own Curve, CurveShape's hit
	// distance is where the ray crosses the curve's own centerline, not a
	// true swept-cylinder surface offset by radius; width only gates the
	// transverse-distance ACCEPTANCE threshold - see CurveShape::intersect's
	// hitWidth/halfW comment, shapes.h). This mirrors curve_shape_tests.cpp's
	// own FlatHitCentre/FlatMissOffset tests (t = exact axis-crossing
	// distance; width changes hit/miss at an offset, not t itself).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 3 0 0\n"
		"Shape \"curve\" \"string type\" [ \"cylinder\" ] \"float width\" [ 0.2 ]\n"
		"  \"point3 P\" [ 0 0 0  0 0 1  0 0 2  0 0 3 ]\n");
	EXPECT_EQ(b.curveCount, 1u);
	double t = 0.0;
	// Curve's world-space axis runs through x=3, y=0; a ray from x=-2 toward
	// +X aimed exactly at the axis (y=0) hits at the axis-crossing distance.
	ASSERT_TRUE(castRay(b, point3(-2, 0, 1.5), vec3(1, 0, 0), t));
	EXPECT_NEAR(t, 5.0, 1e-3);

	// A ray offset by less than the half-width (0.1) still hits...
	ASSERT_TRUE(castRay(b, point3(-2, 0.05, 1.5), vec3(1, 0, 0), t));
	// ...but one offset well past the half-width misses - proving the
	// authored width (not some other default) is what's actually reaching
	// CurveShape, the one thing the axis-crossing hit above doesn't prove.
	EXPECT_FALSE(castRay(b, point3(-2, 0.5, 1.5), vec3(1, 0, 0), t));
}

TEST(PbrtCpuBuildTest, CurveWithInvalidControlPointCountIsNotBuilt) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Shape \"curve\"\n"
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0  4 0 0 ]\n");
	EXPECT_EQ(b.curveCount, 0u);
}

TEST(PbrtCpuBuildTest, HairMaterialSphereIsReachable) {
	// hair_material's scatter() is stochastic (Marschner lobe sampling) and
	// can reject a sample, so this only confirms the built sphere is hit
	// geometrically - the same "did it build and can a ray find it" bar
	// CloudMediumIsReachable/UniformgridMediumIsReachable use below. The
	// closed-form sigma_a math itself (direct/eumelanin-pheomelanin/default)
	// is covered by pbrt_flatten_tests.cpp's own Hair tests.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"hair\" \"float eumelanin\" [ 1.0 ] \"float pheomelanin\" [ 0.3 ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	double t = 0.0;
	ASSERT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 4.0, 1e-6);
}

TEST(PbrtCpuBuildTest, MediumInterfaceWrapsTheSphereInAParticipatingMedium) {
	// The precise structural claim - the sphere resolves to the right
	// FlatScene::media index, with the right coefficients - lives in
	// pbrt_flatten_tests.cpp, which can check it before pbrt_cpu_builder.h's
	// own final BVH-folding step collapses every top-level hittable (sphere,
	// constant_medium wrapper, everything else) into a single bvh_node,
	// making world->objects.size() always 1 regardless of scene content and
	// unusable as a "how many things got added" signal here. This test only
	// confirms the CPU builder actually consumes that index without
	// crashing and the geometry stays reachable.
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.1 0.1 0.1 ] \"rgb sigma_s\" [ 2 2 2 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);

	// constant_medium stochastically decides whether a ray scatters inside
	// the volume, so this doesn't assert a specific outcome - only that the
	// medium-wrapped sphere is still reachable at all, the same "did it
	// build and can a ray find it" bar castRay() checks for plain geometry.
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t))
		<< "a ray toward the medium-wrapped sphere should still hit something";
}

TEST(PbrtCpuBuildTest, CameraMediumIsBuiltFromTheResolvedIndex) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMedium \"haze\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.01 0.01 0.01 ] \"rgb sigma_s\" [ 0.04 0.04 0.04 ]\n"
		"MediumInterface \"\" \"haze\"\n"
		"Camera \"perspective\" \"float fov\" [ 40 ]\n"
		"WorldBegin\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	ASSERT_TRUE(b.cameraMedium != nullptr);
	// sample_scatter() should eventually intercept a ray traveling toward
	// infinity (no real surface in front of it) - sigma_t=0.05 makes this
	// happen with overwhelming probability well before t=1e6.
	ray r(point3(0, 0, 0), vec3(0, 0, -1));
	hit_record rec;
	EXPECT_TRUE(b.cameraMedium->sample_scatter(r, 1e6, rec))
		<< "an unbounded medium with real extinction should (almost) always "
		   "scatter a long ray before it escapes";
}

TEST(PbrtCpuBuildTest, NoCameraMediumWithoutMediumInterfaceBeforeCamera) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Camera \"perspective\" \"float fov\" [ 40 ]\n"
		"WorldBegin\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	EXPECT_TRUE(b.cameraMedium == nullptr);
}

TEST(PbrtCpuBuildTest, EmissiveMediumWrapsTheSphereAndStaysReachable) {
	// "rgb Le"/"float Lescale" - same "builds, geometry stays reachable"
	// bar as MediumInterfaceWrapsTheSphereInAParticipatingMedium above; the
	// precise Le/Lescale resolution lives in pbrt_flatten_tests.cpp.
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMedium \"fire\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.5 0.5 0.5 ] \"rgb sigma_s\" [ 0.5 0.5 0.5 ]\n"
		"  \"rgb Le\" [ 5 2 0.5 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fire\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);

	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t))
		<< "a ray toward the emissive-medium-wrapped sphere should still hit something";
}

TEST(PbrtCpuBuildTest, EmissiveMediumBakesLeWeightedBySigmaAOverSigmaT) {
	// Unlike the reachability test above, this deterministically verifies
	// the ACTUAL emission value addMediumIfPresent (pbrt_cpu_builder.h)
	// computes and bakes into hg_phase_material - not just that something
	// built. sigma_a/sigma_s are astronomically large (not a realistic
	// scene value) so free_path (Beer-Lambert-sampled) is essentially
	// always far shorter than the sphere's own diameter - constant_medium's
	// stochastic collision test (hit(), constant_medium.h) resolves to a
	// real collision on effectively every ray, sidestepping the
	// "stochastic, don't assert specifics" caveat every other medium test
	// in this file carries (P(no collision) = exp(-sigma_t*diameter) =
	// exp(-2000) - not just unlikely, unrepresentable in a double).
	// Expected: sigma_a=800, sigma_s=200 (sigma_t=1000), Le=[5,2,0.5],
	// Lescale=1 -> emission = Le * (sigma_a/sigma_t) = Le * 0.8
	// = [4, 1.6, 0.4].
	// Material "none" (Interface) is essential here, not cosmetic - with no
	// Material directive the sphere gets pbrt's own default (opaque)
	// surface material, and the ray would hit THAT solid boundary first
	// (rec.mat = the default surface material, emitted() = black) rather
	// than ever reaching the medium's own internal stochastic collision -
	// exactly the scene-authoring pitfall documented in docs/PBRT_SUPPORT.md
	// (a medium boundary material must be near-invisible, not opaque).
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMedium \"fire\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 800 800 800 ] \"rgb sigma_s\" [ 200 200 200 ]\n"
		"  \"rgb Le\" [ 5 2 0.5 ]\n"
		"AttributeBegin\n"
		"  Material \"none\"\n"
		"  MediumInterface \"fire\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(b.sphereCount, 1u);

	// addMediumIfPresent (pbrt_cpu_builder.h) unconditionally adds BOTH the
	// raw sphere (its own hittable, carrying the Interface material) AND a
	// SEPARATE constant_medium wrapping the same sphere as its boundary -
	// so a single hit() call always finds the raw sphere's own (closer, at
	// the boundary entry t) surface first, same as the real path tracer's
	// first bounce would. The real integrator (camera.h) recognizes
	// is_medium_boundary() and CONTINUES the same ray from just past that
	// point for its next bounce, which is when it actually reaches the
	// medium's own internal stochastic collision - mirrored here with a
	// second hit() call starting just past the first hit, rather than
	// asserting on the first hit_record directly.
	const ray r(point3(0, 0, -5), vec3(0, 0, 1));
	hit_record entryRec;
	ASSERT_TRUE(b.world->hit(r, interval(0.001, infinity), entryRec))
		<< "a ray toward the emissive-medium-wrapped sphere should hit its boundary";
	ASSERT_TRUE(entryRec.mat != nullptr);
	ASSERT_TRUE(entryRec.mat->is_medium_boundary())
		<< "the boundary hit should be the Interface material, not an opaque surface";

	hit_record rec;
	ASSERT_TRUE(b.world->hit(r, interval(entryRec.t + 1e-4, infinity), rec))
		<< "continuing past the boundary should reach the medium's own internal collision";
	ASSERT_TRUE(rec.mat != nullptr);

	const color Le = rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);
	EXPECT_NEAR(Le.x(), 4.0, 1e-9);
	EXPECT_NEAR(Le.y(), 1.6, 1e-9);
	EXPECT_NEAR(Le.z(), 0.4, 1e-9);
}

TEST(PbrtCpuBuildTest, UnknownMediumInterfaceNameIsTreatedAsVacuum) {
	// No MakeNamedMedium declares "ghost" - the parser should warn and fall
	// back to insideMedium=-1 (vacuum) rather than crash or misindex. See
	// FlattenTest.UnresolvedMediumNameIsVacuumAndWarns for the precise
	// index-level check; this only confirms the CPU builder still produces
	// an ordinary, reachable sphere rather than failing to build at all.
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  MediumInterface \"ghost\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t));
}

TEST(PbrtCpuBuildTest, CloudMediumIsReachable) {
	// Like MediumInterfaceWrapsTheSphereInAParticipatingMedium above, this
	// only confirms the standalone cloud_medium_hittable (added independently
	// of the declaring sphere - see addMediumIfPresent's own comment) builds
	// without crashing and stays reachable; CloudMedium's own density math is
	// covered by cloud_medium_hittable_tests.cpp / cloud_medium_tests.cpp.
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"puff\" \"string type\" [ \"cloud\" ]\n"
		"    \"rgb sigma_s\" [ 5 5 5 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 1 1 1\n"
		"  MediumInterface \"puff\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 2 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(1, 1, -5), vec3(0, 0, 1), t))
		<< "a ray toward the cloud medium's world AABB should still hit something";
}

TEST(PbrtCpuBuildTest, RgbGridMediumIsReachable) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"nebula\" \"string type\" [ \"rgbgrid\" ]\n"
		"    \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"    \"rgb sigma_s\" [ 1 1 1  2 2 2 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 1 1 1\n"
		"  MediumInterface \"nebula\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 2 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(1, 1, -5), vec3(0, 0, 1), t))
		<< "a ray toward the rgbgrid medium's world AABB should still hit something";
}

TEST(PbrtCpuBuildTest, UniformgridMediumIsReachable) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"fog\" \"string type\" [ \"uniformgrid\" ]\n"
		"    \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"    \"float density\" [ 0.5 1.0 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 1 1 1\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 2 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(1, 1, -5), vec3(0, 0, 1), t))
		<< "a ray toward the uniformgrid medium's world AABB should still hit something";
}

TEST(PbrtCpuBuildTest, SharedVerticesAreDeduplicated) {
	// The two triangles of a quad share two corners. FlatScene stores all six
	// vertices explicitly; the builder should recover the original four.
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_EQ(b.uniqueVertexCount, 4u);
}

TEST(PbrtCpuBuildTest, DistinctVerticesAreNotCollapsed) {
	// Guards the dedup against being too eager: two separate triangles sharing
	// no corners must keep all six vertices.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 5 5 5  6 5 5  5 6 5 ]\n");
	EXPECT_EQ(b.triangleCount, 2u);
	EXPECT_EQ(b.uniqueVertexCount, 6u);
}

TEST(PbrtCpuBuildTest, EmissiveShapesGoIntoTheLightList) {
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n")
		+ kQuad + "AttributeEnd\n" + kQuad);
	EXPECT_EQ(b.triangleCount, 4u);
	EXPECT_EQ(b.lights->objects.size(), 2u)
		<< "only the triangles inside the attribute scope are emissive";
}

TEST(PbrtCpuBuildTest, AreaLightFilenameBuildsATextureBackedTwoSidedDiffuseLight) {
	// buildFrom() only runs flatten(), not pbrt_load.h's resolution pass (see
	// DiffuseReflectanceImagemapBuildsATextureBackedLambertian's own comment
	// on the same trade-off), so this only checks that a non-empty
	// Emission::filename makes it all the way to a texture-backed,
	// is_two_sided() diffuse_light - makeMaterial()'s own filename branch -
	// rather than the flat-L solid_color-backed one.
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"string filename\" [ \"glow.png\" ]"
					" \"bool twosided\" [ true ]\n")
		+ kQuad + "AttributeEnd\n");
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *dl = dynamic_cast<diffuse_light *>(rec.mat.get());
	ASSERT_NE(dl, nullptr);
	EXPECT_TRUE(dl->is_two_sided());
	EXPECT_EQ(dynamic_cast<mipmap_texture *>(dl->get_texture().get()), dl->get_texture().get())
		<< "a \"filename\" area light must build a mipmap_texture-backed "
		   "diffuse_light, not a flat solid_color one";
}

TEST(PbrtCpuBuildTest, AreaLightWithoutFilenameBuildsAPlainSolidColorDiffuseLight) {
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n")
		+ kQuad + "AttributeEnd\n");
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *dl = dynamic_cast<diffuse_light *>(rec.mat.get());
	ASSERT_NE(dl, nullptr);
	EXPECT_FALSE(dl->is_two_sided());
	EXPECT_EQ(dynamic_cast<mipmap_texture *>(dl->get_texture().get()), nullptr);
}

TEST(PbrtCpuBuildTest, NonEmissiveSceneHasAnEmptyLightList) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_TRUE(b.lights->objects.empty());
}

TEST(PbrtCpuBuildTest, AnEmptySceneBuildsWithoutCrashing) {
	// flatten() drops unsupported shapes, so a scene can legitimately arrive
	// with nothing in it. A BVH must not be built over zero primitives.
	// "hyperboloid" - not "cylinder"/"disk"/"cone"/"paraboloid", which this
	// loader now supports.
	const pbrt_cpu::BuildResult b = buildFrom("Shape \"hyperboloid\"\n");
	EXPECT_EQ(b.triangleCount, 0u);
	EXPECT_EQ(b.sphereCount, 0u);
	EXPECT_EQ(b.diskCount, 0u);
	EXPECT_EQ(b.cylinderCount, 0u);
	EXPECT_EQ(b.coneCount, 0u);
	EXPECT_EQ(b.paraboloidCount, 0u);
	ASSERT_NE(b.world, nullptr);
	EXPECT_TRUE(b.world->objects.empty());
}

TEST(PbrtCpuBuildTest, MaterialsAreSharedRatherThanCopiedPerPrimitive) {
	// A million-triangle mesh with one material should hold one material
	// object, not a million.
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n") + kQuad);
	ASSERT_EQ(b.triangleCount, 2u);

	// The world is BVH-wrapped, so reach the triangles by hitting them rather
	// than walking the object graph.
	hit_record a, c;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), a));
	ASSERT_TRUE(b.world->hit(ray(point3(0.75, 0.75, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), c));
	EXPECT_EQ(a.mat.get(), c.mat.get());
}

// ---------------------------------------------------------------------------
// makeMaterial() used to fall through to lambertian for these three kinds
// silently - no warning, unlike the genuinely-unsupported case - even though
// coated_diffuse, coated_conductor and diffuse_transmission all already
// existed in material_pbrt.h. That made CPU and GPU render the same pbrt
// file with different materials for any scene using one of them. These pin
// the real class getting built, not just that something renders.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, CoatedDiffuseBuildsTheRealMaterialNotLambertian) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"coateddiffuse\" \"rgb reflectance\" [ .2 .4 .6 ] "
		"\"float roughness\" [ .1 ]\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<coated_diffuse *>(rec.mat.get()), nullptr)
		<< "a pbrt coateddiffuse material must build the real coated_diffuse "
		   "class, not silently fall back to lambertian";
}

TEST(PbrtCpuBuildTest, CoatedConductorBuildsTheRealMaterialNotLambertian) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"coatedconductor\" \"rgb reflectance\" [ .8 .8 .8 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<coated_conductor *>(rec.mat.get()), nullptr)
		<< "a pbrt coatedconductor material must build the real "
		   "coated_conductor class, not silently fall back to lambertian";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceImagemapBuildsATextureBackedLambertian) {
	// buildFrom() only runs flatten(), not pbrt_load.h's resolution pass, so
	// Material::textureFilename stays "as written" here ("t.png") rather than
	// an absolute path - fine for this test, which only checks that a
	// non-empty textureFilename makes it all the way to a mipmap_texture-
	// backed lambertian (this Diffuse case's own branch in makeMaterial()),
	// not a plain solid_color-backed one built from `reflectance`.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	EXPECT_NE(dynamic_cast<mipmap_texture *>(lam->get_texture().get()), nullptr)
		<< "a diffuse material with an imagemap-bound reflectance must build a "
		   "texture-backed lambertian, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceImagemapBuildsATextureBackedCoatedDiffuse) {
	// pbrt's own ganesha scene's exact binding shape (a bare imagemap, no
	// wrapping "scale") - see pbrt_flatten::Material::textureFilename's own
	// comment on why CoatedDiffuse is now gated in alongside Diffuse.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	EXPECT_NE(dynamic_cast<mipmap_texture *>(cd->get_texture().get()), nullptr)
		<< "a coateddiffuse material with an imagemap-bound reflectance must build a "
		   "texture-backed coated_diffuse, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DielectricRoughnessImagemapBuildsATextureBackedRoughDielectric) {
	// pbrt-v4 "texture roughness" on a Dielectric bound to a bare
	// imagemap - see pbrt_flatten::Material::roughnessTextureFilename's
	// own comment. Must build a real texture-backed rough_dielectric (not
	// the flat-roughness overload, and not a plain smooth dielectric).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"scratch\" \"float\" \"imagemap\" \"string filename\" [ \"scratch.png\" ]\n"
		"Material \"dielectric\" \"texture roughness\" [ \"scratch\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *rd = dynamic_cast<rough_dielectric *>(rec.mat.get());
	ASSERT_NE(rd, nullptr);
	EXPECT_NE(dynamic_cast<mipmap_texture *>(rd->get_roughness_texture().get()), nullptr)
		<< "a dielectric material with an imagemap-bound roughness must build a "
		   "texture-backed rough_dielectric, not the flat-roughness/smooth fallback";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceScaleWrappedImagemapAppliesTheScale) {
	// barcelona-pavilion's own dominant binding shape (an imagemap wrapped
	// in a "scale" texture, e.g. materials.pbrt's "concrete-kd") - Material::
	// textureScale's own comment.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Texture \"tmap-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"tmap\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"tmap-scaled\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	auto *scaled = dynamic_cast<scaled_texture *>(cd->get_texture().get());
	ASSERT_NE(scaled, nullptr)
		<< "a scale-wrapped imagemap reflectance must build a scaled_texture "
		   "wrapping the real mipmap_texture, not just the bare image";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceCheckerboardBuildsAUVCheckerBackedCoatedDiffuse) {
	// checkerboard/fbm/marble/mix used to be Diffuse-only; now also resolved
	// for CoatedDiffuse (mirrors DiffuseReflectanceCheckerboardBuildsA
	// UVCheckerBackedLambertian above, just via coated_diffuse's own
	// texture-taking constructor).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"chk\" \"spectrum\" \"checkerboard\" "
		"\"rgb tex1\" [ 1 0 0 ] \"rgb tex2\" [ 0 0 1 ] "
		"\"float uscale\" [ 1 ] \"float vscale\" [ 1 ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	auto *chk = dynamic_cast<uv_checker_texture *>(cd->get_texture().get());
	ASSERT_NE(chk, nullptr)
		<< "a coateddiffuse material with a checkerboard-bound reflectance must "
		   "build a uv_checker_texture-backed coated_diffuse";
	const color c00 = chk->value(0.25, 0.25, rec.p);
	const color c10 = chk->value(1.25, 0.25, rec.p);
	EXPECT_NE(c00.x(), c10.x())
		<< "adjacent UV checker cells (one uscale apart) must differ";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceMixBuildsAMixBackedCoatedDiffuse) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float amount\" [ 0.25 ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	auto *mix = dynamic_cast<mix_texture *>(cd->get_texture().get());
	ASSERT_NE(mix, nullptr)
		<< "a coateddiffuse material with a mix-bound reflectance must build a "
		   "mix_texture-backed coated_diffuse, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceFbmBuildsAnFbmBackedCoatedDiffuse) {
	// Mirrors DiffuseReflectanceFbmBuildsAnFbmBackedLambertian below, for
	// CoatedDiffuse's own separate branch (previously untested - checkerboard
	// and mix were covered above, but fbm/marble were not).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"cloud\" \"float\" \"fbm\" \"integer octaves\" [ 4 ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"cloud\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	EXPECT_NE(dynamic_cast<fbm_texture *>(cd->get_texture().get()), nullptr)
		<< "a coateddiffuse material with an fbm-bound reflectance must build an "
		   "fbm_texture-backed coated_diffuse, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceMarbleBuildsAMarbleBackedCoatedDiffuse) {
	// Mirrors DiffuseReflectanceMarbleBuildsAMarbleBackedLambertian below,
	// for CoatedDiffuse's own separate branch.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"stone\" \"spectrum\" \"marble\"\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"stone\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	EXPECT_NE(dynamic_cast<marble_texture *>(cd->get_texture().get()), nullptr)
		<< "a coateddiffuse material with a marble-bound reflectance must build a "
		   "marble_texture-backed coated_diffuse, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceScaleWrappedImagemapAppliesTheScale) {
	// barcelona-pavilion's own dominant binding shape, now also supported for
	// plain diffuse surfaces (not just coateddiffuse) - Material::
	// textureScale's own comment.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Texture \"tmap-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"tmap\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"tmap-scaled\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *scaled = dynamic_cast<scaled_texture *>(lam->get_texture().get());
	ASSERT_NE(scaled, nullptr)
		<< "a scale-wrapped imagemap reflectance on a plain diffuse material must "
		   "build a scaled_texture wrapping the real mipmap_texture, not just the "
		   "bare image";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceCheckerboardBuildsAUVCheckerBackedLambertian) {
	// Round 5 Phase 2: hasCheckerReflectance must make it all the way to a
	// uv_checker_texture-backed lambertian, and that texture must actually
	// tile by UV (not the unrelated 3D world-space checker_texture) - probed
	// here via value() at two UVs one checker cell apart.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"chk\" \"spectrum\" \"checkerboard\" "
		"\"rgb tex1\" [ 1 0 0 ] \"rgb tex2\" [ 0 0 1 ] "
		"\"float uscale\" [ 1 ] \"float vscale\" [ 1 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *chk = dynamic_cast<uv_checker_texture *>(lam->get_texture().get());
	ASSERT_NE(chk, nullptr)
		<< "a diffuse material with a checkerboard-bound reflectance must "
		   "build a uv_checker_texture-backed lambertian";
	const color c00 = chk->value(0.25, 0.25, rec.p);
	const color c10 = chk->value(1.25, 0.25, rec.p);
	EXPECT_NE(c00.x(), c10.x())
		<< "adjacent UV checker cells (one uscale apart) must differ";
}

TEST(PbrtCpuBuildTest, CheckerboardWithNestedImagemapTex1BuildsAMipmapBackedSlot) {
	// One-level-nested tex1 (bound to a bare imagemap Texture) must build a
	// REAL mipmap_texture for that slot, not silently fall back to reading
	// checkerColor1 (which stays at its unused default here). buildFrom()
	// skips pbrt_load.h's resolution pass, so "leaf.png" never actually
	// resolves/decodes - mipmap_texture's own documented "corrupt-or-missing
	// file -> cyan debug colour (0,1,1)" fallback (texture.h) is exactly the
	// signal this test uses to confirm a REAL mipmap_texture is in that
	// slot, as opposed to any flat colour (which could never coincidentally
	// equal solid cyan).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Texture \"chk\" \"spectrum\" \"checkerboard\" \"texture tex1\" [ \"leaf\" ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float uscale\" [ 1 ] \"float vscale\" [ 1 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *chk = dynamic_cast<uv_checker_texture *>(lam->get_texture().get());
	ASSERT_NE(chk, nullptr);
	const color tex1Cell = chk->value(0.25, 0.25, rec.p);   // (0,0) cell - tex1 (nested imagemap)
	const color tex2Cell = chk->value(1.25, 0.25, rec.p);   // (1,0) cell - tex2 (flat blue literal)
	EXPECT_NEAR(tex1Cell.x(), 0.0, 1e-9);
	EXPECT_NEAR(tex1Cell.y(), 1.0, 1e-9);
	EXPECT_NEAR(tex1Cell.z(), 1.0, 1e-9);
	EXPECT_NEAR(tex2Cell.z(), 1.0, 1e-9) << "tex2 must stay the flat blue literal, unaffected";
}

TEST(PbrtCpuBuildTest, CheckerboardWithSecondLevelNestedCheckerboardBuildsRealNestedTexture) {
	// A second level of nesting: the outer checkerboard's tex1 names ANOTHER
	// checkerboard Texture (itself nesting a bare imagemap in ITS tex1),
	// instead of a flat literal or a bare imagemap directly - real recursive
	// evaluation, not the flat-average-colour approximation GPU falls back
	// to (see pbrt_flatten::NestedProceduralTexture's own comment).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Texture \"inner\" \"spectrum\" \"checkerboard\" \"texture tex1\" [ \"leaf\" ] "
		"\"rgb tex2\" [ 0 1 0 ] \"float uscale\" [ 1 ] \"float vscale\" [ 1 ]\n"
		"Texture \"outer\" \"spectrum\" \"checkerboard\" \"texture tex1\" [ \"inner\" ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float uscale\" [ 1 ] \"float vscale\" [ 1 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"outer\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *outer = dynamic_cast<uv_checker_texture *>(lam->get_texture().get());
	ASSERT_NE(outer, nullptr);
	auto *inner = dynamic_cast<uv_checker_texture *>(outer->get_tex1().get());
	ASSERT_NE(inner, nullptr)
		<< "outer checkerboard's tex1 slot must build a REAL nested "
		   "uv_checker_texture, not fall back to a flat colour";
	// Outer's (0,0) cell is tex1 (the inner checkerboard); at (0.25,0.25) the
	// inner pattern's own (0,0) cell is ITS tex1 - the imagemap, unresolved
	// here so it reads back as mipmap_texture's cyan (0,1,1) debug fallback.
	const color outerTex1Cell = outer->value(0.25, 0.25, rec.p);
	EXPECT_NEAR(outerTex1Cell.x(), 0.0, 1e-9);
	EXPECT_NEAR(outerTex1Cell.y(), 1.0, 1e-9);
	EXPECT_NEAR(outerTex1Cell.z(), 1.0, 1e-9);
	// Outer's (1,0) cell is tex2 (flat blue), unaffected by the nesting.
	const color outerTex2Cell = outer->value(1.25, 0.25, rec.p);
	EXPECT_NEAR(outerTex2Cell.z(), 1.0, 1e-9);
	// Sample the inner checkerboard directly (not through outer) to confirm
	// its own tex2 (flat green) is intact and independently addressable.
	const color innerTex2Cell = inner->value(1.25, 0.25, rec.p);
	EXPECT_NEAR(innerTex2Cell.y(), 1.0, 1e-9);
	EXPECT_NEAR(innerTex2Cell.x(), 0.0, 1e-9);
}

TEST(PbrtCpuBuildTest, MixWithSecondLevelNestedMixBuildsRealNestedTexture) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"inner\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 1 0 ] \"float amount\" [ 1 ]\n"
		"Texture \"outer\" \"spectrum\" \"mix\" \"texture tex1\" [ \"inner\" ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float amount\" [ 0 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"outer\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *outer = dynamic_cast<mix_texture *>(lam->get_texture().get());
	ASSERT_NE(outer, nullptr);
	auto *inner = dynamic_cast<mix_texture *>(outer->get_tex1().get());
	ASSERT_NE(inner, nullptr)
		<< "outer mix's tex1 slot must build a REAL nested mix_texture, not "
		   "fall back to a flat colour";
	// Outer amount=0 -> pure tex1 -> inner amount=1 -> pure inner tex2 (green).
	const color result = outer->value(0.5, 0.5, rec.p);
	EXPECT_NEAR(result.x(), 0.0, 1e-9);
	EXPECT_NEAR(result.y(), 1.0, 1e-9);
	EXPECT_NEAR(result.z(), 0.0, 1e-9);
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceFbmBuildsAnFbmBackedLambertian) {
	// Round 6 Phase 1: hasFbmReflectance must make it all the way to an
	// fbm_texture-backed lambertian, reusing the existing CPU class rather
	// than falling back to a flat colour.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"cloud\" \"float\" \"fbm\" \"integer octaves\" [ 4 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"cloud\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	EXPECT_NE(dynamic_cast<fbm_texture *>(lam->get_texture().get()), nullptr)
		<< "a diffuse material with an fbm-bound reflectance must build an "
		   "fbm_texture-backed lambertian, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceMarbleBuildsAMarbleBackedLambertian) {
	// Round 6 Phase 1: hasMarbleReflectance must make it all the way to a
	// marble_texture-backed lambertian.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"stone\" \"spectrum\" \"marble\"\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"stone\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	EXPECT_NE(dynamic_cast<marble_texture *>(lam->get_texture().get()), nullptr)
		<< "a diffuse material with a marble-bound reflectance must build a "
		   "marble_texture-backed lambertian, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceMixBuildsAMixBackedLambertian) {
	// Round 6 Phase 1: hasMixReflectance must make it all the way to a
	// mix_texture-backed lambertian that actually blends tex1/tex2 by
	// amount (deterministic, unlike fbm/marble - so the resulting colour
	// can be checked exactly).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float amount\" [ 0.25 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *mix = dynamic_cast<mix_texture *>(lam->get_texture().get());
	ASSERT_NE(mix, nullptr)
		<< "a diffuse material with a mix-bound reflectance must build a "
		   "mix_texture-backed lambertian, not the flat-colour fallback";
	const color c = mix->value(0.0, 0.0, rec.p);
	EXPECT_NEAR(c.x(), 0.75, 1e-9);  // (1-0.25)*1 + 0.25*0
	EXPECT_NEAR(c.z(), 0.25, 1e-9);  // (1-0.25)*0 + 0.25*1
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceMixAmountImagemapBuildsATextureBackedBlend) {
	// pbrt-v4's real "amount" bound to a Texture (e.g. an fbm-driven dirt/
	// wear mask) - previously unsupported, now resolved the same one-level
	// nested-bare-imagemap way as tex1/tex2 (Material::
	// mixAmountTextureFilename's own comment). Must build a REAL
	// mipmap_texture-backed amount, not the flat-scalar fallback.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"wear\" \"float\" \"imagemap\" \"string filename\" [ \"wear.png\" ]\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 0 1 ] \"texture amount\" [ \"wear\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *mix = dynamic_cast<mix_texture *>(lam->get_texture().get());
	ASSERT_NE(mix, nullptr);
	EXPECT_NE(dynamic_cast<mipmap_texture *>(mix->get_amount_texture().get()), nullptr)
		<< "a mix reflectance with a texture-bound amount must build a "
		   "texture-backed blend fraction, not the flat-scalar fallback";
}

TEST(PbrtCpuBuildTest, CoatedDiffuseReflectanceMixAmountImagemapBuildsATextureBackedBlend) {
	// Same as DiffuseReflectanceMixAmountImagemapBuildsATextureBackedBlend
	// above, for CoatedDiffuse's own separate branch.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"wear\" \"float\" \"imagemap\" \"string filename\" [ \"wear.png\" ]\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 0 1 ] \"texture amount\" [ \"wear\" ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cd = dynamic_cast<coated_diffuse *>(rec.mat.get());
	ASSERT_NE(cd, nullptr);
	auto *mix = dynamic_cast<mix_texture *>(cd->get_texture().get());
	ASSERT_NE(mix, nullptr);
	EXPECT_NE(dynamic_cast<mipmap_texture *>(mix->get_amount_texture().get()), nullptr)
		<< "a mix reflectance with a texture-bound amount must build a "
		   "texture-backed blend fraction, not the flat-scalar fallback";
}

TEST_F(CpuBuilderTempTree, DisplacementWithARealGrayscaleBumpMapWrapsInBumpMapMaterial) {
	// Round 5 Phase 1: Material::displacementTextureFilename (see that
	// field's own comment) must reach the CPU builder's materialFor()
	// lambda and wrap the base material in bump_map_material - the same
	// decorator mesh.h's own OBJ/MTL map_Bump dispatch already uses for a
	// genuine grayscale height/displacement image. Goes through the real
	// pbrt_load::loadFile() resolution pass (not just flatten()) so the
	// path is a real, existence-tested, absolute file pbrt_cpu_builder.h's
	// rtw_image probe can actually decode.
	write("scene.pbrt",
		  "Texture \"bmap\" \"float\" \"imagemap\" \"string filename\" [ \"bump.bmp\" ]\n"
		  "Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ] "
		  "\"texture displacement\" [ \"bmap\" ]\n"
		  + std::string(kQuad));
	write("bump.bmp", solidGrayBmp1x1());

	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;
	ASSERT_EQ(loaded.scene.materials.size(), 1u);
	ASSERT_EQ(loaded.scene.materials[0].displacementTextureFilename, path("bump.bmp"));

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<bump_map_material *>(rec.mat.get()), nullptr)
		<< "a material with a resolved, decodable displacement texture must "
		   "wrap the base material in bump_map_material";
}

TEST(PbrtCpuBuildTest, DiffuseTransmissionBuildsTheRealMaterialNotLambertian) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"diffusetransmission\"\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<diffuse_transmission *>(rec.mat.get()), nullptr)
		<< "a pbrt diffusetransmission material must build the real "
		   "diffuse_transmission class, not silently fall back to lambertian";
}

TEST(PbrtCpuBuildTest, DiffuseTransmissionReflectanceScaleWrappedImagemapAppliesTheScale) {
	// "scale"-unwrap used to be CoatedDiffuse-only; now also resolved for
	// DiffuseTransmission's own reflectance - Material::textureScale's own
	// comment.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Texture \"leaf-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"leaf\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"leaf-scaled\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *dt = dynamic_cast<diffuse_transmission *>(rec.mat.get());
	ASSERT_NE(dt, nullptr);
	auto *scaled = dynamic_cast<scaled_texture *>(dt->get_reflectance_texture().get());
	ASSERT_NE(scaled, nullptr)
		<< "a scale-wrapped imagemap reflectance must build a scaled_texture "
		   "wrapping the real mipmap_texture, not just the bare image";
}

TEST(PbrtCpuBuildTest, DiffuseTransmissionTransmittanceScaleWrappedImagemapAppliesItsOwnScale) {
	// Same as the reflectance test above, for transmittance's own
	// independent scale (Material::transmittanceTextureScale's own
	// comment) - a DIFFERENT scale value than reflectance's own, to catch
	// the two scales being accidentally cross-applied.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"leaf-r\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf-r.png\" ]\n"
		"Texture \"leaf-t\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf-t.png\" ]\n"
		"Texture \"leaf-t-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"leaf-t\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"leaf-r\" ] "
		"\"texture transmittance\" [ \"leaf-t-scaled\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *dt = dynamic_cast<diffuse_transmission *>(rec.mat.get());
	ASSERT_NE(dt, nullptr);
	EXPECT_EQ(dynamic_cast<scaled_texture *>(dt->get_reflectance_texture().get()), nullptr)
		<< "reflectance was bound to a bare (unscaled) imagemap, must not be scaled_texture-wrapped";
	auto *scaled = dynamic_cast<scaled_texture *>(dt->get_transmittance_texture().get());
	ASSERT_NE(scaled, nullptr)
		<< "a scale-wrapped imagemap transmittance must build a scaled_texture "
		   "wrapping the real mipmap_texture, not just the bare image";
}

// ---------------------------------------------------------------------------
// LightSource point/spot/distant/goniometric/projection
//
// pbrt_flatten_tests.cpp already pins the parsing (parameter defaults, CTM
// handling); these pin the next stage - that a parsed pbrt_flatten::
// PunctualLight actually becomes a real punctual_light_objects.h light a
// shading point can sample, wired the same way build_point_light_punct() et
// al. wire the hand-built C2-C6 showcase scenes (scenes_advanced.h).
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, NoLightSourceLeavesPunctLightsNull) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_EQ(b.punctLights, nullptr);
}

TEST(PbrtCpuBuildTest, PointLightSourceIsSampleableAtAShadingPoint) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"LightSource \"point\" \"point3 from\" [ 0 10 0 ] \"rgb I\" [ 1 1 1 ] "
		"\"float scale\" [ 100 ]\n" + std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);
	ASSERT_FALSE(b.punctLights->empty());

	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		// wi must point from the shading point TOWARD the light: straight up.
		EXPECT_NEAR(s.wi.x(), 0.0, 1e-9);
		EXPECT_NEAR(s.wi.y(), 1.0, 1e-9);
		EXPECT_NEAR(s.wi.z(), 0.0, 1e-9);
		// Li = I * scale / r^2 = 1 * 100 / 100 = 1.
		EXPECT_NEAR(s.Li.x(), 1.0, 1e-6);
		EXPECT_NEAR(s.t_max, 10.0, 1e-6);
	});
	EXPECT_EQ(samples, 1);
}

TEST(PbrtCpuBuildTest, SpotLightSourceAimsFromFromTowardTo) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"LightSource \"spot\" \"point3 from\" [ 0 0 -10 ] \"point3 to\" [ 0 0 0 ] "
		"\"rgb I\" [ 1 1 1 ] \"float scale\" [ 100 ] \"float coneangle\" [ 60 ]\n"
		+ std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	// A point directly in front of the spot (along its axis) sees full
	// intensity; the shadow-ray direction must point back toward the light.
	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		EXPECT_NEAR(s.wi.z(), -1.0, 1e-9);
		EXPECT_GT(s.Li.x(), 0.0) << "on-axis point should be lit, not in the falloff";
	});
	EXPECT_EQ(samples, 1);
}

TEST(PbrtCpuBuildTest, DistantLightSourceHasNoDistanceFalloff) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"LightSource \"distant\" \"point3 from\" [ 0 1 0 ] \"point3 to\" [ 0 0 0 ] "
		"\"rgb L\" [ 2 3 4 ] \"float scale\" [ 1 ]\n" + std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	// Radiance must be identical at two very different distances - that is
	// the entire point of a directional/parallel light.
	color nearL, farL;
	b.punctLights->for_each_sample(point3(0, 0, 0),
		[&](const PunctualLiSample &s) { nearL = s.Li; });
	b.punctLights->for_each_sample(point3(0, 1000, 0),
		[&](const PunctualLiSample &s) { farL = s.Li; });
	EXPECT_NEAR(nearL.x(), 2.0, 1e-6);
	EXPECT_NEAR(nearL.x(), farL.x(), 1e-9);
	EXPECT_NEAR(nearL.y(), farL.y(), 1e-9);
	EXPECT_NEAR(nearL.z(), farL.z(), 1e-9);
}

TEST(PbrtCpuBuildTest, GoniometricLightSourceIsSampleableAtAShadingPoint) {
	// pbrt-v4's GoniometricLight has no "from"/"to" of its own (unlike point/
	// spot/distant) - its position/orientation come purely from the CTM, so
	// this positions it with Translate rather than a "from" parameter (see
	// PunctualLight::pos's own comment). No "filename" - flatten() falls
	// back to an isotropic profile (see FlattenPunctualLightTest::
	// GoniometricLightWithNoFilenameIsIsotropicAndUnwarned in
	// pbrt_flatten_tests.cpp), so this should behave exactly like a plain
	// point light: nonzero, direction-independent intensity.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 10 0\n"
		"LightSource \"goniometric\" \"rgb I\" [ 1 1 1 ] \"float scale\" [ 100 ]\n"
		+ std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		EXPECT_GT(s.Li.x(), 0.0);
	});
	EXPECT_EQ(samples, 1);
}

TEST_F(CpuBuilderTempTree, GoniometricLightWithRealSquareImageProducesNonUniformIntensity) {
	// Goes through the real pbrt_load::loadFile() resolution pass (not
	// buildFrom()'s bare flatten()) so PunctualLight::filename is a real,
	// existence-tested, absolute path pbrt_cpu_builder.h's decode can
	// actually open - see decodePunctualLightImageFile()'s own comment.
	write("scene.pbrt",
		  "LightSource \"goniometric\" \"rgb I\" [ 1 1 1 ] \"float scale\" [ 100 ] "
		  "\"string filename\" [ \"profile.bmp\" ]\n");
	write("profile.bmp", twoToneBmp2x2());

	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;
	ASSERT_EQ(loaded.scene.punctualLights.size(), 1u);
	ASSERT_EQ(loaded.scene.punctualLights[0].filename, path("profile.bmp"));

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	ASSERT_NE(b.punctLights, nullptr);

	// Sample from points spread across genuinely different directions
	// (not just a flat circle at one height, which - since the two-tone
	// split is row-wise, i.e. along the equal-area square's v/polar axis -
	// would keep every point in the same row band and defeat this test) -
	// a real two-tone image gives a different eval_I() at different
	// directions; the flat kUniformImage fallback would give the exact
	// same value everywhere, since telling those two apart is the whole
	// point of this test.
	const point3 samplePoints[] = {
		point3(10, 0, 0),  point3(-10, 0, 0),
		point3(0, 10, 0),  point3(0, -10, 0),
		point3(0, 0, 10),  point3(0, 0, -10),
		point3(7, 7, 7),   point3(-7, -7, -7),
	};
	double maxLi = 0.0;
	bool sawDifferentValue = false;
	bool sawFirst = false;
	double first = 0.0;
	for (const point3 &p : samplePoints) {
		b.punctLights->for_each_sample(p, [&](const PunctualLiSample &s) {
			maxLi = std::max(maxLi, s.Li.x());
			if (!sawFirst) { first = s.Li.x(); sawFirst = true; }
			else if (std::abs(s.Li.x() - first) > 1e-9) sawDifferentValue = true;
		});
	}
	EXPECT_GT(maxLi, 0.0);
	EXPECT_TRUE(sawDifferentValue)
		<< "identical intensity at every sampled direction means the real "
		   "two-tone image was not actually used";
}

TEST_F(CpuBuilderTempTree, ProjectionLightWithRealImageProducesNonUniformIntensity) {
	write("scene.pbrt",
		  "Translate 0 0 -10\n"
		  "LightSource \"projection\" \"float scale\" [ 100 ] \"float fov\" [ 80 ] "
		  "\"string filename\" [ \"slide.bmp\" ]\n");
	write("slide.bmp", twoToneBmp2x2());

	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;
	ASSERT_EQ(loaded.scene.punctualLights.size(), 1u);
	ASSERT_EQ(loaded.scene.punctualLights[0].filename, path("slide.bmp"));

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	ASSERT_NE(b.punctLights, nullptr);

	// Sample across a small grid of points inside the projected beam - a
	// real two-tone slide gives different Li at different points within
	// its fov; kUniformSlide's flat white fallback would give the exact
	// same value everywhere inside the beam.
	double first = -1.0;
	bool sawDifferentValue = false;
	for (int gx = -2; gx <= 2; ++gx) {
		for (int gy = -2; gy <= 2; ++gy) {
			const point3 p(gx * 0.5, gy * 0.5, 0.0);
			b.punctLights->for_each_sample(p, [&](const PunctualLiSample &s) {
				if (s.Li.x() <= 0.0) return;  // outside the beam/fov
				if (first < 0.0) first = s.Li.x();
				else if (std::abs(s.Li.x() - first) > 1e-9) sawDifferentValue = true;
			});
		}
	}
	EXPECT_GT(first, 0.0);
	EXPECT_TRUE(sawDifferentValue)
		<< "identical intensity at every sampled point means the real "
		   "two-tone slide was not actually used";
}

TEST(PbrtCpuBuildTest, ProjectionLightSourceIsSampleableAtAShadingPoint) {
	// Like GoniometricLight, pbrt-v4's ProjectionLight has no "from"/"to" -
	// position/aim come purely from the CTM (Translate here places it at
	// (0,0,-10) looking down its local +z, which with no Rotate is world
	// +z - straight at the quad's shading point at the origin). No
	// "filename" either (see flatten()'s own warning for this case) - the
	// uniform-white-beam fallback should still light a point inside its fov.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 0 -10\n"
		"LightSource \"projection\" \"float scale\" [ 100 ] \"float fov\" [ 60 ]\n"
		+ std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		EXPECT_GT(s.Li.x(), 0.0) << "on-axis point should fall inside the projected beam";
	});
	EXPECT_EQ(samples, 1);
}

// ---------------------------------------------------------------------------
// Material "dielectric" with/without roughness
//
// A nonzero "roughness" on a pbrt dielectric asks for the GGX microfacet
// variant (pbrt-v4 DielectricBxDF's rough path) - this codebase's real model
// for that is rough_dielectric, not plain dielectric. Pins that the roughness
// parameter actually routes to the different class rather than being parsed
// and silently dropped (a pre-existing gap fixed alongside the Metal/
// rough_metal GPU mismatch this session).
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, SmoothDielectricBuildsThePlainDielectricClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"dielectric\" \"float eta\" [ 1.5 ]\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<dielectric *>(rec.mat.get()), nullptr)
		<< "a pbrt dielectric with no roughness must build the plain smooth "
		   "dielectric class";
	EXPECT_EQ(dynamic_cast<rough_dielectric *>(rec.mat.get()), nullptr);
}

TEST(PbrtCpuBuildTest, RoughDielectricRoughnessBuildsTheRealRoughDielectricClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"dielectric\" \"float eta\" [ 1.5 ] \"float roughness\" [ 0.2 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<rough_dielectric *>(rec.mat.get()), nullptr)
		<< "a pbrt dielectric with nonzero roughness must build the real "
		   "rough_dielectric (GGX) class, not silently drop the roughness "
		   "and render a perfect mirror/refractor";
}

// ---------------------------------------------------------------------------
// Material "none" (pbrt-v4's real interface-material idiom)
//
// Must build the dedicated interface_material class, not a near-invisible
// dielectric - real pass-through, no Fresnel/refraction, and a distinct
// classification (is_medium_boundary()=true, is_delta_bsdf()=false) the
// integrators use to skip the crossing entirely instead of treating it as a
// specular bounce.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, InterfaceMaterialBuildsTheDedicatedClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"none\"\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	ASSERT_NE(dynamic_cast<interface_material *>(rec.mat.get()), nullptr)
		<< "Material \"none\" must build the dedicated interface_material "
		   "class, not a near-invisible dielectric";
	EXPECT_FALSE(rec.mat->is_delta_bsdf())
		<< "not a real delta surface - is_medium_boundary() is the correct "
		   "classification instead";
	EXPECT_TRUE(rec.mat->is_medium_boundary());
	EXPECT_TRUE(rec.mat->is_shadow_transmissive(rec));
}

TEST(PbrtCpuBuildTest, InterfaceMaterialScatterIsAnExactPassThrough) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"none\"\n" + std::string(kQuad));
	hit_record rec;
	const vec3 in_dir(0.05, -0.03, 1.0);  // slightly oblique, stays inside the [0,1]x[0,1] quad from z=-5
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), unit_vector(in_dir)),
							 interval(0.001, infinity), rec));
	scatter_record srec;
	ASSERT_TRUE(rec.mat->scatter(ray(point3(0.5, 0.5, -5), unit_vector(in_dir)), rec, srec));
	EXPECT_TRUE(srec.skip_pdf);
	EXPECT_TRUE(srec.is_medium_boundary);
	EXPECT_DOUBLE_EQ(srec.attenuation.x(), 1.0);
	EXPECT_DOUBLE_EQ(srec.attenuation.y(), 1.0);
	EXPECT_DOUBLE_EQ(srec.attenuation.z(), 1.0);
	// Exact same direction as the incoming ray - no Fresnel reflection, no
	// refraction, no critical angle.
	vec3 out_dir = unit_vector(srec.skip_pdf_ray.direction());
	vec3 expected = unit_vector(in_dir);
	EXPECT_NEAR(out_dir.x(), expected.x(), 1e-12);
	EXPECT_NEAR(out_dir.y(), expected.y(), 1e-12);
	EXPECT_NEAR(out_dir.z(), expected.z(), 1e-12);
}

// ---------------------------------------------------------------------------
// Material "conductor"
//
// pbrt describes a conductor's complex IOR via "spectrum eta"/"spectrum k"
// bound to a NAMED spectrum ("metal-Ag-eta"/"metal-Ag-k") - a recognized
// metal name should build the real `conductor` class (GGX + complex
// Fresnel), not the flat-albedo `metal` fuzz-mirror approximation every
// pbrt conductor silently fell back to before.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, NamedMetalSpectrumConductorBuildsTheRealConductorClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"conductor\" \"spectrum eta\" [ \"metal-Ag-eta\" ] "
		"\"spectrum k\" [ \"metal-Ag-k\" ] \"float roughness\" [ 0.1 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<conductor *>(rec.mat.get()), nullptr)
		<< "a pbrt conductor bound to a recognized named metal spectrum "
		   "(metal-Ag-eta/metal-Ag-k) must build the real conductor (GGX + "
		   "complex Fresnel) class, not fall back to the flat-albedo metal "
		   "fuzz-mirror approximation";
}

TEST(PbrtCpuBuildTest, ExplicitRgbEtaKConductorBuildsTheRealConductorClass) {
	// This codebase's own conductor BxDF is already a plain 3-float RGB
	// model (matching its own named-spectrum table's shape), so an explicit
	// "rgb eta"/"rgb k" pair (not a named spectrum) needs no spectral
	// upsampling to build the real class - see flatten()'s own Conductor-
	// OR-CoatedConductor branch.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"conductor\" \"rgb eta\" [ 0.2 0.9 1.4 ] \"rgb k\" [ 3.9 2.5 2.1 ] "
		"\"float roughness\" [ 0.1 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<conductor *>(rec.mat.get()), nullptr)
		<< "a pbrt conductor bound to an explicit rgb eta/k pair must build "
		   "the real conductor (GGX + complex Fresnel) class, not fall back "
		   "to the flat-albedo metal fuzz-mirror approximation";
}

TEST(PbrtCpuBuildTest, ExplicitRgbEtaOnlyConductorFallsBackToMetal) {
	// Giving only ONE of eta/k (not both) as an explicit RGB triple must NOT
	// activate the real model - there's no well-defined "the other one
	// defaults to what" for an arbitrary explicit pair (unlike the
	// documented Cu-default for giving NEITHER).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"conductor\" \"rgb eta\" [ 0.2 0.9 1.4 ] \"float roughness\" [ 0.1 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<metal *>(rec.mat.get()), nullptr)
		<< "a pbrt conductor with only eta (no k) as explicit RGB must keep "
		   "the pre-existing metal fuzz-mirror approximation";
}

TEST(PbrtCpuBuildTest, CoatedConductorNamedSpectrumBuildsTheRealConductorFresnel) {
	// CoatedConductor previously never resolved a named spectrum (or an
	// explicit RGB eta/k) at all - even "metal-Au-eta" was silently ignored,
	// always going through reflectanceToConductorK()'s approximation. Gold's
	// real complex IOR has a strongly chromatic (red >> blue) normal-
	// incidence Fresnel; the approximation would stay grey/achromatic here
	// since no "reflectance" was even given (defaults to m.color's flat
	// 0.5,0.5,0.5), so this distinguishes the two paths unambiguously.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"coatedconductor\" \"spectrum eta\" [ \"metal-Au-eta\" ] "
		"\"spectrum k\" [ \"metal-Au-k\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *cc = dynamic_cast<coated_conductor *>(rec.mat.get());
	ASSERT_NE(cc, nullptr);
	color f0 = cc->get_conductor_f0();
	EXPECT_GT(f0.x(), f0.z())
		<< "a coatedconductor bound to a recognized named metal spectrum "
		   "must build the real complex-IOR Fresnel (gold: red F0 >> blue "
		   "F0), not the pre-existing grey/achromatic approximation";
}

TEST(PbrtCpuBuildTest, UnrecognizedConductorSpectrumFallsBackToMetal) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"conductor\" \"rgb reflectance\" [ .8 .8 .8 ] "
		"\"float roughness\" [ 0.1 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<metal *>(rec.mat.get()), nullptr)
		<< "a pbrt conductor with no eta/k (or an explicit RGB k) must keep "
		   "the pre-existing metal fuzz-mirror approximation";
	EXPECT_EQ(dynamic_cast<conductor *>(rec.mat.get()), nullptr);
}

// ---------------------------------------------------------------------------
// Material "mix"
//
// pbrt_flatten_tests.cpp pins the name resolution; these pin the next stage -
// that a resolved pbrt_flatten::Material with MaterialKind::Mix actually
// becomes a real mix_material wrapping the real CPU classes its two named
// sub-materials asked for (not two more Lambertians), matching how the
// Coated/DiffuseTransmission tests above already pin their own real classes.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, MixMaterialBuildsTheRealMixOfItsTwoSubMaterials) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMaterial \"a\" \"string type\" [ \"conductor\" ] \"rgb reflectance\" [ .2 .4 .6 ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"dielectric\" ] \"float eta\" [ 1.7 ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ] \"float amount\" [ 0.75 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *mix = dynamic_cast<mix_material *>(rec.mat.get());
	ASSERT_NE(mix, nullptr)
		<< "a pbrt mix material must build the real mix_material class, not "
		   "silently fall back to lambertian";
	EXPECT_NE(dynamic_cast<metal *>(mix->get_mat_a().get()), nullptr)
		<< "sub-material 'a' (conductor) must build the real metal class";
	EXPECT_NE(dynamic_cast<dielectric *>(mix->get_mat_b().get()), nullptr)
		<< "sub-material 'b' (dielectric) must build the real dielectric class";
	EXPECT_DOUBLE_EQ(mix->get_weight()->value(0, 0, point3(0, 0, 0)).x(), 0.75);
}

TEST(PbrtCpuBuildTest, MalformedMixFallsBackToLambertianLikeAnyOtherUnsupportedMaterial) {
	// flatten() already downgrades an unresolvable mix to MaterialKind::
	// Unsupported (pinned in pbrt_flatten_tests.cpp) - this just confirms
	// the builder's existing Unsupported->lambertian fallback still applies,
	// rather than the builder crashing or constructing a mix_material with
	// dangling sub-material indices.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"mix\" \"string materials\" [ \"nope\" \"alsonope\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<lambertian *>(rec.mat.get()), nullptr);
}

// ===========================================================================
// Accelerator - a non-default splitmethod must produce the exact same hit
// results as the default SAH bvh_node build, over the same primitives - see
// bvh_aggregate_hittable.h's own top comment (only build strategy/tree shape
// changes, never the converged image).
// ===========================================================================

namespace {

// Several spheres spread across all 3 axes, so the BVH actually branches
// (not just one leaf) regardless of split method.
// AttributeBegin/End around each sphere resets the CTM, so each Translate
// below is an absolute world-space center, not cumulative with the others.
const std::string kScatteredSpheres =
	"Material \"diffuse\" \"rgb reflectance\" [ 0.6 0.3 0.2 ]\n"
	"AttributeBegin\nTranslate -4 0 0\nShape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n"
	"AttributeBegin\nTranslate 4 0 0\nShape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n"
	"AttributeBegin\nTranslate 0 4 0\nShape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n"
	"AttributeBegin\nTranslate 0 -4 0\nShape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n"
	"AttributeBegin\nTranslate 0 0 4\nShape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n"
	"AttributeBegin\nTranslate 0 0 -4\nShape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n";

} // namespace

class AcceleratorParityTest : public ::testing::TestWithParam<const char *> {};

TEST_P(AcceleratorParityTest, MatchesSahExactlyOverTheSameScene) {
	const pbrt_cpu::BuildResult sah = buildFrom(kScatteredSpheres);
	const pbrt_cpu::BuildResult alt = buildFrom(
		"Accelerator \"bvh\" \"string splitmethod\" \"" + std::string(GetParam()) +
		"\"\n" + kScatteredSpheres);

	// One ray toward each of the 6 spheres (at (-4,0,0)/(4,0,0)/(0,4,0)/
	// (0,-4,0)/(0,0,4)/(0,0,-4)), plus one that hits nothing.
	const struct { point3 origin; vec3 dir; } rays[] = {
		{point3(-4, 0, -10), vec3(0, 0, 1)},
		{point3(4, 0, -10), vec3(0, 0, 1)},
		{point3(0, 4, -10), vec3(0, 0, 1)},
		{point3(0, -4, -10), vec3(0, 0, 1)},
		{point3(-10, 0, 4), vec3(1, 0, 0)},
		{point3(0, 0, -10), vec3(0, 0, 1)},
		{point3(20, 20, 20), vec3(1, 0, 0)},
	};
	int rayIdx = 0;
	for (const auto &rd : rays) {
		SCOPED_TRACE(::testing::Message() << "rayIdx=" << rayIdx++);
		hit_record recSah, recAlt;
		const ray r(rd.origin, rd.dir);
		bool hitSah = sah.world->hit(r, interval(0.001, infinity), recSah);
		bool hitAlt = alt.world->hit(r, interval(0.001, infinity), recAlt);
		ASSERT_EQ(hitSah, hitAlt);
		if (!hitSah) continue;
		EXPECT_NEAR(recSah.t, recAlt.t, 1e-9);
		EXPECT_NEAR(recSah.normal.x(), recAlt.normal.x(), 1e-9);
		EXPECT_NEAR(recSah.normal.y(), recAlt.normal.y(), 1e-9);
		EXPECT_NEAR(recSah.normal.z(), recAlt.normal.z(), 1e-9);
		EXPECT_NEAR(recSah.u, recAlt.u, 1e-9);
		EXPECT_NEAR(recSah.v, recAlt.v, 1e-9);
		ASSERT_NE(recAlt.mat, nullptr);
		EXPECT_NE(dynamic_cast<lambertian *>(recAlt.mat.get()), nullptr);
	}
}

INSTANTIATE_TEST_SUITE_P(SplitMethods, AcceleratorParityTest,
						  ::testing::Values("middle", "equal", "hlbvh"),
						  [](const ::testing::TestParamInfo<const char *> &info) {
							  return std::string(info.param);
						  });

TEST(PbrtCpuBuildTest, AcceleratorFallsBackToSahForAMovingSphereEvenIfRequested) {
	// flatten() already forces acceleratorSplitMethod back to "sah" when the
	// scene has object motion blur (pbrt_flatten_tests.cpp pins this) - this
	// confirms the CPU builder actually honors that resolved value end to
	// end: a build that explicitly asked for "hlbvh" on a moving-sphere
	// scene must match a build of the exact same scene with no Accelerator
	// override at all (both should silently resolve to the same "sah"
	// behavior), NOT diverge by rendering the sphere frozen at time 0 the
	// way routing the moving sphere through bvh_aggregate_hittable (no
	// ray-time channel) would.
	const std::string kMovingSphere =
		"ActiveTransform \"StartTime\"\n"
		"ActiveTransform \"EndTime\"\n"
		"Translate 4 0 0\n"
		"ActiveTransform \"All\"\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n";
	const pbrt_cpu::BuildResult plain = buildFrom(kMovingSphere);
	const pbrt_cpu::BuildResult withOverride = buildFrom(
		"Accelerator \"bvh\" \"string splitmethod\" \"hlbvh\"\n" + kMovingSphere);

	for (double t : {0.0, 0.5, 1.0}) {
		hit_record recPlain, recOverride;
		const ray r(point3(1, 0, -10), vec3(0, 0, 1), t);
		bool hitPlain = plain.world->hit(r, interval(0.001, infinity), recPlain);
		bool hitOverride = withOverride.world->hit(r, interval(0.001, infinity), recOverride);
		ASSERT_EQ(hitPlain, hitOverride) << "t=" << t;
		if (!hitPlain) continue;
		EXPECT_NEAR(recPlain.t, recOverride.t, 1e-9) << "t=" << t;
	}
}

TEST(PbrtCpuBuildTest, AcceleratorNonSahRoutesAParticipatingMediumWithoutDroppingScatterEvents) {
	// Regression test for a real bug caught by code review: the first
	// bvh_aggregate_hittable.h design re-queried the winning primitive's
	// hit() a SECOND time to recover the full hit_record BvhHit's slim
	// interface can't carry. constant_medium::hit() (like grid/rgb-grid
	// medium's own hit()) samples its scatter distance via random_double()
	// INSIDE hit() itself - a second independent call draws a fresh random
	// sample, so it could legitimately disagree with (or, on very roughly
	// half of all such rays, simply fail to reproduce) the first call's
	// result, silently punching holes through the medium. The fix caches
	// the ORIGINAL call's own hit_record instead of re-querying - this test
	// casts many rays guaranteed to enter a dense medium-wrapped sphere and
	// confirms every single one still scatters, not just "most of them".
	const pbrt_cpu::BuildResult b = buildFrom(
		"Accelerator \"bvh\" \"string splitmethod\" \"hlbvh\"\n"
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0 0 0 ] \"rgb sigma_s\" [ 500 500 500 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(b.sphereCount, 1u);
	for (int i = 0; i < 200; ++i) {
		hit_record rec;
		ASSERT_TRUE(b.world->hit(ray(point3(0, 0, -5), vec3(0, 0, 1)),
								 interval(0.001, infinity), rec))
			<< "iteration " << i << ": a sigma_s=500 medium across a "
			   "radius-1 sphere (optical depth ~1000) must scatter on "
			   "essentially every ray - a missed hit here means the "
			   "medium's random free-path sample got silently re-drawn "
			   "and lost, not that this particular ray genuinely escaped";
	}
}
