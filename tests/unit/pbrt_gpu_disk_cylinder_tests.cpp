/**
 * @file pbrt_gpu_disk_cylinder_tests.cpp
 * @brief Tests for the GPU-recursive Disk/Cylinder primitive (Round 3 Phase 4b)
 *
 * Two tiers, matching this project's own established split for pbrt GPU
 * builder work (see pbrt_gpu_quad_light_tests.cpp):
 *
 *  - Host-side field-correctness tests: pbrt_gpu::build()'s new disk/cylinder
 *    loop (pbrt_gpu_builder.h) produces DiskData/CylinderData with the right
 *    radius/phiMax/materialIdx, and an o2w/w2o pair that are genuine inverses
 *    of each other - no GPU hardware required, runs everywhere.
 *  - A real end-to-end OptiX-recursive render: proves the new GAS/SBT/
 *    program-group wiring in optix_renderer.cpp actually works on real
 *    hardware, not just that the host-side data looks right. There is no
 *    scene-registry entry for an ad hoc scene, so this calls OptiXRenderer
 *    directly (buildScene()+render()) rather than going through
 *    optix_render_main()'s scene_id dispatch - the same reason pbrt_scene_
 *    render_test.cpp (CPU side) calls pbrt_cpu_builder directly instead of
 *    cpu_render_main().
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "optix_renderer.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

extern "C" {
	#include "optix_interface.h"
}

#include <cmath>

namespace {

pbrt_flatten::FlatScene flattenSource(const std::string &text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

// Multiplies two row-major 3x4 affine transforms (implicit [0,0,0,1] bottom
// row), the same convention DiskData::o2w/w2o (optix_types.h) and
// SceneData::InstancePlacementGPU::transform both use.
void mul3x4(const float a[12], const float b[12], float out[12]) {
	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 4; ++col) {
			float sum = (col == 3) ? a[row * 4 + 3] : 0.0f;
			for (int k = 0; k < 3; ++k) sum += a[row * 4 + k] * b[k * 4 + col];
			out[row * 4 + col] = sum;
		}
	}
}

void expectApproxIdentity3x4(const float m[12]) {
	const float expected[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
	for (int i = 0; i < 12; ++i) EXPECT_NEAR(m[i], expected[i], 1e-3f) << "at index " << i;
}

} // namespace

// ---------------------------------------------------------------------------
// Host-side field correctness
// ---------------------------------------------------------------------------

TEST(PbrtGpuDiskCylinderTest, DiskFieldsMatchTheFlattenedShape) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Translate 0 0 10\n"
		"Shape \"disk\" \"float radius\" [ 3 ] \"float innerradius\" [ 1 ] "
		"\"float height\" [ 2 ] \"float phimax\" [ 270 ]\n");
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	ASSERT_EQ(stats.disks, 1u);
	ASSERT_EQ(scene.disks.size(), 1u);

	const DiskData &d = scene.disks[0];
	EXPECT_FLOAT_EQ(d.radius, 3.0f);
	EXPECT_FLOAT_EQ(d.innerRadius, 1.0f);
	EXPECT_FLOAT_EQ(d.height, 2.0f);
	EXPECT_NEAR(d.phiMax, 270.0f * (3.14159265358979323846f / 180.0f), 1e-4f);
	// Translate 0 0 10 -> the world-space translation column of o2w.
	EXPECT_NEAR(d.o2w[3], 0.0f, 1e-4f);
	EXPECT_NEAR(d.o2w[7], 0.0f, 1e-4f);
	EXPECT_NEAR(d.o2w[11], 10.0f, 1e-4f);
}

TEST(PbrtGpuDiskCylinderTest, CylinderFieldsMatchTheFlattenedShape) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Shape \"cylinder\" \"float radius\" [ 2 ] \"float zmin\" [ -5 ] "
		"\"float zmax\" [ 5 ] \"float phimax\" [ 180 ]\n");
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	ASSERT_EQ(stats.cylinders, 1u);
	ASSERT_EQ(scene.cylinders.size(), 1u);

	const CylinderData &c = scene.cylinders[0];
	EXPECT_FLOAT_EQ(c.radius, 2.0f);
	EXPECT_FLOAT_EQ(c.zMin, -5.0f);
	EXPECT_FLOAT_EQ(c.zMax, 5.0f);
	EXPECT_NEAR(c.phiMax, 180.0f * (3.14159265358979323846f / 180.0f), 1e-4f);
}

TEST(PbrtGpuDiskCylinderTest, W2OIsTheGenuineInverseOfO2WUnderRotationAndTranslation) {
	// The whole point of DiskData/CylinderData carrying an unbaked transform
	// (see optix_types.h's own comment) is exact behavior under rotation,
	// unlike Sphere's baked-and-approximated one - so this specifically
	// exercises a rotated-AND-translated disk, not just a translated one.
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Translate 5 -2 3\nRotate 40 0 1 0\nShape \"disk\" \"float radius\" [ 1 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.disks.size(), 1u);

	float roundTrip[12];
	mul3x4(scene.disks[0].o2w, scene.disks[0].w2o, roundTrip);
	expectApproxIdentity3x4(roundTrip);
}

TEST(PbrtGpuDiskCylinderTest, MaterialAndAreaLightAreResolved) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"disk\"\n"
		"  Shape \"cylinder\"\n"
		"AttributeEnd\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.disks.size(), 1u);
	ASSERT_EQ(scene.cylinders.size(), 1u);

	ASSERT_GE(scene.disks[0].materialIdx, 0);
	ASSERT_GE(scene.cylinders[0].materialIdx, 0);
	EXPECT_EQ(scene.materials[scene.disks[0].materialIdx].type, MaterialType::DiffuseLight);
	EXPECT_EQ(scene.materials[scene.cylinders[0].materialIdx].type, MaterialType::DiffuseLight);
}

// A disk/cylinder AreaLightSource must be registered for real NEE sampling
// (GpuLightKind::Disk/Cylinder), not merely emit when directly hit - see
// docs/PBRT_SUPPORT.md's disk/cylinder entry and optix_disk_cylinder_
// helpers.h. A non-emissive disk/cylinder (no AreaLightSource) must NOT be
// registered at all, matching every other shape kind's own convention.
TEST(PbrtGpuDiskCylinderTest, EmissiveDiskAndCylinderAreRegisteredAsNeeLights) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"disk\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 3 3 3 ]\n"
		"  Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ] \"float zmax\" [ 2 ]\n"
		"AttributeEnd\n"
		// A non-emissive disk must not show up in the light list at all.
		"Material \"diffuse\"\n"
		"Shape \"disk\" \"float radius\" [ 1 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.disks.size(), 2u);
	ASSERT_EQ(scene.cylinders.size(), 1u);
	ASSERT_EQ(scene.lightIndices.size(), scene.lightKinds.size());
	ASSERT_EQ(scene.lightIndices.size(), 2u);

	EXPECT_EQ(scene.lightKinds[0], GpuLightKind::Disk);
	EXPECT_EQ(scene.lightIndices[0], 0);
	EXPECT_EQ(scene.lightKinds[1], GpuLightKind::Cylinder);
	EXPECT_EQ(scene.lightIndices[1], 0);
}

// MediumInterface on a cylinder now resolves to a real MaterialType::Medium
// (see optix_intersection_disk_cylinder.h/wavefront_programs.cu's Medium
// near/far re-intersection and docs/PBRT_SUPPORT.md's cylinder-medium
// entry) - not the shape's own declared surface Material at all, matching
// how the sphere loop already resolves s.medium via mediumMaterialIndex().
// Disk's own medium field stays intentionally unresolved (structurally not
// meaningful - a zero-thickness plane has no "inside" volume).
TEST(PbrtGpuDiskCylinderTest, CylinderMediumInterfaceResolvesToRealMediumMaterial) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" [ \"homogeneous\" ]\n"
		"  \"rgb sigma_a\" [ 0.1 0.1 0.1 ] \"rgb sigma_s\" [ 1.0 1.0 1.0 ]\n"
		"AttributeBegin\n"
		"  Material \"dielectric\" \"float eta\" [ 1.001 ]\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ] \"float zmax\" [ 2 ]\n"
		"AttributeEnd\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.cylinders.size(), 1u);
	ASSERT_GE(scene.cylinders[0].materialIdx, 0);
	EXPECT_EQ(scene.materials[scene.cylinders[0].materialIdx].type, MaterialType::Medium);
}

TEST(PbrtGpuDiskCylinderTest, ObjectMotionBlurIsCountedAsUnsupportedAndRendersStatic) {
	// See BuildStats::animatedDiskCylinderCount's own comment - CPU has real
	// object motion blur for disk/cylinder (disk_cylinder_hittable.h's
	// AnimatedTransform use), neither GPU backend does yet. o2w must resolve
	// to the StartTime position (unaffected by xformEnd), and the shape must
	// be counted so scene_builder.cpp can warn instead of silently dropping
	// the motion.
	const pbrt_flatten::FlatScene flat = flattenSource(
		"ActiveTransform \"StartTime\"\n"
		"ActiveTransform \"EndTime\"\n"
		"Translate 5 0 0\n"
		"ActiveTransform \"All\"\n"
		"Shape \"disk\" \"float radius\" [ 1 ]\n"
		"Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ -1 ] "
		"\"float zmax\" [ 1 ]\n");
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	EXPECT_EQ(stats.animatedDiskCylinderCount, 2);
	ASSERT_EQ(scene.disks.size(), 1u);
	ASSERT_EQ(scene.cylinders.size(), 1u);
	// StartTime translation column is 0 (no shift) - EndTime's Translate 5 0 0
	// must NOT have been picked up.
	EXPECT_NEAR(scene.disks[0].o2w[3], 0.0f, 1e-6f);
	EXPECT_NEAR(scene.cylinders[0].o2w[3], 0.0f, 1e-6f);
}

TEST(PbrtGpuDiskCylinderTest, StaticDiskAndCylinderAreNotCountedAsAnimated) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Shape \"disk\" \"float radius\" [ 1 ]\n"
		"Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ -1 ] "
		"\"float zmax\" [ 1 ]\n");
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	EXPECT_EQ(stats.animatedDiskCylinderCount, 0);
}

// ---------------------------------------------------------------------------
// Real GPU render (skipped when no OptiX-capable device is present) - both
// backends, since Phase 4c ports Disk/Cylinder to wavefront on top of
// Phase 4b's recursive-backend support.
// ---------------------------------------------------------------------------

namespace {

// Same scene as PbrtSceneRender.DiskAndCylinderRenderWithoutNaN
// (tests/integration/pbrt_scene_render_test.cpp) - a disk ceiling light
// (rotated flat via Translate+Rotate, exactly the transform combination
// W2OIsTheGenuineInverseOfO2WUnderRotationAndTranslation above checks) plus
// a free-floating cylinder, so both new GPU program-group/SBT regions (disk
// AND cylinder) are exercised by the same launch.
const char *kDiskCylinderScene =
	"LookAt 0 1 -4    0 1 0    0 1 0\n"
	"Camera \"perspective\" \"float fov\" [ 50 ]\n"
	"WorldBegin\n"
	"AttributeBegin\n"
	"  AreaLightSource \"diffuse\" \"rgb L\" [ 12 12 12 ]\n"
	"  Translate 0 2.0 0\n"
	"  Rotate 90 1 0 0\n"
	"  Shape \"disk\" \"float radius\" [ 0.6 ]\n"
	"AttributeEnd\n"
	"Material \"diffuse\" \"rgb reflectance\" [ 0.75 0.75 0.75 ]\n"
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
	"  \"point3 P\" [ -2 0 -2   2 0 -2   2 0 2   -2 0 2 ]\n"
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
	"  \"point3 P\" [ -2 0 2   2 0 2   2 3 2   -2 3 2 ]\n"
	"Translate 0 0.5 0\n"
	"Shape \"cylinder\" \"float radius\" [ 0.4 ] \"float zmin\" [ -0.5 ] \"float zmax\" [ 0.5 ]\n";

constexpr int kRenderW = 32, kRenderH = 32;

// Pinhole camera, same math as scene_builder.cpp's own (file-local)
// build_pinhole_camera_params(), replicated here since that helper isn't
// exported - matches kDiskCylinderScene's own LookAt/fov exactly.
GpuCameraParams pinholeCameraFor(const pbrt_flatten::FlatScene &flat) {
	const float3 lookfrom = make_float3(static_cast<float>(flat.camera.lookfrom[0]),
										static_cast<float>(flat.camera.lookfrom[1]),
										static_cast<float>(flat.camera.lookfrom[2]));
	const float3 lookat = make_float3(static_cast<float>(flat.camera.lookat[0]),
									  static_cast<float>(flat.camera.lookat[1]),
									  static_cast<float>(flat.camera.lookat[2]));
	const float3 vup = make_float3(static_cast<float>(flat.camera.up[0]),
								   static_cast<float>(flat.camera.up[1]),
								   static_cast<float>(flat.camera.up[2]));
	constexpr float kPi = 3.14159265358979323846f;
	const float aspect = static_cast<float>(kRenderW) / static_cast<float>(kRenderH);
	const float theta = static_cast<float>(flat.camera.vfov) * kPi / 180.0f;
	const float h = tanf(theta / 2.0f);
	const float viewport_height = 2.0f * h;
	const float viewport_width = aspect * viewport_height;
	const float3 view_dir = make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z);
	const float3 w = normalize(view_dir);
	const float3 u = normalize(cross(vup, w));
	const float3 v = cross(w, u);
	const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
	const float3 vertical = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
	const float3 lower_left = make_float3(
		lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
		lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
		lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z);

	GpuCameraParams cam{};
	cam.kind = CameraKind::Perspective;
	cam.origin = lookfrom;
	cam.lower_left_corner = lower_left;
	cam.horizontal = horizontal;
	cam.vertical = vertical;
	return cam;
}

void expectFiniteAndLit(const std::vector<float> &framebuffer, const char *backendName) {
	bool allFinite = true;
	float maxVal = 0.0f;
	double sum = 0.0;
	for (float v : framebuffer) {
		if (!std::isfinite(v)) allFinite = false;
		maxVal = std::max(maxVal, v);
		sum += v;
	}
	EXPECT_TRUE(allFinite) << backendName << ": a NaN or infinity reached the GPU framebuffer";
	EXPECT_GT(maxVal, 0.0f) << backendName << ": the frame is entirely black - disk light or "
		"cylinder geometry may not be where the scene says it is, or the new GAS/SBT region "
		"never got traced";
	EXPECT_GT(sum / framebuffer.size(), 0.0) << backendName << ": the frame is entirely black";
}

} // namespace

TEST(PbrtGpuDiskCylinderRenderTest, DiskLightAndCylinderRenderWithoutNaN) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available on this system";
	}

	const pbrt_flatten::FlatScene flat = flattenSource(kDiskCylinderScene);
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	ASSERT_EQ(stats.disks, 1u);
	ASSERT_EQ(stats.cylinders, 1u);
	ASSERT_FALSE(scene.triangles.empty());

	OptiXRenderer renderer;
	ASSERT_TRUE(renderer.initialize()) << "OptiX device init failed";
	ASSERT_TRUE(renderer.buildScene(
		scene.spheres, scene.quads, scene.materials,
		scene.lightIndices, scene.lightKinds, scene.punctualLights,
		scene.bilinearPatches, scene.triangles, scene.disks, scene.cylinders))
		<< "buildScene() failed - GAS/SBT wiring for the new disk/cylinder region";

	const GpuCameraParams cam = pinholeCameraFor(flat);
	std::vector<float> framebuffer(static_cast<std::size_t>(kRenderW) * kRenderH * 3);
	ASSERT_TRUE(renderer.render(kRenderW, kRenderH, /*spp=*/32, /*maxDepth=*/8, cam, framebuffer.data()))
		<< "render() failed";

	expectFiniteAndLit(framebuffer, "recursive");
}

TEST(PbrtGpuDiskCylinderRenderTest, WavefrontDiskLightAndCylinderRenderWithoutNaN) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available on this system";
	}

	const pbrt_flatten::FlatScene flat = flattenSource(kDiskCylinderScene);
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	ASSERT_EQ(stats.disks, 1u);
	ASSERT_EQ(stats.cylinders, 1u);
	ASSERT_FALSE(scene.triangles.empty());

	OptiXRenderer renderer;
	ASSERT_TRUE(renderer.initialize()) << "OptiX device init failed";
	ASSERT_TRUE(renderer.buildScene(
		scene.spheres, scene.quads, scene.materials,
		scene.lightIndices, scene.lightKinds, scene.punctualLights,
		scene.bilinearPatches, scene.triangles, scene.disks, scene.cylinders))
		<< "buildScene() failed - GAS/SBT wiring for the new disk/cylinder region";

	// Empty ptxPath: enableWavefront() looks for wavefront_programs.ptx next
	// to optix_programs.ptx, same default gpu_render_tests.cpp's own
	// RAY_TRACER_WAVEFRONT=1 path relies on.
	renderer.enableWavefront(true);

	const GpuCameraParams cam = pinholeCameraFor(flat);
	std::vector<float> framebuffer(static_cast<std::size_t>(kRenderW) * kRenderH * 3);
	ASSERT_TRUE(renderer.render(kRenderW, kRenderH, /*spp=*/32, /*maxDepth=*/8, cam, framebuffer.data()))
		<< "render() failed - this is exactly the SBT-layout risk "
		   "WavefrontPathTracer::buildSBT()'s disk/cylinder trailing-region "
		   "placement (mirroring OptiXRenderer::buildScene()'s "
		   "diskCylinderSbtOffset) exists to avoid";

	expectFiniteAndLit(framebuffer, "wavefront");
}

// ---------------------------------------------------------------------------
// Regression test for the wavefront SBT stride mismatch found in code review:
// WavefrontPathTracer::buildSBT() padded every present type-group to only 2
// records, but the shared IAS's baked instance.sbtOffset values are computed
// by OptiXRenderer::buildScene() using stride=RAY_TYPE_COUNT(3) - correct
// only when at most one type-group precedes the affected one. kDiskCylinderScene
// above (triangle, then disk, then cylinder - one group before disk/cylinder)
// happened to stay inside that safe margin and passed even with the bug
// present. This scene adds a SPHERE ahead of the triangle, putting TWO
// type-groups (sphere, then triangle) before the disk/cylinder region, which
// is exactly the case that spilled disk hits into the cylinder's own SBT
// records (and cylinder hits one record past the end of the array) before
// buildSBT()'s pushTriple fix (padding every group to RAY_TYPE_COUNT records
// so its own cumulative offsets match the shared baked ones exactly, for any
// number of preceding groups).
// ---------------------------------------------------------------------------
TEST(PbrtGpuDiskCylinderRenderTest, WavefrontSphereTriangleDiskCylinderRenderWithoutNaN) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available on this system";
	}

	const pbrt_flatten::FlatScene flat = flattenSource(
		"LookAt 0 1 -4    0 1 0    0 1 0\n"
		"Camera \"perspective\" \"float fov\" [ 50 ]\n"
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 12 12 12 ]\n"
		"  Translate 0 2.0 0\n"
		"  Rotate 90 1 0 0\n"
		"  Shape \"disk\" \"float radius\" [ 0.6 ]\n"
		"AttributeEnd\n"
		"Material \"diffuse\" \"rgb reflectance\" [ 0.75 0.75 0.75 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ -2 0 -2   2 0 -2   2 0 2   -2 0 2 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ -2 0 2   2 0 2   2 3 2   -2 3 2 ]\n"
		"Translate -0.8 0.4 0\n"
		"Shape \"sphere\" \"float radius\" [ 0.4 ]\n"
		"Translate 0.8 0.1 0\n"
		"Shape \"cylinder\" \"float radius\" [ 0.4 ] \"float zmin\" [ -0.5 ] \"float zmax\" [ 0.5 ]\n");

	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	ASSERT_EQ(stats.spheres, 1u);
	ASSERT_EQ(stats.disks, 1u);
	ASSERT_EQ(stats.cylinders, 1u);
	ASSERT_FALSE(scene.triangles.empty());

	OptiXRenderer renderer;
	ASSERT_TRUE(renderer.initialize()) << "OptiX device init failed";
	ASSERT_TRUE(renderer.buildScene(
		scene.spheres, scene.quads, scene.materials,
		scene.lightIndices, scene.lightKinds, scene.punctualLights,
		scene.bilinearPatches, scene.triangles, scene.disks, scene.cylinders))
		<< "buildScene() failed - GAS/SBT wiring for the new disk/cylinder region";

	renderer.enableWavefront(true);

	const GpuCameraParams cam = pinholeCameraFor(flat);
	std::vector<float> framebuffer(static_cast<std::size_t>(kRenderW) * kRenderH * 3);
	ASSERT_TRUE(renderer.render(kRenderW, kRenderH, /*spp=*/32, /*maxDepth=*/8, cam, framebuffer.data()))
		<< "render() failed";

	expectFiniteAndLit(framebuffer, "wavefront sphere+triangle+disk+cylinder");
}
