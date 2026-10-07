/**
 * @file scene_builder_tests.cpp
 * @brief The Scene Builder's document model (src/shared/scene_document.h): JSON round trip, pbrt output, validation, and what the renderer makes of it
 *
 * The first groups need nothing but the header. The last group writes a builder scene to disk and renders it with the launcher (ray_tracer.exe, looked for
 * next to the test binary and in the usual build output folders) against a closed form; it skips when the launcher is not built.
 */

#include <gtest/gtest.h>

#include "agreement_test_helpers.h"
#include "../../src/shared/scene_document.h"
#include "../../src/shared/pbrt_load.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using scene_doc::Document;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::Problem;
using scene_doc::ShapeKind;
using scene_doc::fromJson;
using scene_doc::fromPbrt;
using scene_doc::hasErrors;
using scene_doc::makeAreaLightPanel;
using scene_doc::makeObject;
using scene_doc::makeStarterScene;
using scene_doc::toJson;
using scene_doc::toPbrt;
using scene_doc::toString;
using scene_doc::validate;

namespace {

std::string slurp(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	std::ostringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

// A document touching every field, with awkward strings, so a field the writer or reader forgets shows up as a difference.
Document busyDocument() {
	Document d = makeStarterScene();
	d.title = "Quote \" backslash \\ tab \t and unicode \xC3\xA9";
	d.camera.position = {1.25, -2.5, 3.125};
	d.camera.lensRadius = 0.05;
	d.camera.focusDistance = 6.5;
	d.render = {321, 123, 17, 5};
	Object cyl = makeObject(ShapeKind::Cylinder, "Tall \"one\"");
	cyl.rotation = {10, 20, 30};
	cyl.material.kind = MaterialKind::CoatedDiffuse;
	cyl.material.roughness = 0.3;
	cyl.material.ior = 1.7;
	cyl.emissive = true;
	cyl.twoSided = true;
	cyl.emissionStrength = 3.5;
	d.objects.push_back(cyl);
	Object dt = makeObject(ShapeKind::Disk, "Shell");
	dt.material.kind = MaterialKind::DiffuseTransmission;
	dt.material.transmittance = {0.1, 0.2, 0.3};
	d.objects.push_back(dt);
	Object mesh = makeObject(ShapeKind::Mesh, "Mesh");
	mesh.meshFile = "C:\\models\\bunny.ply";
	mesh.meshScale = 2.5;
	d.objects.push_back(mesh);
	Light spot;
	spot.name = "Spot";
	spot.kind = LightKind::Spot;
	spot.coneAngle = 25;
	spot.coneDelta = 4;
	d.lights.push_back(spot);
	Light env;
	env.kind = LightKind::Infinite;
	env.imageFile = "sky.exr";
	d.lights.push_back(env);
	return d;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------------------------------------------------------------

TEST(SceneDocumentTest, JsonRoundTripIsExact) {
	const Document a = busyDocument();
	const std::string j1 = toJson(a);
	Document b;
	std::string err;
	ASSERT_TRUE(fromJson(j1, b, err)) << err;
	EXPECT_EQ(toJson(b), j1);
	EXPECT_EQ(b.title, a.title);
	ASSERT_EQ(b.objects.size(), a.objects.size());
	EXPECT_EQ(b.objects[5].name, "Tall \"one\"");
	EXPECT_EQ(b.objects[7].meshFile, "C:\\models\\bunny.ply");
	EXPECT_EQ(b.objects[5].material.kind, MaterialKind::CoatedDiffuse);
	EXPECT_TRUE(b.objects[5].twoSided);
	EXPECT_EQ(b.render.width, 321);
	EXPECT_EQ(b.lights.back().imageFile, "sky.exr");
}

TEST(SceneDocumentTest, PbrtTextCarriesTheDocumentAndReadsBack) {
	const Document a = busyDocument();
	const std::string text = toPbrt(a);
	Document b;
	std::string err;
	ASSERT_TRUE(fromPbrt(text, b, err)) << err;
	EXPECT_EQ(toJson(b), toJson(a));
	// Writing is deterministic.
	EXPECT_EQ(toPbrt(b), text);
}

TEST(SceneDocumentTest, AFileWithoutBuilderDataIsRefusedWithAReason) {
	Document d;
	std::string err;
	EXPECT_FALSE(fromPbrt("LookAt 0 0 5  0 0 0  0 1 0\nCamera \"perspective\"\nWorldBegin\n", d, err));
	EXPECT_NE(err.find("not made by the Scene Builder"), std::string::npos) << err;
}

TEST(SceneDocumentTest, DamagedJsonIsRefusedNotHalfLoaded) {
	Document d = makeStarterScene();
	const std::string good = toJson(d);
	for (const std::string& bad : {std::string(), std::string("{"), good.substr(0, good.size() / 2), std::string("[1,2]"),
	                               std::string("{\"objects\":[{\"shape\":\"teapot\"}]}"), std::string("{\"camera\":{\"fov\":\"wide\"}}"),
	                               std::string("{\"version\":2}")}) {
		Document out;
		out.title = "untouched";
		std::string err;
		EXPECT_FALSE(fromJson(bad, out, err)) << bad;
		EXPECT_FALSE(err.empty()) << bad;
		EXPECT_EQ(out.title, "untouched") << "a failed read must not change the document: " << bad;
	}
}

TEST(SceneDocumentTest, MissingFieldsKeepTheirDefaults) {
	Document d;
	std::string err;
	ASSERT_TRUE(fromJson("{\"objects\":[{\"name\":\"Ball\",\"shape\":\"sphere\"}]}", d, err)) << err;
	ASSERT_EQ(d.objects.size(), 1u);
	EXPECT_EQ(d.objects[0].radius, 1.0);
	EXPECT_EQ(d.objects[0].material.kind, MaterialKind::Diffuse);
	EXPECT_EQ(d.render.width, 800);
}

// The JSON copy (and so every undo snapshot) must not round: a value needing 17 digits comes back bit for bit.
TEST(SceneDocumentTest, JsonKeepsEveryDigitOfANumber) {
	Document a = makeStarterScene();
	a.camera.position.x = 1.2345678901234567;
	a.objects[1].radius = 0.1 + 0.2;  // 0.30000000000000004
	a.objects[2].position.y = 1e-7 / 3.0;
	Document b;
	std::string err;
	ASSERT_TRUE(fromJson(toJson(a), b, err)) << err;
	EXPECT_EQ(b.camera.position.x, a.camera.position.x);
	EXPECT_EQ(b.objects[1].radius, a.objects[1].radius);
	EXPECT_EQ(b.objects[2].position.y, a.objects[2].position.y);
	// A short number stays short.
	EXPECT_NE(toJson(makeStarterScene()).find("\"fov\":40,"), std::string::npos);
}

// A name that contains the marker word must not be mistaken for the line that holds the document.
TEST(SceneDocumentTest, ANameContainingTheMarkerCannotHideTheDocument) {
	Document a = makeStarterScene();
	a.title = "@rt-builder-doc not json";
	a.objects[0].name = "# @rt-builder-doc {\"title\":\"evil\"}";
	a.lights[0].name = "x # @rt-builder-doc y";
	const std::string text = toPbrt(a);
	Document b;
	std::string err;
	ASSERT_TRUE(fromPbrt(text, b, err)) << err;
	EXPECT_EQ(b.title, a.title) << "the document comes from the real line, with the names as typed";
	EXPECT_EQ(b.objects[0].name, a.objects[0].name);
	// Exactly one line starts with the marker (names also appear, escaped, inside that line's JSON).
	size_t lines = text.compare(0, 18, "# @rt-builder-doc ") == 0 ? 1 : 0;
	for (size_t at = text.find("\n# @rt-builder-doc "); at != std::string::npos; at = text.find("\n# @rt-builder-doc ", at + 1)) ++lines;
	EXPECT_EQ(lines, 1u);
}

// A comma as the decimal separator (a German or French locale) must not leak into the file: pbrt and JSON both want a point.
TEST(SceneDocumentTest, NumbersNeverUseTheProcessLocale) {
	Document d = makeStarterScene();
	d.camera.fov = 37.5;
	const std::string text = toPbrt(d);
	EXPECT_NE(text.find("\"float fov\" [ 37.5 ]"), std::string::npos);
	EXPECT_EQ(text.find("37,5"), std::string::npos);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------------------------------------------

namespace {
bool mentions(const std::vector<Problem>& ps, Problem::Severity sev, const std::string& needle) {
	for (const Problem& p : ps)
		if (p.severity == sev && p.message.find(needle) != std::string::npos) return true;
	return false;
}
}  // namespace

TEST(SceneDocumentValidateTest, TheStarterSceneHasNoProblems) {
	const auto ps = validate(makeStarterScene());
	for (const Problem& p : ps) ADD_FAILURE() << p.message;
}

TEST(SceneDocumentValidateTest, CameraProblemsAreErrors) {
	Document d = makeStarterScene();
	d.camera.target = d.camera.position;
	EXPECT_TRUE(mentions(validate(d), Problem::Severity::Error, "camera is at its target"));
	d = makeStarterScene();
	d.camera.position = {0, 5, 0};
	d.camera.target = {0, 0, 0};
	d.camera.up = {0, 1, 0};
	EXPECT_TRUE(mentions(validate(d), Problem::Severity::Error, "straight along its up"));
	d = makeStarterScene();
	d.camera.fov = 0.0;
	EXPECT_TRUE(hasErrors(validate(d)));
	d = makeStarterScene();
	d.camera.lensRadius = 0.1;
	d.camera.focusDistance = 0.0;
	EXPECT_TRUE(hasErrors(validate(d)));
}

TEST(SceneDocumentValidateTest, ObjectAndLightProblems) {
	Document d = makeStarterScene();
	d.objects[1].radius = 0.0;
	EXPECT_TRUE(mentions(validate(d), Problem::Severity::Error, "Glass ball"));
	d = makeStarterScene();
	d.objects.push_back(makeObject(ShapeKind::Mesh, "Bunny"));
	EXPECT_TRUE(mentions(validate(d), Problem::Severity::Error, "choose a .ply"));
	d = makeStarterScene();
	d.objects[1].material.roughness = 1.5;
	EXPECT_TRUE(hasErrors(validate(d)));
	d = makeStarterScene();
	Light spot;
	spot.kind = LightKind::Spot;
	spot.position = spot.target;
	d.lights.push_back(spot);
	EXPECT_TRUE(mentions(validate(d), Problem::Severity::Error, "aims at"));
}

TEST(SceneDocumentValidateTest, ASceneWithNoLightIsAWarningNotAnError) {
	Document d = makeStarterScene();
	d.lights.clear();
	d.objects.pop_back();  // the ceiling panel
	const auto ps = validate(d);
	EXPECT_TRUE(mentions(ps, Problem::Severity::Warning, "no light"));
	EXPECT_FALSE(hasErrors(ps));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The pbrt text the renderer reads
// ---------------------------------------------------------------------------------------------------------------------------------

namespace {
// Writes text to a temporary scene file and loads it the way the renderer does.
pbrt_load::LoadResult loadText(const std::string& text, const char* tag) {
	const std::string path = std::string("scene_builder_test_") + tag + ".pbrt";
	{
		std::ofstream out(path, std::ios::binary);
		out << text;
	}
	pbrt_load::LoadResult r = pbrt_load::loadFile(path);
	std::remove(path.c_str());
	return r;
}
}  // namespace

TEST(SceneDocumentPbrtTest, EveryShapeMaterialAndLightLoads) {
	Document d = makeStarterScene();
	d.objects.clear();
	d.lights.clear();
	for (ShapeKind sh : {ShapeKind::Sphere, ShapeKind::Box, ShapeKind::Quad, ShapeKind::Disk, ShapeKind::Cylinder, ShapeKind::Cone})
		d.objects.push_back(makeObject(sh, toString(sh)));
	const MaterialKind kinds[] = {MaterialKind::Diffuse, MaterialKind::Conductor, MaterialKind::Dielectric, MaterialKind::CoatedDiffuse,
	                              MaterialKind::DiffuseTransmission};
	for (MaterialKind k : kinds) {
		Object o = makeObject(ShapeKind::Sphere, toString(k));
		o.material.kind = k;
		o.material.roughness = 0.2;
		d.objects.push_back(o);
	}
	Object checks = makeObject(ShapeKind::Sphere, "checks");
	checks.material.checker = true;
	d.objects.push_back(checks);
	d.objects.push_back(makeAreaLightPanel("panel"));
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite}) {
		Light l;
		l.name = toString(k);
		l.kind = k;
		l.target = {0, 0, 1};
		d.lights.push_back(l);
	}
	ASSERT_FALSE(hasErrors(validate(d)));
	const pbrt_load::LoadResult r = loadText(toPbrt(d), "everything");
	ASSERT_TRUE(r.ok) << r.error;
}

TEST(SceneDocumentPbrtTest, TheStarterSceneLoads) {
	const pbrt_load::LoadResult r = loadText(toPbrt(makeStarterScene()), "starter");
	ASSERT_TRUE(r.ok) << r.error;
}

TEST(SceneDocumentPbrtTest, ASceneNamedLikeADirectiveCannotInjectOne) {
	Document d = makeStarterScene();
	d.title = "evil\nWorldEnd";
	d.objects[1].name = "x\"\nShape \"sphere\" \"float radius\" [ 100 ]";
	const std::string text = toPbrt(d);
	// Names appear only in '#' comment lines; a newline in one must not start a new directive.
	std::istringstream in(text);
	std::string line;
	int shapes = 0;
	while (std::getline(in, line))
		if (line.rfind("  Shape", 0) == 0 || line.rfind("Shape", 0) == 0) ++shapes;
	EXPECT_EQ(shapes, 5) << "five objects, five Shape lines";
	const pbrt_load::LoadResult r = loadText(text, "inject");
	EXPECT_TRUE(r.ok) << r.error;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Rendered against a closed form
// ---------------------------------------------------------------------------------------------------------------------------------

namespace {

std::string findLauncher() {
	namespace fs = std::filesystem;
	std::error_code ec;
	for (const char* c : {"ray_tracer.exe", "../ray_tracer.exe", "../../ray_tracer.exe", "x64/Release/ray_tracer.exe", "../x64/Release/ray_tracer.exe",
	                      "../../x64/Release/ray_tracer.exe", "ray_tracer", "../ray_tracer", "../../ray_tracer"})
		if (fs::is_regular_file(c, ec)) return fs::absolute(c, ec).string();
	return "";
}

// Renders scene text with the launcher on the CPU; returns false if it cannot.
bool renderWithLauncher(const std::string& launcher, const std::string& sceneText, const char* tag, int size, int spp, int depth, std::vector<float>& rgb, int& w,
                        int& h) {
	const std::string scenePath = std::string("scene_builder_render_") + tag + ".pbrt";
	const std::string outPath = std::string("scene_builder_render_") + tag + ".exr";
	{
		std::ofstream out(scenePath, std::ios::binary);
		out << sceneText;
	}
	std::remove(outPath.c_str());
	std::ostringstream cmd;
	cmd << "\"" << launcher << "\" --cpu --output \"" << outPath << "\" " << size << " " << spp << " " << depth << " \"" << scenePath << "\" > "
	    << "scene_builder_render_" << tag << ".log 2>&1";
	std::string command = cmd.str();
#ifdef _WIN32
	command = "\"" + command + "\"";  // cmd.exe strips one pair of outer quotes
#endif
	const int rc = std::system(command.c_str());
	const bool ok = rc == 0 && loadLinearRgbPixels(outPath, w, h, rgb);
	if (!ok) ADD_FAILURE() << "launcher failed (exit " << rc << "); log:\n" << slurp(std::string("scene_builder_render_") + tag + ".log");
	std::remove(scenePath.c_str());
	std::remove(outPath.c_str());
	std::remove((std::string("scene_builder_render_") + tag + ".log").c_str());
	return ok;
}

double channelMean(const std::vector<float>& rgb, int c) {
	double s = 0;
	size_t n = rgb.size() / 3;
	for (size_t i = 0; i < n; ++i) s += rgb[3 * i + c];
	return s / static_cast<double>(n);
}

}  // namespace

// A diffuse sphere of albedo (0.5, 0.25, 0.75) fills the view under a uniform white sky of strength 1: at depth 1 every pixel reads its albedo.
// Everything here is made through the document model, so a wrong colour, scale or light multiplier in the writer fails it.
TEST(SceneBuilderRenderTest, DiffuseSphereUnderAUniformSkyReadsItsAlbedo) {
	const std::string launcher = findLauncher();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	Document d;
	d.camera.position = {0, 0, 10};
	d.camera.target = {0, 0, 0};
	d.camera.fov = 5;
	Object ball = makeObject(ShapeKind::Sphere, "Ball");
	ball.position = {0, 0, 0};
	ball.material.color = {0.5, 0.25, 0.75};
	d.objects.push_back(ball);
	Light sky;
	sky.kind = LightKind::Infinite;
	sky.color = {1, 1, 1};
	sky.intensity = 1.0;
	d.lights.push_back(sky);
	ASSERT_FALSE(hasErrors(validate(d)));
	std::vector<float> rgb;
	int w = 0, h = 0;
	ASSERT_TRUE(renderWithLauncher(launcher, toPbrt(d), "furnace", 32, 128, 1, rgb, w, h));
	const double want[3] = {0.5, 0.25, 0.75};
	for (int c = 0; c < 3; ++c) EXPECT_NEAR(channelMean(rgb, c), want[c], 0.015 * want[c] + 0.004) << "channel " << c;
}

// A sphere light of radiance L and radius r, its centre a distance d straight above a diffuse plate of albedo rho, makes the plate under it radiate
// rho * L * (r/d)^2 (irradiance pi * L * r^2/d^2, exactly, for a point straight below a sphere). The quad's front must face +Y for the plate to be lit.
TEST(SceneBuilderRenderTest, AreaLightOverAQuadMatchesTheClosedForm) {
	const std::string launcher = findLauncher();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	Document d;
	Object plate = makeObject(ShapeKind::Quad, "Plate");
	plate.size = {40, 1, 40};
	plate.material.color = {0.6, 0.6, 0.6};
	d.objects.push_back(plate);
	// The camera looks at the point under the light from the side, so the light is out of view.
	d.camera.position = {4.0, 5.0, 0.0};
	d.camera.target = {0.0, 0.0, 0.0};
	d.camera.up = {0, 1, 0};
	d.camera.fov = 1.0;
	Object bulb = makeObject(ShapeKind::Sphere, "Bulb");
	bulb.radius = 0.5;
	bulb.position = {0, 4.0, 0};
	bulb.emissive = true;
	bulb.emission = {1, 1, 1};
	bulb.emissionStrength = 20.0;
	bulb.material.color = {0, 0, 0};
	d.objects.push_back(bulb);
	ASSERT_FALSE(hasErrors(validate(d)));
	std::vector<float> rgb;
	int w = 0, h = 0;
	ASSERT_TRUE(renderWithLauncher(launcher, toPbrt(d), "plate", 24, 256, 1, rgb, w, h));
	const double r = 0.5, dist = 4.0, rho = 0.6, L = 20.0;
	const double expected = rho * L * (r * r) / (dist * dist);
	const double got = channelMean(rgb, 1);
	EXPECT_NEAR(got, expected, 0.04 * expected) << "expected rho*L*(r/d)^2 = " << expected;
}

// "New scene" must render to a lit, finite picture at once. RT_SCENE_BUILDER_STARTER_OUT=<file.pbrt> also writes the scene there, to look at it.
TEST(SceneBuilderRenderTest, TheStarterSceneRendersLitAndFinite) {
	const std::string launcher = findLauncher();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	const std::string text = toPbrt(makeStarterScene());
	if (const char* keep = std::getenv("RT_SCENE_BUILDER_STARTER_OUT")) {
		std::ofstream out(keep, std::ios::binary);
		out << text;
	}
	std::vector<float> rgb;
	int w = 0, h = 0;
	ASSERT_TRUE(renderWithLauncher(launcher, text, "starter", 64, 32, 6, rgb, w, h));
	for (int c = 0; c < 3; ++c) {
		const double m = channelMean(rgb, c);
		EXPECT_GT(m, 0.05) << "channel " << c << " is nearly black";
		EXPECT_LT(m, 3.0) << "channel " << c << " is blown out";
	}
}

// The orientation the header promises: a box, sphere, cylinder and cone emit outward, a quad and a disk emit from their +Y side. A camera whose view lies
// wholly inside the shape sees the emitter's radiance (strength 2) at depth 1, or nothing from behind it.
TEST(SceneBuilderRenderTest, EmissiveShapesFaceTheWayTheDocumentSays) {
	const std::string launcher = findLauncher();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	struct Case {
		ShapeKind shape;
		scene_doc::Float3 camera;  // looks at the origin
		double expected;
	};
	const double L = 2.0;
	const Case cases[] = {
		{ShapeKind::Sphere, {0, 0, 10}, L},
		{ShapeKind::Box, {0, 0, 10}, L},   {ShapeKind::Box, {10, 0.001, 0}, L},  {ShapeKind::Box, {0, 10, 0.001}, L}, {ShapeKind::Box, {0, -10, 0.001}, L},
		{ShapeKind::Quad, {0, 10, 0.001}, L},  {ShapeKind::Quad, {0, -10, 0.001}, 0.0},
		{ShapeKind::Disk, {0, 10, 0.001}, L},  {ShapeKind::Disk, {0, -10, 0.001}, 0.0},
		{ShapeKind::Cylinder, {0, 0, 10}, L},  {ShapeKind::Cylinder, {10, 0, 0}, L},
		{ShapeKind::Cone, {0, 0, 10}, L},
	};
	for (const Case& c : cases) {
		Document d;
		d.camera.position = c.camera;
		d.camera.target = {0, 0, 0};
		d.camera.fov = 3;
		Object o = makeObject(c.shape, toString(c.shape));
		o.position = {0, 0, 0};
		o.emissive = true;
		o.emission = {1, 1, 1};
		o.emissionStrength = L;
		o.material.color = {0, 0, 0};
		d.objects.push_back(o);
		ASSERT_FALSE(hasErrors(validate(d)));
		std::vector<float> rgb;
		int w = 0, h = 0;
		ASSERT_TRUE(renderWithLauncher(launcher, toPbrt(d), "emit", 16, 8, 1, rgb, w, h)) << toString(c.shape);
		EXPECT_NEAR(channelMean(rgb, 1), c.expected, 0.02) << toString(c.shape) << " seen from (" << c.camera.x << "," << c.camera.y << "," << c.camera.z << ")";
	}
}

// The scene list is built once. A file named after that cannot be added: the call says so (2) instead of blaming the file, and a file the list already
// holds simply gets its id.
TEST(SceneBuilderRegistryTest, ANamedFileThatArrivesAfterTheSceneListIsBuiltIsTooLate) {
	const auto& registry = get_scene_registry();  // builds it
	ASSERT_FALSE(pbrt_scene_registry::paths().empty()) << "pbrt_scenes/ was not discovered - run from the repository root";
	char id[32] = {};
	const auto& known = *pbrt_scene_registry::paths().begin();
	EXPECT_EQ(cpu_register_scene_file(known.second.c_str(), id, sizeof id), 0);
	EXPECT_EQ(std::string(id), known.first);
	(void)registry;

	const std::string fresh = "scene_builder_too_late.pbrt";
	{
		std::ofstream out(fresh, std::ios::binary);
		out << toPbrt(makeStarterScene());
	}
	EXPECT_EQ(cpu_register_scene_file(fresh.c_str(), id, sizeof id), 2);
	std::remove(fresh.c_str());
	EXPECT_EQ(cpu_register_scene_file(nullptr, id, sizeof id), 1);
}
