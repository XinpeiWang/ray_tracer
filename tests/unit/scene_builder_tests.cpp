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
#include "../../src/shared/scene_props.h"
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
	Object pic = makeObject(ShapeKind::Quad, "Picture");
	pic.material.imageFile = "C:\\pictures\\my photo.png";
	d.objects.push_back(pic);
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
	EXPECT_EQ(b.objects[8].material.imageFile, "C:\\pictures\\my photo.png");
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
	EXPECT_TRUE(mentions(validate(d), Problem::Severity::Error, "choose a mesh file"));
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
	for (ShapeKind sh : scene_doc::allShapeKinds())
		if (sh != ShapeKind::Mesh) d.objects.push_back(makeObject(sh, toString(sh)));
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

// A picture becomes an imagemap texture bound as the reflectance, replacing the flat colour and the checker.
TEST(SceneDocumentPbrtTest, APictureIsWrittenAsAnImageTextureForDiffuseAndCoatedPaint) {
	Document d = makeStarterScene();
	d.objects.clear();
	Object a = makeObject(ShapeKind::Quad, "Wall");
	a.material.imageFile = "C:\\pictures\\my photo.png";
	a.material.checker = true;
	d.objects.push_back(a);
	Object b = makeObject(ShapeKind::Sphere, "Ball");
	b.material.kind = MaterialKind::CoatedDiffuse;
	b.material.imageFile = "ball.jpg";
	d.objects.push_back(b);
	Object c = makeObject(ShapeKind::Sphere, "Metal");
	c.material.kind = MaterialKind::Conductor;
	c.material.imageFile = "ignored.png";
	d.objects.push_back(c);
	const std::string text = toPbrt(d);
	EXPECT_NE(text.find("Texture \"picture-1\" \"spectrum\" \"imagemap\" \"string filename\" [ \"C:/pictures/my photo.png\" ]"), std::string::npos) << text;
	EXPECT_NE(text.find("Material \"diffuse\" \"texture reflectance\" \"picture-1\""), std::string::npos);
	EXPECT_NE(text.find("Material \"coateddiffuse\" \"texture reflectance\" \"picture-2\" \"float eta\""), std::string::npos);
	EXPECT_EQ(text.find("checkerboard"), std::string::npos) << "the picture replaces the checker";
	EXPECT_EQ(text.find("picture-3"), std::string::npos) << "a conductor has no picture";
	// The same text, read back, still has the builder data with the picture in it.
	Document back;
	std::string err;
	ASSERT_TRUE(fromPbrt(text, back, err)) << err;
	EXPECT_EQ(back.objects[1].material.imageFile, "ball.jpg");
}

TEST(SceneDocumentValidateTest, APictureOnAMaterialThatCannotShowItIsAWarning) {
	Document d = makeStarterScene();
	Object o = makeObject(ShapeKind::Sphere, "Metal");
	o.material.kind = MaterialKind::Conductor;
	o.material.imageFile = "x.png";
	d.objects.push_back(o);
	bool warned = false;
	for (const Problem& p : validate(d))
		if (p.message.find("a picture only applies") != std::string::npos) warned = p.severity == Problem::Severity::Warning;
	EXPECT_TRUE(warned);
	EXPECT_FALSE(hasErrors(validate(d)));
}

// The renderer renders a missing picture without it and says nothing the Builder can show, so the problem list has to: a picture, mesh or sky file
// that is an absolute path to nothing is a warning (not an error: a preview still works), and one that exists is not mentioned.
TEST(SceneDocumentValidateTest, AMissingPictureMeshOrSkyFileIsAWarningNamingIt) {
	const std::string gone = std::filesystem::absolute("scene_builder_no_such_file.png").string();
	ASSERT_FALSE(std::filesystem::exists(gone));
	Document d = makeStarterScene();
	Object pic = makeObject(ShapeKind::Quad, "Photo wall");
	pic.material.imageFile = gone;
	d.objects.push_back(pic);
	Object mesh = makeObject(ShapeKind::Mesh, "Scan");
	mesh.meshFile = std::filesystem::absolute("scene_builder_no_such_mesh.obj").string();
	d.objects.push_back(mesh);
	Light sky;
	sky.name = "Sky";
	sky.kind = LightKind::Infinite;
	sky.imageFile = std::filesystem::absolute("scene_builder_no_such_sky.exr").string();
	d.lights.push_back(sky);
	const std::vector<Problem> ps = validate(d);
	EXPECT_FALSE(hasErrors(ps));
	EXPECT_TRUE(mentions(ps, Problem::Severity::Warning, "picture file was not found"));
	EXPECT_TRUE(mentions(ps, Problem::Severity::Warning, "mesh file was not found"));
	EXPECT_TRUE(mentions(ps, Problem::Severity::Warning, "sky image was not found"));
	EXPECT_TRUE(mentions(ps, Problem::Severity::Warning, gone)) << "the message says which file";
}

TEST(SceneDocumentValidateTest, APictureThatExistsOrIsRelativeIsNotMentioned) {
	const std::string here = "scene_builder_exists_picture.bmp";
	{ std::ofstream out(here, std::ios::binary); out << "x"; }
	Document d = makeStarterScene();
	Object a = makeObject(ShapeKind::Quad, "Real");
	a.material.imageFile = std::filesystem::absolute(here).string();
	d.objects.push_back(a);
	Object b = makeObject(ShapeKind::Quad, "Relative");
	b.material.imageFile = "somewhere/else.png";  // relative to the saved scene: unknowable here
	d.objects.push_back(b);
	const std::vector<Problem> ps = validate(d);
	std::remove(here.c_str());
	EXPECT_FALSE(mentions(ps, Problem::Severity::Warning, "was not found"));
}

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
	std::remove((outPath + ".run_marker.txt").c_str());  // the launcher drops a marker beside its output
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

namespace {

// A 24-bit BMP of w x h pixels; rgb is row-major from the TOP row, 3 bytes per pixel.
std::string bmpFile(int w, int h, const std::vector<unsigned char>& rgb) {
	const int rowBytes = (w * 3 + 3) / 4 * 4;
	const int dataBytes = rowBytes * h;
	std::string out;
	auto put32 = [&out](unsigned v) { for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF)); };
	auto put16 = [&out](unsigned v) { for (int i = 0; i < 2; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF)); };
	out += "BM";
	put32(54 + dataBytes); put32(0); put32(54);
	put32(40); put32(static_cast<unsigned>(w)); put32(static_cast<unsigned>(h)); put16(1); put16(24); put32(0); put32(dataBytes); put32(2835); put32(2835); put32(0); put32(0);
	for (int y = h - 1; y >= 0; --y) {  // BMP rows run bottom to top
		for (int x = 0; x < w; ++x)
			for (int c = 2; c >= 0; --c) out.push_back(static_cast<char>(rgb[(y * w + x) * 3 + c]));  // stored as B, G, R
		for (int p = w * 3; p < rowBytes; ++p) out.push_back('\0');
	}
	return out;
}

double srgbToLinear(double v) { return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); }

// Mean of one channel over the middle of a quarter of the picture: qx, qy in {0, 1} = left/right, top/bottom.
double quarterMean(const std::vector<float>& rgb, int w, int h, int qx, int qy, int c) {
	double s = 0;
	int n = 0;
	for (int y = qy * h / 2 + h / 8; y < (qy + 1) * h / 2 - h / 8; ++y)
		for (int x = qx * w / 2 + w / 8; x < (qx + 1) * w / 2 - w / 8; ++x, ++n) s += rgb[3 * (y * w + x) + c];
	return s / n;
}

}  // namespace

// A sphere whose picture is one colour reads that colour (decoded from sRGB to linear) under a uniform sky at depth 1.
TEST(SceneBuilderRenderTest, APicturedSphereReadsTheColourOfItsPicture) {
	const std::string launcher = findLauncher();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	const std::string bmp = "scene_builder_solid_picture.bmp";
	{
		std::ofstream out(bmp, std::ios::binary);
		out << bmpFile(2, 2, {230, 200, 100, 230, 200, 100, 230, 200, 100, 230, 200, 100});
	}
	Document d;
	d.camera.position = {0, 0, 10};
	d.camera.target = {0, 0, 0};
	d.camera.fov = 5;
	Object ball = makeObject(ShapeKind::Sphere, "Ball");
	ball.position = {0, 0, 0};
	ball.material.imageFile = std::filesystem::absolute(bmp).string();
	d.objects.push_back(ball);
	Light sky;
	sky.kind = LightKind::Infinite;
	sky.intensity = 1.0;
	d.lights.push_back(sky);
	ASSERT_FALSE(hasErrors(validate(d)));
	std::vector<float> rgb;
	int w = 0, h = 0;
	const bool ok = renderWithLauncher(launcher, toPbrt(d), "picture_sphere", 32, 128, 1, rgb, w, h);
	std::remove(bmp.c_str());
	ASSERT_TRUE(ok);
	const double want[3] = {srgbToLinear(230 / 255.0), srgbToLinear(200 / 255.0), srgbToLinear(100 / 255.0)};
	for (int c = 0; c < 3; ++c) EXPECT_NEAR(channelMean(rgb, c), want[c], 0.03 * want[c] + 0.004) << "channel " << c;
}

// A quad stood up as a wall (rotated 90 degrees about X so it faces +Z) and seen from +Z shows its picture the right way up and not mirrored:
// the picture's top-left pixel is at the top left of the render. The picture is red/green over blue/white.
TEST(SceneBuilderRenderTest, APicturedQuadShowsItsPictureUprightAndNotMirrored) {
	const std::string launcher = findLauncher();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	const std::string bmp = "scene_builder_quad_picture.bmp";
	{
		std::ofstream out(bmp, std::ios::binary);
		out << bmpFile(2, 2, {230, 20, 20, 20, 230, 20, 20, 20, 230, 230, 230, 230});
	}
	Document d;
	d.camera.position = {0, 0, 5};
	d.camera.target = {0, 0, 0};
	d.camera.fov = 30;
	Object wall = makeObject(ShapeKind::Quad, "Wall");
	wall.size = {4, 1, 4};
	wall.rotation = {90, 0, 0};
	wall.material.imageFile = std::filesystem::absolute(bmp).string();
	d.objects.push_back(wall);
	Light sky;
	sky.kind = LightKind::Infinite;
	sky.intensity = 1.0;
	d.lights.push_back(sky);
	ASSERT_FALSE(hasErrors(validate(d)));
	std::vector<float> rgb;
	int w = 0, h = 0;
	const bool ok = renderWithLauncher(launcher, toPbrt(d), "picture_quad", 32, 64, 1, rgb, w, h);
	std::remove(bmp.c_str());
	ASSERT_TRUE(ok);
	auto dominant = [&](int qx, int qy) {  // 0 red, 1 green, 2 blue, 3 white
		const double r = quarterMean(rgb, w, h, qx, qy, 0), g = quarterMean(rgb, w, h, qx, qy, 1), b = quarterMean(rgb, w, h, qx, qy, 2);
		if (r > 0.5 && g > 0.5 && b > 0.5) return 3;
		return r > g && r > b ? 0 : (g > b ? 1 : 2);
	};
	EXPECT_EQ(dominant(0, 0), 0) << "top left should be the red pixel";
	EXPECT_EQ(dominant(1, 0), 1) << "top right should be the green pixel";
	EXPECT_EQ(dominant(0, 1), 2) << "bottom left should be the blue pixel";
	EXPECT_EQ(dominant(1, 1), 3) << "bottom right should be the white pixel";
}

// The ready-made shapes that are generated triangle meshes (scene_shapes.h): every triangle faces outward, the normals agree with the winding, the volume
// is positive and the mesh sits where the document says (centred on the position, sized by its fields).
TEST(SceneDocumentShapesTest, GeneratedMeshesAreClosedOutwardFacingAndSizedAsAsked) {
	for (ShapeKind sh : scene_doc::allShapeKinds()) {
		if (!scene_doc::isGeneratedShape(sh)) continue;
		const Object o = makeObject(sh, toString(sh));
		const scene_doc::ShapeMesh m = scene_doc::generatedMesh(o);
		ASSERT_GT(m.triangleCount(), 0u) << toString(sh);
		ASSERT_EQ(m.N.size(), m.P.size()) << toString(sh);
		ASSERT_EQ(m.UV.size() / 2, m.vertexCount()) << toString(sh);
		double volume = 0, lo[3] = {1e9, 1e9, 1e9}, hi[3] = {-1e9, -1e9, -1e9};
		for (std::size_t k = 0; k < m.vertexCount(); ++k)
			for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], m.P[k * 3 + a]); hi[a] = std::max(hi[a], m.P[k * 3 + a]); }
		for (std::size_t t = 0; t < m.triangleCount(); ++t) {
			const int i[3] = {m.indices[t * 3], m.indices[t * 3 + 1], m.indices[t * 3 + 2]};
			for (int v : i) ASSERT_TRUE(v >= 0 && v < static_cast<int>(m.vertexCount())) << toString(sh);
			double p[3][3], n[3] = {0, 0, 0};
			for (int v = 0; v < 3; ++v)
				for (int a = 0; a < 3; ++a) { p[v][a] = m.P[i[v] * 3 + a]; n[a] += m.N[i[v] * 3 + a]; }
			const double e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]}, e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
			const double g[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
			EXPECT_GT(g[0] * n[0] + g[1] * n[1] + g[2] * n[2], 0.0) << toString(sh) << " triangle " << t << " faces against its normals";
			volume += (p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0]) +
			           p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0])) / 6.0;
		}
		EXPECT_GT(volume, 0.01) << toString(sh) << " is not a closed, outward-facing solid";
		for (int a = 0; a < 3; ++a) EXPECT_NEAR(lo[a], -hi[a], 1e-6 + 0.2 * (hi[a] - lo[a])) << toString(sh) << " is not centred on its position (axis " << a << ")";
	}
	// Sizes follow the fields.
	auto extent = [](const Object& o, int axis) {
		const scene_doc::ShapeMesh m = scene_doc::generatedMesh(o);
		double lo = 1e9, hi = -1e9;
		for (std::size_t k = 0; k < m.vertexCount(); ++k) { lo = std::min(lo, m.P[k * 3 + axis]); hi = std::max(hi, m.P[k * 3 + axis]); }
		return hi - lo;
	};
	Object pyr = makeObject(ShapeKind::Pyramid, "p");
	pyr.size = {3, 0, 5};
	pyr.height = 2;
	EXPECT_NEAR(extent(pyr, 0), 3, 1e-9);
	EXPECT_NEAR(extent(pyr, 1), 2, 1e-9);
	EXPECT_NEAR(extent(pyr, 2), 5, 1e-9);
	Object tor = makeObject(ShapeKind::Torus, "t");
	tor.radius = 2;
	tor.radius2 = 0.5;
	EXPECT_NEAR(extent(tor, 0), 5, 1e-6);
	EXPECT_NEAR(extent(tor, 1), 1, 1e-6);
	Object cap = makeObject(ShapeKind::Capsule, "c");
	cap.radius = 0.5;
	cap.height = 3;
	EXPECT_NEAR(extent(cap, 1), 3, 1e-9);
	EXPECT_NEAR(extent(cap, 0), 1, 1e-6);
	cap.height = 0.2;   // shorter than its ends need: as tall as they are
	EXPECT_NEAR(extent(cap, 1), 1, 1e-9);
	Object stairs = makeObject(ShapeKind::Stairs, "s");
	stairs.steps = 1;
	const std::size_t oneStep = scene_doc::generatedMesh(stairs).triangleCount();
	stairs.steps = 8;
	EXPECT_GT(scene_doc::generatedMesh(stairs).triangleCount(), oneStep * 4);
}

TEST(SceneDocumentShapesTest, NewShapesRoundTripValidateAndWriteAsPlainPbrt) {
	Document d = makeStarterScene();
	const std::size_t first = d.objects.size();   // the objects added below follow the starter scene's own
	for (ShapeKind sh : scene_doc::allShapeKinds()) {
		if (sh == ShapeKind::Mesh) continue;
		Object o = makeObject(sh, std::string("n-") + toString(sh));
		o.radius2 = 0.31;
		o.steps = 7;
		d.objects.push_back(o);
	}
	ASSERT_FALSE(hasErrors(validate(d)));
	const std::string text = toPbrt(d);
	EXPECT_NE(text.find("\"normal N\""), std::string::npos);
	Document back;
	std::string error;
	ASSERT_TRUE(fromPbrt(text, back, error)) << error;
	ASSERT_EQ(back.objects.size(), d.objects.size());
	for (std::size_t i = first; i < d.objects.size(); ++i) {
		EXPECT_EQ(back.objects[i].shape, d.objects[i].shape) << i;
		EXPECT_DOUBLE_EQ(back.objects[i].radius2, 0.31) << i;
		EXPECT_EQ(back.objects[i].steps, 7) << i;
	}
	// A document written before these fields existed keeps their defaults.
	Document old;
	ASSERT_TRUE(fromJson("{\"objects\":[{\"name\":\"a\",\"shape\":\"torus\"}]}", old, error)) << error;
	EXPECT_DOUBLE_EQ(old.objects[0].radius2, 0.25);
	EXPECT_EQ(old.objects[0].steps, 5);

	// What does not make a solid is refused.
	auto problem = [](ShapeKind sh, auto mutate) {
		Document one = makeStarterScene();
		Object o = makeObject(sh, "x");
		mutate(o);
		one.objects.push_back(o);
		return hasErrors(validate(one));
	};
	EXPECT_TRUE(problem(ShapeKind::Torus, [](Object& o) { o.radius2 = o.radius; }));
	EXPECT_TRUE(problem(ShapeKind::Tube, [](Object& o) { o.radius2 = o.radius + 0.1; }));
	EXPECT_TRUE(problem(ShapeKind::Stairs, [](Object& o) { o.steps = 0; }));
	EXPECT_TRUE(problem(ShapeKind::Pyramid, [](Object& o) { o.height = 0; }));
	EXPECT_TRUE(problem(ShapeKind::Dome, [](Object& o) { o.radius = 0; }));
	EXPECT_FALSE(problem(ShapeKind::Capsule, [](Object& o) { o.height = 0.1; }));   // only a warning
	EXPECT_TRUE(problem(ShapeKind::Stairs, [](Object& o) { o.steps = scene_doc::kMaxStairSteps + 1; }));
}

// A corrupt file asking for two billion steps is reported by validate(), and drawing it makes at most scene_doc::kMaxStairSteps steps - never a huge mesh.
TEST(SceneBuilderTest, AHugeStepCountMakesABoundedMesh) {
	Object o = makeObject(ShapeKind::Stairs, "s");
	o.steps = 2000000000;
	const scene_doc::ShapeMesh capped = scene_doc::generatedMesh(o);
	o.steps = scene_doc::kMaxStairSteps;
	EXPECT_EQ(capped.P.size(), scene_doc::generatedMesh(o).P.size());
	o.steps = -5;
	EXPECT_FALSE(scene_doc::generatedMesh(o).P.empty());   // at least one step
}

// The props (scene_props.h): ordinary objects that stand on the floor around the origin, validate cleanly, and survive a save and re-open as plain objects.
TEST(SceneDocumentPropsTest, EveryPropIsAValidSetOfObjectsStandingOnTheFloor) {
	for (scene_doc::PropKind kind : scene_doc::allPropKinds()) {
		const std::vector<Object> parts = scene_doc::makeProp(kind);
		ASSERT_GE(parts.size(), 3u) << toString(kind);
		Document d = makeStarterScene();
		const std::size_t first = d.objects.size();
		double lowest = 1e9;
		for (const Object& o : parts) {
			EXPECT_FALSE(o.name.empty()) << toString(kind);
			d.objects.push_back(o);
			// The lowest point of each part: its centre minus half its height (a sphere or box by its size, a cone or cylinder by its height).
			double half = o.radius;
			if (o.shape == ShapeKind::Box) half = o.size.y / 2;
			if (o.shape == ShapeKind::Cylinder || o.shape == ShapeKind::Cone) half = o.height / 2;
			if (o.rotation.x == 0) lowest = std::min(lowest, o.position.y - half);
		}
		EXPECT_NEAR(lowest, 0.0, 1e-9) << toString(kind) << " does not stand on the floor";
		EXPECT_FALSE(hasErrors(validate(d))) << toString(kind);
		Document back;
		std::string error;
		ASSERT_TRUE(fromPbrt(toPbrt(d), back, error)) << error;
		ASSERT_EQ(back.objects.size(), d.objects.size());
		for (std::size_t i = first; i < d.objects.size(); ++i) {
			EXPECT_EQ(back.objects[i].name, d.objects[i].name);
			EXPECT_EQ(back.objects[i].shape, d.objects[i].shape);
		}
	}
	// The street lamp's bulb is the one part that gives light.
	int emitters = 0;
	for (const Object& o : scene_doc::makeProp(scene_doc::PropKind::StreetLamp)) emitters += o.emissive ? 1 : 0;
	EXPECT_EQ(emitters, 1);
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
		{ShapeKind::Pyramid, {0, 0, 10}, L},   {ShapeKind::Wedge, {0, 0, 10}, L},   {ShapeKind::Stairs, {0, 0, 10}, L},  {ShapeKind::Torus, {0, 0, 10}, L},
		{ShapeKind::Capsule, {0, 0, 10}, L},   {ShapeKind::Dome, {0, 0, 10}, L},    {ShapeKind::Tube, {0, 0, 10}, L},
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

// A scene the user deletes (the GUI's "Delete scene" / "Delete all my scenes") leaves the list on the next refresh, which is also what lists a new one: the
// list follows the per-user folder in both directions, and a deleted scene's id no longer finds anything.
TEST(SceneBuilderRegistryTest, ADeletedUserSceneLeavesTheListOnTheNextRefresh) {
	namespace fs = std::filesystem;
	const fs::path root = fs::temp_directory_path() / ("rt_user_scenes_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
	fs::remove_all(root);
	fs::create_directories(root / "user_scenes");
	const char* old = std::getenv("RAY_TRACER_USER_ASSETS");
	const std::string oldValue = old ? old : "";
#ifdef _WIN32
	_putenv_s("RAY_TRACER_USER_ASSETS", root.string().c_str());
#else
	setenv("RAY_TRACER_USER_ASSETS", root.string().c_str(), 1);
#endif
	const size_t before = get_scene_registry().size();
	for (const char* name : {"gone-scene.pbrt", "kept-scene.pbrt"}) {
		std::ofstream out(root / "user_scenes" / name, std::ios::binary);
		out << toPbrt(makeStarterScene());
	}
	EXPECT_EQ(cpu_refresh_user_scenes(), 2) << "both files are listed";
	EXPECT_EQ(get_scene_registry().size(), before + 2);
	std::string goneId, keptId;
	for (const auto& kv : pbrt_scene_registry::paths()) {
		const std::string file = fs::path(kv.second).filename().string();
		if (file == "gone-scene.pbrt") goneId = kv.first;
		if (file == "kept-scene.pbrt") keptId = kv.first;
	}
	ASSERT_FALSE(goneId.empty());
	ASSERT_FALSE(keptId.empty());
	ASSERT_NE(find_scene(goneId), nullptr);

	fs::remove(root / "user_scenes" / "gone-scene.pbrt");
	cpu_refresh_user_scenes();
	EXPECT_EQ(get_scene_registry().size(), before + 1) << "the deleted scene left the list";
	EXPECT_EQ(find_scene(goneId), nullptr);
	EXPECT_EQ(pbrt_scene_registry::paths().count(goneId), 0u);
	ASSERT_NE(find_scene(keptId), nullptr) << "the other scene is untouched";

	fs::remove(root / "user_scenes" / "kept-scene.pbrt");
	cpu_refresh_user_scenes();
	EXPECT_EQ(get_scene_registry().size(), before);
	EXPECT_EQ(cpu_refresh_user_scenes(), 0) << "a refresh with nothing new or gone changes nothing";

#ifdef _WIN32
	_putenv_s("RAY_TRACER_USER_ASSETS", oldValue.c_str());
#else
	if (old) setenv("RAY_TRACER_USER_ASSETS", oldValue.c_str(), 1);
	else unsetenv("RAY_TRACER_USER_ASSETS");
#endif
	fs::remove_all(root);
}
