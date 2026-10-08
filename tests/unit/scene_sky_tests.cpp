// Tests for the Scene Builder's Sun & sky (src/shared/scene_sky.h, hosek_sky.h): the sky model's behaviour, the generated picture and the sun, the file it is
// written to, how it edits the document's lights, and (with the launcher built) that the lights add up to the light a closed form says they should.
#include <gtest/gtest.h>

#include "agreement_test_helpers.h"
#include "../../src/shared/hosek_sky.h"
#include "../../src/shared/pbrt_load.h"
#include "../../src/shared/scene_document.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace scene_doc;

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string tempDir(const char* name) {
	const std::filesystem::path dir = std::filesystem::path("scene_sky_test_dirs") / name;
	std::error_code ec;
	std::filesystem::remove_all(dir, ec);
	std::filesystem::create_directories(dir, ec);
	return dir.string();
}

pbrt_load::LoadResult loadText(const std::string& text, const char* tag) {
	const std::string path = std::string("scene_sky_test_") + tag + ".pbrt";
	{
		std::ofstream out(path, std::ios::binary);
		out << text;
	}
	pbrt_load::LoadResult r = pbrt_load::loadFile(path);
	std::remove(path.c_str());
	return r;
}

double luminance(const float* c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }

const float* pixelTowards(const std::vector<float>& img, int w, int h, double x, double y, double z) {
	const double len = std::sqrt(x * x + y * y + z * z);
	const double theta = std::acos(y / len), phi = std::fmod(std::atan2(-z, x) + 2.0 * kPi, 2.0 * kPi);
	const int row = std::min(h - 1, static_cast<int>(theta / kPi * h)), col = std::min(w - 1, static_cast<int>(phi / (2.0 * kPi) * w));
	return &img[(static_cast<size_t>(row) * w + col) * 3];
}

}  // namespace

TEST(HosekSkyTest, AClearSkyIsBrighterTowardsTheHorizonAndTheSunAndGrowsWithTheSunHeight) {
	double lastZenith = 0;
	for (double el : {10.0, 30.0, 50.0, 70.0}) {
		const hosek_sky::Model m(3.0, 0.3, el * kPi / 180.0);
		double zenith[3], horizon[3], nearSun[3];
		const double sunTheta = (90.0 - el) * kPi / 180.0;
		m.xyz(0.0, sunTheta, zenith);
		m.xyz(85.0 * kPi / 180.0, 90.0 * kPi / 180.0, horizon);
		m.xyz(sunTheta, 4.0 * kPi / 180.0, nearSun);
		EXPECT_GT(zenith[1], lastZenith) << "the zenith brightens as the sun climbs (" << el << ")";
		EXPECT_GT(horizon[1], zenith[1]) << "the horizon is brighter than the zenith";
		EXPECT_GT(nearSun[1], zenith[1]) << "the glow round the sun";
		// Real skies: zenith luminance a few thousand cd/m^2 (Y * 683), the near-sun glow tens of thousands.
		EXPECT_GT(zenith[1] * 683.0, 300.0);
		EXPECT_LT(zenith[1] * 683.0, 20000.0);
		lastZenith = zenith[1];
	}
}

TEST(HosekSkyTest, HazeBrightensTheGlowAndTheTurbidityRangeIsHandled) {
	const double el = 40.0 * kPi / 180.0, sunTheta = kPi / 2 - el;
	double clear[3], hazy[3];
	hosek_sky::Model(2.0, 0.3, el).xyz(sunTheta, 10.0 * kPi / 180.0, clear);
	hosek_sky::Model(8.0, 0.3, el).xyz(sunTheta, 10.0 * kPi / 180.0, hazy);
	EXPECT_NE(clear[1], hazy[1]);
	for (double t : {1.0, 1.7, 5.5, 10.0, 12.0, 0.2})   // out-of-range values are clamped, never read outside the table
		for (double albedo : {0.0, 0.5, 1.0, 2.0}) {
			double out[3];
			hosek_sky::Model(t, albedo, 0.3).xyz(0.7, 0.9, out);
			for (double v : out) ASSERT_TRUE(std::isfinite(v) && v >= 0.0) << "turbidity " << t << " albedo " << albedo;
		}
}

TEST(SceneSkyTest, TheSunIsRedderAndDimmerWhenLowAndItsDirectionFollowsTheAzimuth) {
	double lastE = 0;
	for (double el : {3.0, 10.0, 25.0, 50.0, 80.0}) {
		SkyParams p;
		p.sunElevation = el;
		const sky::SunLight s = sky::sunLight(p);
		EXPECT_GT(s.irradiance, lastE) << el;
		lastE = s.irradiance;
		EXPECT_DOUBLE_EQ(std::max(s.colour.r, std::max(s.colour.g, s.colour.b)), 1.0);
		EXPECT_GE(s.colour.r, s.colour.g);
		EXPECT_GE(s.colour.g, s.colour.b);
		EXPECT_LE(s.irradiance, sky::kSunIrradiance + 1e-9);
	}
	SkyParams low, high;
	low.sunElevation = 3.0;
	high.sunElevation = 70.0;
	EXPECT_LT(sky::sunLight(low).colour.b, 0.2) << "a sun on the horizon is deep orange";
	EXPECT_GT(sky::sunLight(high).colour.b, 0.6) << "a high sun is nearly white";

	SkyParams p;
	p.sunElevation = 30.0;
	for (double az : {0.0, 90.0, 180.0, 270.0}) {
		p.sunAzimuth = az;
		const Float3 d = sky::sunDirection(p);
		EXPECT_NEAR(d.x * d.x + d.y * d.y + d.z * d.z, 1.0, 1e-12);
		EXPECT_NEAR(d.y, 0.5, 1e-12);
		const double c = std::cos(30.0 * kPi / 180.0);
		EXPECT_NEAR(d.x, c * std::cos(az * kPi / 180.0), 1e-12);
		EXPECT_NEAR(d.z, -c * std::sin(az * kPi / 180.0), 1e-12);
	}
}

TEST(SceneSkyTest, TheSkyPictureIsBlueAboveBrightestNearTheSunAndAGroundBelow) {
	SkyParams p;
	p.sunElevation = 45.0;
	p.sunAzimuth = 0.0;   // towards +X
	const int w = 256, h = 128;
	const std::vector<float> img = sky::renderSkyImage(p, w, h);
	ASSERT_EQ(img.size(), static_cast<size_t>(w) * h * 3);
	for (float v : img) ASSERT_TRUE(std::isfinite(v) && v >= 0.0f);
	const float* up = pixelTowards(img, w, h, 0, 1, 0.0001);
	EXPECT_GT(up[2], up[0]) << "blue light overhead";
	const float* nearSun = pixelTowards(img, w, h, 1, 1, 0);          // the sun's own direction (45 degrees up, +X)
	const float* awayFromSun = pixelTowards(img, w, h, -1, 1, 0);     // the same height on the far side
	EXPECT_GT(luminance(nearSun), 2.0 * luminance(awayFromSun)) << "the glow round the sun";
	const float* ground = pixelTowards(img, w, h, 0, -1, 0.0001);
	EXPECT_GT(luminance(ground), 0.0);
	EXPECT_LT(luminance(ground), luminance(nearSun)) << "the ground is dimmer than the sun's glow";
	// (A sunlit ground is brighter than the zenith, as outdoors: its albedo times the sun and sky light on level ground.)
	// The ground is the albedo times what falls on level ground, so a darker albedo gives a darker ground and the sky above does not change.
	SkyParams dark = p;
	dark.groundAlbedo = 0.05;
	const std::vector<float> darker = sky::renderSkyImage(dark, w, h);
	EXPECT_LT(luminance(pixelTowards(darker, w, h, 0, -1, 0.0001)), 0.3 * luminance(ground));
}

TEST(SceneSkyTest, TheHdrFileRoundTripsThroughAnRgbeDecoder) {
	SkyParams p;
	const int w = 128, h = 64;
	const std::vector<float> img = sky::renderSkyImage(p, w, h);
	const std::string data = sky::encodeHdr(img, w, h);
	ASSERT_EQ(data.rfind("#?RADIANCE\n", 0), 0u);
	const size_t hdrEnd = data.find("-Y 64 +X 128\n");
	ASSERT_NE(hdrEnd, std::string::npos);
	size_t pos = hdrEnd + std::string("-Y 64 +X 128\n").size();
	// Decode the new-style run-length scanlines.
	double maxRel = 0;
	for (int y = 0; y < h; ++y) {
		ASSERT_EQ(static_cast<unsigned char>(data[pos]), 2u);
		ASSERT_EQ(static_cast<unsigned char>(data[pos + 1]), 2u);
		ASSERT_EQ((static_cast<unsigned char>(data[pos + 2]) << 8) | static_cast<unsigned char>(data[pos + 3]), w);
		pos += 4;
		std::vector<unsigned char> line(static_cast<size_t>(w) * 4);
		for (int c = 0; c < 4; ++c) {
			int x = 0;
			while (x < w) {
				const unsigned char count = static_cast<unsigned char>(data[pos++]);
				if (count > 128) {
					const unsigned char v = static_cast<unsigned char>(data[pos++]);
					for (int k = 0; k < count - 128; ++k) line[static_cast<size_t>(x++) * 4 + c] = v;
				} else {
					for (int k = 0; k < count; ++k) line[static_cast<size_t>(x++) * 4 + c] = static_cast<unsigned char>(data[pos++]);
				}
			}
			ASSERT_EQ(x, w);
		}
		for (int x = 0; x < w; ++x) {
			const unsigned char* q = &line[static_cast<size_t>(x) * 4];
			const float scale = q[3] ? std::ldexp(1.0f, static_cast<int>(q[3]) - 136) : 0.0f;
			for (int c = 0; c < 3; ++c) {
				const float want = img[(static_cast<size_t>(y) * w + x) * 3 + c];
				const float got = (q[c] + 0.5f) * scale;
				if (want > 1e-3f) maxRel = std::max(maxRel, std::fabs(static_cast<double>(got - want)) / want);
			}
		}
	}
	EXPECT_EQ(pos, data.size()) << "the file is exactly the scanlines";
	EXPECT_LT(maxRel, 0.03) << "RGBE keeps about 1% per pixel (a pixel's dimmer channels lose most)";
}

TEST(SceneSkyTest, ApplyingItWritesThePictureReusesItAndAimsASunLight) {
	const std::string dir = tempDir("apply");
	std::vector<Light> lights;
	Light skyLight;
	skyLight.name = "Sky";
	skyLight.kind = LightKind::Infinite;
	skyLight.sky.sunElevation = 20.0;
	skyLight.sky.sunAzimuth = 90.0;
	skyLight.intensity = 1.0;
	lights.push_back(skyLight);
	std::string error;
	ASSERT_TRUE(sky::applySunAndSky(lights, 0, dir, error)) << error;
	ASSERT_EQ(lights.size(), 2u) << "a Sun was added";
	EXPECT_TRUE(lights[0].physicalSky);
	EXPECT_TRUE(std::filesystem::exists(lights[0].imageFile));
	EXPECT_NE(lights[0].imageFile.find(sky::skyFileName(lights[0].sky)), std::string::npos);
	const Light& sun = lights[1];
	EXPECT_EQ(sun.kind, LightKind::Distant);
	EXPECT_EQ(sun.name, "Sun");
	const Float3 toSun = {sun.position.x - sun.target.x, sun.position.y - sun.target.y, sun.position.z - sun.target.z};
	const Float3 want = sky::sunDirection(lights[0].sky);
	const double len = std::sqrt(toSun.x * toSun.x + toSun.y * toSun.y + toSun.z * toSun.z);
	EXPECT_NEAR(toSun.x / len, want.x, 1e-9);
	EXPECT_NEAR(toSun.y / len, want.y, 1e-9);
	EXPECT_NEAR(toSun.z / len, want.z, 1e-9);
	EXPECT_NEAR(sun.intensity, sky::sunLight(lights[0].sky).irradiance, 1e-9);

	// Changing the sun updates the same Sun light (no second one), writes a new picture and keeps the old one; a repeat does not rewrite the file.
	const std::string first = lights[0].imageFile;
	lights[0].sky.sunElevation = 60.0;
	ASSERT_TRUE(sky::applySunAndSky(lights, 0, dir, error)) << error;
	EXPECT_EQ(lights.size(), 2u);
	EXPECT_NE(lights[0].imageFile, first);
	EXPECT_TRUE(std::filesystem::exists(first));
	const auto stamp = std::filesystem::last_write_time(lights[0].imageFile);
	ASSERT_TRUE(sky::applySunAndSky(lights, 0, dir, error)) << error;
	EXPECT_EQ(std::filesystem::last_write_time(lights[0].imageFile), stamp);

	// The sky light's brightness scales the Sun with it, and a user's own Distant light is the one that is taken over.
	std::vector<Light> mine;
	Light own;
	own.name = "My sun";
	own.kind = LightKind::Distant;
	mine.push_back(own);
	mine.push_back(skyLight);
	mine[1].intensity = 2.0;
	ASSERT_TRUE(sky::applySunAndSky(mine, 1, dir, error)) << error;
	EXPECT_EQ(mine.size(), 2u);
	EXPECT_NEAR(mine[0].intensity, 2.0 * sky::sunLight(mine[1].sky).irradiance, 1e-9);

	EXPECT_FALSE(sky::applySunAndSky(mine, 0, dir, error)) << "a Distant light is not a sky";
	EXPECT_FALSE(sky::applySunAndSky(mine, 7, dir, error));
	std::error_code ec;
	std::filesystem::remove_all("scene_sky_test_dirs", ec);
}

TEST(SceneSkyTest, SkySettingsSurviveTheJsonAndTheSceneStaysValidAndLoads) {
	const std::string dir = tempDir("json");
	Document d = makeStarterScene();
	d.lights[0].sky.sunElevation = 12.5;
	d.lights[0].sky.sunAzimuth = 200.0;
	d.lights[0].sky.turbidity = 4.5;
	d.lights[0].sky.groundAlbedo = 0.45;
	d.lights[0].intensity = 1.0;
	std::string error;
	ASSERT_TRUE(sky::applySunAndSky(d.lights, 0, dir, error)) << error;
	Document back;
	ASSERT_TRUE(fromJson(toJson(d), back, error)) << error;
	EXPECT_TRUE(back.lights[0].physicalSky);
	EXPECT_DOUBLE_EQ(back.lights[0].sky.sunElevation, 12.5);
	EXPECT_DOUBLE_EQ(back.lights[0].sky.sunAzimuth, 200.0);
	EXPECT_DOUBLE_EQ(back.lights[0].sky.turbidity, 4.5);
	EXPECT_DOUBLE_EQ(back.lights[0].sky.groundAlbedo, 0.45);
	EXPECT_EQ(back.lights[0].imageFile, d.lights[0].imageFile);
	EXPECT_EQ(toJson(back), toJson(d));
	// An older file (no sky fields) is an ordinary light.
	Document old;
	ASSERT_TRUE(fromJson(toJson(makeStarterScene()), old, error)) << error;
	EXPECT_FALSE(old.lights[0].physicalSky);
	EXPECT_EQ(old.lights[0].sky.sunElevation, SkyParams().sunElevation);
	EXPECT_FALSE(hasErrors(validate(d)));
	const pbrt_load::LoadResult r = loadText(toPbrt(d), "sun-and-sky");
	ASSERT_TRUE(r.ok) << r.error;
	std::error_code ec;
	std::filesystem::remove_all("scene_sky_test_dirs", ec);
}

namespace {

std::string findLauncherForSky() {
	namespace fs = std::filesystem;
	std::error_code ec;
	for (const char* c : {"ray_tracer.exe", "../ray_tracer.exe", "../../ray_tracer.exe", "x64/Release/ray_tracer.exe", "../x64/Release/ray_tracer.exe",
	                      "../../x64/Release/ray_tracer.exe", "ray_tracer", "../ray_tracer", "../../ray_tracer"})
		if (fs::is_regular_file(c, ec)) return fs::absolute(c, ec).string();
	return "";
}

}  // namespace

// A big level diffuse plate of albedo rho seen from above, lit only by the sky and the sun, at depth 1 radiates rho/pi times the irradiance on level ground:
// the sun's (irradiance * sin(height)) plus the integral of the sky picture over the upper hemisphere. This checks the picture's scale, the Sun light's units and
// the way the renderer reads the picture together.
TEST(SceneSkyRenderTest, ALevelPlateReadsAlbedoOverPiTimesTheIrradiance) {
	const std::string launcher = findLauncherForSky();
	if (launcher.empty()) GTEST_SKIP() << "ray_tracer.exe was not found next to the tests";
	for (double el : {25.0, 60.0}) {
		const std::string dir = tempDir("render");
		Document d;
		Object plate = makeObject(ShapeKind::Quad, "Plate");
		plate.size = {400, 1, 400};
		plate.material.color = {0.5, 0.5, 0.5};
		d.objects.push_back(plate);
		d.camera.position = {0.0, 6.0, 0.0};
		d.camera.target = {0.0, 0.0, 0.0};
		d.camera.up = {0.0, 0.0, -1.0};
		d.camera.fov = 8.0;
		Light skyLight;
		skyLight.name = "Sky";
		skyLight.kind = LightKind::Infinite;
		skyLight.intensity = 1.0;
		skyLight.sky.sunElevation = el;
		skyLight.sky.sunAzimuth = 40.0;
		d.lights.push_back(skyLight);
		std::string error;
		ASSERT_TRUE(sky::applySunAndSky(d.lights, 0, dir, error)) << error;
		ASSERT_FALSE(hasErrors(validate(d)));

		// The closed form: sun + the upper hemisphere of the picture (each channel).
		const int w = sky::kImageWidth, h = sky::kImageHeight;
		const std::vector<float> img = sky::renderSkyImage(d.lights[0].sky, w, h);
		double skyE[3] = {0, 0, 0};
		const double dOmega = (2.0 * kPi / w) * (kPi / h);
		for (int row = 0; row < h / 2; ++row) {
			const double theta = (row + 0.5) / h * kPi;
			for (int col = 0; col < w; ++col)
				for (int c = 0; c < 3; ++c) skyE[c] += img[(static_cast<size_t>(row) * w + col) * 3 + c] * std::cos(theta) * std::sin(theta) * dOmega;
		}
		const sky::SunLight sun = sky::sunLight(d.lights[0].sky);
		const double sunE[3] = {sun.irradiance * sun.colour.r, sun.irradiance * sun.colour.g, sun.irradiance * sun.colour.b};
		const double sinEl = std::sin(el * kPi / 180.0);

		const std::string scenePath = "scene_sky_render.pbrt", outPath = "scene_sky_render.exr";
		{
			std::ofstream out(scenePath, std::ios::binary);
			out << toPbrt(d);
		}
		std::remove(outPath.c_str());
		std::ostringstream cmd;
		cmd << "\"" << launcher << "\" --cpu --output \"" << outPath << "\" 24 256 1 \"" << scenePath << "\" > scene_sky_render.log 2>&1";
		std::string command = cmd.str();
#ifdef _WIN32
		command = "\"" + command + "\"";
#endif
		ASSERT_EQ(std::system(command.c_str()), 0);
		std::vector<float> rgb;
		int rw = 0, rh = 0;
		ASSERT_TRUE(loadLinearRgbPixels(outPath, rw, rh, rgb));
		for (int c = 0; c < 3; ++c) {
			double sum = 0;
			for (size_t i = 0; i < rgb.size() / 3; ++i) sum += rgb[3 * i + c];
			const double got = sum / static_cast<double>(rgb.size() / 3);
			const double want = 0.5 / kPi * (skyE[c] + sunE[c] * sinEl);
			EXPECT_NEAR(got, want, 0.05 * want) << "sun height " << el << ", channel " << c << ": albedo/pi * (sky " << skyE[c] << " + sun " << sunE[c] * sinEl << ")";
		}
		std::remove(scenePath.c_str());
		std::remove(outPath.c_str());
		std::remove((outPath + ".run_marker.txt").c_str());
		std::remove("scene_sky_render.log");
		std::error_code ec;
		std::filesystem::remove_all("scene_sky_test_dirs", ec);
	}
}
