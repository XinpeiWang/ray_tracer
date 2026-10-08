#pragma once
// scene_sky.h -- the Scene Builder's "Sun & sky": a physically based clear sky for a chosen time of day, as the two lights the renderers already understand.
//
// The sky is the Hosek-Wilkie model (hosek_sky.h) sampled into an equirectangular HDR image (row 0 is straight up, the layout every backend's "infinite" image
// light reads) and the sun is a distant light whose colour and strength come from how much atmosphere its light crosses (Preetham's Rayleigh + aerosol
// extinction). Both are in the same units, so a low sun is dimmer, redder and sits in a darker sky, as it does outside. The picture is written once per set of
// parameters and reused (the file name says which they are).
//
// Units: the scene is scaled so that the full, unobstructed sun gives an irradiance of kSunIrradiance (3.0) on a surface facing it, the value the starter
// scene's Sun uses; the sky's luminance is scaled by the same factor (cd/m^2 and lux share the real-world ratio). The Light's `intensity` multiplies both.
//
// std-only.

#include "hosek_sky.h"
#include "scene_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace scene_doc {
namespace sky {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSunIrradiance = 3.0;              // scene units, the unobstructed sun facing a surface
constexpr double kSunIlluminanceLux = 100000.0;     // what that is outdoors, so cd/m^2 can be converted at the same ratio
constexpr double kScenePerLux = kSunIrradiance / kSunIlluminanceLux;
constexpr int kImageWidth = 512, kImageHeight = 256;

inline SkyParams clamped(SkyParams p) {
	p.sunElevation = std::min(90.0, std::max(0.5, p.sunElevation));
	p.sunAzimuth = std::fmod(std::fmod(p.sunAzimuth, 360.0) + 360.0, 360.0);
	p.turbidity = std::min(10.0, std::max(1.7, p.turbidity));
	p.groundAlbedo = std::min(1.0, std::max(0.0, p.groundAlbedo));
	return p;
}

// The unit vector towards the sun. Azimuth 0 is +X, 90 is -Z (away from the starter camera), 180 is -X, 270 is +Z (behind it).
inline Float3 sunDirection(const SkyParams& in) {
	const SkyParams p = clamped(in);
	const double el = p.sunElevation * kPi / 180.0, az = p.sunAzimuth * kPi / 180.0;
	return {std::cos(el) * std::cos(az), std::sin(el), -std::cos(el) * std::sin(az)};
}

struct SunLight {
	Float3 towardSun;
	Rgb colour;          // the brightest channel is 1
	double irradiance;   // the brightest channel, scene units, facing the sun
};

inline SunLight sunLight(const SkyParams& in) {
	const SkyParams p = clamped(in);
	const double elDeg = p.sunElevation, el = elDeg * kPi / 180.0;
	// Kasten-Young relative air mass, then the transmittance of Rayleigh scattering and aerosols per colour channel (Preetham et al., 1999).
	const double airMass = 1.0 / (std::sin(el) + 0.50572 * std::pow(elDeg + 6.07995, -1.6364));
	const double beta = std::max(0.005, 0.04608 * p.turbidity - 0.04586);
	const double wavelengthMicrons[3] = {0.610, 0.550, 0.465};
	double t[3];
	for (int c = 0; c < 3; ++c) {
		const double l = wavelengthMicrons[c];
		t[c] = std::exp(-airMass * (0.008735 * std::pow(l, -4.08) + beta * std::pow(l, -1.3)));
	}
	const double peak = std::max(t[0], std::max(t[1], t[2]));
	SunLight s;
	s.towardSun = sunDirection(p);
	s.colour = {t[0] / peak, t[1] / peak, t[2] / peak};
	s.irradiance = kSunIrradiance * peak;
	return s;
}

// The sky as an equirectangular image of width x height RGB floats (linear, scene units), row 0 = up. Below the horizon it is the ground: its albedo times what
// falls on level ground, so a scene's own floor and the sky agree about the light.
inline std::vector<float> renderSkyImage(const SkyParams& in, int width, int height) {
	const SkyParams p = clamped(in);
	const hosek_sky::Model model(p.turbidity, p.groundAlbedo, p.sunElevation * kPi / 180.0);
	const Float3 sun = sunDirection(p);
	std::vector<float> rgb(static_cast<size_t>(width) * height * 3, 0.0f);
	// CIE XYZ -> linear sRGB, and the luminance scale: Y * 683 is cd/m^2, kScenePerLux turns that into scene units.
	auto toRgb = [](const double xyz[3], double out[3]) {
		const double k = 683.0 * kScenePerLux;
		out[0] = std::max(0.0, k * (3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2]));
		out[1] = std::max(0.0, k * (-0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2]));
		out[2] = std::max(0.0, k * (0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2]));
	};
	double skyIrradiance = 0.0;   // on level ground, from the part of the sky above the horizon (luminance-weighted)
	const double dOmegaRow = (2.0 * kPi / width) * (kPi / height);
	for (int row = 0; row < height; ++row) {
		const double theta = (row + 0.5) / height * kPi;
		const double st = std::sin(theta), ct = std::cos(theta);
		for (int col = 0; col < width; ++col) {
			const double phi = (col + 0.5) / width * 2.0 * kPi;
			const double dx = st * std::cos(phi), dy = ct, dz = -st * std::sin(phi);
			if (dy <= 0.0) continue;   // the ground, below
			const double cosGamma = std::min(1.0, std::max(-1.0, dx * sun.x + dy * sun.y + dz * sun.z));
			double xyz[3], c[3];
			model.xyz(theta, std::acos(cosGamma), xyz);
			toRgb(xyz, c);
			float* px = &rgb[(static_cast<size_t>(row) * width + col) * 3];
			for (int k = 0; k < 3; ++k) px[k] = static_cast<float>(c[k]);
			skyIrradiance += (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) * dy * st * dOmegaRow;
		}
	}
	const SunLight sunL = sunLight(p);
	const double levelIrradiance = skyIrradiance + sunL.irradiance * sun.y;
	const double ground = p.groundAlbedo * levelIrradiance / kPi;
	for (int row = height / 2; row < height; ++row)
		for (int col = 0; col < width; ++col) {
			float* px = &rgb[(static_cast<size_t>(row) * width + col) * 3];
			px[0] = static_cast<float>(ground * (0.9 + 0.1 * sunL.colour.r));   // a hint of the sun's colour in the light the ground returns
			px[1] = static_cast<float>(ground * (0.9 + 0.1 * sunL.colour.g));
			px[2] = static_cast<float>(ground * (0.9 + 0.1 * sunL.colour.b));
		}
	return rgb;
}

// ---- Radiance .hdr (RGBE, run-length encoded): readable by stb_image and every other loader ----

namespace detail {
inline void rgbe(const float* c, unsigned char out[4]) {
	const float v = std::max(c[0], std::max(c[1], c[2]));
	if (!(v > 1e-32f)) { out[0] = out[1] = out[2] = out[3] = 0; return; }
	int e;
	const float m = std::frexp(v, &e) * 256.0f / v;
	out[0] = static_cast<unsigned char>(c[0] * m);
	out[1] = static_cast<unsigned char>(c[1] * m);
	out[2] = static_cast<unsigned char>(c[2] * m);
	out[3] = static_cast<unsigned char>(e + 128);
}
inline void writeChannelRuns(std::string& out, const unsigned char* d, int n) {
	int x = 0;
	while (x < n) {
		int run = 1;
		while (x + run < n && run < 127 && d[x + run] == d[x]) ++run;
		if (run >= 3) {
			out.push_back(static_cast<char>(128 + run));
			out.push_back(static_cast<char>(d[x]));
			x += run;
			continue;
		}
		int lit = 0;   // literal bytes up to the next run of 3 or more
		while (x + lit < n && lit < 128) {
			int r = 1;
			while (x + lit + r < n && r < 3 && d[x + lit + r] == d[x + lit]) ++r;
			if (r >= 3) break;
			++lit;
		}
		if (lit == 0) lit = 1;
		out.push_back(static_cast<char>(lit));
		for (int i = 0; i < lit; ++i) out.push_back(static_cast<char>(d[x + i]));
		x += lit;
	}
}
}  // namespace detail

inline std::string encodeHdr(const std::vector<float>& rgb, int width, int height) {
	std::string out = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " + std::to_string(height) + " +X " + std::to_string(width) + "\n";
	std::vector<unsigned char> line(static_cast<size_t>(width) * 4), plane(static_cast<size_t>(width));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) detail::rgbe(&rgb[(static_cast<size_t>(y) * width + x) * 3], &line[static_cast<size_t>(x) * 4]);
		out.push_back(2);
		out.push_back(2);
		out.push_back(static_cast<char>((width >> 8) & 255));
		out.push_back(static_cast<char>(width & 255));
		for (int c = 0; c < 4; ++c) {
			for (int x = 0; x < width; ++x) plane[x] = line[static_cast<size_t>(x) * 4 + c];
			detail::writeChannelRuns(out, plane.data(), width);
		}
	}
	return out;
}

// The file name that says which sky it is (and which version of this generator wrote it).
inline std::string skyFileName(const SkyParams& in) {
	const SkyParams p = clamped(in);
	char buf[96];
	std::snprintf(buf, sizeof buf, "sky_v1_e%.1f_a%.0f_t%.1f_g%.2f.hdr", p.sunElevation, p.sunAzimuth, p.turbidity, p.groundAlbedo);
	return buf;
}

// Makes sure the sky picture for `p` is in `dir` (created if needed) and says where. Written to a temporary name and renamed, so a half-written file is never read.
inline bool ensureSkyFile(const std::string& dir, const SkyParams& p, std::string& path, std::string& error) {
	namespace fs = std::filesystem;
	std::error_code ec;
	fs::create_directories(dir, ec);
	const fs::path target = fs::path(dir) / skyFileName(p);
	path = target.string();
	if (fs::exists(target, ec) && fs::file_size(target, ec) > 0) return true;
	const std::string data = encodeHdr(renderSkyImage(p, kImageWidth, kImageHeight), kImageWidth, kImageHeight);
	const fs::path tmp = fs::path(dir) / (skyFileName(p) + ".tmp");
	{
		std::ofstream f(tmp, std::ios::binary);
		if (!f) { error = "could not write " + tmp.string(); return false; }
		f.write(data.data(), static_cast<std::streamsize>(data.size()));
		if (!f) { error = "could not write " + tmp.string(); return false; }
	}
	fs::rename(tmp, target, ec);
	if (ec) { error = "could not move the sky picture into place: " + ec.message(); return false; }
	return true;
}

// Turns lights[skyIndex] (an Infinite light) into the sky of its own SkyParams: its picture is the generated sky (written into `imageDir`), and the scene's
// first Distant light, or a new one named "Sun", is aimed along the sun with the sun's colour and strength. Both are multiplied by the sky light's `intensity`.
inline bool applySunAndSky(std::vector<Light>& lights, size_t skyIndex, const std::string& imageDir, std::string& error) {
	if (skyIndex >= lights.size() || lights[skyIndex].kind != LightKind::Infinite) { error = "not a sky light"; return false; }
	Light& sky = lights[skyIndex];
	sky.sky = clamped(sky.sky);
	std::string path;
	if (!ensureSkyFile(imageDir, sky.sky, path, error)) return false;
	sky.physicalSky = true;
	sky.imageFile = path;
	const SunLight sun = sunLight(sky.sky);
	Light* sunLightPtr = nullptr;
	for (Light& l : lights)
		if (l.kind == LightKind::Distant) { sunLightPtr = &l; break; }
	if (!sunLightPtr) {
		lights.emplace_back();
		sunLightPtr = &lights.back();
		sunLightPtr->name = "Sun";
		sunLightPtr->kind = LightKind::Distant;
		sunLightPtr->target = {0.0, 0.0, 0.0};
	}
	Light& s = *sunLightPtr;
	const double distance = 10.0;
	s.position = {s.target.x + sun.towardSun.x * distance, s.target.y + sun.towardSun.y * distance, s.target.z + sun.towardSun.z * distance};
	s.color = sun.colour;
	s.intensity = sun.irradiance * lights[skyIndex].intensity;
	return true;
}

}  // namespace sky
}  // namespace scene_doc
