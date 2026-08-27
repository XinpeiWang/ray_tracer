#pragma once
// pbrt_load.h -- loads a .pbrt file from disk into flattened geometry.
//
// The layers below this are pure by design: pbrt_scene.h takes text and a
// FileResolver, pbrt_flatten.h takes a description and a MeshResolver, and
// ply_mesh.h takes bytes. That made them testable without a filesystem, and
// left exactly one thing unimplemented - actually finding the files.
//
// PATHS ARE RELATIVE TO THE SCENE, NOT THE WORKING DIRECTORY
// ----------------------------------------------------------
// This is the whole reason the resolvers were callbacks. A pbrt scene says
// `Include "geometry/killeroo.pbrt"` and `"string filename" "textures/x.png"`
// meaning relative to the scene file itself, so the same scene loads whether
// it is opened from its own directory or from anywhere else. Resolving against
// the process's working directory instead works exactly once - when you happen
// to have cd'd into the scene folder - and fails for every other caller,
// including the GUI, which runs from the package directory.
//
// Nested includes resolve against the file that named them, which matters
// because a scene's geometry/ subdirectory routinely includes its own
// neighbours by bare filename.

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pbrt_flatten.h"
#include "pbrt_scene.h"
#include "ply_mesh.h"
#include "measured_bxdf_loader.h"

#include "../external/stb_image.h"
#include "../external/tinyexr.h"

namespace pbrt_load {

struct LoadResult {
	bool ok = false;
	std::string error;
	pbrt_flatten::FlatScene scene;
};

namespace detail {

// Everything up to and including the final separator, or empty for a bare
// filename. Both separators are accepted: pbrt scenes are written on every
// platform and a Windows-authored one carries backslashes.
inline std::string directoryOf(const std::string &path) {
	const std::size_t slash = path.find_last_of("/\\");
	return (slash == std::string::npos) ? std::string() : path.substr(0, slash + 1);
}

inline bool isAbsolute(const std::string &path) {
	if (path.empty()) return false;
	if (path[0] == '/' || path[0] == '\\') return true;
	return path.size() > 1 && path[1] == ':';        // C:\...
}

inline std::string join(const std::string &dir, const std::string &path) {
	return (dir.empty() || isAbsolute(path)) ? path : dir + path;
}

inline bool readFile(const std::string &path, std::string &out) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return false;
	std::ostringstream ss;
	ss << in.rdbuf();
	out = ss.str();
	return true;
}

inline bool fileExists(const std::string &path) {
	std::ifstream in(path, std::ios::binary);
	return static_cast<bool>(in);
}

// Resolves `want` the same scene-directory-first, then as-given way as
// everything else in this file, but returns the resolved PATH rather than
// the file's contents - for a measured material's .bsdf tensor file, which
// is 7-15MB of binary data best read with seeks scattered across the file
// (measured_bxdf_loader.h opens it directly), not slurped into a string
// first just to be handed back to another reader. Empty return means
// neither location has the file.
inline std::string resolveExistingPath(const std::string &sceneDir, const std::string &want) {
	const std::string nearScene = join(sceneDir, want);
	if (fileExists(nearScene)) return nearScene;
	if (fileExists(want)) return want;
	return std::string();
}

inline bool endsWithCaseInsensitive(const std::string &path, const std::string &ext) {
	if (path.size() < ext.size()) return false;
	return std::equal(ext.rbegin(), ext.rend(), path.rbegin(),
		[](unsigned char a, unsigned char b) {
			return std::tolower(a) == std::tolower(b);
		});
}

// Re-samples a decoded equirectangular environment image so its pixel data
// bakes in LightSource "infinite"'s own CTM (pbrt_flatten::InfiniteLight::
// xform) - a sun/horizon feature otherwise faces the wrong way, since
// nothing downstream of this loader (sky_light.h on CPU, optix_sky_light.h/
// wavefront_sky_light.h on GPU - three independent sampling
// implementations) ever reads InfiniteLight::xform at all. Doing it once,
// here, on the raw pixel buffer means none of those three need to change:
// each already converts a world-space direction to/from an image (u,v) via
// the same latlong mapping pbrt uses (see sky_light.h's own header
// comment), so pre-rotating the SOURCE pixels is exactly equivalent to
// rotating the light and is invisible to every consumer.
//
// Only the transform's linear (3x3) part is used - translating an infinite
// light is meaningless (it has no position). This deliberately does not
// special-case a non-rotation linear part (anisotropic scale, or a
// reflection like sportscar-sky.pbrt's "Scale -1 1 1") - the general
// inverse-and-resample approach below handles those the same way, just
// with the image locally stretched/mirrored instead of purely rotated,
// matching what pbrt itself would render.
//
// Uses its OWN forward/inverse latlong mapping (matching sky_light.h's
// documented `sample()` formula: dir = (sin(theta)cos(phi), cos(theta),
// -sin(theta)sin(phi))) rather than calling sky_light.h's private
// dir_to_uv() (this loader has no access to it anyway - dir_to_uv is a
// private static member). At the time this was written, dir_to_uv() had a
// real, separate, pre-existing bug (negated y, +pi phase offset instead of
// the closed-form inverse of sample()) discovered while deriving this
// function's own correct mapping; that bug has since been fixed (dir_to_uv
// now matches this function's formula exactly), so the two are equivalent
// again - this still keeps its own copy rather than depending on a private
// member, not because of any remaining disagreement.
inline void applyInfiniteLightOrientation(const pbrt_scene::Matrix4 &xform,
										  std::vector<float> &pixels, int w, int h) {
	// Identity (the overwhelmingly common case - most scenes never rotate
	// their sky) is a free no-op; skip the O(w*h) resample entirely.
	bool isIdentity = true;
	for (int i = 0; i < 16 && isIdentity; ++i) {
		double expected = (i % 5 == 0) ? 1.0 : 0.0;  // diagonal = 1, else 0
		if (xform.m[i] != expected) isIdentity = false;
	}
	if (isIdentity || w <= 0 || h <= 0) return;

	pbrt_scene::Matrix4 inv;
	if (!xform.inverseAffine(inv)) return;  // singular CTM - leave unrotated rather than guess

	const double pi = 3.14159265358979323846;
	std::vector<float> rotated(pixels.size());

	auto wrapX = [w](int xx) { xx %= w; if (xx < 0) xx += w; return xx; };
	auto clampY = [h](int yy) { return yy < 0 ? 0 : (yy >= h ? h - 1 : yy); };

	for (int y = 0; y < h; ++y) {
		double v = (y + 0.5) / h;
		double theta = v * pi;
		double sinTheta = std::sin(theta), cosTheta = std::cos(theta);
		for (int x = 0; x < w; ++x) {
			double u = (x + 0.5) / w;
			double phi = u * 2.0 * pi;
			// World-space direction this OUTPUT pixel represents.
			double dx = sinTheta * std::cos(phi);
			double dy = cosTheta;
			double dz = -sinTheta * std::sin(phi);

			// Rotate back into the image's own local frame via the CTM's
			// inverse linear part (3x3, translation ignored - a direction,
			// not a point).
			double lx = inv.m[0]*dx + inv.m[1]*dy + inv.m[2]*dz;
			double ly = inv.m[4]*dx + inv.m[5]*dy + inv.m[6]*dz;
			double lz = inv.m[8]*dx + inv.m[9]*dy + inv.m[10]*dz;
			double len = std::sqrt(lx*lx + ly*ly + lz*lz);
			if (len < 1e-12) {
				for (int c = 0; c < 3; ++c) rotated[(y*w+x)*3+c] = 0.0f;
				continue;
			}
			lx /= len; ly /= len; lz /= len;

			// Closed-form inverse of dir = (sin(theta)cos(phi), cos(theta),
			// -sin(theta)sin(phi)): theta = acos(dir.y), phi = atan2(-dir.z, dir.x).
			double ct = ly; if (ct > 1.0) ct = 1.0; if (ct < -1.0) ct = -1.0;
			double srcTheta = std::acos(ct);
			double srcPhi = std::atan2(-lz, lx);
			if (srcPhi < 0.0) srcPhi += 2.0 * pi;
			double srcU = srcPhi / (2.0 * pi);
			double srcV = srcTheta / pi;

			// Bilinear sample of the ORIGINAL image at (srcU, srcV): phi
			// wraps around horizontally (periodic), theta clamps vertically
			// (the poles are real edges, not a seam).
			double fx = srcU * w - 0.5, fy = srcV * h - 0.5;
			int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
			double tx = fx - x0, ty = fy - y0;
			int X0 = wrapX(x0), X1 = wrapX(x0 + 1), Y0 = clampY(y0), Y1 = clampY(y0 + 1);
			for (int c = 0; c < 3; ++c) {
				double p00 = pixels[(Y0*w+X0)*3+c], p10 = pixels[(Y0*w+X1)*3+c];
				double p01 = pixels[(Y1*w+X0)*3+c], p11 = pixels[(Y1*w+X1)*3+c];
				double top = p00 + (p10 - p00) * tx;
				double bot = p01 + (p11 - p01) * tx;
				rotated[(y*w+x)*3+c] = static_cast<float>(top + (bot - top) * ty);
			}
		}
	}
	pixels.swap(rotated);
}

// Decodes an infinite light's environment-map bytes into row-major RGB float
// pixels (3 floats/pixel, linear) - the format sky_light's raw-buffer
// constructor (sky_light.h) accepts. Dispatches purely on `filename`'s
// extension, not the bytes themselves: pbrt-v4-scenes environment maps are
// exclusively .exr (needs tinyexr - stb_image cannot decode it) or .hdr
// (Radiance RGBE, stb_image's own native HDR format). Returns false, leaving
// the outputs untouched, on a format this cannot decode or a corrupt file -
// the caller falls back to the scene's constant-colour L the same way it
// would for a scene with no filename at all, rather than failing the load
// over one texture.
inline bool decodeInfiniteLightImage(const std::string &filename, const std::string &bytes,
									 std::vector<float> &outPixels, int &outW, int &outH) {
	if (endsWithCaseInsensitive(filename, ".exr")) {
		float *rgba = nullptr;
		int w = 0, h = 0;
		const char *err = nullptr;
		const int rc = LoadEXRFromMemory(&rgba, &w, &h,
			reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), &err);
		if (err) FreeEXRErrorMessage(err);
		if (rc != TINYEXR_SUCCESS || !rgba || w <= 0 || h <= 0) {
			if (rgba) free(rgba);
			return false;
		}
		outPixels.resize(static_cast<std::size_t>(w) * h * 3);
		for (int i = 0; i < w * h; ++i) {
			outPixels[i * 3 + 0] = rgba[i * 4 + 0];
			outPixels[i * 3 + 1] = rgba[i * 4 + 1];
			outPixels[i * 3 + 2] = rgba[i * 4 + 2];
		}
		free(rgba);
		outW = w; outH = h;
		return true;
	}

	// Everything else goes through stb_image's own HDR path (Radiance
	// .hdr/.pic - the same decoder rtw_image's filename constructor uses).
	int w = 0, h = 0, channels = 0;
	float *pixels = stbi_loadf_from_memory(
		reinterpret_cast<const unsigned char *>(bytes.data()),
		static_cast<int>(bytes.size()), &w, &h, &channels, 3);
	if (!pixels || w <= 0 || h <= 0) {
		if (pixels) stbi_image_free(pixels);
		return false;
	}
	outPixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 3);
	stbi_image_free(pixels);
	outW = w; outH = h;
	return true;
}

} // namespace detail

// Reads `path`, resolving Include and plymesh references relative to it.
inline LoadResult loadFile(const std::string &path) {
	using namespace detail;
	LoadResult r;

	std::string text;
	if (!readFile(path, text)) {
		r.error = "cannot open scene file: " + path;
		return r;
	}
	const std::string sceneDir = directoryOf(path);

	// An included file's own includes resolve against ITS directory, not the
	// top-level scene's - tracked by remembering the directory each resolved
	// path came from. Without this, geometry/a.pbrt including "b.pbrt" would
	// be looked for beside the scene rather than beside a.pbrt.
	std::vector<std::pair<std::string, std::string>> resolvedDirs;
	const pbrt_scene::FileResolver files =
		[&](const std::string &want, std::string &contents) {
			// Try the directory of whichever file most recently resolved,
			// then the scene directory, then the path as given.
			for (auto it = resolvedDirs.rbegin(); it != resolvedDirs.rend(); ++it) {
				if (readFile(join(it->second, want), contents)) {
					resolvedDirs.emplace_back(want, directoryOf(join(it->second, want)));
					return true;
				}
			}
			if (readFile(join(sceneDir, want), contents)) {
				resolvedDirs.emplace_back(want, directoryOf(join(sceneDir, want)));
				return true;
			}
			if (readFile(want, contents)) {
				resolvedDirs.emplace_back(want, directoryOf(want));
				return true;
			}
			return false;
		};

	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(text, files);
	if (!parsed.ok) {
		r.error = path + ": " + parsed.error;
		return r;
	}

	const pbrt_flatten::MeshResolver meshes =
		[&sceneDir](const std::string &want, std::vector<float> &positions,
					std::vector<int> &indices, std::vector<float> &uvs) {
			ply_mesh::LoadResult m = ply_mesh::loadFile(join(sceneDir, want));
			if (!m.ok) m = ply_mesh::loadFile(want);   // already absolute, or cwd-relative
			// A scene naming "x.ply" when the folder ships "x.ply.gz" is common
			// enough to be worth one extra open. Without this the failure is a
			// missing-file message about a file that is plainly right there.
			if (!m.ok) m = ply_mesh::loadFile(join(sceneDir, want) + ".gz");
			if (!m.ok) m = ply_mesh::loadFile(want + ".gz");
			if (!m.ok) return false;
			positions = std::move(m.mesh.positions);
			indices = std::move(m.mesh.indices);
			uvs = std::move(m.mesh.uvs);
			return true;
		};

	r.scene = pbrt_flatten::flatten(parsed.scene, meshes);
	r.ok = true;

	// Infinite light's image, if it has one. Resolved the same
	// scene-directory-then-as-given way as everything else in this file
	// (loadFileNear() below is this exact lookup, factored out for a
	// caller that only has the flattened result to work from - inlined
	// here instead since sceneDir/readFile are already in scope).
	pbrt_flatten::InfiniteLight &sky = r.scene.infiniteLight;
	if (sky.present && !sky.imageFile.empty()) {
		std::string bytes;
		if (readFile(join(sceneDir, sky.imageFile), bytes) || readFile(sky.imageFile, bytes)) {
			if (!decodeInfiniteLightImage(sky.imageFile, bytes,
										  sky.imagePixels, sky.imageWidth, sky.imageHeight)) {
				r.scene.warnings.push_back(
					{0, path, "infinite light image '" + sky.imageFile +
						"' could not be decoded; using its constant colour instead"});
			} else if (!sky.hasPortal) {
				// Bakes the LightSource "infinite" directive's own CTM into
				// the pixels themselves - see applyInfiniteLightOrientation()'s
				// own comment for why this is done once here instead of in
				// either backend's sampling code. Its resample math is
				// hardcoded for the equirectangular dir_to_uv mapping plain
				// sky_light/InfiniteLight use - a portal light's image is a
				// DIFFERENT format (equal-area octahedral, see
				// PortalImageInfiniteLightData's own comment), which this
				// would corrupt if applied, so portal images skip it
				// entirely. Known, pre-existing limitation this inherits
				// (not introduced here): the portal's geometry (its 4
				// corners) IS correctly transformed by this same CTM at
				// parse time (pbrt_flatten.h), but the environment image's
				// own orientation is not - a rotated portal light's image
				// renders as if that rotation were identity. Fixing that
				// would need PortalImageInfiniteLightData itself to accept
				// a transform, which it currently does not (see its own
				// constructor comment).
			}
		} else {
			r.scene.warnings.push_back(
				{0, path, "infinite light image '" + sky.imageFile +
					"' could not be read; using its constant colour instead"});
		}
	}

	// Measured materials' .bsdf tensor files. Same scene-directory-then-as-
	// given resolution as everything else here, but resolved to a PATH
	// (resolveExistingPath) rather than read into a string - the tensor
	// reader wants to fopen/seek the file itself (measured_bxdf_loader.h).
	// This also DOUBLES as validation: parsing the file now, via the same
	// cache pbrt_cpu_builder.h's `class measured` will hit, means a
	// malformed or unreadable .bsdf is reported as a warning here (with the
	// scene path attached, matching every other diagnostic in this
	// function) rather than silently rendering that material as flat black
	// or crashing deeper in scene construction. Material::measuredFilename
	// is overwritten with the resolved path on success so pbrt_cpu_builder.h
	// never needs to know the scene's own directory - same trick
	// FlatScene::InfiniteLight's imageFile/imagePixels split already uses.
	for (pbrt_flatten::Material &m : r.scene.materials) {
		if (m.kind != pbrt_flatten::MaterialKind::Measured) continue;
		if (m.measuredFilename.empty()) continue;   // already warned by flatten()

		const std::string resolved = resolveExistingPath(sceneDir, m.measuredFilename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "measured material's file '" + m.measuredFilename +
					"' could not be found; falling back to a diffuse approximation"});
			m.measuredFilename.clear();
			continue;
		}

		std::string loadError;
		if (!measured_bxdf_io::GetMeasuredBRDFDataCached(resolved, loadError)) {
			r.scene.warnings.push_back(
				{0, path, "measured material's file '" + m.measuredFilename +
					"' could not be loaded (" + loadError +
					"); falling back to a diffuse approximation"});
			m.measuredFilename.clear();
			continue;
		}

		m.measuredFilename = resolved;
	}

	// Diffuse materials' imagemap "reflectance" texture (Material::
	// textureFilename - see that field's own comment). Same scene-directory-
	// then-as-given resolution as everything else here; unlike the measured-
	// material tensor above, decoding is left to pbrt_cpu_builder.h (an
	// mipmap_texture-backed lambertian, mirroring the OBJ/MTL map_Kd path in
	// mesh.h) rather than validated here, since a plain image decode has
	// nothing worth caching ahead of time the way the .bsdf tensor reader
	// does.
	for (pbrt_flatten::Material &m : r.scene.materials) {
		if (m.textureFilename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, m.textureFilename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "material's reflectance texture image '" + m.textureFilename +
					"' could not be found; falling back to a constant colour instead"});
			m.textureFilename.clear();
			continue;
		}

		m.textureFilename = resolved;
	}

	// DiffuseTransmission's own imagemap "transmittance" texture (Material::
	// transmittanceTextureFilename - see that field's own comment). Same
	// resolution convention as textureFilename just above.
	for (pbrt_flatten::Material &m : r.scene.materials) {
		if (m.transmittanceTextureFilename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, m.transmittanceTextureFilename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "material's transmittance texture image '" + m.transmittanceTextureFilename +
					"' could not be found; falling back to a constant colour instead"});
			m.transmittanceTextureFilename.clear();
			continue;
		}

		m.transmittanceTextureFilename = resolved;
	}

	// Dielectric's own imagemap "roughness" texture (Material::
	// roughnessTextureFilename - see that field's own comment). Same
	// resolution convention as textureFilename above.
	for (pbrt_flatten::Material &m : r.scene.materials) {
		if (m.roughnessTextureFilename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, m.roughnessTextureFilename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "material's roughness texture image '" + m.roughnessTextureFilename +
					"' could not be found; falling back to a constant roughness instead"});
			m.roughnessTextureFilename.clear();
			continue;
		}

		m.roughnessTextureFilename = resolved;
	}

	// Shape "alpha" cutout masks (Material::alphaTextureFilename - see that
	// field's own comment). Same resolution convention as textureFilename
	// just above; decoding is likewise left to the CPU/GPU builders.
	for (pbrt_flatten::Material &m : r.scene.materials) {
		if (m.alphaTextureFilename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, m.alphaTextureFilename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "shape's alpha-cutout texture image '" + m.alphaTextureFilename +
					"' could not be found; the shape will render fully opaque"});
			m.alphaTextureFilename.clear();
			continue;
		}

		m.alphaTextureFilename = resolved;
	}

	// Material "texture displacement" (bump mapping - Material::
	// displacementTextureFilename's own comment). Same resolution
	// convention as textureFilename/alphaTextureFilename above;
	// displacementScale needs no resolution (already a final numeric value).
	for (pbrt_flatten::Material &m : r.scene.materials) {
		if (m.displacementTextureFilename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, m.displacementTextureFilename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "material's displacement (bump) texture image '" + m.displacementTextureFilename +
					"' could not be found; the surface will render without bump detail"});
			m.displacementTextureFilename.clear();
			continue;
		}

		m.displacementTextureFilename = resolved;
	}

	// Checkerboard/mix's own nested-imagemap tex1/tex2 (Material::
	// checkerTex1Filename/checkerTex2Filename/mixTex1Filename/
	// mixTex2Filename - see hasCheckerReflectance/hasMixReflectance's own
	// comments in pbrt_flatten.h). Same resolve-or-warn-and-clear shape as
	// every loop above, factored into a small local helper since this adds
	// 4 more near-identical fields at once.
	auto resolveNestedTextureField = [&](std::string pbrt_flatten::Material::*field, const char *what) {
		for (pbrt_flatten::Material &m : r.scene.materials) {
			std::string &f = m.*field;
			if (f.empty()) continue;
			const std::string resolved = resolveExistingPath(sceneDir, f);
			if (resolved.empty()) {
				r.scene.warnings.push_back(
					{0, path, std::string("material's ") + what + " texture image '" + f +
						"' could not be found; falling back to a constant colour instead"});
				f.clear();
				continue;
			}
			f = resolved;
		}
	};
	resolveNestedTextureField(&pbrt_flatten::Material::checkerTex1Filename, "checkerboard tex1");
	resolveNestedTextureField(&pbrt_flatten::Material::checkerTex2Filename, "checkerboard tex2");
	resolveNestedTextureField(&pbrt_flatten::Material::mixTex1Filename, "mix tex1");
	resolveNestedTextureField(&pbrt_flatten::Material::mixTex2Filename, "mix tex2");

	// AreaLightSource "diffuse"'s "filename" (spatially-varying image
	// emission - Emission::filename's own comment). Same resolution
	// convention as textureFilename/alphaTextureFilename/
	// displacementTextureFilename above; decoding is likewise left to the
	// CPU/GPU builders.
	for (pbrt_flatten::Emission &e : r.scene.areaLights) {
		if (e.filename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, e.filename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "area light's image emission file '" + e.filename +
					"' could not be found; falling back to its flat colour instead"});
			e.filename.clear();
			continue;
		}

		e.filename = resolved;
	}

	// LightSource "goniometric"/"projection"'s own "filename" (PunctualLight::
	// filename's own comment) - same resolution convention as the other
	// filename fields above; decoding (a real equal-area profile image for
	// goniometric, an ordinary photo/slide for projection) is likewise left
	// to the CPU/GPU builders, which already have a direct-from-resolved-
	// path image decode utility to reuse (mipmap_texture/
	// getOrBuildPbrtImageTexture) rather than one built specifically for
	// this. On a resolve failure, `filename` is cleared (so the builders'
	// existing empty-means-fall-back-to-uniform branch applies unchanged)
	// but hadImageFilename is left true, so a caller can still tell "named
	// but not found" apart from "never named" after this point.
	for (pbrt_flatten::PunctualLight &pl : r.scene.punctualLights) {
		if (pl.filename.empty()) continue;

		const std::string resolved = resolveExistingPath(sceneDir, pl.filename);
		if (resolved.empty()) {
			r.scene.warnings.push_back(
				{0, path, "punctual light's image file '" + pl.filename +
					"' could not be found; falling back to a uniform profile/slide instead"});
			pl.filename.clear();
			continue;
		}

		pl.filename = resolved;
	}

	return r;
}

// Reads a file a scene references outside the Include/plymesh graphs this
// header already resolves - currently just a "realistic" camera's lensfile,
// which pbrt_flatten.h can name but has no file access of its own to read.
// Same relative-to-the-scene-first, then as-given convention as everything
// else in this file, so a lens file living beside its scene is found without
// the caller having to know the scene's own directory.
inline bool loadFileNear(const std::string &scenePath, const std::string &relPath,
						 std::string &outContents) {
	using namespace detail;
	const std::string sceneDir = directoryOf(scenePath);
	if (readFile(join(sceneDir, relPath), outContents)) return true;
	return readFile(relPath, outContents);
}

// Parses a pbrt-format lens description table: whitespace-separated
// "radius thickness eta aperture-diameter" rows in millimetres, '#' comments
// to end of line, blank lines ignored - the format pbrt-v4 ships its own
// sample lens files in (e.g. dgauss.22deg.dat), matching what
// RealisticCamera's own lens-table constructor argument already expects, so
// the result can be passed straight through with no further conversion.
// Returns however many complete 4-number rows were found; a malformed file
// that does not divide evenly drops its trailing partial row rather than
// misreading every row after it out of phase.
inline std::vector<double> parseLensFile(const std::string &text) {
	std::vector<double> row;
	std::istringstream in(text);
	std::string line;
	while (std::getline(in, line)) {
		const std::size_t hash = line.find('#');
		if (hash != std::string::npos) line.resize(hash);
		std::istringstream ls(line);
		double v;
		while (ls >> v) row.push_back(v);
	}
	row.resize((row.size() / 4) * 4);
	return row;
}

} // namespace pbrt_load
