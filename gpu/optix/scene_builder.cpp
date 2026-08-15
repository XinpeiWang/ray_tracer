// Scene Builder Implementation
// Converts shared scene definitions to OptiX geometry

#include "scene_builder.h"
#include "optix_math_helpers.h"
// Declarations only (extern "C", no scene_registry.h class hierarchy) -
// the actual implementation lives in cpu_renderer.lib, resolved at final
// link time exactly like stb_image.h's stbi_loadf above (see that
// include's own comment) - both launcher.vcxproj and
// tests/ray_tracer_tests.vcxproj already link cpu_renderer.lib alongside
// this project's own output.
#include "../../cpu_renderer/cpu_interface.h"
#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_load.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "../../src/shared/conductor_data.h"
#include "../../src/shared/rgb_nebula_generator.h"
#include "../../src/shared/curve_tessellate.h"
#include "../../src/shared/cameras.h"
#include "../../src/shared/cornell_box_data.h"
// Declarations only (no STB_IMAGE_IMPLEMENTATION) - the actual
// implementation is already compiled once into cpu_renderer.lib (see
// src/external/stb_image_impl.cpp), and launcher.vcxproj always links
// both cpu_renderer.lib and optix_renderer.lib into ray_tracer.exe
// together, so stbi_loadf resolves at final link time without needing a
// second copy of the implementation (which would collide as a duplicate
// symbol if compiled into optix_renderer.lib too).
#include "../../src/external/stb_image.h"

namespace {
	// Constants for Cornell Box dimensions
	constexpr float kBoxSize = 555.0f;
	constexpr float kLightIntensity = 15.0f;
	constexpr float kGlassIOR = 1.5f;

	// Helper to safely cast vector size to int with bounds checking
	inline int safe_cast_to_int(size_t value) {
		assert(value <= static_cast<size_t>(INT_MAX) && "Material index overflow");
		return static_cast<int>(value);
	}

	// ------------------------------------------------------------------
	// Named-field material factories.
	//
	// MaterialData (optix_types.h) is a flat 8-field POD reused across all
	// 16 MaterialTypes -- e.g. `.fuzz` means Metal roughness, GGX alpha,
	// Henyey-Greenstein g, or Hair's beta_m depending on `.type`, and
	// `.eta_c` alone means 4 unrelated things (Conductor's complex IOR,
	// Hair's beta_n/alpha_deg pair, DielectricMedium's sigma_t, Principled's
	// metallic/clearcoat/clearcoat_rough triple). Before these helpers,
	// every call site built a MaterialData by raw positional aggregate-init
	// or manual field-by-field assignment, so the *meaning* of each
	// argument depended entirely on which MaterialType was named on the
	// same line -- a transposed argument (e.g. swapping fuzz/ior) would
	// compile silently and render a wrong-but-plausible image.
	//
	// These functions don't change MaterialData's layout (still a GPU
	// upload requirement: one flat contiguous array, no vtable/RTTI) but
	// give every call site a parameter name for what it's actually
	// setting, matching the field-reuse mapping already documented on each
	// MaterialType enumerator in optix_types.h. Each returns the new
	// material's index (folding in the safe_cast_to_int(size())+push_back
	// pattern every existing call site repeated by hand).
	// ------------------------------------------------------------------

	inline int add_lambertian(SceneData& scene, float3 albedo, int textureIdx = -1) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Lambertian;
		m.albedo = albedo;
		m.textureIdx = textureIdx;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_metal(SceneData& scene, float3 albedo, float roughness) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Metal;
		m.albedo = albedo;
		m.roughness = roughness;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_dielectric(SceneData& scene, float ior) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Dielectric;
		m.ior = ior;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_diffuse_light(SceneData& scene, float3 emission) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::DiffuseLight;
		m.emission = emission;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_rough_dielectric(SceneData& scene, float roughness, float ior) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::RoughDielectric;
		m.roughness = roughness;
		m.ior = ior;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_conductor(SceneData& scene, float3 eta, float3 k, float roughness) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Conductor;
		m.roughness = roughness;
		m.eta_c = eta;
		m.k_c = k;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_coated_diffuse(SceneData& scene, float3 albedo, float coatRoughness, float coatIor) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::CoatedDiffuse;
		m.albedo = albedo;
		m.roughness = coatRoughness;
		m.ior = coatIor;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_thin_dielectric(SceneData& scene, float ior) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::ThinDielectric;
		m.ior = ior;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_coated_conductor(SceneData& scene, float3 eta, float3 k, float coatRoughness, float coatIor) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::CoatedConductor;
		m.roughness = coatRoughness;
		m.ior = coatIor;
		m.eta_c = eta;
		m.k_c = k;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_diffuse_transmission(SceneData& scene, float3 reflectance, float3 transmittance) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::DiffuseTransmission;
		m.reflectance = reflectance;
		m.transmittance = transmittance;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_normalized_fresnel(SceneData& scene, float ior) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::NormalizedFresnel;
		m.ior = ior;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_medium(SceneData& scene, float3 albedo, float g, float sigma_t) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Medium;
		m.medium_albedo = albedo;
		m.g = g;
		m.sigma_t = sigma_t;
		scene.materials.push_back(m);
		return idx;
	}

	// Pushes `medium` into scene.cloudMediums and returns a MaterialData index
	// referencing it via cloud_medium_extra.cloudMediumIdx - see
	// MaterialType::CloudMedium's comment in optix_types.h for why this needs
	// an index into a separate array rather than direct field reuse the way
	// add_medium() above does.
	inline int add_cloud_medium(SceneData& scene, const CloudMedium<float>& medium,
	                             float3 albedo) {
		const int cloudIdx = safe_cast_to_int(scene.cloudMediums.size());
		scene.cloudMediums.push_back(medium);

		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::CloudMedium;
		m.medium_albedo = albedo;
		m.g = medium.phase_g;
		m.cloud_medium_extra.cloudMediumIdx = static_cast<float>(cloudIdx);
		scene.materials.push_back(m);
		return idx;
	}

	// Uploads a heterogeneous RGB grid medium's metadata + flat voxel data
	// (R block, then G, then B - see GpuRgbGridMedium::dataOffset), then adds
	// a material referencing it via rgb_grid_medium_extra.rgbGridMediumIdx -
	// see MaterialType::RgbGridMedium's comment in optix_types.h. `meta`'s
	// dataOffset field is overwritten here; the caller only needs to fill in
	// every other field before calling this.
	inline int add_rgb_grid_medium(SceneData& scene, GpuRgbGridMedium meta,
	                                const std::vector<float>& r,
	                                const std::vector<float>& g,
	                                const std::vector<float>& b) {
		meta.dataOffset = safe_cast_to_int(scene.rgbGridData.size());
		scene.rgbGridData.insert(scene.rgbGridData.end(), r.begin(), r.end());
		scene.rgbGridData.insert(scene.rgbGridData.end(), g.begin(), g.end());
		scene.rgbGridData.insert(scene.rgbGridData.end(), b.begin(), b.end());

		const int mediumIdx = safe_cast_to_int(scene.rgbGridMediums.size());
		scene.rgbGridMediums.push_back(meta);

		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::RgbGridMedium;
		m.rgb_grid_medium_extra.rgbGridMediumIdx = static_cast<float>(mediumIdx);
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_hair(SceneData& scene, float3 sigma_a, float beta_m, float eta, float beta_n, float alpha_deg) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Hair;
		m.sigma_a = sigma_a;
		m.beta_m = beta_m;
		m.eta = eta;
		m.hair_extra.beta_n = beta_n;
		m.hair_extra.alpha_deg = alpha_deg;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_dielectric_medium(SceneData& scene, float3 albedo, float ior, float sigma_t, float g = 0.0f) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::DielectricMedium;
		m.medium_albedo = albedo;
		m.g = g;
		m.ior = ior;
		m.dielectric_medium_extra.sigma_t = sigma_t;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_normal_mapped_lambertian(SceneData& scene, float3 albedo, int normalMapTexIdx) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::NormalMappedLambertian;
		m.albedo = albedo;
		m.textureIdx = normalMapTexIdx;
		scene.materials.push_back(m);
		return idx;
	}

	inline int add_principled(SceneData& scene, float3 baseColor, float ior, float roughness,
			float metallic, float clearcoat, float clearcoatRoughness) {
		const int idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Principled;
		m.base_color = baseColor;
		m.ior = ior;
		m.roughness = roughness;
		m.principled_params.metallic = metallic;
		m.principled_params.clearcoat = clearcoat;
		m.principled_params.clearcoat_rough = clearcoatRoughness;
		scene.materials.push_back(m);
		return idx;
	}

	// Helper to rotate a point around Y axis
	inline float3 rotate_y(const float3& p, float angle_degrees) {
		const float radians = angle_degrees * (3.14159265358979323846f / 180.0f);
		const float cos_theta = std::cos(radians);
		const float sin_theta = std::sin(radians);
		return make_float3(
			cos_theta * p.x + sin_theta * p.z,
			p.y,
			-sin_theta * p.x + cos_theta * p.z
		);
	}

	// Helper to translate a point
	inline float3 translate(const float3& p, const float3& offset) {
		return make_float3(p.x + offset.x, p.y + offset.y, p.z + offset.z);
	}

	// Fills the 12-float camera_params layout (origin, lower_left_corner,
	// horizontal, vertical) shared by every GPU scene's pinhole/thin-lens
	// camera - the same math src/TheRestOfYourLife/camera.h's initialize()
	// uses on the CPU side. focus_dist=1.0 (the default) reproduces the
	// plain pinhole camera every non-depth-of-field scene uses; scenes with
	// actual defocus blur (e.g. scene 22) pass their real focus distance,
	// which scales both the viewport size and the lower-left-corner offset
	// - see camera.h's own viewport_height = 2*h*focus_dist formula. When a
	// caller additionally needs the camera basis vectors (u, v, w) - e.g.
	// for defocus-disk sampling setup - pass non-null out_u/out_v/out_w.
	void build_pinhole_camera_params(
		const float3& lookfrom, const float3& lookat, const float3& vup,
		float vfov_degrees, float aspect, float focus_dist,
		float* camera_params,
		float3* out_u = nullptr, float3* out_v = nullptr, float3* out_w = nullptr
	) {
		constexpr float kPi = 3.14159265358979323846f;
		const float theta = vfov_degrees * kPi / 180.0f;
		const float h = tanf(theta / 2.0f);
		const float viewport_height = 2.0f * h * focus_dist;
		const float viewport_width = aspect * viewport_height;

		const float3 view_direction = make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z);
		const float3 w = normalize(view_direction);
		const float3 u = normalize(cross(vup, w));
		const float3 v = cross(w, u);

		const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
		const float3 vertical = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
		const float3 lower_left_corner = make_float3(
			lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - focus_dist * w.x,
			lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - focus_dist * w.y,
			lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - focus_dist * w.z
		);

		camera_params[0] = lookfrom.x;           camera_params[1] = lookfrom.y;           camera_params[2] = lookfrom.z;
		camera_params[3] = lower_left_corner.x;  camera_params[4] = lower_left_corner.y;  camera_params[5] = lower_left_corner.z;
		camera_params[6] = horizontal.x;         camera_params[7] = horizontal.y;         camera_params[8] = horizontal.z;
		camera_params[9] = vertical.x;           camera_params[10] = vertical.y;          camera_params[11] = vertical.z;

		if (out_u) *out_u = u;
		if (out_v) *out_v = v;
		if (out_w) *out_w = w;
	}

	// Resolves the lookfrom for a Fixed-mode scene (no CameraMode::
	// UserControlled in its src/TheRestOfYourLife/scene_registry.h entry):
	// honor cam_x/y/z only when force_camera_override is set (video mode's
	// per-frame animated position, or an explicit CLI/GUI override) -
	// otherwise fall back to the scene's own registered default, matching
	// cpu_interface.cpp's `(cc.mode == CameraMode::UserControlled ||
	// force_camera_override) ? cam_x,y,z : cc.lookfrom` precedence exactly.
	//
	// Centralized here instead of each scene case re-deriving this ternary
	// by hand: that was the actual root cause of a real bug (see git log
	// "Fix video mode's frozen camera") - nine-plus scenes' camera cases,
	// including one added earlier the same session, simply never got the
	// ternary, so video mode silently rendered the same frozen frame for
	// every one of them. A single call site here can't be skipped by
	// accident the way a hand-copied multi-line ternary can.
	inline float3 resolve_fixed_lookfrom(
		bool force_camera_override, double cam_x, double cam_y, double cam_z,
		float default_x, float default_y, float default_z)
	{
		return force_camera_override
			? make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z))
			: make_float3(default_x, default_y, default_z);
	}

	// Helper to check if a material is emissive
	inline bool is_emissive(const SceneData& scene, int material_idx) {
		if (material_idx < 0 || material_idx >= static_cast<int>(scene.materials.size()))
			return false;
		const auto& mat = scene.materials[material_idx];
		return mat.type == MaterialType::DiffuseLight;
	}

	// Helper to add a quad with optional rotation and translation
	inline void add_transformed_quad(
		SceneData& scene,
		const float3& Q,
		const float3& u,
		const float3& v,
		int material_idx,
		float rotation_y_degrees = 0.0f,
		const float3& translation = make_float3(0, 0, 0))
	{
		// Apply rotation first, then translation (matching CPU transform order)
		float3 Q_transformed = Q;
		float3 u_transformed = u;
		float3 v_transformed = v;

		if (rotation_y_degrees != 0.0f) {
			Q_transformed = rotate_y(Q, rotation_y_degrees);
			u_transformed = rotate_y(u, rotation_y_degrees);
			v_transformed = rotate_y(v, rotation_y_degrees);
		}

		Q_transformed = translate(Q_transformed, translation);

		// Build the quad
		QuadData quad{};
		quad.Q = Q_transformed;
		quad.u = u_transformed;
		quad.v = v_transformed;
		const float3 quad_cross = cross(quad.u, quad.v);
		quad.w = quad_cross;
		quad.normal = normalize(quad_cross);
		quad.D = dot(quad.normal, quad.Q);
		quad.materialIdx = material_idx;
		scene.quads.push_back(quad);

		// Track if this quad is a light
		if (is_emissive(scene, material_idx)) {
			scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
			scene.lightKinds.push_back(GpuLightKind::Quad); // false = quad
		}
	}

	// Helper to add a box (6 quads) with rotation and translation
	// Matches CPU box() function from src/TheRestOfYourLife/quad.h
	inline void add_box(
		SceneData& scene,
		const float3& corner_a,
		const float3& corner_b,
		int material_idx,
		float rotation_y_degrees = 0.0f,
		const float3& translation = make_float3(0, 0, 0))
	{
		// Construct min and max corners
		const float3 min_corner = make_float3(
			fminf(corner_a.x, corner_b.x),
			fminf(corner_a.y, corner_b.y),
			fminf(corner_a.z, corner_b.z)
		);
		const float3 max_corner = make_float3(
			fmaxf(corner_a.x, corner_b.x),
			fmaxf(corner_a.y, corner_b.y),
			fmaxf(corner_a.z, corner_b.z)
		);

		const float3 dx = make_float3(max_corner.x - min_corner.x, 0, 0);
		const float3 dy = make_float3(0, max_corner.y - min_corner.y, 0);
		const float3 dz = make_float3(0, 0, max_corner.z - min_corner.z);

		// Six faces matching CPU box() in quad.h:
		// Front face (min.x, min.y, max.z)
		add_transformed_quad(scene, make_float3(min_corner.x, min_corner.y, max_corner.z), dx, dy, material_idx, rotation_y_degrees, translation);
		// Right face (max.x, min.y, max.z)
		add_transformed_quad(scene, make_float3(max_corner.x, min_corner.y, max_corner.z), make_float3(-dz.z, 0, 0), dy, material_idx, rotation_y_degrees, translation);
		// Back face (max.x, min.y, min.z)
		add_transformed_quad(scene, make_float3(max_corner.x, min_corner.y, min_corner.z), make_float3(-dx.x, 0, 0), dy, material_idx, rotation_y_degrees, translation);
		// Left face (min.x, min.y, min.z)
		add_transformed_quad(scene, make_float3(min_corner.x, min_corner.y, min_corner.z), dz, dy, material_idx, rotation_y_degrees, translation);
		// Top face (min.x, max.y, max.z)
		add_transformed_quad(scene, make_float3(min_corner.x, max_corner.y, max_corner.z), dx, make_float3(0, 0, -dz.z), material_idx, rotation_y_degrees, translation);
		// Bottom face (min.x, min.y, min.z)
		add_transformed_quad(scene, make_float3(min_corner.x, min_corner.y, min_corner.z), dx, dz, material_idx, rotation_y_degrees, translation);
	}

	// Loads an image file into scene.texturePixels (appended) + a new
	// TextureData entry in scene.textures, returning its index. Matches
	// CPU's rtw_stb_image.h + texture.h::image_texture exactly: tries a
	// few relative search paths, converts stb's float pixel data to 8-bit
	// RGB bytes via the same float_to_byte formula (<=0 -> 0, >=1 -> 255,
	// else 256*value, rtw_stb_image.h:120-126). On load failure, still
	// returns a valid index whose width/height are 0 - sample_texture()
	// (optix_device_helpers.h) treats that as CPU's own solid-cyan
	// missing-texture fallback (texture.h:76), not a crash.
	inline int load_image_texture_gpu(SceneData& scene, const char* filename) {
		int width = 0, height = 0, channels = 0;
		float* fdata = nullptr;
		const char* search_prefixes[] = { "", "images/", "../images/", "../../images/" };
		for (const char* prefix : search_prefixes) {
			std::string path = std::string(prefix) + filename;
			fdata = stbi_loadf(path.c_str(), &width, &height, &channels, 3);
			if (fdata) break;
		}

		TextureData tex{};
		tex.kind = TextureKind::Image;
		tex.noiseScale = 0.0f;
		if (!fdata) {
			std::cerr << "[OptiX] Could not load image texture '" << filename
					   << "' (tried a few relative paths) - using solid-cyan "
					   << "debug fallback, matching CPU's own missing-texture behavior.\n";
			tex.pixelOffset = 0;
			tex.width = 0;
			tex.height = 0;
		} else {
			tex.pixelOffset = safe_cast_to_int(scene.texturePixels.size());
			tex.width = width;
			tex.height = height;
			const size_t total = static_cast<size_t>(width) * height * 3;
			scene.texturePixels.resize(scene.texturePixels.size() + total);
			unsigned char* out = scene.texturePixels.data() + tex.pixelOffset;
			for (size_t i = 0; i < total; ++i) {
				const float v = fdata[i];
				out[i] = (v <= 0.0f) ? 0 : (v >= 1.0f ? 255 : static_cast<unsigned char>(256.0f * v));
			}
			stbi_image_free(fdata);
		}
		scene.textures.push_back(tex);
		return static_cast<int>(scene.textures.size()) - 1;
	}

	// Registers a Perlin-noise texture (no pixel data - see
	// TextureKind::Noise in optix_types.h). Matches CPU's
	// noise_texture(scale) constructor exactly - see sample_texture()'s
	// Noise branch (optix_device_helpers.h) for the actual turbulence
	// formula this is evaluated with.
	inline int add_noise_texture_gpu(SceneData& scene, float scale) {
		TextureData tex{};
		tex.kind = TextureKind::Noise;
		tex.pixelOffset = 0;
		tex.width = 0;
		tex.height = 0;
		tex.noiseScale = scale;
		scene.textures.push_back(tex);
		return static_cast<int>(scene.textures.size()) - 1;
	}

	// Registers a checker texture (no pixel data - see TextureKind::Checker
	// in optix_types.h). Matches CPU's checker_texture(scale, c1, c2)
	// constructor exactly (texture.h:50-51) - stores 1/scale directly in
	// noiseScale so sample_texture() (optix_device_helpers.h) can multiply
	// straight through, same as checker_texture's own inv_scale member.
	inline int add_checker_texture_gpu(SceneData& scene, float scale, float3 color1, float3 color2) {
		TextureData tex{};
		tex.kind = TextureKind::Checker;
		tex.pixelOffset = 0;
		tex.width = 0;
		tex.height = 0;
		tex.noiseScale = 1.0f / scale;
		tex.color1 = color1;
		tex.color2 = color2;
		scene.textures.push_back(tex);
		return static_cast<int>(scene.textures.size()) - 1;
	}

	// Minimal Wavefront OBJ triangle loader for GPU scene building: loads
	// positions ("v"), optional per-vertex normals ("vn"), and faces ("f",
	// fan-triangulated, any "p", "p/t", "p//n" or "p/t/n" index format - vt
	// is still ignored, only p and n are used) directly into
	// SceneData::triangles, applying the same scale/offset transform CPU's
	// src/TheRestOfYourLife/mesh.h::load_obj() takes. This is a bare-bones
	// reimplementation rather than a call into mesh.h itself: that loader
	// builds a CPU hittable_list/bvh_node of the full CPU material/hittable
	// class hierarchy, which the GPU scene builder has no use for and
	// otherwise never touches - GPU only needs the flat TriangleData array
	// this writes straight into `scene`. Search path matches mesh.h's
	// load_obj() exactly, so a single models/ asset works from both
	// renderers regardless of the current working directory the renderer
	// happens to run from.
	inline void load_obj_triangles_gpu(SceneData& scene, const char* filename,
			int materialIdx, float scale, float3 offset) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			static const char* kSearchPrefixes[] = {
				"models/", "../models/", "../../models/",
				"../../../models/", "../../../../models/", "../../../../../models/"
			};
			for (const char* prefix : kSearchPrefixes) {
				file.clear();
				file.open(std::string(prefix) + filename);
				if (file.is_open()) break;
			}
		}
		if (!file.is_open()) {
			std::cerr << "[OptiX] Could not load mesh '" << filename
					   << "' (tried a few relative paths) - scene will be missing this geometry.\n";
			return;
		}

		std::vector<float3> positions;
		// Normals are unit direction vectors, unaffected by the uniform
		// scale/offset applied to positions (matches CPU's mesh.h, which
		// likewise only transforms raw_pos, not raw_norm).
		std::vector<float3> normals;
		// OBJ format lists all "v"/"vn"/"vt" data before any "f" line
		// references it, so by the time the first face is parsed, `normals`
		// already holds every vertex normal the file has (or none, if it
		// has none) - checking !normals.empty() per-face is equivalent to
		// (and simpler than) a separate up-front presence scan.
		std::string line;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#') continue;
			std::istringstream ss(line);
			std::string tok;
			ss >> tok;
			if (tok == "v") {
				float x, y, z;
				ss >> x >> y >> z;
				positions.push_back(make_float3(x * scale + offset.x, y * scale + offset.y, z * scale + offset.z));
			} else if (tok == "vn") {
				float x, y, z;
				ss >> x >> y >> z;
				normals.push_back(normalize(make_float3(x, y, z)));
			} else if (tok == "f") {
				std::vector<int> idx, nIdx;
				std::string fv;
				// OBJ indices may be negative ("relative"): -1 refers to the
				// most-recently-defined v/vn, resolved against however many
				// have been parsed so far in the file - matches CPU mesh.h's
				// load_obj() fix (see that function's own comment for why:
				// rungholt.obj, a McGuire Computer Graphics Archive scene,
				// uses this convention throughout, and the previous `p - 1`/
				// `n - 1` here silently dropped every negative-indexed face,
				// same bug as the CPU loader had).
				auto resolveIdx = [](int raw, size_t countSoFar) -> int {
					return raw > 0 ? raw - 1 : static_cast<int>(countSoFar) + raw;
				};
				while (ss >> fv) {
					// Possible formats: p   p/t   p//n   p/t/n
					int p = 0, t = 0, n = 0;
					if (sscanf_s(fv.c_str(), "%d/%d/%d", &p, &t, &n) == 3) {
						idx.push_back(resolveIdx(p, positions.size())); nIdx.push_back(resolveIdx(n, normals.size()));
					} else if (sscanf_s(fv.c_str(), "%d//%d", &p, &n) == 2) {
						idx.push_back(resolveIdx(p, positions.size())); nIdx.push_back(resolveIdx(n, normals.size()));
					} else if (sscanf_s(fv.c_str(), "%d/%d", &p, &t) == 2) {
						idx.push_back(resolveIdx(p, positions.size())); nIdx.push_back(-1);
					} else if (sscanf_s(fv.c_str(), "%d", &p) == 1) {
						idx.push_back(resolveIdx(p, positions.size())); nIdx.push_back(-1);
					}
				}
				auto cornerNormal = [&](int ni) -> float3 {
					if (ni >= 0 && ni < static_cast<int>(normals.size())) return normals[ni];
					return make_float3(0.0f, 1.0f, 0.0f);  // matches CPU mesh.h's fallback
				};
				for (size_t i = 1; i + 1 < idx.size(); ++i) {
					if (idx[0] < 0 || idx[0] >= static_cast<int>(positions.size()) ||
						idx[i] < 0 || idx[i] >= static_cast<int>(positions.size()) ||
						idx[i + 1] < 0 || idx[i + 1] >= static_cast<int>(positions.size()))
						continue;
					TriangleData t{};
					t.p0 = positions[idx[0]];
					t.p1 = positions[idx[i]];
					t.p2 = positions[idx[i + 1]];
					t.materialIdx = materialIdx;
					t.hasNormals = !normals.empty();
					if (t.hasNormals) {
						t.n0 = cornerNormal(nIdx[0]);
						t.n1 = cornerNormal(nIdx[i]);
						t.n2 = cornerNormal(nIdx[i + 1]);
					}
					scene.triangles.push_back(t);
				}
			}
		}
	}

	// Minimal Wavefront .mtl parser (GPU-side counterpart of CPU's
	// src/TheRestOfYourLife/mesh.h::parse_mtl()): maps material name ->
	// diffuse (Kd) color. Only Kd is read, matching this renderer's
	// no-texture mesh convention. Returns an empty map (never throws) if
	// the file can't be found.
	inline std::unordered_map<std::string, float3> parse_mtl_gpu(const std::string& filename) {
		std::unordered_map<std::string, float3> result;
		std::ifstream file(filename);
		if (!file.is_open()) {
			static const char* kSearchPrefixes[] = {
				"models/", "../models/", "../../models/",
				"../../../models/", "../../../../models/", "../../../../../models/"
			};
			for (const char* prefix : kSearchPrefixes) {
				file.clear();
				file.open(std::string(prefix) + filename);
				if (file.is_open()) break;
			}
		}
		if (!file.is_open()) return result;

		std::string line, current;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#') continue;
			std::istringstream ss(line);
			std::string tok;
			ss >> tok;
			if (tok == "newmtl") {
				ss >> current;
			} else if (tok == "Kd" && !current.empty()) {
				float r, g, b;
				ss >> r >> g >> b;
				result[current] = make_float3(r, g, b);
			}
		}
		return result;
	}

	// GPU counterpart of CPU's parse_mtl_textures(): maps material name ->
	// its map_Kd (diffuse texture) path, exactly as written in the .mtl.
	// Only materials with a map_Kd line appear in the result.
	inline std::unordered_map<std::string, std::string> parse_mtl_textures_gpu(const std::string& filename) {
		std::unordered_map<std::string, std::string> result;
		std::ifstream file(filename);
		if (!file.is_open()) {
			static const char* kSearchPrefixes[] = {
				"models/", "../models/", "../../models/",
				"../../../models/", "../../../../models/", "../../../../../models/"
			};
			for (const char* prefix : kSearchPrefixes) {
				file.clear();
				file.open(std::string(prefix) + filename);
				if (file.is_open()) break;
			}
		}
		if (!file.is_open()) return result;

		std::string line, current;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#') continue;
			std::istringstream ss(line);
			std::string tok;
			ss >> tok;
			if (tok == "newmtl") {
				ss >> current;
			} else if (tok == "map_Kd" && !current.empty()) {
				// Rest-of-line rather than a single ss >> token: real
				// texture filenames in these archives can contain literal
				// spaces (e.g. Bistro's "Metal_ RollDoor_01/..."), which a
				// single >> would silently truncate at. See CPU's
				// parse_mtl_textures() comment for the full rationale.
				std::string path;
				std::getline(ss, path);
				size_t start = path.find_first_not_of(" \t");
				if (start != std::string::npos) {
					size_t end = path.find_last_not_of(" \t\r");
					result[current] = path.substr(start, end - start + 1);
				}
			}
		}
		return result;
	}

	// GPU counterpart of CPU's resolve_mtl_texture_path(): normalizes
	// backslashes to forward slashes and strips leading "../"/"./"
	// segments, then joins onto textureDir. See CPU's own comment
	// (mesh.h) for why literal relative-path resolution isn't used.
	inline std::string resolve_mtl_texture_path_gpu(const std::string& relativePath, const std::string& textureDir) {
		std::string p = relativePath;
		for (auto& c : p) if (c == '\\') c = '/';

		size_t pos = 0;
		while (true) {
			if (p.compare(pos, 3, "../") == 0) pos += 3;
			else if (p.compare(pos, 2, "./") == 0) pos += 2;
			else break;
		}
		return textureDir + "/" + p.substr(pos);
	}

	// GPU counterpart of CPU's load_obj_mtl(): like load_obj_triangles_gpu()
	// above, but tracks mtllib/usemtl directives during face parsing and
	// resolves each face's material name to its own MaterialData (added to
	// scene.materials on first use, cached by name so faces sharing a
	// material share one materialIdx), instead of applying a single
	// materialIdx to every triangle. Falls back to fallbackMaterialIdx for
	// faces with no usemtl, an unknown name, or a missing/unreadable .mtl.
	// A standalone duplicate of load_obj_triangles_gpu() rather than a
	// shared helper, for the same reason CPU's load_obj_mtl() duplicates
	// load_obj(): ~50 existing single-material scenes call the original and
	// must not change behavior.
	//
	// textureDir: when non-null/non-empty, materials with a map_Kd entry
	// get a real image-texture-backed MaterialData (sampled via this
	// mesh's own "vt" UVs, interpolated in optix_intersection_triangle.h)
	// instead of a flat Kd color, resolved via resolve_mtl_texture_path_gpu()
	// against foundPrefix + textureDir -- a path *relative to the models/
	// folder itself* (e.g. "sponza_textures", not "models/sponza_textures"),
	// mirroring CPU's load_obj_mtl() exactly (see its own comment for why:
	// textureDir alone can't know how many ".." climbs the current working
	// directory needs to reach models/). Left null (the default), this
	// behaves exactly like the Kd-only version.
	inline void load_obj_triangles_mtl_gpu(SceneData& scene, const char* filename,
			int fallbackMaterialIdx, float scale, float3 offset, const char* textureDir = nullptr) {
		std::string foundPrefix;
		std::ifstream file(filename);
		if (!file.is_open()) {
			static const char* kSearchPrefixes[] = {
				"models/", "../models/", "../../models/",
				"../../../models/", "../../../../models/", "../../../../../models/"
			};
			for (const char* prefix : kSearchPrefixes) {
				file.clear();
				file.open(std::string(prefix) + filename);
				if (file.is_open()) { foundPrefix = prefix; break; }
			}
		}
		if (!file.is_open()) {
			std::cerr << "[OptiX] Could not load mesh '" << filename
					   << "' (tried a few relative paths) - scene will be missing this geometry.\n";
			return;
		}

		std::vector<float3> positions;
		std::vector<float3> normals;
		std::vector<float2> uvs;
		struct Face { int p[3]; int n[3]; int t[3]; std::string mtl; };
		std::vector<Face> faces;
		std::string mtllibName;
		std::string currentMtl;

		std::string line;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#') continue;
			std::istringstream ss(line);
			std::string tok;
			ss >> tok;
			if (tok == "v") {
				float x, y, z;
				ss >> x >> y >> z;
				positions.push_back(make_float3(x * scale + offset.x, y * scale + offset.y, z * scale + offset.z));
			} else if (tok == "vn") {
				float x, y, z;
				ss >> x >> y >> z;
				normals.push_back(normalize(make_float3(x, y, z)));
			} else if (tok == "vt") {
				float u, v;
				ss >> u >> v;
				uvs.push_back(make_float2(u, v));
			} else if (tok == "mtllib") {
				ss >> mtllibName;
			} else if (tok == "usemtl") {
				ss >> currentMtl;
			} else if (tok == "f") {
				std::vector<int> idx, nIdx, tIdx;
				std::string fv;
				auto resolveIdx = [](int raw, size_t countSoFar) -> int {
					return raw > 0 ? raw - 1 : static_cast<int>(countSoFar) + raw;
				};
				while (ss >> fv) {
					int p = 0, t = 0, n = 0;
					if (sscanf_s(fv.c_str(), "%d/%d/%d", &p, &t, &n) == 3) {
						idx.push_back(resolveIdx(p, positions.size())); tIdx.push_back(resolveIdx(t, uvs.size())); nIdx.push_back(resolveIdx(n, normals.size()));
					} else if (sscanf_s(fv.c_str(), "%d//%d", &p, &n) == 2) {
						idx.push_back(resolveIdx(p, positions.size())); tIdx.push_back(-1); nIdx.push_back(resolveIdx(n, normals.size()));
					} else if (sscanf_s(fv.c_str(), "%d/%d", &p, &t) == 2) {
						idx.push_back(resolveIdx(p, positions.size())); tIdx.push_back(resolveIdx(t, uvs.size())); nIdx.push_back(-1);
					} else if (sscanf_s(fv.c_str(), "%d", &p) == 1) {
						idx.push_back(resolveIdx(p, positions.size())); tIdx.push_back(-1); nIdx.push_back(-1);
					}
				}
				for (size_t i = 1; i + 1 < idx.size(); ++i) {
					if (idx[0] < 0 || idx[0] >= static_cast<int>(positions.size()) ||
						idx[i] < 0 || idx[i] >= static_cast<int>(positions.size()) ||
						idx[i + 1] < 0 || idx[i + 1] >= static_cast<int>(positions.size()))
						continue;
					Face f{};
					f.p[0] = idx[0]; f.p[1] = idx[i]; f.p[2] = idx[i + 1];
					f.n[0] = nIdx[0]; f.n[1] = nIdx[i]; f.n[2] = nIdx[i + 1];
					f.t[0] = tIdx[0]; f.t[1] = tIdx[i]; f.t[2] = tIdx[i + 1];
					f.mtl = currentMtl;
					faces.push_back(f);
				}
			}
		}

		// Locate the companion .mtl the same way CPU's load_obj_mtl() does:
		// prefer the file's own mtllib directive, fall back to
		// "<same name as the .obj>.mtl" if that's missing/empty/unreadable.
		std::unordered_map<std::string, float3> mtlColors;
		std::string mtlPathUsed = mtllibName;
		if (!mtllibName.empty())
			mtlColors = parse_mtl_gpu(mtllibName);
		if (mtlColors.empty()) {
			std::string name(filename);
			auto dot = name.find_last_of('.');
			mtlPathUsed = (dot == std::string::npos ? name : name.substr(0, dot)) + ".mtl";
			mtlColors = parse_mtl_gpu(mtlPathUsed);
		}
		std::unordered_map<std::string, std::string> mtlTextures;
		if (textureDir && textureDir[0] != '\0' && !mtlPathUsed.empty())
			mtlTextures = parse_mtl_textures_gpu(mtlPathUsed);

		auto cornerNormal = [&](int ni) -> float3 {
			if (ni >= 0 && ni < static_cast<int>(normals.size())) return normals[ni];
			return make_float3(0.0f, 1.0f, 0.0f);
		};
		auto cornerUV = [&](int ti) -> float2 {
			if (ti >= 0 && ti < static_cast<int>(uvs.size())) return uvs[ti];
			return make_float2(0.0f, 0.0f);
		};

		// One Lambertian MaterialData per unique .mtl name, added to
		// scene.materials on first use and cached by name so faces sharing
		// a material share one materialIdx: a real textureIdx (see
		// load_image_texture_gpu) when textureDir is set and the
		// material's map_Kd image loads successfully, else Kd-only, else
		// fallbackMaterialIdx when the name is empty/unknown or has
		// neither -- mirrors CPU's load_obj_mtl() exactly. Deliberately
		// does NOT require a Kd line to attempt the texture: a material
		// with only map_Kd (no Kd) is valid OBJ and must still resolve,
		// even though none of Sponza/Bistro/Rungholt's .mtl files actually
		// have one (every map_Kd material there also has a Kd line).
		std::unordered_map<std::string, int> matCache;
		for (const auto& f : faces) {
			int materialIdx = fallbackMaterialIdx;
			if (!f.mtl.empty()) {
				auto cached = matCache.find(f.mtl);
				if (cached != matCache.end()) {
					materialIdx = cached->second;
				} else {
					int resolvedIdx = -1;
					auto texIt = mtlTextures.find(f.mtl);
					if (texIt != mtlTextures.end()) {
						std::string imgPath = resolve_mtl_texture_path_gpu(texIt->second, foundPrefix + textureDir);
						int texIdx = load_image_texture_gpu(scene, imgPath.c_str());
						if (scene.textures[texIdx].width > 0) {
							auto colorForTexIt = mtlColors.find(f.mtl);
							float3 albedo = (colorForTexIt != mtlColors.end())
								? colorForTexIt->second : make_float3(1.0f, 1.0f, 1.0f);
							resolvedIdx = safe_cast_to_int(scene.materials.size());
							add_lambertian(scene, albedo);
							scene.materials.back().textureIdx = texIdx;
						}
					}
					if (resolvedIdx < 0) {
						auto colorIt = mtlColors.find(f.mtl);
						if (colorIt != mtlColors.end()) {
							resolvedIdx = safe_cast_to_int(scene.materials.size());
							add_lambertian(scene, colorIt->second);
						}
					}
					materialIdx = (resolvedIdx >= 0) ? resolvedIdx : fallbackMaterialIdx;
					matCache[f.mtl] = materialIdx;
				}
			}
			TriangleData t{};
			t.p0 = positions[f.p[0]];
			t.p1 = positions[f.p[1]];
			t.p2 = positions[f.p[2]];
			t.materialIdx = materialIdx;
			t.hasNormals = !normals.empty();
			if (t.hasNormals) {
				t.n0 = cornerNormal(f.n[0]);
				t.n1 = cornerNormal(f.n[1]);
				t.n2 = cornerNormal(f.n[2]);
			}
			t.hasUVs = !uvs.empty();
			if (t.hasUVs) {
				t.uv0 = cornerUV(f.t[0]);
				t.uv1 = cornerUV(f.t[1]);
				t.uv2 = cornerUV(f.t[2]);
			}
			scene.triangles.push_back(t);
		}
	}
}

/// @brief Build the Cornell Box scene with box primitive
/// @param scene Output scene data container
static void build_cornell_box(SceneData& scene) {
	using namespace cornell_box_data;

	// Walls + lights: one material and one QuadData per kQuads entry, built
	// directly from the shared table so CPU and GPU can't disagree on count,
	// position, color, or which quads are lights - see cornell_box_data.h.
	for (const auto& q : kQuads) {
		const int mat = safe_cast_to_int(scene.materials.size());
		if (q.is_light) {
			add_diffuse_light(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		} else {
			add_lambertian(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		}

		QuadData quad{};
		quad.Q = make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z));
		quad.u = make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z));
		quad.v = make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z));
		const float3 quad_cross = cross(quad.u, quad.v);
		quad.w = quad_cross;
		quad.normal = normalize(quad_cross);
		quad.D = dot(quad.normal, quad.Q);
		quad.materialIdx = mat;
		scene.quads.push_back(quad);
		if (q.is_light) {
			scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
			scene.lightKinds.push_back(GpuLightKind::Quad);
		}
	}

	// Glass sphere
	const int mat_glass = add_dielectric(scene, static_cast<float>(kGlassSphere.glass_ior));
	SphereData glass_sphere{};
	glass_sphere.center = make_float3(
		static_cast<float>(kGlassSphere.center.x), static_cast<float>(kGlassSphere.center.y), static_cast<float>(kGlassSphere.center.z));
	glass_sphere.radius = static_cast<float>(kGlassSphere.radius);
	glass_sphere.materialIdx = mat_glass;
	scene.spheres.push_back(glass_sphere);
	// Glass is never emissive - no lightIndices entry.

	// White rotated box
	const int mat_box = add_lambertian(scene, make_float3(static_cast<float>(kBox.color.r), static_cast<float>(kBox.color.g), static_cast<float>(kBox.color.b)));
	add_box(scene,
		make_float3(static_cast<float>(kBox.corner_min.x), static_cast<float>(kBox.corner_min.y), static_cast<float>(kBox.corner_min.z)),
		make_float3(static_cast<float>(kBox.corner_max.x), static_cast<float>(kBox.corner_max.y), static_cast<float>(kBox.corner_max.z)),
		mat_box,
		static_cast<float>(kBox.rotate_y_degrees),
		make_float3(static_cast<float>(kBox.translate.x), static_cast<float>(kBox.translate.y), static_cast<float>(kBox.translate.z)));
}

/// @brief Build Rough Metal Spheres scene (scene 9)
/// Matches CPU build_rough_metal_spheres(): large ground sphere, 5 rough-metal
/// spheres with roughness 0.05..0.8, and a large quad area light above.
static void build_rough_metal_spheres(SceneData& scene) {
    // Ground (large dark-grey Lambertian sphere)
    const int mat_ground = add_lambertian(scene, make_float3(0.2f, 0.2f, 0.2f));

    // Area light quad material
    constexpr float kRMSLightIntensity = 6.0f;
    const int mat_light = safe_cast_to_int(scene.materials.size());
    add_diffuse_light(scene, make_float3(kRMSLightIntensity, kRMSLightIntensity, kRMSLightIntensity));

    // Five rough-metal sphere materials: roughness 0.05, 0.2, 0.4, 0.6, 0.8
    const float roughnesses[5] = { 0.05f, 0.2f, 0.4f, 0.6f, 0.8f };
    int mat_metal[5];
    for (int i = 0; i < 5; ++i) {
        mat_metal[i] = safe_cast_to_int(scene.materials.size());
        add_metal(scene, make_float3(0.95f, 0.85f, 0.55f), roughnesses[i]);
    }

    // Ground sphere: center (0,-1000,0), radius 1000
    SphereData ground{};
    ground.center = make_float3(0.0f, -1000.0f, 0.0f);
    ground.radius = 1000.0f;
    ground.materialIdx = mat_ground;
    scene.spheres.push_back(ground);

    // Five metal spheres at x = (i-2)*2.5, y=1, z=0, radius=1
    for (int i = 0; i < 5; ++i) {
        SphereData s{};
        s.center = make_float3((i - 2) * 2.5f, 1.0f, 0.0f);
        s.radius = 1.0f;
        s.materialIdx = mat_metal[i];
        scene.spheres.push_back(s);
    }

    // Area light quad: Q=(-6,6,-4), u=(12,0,0), v=(0,0,8)
    QuadData lq{};
    lq.Q = make_float3(-6.0f, 6.0f, -4.0f);
    lq.u = make_float3(12.0f, 0.0f, 0.0f);
    lq.v = make_float3(0.0f, 0.0f, 8.0f);
    const float3 lc = cross(lq.u, lq.v);
    lq.w = lc;
    lq.normal = normalize(lc);
    lq.D = dot(lq.normal, lq.Q);
    lq.materialIdx = mat_light;
    scene.quads.push_back(lq);
    scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
    scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Adds the 5 standard Cornell-box walls (red/white/green/white/white)
/// and the main ceiling light - all of cornell_box_data::kQuads - to scene.
/// Shared by every "Cornell family" GPU builder (scenes 10-13, 15-17) that
/// keeps the standard box shell but swaps in different sphere/box
/// materials, mirroring CPU's add_cornell_walls_and_main_light() in
/// scenes_book.h.
static void add_cornell_walls_and_main_light(SceneData& scene) {
    using namespace cornell_box_data;
    for (int i = 0; i < 6; ++i) {
        const QuadSpec& q = kQuads[i];
        const int mat = safe_cast_to_int(scene.materials.size());
        if (q.is_light) {
            add_diffuse_light(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
        } else {
            add_lambertian(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
        }

        QuadData quad{};
        quad.Q = make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z));
        quad.u = make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z));
        quad.v = make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z));
        const float3 quad_cross = cross(quad.u, quad.v);
        quad.w = quad_cross;
        quad.normal = normalize(quad_cross);
        quad.D = dot(quad.normal, quad.Q);
        quad.materialIdx = mat;
        scene.quads.push_back(quad);
        if (q.is_light) {
            scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
            scene.lightKinds.push_back(GpuLightKind::Quad);
        }
    }
}

/// @brief Build Cornell Rough Metal scene (scene 10)
/// Matches CPU build_cornell_rough_metal(): same walls/light, rough aluminum box + rough gold sphere
static void build_cornell_rough_metal(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Rough aluminum box material (roughness 0.15)
    const int mat_alum = add_metal(scene, make_float3(0.8f, 0.85f, 0.88f), 0.15f);

    // Rough gold sphere material (roughness 0.3)
    const int mat_gold = add_metal(scene, make_float3(0.95f, 0.78f, 0.28f), 0.3f);

    // Rough gold sphere (center 190, 90, 190), radius 90
    SphereData sphere{};
    sphere.center = make_float3(190.0f, 90.0f, 190.0f);
    sphere.radius = 90.0f;
    sphere.materialIdx = mat_gold;
    scene.spheres.push_back(sphere);

    // Rough aluminum box: box(0,0,0 -> 165,330,165), rotated 15 deg, translated (265,0,295)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_alum,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Conductor scene (scene 12)
/// Matches CPU build_cornell_conductor(): Cornell box with a gold sphere and
/// aluminium box using GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF).
static void build_cornell_conductor(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Gold sphere material (conductor, roughness 0.1 -- polished gold)
    const int mat_gold = add_conductor(scene,
        make_float3(kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b),
        make_float3(kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b),
        0.1f);   // roughness; alpha = sqrt(0.1)

    // Aluminium box material (conductor, roughness 0.05 -- polished aluminium)
    const int mat_alum = add_conductor(scene,
        make_float3(kConductorAl.eta_r, kConductorAl.eta_g, kConductorAl.eta_b),
        make_float3(kConductorAl.k_r,   kConductorAl.k_g,   kConductorAl.k_b),
        0.05f);

    // Gold sphere
    SphereData sphere{};
    sphere.center = make_float3(190.0f, 90.0f, 190.0f);
    sphere.radius = 90.0f;
    sphere.materialIdx = mat_gold;
    scene.spheres.push_back(sphere);

    // Polished aluminium box
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_alum,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Coated Diffuse scene (scene 13)
/// Matches CPU build_cornell_coated_diffuse(): Cornell box with a blue coated sphere
/// and a red coated box (rough dielectric coat over Lambertian base, pbrt-v4 CoatedDiffuseBxDF)
static void build_cornell_coated_diffuse(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Blue coated-diffuse sphere (IOR 1.5, roughness 0.1)
    const int mat_coated_blue = add_coated_diffuse(scene,
        make_float3(0.2f, 0.3f, 0.9f),  // diffuse base colour
        0.1f,                            // coat roughness (RoughnessToAlpha done in shader)
        1.5f);                           // coat IOR (glass-like)

    // Orange/terracotta coated-diffuse box (IOR 1.5, roughness 0.2) - was
    // near-identical red to the wall behind it (matches CPU's fix, see
    // build_cornell_coated_diffuse()'s comment there).
    const int mat_coated_red = add_coated_diffuse(scene, make_float3(0.75f, 0.35f, 0.1f), 0.2f, 1.5f);

    // Blue coated-diffuse sphere
    SphereData sph{};
    sph.center    = make_float3(190.0f, 90.0f, 190.0f);
    sph.radius    = 90.0f;
    sph.materialIdx = mat_coated_blue;
    scene.spheres.push_back(sph);

    // Red coated-diffuse box
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_coated_red,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Thin Glass scene (scene 14)
/// Matches CPU build_cornell_thin_glass(): Cornell box with a vertical thin-glass panel
/// using pbrt-v4 ThinDielectricBxDF (analytic multi-bounce Fresnel, no bending).
static void build_cornell_thin_glass(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    add_lambertian(scene, make_float3(0.65f, 0.05f, 0.05f));

    const int mat_white = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));

    const int mat_green = add_lambertian(scene, make_float3(0.12f, 0.45f, 0.15f));

    const int mat_light = safe_cast_to_int(scene.materials.size());
    add_diffuse_light(scene, make_float3(kLightIntensity, kLightIntensity, kLightIntensity));

    // White diffuse box
    const int mat_box = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));

    // Thin-glass panel (IOR 1.5)
    const int mat_panel = add_thin_dielectric(scene, 1.5f);

    // Cornell Box walls
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,0); q.u = make_float3(0,0,kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(0,0,-kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(213,554,227); q.u = make_float3(130,0,0); q.v = make_float3(0,0,105); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1); scene.lightKinds.push_back(GpuLightKind::Quad); }
    { QuadData q{}; q.Q = make_float3(0,kBoxSize,0); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,-kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,kBoxSize); q.u = make_float3(-kBoxSize,0,0); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // White diffuse box (right side)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_box,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));

    // Thin-glass panel: rotated ~28 degrees off the camera's straight-on
    // view axis so it's actually visible (matches CPU build_cornell_thin_glass()
    // - see that function's comment: at 0 degrees incidence, IOR-1.5 Fresnel
    // reflectance is only ~4%, imperceptible). Built centered at local
    // origin, rotated about Y (same convention as CPU's rotate_y: x'=cos*x
    // + sin*z, z'=-sin*x + cos*z, y untouched), then translated into place.
    {
        const float panelAngleRad = 62.0f * 3.14159265358979323846f / 180.0f;  // matches CPU's 62-degree tilt
        const float cosA = cosf(panelAngleRad), sinA = sinf(panelAngleRad);
        auto rotate_y_pt = [&](float x, float y, float z) {
            return make_float3(cosA * x + sinA * z, y, -sinA * x + cosA * z);
        };
        const float3 Q_rot = rotate_y_pt(-177.5f, -277.5f, 0.0f);
        const float3 u_rot = rotate_y_pt(0.0f, 555.0f, 0.0f);  // vertical edge, unchanged by Y rotation
        const float3 v_rot = rotate_y_pt(355.0f, 0.0f, 0.0f);
        const float3 translate = make_float3(277.5f, 277.5f, 200.0f);

        QuadData q{};
        q.Q = make_float3(Q_rot.x + translate.x, Q_rot.y + translate.y, Q_rot.z + translate.z);
        q.u = u_rot;
        q.v = v_rot;
        const float3 c = cross(q.u, q.v);
        q.w      = c;
        q.normal = normalize(c);
        q.D      = dot(q.normal, q.Q);
        q.materialIdx = mat_panel;
        scene.quads.push_back(q);
    }
}

/// @brief Build Cornell Coated Conductor scene (scene 15)
/// Matches CPU build_cornell_coated_conductor(): Cornell box with a lacquered-gold sphere
/// and a lacquered-copper box using pbrt-v4 CoatedConductorBxDF.
/// coat: IOR=1.5, roughness=0.1/0.2; conductor: Au sphere, Cu box.
static void build_cornell_coated_conductor(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Lacquered-gold sphere (Au conductor, IOR-1.5 coat, roughness 0.1)
    const int mat_gold_lacquer = add_coated_conductor(scene,
        make_float3(kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b),
        make_float3(kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b),
        0.1f,   // coat roughness
        1.5f);  // coat IOR

    // Lacquered-copper box (Cu conductor, IOR-1.5 coat, roughness 0.2)
    const int mat_copper_lacquer = add_coated_conductor(scene,
        make_float3(kConductorCu.eta_r, kConductorCu.eta_g, kConductorCu.eta_b),
        make_float3(kConductorCu.k_r,   kConductorCu.k_g,   kConductorCu.k_b),
        0.2f, 1.5f);

    // Lacquered-gold sphere
    SphereData sphere{};
    sphere.center = make_float3(190.0f, 90.0f, 190.0f);
    sphere.radius = 90.0f;
    sphere.materialIdx = mat_gold_lacquer;
    scene.spheres.push_back(sphere);

    // Lacquered-copper box
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_copper_lacquer,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// Matches CPU build_cornell_wax_slab(): Cornell box with a wax sphere (DiffuseTransmissionBxDF).
/// albedo = reflectance R, emission field = transmittance T.
static void build_cornell_wax_slab(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Wax sphere: albedo = R (reflectance), emission = T (transmittance)
    const int mat_wax = add_diffuse_transmission(scene,
        make_float3(0.6f, 0.5f, 0.3f),   // R: reflected diffuse color
        make_float3(0.8f, 0.6f, 0.3f));  // T: transmitted diffuse color

    // White diffuse box material (same albedo as the walls, own index)
    const int mat_box = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));

    // Wax sphere (left)
    SphereData wax_sphere{};
    wax_sphere.center    = make_float3(190.0f, 90.0f, 190.0f);
    wax_sphere.radius    = 90.0f;
    wax_sphere.materialIdx = mat_wax;
    scene.spheres.push_back(wax_sphere);

    // White diffuse box (right)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_box,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Crystal scene (scene 17)
/// Matches CPU build_cornell_crystal(): Cornell box with NormalizedFresnelBxDF crystal sphere.
/// ior stored in mat.ior; normalization constant c computed on GPU via FresnelMoment1.
static void build_cornell_crystal(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Crystal sphere: NormalizedFresnelBxDF, IOR 1.5 (glass/crystal).
    // albedo is unused (weight computed from Fresnel) -- add_normalized_fresnel()
    // leaves it default-zeroed like every other field this type doesn't use.
    const int mat_crystal = add_normalized_fresnel(scene, 1.5f);

    // White diffuse box material (same albedo as the walls, own index)
    const int mat_box = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));

    // Crystal sphere (left)
    SphereData crystal_sphere{};
    crystal_sphere.center    = make_float3(190.0f, 90.0f, 190.0f);
    crystal_sphere.radius    = 90.0f;
    crystal_sphere.materialIdx = mat_crystal;
    scene.spheres.push_back(crystal_sphere);

    // White diffuse box (right)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_box,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Rough Glass scene (scene 11)
/// Matches CPU build_cornell_rough_glass(): same walls/light, diffuse box + rough-glass sphere
static void build_cornell_rough_glass(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Rough glass sphere (roughness 0.2, IOR 1.5 -- frosted glass)
    const int mat_rough_glass = add_rough_dielectric(scene, 0.2f, kGlassIOR);

    // White diffuse box material (same albedo as the walls, own index)
    const int mat_box = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));

    // White diffuse box (right)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_box,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));

    // Rough glass sphere (left, same position as original glass sphere)
    SphereData rg_sphere{};
    rg_sphere.center    = make_float3(190.0f, 90.0f, 190.0f);
    rg_sphere.radius    = 90.0f;
    rg_sphere.materialIdx = mat_rough_glass;
    scene.spheres.push_back(rg_sphere);
}

/**
 * Build bouncing spheres scene (scene 1, "In One Weekend" final scene).
 * Structurally mirrors src/TheRestOfYourLife/scenes_book.h's
 * build_bouncing_spheres() - a checker-ground plane (approximated as flat
 * gray, matching this file's established "GPU has no checker/procedural
 * texture support" simplification - see build_checkered_spheres below and
 * GpuCameraParams::backgroundColor's comment), a grid of small random
 * spheres, and 3 large signature spheres (glass/diffuse/metal).
 *
 * The small diffuse spheres get real GPU motion blur: each one's center1
 * (see SphereData's doc comment) is set to its "bounced" end-of-shutter
 * position, exactly like the CPU's moving-sphere constructor
 * (sphere(center1, center2, radius, mat)). This is the one GPU scene that
 * exercises OptiXRenderer::buildScene()'s sceneHasMotion_ path.
 *
 * Uses a fixed-seed std::mt19937 rather than CPU's random_double() RNG, so
 * the exact sphere layout won't pixel-match the CPU render - consistent
 * with every other procedural GPU scene in this file (e.g. Perlin noise,
 * random material assignment), none of which reproduce the CPU's exact
 * random sequence either.
 */
void build_bouncing_spheres(SceneData& scene) {
	// Ground sphere - checker approximated as flat gray.
	{
		const int mat = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(0.5f, 0.5f, 0.5f));
		SphereData ground{};
		ground.center = make_float3(0.0f, -1000.0f, 0.0f);
		ground.center1 = ground.center;  // static
		ground.radius = 1000.0f;
		ground.materialIdx = mat;
		scene.spheres.push_back(ground);
	}

	std::mt19937 rng(42u);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			const float choose_mat = unit(rng);
			const float3 center = make_float3(
				a + 0.9f * unit(rng),
				0.2f,
				b + 0.9f * unit(rng)
			);

			const float3 d = make_float3(center.x - 4.0f, center.y - 0.2f, center.z - 0.0f);
			const float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
			if (dist <= 0.9f) continue;

			int mat_idx;
			float3 center1 = center;  // default: static

			if (choose_mat < 0.8f) {
				// Diffuse - moving sphere (the "bounce": hops straight up by
				// a random amount between t=0 and t=1, matching CPU's
				// `center2 = center + vec3(0, random_double(0,.5), 0)`).
				const float3 albedo = make_float3(
					unit(rng) * unit(rng),
					unit(rng) * unit(rng),
					unit(rng) * unit(rng)
				);
				mat_idx = safe_cast_to_int(scene.materials.size());
				add_lambertian(scene, albedo);
				center1 = make_float3(center.x, center.y + unit(rng) * 0.5f, center.z);
			} else if (choose_mat < 0.95f) {
				// Metal
				const float3 albedo = make_float3(0.5f + 0.5f * unit(rng), 0.5f + 0.5f * unit(rng), 0.5f + 0.5f * unit(rng));
				const float fuzz = 0.5f * unit(rng);
				mat_idx = safe_cast_to_int(scene.materials.size());
				add_metal(scene, albedo, fuzz);
			} else {
				// Glass
				mat_idx = safe_cast_to_int(scene.materials.size());
				add_dielectric(scene, 1.5f);
			}

			SphereData sph{};
			sph.center = center;
			sph.center1 = center1;
			sph.radius = 0.2f;
			sph.materialIdx = mat_idx;
			scene.spheres.push_back(sph);
		}
	}

	// Three large signature spheres - static.
	{
		const int mat1 = add_dielectric(scene, 1.5f);
		SphereData s1{};
		s1.center = make_float3(0.0f, 1.0f, 0.0f);
		s1.center1 = s1.center;
		s1.radius = 1.0f;
		s1.materialIdx = mat1;
		scene.spheres.push_back(s1);
	}
	{
		const int mat2 = add_lambertian(scene, make_float3(0.4f, 0.2f, 0.1f));
		SphereData s2{};
		s2.center = make_float3(-4.0f, 1.0f, 0.0f);
		s2.center1 = s2.center;
		s2.radius = 1.0f;
		s2.materialIdx = mat2;
		scene.spheres.push_back(s2);
	}
	{
		const int mat3 = add_metal(scene, make_float3(0.7f, 0.6f, 0.5f), 0.0f);
		SphereData s3{};
		s3.center = make_float3(4.0f, 1.0f, 0.0f);
		s3.center1 = s3.center;
		s3.radius = 1.0f;
		s3.materialIdx = mat3;
		scene.spheres.push_back(s3);
	}
}

/// @brief Build checkered spheres scene (scene 2). Matches CPU
/// build_checkered_spheres() (src/TheRestOfYourLife/scenes_book.h) exactly:
/// a single checker_texture(scale=0.32, color(.2,.3,.1), color(.9,.9,.9))
/// shared across both spheres - not two separately flat-colored spheres (an
/// earlier version of this function approximated the checker that way,
/// before this codebase had any GPU checker-texture support; see
/// add_checker_texture_gpu, added for scene 38's ground).
void build_checkered_spheres(SceneData& scene) {
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.32f,
		make_float3(0.2f, 0.3f, 0.1f), make_float3(0.9f, 0.9f, 0.9f));
	const int mat = safe_cast_to_int(scene.materials.size());
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);

	SphereData sphere1{};
	sphere1.center = make_float3(0.0f, -10.0f, 0.0f);
	sphere1.radius = 10.0f;
	sphere1.materialIdx = mat;
	scene.spheres.push_back(sphere1);

	SphereData sphere2{};
	sphere2.center = make_float3(0.0f, 10.0f, 0.0f);
	sphere2.radius = 10.0f;
	sphere2.materialIdx = mat;
	scene.spheres.push_back(sphere2);

	// Small accent spheres resting on the visible cap of the lower
	// "planet" - matches CPU build_checkered_spheres() exactly (see its
	// own comment for the reasoning).
	const int warmMat  = add_lambertian(scene, make_float3(0.55f, 0.15f, 0.10f));
	const int metalMat = add_metal(scene, make_float3(0.8f, 0.75f, 0.6f), 0.05f);
	const int glassMat = add_dielectric(scene, 1.5f);

	SphereData accent1{};
	accent1.center = make_float3(1.6f, 0.5f, 2.2f);
	accent1.radius = 0.9f;
	accent1.materialIdx = warmMat;
	scene.spheres.push_back(accent1);

	SphereData accent2{};
	accent2.center = make_float3(-1.4f, 0.45f, 1.6f);
	accent2.radius = 0.7f;
	accent2.materialIdx = metalMat;
	scene.spheres.push_back(accent2);

	SphereData accent3{};
	accent3.center = make_float3(0.1f, 0.15f, 3.0f);
	accent3.radius = 0.6f;
	accent3.materialIdx = glassMat;
	scene.spheres.push_back(accent3);
}

/// @brief Scene 3: Earth. Matches CPU build_earth() (src/TheRestOfYourLife/
/// scenes_book.h) exactly: a single radius-2 sphere at the origin with the
/// earthmap.jpg image texture (see load_image_texture_gpu's comment for the
/// solid-cyan fallback if that file can't be found - it now can, see
/// images/earthmap.jpg), no other geometry. Illuminated purely by the flat
/// sky background (CameraConfig bg=(0.70,0.80,1.00)), same as scenes 1/2/5.
static void build_earth_gpu(SceneData& scene) {
	const int earthTexIdx = load_image_texture_gpu(scene, "earthmap.jpg");
	const int mat = safe_cast_to_int(scene.materials.size());
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), earthTexIdx);
	SphereData s{};
	s.center = make_float3(0.0f, 0.0f, 0.0f);
	s.center1 = s.center;
	s.radius = 2.0f;
	s.materialIdx = mat;
	scene.spheres.push_back(s);

	// Small grey "moon" for scale/context - matches CPU build_earth() (see
	// its own comment).
	const int moonMat = add_lambertian(scene, make_float3(0.6f, 0.6f, 0.62f));
	SphereData moon{};
	moon.center = make_float3(2.0f, 1.3f, 0.5f);
	moon.center1 = moon.center;
	moon.radius = 0.35f;
	moon.materialIdx = moonMat;
	scene.spheres.push_back(moon);

	// Dim cool rim light behind the globe - matches CPU build_earth()'s
	// quad exactly (see build_earth_lights()).
	const int rimMat = safe_cast_to_int(scene.materials.size());
	add_diffuse_light(scene, make_float3(0.9f, 1.0f, 1.3f));
	QuadData rim{};
	rim.Q = make_float3(-4.0f, -2.5f, -6.0f);
	rim.u = make_float3(3.0f, 0.0f, 0.0f);
	rim.v = make_float3(0.0f, 5.0f, 0.0f);
	const float3 rc = cross(rim.u, rim.v);
	rim.w = rc;
	rim.normal = normalize(rc);
	rim.D = dot(rim.normal, rim.Q);
	rim.materialIdx = rimMat;
	scene.quads.push_back(rim);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Ground + Perlin-noise sphere pair shared by scenes 4 (Perlin
/// Spheres) and 6 (Simple Light) - both start from identical code in CPU
/// (scenes_book.h's build_perlin_spheres() and build_simple_light() both
/// begin with the exact same two spheres before scene 6 adds its lights).
/// One shared noise_texture(scale=4) material for both spheres, matching
/// CPU's single `pertext` object.
static void add_perlin_spheres_pair_gpu(SceneData& scene) {
	const int noiseTexIdx = add_noise_texture_gpu(scene, 4.0f);
	const int mat = safe_cast_to_int(scene.materials.size());
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), noiseTexIdx);

	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat;
	scene.spheres.push_back(ground);

	SphereData s{};
	s.center = make_float3(0.0f, 2.0f, 0.0f);
	s.radius = 2.0f;
	s.materialIdx = mat;
	scene.spheres.push_back(s);
}

/// @brief Scene 4: Perlin Spheres. Matches CPU build_perlin_spheres()
/// (scenes_book.h) exactly: the shared ground+main-sphere pair, plus 2
/// smaller marble companion spheres (noise scale 8, vs. the pair's 4) and a
/// warm key-light quad - this scene used to be lit only by flat sky
/// ambient with no directed light at all.
static void build_perlin_spheres_gpu(SceneData& scene) {
	add_perlin_spheres_pair_gpu(scene);

	const int noiseTex2Idx = add_noise_texture_gpu(scene, 8.0f);
	const int companionMat = safe_cast_to_int(scene.materials.size());
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), noiseTex2Idx);

	SphereData companion1{};
	companion1.center = make_float3(2.2f, 0.8f, 1.0f);
	companion1.radius = 0.8f;
	companion1.materialIdx = companionMat;
	scene.spheres.push_back(companion1);

	SphereData companion2{};
	companion2.center = make_float3(-1.8f, 0.6f, -1.2f);
	companion2.radius = 0.6f;
	companion2.materialIdx = companionMat;
	scene.spheres.push_back(companion2);

	const int keyMat = safe_cast_to_int(scene.materials.size());
	add_diffuse_light(scene, make_float3(8.0f, 6.0f, 3.0f));
	QuadData key{};
	key.Q = make_float3(-4.0f, 6.0f, -3.0f);
	key.u = make_float3(4.0f, 0.0f, 0.0f);
	key.v = make_float3(0.0f, 0.0f, 4.0f);
	const float3 kc = cross(key.u, key.v);
	key.w = kc;
	key.normal = normalize(kc);
	key.D = dot(key.normal, key.Q);
	key.materialIdx = keyMat;
	scene.quads.push_back(key);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Scene 6: Simple Light. Matches CPU build_simple_light()
/// (scenes_book.h) exactly: the same Perlin-sphere pair as scene 4, plus a
/// warm emissive sphere above and a cool emissive quad to the side (two
/// separate materials/colors, not one shared flat-white light, for
/// temperature contrast between them - see CPU's own comment).
static void build_simple_light_gpu(SceneData& scene) {
	add_perlin_spheres_pair_gpu(scene);

	const int warmMat = safe_cast_to_int(scene.materials.size());
	add_diffuse_light(scene, make_float3(6.0f, 3.0f, 1.0f));

	SphereData lightSphere{};
	lightSphere.center = make_float3(0.0f, 7.0f, 0.0f);
	lightSphere.radius = 2.0f;
	lightSphere.materialIdx = warmMat;
	scene.spheres.push_back(lightSphere);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);

	const int coolMat = safe_cast_to_int(scene.materials.size());
	add_diffuse_light(scene, make_float3(2.0f, 3.0f, 6.0f));

	QuadData lightQuad{};
	lightQuad.Q = make_float3(3.5f, 1.0f, -3.0f);
	lightQuad.u = make_float3(2.0f, 0.0f, 0.0f);
	lightQuad.v = make_float3(0.0f, 2.0f, 0.0f);
	const float3 lc = cross(lightQuad.u, lightQuad.v);
	lightQuad.w = lc;
	lightQuad.normal = normalize(lc);
	lightQuad.D = dot(lightQuad.normal, lightQuad.Q);
	lightQuad.materialIdx = coolMat;
	scene.quads.push_back(lightQuad);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/**
 * Build colored quads scene (scene 5)
 * Five colored quads arranged in 3D space
 */
void build_quads_scene(SceneData& scene) {
	// Helper lambda to add a quad with its material
	auto add_quad = [&scene](float Qx, float Qy, float Qz,
							  float ux, float uy, float uz,
							  float vx, float vy, float vz,
							  float r, float g, float b) {
		// Add material
		const int mat_idx = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(r, g, b));

		// Add quad
		QuadData quad{};
		quad.Q = make_float3(Qx, Qy, Qz);
		quad.u = make_float3(ux, uy, uz);
		quad.v = make_float3(vx, vy, vz);
		const float3 cross_prod = cross(quad.u, quad.v);
		quad.normal = normalize(cross_prod);
		quad.D = dot(quad.normal, quad.Q);
		quad.materialIdx = mat_idx;
		scene.quads.push_back(quad);

		// Track if emissive
		if (is_emissive(scene, mat_idx)) {
			scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
			scene.lightKinds.push_back(GpuLightKind::Quad);
		}
	};

	// Left red quad
	add_quad(-3.0f, -2.0f, 5.0f, 0.0f, 0.0f, -4.0f, 0.0f, 4.0f, 0.0f, 1.0f, 0.2f, 0.2f);

	// Back green quad
	add_quad(-2.0f, -2.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.2f, 1.0f, 0.2f);

	// Right blue quad
	add_quad(3.0f, -2.0f, 1.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f, 0.0f, 0.2f, 0.2f, 1.0f);

	// Upper orange quad
	add_quad(-2.0f, 3.0f, 1.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 1.0f, 0.5f, 0.0f);

	// Lower teal quad
	add_quad(-2.0f, -3.0f, 5.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, -4.0f, 0.2f, 0.8f, 0.8f);

	// A real light floating in the room, facing the camera - matches CPU
	// build_quads() exactly (see its own comment). The add_quad lambda
	// above only builds Lambertian materials, so this is added directly
	// rather than through it.
	const int lampMat = safe_cast_to_int(scene.materials.size());
	add_diffuse_light(scene, make_float3(7.0f, 7.0f, 6.5f));
	QuadData lamp{};
	lamp.Q = make_float3(-1.0f, 0.5f, 3.0f);
	lamp.u = make_float3(2.0f, 0.0f, 0.0f);
	lamp.v = make_float3(0.0f, 1.0f, 0.0f);
	const float3 lc = cross(lamp.u, lamp.v);
	lamp.w = lc;
	lamp.normal = normalize(lc);
	lamp.D = dot(lamp.normal, lamp.Q);
	lamp.materialIdx = lampMat;
	scene.quads.push_back(lamp);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Cornell box walls (no light quad) + two spheres, for scenes lit by
/// a punctual (point/spot/distant) light instead of an emissive quad.
/// Matches CPU src/TheRestOfYourLife/scenes_advanced.h cornell_walls_no_light()
/// exactly: same 5 walls, same two sphere positions/materials.
static void build_punctual_light_walls(SceneData& scene) {
	using namespace cornell_box_data;

	// The 5 standard walls (green/red/ceiling/floor/back), no light quad -
	// shares kQuads[0..4] with CPU's cornell_walls_no_light() so the two
	// can't drift apart, same pattern as scene 0's build_cornell_box().
	for (int i = 0; i < 5; ++i) {
		const QuadSpec& q = kQuads[i];
		const int mat = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		add_transformed_quad(scene,
			make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z)),
			make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z)),
			make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z)),
			mat);
	}

	// Sphere materials: white lambertian + blue-tinted fuzzy metal (matches CPU)
	const int mat_white_sphere = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));

	const int mat_metal_sphere = add_metal(scene, make_float3(0.8f, 0.8f, 0.9f), 0.1f);

	// Two spheres (matches CPU cornell_walls_no_light exactly)
	{ SphereData s{}; s.center = make_float3(190.0f, 90.0f, 190.0f); s.radius = 90.0f; s.materialIdx = mat_white_sphere; scene.spheres.push_back(s); }
	{ SphereData s{}; s.center = make_float3(370.0f, 120.0f, 380.0f); s.radius = 120.0f; s.materialIdx = mat_metal_sphere; scene.spheres.push_back(s); }
}

/// @brief Scene 25: Spotlight Cornell. Matches CPU build_spotlight_punct().
static void build_spotlight_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	constexpr float kPi = 3.14159265358979323846f;
	auto deg2rad = [](float d) { return d * kPi / 180.0f; };

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Spot;
	light.spot.pos_x = 278.0f; light.spot.pos_y = 548.0f; light.spot.pos_z = 278.0f;
	light.spot.dir_x = 0.0f;   light.spot.dir_y = -1.0f;  light.spot.dir_z = 0.0f;  // already unit
	light.spot.ir = 1.0f; light.spot.ig = 0.95f; light.spot.ib = 0.85f;
	light.spot.scale = 600000.0f;
	light.spot.cos_falloff_start = cosf(deg2rad(15.0f));
	light.spot.cos_falloff_end   = cosf(deg2rad(30.0f));
	scene.punctualLights.push_back(light);
}

/// @brief Scene 26: Distant Light Cornell. Matches CPU build_distant_light_punct().
static void build_distant_light_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	float3 dir = normalize(make_float3(-0.4f, -1.0f, -0.2f));

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Distant;
	light.distant.dir_x = dir.x; light.distant.dir_y = dir.y; light.distant.dir_z = dir.z;
	light.distant.ir = 1.0f; light.distant.ig = 0.98f; light.distant.ib = 0.92f;
	// No 1/r^2 falloff for a distant light - this scale directly IS the
	// incident irradiance, not a huge r^2-compensating number like the
	// point/spot/goniometric lights below (matches CPU's fix, see
	// build_distant_light_punct()'s comment).
	light.distant.scale = 14.0f;
	light.distant.scene_radius = 1000.0f;
	scene.punctualLights.push_back(light);
}

/// @brief Scene 27: Point Light Cornell. Matches CPU build_point_light_punct().
static void build_point_light_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Point;
	light.point.pos_x = 278.0f; light.point.pos_y = 540.0f; light.point.pos_z = 278.0f;
	light.point.ir = 1.0f; light.point.ig = 0.98f; light.point.ib = 0.90f;
	// Matches CPU's fix (see build_point_light_punct()'s comment) - was
	// ~8x too bright, blowing the room to near-white.
	light.point.scale = 600000.0f;
	scene.punctualLights.push_back(light);
}

/// @brief Scene 28: Goniometric Light Cornell. Matches CPU build_goniometric_punct().
static void build_goniometric_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Goniometric;
	GoniometricLightGPU& g = light.gonio;
	g.pos_x = 278.0f; g.pos_y = 520.0f; g.pos_z = 278.0f;
	// Identity rotation (matches CPU's id[9] = {1,0,0, 0,1,0, 0,0,1})
	g.world_to_light[0] = 1.0f; g.world_to_light[1] = 0.0f; g.world_to_light[2] = 0.0f;
	g.world_to_light[3] = 0.0f; g.world_to_light[4] = 1.0f; g.world_to_light[5] = 0.0f;
	g.world_to_light[6] = 0.0f; g.world_to_light[7] = 0.0f; g.world_to_light[8] = 1.0f;
	g.ir = 1.0f; g.ig = 0.9f; g.ib = 0.7f;
	// Matches CPU's fix (see build_goniometric_punct()'s comment) - was
	// blowing the room to near-white.
	g.scale = 600000.0f;
	// Same synthetic profile as CPU build_goniometric_punct(): bright toward
	// the bottom hemisphere (v > NV/2), dim toward the top.
	g.nu = 16; g.nv = 8;
	for (int v = 0; v < g.nv; ++v) {
		float t = (float)v / (float)g.nv;
		for (int u = 0; u < g.nu; ++u)
			g.image[v * g.nu + u] = 0.2f + 0.8f * t;
	}
	scene.punctualLights.push_back(light);
}

/// @brief Scene 29: Projection Light Cornell. Matches CPU build_projection_punct().
static void build_projection_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Projection;
	ProjectionLightGPU& pr = light.proj;
	pr.pos_x = 278.0f; pr.pos_y = 278.0f; pr.pos_z = -50.0f;
	// Identity rotation (matches CPU's wtl[9] = {1,0,0, 0,1,0, 0,0,1})
	pr.world_to_light[0] = 1.0f; pr.world_to_light[1] = 0.0f; pr.world_to_light[2] = 0.0f;
	pr.world_to_light[3] = 0.0f; pr.world_to_light[4] = 1.0f; pr.world_to_light[5] = 0.0f;
	pr.world_to_light[6] = 0.0f; pr.world_to_light[7] = 0.0f; pr.world_to_light[8] = 1.0f;
	pr.scale = 1000000.0f;
	pr.hither = 1e-3f;
	pr.nx = 8; pr.ny = 8;
	constexpr float kPi = 3.14159265358979323846f;
	const float fov_deg = 40.0f;
	// Screen bounds (mirrors ProjectionLight<T>::make, cameras.h aspect logic)
	const float aspect = (float)pr.nx / (float)pr.ny;
	if (aspect >= 1.0f) {
		pr.sb_xmin = -aspect; pr.sb_xmax = aspect;
		pr.sb_ymin = -1.0f;   pr.sb_ymax = 1.0f;
	} else {
		pr.sb_xmin = -1.0f;         pr.sb_xmax = 1.0f;
		pr.sb_ymin = -1.0f/aspect;  pr.sb_ymax = 1.0f/aspect;
	}
	// screenFromLight reduces to a single scalar - see ProjectionLightGPU's
	// comment in optix_types.h for why the full 4x4 matrix isn't needed.
	pr.inv_tan = 1.0f / tanf((kPi / 180.0f) * fov_deg / 2.0f);
	// Same 8x8 checkerboard slide as CPU build_projection_punct().
	for (int y = 0; y < pr.ny; ++y) {
		for (int x = 0; x < pr.nx; ++x) {
			float v = ((x + y) % 2 == 0) ? 1.0f : 0.05f;
			int idx = (y * pr.nx + x) * 3;
			pr.image_rgb[idx] = v; pr.image_rgb[idx + 1] = v; pr.image_rgb[idx + 2] = v;
		}
	}
	scene.punctualLights.push_back(light);
}

/// @brief Scene 22: Depth of Field. Matches CPU build_depth_of_field() in
/// spirit (ground + a row of spheres spanning near/far of the focus plane to
/// show defocus blur) - simplified to solid-color materials since GPU has no
/// checker/procedural texture support. No emissive geometry, matching CPU
/// exactly (build_depth_of_field() has none either) - both are lit purely by
/// a flat sky background color, set via backgroundColor in this scene's
/// build_scene() case below (matching CameraConfig's bg for scene 22) rather
/// than a synthetic light. An earlier version of this function added a
/// hand-placed overhead area-light quad instead, under the mistaken belief
/// that GPU had no background/miss-color mechanism (it does - see
/// optix_miss.h's __miss__ms() and scene 5/24's own use of this same field)
/// - that extra light was a real, NEE-sampled light CPU doesn't have,
/// silently breaking CpuGpuLightParityTest for this scene.
static void build_depth_of_field_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.5f, 0.5f, 0.5f));
	SphereData ground{};
	// Modest flat ground (not the usual radius-1000 "planet" sphere used
	// elsewhere in this file) - at this camera's close distance/narrow fov,
	// a radius-1000 ground's curvature toward the horizon caught the
	// overhead light at a bad grazing angle and blew out most of the frame.
	ground.center = make_float3(0.0f, -50.0f, 0.0f);
	ground.radius = 50.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Five spheres spanning z = -4..+4 around the lookat point (z=0, matching
	// CPU's focus_dist=9 from lookfrom z=9), alternating material, so the
	// defocus blur visibly increases toward the near/far ends.
	const float3 colors[5] = {
		make_float3(0.8f, 0.2f, 0.2f), make_float3(0.2f, 0.8f, 0.2f), make_float3(0.9f, 0.9f, 0.9f),
		make_float3(0.2f, 0.2f, 0.8f), make_float3(0.8f, 0.8f, 0.2f)
	};
	const MaterialType kinds[5] = {
		MaterialType::Lambertian, MaterialType::Metal, MaterialType::Dielectric,
		MaterialType::Metal, MaterialType::Lambertian
	};
	for (int i = 0; i < 5; ++i) {
		// Material kind varies per sphere (see kinds[] above), so this picks
		// the matching named factory per-iteration instead of one fixed call.
		int mat;
		switch (kinds[i]) {
			case MaterialType::Metal:      mat = add_metal(scene, colors[i], 0.05f); break;
			case MaterialType::Dielectric: mat = add_dielectric(scene, 1.5f); break;
			default:                       mat = add_lambertian(scene, colors[i]); break;
		}
		SphereData s{};
		// Smaller radius than a first attempt at this scene used: at only 5
		// world units from the camera (nearest sphere, z=+4) with a narrow
		// 20-degree vfov, radius-1 spheres subtended more than the whole
		// frame and blew out to a solid color filling the image.
		s.center = make_float3(0.0f, 0.5f, (i - 2) * 2.0f);
		s.radius = 0.5f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}
}

/// @brief Scene 32: Orthographic Camera. Matches CPU build_ortho_camera_scene()
/// in spirit (ground + a row of colored lambertian spheres) - simplified to
/// solid-color materials, same reasoning as scene 22 above. No emissive
/// geometry either, matching CPU - build_ortho_sky() is a flat-color
/// sky_light there, mirrored via backgroundColor in this scene's
/// build_scene() case below, same as scene 22 (see that scene's comment for
/// why - this scene had the identical spurious-overhead-light bug).
static void build_ortho_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.5f, 0.5f, 0.5f));
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const float3 colors[5] = {
		make_float3(0.8f, 0.2f, 0.2f), make_float3(0.8f, 0.6f, 0.2f), make_float3(0.2f, 0.8f, 0.3f),
		make_float3(0.2f, 0.4f, 0.9f), make_float3(0.7f, 0.2f, 0.8f)
	};
	for (int i = 0; i < 5; ++i) {
		const int mat = add_lambertian(scene, colors[i]);
		SphereData s{};
		s.center = make_float3((i - 2) * 2.5f, 1.0f, 0.0f);
		s.radius = 1.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}
}

/// @brief Scene 33: Spherical Camera. Matches CPU build_spherical_camera_scene()
/// in spirit (ground + a ring of colored spheres + one emissive sphere) -
/// self-illuminating, needs no extra light unlike scenes 22/32 above.
static void build_spherical_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.5f, 0.5f, 0.5f));
	SphereData ground{};
	// CPU's SphericalCamera uses no camera_to_world (identity), so the
	// camera sits exactly at world origin - keep the ground surface below
	// that (top at y=-2) rather than tangent to it, matching how the row of
	// spheres/light below are all placed comfortably above the camera.
	ground.center = make_float3(0.0f, -1002.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	constexpr float kPi = 3.14159265358979323846f;
	constexpr int kRingCount = 8;
	for (int i = 0; i < kRingCount; ++i) {
		float ang = (2.0f * kPi * i) / (float)kRingCount;
		float hue = (float)i / (float)kRingCount;
		float3 col = make_float3(0.5f + 0.5f * cosf(2.0f * kPi * hue),
								   0.5f + 0.5f * cosf(2.0f * kPi * (hue + 0.33f)),
								   0.5f + 0.5f * cosf(2.0f * kPi * (hue + 0.67f)));
		const int mat = add_lambertian(scene, col);
		SphereData s{};
		s.center = make_float3(4.0f * cosf(ang), 1.0f, 4.0f * sinf(ang));
		s.radius = 0.8f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Central emissive sphere, matches CPU's central diffuse_light sphere.
	const int mat_light = safe_cast_to_int(scene.materials.size());
	constexpr float kSphericalLightIntensity = 8.0f;
	add_diffuse_light(scene, make_float3(kSphericalLightIntensity, kSphericalLightIntensity, kSphericalLightIntensity));
	SphereData lightSphere{};
	lightSphere.center = make_float3(0.0f, 3.0f, 0.0f);
	lightSphere.radius = 1.0f;
	lightSphere.materialIdx = mat_light;
	scene.spheres.push_back(lightSphere);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 36: Realistic Camera. Matches CPU build_realistic_camera_scene()
/// (ground + 5 colored spheres at increasing depth to show bokeh + one area
/// light) - ground uses a flat gray instead of CPU's checker_texture, matching
/// this file's established "no procedural textures on GPU" simplification
/// used elsewhere (e.g. build_triangle_mesh_scene_gpu).
static void build_realistic_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.5f, 0.5f, 0.5f));
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const float3 sphere_colors[5] = {
		make_float3(0.9f, 0.2f, 0.2f), make_float3(0.2f, 0.8f, 0.2f), make_float3(0.2f, 0.2f, 0.9f),
		make_float3(0.8f, 0.8f, 0.2f), make_float3(0.8f, 0.2f, 0.8f)
	};
	for (int i = 0; i < 5; ++i) {
		const float z = 2.0f + i * 1.5f;
		const int mat = add_lambertian(scene, sphere_colors[i]);
		SphereData s{};
		s.center = make_float3(0.0f, 1.0f, z);
		s.radius = 0.8f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData lightSphere{};
	lightSphere.center = make_float3(0.0f, 8.0f, 5.0f);
	lightSphere.radius = 2.0f;
	lightSphere.materialIdx = mat_light;
	scene.spheres.push_back(lightSphere);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 24: HDRI Sky. Matches CPU build_hdri_sky_world() (ground +
/// three spheres showcasing lambertian/metal/dielectric under sky light).
/// The CPU "HDRI" is actually a flat-color sky_light in practice (see
/// GpuCameraParams::backgroundColor's comment) - the caller sets that
/// separately, this only builds the ground+spheres geometry.
static void build_hdri_sky_world_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.4f, 0.4f, 0.4f));
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_lambertian = add_lambertian(scene, make_float3(0.7f, 0.3f, 0.2f));
	SphereData s1{};
	s1.center = make_float3(-3.0f, 1.0f, 0.0f);
	s1.radius = 1.0f;
	s1.materialIdx = mat_lambertian;
	scene.spheres.push_back(s1);

	const int mat_metal = add_metal(scene, make_float3(0.8f, 0.8f, 0.9f), 0.05f);
	SphereData s2{};
	s2.center = make_float3(0.0f, 1.0f, 0.0f);
	s2.radius = 1.0f;
	s2.materialIdx = mat_metal;
	scene.spheres.push_back(s2);

	const int mat_glass = add_dielectric(scene, 1.5f);
	SphereData s3{};
	s3.center = make_float3(3.0f, 1.0f, 0.0f);
	s3.radius = 1.0f;
	s3.materialIdx = mat_glass;
	scene.spheres.push_back(s3);
}

/// @brief Scene 35: Portal Infinite Light. Matches CPU build_portal_light_scene()
/// (5-wall room, no front wall - "portal" for the sky to enter - + one metal
/// sphere). Uses the same wall-quad layout as build_punctual_light_walls,
/// duplicated here rather than shared since materials/sphere content differ.
static void build_portal_light_scene_gpu(SceneData& scene) {
	const int mat_red = add_lambertian(scene, make_float3(0.65f, 0.05f, 0.05f));
	const int mat_white = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));
	const int mat_green = add_lambertian(scene, make_float3(0.12f, 0.45f, 0.15f));
	const int mat_metal_sphere = add_metal(scene, make_float3(0.8f, 0.8f, 0.9f), 0.05f);

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor

	// Back wall with an actual window cut into it (matches CPU
	// build_portal_light_scene() - see that function's comment) instead of
	// one solid quad, so this scene visually has something a "portal"
	// description can point at.
	add_transformed_quad(scene, make_float3(555, 400, 555), make_float3(-555, 0, 0), make_float3(0, 155, 0), mat_white); // top strip
	add_transformed_quad(scene, make_float3(555, 0, 555),   make_float3(-555, 0, 0), make_float3(0, 155, 0), mat_white); // bottom strip
	add_transformed_quad(scene, make_float3(555, 155, 555), make_float3(-155, 0, 0), make_float3(0, 245, 0), mat_white); // right-of-window strip
	add_transformed_quad(scene, make_float3(155, 155, 555), make_float3(-155, 0, 0), make_float3(0, 245, 0), mat_white); // left-of-window strip

	SphereData s{};
	s.center = make_float3(190.0f, 100.0f, 190.0f);
	s.radius = 100.0f;
	s.materialIdx = mat_metal_sphere;
	scene.spheres.push_back(s);
}

/// @brief Scene 7: Cornell Smoke. Matches CPU build_cornell_smoke() (full
/// Cornell box + two constant_medium boxes, density 0.01, black/white).
/// GPU approximates the two rotated boxes as spheres (MaterialType::Medium
/// is sphere-only - see its comment in optix_types.h) positioned at roughly
/// the same locations, since a box boundary would need a second, more
/// involved AABB-slab intersection path not worth the complexity here.
static void build_cornell_smoke_gpu(SceneData& scene) {
	using namespace cornell_box_data;

	// The 5 standard walls (green/red/ceiling/floor/back) - shares
	// cornell_box_data::kQuads[0..4] with CPU's build_cornell_smoke(). This
	// scene's own light is a different size/color than kQuads[5], so it's
	// added separately below rather than looping through index 5.
	for (int i = 0; i < 5; ++i) {
		const QuadSpec& q = kQuads[i];
		const int mat = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		add_transformed_quad(scene,
			make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z)),
			make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z)),
			make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z)),
			mat);
	}

	const int mat_light = add_diffuse_light(scene, make_float3(7.0f, 7.0f, 7.0f));
	{
		QuadData lq{};
		lq.Q = make_float3(113.0f, 554.0f, 127.0f);
		lq.u = make_float3(330.0f, 0.0f, 0.0f);
		lq.v = make_float3(0.0f, 0.0f, 305.0f);
		const float3 lc = cross(lq.u, lq.v);
		lq.w = lc;
		lq.normal = normalize(lc);
		lq.D = dot(lq.normal, lq.Q);
		lq.materialIdx = mat_light;
		scene.quads.push_back(lq);
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.lightKinds.push_back(GpuLightKind::Quad);
	}

	// Two medium spheres approximating CPU's two rotated boxes (centered
	// roughly where box1 [265,0,295]+165/2 and box2 [130,0,65]+82.5 sit).
	// Tinted (cool blue-grey / warm amber) instead of black/white - matches
	// CPU build_cornell_smoke() exactly.
	const int mat_medium_dark = add_medium(scene, make_float3(0.05f, 0.07f, 0.12f), 0.0f, 0.01f);
	const int mat_medium_white = add_medium(scene, make_float3(1.0f, 0.85f, 0.6f), 0.0f, 0.01f);

	SphereData m1{}; m1.center = make_float3(347.0f, 165.0f, 377.0f); m1.radius = 115.0f; m1.materialIdx = mat_medium_dark;
	scene.spheres.push_back(m1);
	SphereData m2{}; m2.center = make_float3(212.0f, 82.0f, 147.0f); m2.radius = 82.0f; m2.materialIdx = mat_medium_white;
	scene.spheres.push_back(m2);
}

/// @brief Scene 30: Homogeneous Medium. Matches CPU
/// build_homogeneous_medium_scene() exactly (full Cornell box + a single
/// medium sphere at the box's center, radius 400, density 0.005, HG g=0.3).
static void build_homogeneous_medium_scene_gpu(SceneData& scene) {
	const int mat_red = add_lambertian(scene, make_float3(0.65f, 0.05f, 0.05f));
	const int mat_white = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));
	const int mat_green = add_lambertian(scene, make_float3(0.12f, 0.45f, 0.15f));
	const int mat_light = add_diffuse_light(scene, make_float3(15.0f, 15.0f, 15.0f));

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back
	{
		QuadData lq{};
		lq.Q = make_float3(213.0f, 554.0f, 227.0f);
		lq.u = make_float3(130.0f, 0.0f, 0.0f);
		lq.v = make_float3(0.0f, 0.0f, 105.0f);
		const float3 lc = cross(lq.u, lq.v);
		lq.w = lc;
		lq.normal = normalize(lc);
		lq.D = dot(lq.normal, lq.Q);
		lq.materialIdx = mat_light;
		scene.quads.push_back(lq);
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.lightKinds.push_back(GpuLightKind::Quad);
	}

	// Radius shrunk from 400 (matches CPU's fix, see build_homogeneous_medium_scene's
	// comment) - GPU media only support sphere boundaries (see
	// optix_intersection_sphere.h), unlike CPU which can use an inset box, so this
	// stays a sphere but sized to stay safely inside the 277.5-unit center-to-wall
	// distance instead of poking through every wall (the r=400 case reached past
	// even the room's 480.6-unit corner-to-corner distance in the diagonal
	// direction). This leaves the room's corners visibly less foggy than CPU's
	// wall-to-wall box, an accepted CPU/GPU divergence matching build_cornell_smoke_gpu's
	// own box-approximated-as-spheres precedent above.
	const int mat_medium = add_medium(scene, make_float3(0.8f, 0.9f, 1.0f), 0.3f, 0.005f);
	SphereData fog{};
	fog.center = make_float3(277.5f, 277.5f, 277.5f);
	fog.radius = 270.0f;
	fog.materialIdx = mat_medium;
	scene.spheres.push_back(fog);
}

/// @brief Scene 20: Normal Mapped Cornell. Matches CPU
/// build_normal_mapped_cornell() (scenes_advanced.h). CPU's bump-mapped
/// back wall and rotated box (bump_map_material wrapping a noise_texture
/// displacement source) are a confirmed no-op on CPU itself:
/// bump_map_material::apply() samples the texture at 3 different (u,v) but
/// the SAME hit point p, and noise_texture::value() (texture.h:127-129)
/// ignores u/v entirely and depends only on p - so disp==disp_u==disp_v
/// bit-for-bit, the finite-difference gradient apply_bump_map() computes
/// is always exactly zero, and it returns the unperturbed geometric normal
/// unchanged. Verified empirically too, not just from reading the code: a
/// CPU render of this scene shows the back wall and box as flat white,
/// zero visible bump texture. So this GPU port renders them as plain flat
/// Lambertian white (same kBox/kQuads[] geometry and color already used by
/// the standard Cornell Box scene) - matching CPU's actual rendered pixels
/// exactly, rather than implementing bump-map device code that would never
/// be visually exercised by any current scene.
/// The sphere's normal_map_material IS non-degenerate (its normal source,
/// checker_texture, uses world-position p directly per-sample with no
/// finite-difference step, so it varies meaningfully across the sphere)
/// and is ported for real via MaterialType::NormalMappedLambertian - see
/// that type's comment in optix_types.h and its handling in
/// optix_intersection_sphere.h.
static void build_normal_mapped_cornell_gpu(SceneData& scene) {
	using namespace cornell_box_data;

	// 5 walls + the one light this scene actually has - kQuads[5]'s
	// position and color (15,15,15) match CPU's light exactly. Not
	// kQuads[6] (the secondary accent light), which this scene doesn't use.
	for (int i = 0; i < 6; ++i) {
		const QuadSpec& q = kQuads[i];
		const int mat = safe_cast_to_int(scene.materials.size());
		if (q.is_light) {
			add_diffuse_light(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		} else {
			add_lambertian(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		}
		// add_transformed_quad() already auto-registers emissive quads into
		// lightIndices/lightKinds itself (it checks the material type) -
		// do not also push here, or the one light double-counts.
		add_transformed_quad(scene,
			make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z)),
			make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z)),
			make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z)),
			mat);
	}

	// Rotated box: same kBox geometry/color as the standard Cornell Box
	// scene - matches CPU's (effectively-flat-white, see header comment
	// above) bumped_box exactly.
	const int mat_box = safe_cast_to_int(scene.materials.size());
	add_lambertian(scene, make_float3(static_cast<float>(kBox.color.r), static_cast<float>(kBox.color.g), static_cast<float>(kBox.color.b)));
	add_box(scene,
		make_float3(static_cast<float>(kBox.corner_min.x), static_cast<float>(kBox.corner_min.y), static_cast<float>(kBox.corner_min.z)),
		make_float3(static_cast<float>(kBox.corner_max.x), static_cast<float>(kBox.corner_max.y), static_cast<float>(kBox.corner_max.z)),
		mat_box,
		static_cast<float>(kBox.rotate_y_degrees),
		make_float3(static_cast<float>(kBox.translate.x), static_cast<float>(kBox.translate.y), static_cast<float>(kBox.translate.z)));

	// Normal-mapped sphere: matches CPU's normal_map_material(
	// checker_texture(8.0, (0.5,0.5,1.0), (0.8,0.8,1.0)),
	// lambertian(0.2,0.3,0.8)) exactly.
	const int checkerTexIdx = add_checker_texture_gpu(scene, 8.0f,
		make_float3(0.5f, 0.5f, 1.0f), make_float3(0.8f, 0.8f, 1.0f));
	const int mat_sphere = add_normal_mapped_lambertian(scene, make_float3(0.2f, 0.3f, 0.8f), checkerTexIdx);
	SphereData s{};
	s.center = make_float3(190.0f, 90.0f, 190.0f);
	s.radius = 90.0f;
	s.materialIdx = mat_sphere;
	scene.spheres.push_back(s);
}

/// @brief Scene 21: Subsurface Slab. Matches CPU build_subsurface_slab()
/// (scenes_advanced.h) - CPU's own header comment there is explicit this
/// isn't a real BSSRDF: it's the same "dielectric boundary + internal
/// constant_medium" trick as scene 8's fog spheres, which
/// MaterialType::DielectricMedium already implements exactly (see that
/// type's comment in optix_types.h) - no new device code needed. The jade
/// sphere maps onto it with zero approximation (it's already a sphere);
/// the wax slab's box boundary is approximated as a sphere sized to
/// roughly its footprint, same "box boundary would need a second AABB-slab
/// intersection path not worth the complexity" precedent as scene 7's own
/// two constant_medium boxes (see build_cornell_smoke_gpu's comment).
static void build_subsurface_slab_gpu(SceneData& scene) {
	using namespace cornell_box_data;

	// The 5 standard walls (green/red/ceiling/floor/back) - shares
	// cornell_box_data::kQuads[0..4] with CPU's build_subsurface_slab().
	// This scene's own light is a different color than kQuads[5], so it's
	// added separately below rather than looping through index 5.
	for (int i = 0; i < 5; ++i) {
		const QuadSpec& q = kQuads[i];
		const int mat = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)));
		add_transformed_quad(scene,
			make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z)),
			make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z)),
			make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z)),
			mat);
	}

	const int mat_light = add_diffuse_light(scene, make_float3(12.0f, 12.0f, 12.0f));
	{
		QuadData lq{};
		lq.Q = make_float3(213.0f, 554.0f, 227.0f);
		lq.u = make_float3(130.0f, 0.0f, 0.0f);
		lq.v = make_float3(0.0f, 0.0f, 105.0f);
		const float3 lc = cross(lq.u, lq.v);
		lq.w = lc;
		lq.normal = normalize(lc);
		lq.D = dot(lq.normal, lq.Q);
		lq.materialIdx = mat_light;
		scene.quads.push_back(lq);
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.lightKinds.push_back(GpuLightKind::Quad);
	}

	// Wax slab: CPU's box(0,0,0)-(200,300,160) translated by (270,0,230),
	// i.e. world-space [270,470]x[0,300]x[230,390] - approximated as a
	// sphere at the box's center with a radius chosen to roughly match its
	// footprint. ior=1.4/sigma_t=0.04/albedo=(0.98,0.96,0.90), matching
	// CPU's dielectric(1.4) + constant_medium(...,0.04,milky-white) exactly.
	{
		const int mat_slab = add_dielectric_medium(scene, make_float3(0.98f, 0.96f, 0.90f), 1.4f, 0.04f);
		SphereData s{};
		s.center = make_float3(370.0f, 150.0f, 310.0f);
		s.radius = 140.0f;
		s.materialIdx = mat_slab;
		scene.spheres.push_back(s);
	}

	// Jade sphere: CPU's sphere(160,90,160,r=90) - matches exactly, no
	// approximation needed (it's already a sphere). ior=1.5/sigma_t=0.06/
	// albedo=(0.1,0.5,0.2), matching CPU's dielectric(1.5) +
	// constant_medium(...,0.06,jade-green) exactly.
	{
		const int mat_jade = add_dielectric_medium(scene, make_float3(0.1f, 0.5f, 0.2f), 1.5f, 0.06f);
		SphereData s{};
		s.center = make_float3(160.0f, 90.0f, 160.0f);
		s.radius = 90.0f;
		s.materialIdx = mat_jade;
		scene.spheres.push_back(s);
	}
}

/// @brief Scene 31: Cloud Medium. Matches CPU build_cloud_medium_scene()
/// (ground + one medium sphere, density 0.8, HG g=0.05) - the CPU's Perlin-
/// noise density texture is dead code there too (constructed but never
/// actually used by the constant_medium call, which passes a constant
/// density), so this is a faithful, not simplified, port.
// Matches CPU build_cloud_medium_scene() exactly (scenes_advanced.h) - see
// that function's comment for the full reasoning (CloudMedium is a real
// heterogeneous, Perlin-FBm-density medium, not the old uniform-density
// constant_medium sphere this scene used to render as).
//
// GPU's medium handling (both Medium and now CloudMedium) is triggered by
// hitting a real SPHERE primitive - see optix_intersection_sphere.h's
// closest-hit program - so CloudMedium still needs *a* sphere to attach its
// materialIdx to, even though CloudMedium's own axis-aligned world AABB (not
// this sphere) is what actually bounds the medium: the sphere here is sized
// to comfortably contain that AABB (its half-diagonal, from the box's
// center) and only serves as the trigger geometry. The tight [tMin,tMax]
// used for delta tracking comes from CloudMedium::sample_ray()'s own AABB
// test against the ray, clipped to the sphere's own entry/exit - see
// optix_intersection_sphere.h's CloudMedium case.
static void build_cloud_medium_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.4f, 0.5f, 0.3f));
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// World AABB - matches CPU's cloud_min/cloud_max exactly.
	const float3 cloud_min = make_float3(-4.0f, 1.0f, -3.0f);
	const float3 cloud_max = make_float3(4.0f, 4.0f, 3.0f);
	const float sx = 1.0f / (cloud_max.x - cloud_min.x);
	const float sy = 1.0f / (cloud_max.y - cloud_min.y);
	const float sz = 1.0f / (cloud_max.z - cloud_min.z);
	const float world_to_medium_mat[9] = { sx,0,0,  0,sy,0,  0,0,sz };
	const float world_to_medium_translate[3] = {
		-cloud_min.x*sx, -cloud_min.y*sy, -cloud_min.z*sz
	};
	CloudMedium<float> cloud_medium = CloudMedium<float>::make(
		0.0f, 0.0f, 0.0f,   1.0f, 1.0f, 1.0f,   // medium-space bounds: unit cube
		world_to_medium_mat, world_to_medium_translate,
		0.0f,   // sigma_a: pure scattering (matches CPU)
		10.0f,  // sigma_s: matches CPU's build_cloud_medium_scene() - see that
		        // function's comment.
		0.3f,   // phase_g
		1.0f,   // density
		1.0f,   // wispiness
		4.0f    // frequency
	);
	const int mat_medium = add_cloud_medium(scene, cloud_medium, make_float3(1.0f, 1.0f, 1.0f));

	const float3 cloud_center = make_float3(
		0.5f*(cloud_min.x+cloud_max.x), 0.5f*(cloud_min.y+cloud_max.y), 0.5f*(cloud_min.z+cloud_max.z));
	const float3 half = make_float3(
		0.5f*(cloud_max.x-cloud_min.x), 0.5f*(cloud_max.y-cloud_min.y), 0.5f*(cloud_max.z-cloud_min.z));
	const float trigger_radius = sqrtf(half.x*half.x + half.y*half.y + half.z*half.z);
	SphereData cloud{};
	cloud.center = cloud_center;
	cloud.radius = trigger_radius;
	cloud.materialIdx = mat_medium;
	scene.spheres.push_back(cloud);

	// Background spheres for context - matches CPU exactly.
	const int mat_orange = add_lambertian(scene, make_float3(0.9f, 0.3f, 0.2f));
	SphereData s1{};
	s1.center = make_float3(-6.0f, 0.5f, 4.0f);
	s1.radius = 0.5f;
	s1.materialIdx = mat_orange;
	scene.spheres.push_back(s1);

	const int mat_metal = add_metal(scene, make_float3(0.8f, 0.8f, 0.9f), 0.05f);
	SphereData s2{};
	s2.center = make_float3(6.0f, 0.5f, 4.0f);
	s2.radius = 0.5f;
	s2.materialIdx = mat_metal;
	scene.spheres.push_back(s2);
}

/// @brief Scene E3: Dielectric Medium Showcase. Matches CPU
/// build_dielectric_medium_scene() - three glass spheres with colored
/// internal fog at varying density, using the already-wired single-material
/// MaterialType::DielectricMedium (add_dielectric_medium()) rather than
/// CPU's two-hittable dielectric+constant_medium pair, since the whole
/// point of that material type is to fuse the two into one.
static void build_dielectric_medium_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.4f, 0.5f, 0.3f));
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	struct fog_sphere { float x; float3 albedo; float sigma_t; };
	const fog_sphere spheres[3] = {
		{ -4.0f, make_float3(0.9f, 0.2f, 0.2f), 0.5f },  // thin red mist
		{  0.0f, make_float3(0.2f, 0.8f, 0.3f), 1.5f },  // medium green haze
		{  4.0f, make_float3(0.3f, 0.4f, 0.9f), 3.0f },  // dense blue fog
	};
	const float radius = 1.5f;
	for (const auto& s : spheres) {
		const int mat = add_dielectric_medium(scene, s.albedo, 1.5f, s.sigma_t);
		SphereData sp{};
		sp.center = make_float3(s.x, radius, 0.0f);
		sp.radius = radius;
		sp.materialIdx = mat;
		scene.spheres.push_back(sp);
	}
}

/// @brief Scene E4: RGB Grid Medium ("nebula"). Matches CPU
/// build_rgb_grid_medium_scene() exactly - same world AABB, same
/// generate_nebula_channel() calls (so both backends render identical voxel
/// data), same sigma_scale/phase_g. Uses the new MaterialType::RgbGridMedium
/// (add_rgb_grid_medium()) with a "trigger sphere" the same way CloudMedium
/// does (see build_cloud_medium_scene_gpu's comment) - the sphere's own
/// geometry only exists to get the ray into this branch at all; the medium's
/// real bounds are GpuRgbGridMedium::bounds_min/max, tested fresh in the
/// closesthit branch.
static void build_rgb_grid_medium_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.4f, 0.5f, 0.3f));
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int nx = 24, ny = 24, nz = 24;
	std::vector<float> ss_r, ss_g, ss_b;
	generate_nebula_channel<float>(nx, ny, nz, 3.0f, 0.0f,  0.0f, 0.0f, ss_r);
	generate_nebula_channel<float>(nx, ny, nz, 3.0f, 5.2f,  1.7f, 3.3f, ss_g);
	generate_nebula_channel<float>(nx, ny, nz, 3.0f, 11.4f, 8.8f, 2.1f, ss_b);

	float max_density = 0.0f;
	for (float v : ss_r) max_density = fmaxf(max_density, v);
	for (float v : ss_g) max_density = fmaxf(max_density, v);
	for (float v : ss_b) max_density = fmaxf(max_density, v);

	const float3 world_min = make_float3(-4.0f, 1.0f, -4.0f);
	const float3 world_max = make_float3(4.0f, 5.0f, 4.0f);
	const float sx = 1.0f / (world_max.x - world_min.x);
	const float sy = 1.0f / (world_max.y - world_min.y);
	const float sz = 1.0f / (world_max.z - world_min.z);
	const float sigma_scale = 4.0f;  // matches CPU's sigma_scale

	GpuRgbGridMedium meta{};
	meta.bounds_min[0] = world_min.x; meta.bounds_min[1] = world_min.y; meta.bounds_min[2] = world_min.z;
	meta.bounds_max[0] = world_max.x; meta.bounds_max[1] = world_max.y; meta.bounds_max[2] = world_max.z;
	meta.mat[0] = sx;   meta.mat[1] = 0.0f; meta.mat[2] = 0.0f;
	meta.mat[3] = 0.0f; meta.mat[4] = sy;   meta.mat[5] = 0.0f;
	meta.mat[6] = 0.0f; meta.mat[7] = 0.0f; meta.mat[8] = sz;
	meta.translate[0] = -world_min.x*sx;
	meta.translate[1] = -world_min.y*sy;
	meta.translate[2] = -world_min.z*sz;
	meta.nx = nx; meta.ny = ny; meta.nz = nz;
	meta.sigma_scale = sigma_scale;
	meta.sigma_maj = max_density * sigma_scale * 1.01f;  // small safety margin
	meta.phase_g = 0.2f;

	const int mat_medium = add_rgb_grid_medium(scene, meta, ss_r, ss_g, ss_b);

	const float3 center = make_float3(
		0.5f*(world_min.x+world_max.x), 0.5f*(world_min.y+world_max.y), 0.5f*(world_min.z+world_max.z));
	const float3 half = make_float3(
		0.5f*(world_max.x-world_min.x), 0.5f*(world_max.y-world_min.y), 0.5f*(world_max.z-world_min.z));
	const float trigger_radius = sqrtf(half.x*half.x + half.y*half.y + half.z*half.z);
	SphereData trigger{};
	trigger.center = center;
	trigger.radius = trigger_radius;
	trigger.materialIdx = mat_medium;
	scene.spheres.push_back(trigger);

	// Context spheres - matches CPU exactly.
	const int mat_orange = add_lambertian(scene, make_float3(0.9f, 0.3f, 0.2f));
	SphereData s1{};
	s1.center = make_float3(-6.0f, 0.5f, 4.0f);
	s1.radius = 0.5f;
	s1.materialIdx = mat_orange;
	scene.spheres.push_back(s1);

	const int mat_metal = add_metal(scene, make_float3(0.8f, 0.8f, 0.9f), 0.05f);
	SphereData s2{};
	s2.center = make_float3(6.0f, 0.5f, 4.0f);
	s2.radius = 0.5f;
	s2.materialIdx = mat_metal;
	scene.spheres.push_back(s2);
}

/// @brief Scene 23: Bilinear Patch Scene. Matches CPU build_bilinear_patch_scene()
/// exactly - standard Cornell box + two genuinely curved (non-planar) metal
/// bilinear patches (see optix_intersection_bilinear_patch.h). Unlike scene 7's
/// medium boxes, these are NOT approximated as another shape - bilinear
/// patches have their own GPU geometry type.
static void build_bilinear_patch_scene_gpu(SceneData& scene) {
	const int mat_red = add_lambertian(scene, make_float3(0.65f, 0.05f, 0.05f));
	const int mat_white = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));
	const int mat_green = add_lambertian(scene, make_float3(0.12f, 0.45f, 0.15f));
	const int mat_light = add_diffuse_light(scene, make_float3(15.0f, 15.0f, 15.0f));

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back
	{
		QuadData lq{};
		lq.Q = make_float3(213.0f, 554.0f, 227.0f);
		lq.u = make_float3(130.0f, 0.0f, 0.0f);
		lq.v = make_float3(0.0f, 0.0f, 105.0f);
		const float3 lc = cross(lq.u, lq.v);
		lq.w = lc;
		lq.normal = normalize(lc);
		lq.D = dot(lq.normal, lq.Q);
		lq.materialIdx = mat_light;
		scene.quads.push_back(lq);
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.lightKinds.push_back(GpuLightKind::Quad);
	}

	// Patch 1: classic hyperbolic paraboloid saddle, gold metal. Roughness
	// matches CPU's build_bilinear_patch_scene() - see that function's
	// comment for why (0.05 read as flat/mirror-like, hiding the curvature).
	const int mat_gold = add_metal(scene, make_float3(0.8f, 0.7f, 0.3f), 0.15f);
	{
		BilinearPatchData p{};
		p.p00 = make_float3(150.0f, 80.0f, 200.0f);
		p.p10 = make_float3(400.0f, 50.0f, 200.0f);
		p.p01 = make_float3(150.0f, 50.0f, 400.0f);
		p.p11 = make_float3(400.0f, 80.0f, 400.0f);
		p.materialIdx = mat_gold;
		scene.bilinearPatches.push_back(p);
	}

	// Patch 2: curved ramp (linear in u, curved in v), blue metal. Roughness
	// matches CPU, same reason as the gold patch above.
	const int mat_blue = add_metal(scene, make_float3(0.2f, 0.4f, 0.8f), 0.25f);
	{
		BilinearPatchData p{};
		p.p00 = make_float3(200.0f, 200.0f, 220.0f);
		p.p10 = make_float3(370.0f, 200.0f, 220.0f);
		p.p01 = make_float3(150.0f, 380.0f, 420.0f);
		p.p11 = make_float3(420.0f, 320.0f, 420.0f);
		p.materialIdx = mat_blue;
		scene.bilinearPatches.push_back(p);
	}
}

/// @brief Scene 19: Hair Fibers. Matches CPU build_hair_fibers() exactly - a
/// dark-floor ground sphere plus 5 spheres shaded with MaterialType::Hair
/// (Marschner/Chiang fiber scattering; no literal fiber geometry - see
/// MaterialType::Hair's comment in optix_types.h).
static void build_hair_fibers_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.05f, 0.05f, 0.06f));
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Hair MaterialData reuse: albedo=sigma_a(r,g,b), fuzz=beta_m, ior=eta(1.55),
	// eta_c.x=beta_n, eta_c.y=alpha_deg.
	struct HairSphere { float3 center; float3 sigma_a; float beta_m; float beta_n; float alpha_deg; };
	// Spacing widened (matches CPU build_hair_fibers() - see that function's
	// comment) so the 5 distinct hair colors read as 5 distinct spheres
	// instead of fusing into one shape.
	const HairSphere hairs[5] = {
		{ make_float3(-3.5f, 1.0f, 0.0f), make_float3(0.06f, 0.10f, 0.20f), 0.25f, 0.25f, 2.0f }, // dark brown
		{ make_float3(-1.2f, 1.0f, 0.4f), make_float3(0.01f, 0.015f, 0.03f), 0.30f, 0.30f, 2.0f }, // blonde
		{ make_float3(1.2f, 1.0f, -0.4f), make_float3(0.02f, 0.08f, 0.18f), 0.20f, 0.20f, 3.0f }, // auburn
		{ make_float3(3.5f, 1.0f, 0.0f), make_float3(0.001f, 0.001f, 0.002f), 0.45f, 0.45f, 1.0f }, // white/silver fur
		{ make_float3(0.0f, 1.0f, 2.3f), make_float3(0.50f, 0.55f, 0.60f), 0.15f, 0.15f, 2.0f }, // fine black fur
	};
	for (const auto& h : hairs) {
		// 1.55f: fiber eta, matches CPU hair_material's default
		const int mat_idx = add_hair(scene, h.sigma_a, h.beta_m, 1.55f, h.beta_n, h.alpha_deg);
		SphereData s{}; s.center = h.center; s.radius = 1.0f; s.materialIdx = mat_idx;
		scene.spheres.push_back(s);
	}

	// Overhead area light -- matches CPU build_hair_fibers()'s own light
	// (see that function's comment for the intensity-calibration rationale:
	// hair's peak BSDF response is far brighter than diffuse/glossy
	// surfaces, so this codebase's usual 6,6,6 light-quad intensity blew
	// the whole visible hemisphere to white under the ACES tone map).
	const int mat_light = add_diffuse_light(scene, make_float3(0.22f, 0.22f, 0.19f));
	QuadData lq{};
	lq.Q = make_float3(-5.0f, 6.0f, -5.0f);
	lq.u = make_float3(10.0f, 0.0f, 0.0f);
	lq.v = make_float3(0.0f, 0.0f, 7.0f);
	const float3 lc = cross(lq.u, lq.v);
	lq.w = lc;
	lq.normal = normalize(lc);
	lq.D = dot(lq.normal, lq.Q);
	lq.materialIdx = mat_light;
	scene.quads.push_back(lq);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Scene 18: Principled Showcase. Matches CPU build_principled_showcase()
/// exactly: 7 spheres sweeping the Disney/pbrt-v4 principled BSDF parameter
/// space (matte -> plastic -> semi-metallic -> fully metallic -> clearcoated
/// metal) over a checkered ground, using MaterialType::Principled - see that
/// type's comment in optix_types.h for how it reuses the shared CPU_GPU
/// PrincipledBxDF<T> struct directly instead of reimplementing the multi-lobe
/// math by hand.
static void build_principled_showcase_gpu(SceneData& scene) {
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.5f,
		make_float3(0.1f, 0.1f, 0.12f), make_float3(0.2f, 0.2f, 0.22f));
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	add_lambertian(scene, make_float3(0.0f, 0.0f, 0.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// x position, base color, metallic, roughness, clearcoat, clearcoat_rough
	// - matches CPU's 7 principled(...) calls exactly (ior=1.5 for all).
	// Spacing 2.0 (radius 1.0 each) so neighbors don't overlap/fuse.
	struct PrincipledSphere { float x; float3 base; float metallic; float roughness; float clearcoat; float clearcoat_rough; };
	const PrincipledSphere spheres[7] = {
		{ -6.0f, make_float3(0.8f, 0.1f, 0.1f),  0.0f, 0.9f,  0.0f, 0.1f  }, // 0: matte diffuse (red)
		{ -4.0f, make_float3(0.1f, 0.2f, 0.8f),  0.0f, 0.2f,  0.0f, 0.1f  }, // 1: plastic, low roughness (blue)
		{ -2.0f, make_float3(0.1f, 0.7f, 0.2f),  0.0f, 0.3f,  1.0f, 0.05f }, // 2: plastic, clearcoated (green)
		{  0.0f, make_float3(0.9f, 0.7f, 0.2f),  0.5f, 0.3f,  0.0f, 0.1f  }, // 3: semi-metallic (gold-tinted)
		{  2.0f, make_float3(0.8f, 0.45f, 0.2f), 0.8f, 0.4f,  0.0f, 0.1f  }, // 4: near-metallic, rough (copper-ish)
		{  4.0f, make_float3(0.9f, 0.9f, 0.9f),  1.0f, 0.05f, 0.0f, 0.1f  }, // 5: fully metallic, smooth (silver)
		{  6.0f, make_float3(0.9f, 0.7f, 0.1f),  1.0f, 0.1f,  1.0f, 0.08f }, // 6: fully metallic, clearcoated (lacquered gold)
	};
	for (const auto& p : spheres) {
		const int mat_idx = add_principled(scene, p.base, 1.5f, p.roughness, p.metallic, p.clearcoat, p.clearcoat_rough);
		SphereData s{}; s.center = make_float3(p.x, 1.0f, 0.0f); s.radius = 1.0f; s.materialIdx = mat_idx;
		scene.spheres.push_back(s);
	}

	// Overhead area light -- matches CPU build_principled_showcase()'s own
	// light (see that function's comment). Without it the clearcoat/metallic
	// spheres showed no specular highlight.
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	QuadData lq{};
	lq.Q = make_float3(-7.0f, 7.0f, -5.0f);
	lq.u = make_float3(14.0f, 0.0f, 0.0f);
	lq.v = make_float3(0.0f, 0.0f, 10.0f);
	const float3 lc = cross(lq.u, lq.v);
	lq.w = lc;
	lq.normal = normalize(lc);
	lq.D = dot(lq.normal, lq.Q);
	lq.materialIdx = mat_light;
	scene.quads.push_back(lq);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Scene 34: Measured BRDF. Matches CPU build_measured_brdf_scene()'s
/// ACTUAL rendered behavior, not its name: src/TheRestOfYourLife/scenes_advanced.h's
/// `measured_material::scatter()` never reads its MeasuredBRDFData member at
/// all (built from synthetic all-1.0 tabulated data, but the real pbrt-v4
/// MeasuredBxDF importance-sampling chain in src/shared/measured_bxdf.h is
/// never called) - it's byte-for-byte a Lambertian material with a flat tint,
/// cosine-hemisphere sampling and constant attenuation. GPU parity means
/// matching that actual behavior with MaterialType::Lambertian, not porting
/// the unused tensor-BRDF machinery.
static void build_measured_brdf_scene_gpu(SceneData& scene) {
	const int mat_ground = add_lambertian(scene, make_float3(0.3f, 0.3f, 0.3f));
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_measured = add_lambertian(scene, make_float3(0.7f, 0.5f, 0.3f));
	for (int i = -2; i <= 2; ++i) {
		SphereData s{}; s.center = make_float3(static_cast<float>(i) * 2.5f, 1.0f, 0.0f); s.radius = 1.0f; s.materialIdx = mat_measured;
		scene.spheres.push_back(s);
	}

	const int mat_light = add_diffuse_light(scene, make_float3(8.0f, 8.0f, 8.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 1.5f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 37: Triangle Mesh. Matches CPU build_triangle_mesh_scene()
/// exactly - same golden-ratio icosahedron vertex/face construction, same
/// gold metal material, same ground/light placement, and (via
/// add_checker_texture_gpu(), added for scene 38's ground) the same real
/// checker-textured ground rather than the flat-gray approximation this
/// scene used before that helper existed.
static void build_triangle_mesh_scene_gpu(SceneData& scene) {
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	const int mat_ground = add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Regular icosahedron: 12 vertices at golden-ratio coordinates, 20 faces.
	// Matches src/TheRestOfYourLife/scenes_advanced.h's build_triangle_mesh_scene() exactly.
	const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
	const float radius = 1.5f;
	const float3 raw_verts[12] = {
		make_float3(-1,  phi,  0), make_float3( 1,  phi,  0), make_float3(-1, -phi,  0), make_float3( 1, -phi,  0),
		make_float3( 0, -1,  phi), make_float3( 0,  1,  phi), make_float3( 0, -1, -phi), make_float3( 0,  1, -phi),
		make_float3( phi,  0, -1), make_float3( phi,  0,  1), make_float3(-phi,  0, -1), make_float3(-phi,  0,  1),
	};
	const float vert_len = length(raw_verts[0]);
	const float3 center = make_float3(0.0f, 2.5f, 0.0f);

	float3 verts[12];
	for (int i = 0; i < 12; ++i) {
		float3 v = raw_verts[i];
		verts[i] = center + (radius / vert_len) * v;
	}
	const int faces[20][3] = {
		{0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
		{1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
		{3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
		{4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1},
	};

	const int mat_mesh = add_metal(scene, make_float3(0.8f, 0.6f, 0.2f), 0.15f);
	for (const auto& f : faces) {
		TriangleData t{};
		t.p0 = verts[f[0]];
		t.p1 = verts[f[1]];
		t.p2 = verts[f[2]];
		t.materialIdx = mat_mesh;
		scene.triangles.push_back(t);
	}

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 72: Curve Fibers. Matches CPU build_curve_fibers_scene()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly in strand placement
/// (same 70-strand Fibonacci-disk arrangement, same deterministic hash01
/// pseudo-random, same windswept lean/taper), but NOT in intersection
/// method: the CPU side renders each strand as a real CurveShape with an
/// exact recursive-subdivision ray-curve test, while this GPU builder
/// tessellates each strand into a tapered tube of bilinear patches
/// (curve_tessellate.h) and feeds them into the SAME bilinear-patch GPU
/// pipeline scene F1 already uses - no new OptiX intersection program,
/// hit group, or GAS/SBT wiring needed. This mirrors pbrt-v4's own GPU
/// backend, which dices curves into bilinear patches for the identical
/// reason (see curve_tessellate.h's header comment): the real recursive
/// curve-intersection algorithm is a poor fit for the GPU. The tube reads
/// as slightly faceted up close compared to the CPU's perfectly smooth
/// curve - an expected, documented tessellation trade-off, not a bug.
static void build_curve_fibers_scene_gpu(SceneData& scene) {
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	const int mat_ground = add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Same 5 hair tones as build_curve_fibers_scene() (CPU) / build_hair_fibers()
	// (scene 19): dark brown, blonde, auburn, silver, black.
	const float3 palette[5] = {
		make_float3(0.25f, 0.14f, 0.06f),
		make_float3(0.80f, 0.65f, 0.35f),
		make_float3(0.45f, 0.13f, 0.05f),
		make_float3(0.75f, 0.75f, 0.78f),
		make_float3(0.03f, 0.03f, 0.03f),
	};
	int matIdx[5];
	for (int c = 0; c < 5; ++c) matIdx[c] = add_lambertian(scene, palette[c]);

	// Deterministic per-strand pseudo-random in [0,1) - identical hash to
	// build_curve_fibers_scene()'s own hash01 lambda, so both backends grow
	// the same 70 strands from the same roots/heights/leans.
	auto hash01 = [](int i, int salt) -> float {
		unsigned int h = static_cast<unsigned int>(i) * 374761393u
		                + static_cast<unsigned int>(salt) * 668265263u;
		h = (h ^ (h >> 13)) * 1274126177u;
		h ^= (h >> 16);
		return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
	};

	const int strand_count = 70;
	const float disk_radius = 1.4f;
	const float golden_angle = 2.399963229728653f;  // sunflower packing (~137.5 deg)
	const int n_length = 10, n_radial = 8;           // tube tessellation density

	std::vector<curve_tessellate::Quad> quads;
	for (int i = 0; i < strand_count; ++i) {
		float frac = (i + 0.5f) / strand_count;
		float r = disk_radius * std::sqrt(frac);
		float angle = i * golden_angle;
		float bx = r * std::cos(angle);
		float bz = r * std::sin(angle);

		float height = 0.9f + 0.5f * hash01(i, 1);
		float lean   = height * (0.35f + 0.35f * hash01(i, 2));

		float cp[4][3] = {
			{ bx,               0.0f,          bz },
			{ bx + 0.15f*lean,  height*0.33f,  bz },
			{ bx + 0.55f*lean,  height*0.70f,  bz },
			{ bx + lean,        height,        bz },
		};

		quads.clear();
		curve_tessellate::tessellate(cp, 0.0f, 1.0f, 0.045f, 0.006f, n_length, n_radial, quads);

		const int mat = matIdx[i % 5];
		for (const curve_tessellate::Quad& q : quads) {
			BilinearPatchData p{};
			p.p00 = make_float3(q.p00[0], q.p00[1], q.p00[2]);
			p.p10 = make_float3(q.p10[0], q.p10[1], q.p10[2]);
			p.p01 = make_float3(q.p01[0], q.p01[1], q.p01[2]);
			p.p11 = make_float3(q.p11[0], q.p11[1], q.p11[2]);
			p.materialIdx = mat;
			scene.bilinearPatches.push_back(p);
		}
	}

	// Overhead area light - matches CPU's quad(-2.5,4.0,-2.5)/(5,0,0)/(0,0,5).
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	QuadData lq{};
	lq.Q = make_float3(-2.5f, 4.0f, -2.5f);
	lq.u = make_float3(5.0f, 0.0f, 0.0f);
	lq.v = make_float3(0.0f, 0.0f, 5.0f);
	const float3 lc = cross(lq.u, lq.v);
	lq.w = lc;
	lq.normal = normalize(lc);
	lq.D = dot(lq.normal, lq.Q);
	lq.materialIdx = mat_light;
	scene.quads.push_back(lq);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Quad);
}

/// @brief Scene 38: Stanford Bunny. Matches CPU build_stanford_bunny()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered ground
/// (now a real GPU checker texture, see add_checker_texture_gpu - unlike
/// scene 37's flat-gray ground approximation, added before this codebase
/// had any GPU texture support at all), same bronze metal material, same
/// scale/offset/light placement. The bunny geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/stanford-bunny.obj CPU
/// loads - 69,451 real triangles, not a procedural stand-in like scene 37's
/// icosahedron, so this is the actual GPU exercise of load_obj_triangles_gpu
/// against a large external asset.
static void build_stanford_bunny_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Bunny mesh, in polished bronze - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_bunny() comment for the numbers).
	const int mat_bunny = add_metal(scene, make_float3(0.71f, 0.43f, 0.20f), 0.15f);
	load_obj_triangles_gpu(scene, "stanford-bunny.obj", mat_bunny,
		/*scale=*/19.4f, make_float3(0.3267f, -0.6398f, 0.0298f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 39: Stanford Armadillo. Matches CPU build_stanford_armadillo()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered ground,
/// same gunmetal metal material, same scale/offset/light placement. The
/// armadillo geometry itself is loaded via load_obj_triangles_gpu() from
/// the same models/armadillo.obj CPU loads - 99,976 real triangles, same
/// "positions only, no vn/vt" situation as scene 38's bunny (confirmed via
/// grep - zero `vn` lines in the source file).
static void build_stanford_armadillo_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Armadillo mesh, in gunmetal - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_armadillo() comment for the numbers).
	const int mat_armadillo = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.08f);
	load_obj_triangles_gpu(scene, "armadillo.obj", mat_armadillo,
		/*scale=*/0.0198f, make_float3(0.0f, 1.0736f, 0.0f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 40: Stanford Happy Buddha. Matches CPU
/// build_stanford_happy_buddha() (src/TheRestOfYourLife/scenes_advanced.h)
/// exactly: same checkered ground, same polished-gold metal material, same
/// scale/offset/light placement. The buddha geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/happy-buddha.obj CPU
/// loads - 98,601 real triangles, same "positions only, no vn/vt" situation
/// as scenes 38/39 (confirmed via grep - zero `vn` lines in the source
/// file).
static void build_stanford_happy_buddha_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Buddha mesh, in polished gold - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_happy_buddha() comment for the numbers).
	const int mat_buddha = add_metal(scene, make_float3(0.83f, 0.69f, 0.22f), 0.05f);
	load_obj_triangles_gpu(scene, "happy-buddha.obj", mat_buddha,
		/*scale=*/15.1496f, make_float3(0.0824f, -0.7539f, 0.1015f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 41: Stanford Lucy. Matches CPU build_stanford_lucy()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The Lucy geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/lucy.obj CPU loads -
/// 99,970 real triangles, same "positions only, no vn/vt" situation as
/// scenes 37-40 (confirmed via grep - zero `vn` lines in the source
/// file). That file was rotated to Y-up once, directly, before being
/// committed to this repo - see CPU's build_stanford_lucy() comment for
/// why (this mirror's lucy.obj ships Z-up) - so no rotation is needed
/// here either, same scale+translate-only transform as every other mesh
/// scene.
static void build_stanford_lucy_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Lucy mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_lucy() comment for the numbers).
	const int mat_lucy = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "lucy.obj", mat_lucy,
		/*scale=*/0.0018783f, make_float3(-1.2978f, 1.1380f, -0.2283f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 42: Stanford XYZRGB Dragon. Matches CPU build_stanford_dragon()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered ground,
/// same bright-silver metal material, same scale/offset/light placement.
/// The dragon geometry itself is loaded via load_obj_triangles_gpu() from
/// the same models/xyzrgb_dragon.obj CPU loads - 249,882 real triangles,
/// same "positions only, no vn/vt" situation as scenes 37-41. Unlike Lucy,
/// this source file is already Y-up (confirmed via its own bounding box),
/// so no pre-rotation was needed - same scale+translate-only transform as
/// every other mesh scene.
static void build_stanford_dragon_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Dragon mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_dragon() comment for the numbers).
	const int mat_dragon = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "xyzrgb_dragon.obj", mat_dragon,
		/*scale=*/0.0267772f, make_float3(-0.0600f, 1.6803f, -0.2640f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 50: Glass Dragon. Matches CPU build_glass_dragon()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same mesh/scale/
/// offset/light placement as scene 42's dragon, but MaterialType::Dielectric
/// (IOR 1.5, matches CPU's dielectric(1.5)) instead of Metal. See CPU
/// build_glass_dragon()'s own comment for why neither the regular path
/// tracer NOR --sppm render this scene's dragon surface itself cleanly
/// (a genuinely hard case, not a bug) - this GPU builder only targets the
/// regular GPU path tracer; GPU SPPM is Phase-1-scoped to scene 11 only.
static void build_glass_dragon_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Dragon mesh, in clear glass - matches CPU's exact scale/offset (same
	// numbers as scene 42's metal dragon, see build_stanford_dragon_gpu's
	// comment).
	const int mat_dragon = add_dielectric(scene, 1.5f);
	load_obj_triangles_gpu(scene, "xyzrgb_dragon.obj", mat_dragon,
		/*scale=*/0.0267772f, make_float3(-0.0600f, 1.6803f, -0.2640f));

	// Area light
	const int mat_light2 = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light2{}; light2.center = make_float3(0.0f, 8.0f, 0.0f); light2.radius = 2.0f; light2.materialIdx = mat_light2;
	scene.spheres.push_back(light2);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 51: Beast. Matches CPU build_beast() exactly.
static void build_beast_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_beast = add_metal(scene, make_float3(0.71f, 0.43f, 0.20f), 0.15f);
	load_obj_triangles_gpu(scene, "beast.obj", mat_beast,
		/*scale=*/0.0118956f, make_float3(0.0000f, 0.0103f, -0.3401f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 52: VW Beetle. Matches CPU build_beetle() exactly.
static void build_beetle_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_beetle = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.08f);
	load_obj_triangles_gpu(scene, "beetle.obj", mat_beetle,
		/*scale=*/9.9009901f, make_float3(0.3614f, -3.0297f, -1.9010f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 53: VW Beetle (alternate mesh). Matches CPU build_beetle_alt() exactly.
static void build_beetle_alt_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_beetle = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.12f);
	load_obj_triangles_gpu(scene, "beetle-alt.obj", mat_beetle,
		/*scale=*/8.8757396f, make_float3(0.0000f, 1.5000f, 0.0000f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 54: Bimba. Matches CPU build_bimba() exactly.
static void build_bimba_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_bimba = add_metal(scene, make_float3(0.83f, 0.69f, 0.22f), 0.05f);
	load_obj_triangles_gpu(scene, "bimba.obj", mat_bimba,
		/*scale=*/9.2592593f, make_float3(0.3333f, 2.2963f, 10.7454f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 55: Cow. Matches CPU build_cow() exactly.
static void build_cow_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_cow = add_metal(scene, make_float3(0.80f, 0.65f, 0.28f), 0.15f);
	load_obj_triangles_gpu(scene, "cow.obj", mat_cow,
		/*scale=*/0.4689698f, make_float3(-0.3639f, 1.7056f, 0.0000f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 56: Fandisk. Matches CPU build_fandisk() exactly.
static void build_fandisk_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_fandisk = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.1f);
	load_obj_triangles_gpu(scene, "fandisk.obj", mat_fandisk,
		/*scale=*/0.5719733f, make_float3(-1.3807f, -7.2097f, 0.7664f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 57: Homer. Matches CPU build_homer() exactly.
static void build_homer_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_homer = add_metal(scene, make_float3(0.85f, 0.70f, 0.25f), 0.1f);
	load_obj_triangles_gpu(scene, "homer.obj", mat_homer,
		/*scale=*/3.5671819f, make_float3(-1.7818f, -0.5565f, -1.7568f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 58: Igea. Matches CPU build_igea() exactly (unrotated mesh --
/// this scan's face points up rather than forward, compensated by the
/// registry's camera position, not a mesh transform -- see build_igea()'s
/// own comment).
static void build_igea_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_igea = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "igea.obj", mat_igea,
		/*scale=*/30.0000000f, make_float3(0.0000f, 1.5000f, 0.0000f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 59: Max Planck. Matches CPU build_max_planck() exactly.
static void build_max_planck_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_planck = add_metal(scene, make_float3(0.65f, 0.45f, 0.30f), 0.2f);
	load_obj_triangles_gpu(scene, "max-planck.obj", mat_planck,
		/*scale=*/0.0092252f, make_float3(-0.2823f, 1.6670f, -0.7592f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 60: Ogre. Matches CPU build_ogre() exactly.
static void build_ogre_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_ogre = add_metal(scene, make_float3(0.45f, 0.50f, 0.35f), 0.2f);
	load_obj_triangles_gpu(scene, "ogre.obj", mat_ogre,
		/*scale=*/0.0936388f, make_float3(0.0000f, 0.0007f, 0.0557f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 61: Rocker Arm. Matches CPU build_rocker_arm() exactly.
static void build_rocker_arm_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_rocker = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.1f);
	load_obj_triangles_gpu(scene, "rocker-arm.obj", mat_rocker,
		/*scale=*/5.8365759f, make_float3(0.0000f, 1.5000f, 0.0000f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 43: Utah Teapot. Matches CPU build_utah_teapot()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The teapot geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/teapot.obj CPU loads -
/// 6,320 real triangles, same "positions only, no vn/vt" situation as
/// scenes 37-42. Already Y-up and already sitting on y=0 (confirmed via
/// its own bounding box), so no rotation and no y-offset needed.
static void build_utah_teapot_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Teapot mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_utah_teapot() comment for the numbers).
	const int mat_teapot = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "teapot.obj", mat_teapot,
		/*scale=*/0.952381f, make_float3(-1.6352f, 0.0f, 0.0f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 44: Spot the Cow (Keenan Crane). Matches CPU build_spot_cow()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The cow geometry itself is loaded via load_obj_triangles_gpu()
/// from the same models/spot.obj CPU loads - 5,856 real triangles. Source
/// faces use "position/uv" indices (no vn), same "position index only"
/// situation load_obj_triangles_gpu() already handles for scenes 37-44's
/// other meshes. Confirmed Y-up (no rotation) via CPU's first render.
static void build_spot_cow_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Cow mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_spot_cow() comment for the numbers).
	const int mat_cow = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "spot.obj", mat_cow,
		/*scale=*/1.7747f, make_float3(0.0f, 1.3076f, -0.3373f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 45: Suzanne (Blender's monkey-head mascot). Matches CPU
/// build_suzanne() (src/TheRestOfYourLife/scenes_advanced.h) exactly: same
/// checkered ground, same bright-silver metal material, same scale/offset/
/// light placement. The mesh itself is loaded via load_obj_triangles_gpu()
/// from the same models/suzanne.obj CPU loads - mostly quad faces (468 of
/// 500), fan-triangulated by the loader into 968 real triangles, same
/// mechanism already exercised by every other mesh scene's occasional
/// n-gon. Confirmed Y-up (no rotation) via CPU's first render.
static void build_suzanne_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Suzanne mesh, in bright silver - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_suzanne() comment for the numbers).
	const int mat_suzanne = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "suzanne.obj", mat_suzanne,
		/*scale=*/1.52381f, make_float3(3.8005f, -0.4073f, -6.2536f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 46: Nefertiti Bust. Matches CPU build_nefertiti()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The bust geometry itself is loaded via load_obj_triangles_gpu()
/// from the same models/nefertiti.obj CPU loads - 99,938 real triangles,
/// no vn/vt data (flat-shaded, same as most other mesh scenes). That file
/// was rotated to Y-up once, directly, before being committed to this repo
/// - see CPU's build_nefertiti() comment for why (this mirror's
/// nefertiti.obj ships Z-up, same situation Lucy hit) - so no rotation is
/// needed here either, same scale+translate-only transform as every other
/// mesh scene.
static void build_nefertiti_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Nefertiti mesh, in bright silver - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_nefertiti() comment for the numbers).
	const int mat_nefertiti = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "nefertiti.obj", mat_nefertiti,
		/*scale=*/0.0060654f, make_float3(-0.0001f, 1.4998f, -0.0002f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 47: Horse (classic geometry-processing test model - a
/// chess-knight-style head+neck bust, not a full body). Matches CPU
/// build_horse() (src/TheRestOfYourLife/scenes_advanced.h) exactly: same
/// checkered ground, same bright-silver metal material, same scale/offset/
/// light placement. The mesh itself is loaded via load_obj_triangles_gpu()
/// from the same models/horse.obj CPU loads - 96,966 real triangles, no
/// vn/vt data (flat-shaded). Confirmed Y-up (no rotation) via CPU's first
/// render.
static void build_horse_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Horse mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_horse() comment for the numbers).
	const int mat_horse = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "horse.obj", mat_horse,
		/*scale=*/16.36295f, make_float3(0.0f, 1.5f, 0.0f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 48: Cheburashka (beloved cartoon-character bust from Keenan
/// Crane's geometry-processing course). Matches CPU build_cheburashka()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The mesh itself is loaded via load_obj_triangles_gpu() from
/// the same models/cheburashka.obj CPU loads - 13,334 real triangles, no
/// vn/vt data (flat-shaded). Confirmed Y-up (no rotation) via CPU's first
/// render.
static void build_cheburashka_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Cheburashka mesh, in bright silver - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_cheburashka() comment for the numbers).
	const int mat_cheb = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "cheburashka.obj", mat_cheb,
		/*scale=*/3.5648929f, make_float3(-1.7824465f, -0.2824465f, -1.7824465f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 49: Trophy Room. Matches CPU build_trophy_room()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same four meshes
/// (bunny/teapot/Suzanne/Spot the Cow) at the same shrunk-and-shifted
/// scale/offset, same four metal tones (bronze/chrome/gold/gunmetal), same
/// checkered ground and light. First scene to place multiple external
/// meshes in one composition - see CPU's comment for why this one uses
/// four opaque metals rather than glass (an earlier dielectric-on-mesh
/// attempt produced non-converging noise, flagged separately).
static void build_trophy_room_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Bunny (bronze) - matches CPU's shrunk scale/offset exactly.
	const int mat_bronze = add_metal(scene, make_float3(0.71f, 0.43f, 0.20f), 0.15f);
	load_obj_triangles_gpu(scene, "stanford-bunny.obj", mat_bronze,
		/*scale=*/10.3467f, make_float3(-3.32576f, -0.34123f, 0.01589f));

	// Teapot (chrome)
	const int mat_chrome = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.10f);
	load_obj_triangles_gpu(scene, "teapot.obj", mat_chrome,
		/*scale=*/0.50794f, make_float3(-2.07211f, 0.0f, 0.0f));

	// Suzanne (gold)
	const int mat_gold = add_metal(scene, make_float3(0.83f, 0.69f, 0.22f), 0.05f);
	load_obj_triangles_gpu(scene, "suzanne.obj", mat_gold,
		/*scale=*/0.81270f, make_float3(3.22694f, -0.21723f, -3.33526f));

	// Spot the Cow (gunmetal)
	const int mat_gunmetal = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.08f);
	load_obj_triangles_gpu(scene, "spot.obj", mat_gunmetal,
		/*scale=*/0.94651f, make_float3(3.5f, 0.69739f, -0.17989f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 8: Final Scene (Ray Tracing: The Next Week finale).
/// Matches CPU build_final_scene() (src/TheRestOfYourLife/scenes_book.h)
/// structurally: 400-box randomized-height ground, area light quad, moving
/// sphere, dielectric + metal spheres, Earth-image and Perlin-noise
/// textured spheres, and a 1000-sphere rotated/translated cluster - all
/// now real GPU features (boxes-as-quads, motion blur, large static sphere
/// counts, and - as of this function - image/noise textures via
/// load_image_texture_gpu/add_noise_texture_gpu and shade_material()'s
/// texture sampling, see optix_types.h's MaterialData::textureIdx).
/// The two constant_medium fog spheres (small blue fog + giant whole-scene
/// haze) use MaterialType::DielectricMedium, which collapses CPU's "same
/// dielectric boundary sphere added to the world twice - once directly,
/// once wrapped in constant_medium, whichever hits closer each bounce wins"
/// trick into a single material (see that type's comment in optix_types.h).
/// Uses its own fixed-seed RNG for the ground/sphere-cluster randomization,
/// like every other procedural GPU scene (e.g. build_bouncing_spheres) -
/// not intended to pixel-match CPU's independently-seeded layout.
void build_final_scene_gpu(SceneData& scene) {
	std::mt19937 rng(8u);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	// Ground: 20x20 grid of boxes with randomized height, matching CPU's
	// loop bounds/spacing (w=100, x0/z0 in [-1000,1000)) exactly.
	{
		const int mat_ground = add_lambertian(scene, make_float3(0.48f, 0.83f, 0.53f));
		constexpr int kBoxesPerSide = 20;
		constexpr float w = 100.0f;
		for (int i = 0; i < kBoxesPerSide; ++i) {
			for (int j = 0; j < kBoxesPerSide; ++j) {
				const float x0 = -1000.0f + i * w;
				const float z0 = -1000.0f + j * w;
				const float y1 = 1.0f + unit(rng) * 100.0f;  // random_double(1,101)
				add_box(scene, make_float3(x0, 0.0f, z0), make_float3(x0 + w, y1, z0 + w), mat_ground);
			}
		}
	}

	// Area light quad (matches CPU exactly: Q=(123,554,147), u=(300,0,0), v=(0,0,265))
	{
		const int mat_light = add_diffuse_light(scene, make_float3(7.0f, 7.0f, 7.0f));
		add_transformed_quad(scene, make_float3(123.0f, 554.0f, 147.0f), make_float3(300.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 265.0f), mat_light);
	}

	// Moving sphere (real GPU motion blur - center1 differs from center).
	{
		const int mat = add_lambertian(scene, make_float3(0.7f, 0.3f, 0.1f));
		SphereData s{};
		s.center = make_float3(400.0f, 400.0f, 200.0f);
		s.center1 = make_float3(430.0f, 400.0f, 200.0f);  // center + (30,0,0)
		s.radius = 50.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Dielectric (glass) sphere.
	{
		const int mat = add_dielectric(scene, 1.5f);
		SphereData s{};
		s.center = make_float3(260.0f, 150.0f, 45.0f);
		s.center1 = s.center;
		s.radius = 50.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Metal sphere.
	{
		const int mat = add_metal(scene, make_float3(0.8f, 0.8f, 0.9f), 1.0f);
		SphereData s{};
		s.center = make_float3(0.0f, 150.0f, 145.0f);
		s.center1 = s.center;
		s.radius = 50.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Small fog sphere: dielectric(1.5) boundary with an internal blue-
	// tinted medium (matches CPU's constant_medium(boundary, 0.2,
	// color(0.2,0.4,0.9)) wrapping the same dielectric(1.5) boundary).
	// MaterialData.eta_c.x carries sigma_t=0.2 (density); .fuzz=0 is the
	// HG asymmetry g, matching CPU's legacy constructor's g=0.0 default.
	{
		const int mat = add_dielectric_medium(scene, make_float3(0.2f, 0.4f, 0.9f), 1.5f, 0.2f);
		SphereData s{};
		s.center = make_float3(360.0f, 150.0f, 145.0f);
		s.center1 = s.center;
		s.radius = 70.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Giant whole-scene haze sphere: extremely subtle white atmospheric
	// tint (matches CPU's constant_medium(boundary_r5000, .0001,
	// color(1,1,1))). The camera and every other object in this scene sit
	// well within its radius-5000 boundary, so every primary ray starts
	// already inside it - see MaterialType::DielectricMedium's comment in
	// optix_types.h for why that makes the "entry from outside" half of
	// this material's logic unreachable here, which is fine, it's still
	// correct.
	{
		const int mat = add_dielectric_medium(scene, make_float3(1.0f, 1.0f, 1.0f), 1.5f, 0.0001f);
		SphereData s{};
		s.center = make_float3(0.0f, 0.0f, 0.0f);
		s.center1 = s.center;
		s.radius = 5000.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Earth-image-texture sphere. Falls back to CPU's own solid-cyan
	// missing-texture color if earthmap.jpg can't be found (see
	// load_image_texture_gpu's comment) - this matches CPU's behavior
	// exactly rather than being a GPU-specific limitation.
	{
		const int earthTexIdx = load_image_texture_gpu(scene, "earthmap.jpg");
		const int mat = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), earthTexIdx);
		SphereData s{};
		s.center = make_float3(400.0f, 200.0f, 400.0f);
		s.center1 = s.center;
		s.radius = 100.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Perlin-noise-texture sphere (matches CPU's noise_texture(0.2) exactly).
	{
		const int noiseTexIdx = add_noise_texture_gpu(scene, 0.2f);
		const int mat = safe_cast_to_int(scene.materials.size());
		add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), noiseTexIdx);
		SphereData s{};
		s.center = make_float3(220.0f, 280.0f, 300.0f);
		s.center1 = s.center;
		s.radius = 80.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// 1000-sphere white cluster, rotated 15deg around Y then translated -
	// matches CPU's random_double(0,165) box + rotate_y(15) + translate
	// (-100,270,395) exactly in structure (own RNG sequence, not CPU's).
	{
		const int mat_white = add_lambertian(scene, make_float3(0.73f, 0.73f, 0.73f));
		constexpr int kNumSpheres = 1000;
		constexpr float kTranslate_x = -100.0f, kTranslate_y = 270.0f, kTranslate_z = 395.0f;
		for (int i = 0; i < kNumSpheres; ++i) {
			const float3 local = make_float3(unit(rng) * 165.0f, unit(rng) * 165.0f, unit(rng) * 165.0f);
			const float3 rotated = rotate_y(local, 15.0f);
			SphereData s{};
			s.center = make_float3(rotated.x + kTranslate_x, rotated.y + kTranslate_y, rotated.z + kTranslate_z);
			s.center1 = s.center;
			s.radius = 10.0f;
			s.materialIdx = mat_white;
			scene.spheres.push_back(s);
		}
	}
}

/// @brief Scene 62: Crytek Sponza. Matches CPU build_sponza() exactly: no
/// separate ground sphere (the mesh's own floor is part of the geometry),
/// real per-face textures sampled via the mesh's own UVs from the companion
/// sponza.mtl's map_Kd images (models/sponza_textures/, falling back to a
/// flat sandstone lambertian for any face with no usemtl/unknown material/
/// missing texture), no explicit light source object -- lit purely via the
/// flat-color "sky" set on GpuCameraParams::backgroundColor at the case-62
/// dispatch site below (GPU has no sky_light/infinite-light object the way
/// the CPU registry's build_sky field does -- background color IS the GPU
/// sky, same convention as scene 24's HDRI Sky).
static void build_sponza_gpu(SceneData& scene) {
	const int mat_stone = add_lambertian(scene, make_float3(0.80f, 0.74f, 0.62f));
	load_obj_triangles_mtl_gpu(scene, "sponza.obj", mat_stone,
		/*scale=*/1.0f, make_float3(60.52f, 126.44f, 38.69f), "sponza_textures");
}

/// @brief Scene 63: Amazon Lumberyard Bistro (Exterior). Matches CPU
/// build_bistro_exterior() exactly. See build_sponza_gpu()'s own comment
/// for the shared design rationale (real per-face textures via map_Kd, sky
/// via backgroundColor). 2.84M triangles -- the largest mesh in this
/// codebase.
static void build_bistro_exterior_gpu(SceneData& scene) {
	const int mat_plaster = add_lambertian(scene, make_float3(0.75f, 0.62f, 0.50f));
	load_obj_triangles_mtl_gpu(scene, "bistro_exterior.obj", mat_plaster,
		/*scale=*/1.0f, make_float3(-1526.37f, 472.62f, -267.01f), "bistro_textures");
}

/// @brief Scene 64: Rungholt. Matches CPU build_rungholt() exactly. See
/// build_sponza_gpu()'s comment for the shared design rationale, and CPU
/// build_rungholt()'s own comment for the real negative-face-index OBJ
/// loader bug this mesh exposed (fixed in load_obj_triangles_gpu()'s
/// underlying parser the same way as the CPU loader -- see that function;
/// load_obj_triangles_mtl_gpu() shares the same fix).
static void build_rungholt_gpu(SceneData& scene) {
	const int mat_wood = add_lambertian(scene, make_float3(0.62f, 0.48f, 0.34f));
	load_obj_triangles_mtl_gpu(scene, "rungholt.obj", mat_wood,
		/*scale=*/1.0f, make_float3(0.0f, 0.0f, 0.0f));
}

/// @brief Build a scene and configure the camera
/// @param scene_id Scene identifier, category letter + number ("A1" = Cornell Box)
/// @param image_width Output image width in pixels
/// @param image_height Output image height in pixels
/// @param scene Output scene data to populate
/// @param camera_params Output camera parameters array [origin(3), lower_left(3), horizontal(3), vertical(3)]
/// @return true if scene was built successfully, false for unknown scene_id
// Builds a scene that came from a .pbrt file on disk. Separate from the
// switch below because there is nothing to switch on: these scenes are
// discovered at startup, so the code path is one function rather than one
// case per scene.
static bool build_loaded_pbrt_scene(
	const char* path,
	SceneData& scene,
	float* camera_params,
	const int image_width,
	const int image_height,
	const double cam_x,
	const double cam_y,
	const double cam_z,
	const bool force_camera_override,
	GpuCameraParams* out_camera_extra
) {
	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path);
	if (!loaded.ok) {
		std::cerr << "[OptiX] " << loaded.error << "\n";
		return false;
	}
	for (const pbrt_scene::Warning& w : loaded.scene.warnings)
		std::cerr << "[OptiX] warning: " << path << ": " << w.message << "\n";

	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(loaded.scene, scene);
	std::cerr << "[OptiX] Loaded " << path << ": " << stats.triangles
		  << " triangles, " << stats.spheres << " spheres, "
		  << stats.quadLights << " quads, "
		  << scene.lightIndices.size() << " sampled lights\n";

	// Flat-colour GPU approximation of the scene's own LightSource "infinite"
	// (see GpuCameraParams::backgroundColor's comment and pbrt_gpu_builder.h's
	// BuildStats::backgroundColor) - same shape as every hand-written HDRI
	// scene's own GPU port a few lines up in this file. Left at zero-init
	// (matches CPU bg=(0,0,0)) for a scene with no infinite light.
	if (out_camera_extra) out_camera_extra->backgroundColor = stats.backgroundColor;

	// Reported, not warned about: these are sampled properly now (as
	// GpuLightKind::Triangle), so the only thing worth saying is that they
	// took the per-triangle path rather than the cheaper merged-quad one.
	if (stats.emissiveTrianglesSampledIndividually > 0) {
		std::cerr << "[OptiX] " << stats.emissiveTrianglesSampledIndividually
			  << " emissive triangle(s) are not parallelograms and are sampled "
			     "individually\n";
	}
	if (scene.lightIndices.empty()) {
		std::cerr << "[OptiX] warning: no samplable lights in this scene - "
			     "expect a very dark image.\n";
	}
	if (stats.instancePlacements > 0) {
		std::cerr << "[OptiX] " << stats.instancePlacements << " object instance placement(s)\n";
	}

	// The scene's own camera, unless the user moved it.
	const pbrt_flatten::Camera& c = loaded.scene.camera;
	const float3 lookfrom = force_camera_override
		? make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y),
			      static_cast<float>(cam_z))
		: make_float3(static_cast<float>(c.lookfrom[0]),
			      static_cast<float>(c.lookfrom[1]),
			      static_cast<float>(c.lookfrom[2]));
	const float3 lookat = make_float3(static_cast<float>(c.lookat[0]),
					  static_cast<float>(c.lookat[1]),
					  static_cast<float>(c.lookat[2]));
	const float3 vup = make_float3(static_cast<float>(c.up[0]),
				       static_cast<float>(c.up[1]),
				       static_cast<float>(c.up[2]));
	const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
	// focus_dist is 1.0f, NOT the scene's focal distance - and that matches
	// every other GPU scene in this file, all of which pass 1.0f.
	//
	// The two backends use this parameter differently. camera.h places the
	// CPU viewport AT focus_dist, so there it must be a real world distance
	// (see focusDistanceFor()'s comment in pbrt_flatten.h - feeding it pbrt's
	// 1e6 "no depth of field" sentinel deleted near geometry). Here the
	// viewport sits at unit distance, so anything other than 1.0f just scales
	// it: passing a real focal distance like 800 makes the viewport 800x too
	// wide and every primary ray misses the scene.
	build_pinhole_camera_params(
		lookfrom, lookat, vup, static_cast<float>(c.vfov), aspect,
		1.0f, camera_params);

	// Route non-perspective Camera directives to their GPU camera model, the
	// same generic, data-driven way CPU's setup_camera lambda does (see
	// scene_registry.h) - camera_params above stays populated regardless, as
	// the fallback optix_interface.cpp uses when kind ends up Perspective.
	if (out_camera_extra && c.type != "perspective") {
		if (c.type == "orthographic") {
			// Mirrors case 32's own math (scene_builder.cpp, build_scene()),
			// just driven by this scene's own lookfrom/lookat/up/screenwindow
			// instead of a hardcoded built-in-scene camera.
			float xmin, xmax, ymin, ymax;
			if (c.hasScreenWindow) {
				xmin = static_cast<float>(c.screenWindow[0]);
				xmax = static_cast<float>(c.screenWindow[1]);
				ymin = static_cast<float>(c.screenWindow[2]);
				ymax = static_cast<float>(c.screenWindow[3]);
			} else {
				compute_screen_window<float>(image_width, image_height, xmin, xmax, ymin, ymax);
			}
			const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
			const float3 u = normalize(cross(vup, w));
			const float3 v = cross(w, u);
			const float3 horizontal = make_float3((xmax - xmin) * u.x, (xmax - xmin) * u.y, (xmax - xmin) * u.z);
			const float3 vertical   = make_float3((ymax - ymin) * v.x, (ymax - ymin) * v.y, (ymax - ymin) * v.z);
			const float3 lower_left_corner = make_float3(
				lookfrom.x + xmin * u.x + ymin * v.x,
				lookfrom.y + xmin * u.y + ymin * v.y,
				lookfrom.z + xmin * u.z + ymin * v.z);
			out_camera_extra->kind = CameraKind::Orthographic;
			out_camera_extra->lower_left_corner = lower_left_corner;
			out_camera_extra->horizontal = horizontal;
			out_camera_extra->vertical = vertical;
			out_camera_extra->w = make_float3(-w.x, -w.y, -w.z);
		} else if (c.type == "spherical" || c.type == "environment") {
			if (c.sphericalMapping == "equalarea") {
				// The device raygen (optix_device_helpers.h / wavefront_kernels.cu)
				// only implements EquiRectangular - no GPU EqualArea mapping
				// exists yet. Still route to Spherical (closer than staying
				// Perspective) but say so, the same "warn, use the closer
				// approximation" convention as pbrt_flatten.h's own gaps.
				std::cerr << "[OptiX] warning: " << path
					  << ": spherical camera's \"equalarea\" mapping has no GPU "
					     "implementation; rendering as equirectangular instead\n";
			}
			const Mat4<float> ctw = make_look_at<float>(
				lookfrom.x, lookfrom.y, lookfrom.z,
				lookat.x, lookat.y, lookat.z,
				vup.x, vup.y, vup.z);
			const CamVec3<float> su = ctw.transform_vec(1.0f, 0.0f, 0.0f);
			const CamVec3<float> sv = ctw.transform_vec(0.0f, 1.0f, 0.0f);
			const CamVec3<float> sw = ctw.transform_vec(0.0f, 0.0f, 1.0f);
			out_camera_extra->kind = CameraKind::Spherical;
			out_camera_extra->origin = lookfrom;
			out_camera_extra->su = make_float3(su.x, su.y, su.z);
			out_camera_extra->sv = make_float3(sv.x, sv.y, sv.z);
			out_camera_extra->sw = make_float3(sw.x, sw.y, sw.z);
		} else if (c.type == "realistic") {
			// c.lensFile is guaranteed non-empty here - pbrt_flatten.h already
			// fell back to "perspective" at flatten() time if the scene gave
			// none (see Camera's own comment). A missing/malformed file on
			// disk is still possible and only detectable here, where actual
			// file access happens.
			std::string lensText;
			if (!pbrt_load::loadFileNear(path, c.lensFile, lensText)) {
				std::cerr << "[OptiX] warning: " << path << ": realistic camera lensfile '"
					  << c.lensFile << "' not found; rendering as perspective instead\n";
			} else {
				const std::vector<double> lensD = pbrt_load::parseLensFile(lensText);
				if (lensD.empty()) {
					std::cerr << "[OptiX] warning: " << path << ": realistic camera lensfile '"
						  << c.lensFile << "' has no usable rows; rendering as perspective instead\n";
				} else {
					std::vector<float> lens;
					lens.reserve(lensD.size());
					for (double v : lensD) lens.push_back(static_cast<float>(v));
					const float aspectF = (image_height > 0)
						? static_cast<float>(image_width) / static_cast<float>(image_height) : 1.0f;
					const float halfY = static_cast<float>(c.filmDiagonalMM) / (2.0f * std::sqrt(aspectF * aspectF + 1.0f));
					const float halfX = aspectF * halfY;
					const Mat4<float> ctw = make_look_at<float>(
						lookfrom.x, lookfrom.y, lookfrom.z,
						lookat.x, lookat.y, lookat.z,
						vup.x, vup.y, vup.z);
					const float focusDist = static_cast<float>(pbrt_flatten::focusDistanceFor(c));
					RealisticCamera<float> realCam(ctw, halfX, halfY, focusDist,
						static_cast<float>(c.apertureDiameterMM), lens);

					scene.lensElements.clear();
					for (int i = 0; i < realCam.num_elements(); ++i) {
						GpuLensElement le{};
						le.curvatureRadius = realCam.lens_curvature_radius(i);
						le.thickness       = realCam.lens_thickness(i);
						le.eta              = realCam.lens_eta(i);
						le.apertureRadius   = realCam.lens_aperture_radius(i);
						scene.lensElements.push_back(le);
					}
					scene.exitPupilBounds.clear();
					for (int i = 0; i < realCam.num_exit_pupil_bounds(); ++i) {
						GpuExitPupilBounds b{};
						b.xMin = realCam.exit_pupil_xmin(i);
						b.xMax = realCam.exit_pupil_xmax(i);
						b.yMin = realCam.exit_pupil_ymin(i);
						b.yMax = realCam.exit_pupil_ymax(i);
						b.degenerate = realCam.exit_pupil_degenerate(i) ? 1 : 0;
						scene.exitPupilBounds.push_back(b);
					}

					const CamVec3<float> wo = realCam.world_origin();
					const CamVec3<float> wr = realCam.world_right();
					const CamVec3<float> wu = realCam.world_up();
					const CamVec3<float> wf = realCam.world_forward();

					out_camera_extra->kind = CameraKind::Realistic;
					out_camera_extra->origin = make_float3(wo.x, wo.y, wo.z);
					out_camera_extra->su = make_float3(wr.x, wr.y, wr.z);
					out_camera_extra->sv = make_float3(wu.x, wu.y, wu.z);
					out_camera_extra->sw = make_float3(wf.x, wf.y, wf.z);
					out_camera_extra->film_half_x = realCam.film_half_x();
					out_camera_extra->film_half_y = realCam.film_half_y();
					out_camera_extra->lens_rear_z = realCam.lens_rear_z();
					out_camera_extra->numLensElements = static_cast<int>(scene.lensElements.size());
					out_camera_extra->numExitPupilBounds = static_cast<int>(scene.exitPupilBounds.size());
				}
			}
		}
	}

	return true;
}

bool build_scene(
	const char* scene_id,
	const int image_width,
	const int image_height,
	SceneData& scene,
	float* camera_params,
	const double cam_x,
	const double cam_y,
	const double cam_z,
	GpuCameraParams* out_camera_extra,
	bool force_camera_override
) {
	if (camera_params == nullptr) {
		return false;  // Invalid camera parameter buffer
	}

	// The switch below still keys on the OLD flat 0..68 int id (unchanged,
	// on purpose - see SceneDescriptor::legacy_id's comment in
	// scene_registry.h: rewriting ~900 lines of case bodies into a
	// string-keyed dispatch wasn't worth the risk for what's purely an
	// internal implementation detail). cpu_scene_legacy_id_by_id() is the
	// one place that translates the new id back to it; -1 (not found, e.g.
	// a garbled scene_id) falls through to the same `default:` case an
	// out-of-range legacy id always did.
	const int legacy_scene_id = cpu_scene_legacy_id_by_id(scene_id);

	// Clear previous scene data
	scene.spheres.clear();
	scene.quads.clear();
	scene.bilinearPatches.clear();
	scene.triangles.clear();
	scene.materials.clear();

	// Shared camera setup for the Cornell-box-shell scenes (rough metal/glass,
	// conductor, coated diffuse/conductor, thin glass, wax slab, crystal,
	// spotlight/distant/point/goniometric/projection light, portal light,
	// smoke, homogeneous medium, subsurface slab, normal-mapped, bilinear
	// patch) -- all share the same lookat-center-of-box camera.
	const auto setup_cornell_box_camera = [&]() {
		const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
		const float3 lookat = make_float3(278.0f, 278.0f, 278.0f);
		const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
		const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
		build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);
	};

	// Build requested scene
	switch (legacy_scene_id) {
		case 0:  // Cornell Box
			build_cornell_box(scene);

			// Configure camera for Cornell Box
			{
				const float3 lookfrom = make_float3(
					static_cast<float>(cam_x),
					static_cast<float>(cam_y),
					static_cast<float>(cam_z)
				);
				const float3 lookat = make_float3(278.0f, 278.0f, 278.0f);  // Center of box
				const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
				const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
				build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);
			}
			break;

				case 1:  // Bouncing Spheres (motion blur - see build_bouncing_spheres)
					build_bouncing_spheres(scene);

					// Configure camera. This scene's CameraConfig in
					// scene_registry.h doesn't set CameraMode::UserControlled,
					// so it defaults to Fixed - the CPU renderer (cpu_interface.cpp)
					// ignores cam_x/y/z for Fixed scenes and always uses the
					// registry's own lookfrom (13,2,3), UNLESS force_camera_override
					// is set (main.cpp's video-mode frame loop, which must animate
					// the camera every frame). Match that here rather than always
					// forwarding cam_x/y/z verbatim: this scene's spheres are
					// clustered within roughly +-15 units of the origin, so a
					// leftover Cornell-Box-scale camera position (e.g. (278,278,-800),
					// a common default/preset for other scenes) would place the
					// camera absurdly far away, rendering an unrecognizable speck,
					// for any single-image render that doesn't opt into the override.
					{
						const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 13.0f, 2.0f, 3.0f);
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

						// defocus_angle=0.6, focus_dist=10.0: matches CPU
						// CameraConfig's DOF values for this scene (the
						// book's own final-render "beauty shot" values,
						// focused near the 3 hero spheres) - same thin-lens
						// wiring as case 22 (Depth of Field scene).
						constexpr float kPi = 3.14159265358979323846f;
						constexpr float defocus_angle = 0.6f;
						constexpr float focus_dist    = 10.0f;
						float3 dof_u, dof_v;
						build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, focus_dist, camera_params, &dof_u, &dof_v);

						if (out_camera_extra) {
							// A nonzero defocus disk opts this scene out of
							// optix_interface.cpp's generic camera_params->
							// cameraExtra fallback (see its own comment on
							// defocusDiskZero), so kind/origin/lower_left_corner/
							// horizontal/vertical must be set explicitly here too -
							// same full set case 22 (Depth of Field scene) sets.
							out_camera_extra->kind = CameraKind::Perspective;
							out_camera_extra->origin = lookfrom;
							out_camera_extra->lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
							out_camera_extra->horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
							out_camera_extra->vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);

							const float defocus_radius = focus_dist * tanf((defocus_angle * kPi / 180.0f) / 2.0f);
							out_camera_extra->defocus_disk_u = make_float3(dof_u.x * defocus_radius, dof_u.y * defocus_radius, dof_u.z * defocus_radius);
							out_camera_extra->defocus_disk_v = make_float3(dof_v.x * defocus_radius, dof_v.y * defocus_radius, dof_v.z * defocus_radius);

							// Flat light-blue background, matching CPU registry's
							// bg=(0.70,0.80,1.00) for this scene (see
							// GpuCameraParams::backgroundColor's comment).
							out_camera_extra->backgroundColor = make_float3(0.70f, 0.80f, 1.00f);
						}
					}
					break;

				case 2:  // Checkered Spheres
					build_checkered_spheres(scene);

					// Configure camera. Same Fixed-mode situation as scene 1
					// above (no CameraMode::UserControlled in this scene's
					// registry entry, and the same force_camera_override
					// escape hatch for video mode) - ignore cam_x/y/z by
					// default, matching CPU exactly, rather than placing the
					// camera at whatever Cornell-Box-scale position happened
					// to be leftover from a previous scene.
					{
						const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 13.0f, 2.0f, 3.0f);
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, 1.0f, camera_params);

						// Warm sunset-ish flat background, matching CPU registry's
						// bg=(0.90,0.75,0.55) for this scene - fits the "planet"
						// motif and contrasts with the new accent spheres.
						if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.90f, 0.75f, 0.55f);
					}
					break;

				case 3:  // Earth (see build_earth_gpu's comment)
					build_earth_gpu(scene);

					// Same Fixed-mode situation as scenes 1/2 above - ignore
					// cam_x/y/z by default (this scene's single sphere sits
					// right at the origin, so a leftover Cornell-Box-scale
					// camera position would place it out of frame entirely).
					{
						const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 0.0f, 12.0f);
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						// vfov 20->25: matches CPU registry, widened to leave
						// room for the new moon accent sphere near the frame
						// edge without cropping it.
						build_pinhole_camera_params(lookfrom, lookat, vup, 25.0f, aspect, 1.0f, camera_params);

						// Flat light-blue background, matching CPU registry's
						// bg=(0.70,0.80,1.00) for this scene (see
						// GpuCameraParams::backgroundColor's comment).
						if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.70f, 0.80f, 1.00f);
					}
					break;

				case 4:  // Perlin Spheres (see add_perlin_spheres_pair_gpu's comment)
					build_perlin_spheres_gpu(scene);

					// Same Fixed-mode situation as scenes 1/2/3 above.
					{
						const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 13.0f, 2.0f, 3.0f);
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, 1.0f, camera_params);

						// Flat light-blue background, matching CPU registry's
						// bg=(0.70,0.80,1.00) for this scene (see
						// GpuCameraParams::backgroundColor's comment).
						if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.70f, 0.80f, 1.00f);
					}
					break;

				case 5:  // Colored Quads
						build_quads_scene(scene);

						// Configure camera
					{
						const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						build_pinhole_camera_params(lookfrom, lookat, vup, 80.0f, aspect, 1.0f, camera_params);  // 80: wide angle for quads

						// Flat sky-blue fill background, matching CPU registry's
						// bg=(0.70,0.80,1.00) for scene 5 (see
						// GpuCameraParams::backgroundColor's comment) - now a
						// secondary fill alongside the floating lamp quad
						// build_quads_scene() adds (that quad used to be this
						// scene's only possible radiance source before it had
						// any registered light of its own; see its comment).
						if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.70f, 0.80f, 1.00f);
					}
					break;

				case 6: {  // Simple Light (see build_simple_light_gpu's comment)
					build_simple_light_gpu(scene);

					const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
					const float3 lookat = make_float3(0.0f, 2.0f, 0.0f);
					const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
					const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
					build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, 1.0f, camera_params);
					// backgroundColor left at zero-init (matches CPU bg=(0,0,0)) -
					// this scene has real emissive geometry (the light sphere and
					// light quad above).
					break;
				}

				case 8:  // Final Scene (see build_final_scene_gpu's own comment
						 // for what's ported vs. placeholder-approximated)
					build_final_scene_gpu(scene);
					{
						// Fixed-mode scene (no CameraMode::UserControlled in its
						// registry entry) - ignore cam_x/y/z by default, matching
						// CPU, UNLESS force_camera_override is set (video mode's
						// per-frame animated position), same convention as
						// scenes 1-4 above.
						const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 478.0f, 278.0f, -600.0f);
						const float3 lookat = make_float3(278.0f, 278.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);
						// Subtle deep ambient instead of pure black, matching CPU
						// registry's bg=(0.03,0.025,0.02) - the box-grid ground and
						// negative space used to render into a stark void even
						// though this scene has a real area light (the light quad
						// above) doing the actual illumination.
						if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.03f, 0.025f, 0.02f);
					}
					break;

				case 9: {  // Rough Metal Spheres (GGX)
										build_rough_metal_spheres(scene);

										// Camera: vfov=42, lookfrom=(cam_x,cam_y,cam_z), lookat=(0,1,0) -
										// matches CPU CameraConfig row for scene 9 (widened/pulled back so
										// all 5 spheres, spanning x=+-6, actually fit in frame).
										const float3 lookfrom9 = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
										const float3 lookat9   = make_float3(0.0f, 1.0f, 0.0f);
										const float3 vup9      = make_float3(0.0f, 1.0f, 0.0f);
										const float aspect9    = static_cast<float>(image_width) / static_cast<float>(image_height);
										build_pinhole_camera_params(lookfrom9, lookat9, vup9, 42.0f, aspect9, 1.0f, camera_params);
										break;
									}

							case 10:  // Cornell Rough Metal (GGX)
								build_cornell_rough_metal(scene);
								setup_cornell_box_camera();
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 10 - a flat black background
									// made the box's near-mirror faces (which mostly reflect back out
									// the box's open front) read as solid black instead of shiny metal;
									// a dim fill fixes that.
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.055f, 0.07f);
								}
								break;

							case 11:  // Cornell Rough Glass (GGX)
								build_cornell_rough_glass(scene);
								setup_cornell_box_camera();
								break;

							case 12:  // Cornell Conductor (GGX + complex Fresnel, pbrt-v4 ConductorBxDF)
								build_cornell_conductor(scene);
								setup_cornell_box_camera();
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 12 (same reasoning as scene 10).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.055f, 0.07f);
								}
								break;

							case 13:  // Cornell Coated Diffuse (pbrt-v4 CoatedDiffuseBxDF)
								build_cornell_coated_diffuse(scene);
								setup_cornell_box_camera();
								break;

							case 14:  // Cornell Thin Glass (pbrt-v4 ThinDielectricBxDF)
								build_cornell_thin_glass(scene);
								setup_cornell_box_camera();
								break;

							case 15:  // Cornell Coated Conductor (pbrt-v4 CoatedConductorBxDF)
								build_cornell_coated_conductor(scene);
								setup_cornell_box_camera();
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 15 (same reasoning as scene 10).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.055f, 0.07f);
								}
								break;

							case 16:  // Cornell Wax Slab (pbrt-v4 DiffuseTransmissionBxDF)
								build_cornell_wax_slab(scene);
								setup_cornell_box_camera();
								break;

							case 17:  // Cornell Crystal (pbrt-v4 NormalizedFresnelBxDF)
								build_cornell_crystal(scene);
								setup_cornell_box_camera();
								break;

							case 25:  // Spotlight Cornell (pbrt-v4 SpotLight)
								build_spotlight_cornell_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 26:  // Distant Light Cornell (pbrt-v4 DistantLight)
								build_distant_light_cornell_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 27:  // Point Light Cornell (pbrt-v4 PointLight)
								build_point_light_cornell_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 28:  // Goniometric Light Cornell (pbrt-v4 GoniometricLight)
								build_goniometric_cornell_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 29:  // Projection Light Cornell (pbrt-v4 ProjectionLight)
								build_projection_cornell_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 35:  // Portal Infinite Light (pbrt-v4 PortalImageInfiniteLight)
								build_portal_light_scene_gpu(scene);
								setup_cornell_box_camera();
								// Matches CPU build_portal_sky()'s fix (see that function's
								// comment) - dimmed from (1.0,1.2,1.5), which pushed the room
								// toward overexposed.
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.55f, 0.65f, 0.85f);
								break;

							case 7:  // Cornell Smoke (constant_medium)
								build_cornell_smoke_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 30:  // Homogeneous Medium (constant_medium, HG g=0.3)
								build_homogeneous_medium_scene_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 21:  // Subsurface Slab (see build_subsurface_slab_gpu's comment)
								build_subsurface_slab_gpu(scene);
								setup_cornell_box_camera();
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 21 (same reasoning as scene 10 -
									// the dielectric shell's reflection/refraction rays hit the same
									// open-front-of-box black-background issue).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.055f, 0.07f);
								}
								break;

							case 20:  // Normal Mapped Cornell (see build_normal_mapped_cornell_gpu's comment)
								build_normal_mapped_cornell_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 23:  // Bilinear Patch Scene (pbrt-v4 BilinearPatch shape)
								build_bilinear_patch_scene_gpu(scene);
								setup_cornell_box_camera();
								break;

							case 22: {  // Depth of Field (thin-lens perspective camera)
								build_depth_of_field_gpu(scene);
								constexpr float kPi = 3.14159265358979323846f;
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.0f, 9.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								constexpr float defocus_angle = 10.0f;   // matches CPU CameraConfig row for scene 22
								constexpr float focus_dist    = 9.0f;    // ditto
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								// vfov widened from 20 to 62 so the row of spheres (spanning
								// x=+-5) actually fits in frame at focus_dist=9 - matches CPU
								// CameraConfig row for scene 22. defocus_angle/focus_dist
								// unchanged so the DOF blur physics stays as designed.
								float3 u, v;
								build_pinhole_camera_params(lookfrom, lookat, vup, 62.0f, aspect, focus_dist, camera_params, &u, &v);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Perspective;
									out_camera_extra->origin = lookfrom;
									out_camera_extra->lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
									out_camera_extra->horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
									out_camera_extra->vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
									// pbrt-v4/book-style thin-lens disk basis, scaled by focus_dist and
									// half the defocus cone angle - matches src/TheRestOfYourLife/
									// camera.h's defocus_disk_u/v exactly.
									const float defocus_radius = focus_dist * tanf((defocus_angle * kPi / 180.0f) / 2.0f);
									out_camera_extra->defocus_disk_u = make_float3(u.x * defocus_radius, u.y * defocus_radius, u.z * defocus_radius);
									out_camera_extra->defocus_disk_v = make_float3(v.x * defocus_radius, v.y * defocus_radius, v.z * defocus_radius);
									// Matches CPU CameraConfig bg for scene 22 - see
									// build_depth_of_field_gpu's comment for why this replaced a
									// synthetic overhead light.
									out_camera_extra->backgroundColor = make_float3(0.70f, 0.80f, 1.00f);
								}
								break;
							}

							case 32: {  // Orthographic Camera (parallel projection)
								build_ortho_camera_scene_gpu(scene);
								// lookfrom moved higher/farther back (was (0,3,12)) and the
								// screen-window scale reduced (was 8) - matches CPU's
								// setup_camera lambda for scene 32; see that lambda's comment
								// for why the old values put ray origins below the giant
								// ground sphere's surface for the bottom rows.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 10.0f, 20.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								float xmin, xmax, ymin, ymax;
								if (aspect >= 1.0f) { xmin = -aspect; xmax = aspect; ymin = -1.0f; ymax = 1.0f; }
								else                { xmin = -1.0f; xmax = 1.0f; ymin = -1.0f / aspect; ymax = 1.0f / aspect; }
								constexpr float kScreenScale = 5.0f;
								xmin *= kScreenScale; xmax *= kScreenScale; ymin *= kScreenScale; ymax *= kScreenScale;

								const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
								const float3 u = normalize(cross(vup, w));
								const float3 v = cross(w, u);

								const float3 horizontal = make_float3((xmax - xmin) * u.x, (xmax - xmin) * u.y, (xmax - xmin) * u.z);
								const float3 vertical   = make_float3((ymax - ymin) * v.x, (ymax - ymin) * v.y, (ymax - ymin) * v.z);
								const float3 lower_left_corner = make_float3(
									lookfrom.x + xmin * u.x + ymin * v.x,
									lookfrom.y + xmin * u.y + ymin * v.y,
									lookfrom.z + xmin * u.z + ymin * v.z
								);

								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, lower_left_corner);
								pack_float3(camera_params, 6, horizontal);
								pack_float3(camera_params, 9, vertical);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Orthographic;
									out_camera_extra->lower_left_corner = lower_left_corner;
									out_camera_extra->horizontal = horizontal;
									out_camera_extra->vertical = vertical;
									out_camera_extra->w = make_float3(-w.x, -w.y, -w.z);  // forward = negated look-from/look-at "backward" w
									// Matches CPU's build_ortho_sky() flat sky_light(0.5,0.7,1.0) - see
									// build_ortho_camera_scene_gpu's comment for why this replaced a
									// synthetic overhead light.
									out_camera_extra->backgroundColor = make_float3(0.5f, 0.7f, 1.0f);
								}
								break;
							}

							case 33: {  // Spherical (equirectangular) Camera
								build_spherical_camera_scene_gpu(scene);
								// SphericalCamera captures the full 360-degree sphere around
								// its origin, so orientation doesn't gate a field of view -
								// keep the original fixed su/sv/sw basis (right=+X, up=+Y,
								// forward=+Z, matching an identity camera-to-world) so the
								// panorama's default orientation is unchanged, but let the
								// origin track cam_x/y/z (only under force_camera_override,
								// same convention as scenes 1-4 above - matches CPU: origin
								// previously hardcoded to the world origin regardless, so
								// video mode's animated camera position had no effect and
								// every frame was identical).
								const float3 origin = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 0.0f, 0.0f);
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, origin);
								pack_float3(camera_params, 3, zero);
								pack_float3(camera_params, 6, zero);
								pack_float3(camera_params, 9, zero);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Spherical;
									out_camera_extra->origin = origin;
									out_camera_extra->su = make_float3(1.0f, 0.0f, 0.0f);
									out_camera_extra->sv = make_float3(0.0f, 1.0f, 0.0f);
									out_camera_extra->sw = make_float3(0.0f, 0.0f, 1.0f);
								}
								break;
							}

							case 36: {  // Realistic Camera (pbrt-v4 multi-element lens)
								build_realistic_camera_scene_gpu(scene);
								// Fixed-mode scene - let lookfrom track cam_x/y/z only under
								// force_camera_override (video mode), matching CPU's scene 36
								// setup_camera lambda; lookat stays fixed (this scene never
								// overrides it, matching every other scene's convention). The
								// oblique default (not dead-on with the sphere row) is required -
								// see scene_registry.h's scene 36 comment: the row sits exactly on
								// the old dead-on viewing axis, so the near sphere fully occluded
								// the rest from every lens sample.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 1.65f, 1.07f, -6.85f);
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, zero);
								pack_float3(camera_params, 6, zero);
								pack_float3(camera_params, 9, zero);

								if (out_camera_extra) {
									// Directly instantiate a host-side RealisticCamera<float> - reusing
									// the CPU C++ class from cameras.h - so FocusThickLens/
									// BoundExitPupil (both expensive, one-time precomputes) never need
									// a CUDA port. Same lens table, focus distance, aperture, and
									// camera-to-world as the CPU scene 36 (src/TheRestOfYourLife/
									// scene_registry.h) - keep both in sync if either changes.
									std::vector<float> lens = {
										 35.98738f,  1.21638f, 1.54f,  23.716f,
										 11.69718f,  9.9957f,  1.0f,   17.996f,
										 13.08714f, 15.9948f,  1.77f,  12.364f,
										-22.63294f,  2.7757f,  1.617f, 9.812f,
										  0.0f,      2.75f,    0.0f,   7.4f,     // aperture stop
										 36.3581f,   8.9722f,  1.617f, 12.7f,
										-17.8595f,   1.2f,     1.0f,   12.7f,
										100.0f,      2.9804f,  1.567f, 14.478f,
										-24.5656f,   0.0f,     1.0f,   15.0f
									};
									Mat4<float> ctw = make_look_at<float>(
										lookfrom.x, lookfrom.y, lookfrom.z,   // from
										1.4f, 1.0f,  5.5f,   // to
										0.0f, 1.0f,  0.0f    // up
									);
									// Film half-extents shrunk from 18/12mm (a full 35mm frame) to
									// 3.0/2.0mm, and focus distance/camera position updated to an
									// oblique framing of the sphere row - matches CPU's fix in
									// scene_registry.h's scene 36 setup_camera lambda, see that
									// comment for the full reasoning (lens vignetting at the old film
									// size, plus the row sitting on the old dead-on viewing axis).
									RealisticCamera<float> realCam(ctw, 3.0f, 2.0f, 12.4f, 8.0f, lens, 512);

									scene.lensElements.clear();
									for (int i = 0; i < realCam.num_elements(); ++i) {
										GpuLensElement le{};
										le.curvatureRadius = realCam.lens_curvature_radius(i);
										le.thickness       = realCam.lens_thickness(i);
										le.eta              = realCam.lens_eta(i);
										le.apertureRadius   = realCam.lens_aperture_radius(i);
										scene.lensElements.push_back(le);
									}
									scene.exitPupilBounds.clear();
									for (int i = 0; i < realCam.num_exit_pupil_bounds(); ++i) {
										GpuExitPupilBounds b{};
										b.xMin = realCam.exit_pupil_xmin(i);
										b.xMax = realCam.exit_pupil_xmax(i);
										b.yMin = realCam.exit_pupil_ymin(i);
										b.yMax = realCam.exit_pupil_ymax(i);
										b.degenerate = realCam.exit_pupil_degenerate(i) ? 1 : 0;
										scene.exitPupilBounds.push_back(b);
									}

									CamVec3<float> wo = realCam.world_origin();
									CamVec3<float> wr = realCam.world_right();
									CamVec3<float> wu = realCam.world_up();
									CamVec3<float> wf = realCam.world_forward();

									out_camera_extra->kind = CameraKind::Realistic;
									out_camera_extra->origin = make_float3(wo.x, wo.y, wo.z);
									out_camera_extra->su = make_float3(wr.x, wr.y, wr.z);
									out_camera_extra->sv = make_float3(wu.x, wu.y, wu.z);
									out_camera_extra->sw = make_float3(wf.x, wf.y, wf.z);
									out_camera_extra->film_half_x = realCam.film_half_x();
									out_camera_extra->film_half_y = realCam.film_half_y();
									out_camera_extra->lens_rear_z = realCam.lens_rear_z();
									out_camera_extra->numLensElements = static_cast<int>(scene.lensElements.size());
									out_camera_extra->numExitPupilBounds = static_cast<int>(scene.exitPupilBounds.size());
								}
								break;
							}

							// D5-D8: same classic Cornell box as scene 0 (A1) - see CPU
							// scene_registry.h's comment above these 4 rows for why - each
							// showing a different camera model. build_cornell_box(scene) is
							// the exact same shared-data-driven builder scene 0 uses below.
							case 65: {  // Depth of Field (Cornell Box)
								build_cornell_box(scene);
								constexpr float kPi = 3.14159265358979323846f;
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 278.0f, 278.0f, -800.0f);
								const float3 lookat   = make_float3(278.0f, 278.0f, 278.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								constexpr float defocus_angle = 2.0f;    // matches CPU CameraConfig row for scene 65
								constexpr float focus_dist    = 800.0f; // ditto
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								float3 u, v;
								build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, focus_dist, camera_params, &u, &v);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Perspective;
									out_camera_extra->origin = lookfrom;
									out_camera_extra->lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
									out_camera_extra->horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
									out_camera_extra->vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
									const float defocus_radius = focus_dist * tanf((defocus_angle * kPi / 180.0f) / 2.0f);
									out_camera_extra->defocus_disk_u = make_float3(u.x * defocus_radius, u.y * defocus_radius, u.z * defocus_radius);
									out_camera_extra->defocus_disk_v = make_float3(v.x * defocus_radius, v.y * defocus_radius, v.z * defocus_radius);
								}
								break;
							}

							case 66: {  // Orthographic Camera (Cornell Box)
								build_cornell_box(scene);
								// Dead-on, same lookfrom/lookat as scene 0 (A1) and scene 65
								// (D5) - matches CPU's setup_camera lambda for scene 66; see
								// that lambda's comment for why an angled "isometric" vantage
								// was tried and abandoned (CPU's camera_to_world convention
								// made the angled framing unpredictable to hand-tune, and this
								// dead-on view still unambiguously demonstrates parallel
								// projection's defining trait: perfectly parallel edges,
								// unlike scene 65's converging perspective lines).
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 278.0f, 278.0f, -800.0f);
								const float3 lookat   = make_float3(278.0f, 278.0f, 278.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								float xmin, xmax, ymin, ymax;
								if (aspect >= 1.0f) { xmin = -aspect; xmax = aspect; ymin = -1.0f; ymax = 1.0f; }
								else                { xmin = -1.0f; xmax = 1.0f; ymin = -1.0f / aspect; ymax = 1.0f / aspect; }
								// 320: matches CPU CameraConfig/setup_camera lambda for scene
								// 66 - fills close to half the frame with the box, centered.
								constexpr float kScreenScale = 320.0f;
								xmin *= kScreenScale; xmax *= kScreenScale; ymin *= kScreenScale; ymax *= kScreenScale;

								const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
								const float3 u = normalize(cross(vup, w));
								const float3 v = cross(w, u);

								const float3 horizontal = make_float3((xmax - xmin) * u.x, (xmax - xmin) * u.y, (xmax - xmin) * u.z);
								const float3 vertical   = make_float3((ymax - ymin) * v.x, (ymax - ymin) * v.y, (ymax - ymin) * v.z);
								const float3 lower_left_corner = make_float3(
									lookfrom.x + xmin * u.x + ymin * v.x,
									lookfrom.y + xmin * u.y + ymin * v.y,
									lookfrom.z + xmin * u.z + ymin * v.z
								);

								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, lower_left_corner);
								pack_float3(camera_params, 6, horizontal);
								pack_float3(camera_params, 9, vertical);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Orthographic;
									out_camera_extra->lower_left_corner = lower_left_corner;
									out_camera_extra->horizontal = horizontal;
									out_camera_extra->vertical = vertical;
									out_camera_extra->w = make_float3(-w.x, -w.y, -w.z);
								}
								break;
							}

							case 67: {  // Spherical Camera (Cornell Box)
								build_cornell_box(scene);
								// Origin at the box's center (278,278,278) - only under
								// force_camera_override (video mode), matching D3's own
								// convention; fixed axis-aligned su/sv/sw basis (identical
								// to what CPU's setup_camera lambda's make_look_at produces
								// from an origin looking toward origin+(0,0,1) with up=+Y)
								// so the panorama shows all 5 walls, the ceiling light, the
								// glass sphere, and the rotated box wrapped around it.
								const float3 origin = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 278.0f, 278.0f, 278.0f);
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, origin);
								pack_float3(camera_params, 3, zero);
								pack_float3(camera_params, 6, zero);
								pack_float3(camera_params, 9, zero);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Spherical;
									out_camera_extra->origin = origin;
									out_camera_extra->su = make_float3(1.0f, 0.0f, 0.0f);
									out_camera_extra->sv = make_float3(0.0f, 1.0f, 0.0f);
									out_camera_extra->sw = make_float3(0.0f, 0.0f, 1.0f);
								}
								break;
							}

							case 68: {  // Realistic Camera (Cornell Box)
								build_cornell_box(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 278.0f, 278.0f, -420.0f);
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, zero);
								pack_float3(camera_params, 6, zero);
								pack_float3(camera_params, 9, zero);

								if (out_camera_extra) {
									// Same simplified 9-element dgauss lens as scene 36 (D4) -
									// reused host-side RealisticCamera<float>, same reasoning
									// as that case for why FocusThickLens/BoundExitPupil don't
									// need a CUDA port. Aperture scaled way up from D4's 8mm to
									// 350mm - matches CPU's setup_camera lambda comment for
									// scene 68 (this box is ~550 units across vs D4's few-unit
									// scene, so D4's own aperture would give an imperceptibly
									// narrow defocus cone at a comparable framing distance).
									std::vector<float> lens = {
										 35.98738f,  1.21638f, 1.54f,  23.716f,
										 11.69718f,  9.9957f,  1.0f,   17.996f,
										 13.08714f, 15.9948f,  1.77f,  12.364f,
										-22.63294f,  2.7757f,  1.617f, 9.812f,
										  0.0f,      2.75f,    0.0f,   7.4f,     // aperture stop
										 36.3581f,   8.9722f,  1.617f, 12.7f,
										-17.8595f,   1.2f,     1.0f,   12.7f,
										100.0f,      2.9804f,  1.567f, 14.478f,
										-24.5656f,   0.0f,     1.0f,   15.0f
									};
									Mat4<float> ctw = make_look_at<float>(
										lookfrom.x, lookfrom.y, lookfrom.z,   // from
										278.0f, 278.0f, 278.0f,   // to
										0.0f, 1.0f,  0.0f    // up
									);
									RealisticCamera<float> realCam(ctw, 3.0f, 2.0f, 420.0f, 350.0f, lens, 512);

									scene.lensElements.clear();
									for (int i = 0; i < realCam.num_elements(); ++i) {
										GpuLensElement le{};
										le.curvatureRadius = realCam.lens_curvature_radius(i);
										le.thickness       = realCam.lens_thickness(i);
										le.eta              = realCam.lens_eta(i);
										le.apertureRadius   = realCam.lens_aperture_radius(i);
										scene.lensElements.push_back(le);
									}
									scene.exitPupilBounds.clear();
									for (int i = 0; i < realCam.num_exit_pupil_bounds(); ++i) {
										GpuExitPupilBounds b{};
										b.xMin = realCam.exit_pupil_xmin(i);
										b.xMax = realCam.exit_pupil_xmax(i);
										b.yMin = realCam.exit_pupil_ymin(i);
										b.yMax = realCam.exit_pupil_ymax(i);
										b.degenerate = realCam.exit_pupil_degenerate(i) ? 1 : 0;
										scene.exitPupilBounds.push_back(b);
									}

									CamVec3<float> wo = realCam.world_origin();
									CamVec3<float> wr = realCam.world_right();
									CamVec3<float> wu = realCam.world_up();
									CamVec3<float> wf = realCam.world_forward();

									out_camera_extra->kind = CameraKind::Realistic;
									out_camera_extra->origin = make_float3(wo.x, wo.y, wo.z);
									out_camera_extra->su = make_float3(wr.x, wr.y, wr.z);
									out_camera_extra->sv = make_float3(wu.x, wu.y, wu.z);
									out_camera_extra->sw = make_float3(wf.x, wf.y, wf.z);
									out_camera_extra->film_half_x = realCam.film_half_x();
									out_camera_extra->film_half_y = realCam.film_half_y();
									out_camera_extra->lens_rear_z = realCam.lens_rear_z();
									out_camera_extra->numLensElements = static_cast<int>(scene.lensElements.size());
									out_camera_extra->numExitPupilBounds = static_cast<int>(scene.exitPupilBounds.size());
								}
								break;
							}

							case 24: {  // HDRI Sky (flat-color background - see backgroundColor's comment)
								build_hdri_sky_world_gpu(scene);
								// lookfrom/vfov widened/pulled back so all 3 spheres (spanning
								// x=+-4) actually fit in frame - matches CPU CameraConfig row
								// for scene 24.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.3f, 15.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 42.0f, aspect, 1.0f, camera_params);  // 42: matches CPU CameraConfig row for scene 24

								if (out_camera_extra) {
									// CPU's build_hdri_sky() now actually uses its procedural
									// blue-to-warm-horizon gradient (previously built the image
									// then discarded it, returning a flat solid_color sky_light -
									// see that function's comment). GPU has no per-pixel
									// environment-map sampling (see GpuCameraParams::backgroundColor's
									// comment), so this flat color approximates the gradient's
									// average tone instead of matching it exactly.
									out_camera_extra->backgroundColor = make_float3(0.4f, 0.5f, 0.53f);
								}
								break;
							}

							case 31: {  // Cloud Medium (CloudMedium: heterogeneous Perlin-noise density)
								build_cloud_medium_scene_gpu(scene);
								// lookfrom/vfov widened/pulled back (was 20 deg at (0,5,20)) - matches
								// CPU CameraConfig row for scene 31; see that row's comment.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 4.0f, 26.0f);
								const float3 lookat   = make_float3(0.0f, 2.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);  // 40: matches CPU CameraConfig row for scene 31
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 31 - this is the scene's
									// ONLY light source (no emissive geometry), so a missing/black
									// background here means zero illumination anywhere in the image.
									out_camera_extra->backgroundColor = make_float3(0.5f, 0.7f, 1.0f);
								}
								break;
							}

							case 69: {  // Dielectric Medium Showcase (glass spheres w/ colored fog)
								build_dielectric_medium_scene_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 18.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);  // 40: matches CPU CameraConfig row for scene 69
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 69 - same reasoning as
									// scene 31 (Cloud Medium): this scene's only light source.
									out_camera_extra->backgroundColor = make_float3(0.5f, 0.7f, 1.0f);
								}
								break;
							}

							case 70: {  // RGB Grid Medium (heterogeneous per-voxel R/G/B nebula)
								build_rgb_grid_medium_scene_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 5.0f, 30.0f);
								const float3 lookat   = make_float3(0.0f, 3.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 45.0f, aspect, 1.0f, camera_params);  // 45: matches CPU CameraConfig row for scene 70
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 70 - same reasoning as
									// scene 31 (Cloud Medium): this scene's only light source.
									out_camera_extra->backgroundColor = make_float3(0.5f, 0.7f, 1.0f);
								}
								break;
							}

							case 72: {  // Curve Fibers (see build_curve_fibers_scene_gpu's comment)
								build_curve_fibers_scene_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.0f, 6.5f);
								const float3 lookat   = make_float3(0.0f, 0.7f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 38.0f, aspect, 1.0f, camera_params);  // 38: matches CPU CameraConfig row for scene 72
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 72.
									out_camera_extra->backgroundColor = make_float3(0.04f, 0.045f, 0.06f);
								}
								break;
							}

							case 19: {  // Hair Fibers (pbrt-v4 HairBxDF)
								build_hair_fibers_gpu(scene);
								// lookfrom/vfov widened/pulled back to fit the now wider-spaced
								// cluster - matches CPU CameraConfig row for scene 19.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.5f, 14.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 45.0f, aspect, 1.0f, camera_params);  // 45: matches CPU CameraConfig row for scene 19
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 19 (dim ambient - the
									// only light source, no emissive geometry in this scene).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.07f);
								}
								break;
							}

							case 34: {  // Measured BRDF (see build_measured_brdf_scene_gpu's comment)
								build_measured_brdf_scene_gpu(scene);
								// lookfrom/vfov widened/pulled back so all 5 spheres (spanning
								// x=+-5) actually fit in frame - matches CPU CameraConfig row for
								// scene 34.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.2f, 17.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 42.0f, aspect, 1.0f, camera_params);  // 42: matches CPU CameraConfig row for scene 34
								// backgroundColor left at zero-init (matches CPU bg=(0,0,0)) - this
								// scene has a real emissive light sphere, unlike scenes 19/31.
								break;
							}

							case 37: {  // Triangle Mesh (see build_triangle_mesh_scene_gpu's comment)
								build_triangle_mesh_scene_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 4.0f, 8.0f);
								const float3 lookat   = make_float3(0.0f, 2.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 37
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 37 (dim ambient - real
									// light sphere is the main source, matches scene 19's style).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 18: {  // Principled Showcase (see build_principled_showcase_gpu's comment)
								build_principled_showcase_gpu(scene);
								// lookfrom/vfov widened/pulled back so all 7 spheres (spanning
								// x=+-7 after the spacing fix) actually fit in frame - matches
								// CPU CameraConfig row for scene 18.
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.7f, 17.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 45.0f, aspect, 1.0f, camera_params);  // 45: matches CPU CameraConfig row for scene 18
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 18 (dim ambient - no
									// emissive geometry in this scene, same style as scenes 19/31/37).
									out_camera_extra->backgroundColor = make_float3(0.10f, 0.10f, 0.12f);
								}
								break;
							}

							case 38: {  // Stanford Bunny (see build_stanford_bunny_gpu's comment)
								build_stanford_bunny_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 38
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 38.
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 39: {  // Stanford Armadillo (see build_stanford_armadillo_gpu's comment)
								build_stanford_armadillo_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 39
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 39 (same as scene 38's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 40: {  // Stanford Happy Buddha (see build_stanford_happy_buddha_gpu's comment)
								build_stanford_happy_buddha_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 40
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 40 (same as scenes 38/39's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 41: {  // Stanford Lucy (see build_stanford_lucy_gpu's comment)
								build_stanford_lucy_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 41
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 41 (same as scenes 38-40's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 42: {  // Stanford XYZRGB Dragon (see build_stanford_dragon_gpu's comment)
								build_stanford_dragon_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 42
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 42 (same as scenes 38-41's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 43: {  // Utah Teapot (see build_utah_teapot_gpu's comment)
								build_utah_teapot_gpu(scene);
								// Camera pulled back further than the other mesh scenes'
								// default (0,3,7) - matches CPU CameraConfig row for scene 43,
								// see its comment in scene_registry.h for why (the teapot's
								// spout+handle make it much wider than tall).
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 6.0f, 20.0f);
								const float3 lookat   = make_float3(0.0f, 1.2f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 43
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 43 (same as scenes 38-42's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 44: {  // Spot the Cow (see build_spot_cow_gpu's comment)
								build_spot_cow_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 44
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 44 (same as scenes 38-43's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 45: {  // Suzanne (see build_suzanne_gpu's comment)
								build_suzanne_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 45
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 45 (same as scenes 38-44's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 46: {  // Nefertiti Bust (see build_nefertiti_gpu's comment)
								build_nefertiti_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 46
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 46 (same as scenes 38-45's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 47: {  // Horse (see build_horse_gpu's comment)
								build_horse_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 47
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 47 (same as scenes 38-46's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 48: {  // Cheburashka (see build_cheburashka_gpu's comment)
								build_cheburashka_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 48
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 48 (same as scenes 38-47's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 49: {  // Trophy Room (see build_trophy_room_gpu's comment)
								build_trophy_room_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.3f, 14.0f);
								const float3 lookat   = make_float3(0.0f, 0.9f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 34.0f, aspect, 1.0f, camera_params);  // 34: matches CPU CameraConfig row for scene 49
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 49 (same as scenes 38-48's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}

							case 50: {  // Glass Dragon (see build_glass_dragon_gpu's comment)
								build_glass_dragon_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);  // 35: matches CPU CameraConfig row for scene 50
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 50 (same as scenes 38-49's).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								}
								break;
							}


							case 51: {  // Beast (see build_beast_gpu's comment)
								build_beast_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 52: {  // VW Beetle (see build_beetle_gpu's comment)
								build_beetle_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 16.0f);
								const float3 lookat   = make_float3(0.0f, 1.2f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 53: {  // VW Beetle (alternate mesh) (see build_beetle_alt_gpu's comment)
								build_beetle_alt_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 16.0f);
								const float3 lookat   = make_float3(0.0f, 1.2f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 54: {  // Bimba (see build_bimba_gpu's comment)
								build_bimba_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 55: {  // Cow (see build_cow_gpu's comment)
								build_cow_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 56: {  // Fandisk (see build_fandisk_gpu's comment)
								build_fandisk_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 57: {  // Homer (see build_homer_gpu's comment)
								build_homer_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 58: {  // Igea (see build_igea_gpu's comment)
								build_igea_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 5.0f, 3.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 59: {  // Max Planck (see build_max_planck_gpu's comment)
								build_max_planck_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, -7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 60: {  // Ogre (see build_ogre_gpu's comment)
								build_ogre_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 3.0f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.5f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 61: {  // Rocker Arm (see build_rocker_arm_gpu's comment)
								build_rocker_arm_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 0.0f, 2.5f, 7.0f);
								const float3 lookat   = make_float3(0.0f, 1.2f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 35.0f, aspect, 1.0f, camera_params);
								if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.08f);
								break;
							}

							case 62: {  // Crytek Sponza (see build_sponza_gpu's comment)
								build_sponza_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, -800.0f, 300.0f, 0.0f);
								const float3 lookat   = make_float3(800.0f, 300.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 70.0f, aspect, 1.0f, camera_params);  // 70: matches CPU CameraConfig row for scene 62
								if (out_camera_extra) {
									// Matches CPU build_sponza_sky()'s solid-color sky_light(0.65,0.78,0.95).
									out_camera_extra->backgroundColor = make_float3(0.65f, 0.78f, 0.95f);
								}
								break;
							}

							case 63: {  // Amazon Lumberyard Bistro, Exterior (see build_bistro_exterior_gpu's comment)
								build_bistro_exterior_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 1500.0f, 700.0f, 2000.0f);
								const float3 lookat   = make_float3(4000.0f, 700.0f, 2000.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 60.0f, aspect, 1.0f, camera_params);  // 60: matches CPU CameraConfig row for scene 63
								if (out_camera_extra) {
									// Matches CPU build_bistro_exterior_sky()'s solid-color sky_light(0.55,0.72,0.95).
									out_camera_extra->backgroundColor = make_float3(0.55f, 0.72f, 0.95f);
								}
								break;
							}

							case 64: {  // Rungholt (see build_rungholt_gpu's comment)
								build_rungholt_gpu(scene);
								const float3 lookfrom = resolve_fixed_lookfrom(force_camera_override, cam_x, cam_y, cam_z, 400.0f, 300.0f, 400.0f);
								const float3 lookat   = make_float3(0.0f, 40.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 45.0f, aspect, 1.0f, camera_params);  // 45: matches CPU CameraConfig row for scene 64
								if (out_camera_extra) {
									// Matches CPU build_rungholt_sky()'s solid-color sky_light(0.55,0.72,0.95).
									out_camera_extra->backgroundColor = make_float3(0.55f, 0.72f, 0.95f);
								}
								break;
							}

							default: {
									// A scene loaded from a .pbrt file has no case of its
									// own - it is not known at compile time. The CPU
									// registry already resolved which file this id is, so
									// ask it rather than re-deriving the directory search
									// order here and risking the two sides disagreeing
									// about which file is scene 65.
									const char* pbrtPath = cpu_scene_pbrt_path_by_id(scene_id);
									if (pbrtPath && pbrtPath[0] != '\0') {
										return build_loaded_pbrt_scene(
											pbrtPath, scene, camera_params,
											image_width, image_height,
											cam_x, cam_y, cam_z, force_camera_override,
											out_camera_extra);
									}

									const char* name = cpu_scene_name_by_id(scene_id);
									if (name && name[0] != '\0') {
										std::cerr << "[OptiX] Scene '" << name
											  << "' (id=" << scene_id << ") is not implemented for GPU rendering.\n"
											  << "[OptiX] Use CPU renderer for this scene.\n";
									} else {
										std::cerr << "[OptiX] Unknown scene id=" << scene_id << ".\n";
									}
									return false;
								}
							}

							// Every static sphere across every scene case above is built via
							// `SphereData s{};` (or equivalent), which value-initializes
							// center1 to (0,0,0) - "safe" only as long as ray_time stays
							// provably 0.0f for that scene (see SphereData::center1's doc
							// comment in optix_types.h). But ray_time is 0.0f only when
							// motionBlurEnabled is false, and that flag is auto-detected
							// below in OptiXRenderer::buildScene() by checking whether ANY
							// sphere's center1 differs from its center - which a merely
							// *unset* center1 satisfies just as well as a real moving
							// sphere does, for any static sphere not centered at the exact
							// origin (e.g. every scene's ground sphere). That falsely
							// enabled motion blur for every such scene, randomizing every
							// sphere's ray-time-interpolated position per sample - the
							// actual cause of scenes 19/24/31 (etc. - any sphere-using scene
							// without an explicit light source) rendering as near-black
							// noise on GPU: their camera rays were hitting spheres at
							// effectively random positions instead of their real ones,
							// almost never reaching the open background.
							//
							// build_bouncing_spheres() (scene 1) is the only builder that
							// wants real motion and already explicitly sets center1 on
							// every sphere it creates (to itself for static ones, to a real
							// bounce target for moving ones) - this loop only touches
							// spheres that never got an explicit center1 at all, so it
							// can't undo that.
							for (auto& s : scene.spheres) {
								if (s.center1.x == 0.0f && s.center1.y == 0.0f && s.center1.z == 0.0f &&
									!(s.center.x == 0.0f && s.center.y == 0.0f && s.center.z == 0.0f)) {
									s.center1 = s.center;
								}
							}

							return true;
						}

