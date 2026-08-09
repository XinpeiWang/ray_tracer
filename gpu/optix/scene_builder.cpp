// Scene Builder Implementation
// Converts shared scene definitions to OptiX geometry

#include "scene_builder.h"
#include "optix_math_helpers.h"
#include "../../src/shared/scene_descriptor.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <random>
#include <string>

#include "../../src/shared/conductor_data.h"
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
			scene.isLightSphere.push_back(false); // false = quad
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
			scene.materials.push_back({
				MaterialType::DiffuseLight,
				make_float3(0.0f, 0.0f, 0.0f),
				0.0f, 0.0f,
				make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b))
			});
		} else {
			scene.materials.push_back({
				MaterialType::Lambertian,
				make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)),
				0.0f, 0.0f,
				make_float3(0.0f, 0.0f, 0.0f)
			});
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
			scene.isLightSphere.push_back(false);
		}
	}

	// Glass sphere
	const int mat_glass = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Dielectric,
		make_float3(1.0f, 1.0f, 1.0f),  // albedo (unused for dielectric)
		0.0f,
		static_cast<float>(kGlassSphere.glass_ior),
		make_float3(0.0f, 0.0f, 0.0f)
	});
	SphereData glass_sphere{};
	glass_sphere.center = make_float3(
		static_cast<float>(kGlassSphere.center.x), static_cast<float>(kGlassSphere.center.y), static_cast<float>(kGlassSphere.center.z));
	glass_sphere.radius = static_cast<float>(kGlassSphere.radius);
	glass_sphere.materialIdx = mat_glass;
	scene.spheres.push_back(glass_sphere);
	// Glass is never emissive - no lightIndices entry.

	// White rotated box
	const int mat_box = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(static_cast<float>(kBox.color.r), static_cast<float>(kBox.color.g), static_cast<float>(kBox.color.b)),
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)
	});
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
    const int mat_ground = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.2f, 0.2f, 0.2f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Area light quad material
    constexpr float kRMSLightIntensity = 6.0f;
    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kRMSLightIntensity, kRMSLightIntensity, kRMSLightIntensity) });

    // Five rough-metal sphere materials: roughness 0.05, 0.2, 0.4, 0.6, 0.8
    const float roughnesses[5] = { 0.05f, 0.2f, 0.4f, 0.6f, 0.8f };
    int mat_metal[5];
    for (int i = 0; i < 5; ++i) {
        mat_metal[i] = safe_cast_to_int(scene.materials.size());
        scene.materials.push_back({ MaterialType::Metal, make_float3(0.95f, 0.85f, 0.55f),
                                     roughnesses[i], 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
    scene.isLightSphere.push_back(false);
}

/// @brief Adds the 5 standard Cornell-box walls (red/white/green/white/white)
/// and the main ceiling light - cornell_box_data::kQuads[0..5], skipping the
/// dim warm accent light at index 6 - to scene. Shared by every "Cornell
/// family" GPU builder (scenes 10-13, 15-17) that keeps the standard box
/// shell but swaps in different sphere/box materials, mirroring CPU's
/// add_cornell_walls_and_main_light() in scenes_book.h. Scene 0's
/// build_cornell_box() above doesn't use this - it needs the accent light
/// too, so it loops over the full kQuads itself.
static void add_cornell_walls_and_main_light(SceneData& scene) {
    using namespace cornell_box_data;
    for (int i = 0; i < 6; ++i) {
        const QuadSpec& q = kQuads[i];
        const int mat = safe_cast_to_int(scene.materials.size());
        if (q.is_light) {
            scene.materials.push_back({
                MaterialType::DiffuseLight,
                make_float3(0.0f, 0.0f, 0.0f),
                0.0f, 0.0f,
                make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b))
            });
        } else {
            scene.materials.push_back({
                MaterialType::Lambertian,
                make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)),
                0.0f, 0.0f,
                make_float3(0.0f, 0.0f, 0.0f)
            });
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
            scene.isLightSphere.push_back(false);
        }
    }
}

/// @brief Build Cornell Rough Metal scene (scene 10)
/// Matches CPU build_cornell_rough_metal(): same walls/light, rough aluminum box + rough gold sphere
static void build_cornell_rough_metal(SceneData& scene) {
    add_cornell_walls_and_main_light(scene);

    // Rough aluminum box material (roughness 0.15)
    const int mat_alum = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.85f, 0.88f), 0.15f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Rough gold sphere material (roughness 0.3)
    const int mat_gold = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Metal, make_float3(0.95f, 0.78f, 0.28f), 0.3f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

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
    const int mat_gold = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::Conductor;
        md.fuzz  = 0.1f;   // roughness; alpha = sqrt(0.1)
        md.eta_c = make_float3(kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b);
        md.k_c   = make_float3(kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b);
        scene.materials.push_back(md);
    }

    // Aluminium box material (conductor, roughness 0.05 -- polished aluminium)
    const int mat_alum = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::Conductor;
        md.fuzz  = 0.05f;
        md.eta_c = make_float3(kConductorAl.eta_r, kConductorAl.eta_g, kConductorAl.eta_b);
        md.k_c   = make_float3(kConductorAl.k_r,   kConductorAl.k_g,   kConductorAl.k_b);
        scene.materials.push_back(md);
    }

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
    const int mat_coated_blue = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type   = MaterialType::CoatedDiffuse;
        md.albedo = make_float3(0.2f, 0.3f, 0.9f);  // diffuse base colour
        md.fuzz   = 0.1f;                            // coat roughness (RoughnessToAlpha done in shader)
        md.ior    = 1.5f;                            // coat IOR (glass-like)
        scene.materials.push_back(md);
    }

    // Red coated-diffuse box (IOR 1.5, roughness 0.2)
    const int mat_coated_red = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type   = MaterialType::CoatedDiffuse;
        md.albedo = make_float3(0.8f, 0.1f, 0.1f);
        md.fuzz   = 0.2f;
        md.ior    = 1.5f;
        scene.materials.push_back(md);
    }

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
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // White diffuse box
    const int mat_box = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Thin-glass panel (IOR 1.5)
    const int mat_panel = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type = MaterialType::ThinDielectric;
        md.ior  = 1.5f;
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,0); q.u = make_float3(0,0,kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(0,0,-kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(213,554,227); q.u = make_float3(130,0,0); q.v = make_float3(0,0,105); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1); scene.isLightSphere.push_back(false); }
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

    // Thin-glass panel: vertical slab spanning box interior
    // Q=(100,0,200), u=(0,555,0), v=(355,0,0)  -- horizontal quad lying in xz plane rotated
    {
        QuadData q{};
        q.Q = make_float3(100.0f, 0.0f, 200.0f);
        q.u = make_float3(0.0f, 555.0f, 0.0f);
        q.v = make_float3(355.0f, 0.0f, 0.0f);
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
    const int mat_gold_lacquer = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::CoatedConductor;
        md.fuzz  = 0.1f;   // coat roughness
        md.ior   = 1.5f;   // coat IOR
        md.eta_c = make_float3(kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b);
        md.k_c   = make_float3(kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b);
        scene.materials.push_back(md);
    }

    // Lacquered-copper box (Cu conductor, IOR-1.5 coat, roughness 0.2)
    const int mat_copper_lacquer = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::CoatedConductor;
        md.fuzz  = 0.2f;
        md.ior   = 1.5f;
        md.eta_c = make_float3(kConductorCu.eta_r, kConductorCu.eta_g, kConductorCu.eta_b);
        md.k_c   = make_float3(kConductorCu.k_r,   kConductorCu.k_g,   kConductorCu.k_b);
        scene.materials.push_back(md);
    }

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
    const int mat_wax = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type    = MaterialType::DiffuseTransmission;
        md.albedo  = make_float3(0.6f, 0.5f, 0.3f);   // R: reflected diffuse color
        md.emission = make_float3(0.8f, 0.6f, 0.3f);  // T: transmitted diffuse color
        md.fuzz    = 0.0f;
        md.ior     = 0.0f;
        scene.materials.push_back(md);
    }

    // White diffuse box material (same albedo as the walls, own index)
    const int mat_box = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

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

    // Crystal sphere: NormalizedFresnelBxDF, IOR 1.5 (glass/crystal)
    const int mat_crystal = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type    = MaterialType::NormalizedFresnel;
        md.albedo  = make_float3(1.0f, 1.0f, 1.0f);  // unused -- weight computed from Fresnel
        md.ior     = 1.5f;
        md.fuzz    = 0.0f;
        md.emission = make_float3(0.0f, 0.0f, 0.0f);
        scene.materials.push_back(md);
    }

    // White diffuse box material (same albedo as the walls, own index)
    const int mat_box = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

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
    const int mat_rough_glass = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::RoughDielectric, make_float3(1.0f, 1.0f, 1.0f), 0.2f, kGlassIOR, make_float3(0.0f, 0.0f, 0.0f) });

    // White diffuse box material (same albedo as the walls, own index)
    const int mat_box = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

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
		scene.materials.push_back({
			MaterialType::Lambertian,
			make_float3(0.5f, 0.5f, 0.5f),
			0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f)
		});
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
				scene.materials.push_back({ MaterialType::Lambertian, albedo, 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
				center1 = make_float3(center.x, center.y + unit(rng) * 0.5f, center.z);
			} else if (choose_mat < 0.95f) {
				// Metal
				const float3 albedo = make_float3(0.5f + 0.5f * unit(rng), 0.5f + 0.5f * unit(rng), 0.5f + 0.5f * unit(rng));
				const float fuzz = 0.5f * unit(rng);
				mat_idx = safe_cast_to_int(scene.materials.size());
				scene.materials.push_back({ MaterialType::Metal, albedo, fuzz, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
			} else {
				// Glass
				mat_idx = safe_cast_to_int(scene.materials.size());
				scene.materials.push_back({ MaterialType::Dielectric, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, make_float3(0.0f, 0.0f, 0.0f) });
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
		const int mat1 = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Dielectric, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s1{};
		s1.center = make_float3(0.0f, 1.0f, 0.0f);
		s1.center1 = s1.center;
		s1.radius = 1.0f;
		s1.materialIdx = mat1;
		scene.spheres.push_back(s1);
	}
	{
		const int mat2 = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.4f, 0.2f, 0.1f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s2{};
		s2.center = make_float3(-4.0f, 1.0f, 0.0f);
		s2.center1 = s2.center;
		s2.radius = 1.0f;
		s2.materialIdx = mat2;
		scene.spheres.push_back(s2);
	}
	{
		const int mat3 = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Metal, make_float3(0.7f, 0.6f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s3{};
		s3.center = make_float3(4.0f, 1.0f, 0.0f);
		s3.center1 = s3.center;
		s3.radius = 1.0f;
		s3.materialIdx = mat3;
		scene.spheres.push_back(s3);
	}
}

/**
 * Build checkered spheres scene (scene 2)
 * Two spheres with different albedos
 */
void build_checkered_spheres(SceneData& scene) {
	// Bottom sphere (darker)
	const int mat1 = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.2f, 0.3f, 0.1f),  // albedo
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)  // no emission
	});

	SphereData sphere1{};
	sphere1.center = make_float3(0.0f, -10.0f, 0.0f);
	sphere1.radius = 10.0f;
	sphere1.materialIdx = mat1;
	scene.spheres.push_back(sphere1);
	if (is_emissive(scene, mat1)) {
		scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
		scene.isLightSphere.push_back(true);
	}

	// Top sphere (lighter)
	const int mat2 = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.9f, 0.9f, 0.9f),  // albedo
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)  // no emission
	});

	SphereData sphere2{};
	sphere2.center = make_float3(0.0f, 10.0f, 0.0f);
	sphere2.radius = 10.0f;
	sphere2.materialIdx = mat2;
	scene.spheres.push_back(sphere2);
	if (is_emissive(scene, mat2)) {
		scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
		scene.isLightSphere.push_back(true);
	}
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
		scene.materials.push_back({
			MaterialType::Lambertian,
			make_float3(r, g, b),
			0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f)
		});

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
			scene.isLightSphere.push_back(false);
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
		scene.materials.push_back({
			MaterialType::Lambertian,
			make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)),
			0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f)
		});
		add_transformed_quad(scene,
			make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z)),
			make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z)),
			make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z)),
			mat);
	}

	// Sphere materials: white lambertian + blue-tinted fuzzy metal (matches CPU)
	const int mat_white_sphere = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	const int mat_metal_sphere = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 0.1f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

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
	light.distant.scale = 800000.0f;
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
	light.point.scale = 5000000.0f;
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
	g.scale = 4000000.0f;
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

// Shared overhead area light used by scenes 22/32 below. GPU has no
// background/miss-color model (unlike CPU, which lights these scenes purely
// via a flat sky background color - see build_scene()'s camera-only comment
// for scenes 22/32) - without this, either GPU scene would render fully
// black despite the camera itself working correctly.
static void add_overhead_area_light(SceneData& scene, float3 center, float halfSize, float intensity) {
	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
								 make_float3(intensity, intensity, intensity) });
	QuadData lq{};
	lq.Q = make_float3(center.x - halfSize, center.y, center.z - halfSize);
	lq.u = make_float3(2.0f * halfSize, 0.0f, 0.0f);
	lq.v = make_float3(0.0f, 0.0f, 2.0f * halfSize);
	const float3 lc = cross(lq.u, lq.v);
	lq.w = lc;
	lq.normal = normalize(lc);
	lq.D = dot(lq.normal, lq.Q);
	lq.materialIdx = mat_light;
	scene.quads.push_back(lq);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.isLightSphere.push_back(false);
}

/// @brief Scene 22: Depth of Field. Matches CPU build_depth_of_field() in
/// spirit (ground + a row of spheres spanning near/far of the focus plane to
/// show defocus blur) - simplified to solid-color materials since GPU has no
/// checker/procedural texture support.
static void build_depth_of_field_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
		const int mat = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = kinds[i];
		m.albedo = colors[i];
		m.fuzz = (kinds[i] == MaterialType::Metal) ? 0.05f : 0.0f;
		m.ior = (kinds[i] == MaterialType::Dielectric) ? 1.5f : 0.0f;
		scene.materials.push_back(m);
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
	add_overhead_area_light(scene, make_float3(0.0f, 15.0f, 0.0f), 8.0f, 1.0f);
}

/// @brief Scene 32: Orthographic Camera. Matches CPU build_ortho_camera_scene()
/// in spirit (ground + a row of colored lambertian spheres) - simplified to
/// solid-color materials, same reasoning as scene 22 above.
static void build_ortho_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, colors[i], 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3((i - 2) * 2.5f, 1.0f, 0.0f);
		s.radius = 1.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}
	add_overhead_area_light(scene, make_float3(0.0f, 15.0f, 0.0f), 8.0f, 4.0f);
}

/// @brief Scene 33: Spherical Camera. Matches CPU build_spherical_camera_scene()
/// in spirit (ground + a ring of colored spheres + one emissive sphere) -
/// self-illuminating, needs no extra light unlike scenes 22/32 above.
static void build_spherical_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, col, 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(4.0f * cosf(ang), 1.0f, 4.0f * sinf(ang));
		s.radius = 0.8f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Central emissive sphere, matches CPU's central diffuse_light sphere.
	const int mat_light = safe_cast_to_int(scene.materials.size());
	constexpr float kSphericalLightIntensity = 8.0f;
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
								 make_float3(kSphericalLightIntensity, kSphericalLightIntensity, kSphericalLightIntensity) });
	SphereData lightSphere{};
	lightSphere.center = make_float3(0.0f, 3.0f, 0.0f);
	lightSphere.radius = 1.0f;
	lightSphere.materialIdx = mat_light;
	scene.spheres.push_back(lightSphere);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.isLightSphere.push_back(true);
}

/// @brief Scene 36: Realistic Camera. Matches CPU build_realistic_camera_scene()
/// (ground + 5 colored spheres at increasing depth to show bokeh + one area
/// light) - ground uses a flat gray instead of CPU's checker_texture, matching
/// this file's established "no procedural textures on GPU" simplification
/// used elsewhere (e.g. build_triangle_mesh_scene_gpu).
static void build_realistic_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, sphere_colors[i], 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(0.0f, 1.0f, z);
		s.radius = 0.8f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(6.0f, 6.0f, 6.0f) });
	SphereData lightSphere{};
	lightSphere.center = make_float3(0.0f, 8.0f, 5.0f);
	lightSphere.radius = 2.0f;
	lightSphere.materialIdx = mat_light;
	scene.spheres.push_back(lightSphere);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.isLightSphere.push_back(true);
}

/// @brief Scene 24: HDRI Sky. Matches CPU build_hdri_sky_world() (ground +
/// three spheres showcasing lambertian/metal/dielectric under sky light).
/// The CPU "HDRI" is actually a flat-color sky_light in practice (see
/// GpuCameraParams::backgroundColor's comment) - the caller sets that
/// separately, this only builds the ground+spheres geometry.
static void build_hdri_sky_world_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.4f, 0.4f, 0.4f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_lambertian = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.7f, 0.3f, 0.2f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData s1{};
	s1.center = make_float3(-3.0f, 1.0f, 0.0f);
	s1.radius = 1.0f;
	s1.materialIdx = mat_lambertian;
	scene.spheres.push_back(s1);

	const int mat_metal = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 0.05f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData s2{};
	s2.center = make_float3(0.0f, 1.0f, 0.0f);
	s2.radius = 1.0f;
	s2.materialIdx = mat_metal;
	scene.spheres.push_back(s2);

	const int mat_glass = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Dielectric, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, make_float3(0.0f, 0.0f, 0.0f) });
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
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_metal_sphere = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 0.05f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back

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
		scene.materials.push_back({
			MaterialType::Lambertian,
			make_float3(static_cast<float>(q.color.r), static_cast<float>(q.color.g), static_cast<float>(q.color.b)),
			0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f)
		});
		add_transformed_quad(scene,
			make_float3(static_cast<float>(q.Q.x), static_cast<float>(q.Q.y), static_cast<float>(q.Q.z)),
			make_float3(static_cast<float>(q.u.x), static_cast<float>(q.u.y), static_cast<float>(q.u.z)),
			make_float3(static_cast<float>(q.v.x), static_cast<float>(q.v.y), static_cast<float>(q.v.z)),
			mat);
	}

	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(7.0f, 7.0f, 7.0f) });
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
		scene.isLightSphere.push_back(false);
	}

	// Two medium spheres approximating CPU's two rotated boxes (centered
	// roughly where box1 [265,0,295]+165/2 and box2 [130,0,65]+82.5 sit).
	const int mat_medium_dark = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.01f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_medium_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 0.01f, make_float3(0.0f, 0.0f, 0.0f) });

	SphereData m1{}; m1.center = make_float3(347.0f, 165.0f, 377.0f); m1.radius = 115.0f; m1.materialIdx = mat_medium_dark;
	scene.spheres.push_back(m1);
	SphereData m2{}; m2.center = make_float3(212.0f, 82.0f, 147.0f); m2.radius = 82.0f; m2.materialIdx = mat_medium_white;
	scene.spheres.push_back(m2);
}

/// @brief Scene 30: Homogeneous Medium. Matches CPU
/// build_homogeneous_medium_scene() exactly (full Cornell box + a single
/// medium sphere at the box's center, radius 400, density 0.005, HG g=0.3).
static void build_homogeneous_medium_scene_gpu(SceneData& scene) {
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(15.0f, 15.0f, 15.0f) });

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
		scene.isLightSphere.push_back(false);
	}

	const int mat_medium = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(0.8f, 0.9f, 1.0f), 0.3f, 0.005f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData fog{};
	fog.center = make_float3(277.5f, 277.5f, 277.5f);
	fog.radius = 400.0f;
	fog.materialIdx = mat_medium;
	scene.spheres.push_back(fog);
}

/// @brief Scene 31: Cloud Medium. Matches CPU build_cloud_medium_scene()
/// (ground + one medium sphere, density 0.8, HG g=0.05) - the CPU's Perlin-
/// noise density texture is dead code there too (constructed but never
/// actually used by the constant_medium call, which passes a constant
/// density), so this is a faithful, not simplified, port.
static void build_cloud_medium_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.4f, 0.5f, 0.3f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_medium = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(1.0f, 1.0f, 1.0f), 0.05f, 0.8f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData cloud{};
	cloud.center = make_float3(0.0f, 3.0f, 0.0f);
	cloud.radius = 3.0f;
	cloud.materialIdx = mat_medium;
	scene.spheres.push_back(cloud);
}

/// @brief Scene 23: Bilinear Patch Scene. Matches CPU build_bilinear_patch_scene()
/// exactly - standard Cornell box + two genuinely curved (non-planar) metal
/// bilinear patches (see optix_intersection_bilinear_patch.h). Unlike scene 7's
/// medium boxes, these are NOT approximated as another shape - bilinear
/// patches have their own GPU geometry type.
static void build_bilinear_patch_scene_gpu(SceneData& scene) {
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(15.0f, 15.0f, 15.0f) });

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
		scene.isLightSphere.push_back(false);
	}

	// Patch 1: classic hyperbolic paraboloid saddle, gold metal
	const int mat_gold = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.7f, 0.3f), 0.05f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	{
		BilinearPatchData p{};
		p.p00 = make_float3(150.0f, 80.0f, 200.0f);
		p.p10 = make_float3(400.0f, 50.0f, 200.0f);
		p.p01 = make_float3(150.0f, 50.0f, 400.0f);
		p.p11 = make_float3(400.0f, 80.0f, 400.0f);
		p.materialIdx = mat_gold;
		scene.bilinearPatches.push_back(p);
	}

	// Patch 2: curved ramp (linear in u, curved in v), blue metal
	const int mat_blue = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.2f, 0.4f, 0.8f), 0.1f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.05f, 0.05f, 0.06f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Hair MaterialData reuse: albedo=sigma_a(r,g,b), fuzz=beta_m, ior=eta(1.55),
	// eta_c.x=beta_n, eta_c.y=alpha_deg.
	struct HairSphere { float3 center; float3 sigma_a; float beta_m; float beta_n; float alpha_deg; };
	const HairSphere hairs[5] = {
		{ make_float3(-2.5f, 1.0f, 0.0f), make_float3(0.06f, 0.10f, 0.20f), 0.25f, 0.25f, 2.0f }, // dark brown
		{ make_float3(-0.8f, 1.0f, 0.3f), make_float3(0.01f, 0.015f, 0.03f), 0.30f, 0.30f, 2.0f }, // blonde
		{ make_float3(0.9f, 1.0f, -0.3f), make_float3(0.02f, 0.08f, 0.18f), 0.20f, 0.20f, 3.0f }, // auburn
		{ make_float3(2.5f, 1.0f, 0.0f), make_float3(0.001f, 0.001f, 0.002f), 0.45f, 0.45f, 1.0f }, // white/silver fur
		{ make_float3(0.0f, 1.0f, 1.8f), make_float3(0.50f, 0.55f, 0.60f), 0.15f, 0.15f, 2.0f }, // fine black fur
	};
	for (const auto& h : hairs) {
		const int mat_idx = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = MaterialType::Hair;
		m.albedo = h.sigma_a;
		m.fuzz = h.beta_m;
		m.ior = 1.55f;  // fiber eta, matches CPU hair_material's default
		m.eta_c = make_float3(h.beta_n, h.alpha_deg, 0.0f);
		scene.materials.push_back(m);

		SphereData s{}; s.center = h.center; s.radius = 1.0f; s.materialIdx = mat_idx;
		scene.spheres.push_back(s);
	}
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
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.3f, 0.3f, 0.3f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{}; ground.center = make_float3(0.0f, -1000.0f, 0.0f); ground.radius = 1000.0f; ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_measured = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.7f, 0.5f, 0.3f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	for (int i = -2; i <= 2; ++i) {
		SphereData s{}; s.center = make_float3(static_cast<float>(i) * 2.5f, 1.0f, 0.0f); s.radius = 1.0f; s.materialIdx = mat_measured;
		scene.spheres.push_back(s);
	}

	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(8.0f, 8.0f, 8.0f) });
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 1.5f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.isLightSphere.push_back(true);
}

/// @brief Scene 37: Triangle Mesh. Matches CPU build_triangle_mesh_scene()
/// exactly - same golden-ratio icosahedron vertex/face construction, same
/// gold metal material, same ground/light placement. GPU MaterialType has no
/// procedural-texture support (no scene has ever needed it - every prior
/// checker-textured CPU scene ported to GPU this session used a flat
/// approximation instead), so the ground's CPU checker texture becomes a
/// solid mid-gray here.
static void build_triangle_mesh_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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

	const int mat_mesh = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.6f, 0.2f), 0.15f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	for (const auto& f : faces) {
		TriangleData t{};
		t.p0 = verts[f[0]];
		t.p1 = verts[f[1]];
		t.p2 = verts[f[2]];
		t.materialIdx = mat_mesh;
		scene.triangles.push_back(t);
	}

	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(6.0f, 6.0f, 6.0f) });
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.isLightSphere.push_back(true);
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
/// One piece is deliberately NOT ported yet and uses a plain-glass
/// approximation instead: the two constant_medium fog spheres (small blue
/// fog + giant whole-scene haze). CPU achieves this by adding the SAME
/// dielectric boundary sphere to the world twice (once directly, once
/// wrapped in constant_medium), letting whichever hits closer each bounce
/// win. GPU's MaterialType::Medium is a standalone material, mutually
/// exclusive with Dielectric, so this needs its own material type - the
/// small fog sphere is approximated as plain dielectric glass for now
/// (visually reasonable - it's still a glass sphere, just without the
/// interior fog tint); the giant radius-5000 whole-scene haze sphere is
/// skipped entirely (CPU's own version is barely visible - an extremely
/// subtle atmospheric tint at density 0.0001 - and naively wrapping the
/// whole scene in glass would look nothing like it).
/// Uses its own fixed-seed RNG for the ground/sphere-cluster randomization,
/// like every other procedural GPU scene (e.g. build_bouncing_spheres) -
/// not intended to pixel-match CPU's independently-seeded layout.
void build_final_scene_gpu(SceneData& scene) {
	std::mt19937 rng(8u);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	// Ground: 20x20 grid of boxes with randomized height, matching CPU's
	// loop bounds/spacing (w=100, x0/z0 in [-1000,1000)) exactly.
	{
		const int mat_ground = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.48f, 0.83f, 0.53f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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
		const int mat_light = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(7.0f, 7.0f, 7.0f) });
		add_transformed_quad(scene, make_float3(123.0f, 554.0f, 147.0f), make_float3(300.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 265.0f), mat_light);
	}

	// Moving sphere (real GPU motion blur - center1 differs from center).
	{
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.7f, 0.3f, 0.1f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(400.0f, 400.0f, 200.0f);
		s.center1 = make_float3(430.0f, 400.0f, 200.0f);  // center + (30,0,0)
		s.radius = 50.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Dielectric (glass) sphere.
	{
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Dielectric, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(260.0f, 150.0f, 45.0f);
		s.center1 = s.center;
		s.radius = 50.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Metal sphere.
	{
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 1.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(0.0f, 150.0f, 145.0f);
		s.center1 = s.center;
		s.radius = 50.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Small fog-sphere placeholder (plain dielectric for now - see this
	// function's header comment; Piece 3 upgrades this to a real
	// dielectric+medium combined material).
	{
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Dielectric, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(360.0f, 150.0f, 145.0f);
		s.center1 = s.center;
		s.radius = 70.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}
	// Giant whole-scene haze sphere intentionally omitted - see this
	// function's header comment.

	// Earth-image-texture sphere. Falls back to CPU's own solid-cyan
	// missing-texture color if earthmap.jpg can't be found (see
	// load_image_texture_gpu's comment) - this matches CPU's behavior
	// exactly rather than being a GPU-specific limitation.
	{
		const int earthTexIdx = load_image_texture_gpu(scene, "earthmap.jpg");
		const int mat = safe_cast_to_int(scene.materials.size());
		MaterialData m{ MaterialType::Lambertian, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) };
		m.textureIdx = earthTexIdx;
		scene.materials.push_back(m);
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
		MaterialData m{ MaterialType::Lambertian, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) };
		m.textureIdx = noiseTexIdx;
		scene.materials.push_back(m);
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
		const int mat_white = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
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

/// @brief Build a scene and configure the camera
/// @param scene_id Scene identifier (0 = Cornell Box)
/// @param image_width Output image width in pixels
/// @param image_height Output image height in pixels
/// @param scene Output scene data to populate
/// @param camera_params Output camera parameters array [origin(3), lower_left(3), horizontal(3), vertical(3)]
/// @return true if scene was built successfully, false for unknown scene_id
bool build_scene(
	const int scene_id,
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

	// Clear previous scene data
	scene.spheres.clear();
	scene.quads.clear();
	scene.bilinearPatches.clear();
	scene.triangles.clear();
	scene.materials.clear();

	// Build requested scene
	switch (scene_id) {
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
						const float3 lookfrom = force_camera_override
							? make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z))
							: make_float3(13.0f, 2.0f, 3.0f);
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
						const float3 lookfrom = force_camera_override
							? make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z))
							: make_float3(13.0f, 2.0f, 3.0f);
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, 1.0f, camera_params);

						// Flat light-blue background, matching CPU registry's
						// bg=(0.70,0.80,1.00) for this scene (see
						// GpuCameraParams::backgroundColor's comment) - this was
						// previously left at the zero-init default (black).
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

						// This scene has no emissive geometry at all (5 plain
						// Lambertian quads) - matches CPU registry's
						// bg=(0.70,0.80,1.00) for scene 5 (see
						// GpuCameraParams::backgroundColor's comment). Without
						// this the scene rendered totally black on GPU: no
						// lights meant every path's only possible radiance was
						// this background color, and it was never set.
						if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(0.70f, 0.80f, 1.00f);
					}
					break;

				case 8:  // Final Scene (see build_final_scene_gpu's own comment
						 // for what's ported vs. placeholder-approximated)
					build_final_scene_gpu(scene);
					{
						const float3 lookfrom = make_float3(478.0f, 278.0f, -600.0f);
						const float3 lookat = make_float3(278.0f, 278.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
						build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);
						// backgroundColor left at zero-init (matches CPU bg=(0,0,0)) -
						// this scene has a real area light (the light quad above).
					}
					break;

				case 9: {  // Rough Metal Spheres (GGX)
										build_rough_metal_spheres(scene);

										// Camera: vfov=35, lookfrom=(cam_x,cam_y,cam_z), lookat=(0,1,0)
										const float3 lookfrom9 = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
										const float3 lookat9   = make_float3(0.0f, 1.0f, 0.0f);
										const float3 vup9      = make_float3(0.0f, 1.0f, 0.0f);
										const float aspect9    = static_cast<float>(image_width) / static_cast<float>(image_height);
										build_pinhole_camera_params(lookfrom9, lookat9, vup9, 35.0f, aspect9, 1.0f, camera_params);
										break;
									}

									case 10:  // Cornell Rough Metal (GGX)
								build_cornell_rough_metal(scene);

								// Same camera as Cornell Box (lookat center of box)
								// (camera block below handles this)
								// fallthrough intentional -- camera identical to case 11
								goto cornell_box_camera;

							case 11:  // Cornell Rough Glass (GGX)
											build_cornell_rough_glass(scene);
											cornell_box_camera:

											case 12:  // Cornell Conductor (GGX + complex Fresnel, pbrt-v4 ConductorBxDF)
												if (scene_id == 12) build_cornell_conductor(scene);
												// fallthrough

												case 13:  // Cornell Coated Diffuse (pbrt-v4 CoatedDiffuseBxDF)
												if (scene_id == 13) build_cornell_coated_diffuse(scene);
												// fallthrough

													case 14:  // Cornell Thin Glass (pbrt-v4 ThinDielectricBxDF)
													if (scene_id == 14) build_cornell_thin_glass(scene);
													// fallthrough

														case 15:  // Cornell Coated Conductor (pbrt-v4 CoatedConductorBxDF)
														if (scene_id == 15) build_cornell_coated_conductor(scene);
														// fallthrough

														case 16:  // Cornell Wax Slab (pbrt-v4 DiffuseTransmissionBxDF)
														if (scene_id == 16) build_cornell_wax_slab(scene);
														// fallthrough

														case 17:  // Cornell Crystal (pbrt-v4 NormalizedFresnelBxDF)
														if (scene_id == 17) build_cornell_crystal(scene);
														// fallthrough

														case 25:  // Spotlight Cornell (pbrt-v4 SpotLight)
														if (scene_id == 25) build_spotlight_cornell_gpu(scene);
														// fallthrough

														case 26:  // Distant Light Cornell (pbrt-v4 DistantLight)
														if (scene_id == 26) build_distant_light_cornell_gpu(scene);
														// fallthrough

														case 27:  // Point Light Cornell (pbrt-v4 PointLight)
														if (scene_id == 27) build_point_light_cornell_gpu(scene);
														// fallthrough

														case 28:  // Goniometric Light Cornell (pbrt-v4 GoniometricLight)
														if (scene_id == 28) build_goniometric_cornell_gpu(scene);
														// fallthrough

														case 29:  // Projection Light Cornell (pbrt-v4 ProjectionLight)
														if (scene_id == 29) build_projection_cornell_gpu(scene);
														// fallthrough

														case 35:  // Portal Infinite Light (pbrt-v4 PortalImageInfiniteLight)
														if (scene_id == 35) {
															build_portal_light_scene_gpu(scene);
															if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(1.0f, 1.2f, 1.5f);
														}
														// fallthrough

														case 7:  // Cornell Smoke (constant_medium)
														if (scene_id == 7) build_cornell_smoke_gpu(scene);
														// fallthrough

														case 30:  // Homogeneous Medium (constant_medium, HG g=0.3)
														if (scene_id == 30) build_homogeneous_medium_scene_gpu(scene);
														// fallthrough

														case 23:  // Bilinear Patch Scene (pbrt-v4 BilinearPatch shape)
														if (scene_id == 23) build_bilinear_patch_scene_gpu(scene);
													{
									const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
									const float3 lookat = make_float3(278.0f, 278.0f, 278.0f);
									const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
									const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
									build_pinhole_camera_params(lookfrom, lookat, vup, 40.0f, aspect, 1.0f, camera_params);
								}
								break;

							case 22: {  // Depth of Field (thin-lens perspective camera)
								build_depth_of_field_gpu(scene);
								constexpr float kPi = 3.14159265358979323846f;
								const float3 lookfrom = make_float3(0.0f, 2.0f, 9.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								constexpr float defocus_angle = 10.0f;   // matches CPU CameraConfig row for scene 22
								constexpr float focus_dist    = 9.0f;    // ditto
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								float3 u, v;
								build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, focus_dist, camera_params, &u, &v);

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
								}
								break;
							}

							case 32: {  // Orthographic Camera (parallel projection)
								build_ortho_camera_scene_gpu(scene);
								const float3 lookfrom = make_float3(0.0f, 3.0f, 12.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								// compute_screen_window-equivalent aspect-correct default, then
								// scaled x8 - matches CPU's setup_camera lambda for scene 32.
								float xmin, xmax, ymin, ymax;
								if (aspect >= 1.0f) { xmin = -aspect; xmax = aspect; ymin = -1.0f; ymax = 1.0f; }
								else                { xmin = -1.0f; xmax = 1.0f; ymin = -1.0f / aspect; ymax = 1.0f / aspect; }
								constexpr float kScreenScale = 8.0f;
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
								}
								break;
							}

							case 33: {  // Spherical (equirectangular) Camera
								build_spherical_camera_scene_gpu(scene);
								// Matches CPU: SphericalCamera constructed with no camera_to_world
								// arg, defaulting to identity - camera at world origin, world-axis
								// basis (see build_spherical_camera_scene_gpu's comment on why the
								// ground is placed below y=0 to accommodate this).
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, zero);
								pack_float3(camera_params, 3, zero);
								pack_float3(camera_params, 6, zero);
								pack_float3(camera_params, 9, zero);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Spherical;
									out_camera_extra->origin = zero;
									out_camera_extra->su = make_float3(1.0f, 0.0f, 0.0f);
									out_camera_extra->sv = make_float3(0.0f, 1.0f, 0.0f);
									out_camera_extra->sw = make_float3(0.0f, 0.0f, 1.0f);
								}
								break;
							}

							case 36: {  // Realistic Camera (pbrt-v4 multi-element lens)
								build_realistic_camera_scene_gpu(scene);
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, zero);
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
										0.0f, 2.0f, -2.0f,   // from
										0.0f, 1.0f,  5.0f,   // to
										0.0f, 1.0f,  0.0f    // up
									);
									RealisticCamera<float> realCam(ctw, 18.0f, 12.0f, 7.0f, 8.0f, lens, 512);

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
								const float3 lookfrom = make_float3(0.0f, 2.0f, 10.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 30.0f, aspect, 1.0f, camera_params);  // 30: matches CPU CameraConfig row for scene 24

								if (out_camera_extra) {
									// Matches CPU build_hdri_sky()'s solid-color sky_light(0.3,0.6,1.0)
									// (see GpuCameraParams::backgroundColor's comment for why this is a
									// flat color, not an importance-sampled environment map).
									out_camera_extra->backgroundColor = make_float3(0.3f, 0.6f, 1.0f);
								}
								break;
							}

							case 31: {  // Cloud Medium (constant_medium, HG g=0.05)
								build_cloud_medium_scene_gpu(scene);
								const float3 lookfrom = make_float3(0.0f, 5.0f, 20.0f);
								const float3 lookat   = make_float3(0.0f, 2.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 20.0f, aspect, 1.0f, camera_params);  // 20: matches CPU CameraConfig row for scene 31
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 31 - this is the scene's
									// ONLY light source (no emissive geometry), so a missing/black
									// background here means zero illumination anywhere in the image.
									out_camera_extra->backgroundColor = make_float3(0.5f, 0.7f, 1.0f);
								}
								break;
							}

							case 19: {  // Hair Fibers (pbrt-v4 HairBxDF)
								build_hair_fibers_gpu(scene);
								const float3 lookfrom = make_float3(0.0f, 2.0f, 8.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 30.0f, aspect, 1.0f, camera_params);  // 30: matches CPU CameraConfig row for scene 19
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 19 (dim ambient - the
									// only light source, no emissive geometry in this scene).
									out_camera_extra->backgroundColor = make_float3(0.05f, 0.05f, 0.07f);
								}
								break;
							}

							case 34: {  // Measured BRDF (see build_measured_brdf_scene_gpu's comment)
								build_measured_brdf_scene_gpu(scene);
								const float3 lookfrom = make_float3(0.0f, 3.0f, 12.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
								build_pinhole_camera_params(lookfrom, lookat, vup, 25.0f, aspect, 1.0f, camera_params);  // 25: matches CPU CameraConfig row for scene 34
								// backgroundColor left at zero-init (matches CPU bg=(0,0,0)) - this
								// scene has a real emissive light sphere, unlike scenes 19/31.
								break;
							}

							case 37: {  // Triangle Mesh (see build_triangle_mesh_scene_gpu's comment)
								build_triangle_mesh_scene_gpu(scene);
								const float3 lookfrom = make_float3(0.0f, 4.0f, 8.0f);
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

							default: {
									const SceneDesc* desc = find_scene_desc(scene_id);
									if (desc) {
										std::cerr << "[OptiX] Scene '" << desc->name
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

