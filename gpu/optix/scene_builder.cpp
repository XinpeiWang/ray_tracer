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
#include "../../src/shared/scene_descriptor.h"
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "../../src/shared/cameras.h"
#include "../../src/shared/fresnel.h"  // CauchyCoefficientsFromAbbe() - add_dispersive_dielectric/add_dispersive_rough_dielectric
#include "../../src/shared/animated_transform.h"  // AnimatedTransform - build_gpu_animated_camera_params()
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

	// Shared "process-lifetime cache, keyed by file path" helper for the
	// scene-loading caches below (image decode, OBJ/OBJ+MTL raw geometry,
	// pbrt parse) - each used to hand-roll its own find/build/emplace
	// boilerplate; this collapses all of them to one call, the same
	// "stop repeating the same alloc-or-reuse pattern" reasoning that
	// motivated wavefront_path_tracer.h's own reallocateDeviceBufferIfNeeded<T>()
	// template. `mutex` guards concurrent access - scene building only ever
	// happens on one thread today, but a per-cache mutex costs nothing here
	// and matches the precedent gpu/optix/pbrt_gpu_builder_materials.h's own
	// gammaMutex already sets ("for the realistic future case of building
	// more than one GPU scene concurrently within one process").
	// `builder(V&)` fills its argument and returns true on success; on
	// false, nothing is cached (a transient failure - a momentarily locked
	// or mid-save file - is retried fresh on the very next call instead of
	// failing permanently for the rest of the process) and this returns
	// nullptr. `builder` is responsible for its own diagnostics (a warning/
	// error print, since it only ever runs once per successful OR failed
	// key rather than once per call the way an outer, always-run print
	// would).
	// maxEntries: 0 (default) means unbounded, matching every existing
	// caller's behavior unchanged. A nonzero value bounds a cache whose
	// VALUES are large enough that unbounded growth is itself a concern
	// (e.g. a per-mesh transformed-triangle cache, where a long session
	// visiting several multi-million-triangle scenes could otherwise pin
	// every one of them in memory forever) - eviction here is a full clear
	// on overflow, not real LRU: simple, and adequate for a cache whose
	// point is avoiding repeated work within roughly the current working
	// set (the scene(s) actually in view), not remembering every scene ever
	// visited in a session.
	template <typename V, typename Builder>
	const V* get_or_build_cached(std::unordered_map<std::string, V>& cache, std::mutex& mutex,
								  const std::string& key, Builder&& builder, std::size_t maxEntries = 0) {
		std::lock_guard<std::mutex> lock(mutex);
		auto it = cache.find(key);
		if (it != cache.end()) return &it->second;
		V value{};
		if (!builder(value)) return nullptr;
		if (maxEntries > 0 && cache.size() >= maxEntries) cache.clear();
		return &cache.emplace(key, std::move(value)).first->second;
	}

	// Encodes a float's exact IEEE-754 bit pattern as a cache-key fragment -
	// std::to_string(float) (fixed 6 decimal digits) would risk two distinct-
	// but-close values formatting identically and colliding, silently
	// sharing a cache entry baked for the wrong value. Used by every
	// transform-cache key in this file that includes a scale/offset float
	// (previously hand-duplicated at each call site) - mirrors
	// getOrBuildPbrtImageTexture's identical gammaBits encoding
	// (pbrt_gpu_builder_materials.h), kept separate since that file doesn't
	// share a common header with this one worth adding just for this.
	inline std::string float_bits_key(float v) {
		std::uint32_t bits;
		std::memcpy(&bits, &v, sizeof(bits));
		return std::to_string(bits);
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

	// Lookat basis (right/up/back-from-lookat), shared by
	// build_pinhole_camera_params (static path, below) and
	// build_gpu_animated_camera_params's own per-keyframe basis (animated
	// path) - factored out so the two formulas can't independently drift,
	// mirroring src/TheRestOfYourLife/camera.h's own compute_lookat_basis(),
	// added there for the identical reason (see that function's own
	// comment: "so the two can't drift apart from independent copy-paste
	// edits").
	inline void compute_lookat_basis_gpu(
		const float3& from, const float3& at, const float3& vup,
		float3& out_u, float3& out_v, float3& out_w)
	{
		const float3 view_direction = make_float3(from.x - at.x, from.y - at.y, from.z - at.z);
		out_w = normalize(view_direction);
		out_u = normalize(cross(vup, out_w));
		out_v = cross(out_w, out_u);
	}

	// Perspective viewport dimensions from vertical FOV/aspect/focus
	// distance - shared for the same drift-prevention reason as
	// compute_lookat_basis_gpu above (mirrors camera.h's own
	// compute_viewport_geometry()).
	inline void compute_viewport_dims_gpu(
		float vfov_degrees, float aspect, float focus_dist,
		float& out_width, float& out_height)
	{
		constexpr float kPi = 3.14159265358979323846f;
		const float theta = vfov_degrees * kPi / 180.0f;
		const float h = tanf(theta / 2.0f);
		out_height = 2.0f * h * focus_dist;
		out_width = aspect * out_height;
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
	// screen_window (optional): pbrt-v4 Camera "perspective" "float
	// screenwindow" as [xmin,xmax,ymin,ymax], or nullptr (every call site
	// but the one that actually wires this - build_loaded_pbrt_scene()'s
	// own perspective-camera branch) for the plain vfov-only viewport
	// below. Mirrors CPU's own camera::initialize() (camera.h) - a code-
	// review pass found this was silently unsupported on GPU (unlike this
	// same round's other CPU-only gaps, which all warn) even though CPU
	// wires it for exactly this camera type.
	void build_pinhole_camera_params(
		const float3& lookfrom, const float3& lookat, const float3& vup,
		float vfov_degrees, float aspect, float focus_dist,
		float* camera_params,
		float3* out_u = nullptr, float3* out_v = nullptr, float3* out_w = nullptr,
		const float* screen_window = nullptr
	) {
		float viewport_width, viewport_height;
		float center_shift_u = 0.0f, center_shift_v = 0.0f;
		if (screen_window) {
			constexpr float kPi = 3.14159265358979323846f;
			const float h = tanf((vfov_degrees * kPi / 180.0f) / 2.0f);
			viewport_width  = (screen_window[1] - screen_window[0]) * h * focus_dist;
			viewport_height = (screen_window[3] - screen_window[2]) * h * focus_dist;
			center_shift_u  = (screen_window[0] + screen_window[1]) * 0.5f * h * focus_dist;
			center_shift_v  = (screen_window[2] + screen_window[3]) * 0.5f * h * focus_dist;
		} else {
			compute_viewport_dims_gpu(vfov_degrees, aspect, focus_dist, viewport_width, viewport_height);
		}

		float3 u, v, w;
		compute_lookat_basis_gpu(lookfrom, lookat, vup, u, v, w);

		const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
		const float3 vertical = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
		const float3 lower_left_corner = make_float3(
			lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - focus_dist * w.x + center_shift_u * u.x + center_shift_v * v.x,
			lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - focus_dist * w.y + center_shift_u * u.y + center_shift_v * v.y,
			lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - focus_dist * w.z + center_shift_u * u.z + center_shift_v * v.z
		);

		camera_params[0] = lookfrom.x;           camera_params[1] = lookfrom.y;           camera_params[2] = lookfrom.z;
		camera_params[3] = lower_left_corner.x;  camera_params[4] = lower_left_corner.y;  camera_params[5] = lower_left_corner.z;
		camera_params[6] = horizontal.x;         camera_params[7] = horizontal.y;         camera_params[8] = horizontal.z;
		camera_params[9] = vertical.x;           camera_params[10] = vertical.y;          camera_params[11] = vertical.z;

		if (out_u) *out_u = u;
		if (out_v) *out_v = v;
		if (out_w) *out_w = w;
	}

	// Populates GpuCameraParams for real per-ray camera motion blur (GPU
	// twin of src/TheRestOfYourLife/camera.h's camera_is_animated path -
	// see GpuCameraParams::animated's own comment, optix_types.h, and
	// generate_primary_ray()'s consumption of it, optix_device_helpers.h).
	// Always fills camera_params (the legacy 12-float static buffer) too,
	// via build_pinhole_camera_params at keyframe0 - both the "keyframes
	// turned out identical" fallback below and optix_interface.cpp's own
	// generic camera_params->cameraExtra copy (for scenes that don't set
	// cameraExtra explicitly) need it valid regardless of whether the
	// animated fields end up used.
	//
	// The decomposition into translate+rotate keyframes (iterative polar
	// decomposition + quaternion extraction) runs ONCE here, host-side, via
	// the real double-precision AnimatedTransform class
	// (src/shared/animated_transform.h) - GPU only interpolates the two
	// already-decomposed keyframes per ray. Camera-to-world matrices here
	// are always built from an orthonormal lookat basis (u/v/w, always
	// unit-length via normalize()/cross(), same construction as
	// build_pinhole_camera_params), so there is never a scale term to
	// store or interpolate, unlike the general AnimatedTransform.
	void build_gpu_animated_camera_params(
		const float3& lookfrom0, const float3& lookat0,
		const float3& lookfrom1, const float3& lookat1,
		const float3& vup, float vfov_degrees, float aspect,
		float defocus_angle_deg, float focus_dist,
		float* camera_params,
		GpuCameraParams* out_camera_extra
	) {
		// Viewport scale is arbitrary when there's no lens to blur through -
		// see build_pinhole_camera_params's own call sites for this exact
		// "1.0f, not a real distance, is deliberate" precedent.
		const float fd_for_viewport = (defocus_angle_deg > 0.0f) ? focus_dist : 1.0f;

		float3 u0, v0, w0;
		build_pinhole_camera_params(lookfrom0, lookat0, vup, vfov_degrees, aspect,
			fd_for_viewport, camera_params, &u0, &v0, &w0);

		// camera_params (the static-fallback buffer) is now valid regardless
		// of out_camera_extra - scene_builder.h documents out_camera_extra
		// as optional/nullable, and every other camera-writing case in this
		// file guards its own writes the same way.
		if (!out_camera_extra) return;

		out_camera_extra->kind = CameraKind::Perspective;

		if (defocus_angle_deg > 0.0f) {
			// Opts out of optix_interface.cpp's generic camera_params->
			// cameraExtra fallback (see CameraKind::Perspective's own
			// comment there) - set the full static set explicitly too, the
			// same as build_scene()'s own DOF call sites do.
			constexpr float kPi = 3.14159265358979323846f;
			const float defocus_radius = focus_dist * tanf((defocus_angle_deg * kPi / 180.0f) / 2.0f);
			out_camera_extra->origin = lookfrom0;
			out_camera_extra->lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
			out_camera_extra->horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
			out_camera_extra->vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
			out_camera_extra->defocus_disk_u = make_float3(u0.x * defocus_radius, u0.y * defocus_radius, u0.z * defocus_radius);
			out_camera_extra->defocus_disk_v = make_float3(v0.x * defocus_radius, v0.y * defocus_radius, v0.z * defocus_radius);
		}

		auto make_cam_to_world_matrix = [](const float3& from, const float3& u, const float3& v, const float3& w) -> AT_Mat44 {
			AT_Mat44 m;
			m.m[0][0]=u.x; m.m[0][1]=v.x; m.m[0][2]=w.x; m.m[0][3]=from.x;
			m.m[1][0]=u.y; m.m[1][1]=v.y; m.m[1][2]=w.y; m.m[1][3]=from.y;
			m.m[2][0]=u.z; m.m[2][1]=v.z; m.m[2][2]=w.z; m.m[2][3]=from.z;
			m.m[3][0]=0;   m.m[3][1]=0;   m.m[3][2]=0;   m.m[3][3]=1;
			return m;
		};

		// Keyframe0 reuses u0/v0/w0 (already computed above by
		// build_pinhole_camera_params) instead of re-deriving the same
		// basis a second time; keyframe1's basis is computed once via the
		// same shared helper.
		float3 u1, v1, w1;
		compute_lookat_basis_gpu(lookfrom1, lookat1, vup, u1, v1, w1);

		AnimatedTransform anim(make_cam_to_world_matrix(lookfrom0, u0, v0, w0), 0.0,
								make_cam_to_world_matrix(lookfrom1, u1, v1, w1), 1.0);

		out_camera_extra->animated = anim.IsAnimated() ? 1 : 0;
		if (!out_camera_extra->animated) {
			// Keyframes resolved identical (e.g. lookfrom1==lookfrom0) -
			// the static path above (camera_params + the DOF fields, if
			// any) is already a complete, correct single-frame camera;
			// nothing more to fill in.
			return;
		}

		out_camera_extra->animT0 = make_float3(
			static_cast<float>(anim.T[0][0]), static_cast<float>(anim.T[0][1]), static_cast<float>(anim.T[0][2]));
		out_camera_extra->animT1 = make_float3(
			static_cast<float>(anim.T[1][0]), static_cast<float>(anim.T[1][1]), static_cast<float>(anim.T[1][2]));
		out_camera_extra->animR0 = make_float4(
			static_cast<float>(anim.R[0].x), static_cast<float>(anim.R[0].y),
			static_cast<float>(anim.R[0].z), static_cast<float>(anim.R[0].w));
		out_camera_extra->animR1 = make_float4(
			static_cast<float>(anim.R[1].x), static_cast<float>(anim.R[1].y),
			static_cast<float>(anim.R[1].z), static_cast<float>(anim.R[1].w));

		float viewport_width, viewport_height;
		compute_viewport_dims_gpu(vfov_degrees, aspect, fd_for_viewport, viewport_width, viewport_height);
		out_camera_extra->localHorizontal = make_float3(viewport_width, 0.0f, 0.0f);
		out_camera_extra->localVertical   = make_float3(0.0f, viewport_height, 0.0f);
		out_camera_extra->localLowerLeftCorner = make_float3(-viewport_width / 2.0f, -viewport_height / 2.0f, -fd_for_viewport);

		if (defocus_angle_deg > 0.0f) {
			// Untested: no bundled scene currently combines an animated
			// camera with defocus_angle_deg > 0 (D13 hardcodes 0.0f, and no
			// pbrt_scenes/*.pbrt file uses ActiveTransform at all) - this
			// branch has no render or unit test exercising it yet.
			constexpr float kPi = 3.14159265358979323846f;
			const float defocus_radius = focus_dist * tanf((defocus_angle_deg * kPi / 180.0f) / 2.0f);
			out_camera_extra->localDefocusDiskU = make_float3(defocus_radius, 0.0f, 0.0f);
			out_camera_extra->localDefocusDiskV = make_float3(0.0f, defocus_radius, 0.0f);
		} else {
			out_camera_extra->localDefocusDiskU = make_float3(0.0f, 0.0f, 0.0f);
			out_camera_extra->localDefocusDiskV = make_float3(0.0f, 0.0f, 0.0f);
		}
	}

	// load_obj_triangles_gpu() (a bare-bones OBJ -> SceneData::triangles loader) and
	// load_obj_triangles_mtl_gpu() (its .mtl-aware sibling, with its parse_mtl_*_gpu wrappers,
	// map_Kd/map_Bump/map_d texture loading and bump-vs-normal detection) were deleted with the
	// G1-G24 mesh gallery and the H1-H12 environment scenes they served: those scenes are
	// pbrt-backed now and read their .obj through Shape "plymesh" (src/shared/ply_mesh.h's OBJ
	// support, including its "file.obj#material" group paths), the one mesh loader both backends
	// share; the per-material rules live in pbrt_scenes/environment-*.pbrt.
}

// build_rough_metal_spheres() (former "scene 9" / B1 GPU builder) deleted - B1
// migrated to pbrt-backed, see pbrt_scenes/rough-metal-spheres.pbrt and its case-9
// removal below.


// build_cornell_rough_metal() (former "scene 10" / B2 GPU builder) deleted - B2
// migrated to pbrt-backed, see pbrt_scenes/cornell-rough-metal.pbrt and its case-10
// removal below.


/// @brief Build Cornell Conductor scene (scene 12)
// build_cornell_conductor() (former "scene 12" / B4 Cornell Conductor GPU
// builder) deleted - B4 migrated to pbrt-backed, see pbrt_scenes/
// cornell-conductor.pbrt and its case-12 removal below. add_cornell_walls_
// and_main_light() itself is NOT deleted - B5-B9 and other cases still call
// it directly for their own (still-native) scenes.

/// @brief Build Cornell Coated Diffuse scene (scene 13)
// build_cornell_coated_diffuse() (GPU, former "scene 13" / B5) deleted - B5
// migrated to pbrt-backed, see pbrt_scenes/cornell-coated-diffuse.pbrt and
// its case-13 removal above. The CPU function of the same name is NOT
// deleted (tests/integration/skip_pdf_material_brightness_tests.cpp calls
// it directly) - this GPU-only deletion has no effect on that.

// build_cornell_thin_glass() (GPU, former "scene 14" / B6), build_cornell_
// coated_conductor() (GPU, former "scene 15" / B7), and build_cornell_
// wax_slab() (GPU, former "scene 16" / B8) all deleted - B6/B7/B8 migrated
// to pbrt-backed, see their own cornell-thin-glass.pbrt/cornell-coated-
// conductor.pbrt/cornell-wax-slab.pbrt and the case-14/15/16 removal above.
// None of these three has any other consumer (unlike build_cornell_box/
// build_cornell_conductor/build_cornell_coated_diffuse/build_cornell_rough_
// glass, which stay - other scenes or tests still call them directly).

// build_cornell_crystal() (former "scene 17" / B9 GPU builder) deleted - B9
// migrated to pbrt-backed, see pbrt_scenes/cornell-crystal.pbrt and its
// case-17 removal below. add_normalized_fresnel() (this file's own
// convenience wrapper) had no other caller - deleted too. The new generic
// pbrt Material "normalizedfresnel" dispatch (gpu/optix/pbrt_gpu_builder_
// materials.h) sets MaterialData's type directly instead, same convention
// as the Principled case there. MaterialType::NormalizedFresnel itself
// stays - that's the real, shared GPU material type (also used by
// Subsurface's own exit-point shading), not a now-dead convenience wrapper.

// build_bouncing_spheres() (former "scene 1" / A2 GPU builder) deleted - A2
// migrated to pbrt-backed, see pbrt_scenes/bouncing-spheres.pbrt and its
// case-1 removal below. That file's grid layout is this exact function's
// own fixed std::mt19937(42) sequence, dumped once by a throwaway C++
// program - not a new or different layout.

// build_checkered_spheres() (former "scene 2" / A3 Checkered Spheres GPU
// builder) deleted - A3 migrated to pbrt-backed, see
// pbrt_scenes/checkered-spheres.pbrt and its case-2 removal above.

// build_earth_gpu() (former "scene 3" / A4 Earth GPU builder) deleted - A4
// migrated to pbrt-backed, see pbrt_scenes/earth-globe.pbrt and its case-3
// removal above.

// add_perlin_spheres_pair_gpu()/build_perlin_spheres_gpu()/build_simple_light_gpu()
// (former "scene 4" / A5 Perlin Spheres and "scene 6" / A7 Simple Light GPU builders)
// deleted - both migrated to pbrt-backed, see pbrt_scenes/perlin-spheres.pbrt and
// pbrt_scenes/simple-light.pbrt and their case-4/case-6 removals below.


// build_quads_scene() (former "scene 5" / A6 Colored Quads GPU builder)
// deleted - A6 migrated to pbrt-backed, see pbrt_scenes/colored-quads.pbrt
// and its case-5 removal above.

// build_spotlight_cornell_gpu()/build_distant_light_cornell_gpu()/
// build_point_light_cornell_gpu() (former scenes 25/26/27 - C2/C3/C4) all
// deleted - migrated to pbrt-backed, see pbrt_scenes/cornell-spotlight.pbrt/
// cornell-distant-light.pbrt/cornell-point-light.pbrt and their case
// removal above. build_goniometric_cornell_gpu()/build_projection_cornell_gpu()
// (former scenes 28/29 - C5/C6) and the build_punctual_light_walls() helper
// they both shared are likewise deleted now that C5/C6 are migrated too -
// see pbrt_scenes/cornell-goniometric.pbrt/cornell-projection.pbrt and their
// case removal above. No remaining GPU builder calls build_punctual_light_walls().

// Scene 22 (Depth of Field / D1) migrated to pbrt-backed - see
// pbrt_scenes/depth-of-field-spheres.pbrt and scene_registry_data.h's own
// D1 entry. build_depth_of_field_gpu() itself is deleted - its own case 22
// (below) was its only caller, and the new pbrt file reunifies GPU with
// CPU's original design (see that file's own header comment on why GPU's
// render of this scene changes as a result).

// build_ortho_camera_scene_gpu() (former "scene 32" / D2 Orthographic
// Camera GPU builder) deleted - D2 migrated to pbrt-backed, see
// pbrt_scenes/ortho-camera-scene.pbrt and its case-32 removal above. Found
// along the way (worth recording since it's now permanently gone rather
// than just quietly fixed): this function's 5 sphere colors never matched
// CPU's own build_ortho_camera_scene() formula
// ((0.2+0.15*i, 0.3, 0.8-0.1*i)) at all - a real, pre-existing native
// CPU/GPU divergence for this scene, presumably introduced the same way as
// scene 33's own since-fixed drift noted just below (written independently
// of the CPU version rather than ported from it). The migrated .pbrt file
// uses CPU's real formula, so both backends now render identically.

// build_spherical_camera_scene_gpu() (former "scene 33" / D3 Spherical
// Camera GPU builder) deleted - D3 migrated to pbrt-backed, see
// pbrt_scenes/spherical-camera-scene.pbrt and its case-33 removal above.

// build_realistic_camera_scene_gpu() (former "scene 36" / D4 Realistic
// Camera GPU builder) deleted - D4 migrated to pbrt-backed, see
// pbrt_scenes/realistic-camera-scene.pbrt and its case-36 removal above.
// Unlike scene 32/D2 just above, this one's 5 sphere colors DID already
// match CPU's own sphere_colors[] exactly - only the ground (flat gray vs.
// CPU's checker_texture) and lens (identical to D8's own, already carried
// forward into geometry/dgauss-9-element.dat) needed reconciling, both
// resolved by this migration.

// build_prism_dispersion_gpu() (former B23/B24 shared GPU geometry builder)
// deleted - both B23 and B24 migrated to pbrt-backed, see
// pbrt_scenes/prism-dispersion.pbrt/frosted-prism-dispersion.pbrt and their
// case removal below. add_dispersive_dielectric()/
// add_dispersive_rough_dielectric() (this file's own convenience wrappers)
// had no other caller - deleted too. The new generic pbrt Material
// "dielectric" dispatch's "abbenumber" handling (gpu/optix/
// pbrt_gpu_builder_materials.h) sets MaterialData's dispersive_extra fields
// directly instead, same convention as every other generic pbrt material
// case. I2 (CPU-only, out of scope) still uses the CPU-side
// build_prism_dispersion()/build_prism_dispersion_geometry()/
// build_prism_dispersion_punct(), none of which this GPU-only deletion
// touches.

// build_hdri_sky_world_gpu() (former "scene 24"/C1, and its case-134/I3
// alias) deleted - both migrated to pbrt-backed, see pbrt_scenes/
// hdri-sky-gradient.pbrt and scene_registry_data.h's own C1/I3 entries.

// build_portal_light_scene_gpu()/build_cornell_smoke_gpu()/
// build_homogeneous_medium_scene_gpu() (former "scenes 35/7/30" / C7/A8/E1
// GPU builders) deleted - all three migrated to pbrt-backed, see
// pbrt_scenes/portal-window-room.pbrt/cornell-smoke.pbrt/
// homogeneous-medium.pbrt and their case removal above. Worth recording
// since these are now permanently gone rather than just quietly fixed: all
// three native GPU builders APPROXIMATED their fog/medium boundary as a
// sphere where CPU used a real box (GPU media were sphere-only when these
// were written) - a real, pre-existing, disclosed CPU/GPU geometry
// divergence for all three scenes. The migrated .pbrt files use a real
// box boundary (GpuMediumShapeKind::Box, the same mechanism
// build_subsurface_slab_gpu's own wax slab already used below) on GPU too,
// via this loader's generic pbrt medium builder - a genuine fidelity
// improvement over the deleted sphere approximation, not a regression.

// build_normal_mapped_cornell_gpu() (former "scene 20" / B12 GPU builder)
// deleted - B12 migrated to pbrt-backed, see pbrt_scenes/
// normal-mapped-cornell.pbrt and its case-20 removal below. That file's own
// header comment documents an important finding this function's own removed
// comment first surfaced: native CPU's bump-mapped wall/box was a CONFIRMED
// NO-OP (bump_map_material sampling 3 different (u,v) but the same 3D hit
// point, combined with noise_texture ignoring u,v entirely, producing an
// always-zero gradient) - this function deliberately matched that flat
// behavior rather than building unexercised bump-map device code. The new
// pbrt file uses a real UV-indexed image texture instead, so CPU now shows
// a real, visible bump effect for the first time - a genuine fidelity
// improvement, not a regression, even though it means GPU (which still has
// no scalar bump-displacement material type - the same already-accepted
// gap as every other grayscale "texture displacement" scene) now diverges
// MORE from CPU on this scene than before.

// build_subsurface_slab_gpu() (former "scene 21" / B13 Subsurface Slab GPU
// builder) deleted - B13 migrated to pbrt-backed, see
// pbrt_scenes/subsurface-slab.pbrt and its case-21 removal above. This one
// already used a real box boundary (GpuMediumShapeKind::Box) for the wax
// slab, matching CPU exactly - no geometry approximation to note here,
// unlike A8/C7/E1's deleted sphere-approximated builders just above.

// build_cloud_medium_scene_gpu()/build_dielectric_medium_scene_gpu() (former
// "scenes 31/69" / E2/E3 GPU builders) deleted - both migrated to pbrt-
// backed, see pbrt_scenes/cloud-medium-scene.pbrt/
// dielectric-medium-showcase.pbrt and their case removal above. Both
// already matched CPU exactly (no geometry approximation to record, unlike
// A8/C7/E1 above) - the cloud used a sphere trigger the same way its
// pbrt-authored replacement does (see cloud-medium-scene.pbrt's own header
// comment), and the dielectric spheres were already a single fused
// MaterialType::DielectricMedium matching CPU's own two-hittable
// dielectric+constant_medium pair.

// build_rgb_grid_medium_scene_gpu() (former "scene 70" / E4 GPU builder)
// deleted - E4 migrated to pbrt-backed, see pbrt_scenes/rgb-grid-nebula.pbrt
// and its case removal below. Already matched CPU exactly (same world AABB,
// same generate_nebula_channel() calls, same sigma_scale/phase_g - no
// geometry approximation to record), so this migration doesn't change
// either backend's render.

// build_bilinear_patch_scene_gpu()/build_hair_fibers_gpu() (former "scenes
// 23/19" / F1/B11 GPU builders) deleted - both migrated to pbrt-backed, see
// pbrt_scenes/bilinear-patch-scene.pbrt/hair-fibers-scene.pbrt and their
// case removal above. Both already matched CPU exactly (bilinear patches
// and MaterialType::Hair are both real, non-approximated GPU geometry/
// material types already).

// build_principled_showcase_gpu() (former "scene 18" / B10 GPU builder)
// deleted - B10 migrated to pbrt-backed, see pbrt_scenes/
// principled-showcase.pbrt and its case-18 removal below. add_principled()
// (this file's own convenience wrapper) had no other caller - deleted too.
// The new generic pbrt Material "principled" dispatch (gpu/optix/
// pbrt_gpu_builder_materials.h) sets MaterialData's fields directly instead,
// same convention as every other generic pbrt material case there.
// MaterialType::Principled itself stays - that's the real, shared GPU
// material type, not a now-dead convenience wrapper.

// build_measured_brdf_scene_gpu() (former "scene 34" / B14 Measured BRDF
// GPU builder) deleted - B14 migrated to pbrt-backed, see pbrt_scenes/
// measured-brdf-showroom.pbrt and scene_registry_data.h's own entry. This
// native builder only ever matched CPU's own mislabeled-Lambertian
// behavior (MaterialType::Lambertian, not real measured-BRDF math); the
// pbrt-backed version uses this project's REAL, working, importance-
// sampled Measured BRDF support instead (gpu/optix/optix_measured_bxdf.h/
// wavefront_measured_bxdf.h), already proven on 3 downloaded
// pbrt-v4-scenes bundles, now reachable from this small self-contained
// scene too.

// build_triangle_mesh_scene_gpu() (former "scene 37" / F2 Triangle Mesh
// GPU builder) deleted - F2 migrated to pbrt-backed, see
// pbrt_scenes/triangle-mesh-scene.pbrt and its case-37 removal above.

// build_curve_fibers_scene_gpu() (former "scene 72" / F4 Curve Fibers GPU
// builder) deleted - F4 migrated to pbrt-backed, see
// pbrt_scenes/curve-fibers-scene.pbrt and its case-72 removal above. Real
// pbrt Shape "curve" already tessellates into bilinear patches on this
// loader's own generic pbrt_gpu_builder.h path (see that file's own "----
// curves ----" section), the identical strategy this deleted function used
// by hand.


// build_final_scene_gpu() (former "scene 8" / A9 GPU builder) deleted - A9
// migrated to pbrt-backed, see pbrt_scenes/final-scene.pbrt and its case-8
// removal below. That file's ground-box/sphere-cluster layout is this exact
// function's own fixed std::mt19937(8) sequence, dumped once by a throwaway
// C++ program - not a new or different layout.


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
	const bool has_custom_lookat,
	const double lookat_x,
	const double lookat_y,
	const double lookat_z,
	GpuCameraParams* out_camera_extra,
	// Depth-of-field override - see build_scene()'s own comment
	// (scene_builder.h) and RenderOptions::aperture_override's own comment
	// (render_options.h). This is the ONE function that reads a scene's
	// own parsed Camera::aperture/focusDistance, so it's the one place the
	// override needs to plug in.
	const bool has_dof_override = false,
	const double aperture_override = 0.0,
	const double focus_distance_override = 0.0
) {
	// pbrt_load::loadFile() does real, scene-size-scaling work - disk I/O,
	// full text parsing, PLY mesh loading, and infinite-light image decode -
	// none of which depends on the camera. This function is called on EVERY
	// Live Preview frame the camera moves (rt_realtime_render_frame()'s own
	// cache only covers the GPU-side upload a few frames down the call
	// chain, not this CPU-side parse - see prepareSceneAndCamera()'s own
	// comment), so re-parsing a large external .pbrt scene from scratch on
	// every WASD/orbit frame is what made Live Preview unusably slow there -
	// small/procedural scenes never hit this function at all (they're built
	// directly in the switch below), which is why the slowdown was specific
	// to large, file-loaded scenes. Cached by path for the life of the
	// process - same "no hot-reload, cache lives for the process" precedent
	// g_uploaded_scene_id's own GPU-side scene cache already sets (editing a
	// scene's file mid-session already isn't picked up by that cache
	// either). A FAILED load is deliberately NOT cached, unlike an earlier
	// version of this cache: a scene file that's mid-save, briefly
	// malformed, or momentarily locked by another process at the exact
	// moment Live Preview first requests it would otherwise stay marked
	// failed for the rest of the process even after the file is fixed on
	// disk - the same "never cache a failure" choice the since-deleted OBJ
	// loaders made.
	// pbrt_gpu::build() below takes its FlatScene by const& and never
	// mutates it, so the cached entry can be reused directly by every
	// subsequent call with no copy. The per-warning print happens inside the
	// builder below (only on an actual, successful parse), not out here -
	// this function now runs every Live Preview frame the camera moves, so
	// printing on every cache HIT too would spam stderr continuously for
	// any scene with warnings, instead of the one-time diagnostic it used
	// to be back when this function was slow enough that repeat calls were
	// rare.
	static std::unordered_map<std::string, pbrt_load::LoadResult> s_pbrtLoadCache;
	static std::mutex s_pbrtLoadCacheMutex;
	const pbrt_load::LoadResult* loadedPtr = get_or_build_cached(s_pbrtLoadCache, s_pbrtLoadCacheMutex, std::string(path),
		[&](pbrt_load::LoadResult& out) -> bool {
			out = pbrt_load::loadFile(path);
			if (!out.ok) {
				std::cerr << "[OptiX] " << out.error << "\n";
				return false;
			}
			for (const pbrt_scene::Warning& w : out.scene.warnings)
				std::cerr << "[OptiX] warning: " << path << ": " << w.message << "\n";
			return true;
		});
	if (!loadedPtr) return false;
	const pbrt_load::LoadResult& loaded = *loadedPtr;

	// pbrt_gpu::build() itself is the SOLE populator of `scene` in this
	// function (everything after this call only reads loaded.scene/stats to
	// fill out_camera_extra, never scene) and is a pure function of
	// loaded.scene alone - already cached above, and never mutated by
	// build() (takes it by const&) - so its own output is exactly as
	// cacheable, and for the identical reason: this function reruns on every
	// Live Preview frame the camera moves, and build()'s own triangle-
	// flattening loop (proportional to triangle count, same cost class as
	// the since-deleted OBJ loaders that used to live in scene_builder.cpp)
	// was measured to still dominate per-frame cost on a
	// heavy scene even with pbrt_load::loadFile() itself cached (villa-
	// daylight: ~1.6s/frame before this cache, a known, previously-flagged
	// gap - see this cache's own commit message). Cached by the same `path`
	// key as s_pbrtLoadCache above; a full SceneData copy (not a pointer) on
	// both store and retrieve, same "bulk-copy beats re-derive" trade this
	// file's OBJ-loader caches already make.
	struct PbrtBuiltScene {
		SceneData sceneData;
		pbrt_gpu::BuildStats stats;
	};
	static std::unordered_map<std::string, PbrtBuiltScene> s_pbrtBuiltSceneCache;
	static std::mutex s_pbrtBuiltSceneCacheMutex;

	const PbrtBuiltScene* built = get_or_build_cached(s_pbrtBuiltSceneCache, s_pbrtBuiltSceneCacheMutex, std::string(path),
		[&](PbrtBuiltScene& out) -> bool {
			out.stats = pbrt_gpu::build(loaded.scene, out.sceneData);
			return true;
		});
	if (!built) return false;
	if (scene.skipExpensiveGeometryLoad) {
		// See SceneData::skipExpensiveGeometryLoad's own comment - this pbrt
		// scene is already GPU-resident, so the full built->sceneData copy
		// below (proportional to triangle/texture-byte count) is skipped.
		// lensElements/exitPupilBounds/lightIndices/lightKinds are the
		// exceptions: small, camera- or light-count-sized tables that code
		// further down THIS SAME function still reads every call, skip or
		// not (RealisticCamera setup's numLensElements/numExitPupilBounds,
		// and the "N sampled lights"/"no samplable lights" diagnostics a few
		// lines below) - copying just these avoids the two silently
		// reporting stale/zero counts on every skip-path call.
		scene.lensElements = built->sceneData.lensElements;
		scene.exitPupilBounds = built->sceneData.exitPupilBounds;
		scene.lightIndices = built->sceneData.lightIndices;
		scene.lightKinds = built->sceneData.lightKinds;
	} else {
		scene = built->sceneData;
	}
	const pbrt_gpu::BuildStats& stats = built->stats;
	std::cerr << "[OptiX] Loaded " << path << ": " << stats.triangles
		  << " triangles, " << stats.spheres << " spheres, "
		  << stats.quadLights << " quads, "
		  << stats.disks << " disks, " << stats.cylinders << " cylinders, "
		  << scene.lightIndices.size() << " sampled lights\n";

	// Flat-colour GPU approximation of the scene's own LightSource "infinite"
	// (see GpuCameraParams::backgroundColor's comment and pbrt_gpu_builder.h's
	// BuildStats::backgroundColor) - same shape as every hand-written HDRI
	// scene's own GPU port a few lines up in this file. Left at zero-init
	// (matches CPU bg=(0,0,0)) for a scene with no infinite light.
	if (out_camera_extra) out_camera_extra->backgroundColor = stats.backgroundColor;

	// PixelFilter directive - matches CPU's own scene_registry.h wiring
	// (setup_camera's cam.filter_kind/B/C/sigma/tau, unconditional like
	// this) via the same FlatScene::filter field. See GpuCameraParams::
	// filterKind's own comment for the string->int mapping and defaults.
	if (out_camera_extra) {
		const pbrt_flatten::PixelFilter& pf = loaded.scene.filter;
		out_camera_extra->filterKind = (pf.kind == "box") ? 1
			: (pf.kind == "triangle") ? 2
			: (pf.kind == "mitchell") ? 3
			: (pf.kind == "sinc") ? 4
			: 0;  // "gaussian", or anything unrecognized
		out_camera_extra->filterB = static_cast<float>(pf.B);
		out_camera_extra->filterC = static_cast<float>(pf.C);
		out_camera_extra->filterSigma = static_cast<float>(pf.sigma);
		out_camera_extra->filterTau = static_cast<float>(pf.tau);
		out_camera_extra->filterRadius = static_cast<float>(pf.radius);  // real support radius - see GpuCameraParams::filterSampler
	}

	// Film "float maxcomponentvalue" - CPU's own per-sample firefly clamp
	// (camera::max_component_value, camera.h) is now real on GPU too - see
	// GpuCameraParams::maxComponentValue's own comment for the
	// recursive-exact/wavefront-approximate split. Applied unconditionally
	// from the scene's own declaration, same "PixelFilter"/"regularize"
	// shape as the block just below this one - CPU's own class-level
	// default (1e9, effectively unbounded) already matches this field's
	// own default, so a scene that never declares this behaves exactly as
	// before this feature existed.
	if (out_camera_extra) out_camera_extra->maxComponentValue = static_cast<float>(loaded.scene.maxComponentValue);

	// Integrator "bool regularize" - same unconditional-from-the-scene shape
	// as PixelFilter just above. See GpuCameraParams::regularize's own comment.
	if (out_camera_extra) out_camera_extra->regularize = loaded.scene.regularize ? 1 : 0;

	// Film "cropwindow"/"pixelbounds" - resolved to concrete PIXEL bounds
	// here (image_width/image_height, this function's own params, are
	// already the render's real resolution - unlike CPU's camera class,
	// GPU has no lazy "resolve once width/height are known" step of its
	// own, so this IS that step) via the shared resolve_crop_pixel_bounds()
	// (src/shared/cameras.h - see its own comment for why CPU's own
	// camera::initialize() doesn't call it too). A degenerate collapse
	// warns unconditionally, matching CPU's own initialize() shape exactly -
	// image_width/image_height are already validated nonzero well before
	// this point (this function can't be reached otherwise), so the only
	// way to resolve to a degenerate range is a genuinely tiny non-full
	// crop rectangle, never the trivial "no cropwindow at all" default.
	if (out_camera_extra) {
		const CropPixelBounds bounds = resolve_crop_pixel_bounds(
			loaded.scene.cropX0, loaded.scene.cropX1,
			loaded.scene.cropY0, loaded.scene.cropY1,
			image_width, image_height);
		if (bounds.wasDegenerate) {
			std::cerr << "[OptiX] Warning: Film \"cropwindow\"/\"pixelbounds\" "
						 "resolved to an empty pixel range at this render "
						 "resolution (" << image_width << "x" << image_height
					  << "); rendering the full frame instead.\n";
		}
		out_camera_extra->cropX0 = bounds.x0;
		out_camera_extra->cropX1 = bounds.x1;
		out_camera_extra->cropY0 = bounds.y0;
		out_camera_extra->cropY1 = bounds.y1;
	}

	// Texture "imagemap" "string encoding"/"string wrap"/"bool invert" - GPU
	// now implements all three for the primary "reflectance" slot of
	// Diffuse/CoatedDiffuse/DiffuseTransmission (getOrBuildPbrtImageTexture()/
	// sampleImage() in this file / optix_device_helpers.h - gpuWrapModeFor(),
	// TextureData::wrapMode/GpuWrapMode), matching CPU's own scope exactly
	// (imageMapOptionsFor(), pbrt_cpu_builder.h) - Material::textureFilename
	// is never populated for any OTHER material kind (verified: both
	// builders' only 3 read sites are exactly these kinds), so this closes
	// the divergence the old warning here used to flag entirely, not just
	// narrow it.

	// MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4) - implemented
	// on both GPU backends for BOTH "homogeneous" media (MaterialType::
	// Medium, a single build-time-baked sigma_a/sigma_t-weighted constant -
	// see MaterialData::medium_emission's own comment, optix_types.h, and
	// each backend's own Medium closest-hit case for the real-collision
	// gate) AND "rgbgrid"'s real per-voxel "Le" array (MaterialType::
	// RgbGridMedium - see GpuRgbGridMedium::leDataOffset's own comment,
	// optix_types.h, and each backend's own RgbGridMedium closest-hit case).
	// The rgbgrid case is NOT weighted by a sigma_a/sigma_t fraction like
	// the homogeneous case is (so a scene mixing "rgb sigma_s" and "rgb Le"
	// on the same medium can render noticeably brighter on GPU than --cpu -
	// see GpuRgbGridMedium::Le_scale's own comment for the full reasoning) -
	// GPU RgbGridMedium has no sigma_a grid at all
	// (a real, pre-existing, documented simplification over CPU's fuller
	// RGBGridMediumData support - see GpuRgbGridMedium's own comment) - so
	// every accepted scatter collision emits the full per-voxel Le rather
	// than only the "absorption fraction" CPU models.
	//
	// "rgbgrid" needs its own hasLe check here: its real Le lives in the
	// per-voxel Le_r/Le_g/Le_b arrays now (Medium::Le_r's own comment), not
	// the flat Le[3] field every OTHER type still uses - the flat field
	// stays permanently zero for rgbgrid (pbrt_flatten.h's own `if
	// (!isRgbGrid)` guard around that generic read), so `isNonzeroRGB(med.Le)`
	// alone can never see a real rgbgrid emission request. Checking
	// Le_r/Le_g/Le_b (gated on Le_scale>0, matching RGBGridMediumData::
	// is_emissive()'s own "Le_grids && Le_scale>0" convention) whenever the
	// medium is rgbgrid specifically.
	//
	// "homogeneous" is unconditionally excluded from the warning below - GPU
	// always implements its real medium emission (see MaterialData::
	// medium_emission's own comment). "rgbgrid" is excluded ONLY when GPU can
	// actually realize it: pbrt_gpu_builder.h's rgbgrid branch (hasScattering)
	// requires a valid "rgb sigma_s" array of exactly nx*ny*nz*3 numbers to
	// even build a nonzero majorant (GpuRgbGridMedium::sigma_maj) - without
	// one the delta-tracking loop in each backend's own RgbGridMedium
	// closest-hit case never runs a single iteration, so the new Le-sampling
	// code inside it (gated on an accepted scatter collision) never executes
	// either. A scene with real "rgb Le" but no (or a malformed) "rgb
	// sigma_s" would otherwise render that medium as silently invisible on
	// GPU - same voxels*3 size check as pbrt_gpu_builder.h's own
	// `hasScattering`, duplicated here since that function's local isn't
	// reachable from this loop. Only "cloud"/"uniformgrid" still have no GPU
	// emission concept at all.
	for (const pbrt_flatten::Medium& med : loaded.scene.media) {
		bool hasLe = pbrt_flatten::isNonzeroRGB(med.Le);
		bool gpuCanRealizeRgbGridLe = false;
		if (med.type == "rgbgrid") {
			if (!hasLe && med.Le_scale > 0.0) {
				const auto anyNonzero = [](const std::vector<double>& v) {
					return std::any_of(v.begin(), v.end(), [](double x) { return x > 1e-9; });
				};
				hasLe = anyNonzero(med.Le_r) || anyNonzero(med.Le_g) || anyNonzero(med.Le_b);
			}
			const std::size_t voxels = static_cast<std::size_t>(med.nx)
				* static_cast<std::size_t>(med.ny) * static_cast<std::size_t>(med.nz);
			gpuCanRealizeRgbGridLe = med.sigma_s_r.size() == voxels
				&& med.sigma_s_g.size() == voxels && med.sigma_s_b.size() == voxels;
		}
		const bool gpuSupportsThisMediumsLe =
			med.type == "homogeneous" || (med.type == "rgbgrid" && gpuCanRealizeRgbGridLe);
		if (!gpuSupportsThisMediumsLe && hasLe) {
			std::cerr << "[OptiX] Warning: scene requests a MakeNamedMedium "
						 "\"Le\"/\"Lescale\" (a self-emitting medium) on a \""
					  << med.type << "\" medium, but GPU rendering only "
						 "implements medium emission for \"homogeneous\" media "
						 "and \"rgbgrid\" media that also have a valid \"rgb "
						 "sigma_s\" array - this medium will scatter/absorb "
						 "but not glow. Use --cpu to honor this request.\n";
			break;
		}
	}

	// pbrt-v4's own "camera medium" (FlatScene::cameraMediumIndex's own
	// comment, pbrt_flatten.h) - now real on BOTH GPU backends (recursive:
	// optix_raygen.h's own call site; wavefront: wavefront_kernels.cu's own
	// call site), sharing this one GpuCameraParams-level construction since
	// GpuCameraParams is already threaded to both (see GpuCameraParams::
	// cameraMediumSigmaT's own comment, optix_types.h). The homogeneous collapse
	// (luminance sigma_a/sigma_s, chromatic tint, single-scatter albedo) is
	// pbrt_gpu::cameraMediumGpu(), shared with the synthetic wavefront material
	// pbrt_gpu::build() creates for the same medium. flatten() only ever resolves
	// cameraMediumIndex to a homogeneous medium, warning and leaving it at -1 for
	// anything else - see pbrt_flatten.h's own resolution block.
	if (out_camera_extra && loaded.scene.cameraMediumIndex >= 0
		&& static_cast<std::size_t>(loaded.scene.cameraMediumIndex) < loaded.scene.media.size()) {
		const pbrt_gpu::CameraMediumGpu cm = pbrt_gpu::cameraMediumGpu(
			loaded.scene.media[static_cast<std::size_t>(loaded.scene.cameraMediumIndex)]);
		out_camera_extra->cameraMediumSigmaT = cm.sigmaT;
		out_camera_extra->cameraMediumAlbedo = cm.albedo;   // tint * sigma_s/sigma_t - see cameraMediumGpu()
		out_camera_extra->cameraMediumG = cm.g;
		out_camera_extra->cameraMediumEmission = cm.emission;
		out_camera_extra->cameraMediumSigmaA = cm.sigmaA;
		out_camera_extra->cameraMediumSigmaS = cm.sigmaS;
		out_camera_extra->cameraMediumLeRaw = cm.le;
		out_camera_extra->cameraMediumMaterialIdx = built->sceneData.cameraMediumMaterialIdx;
	}

	// Shape "cone"/"paraboloid" (pbrt_flatten::Cone/Paraboloid's own comment)
	// - CPU-only, v1 scope; GPU (both backends) has no counterpart at all yet.
	// Warned rather than silently rendering with the shape simply missing,
	// same "warn rather than silently drop" precedent as every other CPU-
	// only feature in this codebase.
	if (!loaded.scene.cones.empty() || !loaded.scene.paraboloids.empty()) {
		std::cerr << "[OptiX] Warning: scene has " << loaded.scene.cones.size()
			<< " cone(s) and " << loaded.scene.paraboloids.size()
			<< " paraboloid(s), which are not supported on GPU - they will "
			   "not be rendered; use --cpu instead if this geometry matters "
			   "for this render.\n";
	}

	// checkerboard/mix "tex1"/"tex2" nesting a further procedural texture
	// (pbrt_flatten::NestedProceduralTexture's own comment) - CPU-only;
	// TextureData's tex1ImageIdx/tex2ImageIdx (optix_types.h) are image-only,
	// with no representation for a nested Checker/Mix-kind entry. flatten()
	// already leaves a flat average-of-the-nested-pattern colour behind for
	// this case (so GPU still renders something reasonable, not black/
	// garbage) - warned here so that approximation is a known, disclosed
	// divergence from CPU's real nested rendering, not a silent one.
	{
		int nestedCount = 0;
		for (const pbrt_flatten::Material &m : loaded.scene.materials) {
			if (m.hasCheckerReflectance &&
				(!m.checkerTex1Nested.kind.empty() || !m.checkerTex2Nested.kind.empty()))
				++nestedCount;
			if (m.hasMixReflectance &&
				(!m.mixTex1Nested.kind.empty() || !m.mixTex2Nested.kind.empty() ||
				 !m.mixAmountNested.kind.empty()))
				++nestedCount;
		}
		if (nestedCount > 0) {
			std::cerr << "[OptiX] Warning: " << nestedCount << " material(s) bind a "
				   "checkerboard/mix texture's tex1/tex2 to a further nested "
				   "procedural texture, which GPU approximates as a flat "
				   "average colour instead of rendering for real - use --cpu "
				   "instead if the nested pattern matters for this render.\n";
		}
	}

	// Any pbrt_flatten::Medium::type mediumMaterialIndex() (pbrt_gpu_
	// builder.h) had no real branch for - e.g. "nanovdb" (CPU-only, v1
	// scope; GPU has no NanoVDB reader at all) - falls through to the
	// generic homogeneous-medium path (MaterialType::Medium), using the
	// SAME sigma_a/sigma_s the scene declared but filling the ENTIRE
	// boundary shape uniformly - a flat fog, not whatever real
	// (heterogeneous, sparse, etc) field CPU renders for that type. Warned
	// explicitly since this is a visibly WRONG render, not merely an
	// absent one. Driven by `stats.unsupportedMediumTypeCounts`
	// (BuildStats's own comment) rather than re-scanning loaded.scene.media
	// for one hardcoded type name here - that field is populated at the
	// SAME place the fallthrough decision is actually made, so a future
	// medium type left unhandled in pbrt_gpu_builder.h gets this warning
	// automatically instead of needing a matching edit in this file too.
	for (const auto &[type, count] : stats.unsupportedMediumTypeCounts) {
		std::cerr << "[OptiX] Warning: scene has " << count << " \"" << type << "\" "
			   "medium/media, which GPU has no real support for - it/they will "
			   "render as flat homogeneous fog (using the same sigma_a/sigma_s) "
			   "filling the whole boundary shape instead of the real field CPU "
			   "renders; use --cpu instead if that data matters for this render.\n";
	}

	// Reported, not warned about: these are sampled properly now (as
	// GpuLightKind::Triangle), so the only thing worth saying is that they
	// took the per-triangle path rather than the cheaper merged-quad one.
	if (stats.emissiveTrianglesSampledIndividually > 0) {
		std::cerr << "[OptiX] " << stats.emissiveTrianglesSampledIndividually
			  << " emissive triangle(s) are not parallelograms and are sampled "
			     "individually\n";
	}
	// backgroundColor is zero-init unless the scene has a real LightSource
	// "infinite" (see BuildStats::backgroundColor's comment) - a scene lit
	// purely by one, like sportscar-sky.pbrt (whose 5 AreaLightSource blocks
	// all have their Shape commented out in the source file itself, leaving
	// zero real discrete lights), still renders properly via sky NEE
	// (background color feeds the same NEE path a real light would), so
	// warning "expect a very dark image" here would be actively wrong, not
	// just uninformative.
	if (scene.lightIndices.empty() &&
	    stats.backgroundColor.x == 0.0f && stats.backgroundColor.y == 0.0f &&
	    stats.backgroundColor.z == 0.0f) {
		std::cerr << "[OptiX] warning: no samplable lights in this scene - "
			     "expect a very dark image.\n";
	}
	if (stats.instancePlacements > 0) {
		std::cerr << "[OptiX] " << stats.instancePlacements << " object instance placement(s)\n";
	}
	// See BuildStats::animatedDiskCylinderCount's own comment - real on CPU
	// (disk_cylinder_hittable.h's AnimatedTransform), not yet ported to
	// either GPU backend.
	if (stats.animatedDiskCylinderCount > 0) {
		std::cerr << "[OptiX] Warning: " << stats.animatedDiskCylinderCount
			  << " disk/cylinder shape(s) have an ActiveTransform \"StartTime\"/"
			     "\"EndTime\" motion pair, which GPU has no real support for - "
			     "rendered static at their StartTime position instead of "
			     "blurred; use --cpu instead if that motion matters for this "
			     "render.\n";
	}
	// See BuildStats::animatedMeshCount's own comment - real on CPU
	// (animated_transform_instance.h), not yet ported to either GPU backend.
	if (stats.animatedMeshCount > 0) {
		std::cerr << "[OptiX] Warning: " << stats.animatedMeshCount
			  << " trianglemesh/plymesh/loopsubdiv shape(s) have an "
			     "ActiveTransform \"StartTime\"/\"EndTime\" motion pair, which "
			     "GPU has no real support for - rendered static at their "
			     "StartTime position instead of blurred; use --cpu instead if "
			     "that motion matters for this render.\n";
	}
	// See BuildStats::animatedBilinearPatchCount/animatedCurveCount's own
	// comment - real on CPU (animated_transform_instance.h), not yet ported
	// to either GPU backend.
	if (stats.animatedBilinearPatchCount > 0) {
		std::cerr << "[OptiX] Warning: " << stats.animatedBilinearPatchCount
			  << " bilinearmesh shape(s) have an ActiveTransform \"StartTime\"/"
			     "\"EndTime\" motion pair, which GPU has no real support for - "
			     "rendered static at their StartTime position instead of "
			     "blurred; use --cpu instead if that motion matters for this "
			     "render.\n";
	}
	if (stats.animatedCurveCount > 0) {
		std::cerr << "[OptiX] Warning: " << stats.animatedCurveCount
			  << " curve shape(s) have an ActiveTransform \"StartTime\"/"
			     "\"EndTime\" motion pair, which GPU has no real support for - "
			     "rendered static at their StartTime position instead of "
			     "blurred; use --cpu instead if that motion matters for this "
			     "render.\n";
	}

	// The scene's own camera, unless the user moved it.
	const pbrt_flatten::Camera& c = loaded.scene.camera;
	const float3 lookfrom = force_camera_override
		? make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y),
			      static_cast<float>(cam_z))
		: make_float3(static_cast<float>(c.lookfrom[0]),
			      static_cast<float>(c.lookfrom[1]),
			      static_cast<float>(c.lookfrom[2]));
	// The scene's own look-at, unless Live Preview's free-fly camera has
	// moved it - mirrors lookfrom's own force_camera_override substitution
	// just above. The animated-camera branch below deliberately rebuilds
	// its own lookat from c.lookat directly instead of using this, for the
	// same reason its lookfrom counterpart does (see that branch's comment).
	const float3 lookat = has_custom_lookat
		? make_float3(static_cast<float>(lookat_x), static_cast<float>(lookat_y),
			      static_cast<float>(lookat_z))
		: make_float3(static_cast<float>(c.lookat[0]),
			      static_cast<float>(c.lookat[1]),
			      static_cast<float>(c.lookat[2]));
	const float3 vup = make_float3(static_cast<float>(c.up[0]),
				       static_cast<float>(c.up[1]),
				       static_cast<float>(c.up[2]));
	const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
	// c.aperture (pbrt's lensradius*2, a world-space lens diameter) was
	// never read here at all: every loaded .pbrt scene rendered pinhole-sharp
	// on GPU regardless of what its own Camera directive's "lensradius"
	// asked for, while the identical scene on CPU (scene_registry.h, which
	// does read it via defocusAngleDegreesFor()) correctly blurred. Case 1 and the generic pbrt
	// Depth-of-Field (D1) path elsewhere in this file show the pattern this now follows: pass
	// the real focus_dist into build_pinhole_camera_params (capturing its u/v
	// basis) only when there's a nonzero aperture to blur with, and derive
	// defocus_disk_u/v from it exactly as camera.h does - a zero-aperture
	// scene keeps the prior focus_dist=1.0f/no-DOF behavior untouched (see
	// build_pinhole_camera_params's own viewport-scaling comment this
	// replaces for why 1.0f, not a real distance, is deliberate there).
	float focus_dist_world = static_cast<float>(pbrt_flatten::focusDistanceFor(c));
	float defocus_angle_deg = static_cast<float>(
		pbrt_flatten::defocusAngleDegreesFor(c, focus_dist_world));
	// Depth-of-field override - computed BEFORE the "if (defocus_angle_deg
	// > 0.0f)" gate below so the override can both ADD DOF to a pinhole
	// scene and change/remove DOF on a scene that already has its own -
	// see this project's own DOF plan.
	if (has_dof_override) {
		if (focus_distance_override > 0.0) focus_dist_world = static_cast<float>(focus_distance_override);
		if (aperture_override >= 0.0) {
			const double lens_radius = aperture_override * 0.5;
			defocus_angle_deg = (lens_radius > 0.0 && focus_dist_world > 0.0)
				? static_cast<float>(2.0 * std::atan(lens_radius / static_cast<double>(focus_dist_world)) * 180.0 / 3.14159265358979323846)
				: 0.0f;
		} else if (focus_distance_override > 0.0) {
			// Aperture untouched but focus distance changed - the defocus
			// angle depends on both, so recompute it for the NEW focus
			// distance using the scene's OWN aperture (c.aperture) rather
			// than silently scaling the effective aperture. Reuses
			// defocusAngleDegreesFor() directly instead of duplicating its
			// formula a second time (unlike the aperture-override branch
			// above, which has no Camera object to call it on for an
			// arbitrary override aperture).
			defocus_angle_deg = static_cast<float>(pbrt_flatten::defocusAngleDegreesFor(c, focus_dist_world));
		}
	}
	// Camera motion blur (pbrt-v4's real ActiveTransform "StartTime"/
	// "EndTime" idiom, c.isAnimated - see pbrt_flatten::Camera::isAnimated's
	// own comment). Perspective-only, matching CPU's camera_is_animated
	// (mutually exclusive with an alt camera model there too - the
	// `c.type != "perspective"` block below is skipped by the `return`
	// this branch takes). Uses c.lookfrom directly, NOT the possibly-
	// force_camera_override-substituted `lookfrom` local above: an
	// animated camera always uses its own registered keyframes regardless
	// of an override, exactly like D13's own case in build_scene() and
	// CPU's applyCameraConfig() (cpu_interface.cpp) - a moving camera has
	// no single "current position" for an override to mean.
	if (c.isAnimated && c.type != "perspective") {
		// Mirrors camera.h's own camera_is_animated + alt-camera-model
		// warning: the alternate camera model (handled by the `c.type !=
		// "perspective"` block below) takes priority and motion blur is
		// NOT applied.
		std::cerr << "Warning: this scene's camera is animated together with a "
					 "non-perspective Camera type (\"" << c.type << "\") - the "
					 "alternate camera model takes priority and motion blur will "
					 "NOT be applied.\n";
	} else if (c.isAnimated && out_camera_extra) {
		const float3 animLookfrom0 = make_float3(
			static_cast<float>(c.lookfrom[0]), static_cast<float>(c.lookfrom[1]), static_cast<float>(c.lookfrom[2]));
		const float3 animLookfrom1 = make_float3(
			static_cast<float>(c.lookfrom1[0]), static_cast<float>(c.lookfrom1[1]), static_cast<float>(c.lookfrom1[2]));
		// Uses c.lookat directly, NOT the possibly-has_custom_lookat-substituted
		// `lookat` local above - same reason animLookfrom0 uses c.lookfrom
		// directly just above: an animated camera always uses its own
		// registered keyframes regardless of an override.
		const float3 animLookat0 = make_float3(
			static_cast<float>(c.lookat[0]), static_cast<float>(c.lookat[1]), static_cast<float>(c.lookat[2]));
		const float3 animLookat1 = make_float3(
			static_cast<float>(c.lookat1[0]), static_cast<float>(c.lookat1[1]), static_cast<float>(c.lookat1[2]));
		build_gpu_animated_camera_params(animLookfrom0, animLookat0, animLookfrom1, animLookat1,
			vup, static_cast<float>(c.vfov), aspect, defocus_angle_deg, focus_dist_world,
			camera_params, out_camera_extra);
		return true;
	}
	// pbrt-v4 Camera "perspective" "float screenwindow" - mirrors CPU's own
	// camera::has_screen_window/screen_window (camera.h). float sw[4] (not
	// double) matches build_pinhole_camera_params()'s own screen_window
	// param type; screenWindowPtr stays null for the overwhelmingly common
	// "no screenwindow directive" case, exactly like CPU's has_screen_window
	// gate.
	float sw[4];
	const float* screenWindowPtr = nullptr;
	if (c.hasScreenWindow) {
		sw[0] = static_cast<float>(c.screenWindow[0]);
		sw[1] = static_cast<float>(c.screenWindow[1]);
		sw[2] = static_cast<float>(c.screenWindow[2]);
		sw[3] = static_cast<float>(c.screenWindow[3]);
		screenWindowPtr = sw;
	}
	if (defocus_angle_deg > 0.0f) {
		float3 dof_u, dof_v;
		build_pinhole_camera_params(
			lookfrom, lookat, vup, static_cast<float>(c.vfov), aspect,
			focus_dist_world, camera_params, &dof_u, &dof_v, nullptr, screenWindowPtr);
		if (out_camera_extra) {
			// A nonzero defocus disk opts this scene out of
			// optix_interface.cpp's generic camera_params->cameraExtra
			// fallback (see CameraKind::Perspective's own comment there), so
			// kind/origin/lower_left_corner/horizontal/vertical must be set
			// explicitly here too - same full set case 1/the generic pbrt
			// Depth-of-Field (D1) path set.
			out_camera_extra->kind = CameraKind::Perspective;
			out_camera_extra->origin = lookfrom;
			out_camera_extra->lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
			out_camera_extra->horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
			out_camera_extra->vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);

			constexpr float kPi = 3.14159265358979323846f;
			const float defocus_radius = focus_dist_world * tanf((defocus_angle_deg * kPi / 180.0f) / 2.0f);
			out_camera_extra->defocus_disk_u = make_float3(dof_u.x * defocus_radius, dof_u.y * defocus_radius, dof_u.z * defocus_radius);
			out_camera_extra->defocus_disk_v = make_float3(dof_v.x * defocus_radius, dof_v.y * defocus_radius, dof_v.z * defocus_radius);
		}
	} else {
		build_pinhole_camera_params(
			lookfrom, lookat, vup, static_cast<float>(c.vfov), aspect,
			1.0f, camera_params, nullptr, nullptr, nullptr, screenWindowPtr);
	}

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
			// pbrt-v4's own (left-handed) LookAt basis, matching CPU's
			// OrthographicCamera (src/shared/cameras.h, built from
			// make_look_at) and this same file's spherical/realistic cases
			// below: right = cross(up, forward) = cross(w, up) (forward = -w),
			// up' = cross(right, w). This block previously used the RTiOW
			// perspective basis (u = cross(up, w), the NEGATIVE of this), which
			// mirrored every GPU orthographic render left-to-right relative to
			// CPU - found by the Phase 2 Cameras-category sweep (D6, see
			// tests/integration/material_cpu_gpu_parity_tests.cpp).
			const float3 u = normalize(cross(w, vup));
			const float3 v = cross(u, w);
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
			out_camera_extra->sphericalMapping = (c.sphericalMapping == "equalarea") ? 1 : 0;
		} else if (c.type == "realistic") {
			// c.lensFile is guaranteed non-empty here - pbrt_flatten.h already
			// fell back to "perspective" at flatten() time if the scene gave
			// none (see Camera's own comment). A missing/malformed file on
			// disk is still possible and only detectable here, where actual
			// file access happens.
			//
			// The expensive part - RealisticCamera's constructor, which
			// traces 64 slabs x 1024 samples of real lens-refraction rays to
			// bound the exit pupil per element (bound_exit_pupil(),
			// realistic_camera.h) - depends only on the LENS SYSTEM (lens
			// data, half-film-extents, focus distance, aperture diameter),
			// never on camera_to_world (confirmed by reading that class:
			// camera_to_world_ is only ever read by world_origin/right/up/
			// forward() and generate_ray(), never by the constructor or the
			// exit-pupil-bounding it does). This function reruns on every
			// Live Preview frame the camera moves, so - like every other
			// expensive per-triangle/per-file cost in this file - the lens
			// system's own build output (lensElements/exitPupilBounds/
			// filmHalf{X,Y}/lensRearZ) is cached here, keyed by everything
			// that determines it; only the cheap world-space basis vectors
			// (a plain 4x3 matrix-transform of the CURRENT ctw, computed
			// fresh below without needing a RealisticCamera object at all -
			// the identical math world_origin()/etc. themselves wrap) still
			// depend on the moving camera and are recomputed every call.
			struct RealisticCameraLensData {
				std::vector<GpuLensElement> lensElements;
				std::vector<GpuExitPupilBounds> exitPupilBounds;
				float filmHalfX = 0.0f, filmHalfY = 0.0f, lensRearZ = 0.0f;
			};
			static std::unordered_map<std::string, RealisticCameraLensData> s_realisticLensCache;
			static std::mutex s_realisticLensCacheMutex;

			const float aspectF = (image_height > 0)
				? static_cast<float>(image_width) / static_cast<float>(image_height) : 1.0f;
			const float halfY = static_cast<float>(c.filmDiagonalMM) / (2.0f * std::sqrt(aspectF * aspectF + 1.0f));
			const float halfX = aspectF * halfY;
			const float focusDist = static_cast<float>(pbrt_flatten::focusDistanceFor(c));
			const float apertureDiameter = static_cast<float>(c.apertureDiameterMM);
			const std::string lensCacheKey = std::string(path) + "|" + c.lensFile +
				"|hx" + float_bits_key(halfX) + "|hy" + float_bits_key(halfY) +
				"|fd" + float_bits_key(focusDist) + "|ap" + float_bits_key(apertureDiameter);

			bool lensFileMissing = false, lensFileEmpty = false;
			const RealisticCameraLensData* lensData = get_or_build_cached(
				s_realisticLensCache, s_realisticLensCacheMutex, lensCacheKey,
				[&](RealisticCameraLensData& out) -> bool {
					std::string lensText;
					if (!pbrt_load::loadFileNear(path, c.lensFile, lensText)) {
						lensFileMissing = true;
						return false;
					}
					const std::vector<double> lensD = pbrt_load::parseLensFile(lensText);
					if (lensD.empty()) {
						lensFileEmpty = true;
						return false;
					}
					std::vector<float> lens;
					lens.reserve(lensD.size());
					for (double v : lensD) lens.push_back(static_cast<float>(v));
					// ctw doesn't matter here - identity is fine, since none
					// of the fields read below depend on it (see this
					// block's own comment above).
					RealisticCamera<float> realCam(Mat4<float>{}, halfX, halfY, focusDist, apertureDiameter, lens);
					for (int i = 0; i < realCam.num_elements(); ++i) {
						GpuLensElement le{};
						le.curvatureRadius = realCam.lens_curvature_radius(i);
						le.thickness       = realCam.lens_thickness(i);
						le.eta              = realCam.lens_eta(i);
						le.apertureRadius   = realCam.lens_aperture_radius(i);
						out.lensElements.push_back(le);
					}
					for (int i = 0; i < realCam.num_exit_pupil_bounds(); ++i) {
						GpuExitPupilBounds b{};
						b.xMin = realCam.exit_pupil_xmin(i);
						b.xMax = realCam.exit_pupil_xmax(i);
						b.yMin = realCam.exit_pupil_ymin(i);
						b.yMax = realCam.exit_pupil_ymax(i);
						b.degenerate = realCam.exit_pupil_degenerate(i) ? 1 : 0;
						out.exitPupilBounds.push_back(b);
					}
					out.filmHalfX = realCam.film_half_x();
					out.filmHalfY = realCam.film_half_y();
					out.lensRearZ = realCam.lens_rear_z();
					return true;
				});

			if (lensFileMissing) {
				std::cerr << "[OptiX] warning: " << path << ": realistic camera lensfile '"
					  << c.lensFile << "' not found; rendering as perspective instead\n";
			} else if (lensFileEmpty) {
				std::cerr << "[OptiX] warning: " << path << ": realistic camera lensfile '"
					  << c.lensFile << "' has no usable rows; rendering as perspective instead\n";
			} else if (lensData) {
				scene.lensElements = lensData->lensElements;
				scene.exitPupilBounds = lensData->exitPupilBounds;

				const Mat4<float> ctw = make_look_at<float>(
					lookfrom.x, lookfrom.y, lookfrom.z,
					lookat.x, lookat.y, lookat.z,
					vup.x, vup.y, vup.z);
				const CamVec3<float> wo = ctw.transform_point(0.0f, 0.0f, 0.0f);
				const CamVec3<float> wr = ctw.transform_vec(1.0f, 0.0f, 0.0f);
				const CamVec3<float> wu = ctw.transform_vec(0.0f, 1.0f, 0.0f);
				const CamVec3<float> wf = ctw.transform_vec(0.0f, 0.0f, 1.0f);

				out_camera_extra->kind = CameraKind::Realistic;
				out_camera_extra->origin = make_float3(wo.x, wo.y, wo.z);
				out_camera_extra->su = make_float3(wr.x, wr.y, wr.z);
				out_camera_extra->sv = make_float3(wu.x, wu.y, wu.z);
				out_camera_extra->sw = make_float3(wf.x, wf.y, wf.z);
				out_camera_extra->film_half_x = lensData->filmHalfX;
				out_camera_extra->film_half_y = lensData->filmHalfY;
				out_camera_extra->lens_rear_z = lensData->lensRearZ;
				out_camera_extra->numLensElements = static_cast<int>(scene.lensElements.size());
				out_camera_extra->numExitPupilBounds = static_cast<int>(scene.exitPupilBounds.size());
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
	bool force_camera_override,
	bool has_custom_lookat,
	double lookat_x,
	double lookat_y,
	double lookat_z,
	bool has_dof_override,
	double aperture_override,
	double focus_distance_override
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

	// Every scene is pbrt-backed: the registry hands back the .pbrt file for this id and
	// build_loaded_pbrt_scene() builds geometry, materials, lights and camera from it. (This
	// function used to be a switch over ~80 hand-written per-scene builders; they were
	// migrated to pbrt_scenes/*.pbrt one batch at a time and the last went with the H1-H12
	// environment scenes - see git history for the old cases.)
	const char* pbrtPath = cpu_scene_pbrt_path_by_id(scene_id);
	if (pbrtPath && pbrtPath[0] != '\0') {
		// A scene whose camera the user controls (the Cornell-box family) keeps honouring
		// --cam_x/y/z and Live Preview's free-fly camera even when the caller did not ask
		// for an override explicitly.
		const bool effectiveForceOverride = force_camera_override ||
			(cpu_scene_camera_is_user_controlled_by_id(scene_id) != 0);
		return build_loaded_pbrt_scene(
			pbrtPath, scene, camera_params,
			image_width, image_height,
			cam_x, cam_y, cam_z, effectiveForceOverride,
			has_custom_lookat, lookat_x, lookat_y, lookat_z,
			out_camera_extra,
			has_dof_override, aperture_override, focus_distance_override);
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

