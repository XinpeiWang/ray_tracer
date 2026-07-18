#include "scene_serializer.h"
#include "cuda_scene.h"

// C++ scene headers (safe for C++ compiler, not CUDA compiler)
#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/TheRestOfYourLife/vec3.h"
#include "../../src/TheRestOfYourLife/ray.h"
#include "../../src/TheRestOfYourLife/interval.h"
#include "../../src/TheRestOfYourLife/hittable.h"
#include "../../src/TheRestOfYourLife/hittable_list.h"
#include "../../src/TheRestOfYourLife/sphere.h"
#include "../../src/TheRestOfYourLife/quad.h"
#include "../../src/TheRestOfYourLife/material.h"
#include "../../src/TheRestOfYourLife/texture.h"
#include "../../src/TheRestOfYourLife/camera.h"
#include "../../src/TheRestOfYourLife/cornell_box_scene.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

// Implementation of CameraToPOD (moved from cuda_scene.h to avoid nvcc seeing incomplete camera type)
static CameraPOD CameraToPOD(const camera &cam) {
	CameraPOD c{};
	c.lookfrom_x = static_cast<float>(cam.lookfrom.x());
	c.lookfrom_y = static_cast<float>(cam.lookfrom.y());
	c.lookfrom_z = static_cast<float>(cam.lookfrom.z());
	c.lookat_x = static_cast<float>(cam.lookat.x());
	c.lookat_y = static_cast<float>(cam.lookat.y());
	c.lookat_z = static_cast<float>(cam.lookat.z());
	c.vup_x = static_cast<float>(cam.vup.x());
	c.vup_y = static_cast<float>(cam.vup.y());
	c.vup_z = static_cast<float>(cam.vup.z());
	c.vfov = static_cast<float>(cam.vfov);
	c.aspect_ratio = static_cast<float>(cam.aspect_ratio);
	c.defocus_angle = static_cast<float>(cam.defocus_angle);
	return c;
}

extern "C" void serialize_scene_arrays(
	int image_width, int image_height, int samples_per_pixel, int max_depth,
	SpherePOD** out_spheres, int* out_num_spheres,
	QuadPOD** out_quads, int* out_num_quads,
	MaterialPOD** out_materials, int* out_num_materials,
	CameraPOD* out_camera
) {
	std::vector<SpherePOD> spheres;
	std::vector<QuadPOD> quads;
	std::vector<MaterialPOD> materials;

	// Build the Cornell box scene using the shared scene builder
	hittable_list world = build_cornell_box_scene();

	// Setup camera
	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = image_width;
	cam.samples_per_pixel = samples_per_pixel;
	cam.max_depth = max_depth;
	cam.background = color(0,0,0);
	cam.vfov = 40;
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 0);
	cam.vup = vec3(0,1,0);
	cam.defocus_angle = 0;

	CameraPOD cam_pod = CameraToPOD(cam);

	// Recursive flattening: walk hittable_list, unwrap translate and rotate_y, and collect spheres/quads
	std::function<void(shared_ptr<hittable>, const vec3&, double)> walk =
		[&](shared_ptr<hittable> o, const vec3& acc_offset, double acc_angle) {
			if (auto hl = std::dynamic_pointer_cast<hittable_list>(o)) {
				for (const auto& child : hl->objects) {
					walk(child, acc_offset, acc_angle);
				}
				return;
			}

			if (auto tr = std::dynamic_pointer_cast<translate>(o)) {
				walk(tr->get_object(), acc_offset + tr->get_offset(), acc_angle);
				return;
			}

			if (auto ry = std::dynamic_pointer_cast<rotate_y>(o)) {
				walk(ry->get_object(), acc_offset, acc_angle + ry->get_angle());
				return;
			}

			auto rotate_y_vec = [&](const vec3& v) -> vec3 {
				double c = std::cos(acc_angle);
				double s = std::sin(acc_angle);
				return vec3(c * v.x() + s * v.z(), v.y(), -s * v.x() + c * v.z());
			};

			if (auto sp = std::dynamic_pointer_cast<sphere>(o)) {
				SpherePOD sph{};
				auto c = sp->center_at_time0();
				auto cr = rotate_y_vec(c) + acc_offset;
				sph.cx = static_cast<float>(cr.x());
				sph.cy = static_cast<float>(cr.y());
				sph.cz = static_cast<float>(cr.z());
				sph.radius = static_cast<float>(sp->get_radius());
				sph.matIndex = static_cast<int>(materials.size());

				MaterialPOD m{};
				auto mat = sp->get_material();
				if (auto lam = std::dynamic_pointer_cast<lambertian>(mat)) {
					m.type = 0;
					try {
						auto sc = std::dynamic_pointer_cast<solid_color>(lam->get_texture());
						auto col = sc->color_value();
						m.r = static_cast<float>(col.x());
						m.g = static_cast<float>(col.y());
						m.b = static_cast<float>(col.z());
					} catch(...) {
						m.r = 0.73f; m.g = 0.73f; m.b = 0.73f;
					}
				} else if (auto di = std::dynamic_pointer_cast<dielectric>(mat)) {
					m.type = 2;
					m.ref_idx = static_cast<float>(di->get_refraction_index());
					m.r = m.g = m.b = 1.0f;
				} else {
					m.type = 0; m.r = m.g = m.b = 0.5f;
				}

				materials.push_back(m);
				spheres.push_back(sph);
				return;
			}

			if (auto q = std::dynamic_pointer_cast<quad>(o)) {
				QuadPOD qp{};
				auto Q = rotate_y_vec(q->get_Q()) + acc_offset;
				auto uu = rotate_y_vec(q->get_u());
				auto vv = rotate_y_vec(q->get_v());
				qp.Qx = static_cast<float>(Q.x()); qp.Qy = static_cast<float>(Q.y()); qp.Qz = static_cast<float>(Q.z());
				qp.ux = static_cast<float>(uu.x()); qp.uy = static_cast<float>(uu.y()); qp.uz = static_cast<float>(uu.z());
				qp.vx = static_cast<float>(vv.x()); qp.vy = static_cast<float>(vv.y()); qp.vz = static_cast<float>(vv.z());
				qp.matIndex = static_cast<int>(materials.size());

				MaterialPOD m{};
				auto mat = q->get_material();
				if (auto lam = std::dynamic_pointer_cast<lambertian>(mat)) {
					m.type = 0;
					try {
						auto sc = std::dynamic_pointer_cast<solid_color>(lam->get_texture());
						auto col = sc->color_value();
						m.r = static_cast<float>(col.x());
						m.g = static_cast<float>(col.y());
						m.b = static_cast<float>(col.z());
					} catch(...) {
						m.r = m.g = m.b = 0.8f;
					}
				} else if (auto dl = std::dynamic_pointer_cast<diffuse_light>(mat)) {
					m.type = 3; // emissive
					m.is_emissive = true;
					try {
						auto sc = std::dynamic_pointer_cast<solid_color>(dl->get_texture());
						auto col = sc->color_value();
						m.r = static_cast<float>(col.x());
						m.g = static_cast<float>(col.y());
						m.b = static_cast<float>(col.z());
					} catch(...) {
						m.r = m.g = m.b = 1.0f;
					}
				} else {
					m.type = 0; m.r = m.g = m.b = 0.8f;
				}

				materials.push_back(m);
				quads.push_back(qp);
				return;
			}
		};

	for (const auto& obj : world.objects) {
		walk(obj, vec3(0,0,0), 0.0);
	}

	// Copy data to output arrays
	*out_num_spheres = (int)spheres.size();
	*out_num_quads = (int)quads.size();
	*out_num_materials = (int)materials.size();

	if (!spheres.empty()) {
		*out_spheres = (SpherePOD*)malloc(spheres.size() * sizeof(SpherePOD));
		std::memcpy(*out_spheres, spheres.data(), spheres.size() * sizeof(SpherePOD));
	} else {
		*out_spheres = nullptr;
	}

	if (!quads.empty()) {
		*out_quads = (QuadPOD*)malloc(quads.size() * sizeof(QuadPOD));
		std::memcpy(*out_quads, quads.data(), quads.size() * sizeof(QuadPOD));
	} else {
		*out_quads = nullptr;
	}

	if (!materials.empty()) {
		*out_materials = (MaterialPOD*)malloc(materials.size() * sizeof(MaterialPOD));
		std::memcpy(*out_materials, materials.data(), materials.size() * sizeof(MaterialPOD));
	} else {
		*out_materials = nullptr;
	}

	*out_camera = cam_pod;

	// Log serialized scene for debugging
	std::printf("[scene_serializer] Serialized scene: nspheres=%d nquads=%d nmaterials=%d\n",
		*out_num_spheres, *out_num_quads, *out_num_materials);
	std::fflush(stdout);

	if (!spheres.empty()) {
		auto& s0 = spheres[0];
		std::printf("[scene_serializer] sphere[0]: center=(%f,%f,%f) radius=%f mat=%d\n",
			s0.cx, s0.cy, s0.cz, s0.radius, s0.matIndex);
	}
	if (!quads.empty()) {
		auto& q0 = quads[0];
		std::printf("[scene_serializer] quad[0]: Q=(%f,%f,%f) u=(%f,%f,%f) v=(%f,%f,%f) mat=%d\n",
			q0.Qx, q0.Qy, q0.Qz, q0.ux, q0.uy, q0.uz, q0.vx, q0.vy, q0.vz, q0.matIndex);
	}
	std::fflush(stdout);
}

extern "C" void free_scene_arrays(SpherePOD* spheres, QuadPOD* quads, MaterialPOD* materials) {
	if (spheres) free(spheres);
	if (quads) free(quads);
	if (materials) free(materials);
}
