#pragma once

// PODs shared between host and device for simple sphere-only scenes
struct SpherePOD {
	float cx, cy, cz;
	float radius;
	int matIndex;
};

struct MaterialPOD {
	int type; // 0 = lambertian, 1 = metal, 2 = dielectric, 3 = diffuse_light
	float r, g, b; // color/albedo or emission
	float fuzz;    // for metal
	float ref_idx; // for dielectric
	int is_emissive; // 0 = no, 1 = yes
};

struct CameraPOD {
	float lookfrom_x, lookfrom_y, lookfrom_z;
	float lookat_x, lookat_y, lookat_z;
	float vup_x, vup_y, vup_z;
	float vfov; // vertical field of view degrees
	float aspect_ratio;
	float defocus_angle;
};

struct QuadPOD {
	float Qx, Qy, Qz;
	float ux, uy, uz;
	float vx, vy, vz;
	int matIndex;
};

// For simplicity, expose counts for arrays passed to CUDA kernel
struct ScenePODHeader {
	int nspheres;
	int nquads;
	int nmaterials;
};

// Forward declare; implementation in scene_serializer.cpp (needs full camera definition)
class camera;
CameraPOD CameraToPOD(const camera &cam);

