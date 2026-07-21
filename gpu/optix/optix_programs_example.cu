// Example OptiX Ray Generation Program for ray_tracer
// This replaces gpu/cuda/gpu_interface.cu when using OptiX
// Reference: Based on pbrt-v4's wavefront architecture

#include <optix.h>
#include <cuda_runtime.h>

// Launch parameters passed from host
struct LaunchParams {
	// Framebuffer
	float3* framebuffer;
	int width;
	int height;
	int samplesPerPixel;
	int maxDepth;

	// Camera
	struct {
		float3 origin;
		float3 lower_left_corner;
		float3 horizontal;
		float3 vertical;
	} camera;

	// Scene
	OptixTraversableHandle traversable;  // BVH handle

	// Random seed
	unsigned int frameNumber;
};

// Payload for ray tracing (passed through optixTrace)
struct PayloadRadiance {
	float3 emission;      // Light emitted at hit point
	float3 attenuation;   // Color filter (albedo)
	float3 origin;        // Next ray origin
	float3 direction;     // Next ray direction
	int depth;            // Current path depth
	bool done;            // True if path terminated
	unsigned int packed;  // Pack/unpack for optixTrace
};

// Helper to pack/unpack payload (OptiX passes data as uint32s)
static __forceinline__ __device__ void packPayload(PayloadRadiance* payload, unsigned int& p0, unsigned int& p1) {
	// Pack payload data into uint32s for optixTrace
	// Details omitted for brevity
}

static __forceinline__ __device__ void unpackPayload(unsigned int p0, unsigned int p1, PayloadRadiance* payload) {
	// Unpack payload from optixTrace
}

// Random number generator (per-thread)
__device__ unsigned int tea(unsigned int val0, unsigned int val1) {
	unsigned int v0 = val0;
	unsigned int v1 = val1;
	unsigned int s0 = 0;

	for(unsigned int n = 0; n < 4; n++) {
		s0 += 0x9e3779b9;
		v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
		v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
	}

	return v0;
}

__device__ float rnd(unsigned int& seed) {
	seed = tea(seed, seed);
	return float(seed) / float(0xffffffffu);
}

__device__ float3 randomInUnitSphere(unsigned int& seed) {
	while (true) {
		float3 p = make_float3(
			rnd(seed) * 2.0f - 1.0f,
			rnd(seed) * 2.0f - 1.0f,
			rnd(seed) * 2.0f - 1.0f
		);
		if (dot(p, p) < 1.0f)
			return p;
	}
}

__device__ float3 randomUnitVector(unsigned int& seed) {
	return normalize(randomInUnitSphere(seed));
}

// Main ray generation program (OptiX entry point)
extern "C" __global__ void __raygen__rg() {
	// Get pixel coordinates
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	const int x = idx.x;
	const int y = idx.y;

	// Get launch parameters
	const LaunchParams* params = (LaunchParams*)optixGetSbtDataPointer();

	// Initialize random seed (per pixel, per frame)
	unsigned int seed = tea(x * dim.y + y, params->frameNumber);

	// Accumulate samples for this pixel
	float3 pixelColor = make_float3(0, 0, 0);

	for (int sample = 0; sample < params->samplesPerPixel; sample++) {
		// Generate camera ray with anti-aliasing jitter
		float u = (float(x) + rnd(seed)) / float(dim.x);
		float v = (float(y) + rnd(seed)) / float(dim.y);

		float3 rayOrigin = params->camera.origin;
		float3 rayDirection = params->camera.lower_left_corner +
							 u * params->camera.horizontal +
							 v * params->camera.vertical - rayOrigin;
		rayDirection = normalize(rayDirection);

		// Path tracing loop
		float3 color = make_float3(0, 0, 0);
		float3 throughput = make_float3(1, 1, 1);

		for (int depth = 0; depth < params->maxDepth; depth++) {
			// Prepare payload
			PayloadRadiance payload;
			payload.depth = depth;
			payload.done = false;

			unsigned int p0, p1;
			packPayload(&payload, p0, p1);

			// Trace ray through scene
			// OptiX will call appropriate closest-hit or miss program
			optixTrace(
				params->traversable,        // Scene BVH
				rayOrigin,                  // Ray origin
				rayDirection,               // Ray direction
				0.001f,                     // tmin (avoid self-intersection)
				1e16f,                      // tmax
				0.0f,                       // ray time
				OptixVisibilityMask(255),   // Visibility mask
				OPTIX_RAY_FLAG_NONE,        // Ray flags
				0,                          // SBT offset
				1,                          // SBT stride
				0,                          // Miss SBT index
				p0, p1                      // Payload (packed)
			);

			// Unpack payload after trace
			unpackPayload(p0, p1, &payload);

			// Accumulate emission (from lights)
			color += throughput * payload.emission;

			// If path terminated (absorbed or max depth), stop
			if (payload.done || depth >= params->maxDepth - 1) {
				break;
			}

			// Update throughput with surface albedo
			throughput *= payload.attenuation;

			// Russian roulette path termination
			if (depth > 3) {
				float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
				if (rnd(seed) > p) {
					break;  // Terminate path early
				}
				throughput *= 1.0f / p;  // Unbiased compensation
			}

			// Continue with scattered ray from payload
			rayOrigin = payload.origin;
			rayDirection = payload.direction;
		}

		// Accumulate this sample
		pixelColor += color;
	}

	// Average over samples
	pixelColor *= 1.0f / float(params->samplesPerPixel);

	// Gamma correction
	pixelColor.x = sqrtf(pixelColor.x);
	pixelColor.y = sqrtf(pixelColor.y);
	pixelColor.z = sqrtf(pixelColor.z);

	// Write to framebuffer
	const int pixelIdx = y * dim.x + x;
	params->framebuffer[pixelIdx] = pixelColor;
}

// Miss program (called when ray doesn't hit anything)
extern "C" __global__ void __miss__ms() {
	const LaunchParams* params = (LaunchParams*)optixGetSbtDataPointer();

	// Sky gradient background
	float3 rayDir = optixGetWorldRayDirection();
	float t = 0.5f * (rayDir.y + 1.0f);
	float3 color = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) +  // White
						 t * make_float3(0.5f, 0.7f, 1.0f);      // Sky blue

	// Set payload
	PayloadRadiance* payload = (PayloadRadiance*)optixGetPayloadPointer();
	payload->emission = color;
	payload->attenuation = make_float3(0, 0, 0);
	payload->done = true;  // No scattering
}

// Closest-hit program for Lambertian (diffuse) material
extern "C" __global__ void __closesthit__lambertian() {
	// Get hit data from SBT
	const HitGroupData* hitData = (HitGroupData*)optixGetSbtDataPointer();
	const float3 albedo = hitData->albedo;

	// Get hit point and normal
	const float3 rayOrigin = optixGetWorldRayOrigin();
	const float3 rayDir = optixGetWorldRayDirection();
	const float tHit = optixGetRayTmax();
	const float3 hitPoint = rayOrigin + tHit * rayDir;

	// For spheres, normal is (hitPoint - center) / radius
	// This should come from intersection program
	const float3 center = hitData->center;
	const float radius = hitData->radius;
	const float3 normal = normalize((hitPoint - center) / radius);

	// Generate random seed from hit point
	unsigned int seed = tea(
		__float_as_uint(hitPoint.x) ^ __float_as_uint(hitPoint.y),
		__float_as_uint(hitPoint.z)
	);

	// Lambertian scatter: random direction in hemisphere
	float3 scatterDir = normal + randomUnitVector(seed);

	// Handle degenerate scatter direction
	if (fabs(scatterDir.x) < 1e-8f &&
		fabs(scatterDir.y) < 1e-8f &&
		fabs(scatterDir.z) < 1e-8f) {
		scatterDir = normal;
	}

	// Set payload for next bounce
	PayloadRadiance* payload = (PayloadRadiance*)optixGetPayloadPointer();
	payload->origin = hitPoint + 0.001f * normal;  // Offset to avoid self-intersection
	payload->direction = normalize(scatterDir);
	payload->attenuation = albedo;
	payload->emission = make_float3(0, 0, 0);  // Lambertian doesn't emit
	payload->done = false;
}

// Closest-hit program for Metal material
extern "C" __global__ void __closesthit__metal() {
	const HitGroupData* hitData = (HitGroupData*)optixGetSbtDataPointer();
	const float3 albedo = hitData->albedo;
	const float fuzz = hitData->fuzz;  // Roughness

	const float3 rayOrigin = optixGetWorldRayOrigin();
	const float3 rayDir = optixGetWorldRayDirection();
	const float tHit = optixGetRayTmax();
	const float3 hitPoint = rayOrigin + tHit * rayDir;

	const float3 center = hitData->center;
	const float radius = hitData->radius;
	const float3 normal = normalize((hitPoint - center) / radius);

	// Perfect reflection
	float3 reflected = rayDir - 2.0f * dot(rayDir, normal) * normal;

	// Add fuzz (roughness)
	unsigned int seed = tea(
		__float_as_uint(hitPoint.x) ^ __float_as_uint(hitPoint.y),
		__float_as_uint(hitPoint.z)
	);
	reflected = normalize(reflected + fuzz * randomInUnitSphere(seed));

	// Check if reflection is valid (not into surface)
	bool validScatter = dot(reflected, normal) > 0;

	PayloadRadiance* payload = (PayloadRadiance*)optixGetPayloadPointer();
	if (validScatter) {
		payload->origin = hitPoint + 0.001f * normal;
		payload->direction = reflected;
		payload->attenuation = albedo;
		payload->emission = make_float3(0, 0, 0);
		payload->done = false;
	} else {
		// Absorbed
		payload->done = true;
		payload->attenuation = make_float3(0, 0, 0);
	}
}

// Closest-hit program for Dielectric (glass) material
extern "C" __global__ void __closesthit__dielectric() {
	const HitGroupData* hitData = (HitGroupData*)optixGetSbtDataPointer();
	const float ior = hitData->ior;  // Index of refraction

	const float3 rayOrigin = optixGetWorldRayOrigin();
	const float3 rayDir = optixGetWorldRayDirection();
	const float tHit = optixGetRayTmax();
	const float3 hitPoint = rayOrigin + tHit * rayDir;

	const float3 center = hitData->center;
	const float radius = hitData->radius;
	float3 outwardNormal = (hitPoint - center) / radius;

	// Determine if ray is entering or exiting
	bool frontFace = dot(rayDir, outwardNormal) < 0;
	float3 normal = frontFace ? outwardNormal : -outwardNormal;
	float etaiOverEtat = frontFace ? (1.0f / ior) : ior;

	float cosTheta = fminf(dot(-rayDir, normal), 1.0f);
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	bool cannotRefract = etaiOverEtat * sinTheta > 1.0f;

	unsigned int seed = tea(
		__float_as_uint(hitPoint.x) ^ __float_as_uint(hitPoint.y),
		__float_as_uint(hitPoint.z)
	);

	// Schlick's approximation for Fresnel
	float r0 = (1.0f - etaiOverEtat) / (1.0f + etaiOverEtat);
	r0 = r0 * r0;
	float reflectProb = r0 + (1.0f - r0) * powf((1.0f - cosTheta), 5.0f);

	float3 direction;
	if (cannotRefract || rnd(seed) < reflectProb) {
		// Reflect
		direction = normalize(rayDir - 2.0f * dot(rayDir, normal) * normal);
	} else {
		// Refract
		float3 perpendicular = etaiOverEtat * (rayDir + cosTheta * normal);
		float3 parallel = -sqrtf(fabsf(1.0f - dot(perpendicular, perpendicular))) * normal;
		direction = normalize(perpendicular + parallel);
	}

	PayloadRadiance* payload = (PayloadRadiance*)optixGetPayloadPointer();
	payload->origin = hitPoint + 0.001f * direction;
	payload->direction = direction;
	payload->attenuation = make_float3(1, 1, 1);  // Glass doesn't absorb (clear)
	payload->emission = make_float3(0, 0, 0);
	payload->done = false;
}

// Emissive material (light source)
extern "C" __global__ void __closesthit__emissive() {
	const HitGroupData* hitData = (HitGroupData*)optixGetSbtDataPointer();
	const float3 emission = hitData->emission;

	PayloadRadiance* payload = (PayloadRadiance*)optixGetPayloadPointer();
	payload->emission = emission;
	payload->attenuation = make_float3(0, 0, 0);
	payload->done = true;  // Lights don't scatter rays
}
