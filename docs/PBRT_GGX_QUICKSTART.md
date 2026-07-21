# Quick Start: Add GGX Microfacet BRDF

## Goal
Add physically-based glossy materials (plastic, car paint, brushed metal) in one day.

---

## Step 1: Add GGX Distribution Function

**File**: `src/TheRestOfYourLife/material.h`

```cpp
// Add after line 46 (after material base class)

// GGX/Trowbridge-Reitz microfacet distribution
__device__ __host__ inline float ggx_distribution(float roughness, float NdotH) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH2 = NdotH * NdotH;

	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = M_PI * denom * denom;

	return a2 / denom;
}

// Smith geometric shadowing term for GGX
__device__ __host__ inline float ggx_geometry_schlick(float roughness, float NdotV) {
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	return NdotV / (NdotV * (1.0 - k) + k);
}

__device__ __host__ inline float ggx_geometry_smith(
	float roughness, float NdotV, float NdotL
) {
	return ggx_geometry_schlick(roughness, NdotV) 
		 * ggx_geometry_schlick(roughness, NdotL);
}

// Schlick Fresnel approximation
__device__ __host__ inline vec3 fresnel_schlick(float cosTheta, const vec3& F0) {
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// Microfacet BRDF class
class microfacet : public material {
  public:
	microfacet(const color& albedo, float roughness, float metallic)
	  : albedo(albedo), roughness(roughness), metallic(metallic) {}

	bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
		// View direction
		vec3 V = unit_vector(-r_in.direction());
		vec3 N = rec.normal;

		// Sample microfacet normal (GGX importance sampling)
		float r1 = random_double();
		float r2 = random_double();

		float a = roughness * roughness;
		float phi = 2.0 * M_PI * r1;
		float cosTheta = sqrt((1.0 - r2) / (1.0 + (a*a - 1.0) * r2));
		float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

		// Half vector in tangent space
		vec3 H_tangent = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

		// Transform to world space (assume N = (0,0,1) for now; needs TBN matrix)
		vec3 H = H_tangent; // TODO: transform by TBN

		// Reflect view direction around microfacet normal
		vec3 L = reflect(-V, H);

		if (dot(L, N) <= 0.0) return false; // Below surface

		// Compute BRDF
		float NdotL = fmax(dot(N, L), 0.0);
		float NdotV = fmax(dot(N, V), 0.0);
		float NdotH = fmax(dot(N, H), 0.0);
		float VdotH = fmax(dot(V, H), 0.0);

		// F0 (reflectance at normal incidence)
		vec3 F0 = mix(vec3(0.04), albedo, metallic);

		// Cook-Torrance specular BRDF
		float D = ggx_distribution(roughness, NdotH);
		float G = ggx_geometry_smith(roughness, NdotV, NdotL);
		vec3 F = fresnel_schlick(VdotH, F0);

		vec3 specular = (D * G * F) / fmax(4.0 * NdotV * NdotL, 0.001);

		// Diffuse component (for non-metals)
		vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
		vec3 diffuse = kD * albedo / M_PI;

		srec.attenuation = diffuse + specular;
		srec.skip_pdf = false;
		srec.pdf_ptr = make_shared<cosine_pdf>(rec.normal); // TODO: use GGX PDF

		return true;
	}

  private:
	color albedo;
	float roughness;  // 0.0 = mirror, 1.0 = diffuse
	float metallic;   // 0.0 = dielectric, 1.0 = metal
};
```

---

## Step 2: Test Scene

**File**: `src/TheRestOfYourLife/scenes.h` (add new scene)

```cpp
// Scene 6: GGX Material Test
inline void build_ggx_test(hittable_list& world, camera& cam, int image_width) {
	// Ground
	auto ground = make_shared<lambertian>(color(0.8, 0.8, 0.8));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground));

	// Varying roughness (left to right: 0.0 → 1.0)
	auto mat_smooth   = make_shared<microfacet>(color(0.8, 0.6, 0.2), 0.0, 1.0);
	auto mat_medium   = make_shared<microfacet>(color(0.8, 0.6, 0.2), 0.3, 1.0);
	auto mat_rough    = make_shared<microfacet>(color(0.8, 0.6, 0.2), 0.7, 1.0);
	auto mat_diffuse  = make_shared<microfacet>(color(0.8, 0.6, 0.2), 1.0, 1.0);

	world.add(make_shared<sphere>(point3(-3, 1, 0), 1.0, mat_smooth));
	world.add(make_shared<sphere>(point3(-1, 1, 0), 1.0, mat_medium));
	world.add(make_shared<sphere>(point3( 1, 1, 0), 1.0, mat_rough));
	world.add(make_shared<sphere>(point3( 3, 1, 0), 1.0, mat_diffuse));

	// Camera
	point3 lookfrom(0, 2, 10);
	point3 lookat(0, 1, 0);
	vec3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	auto aperture = 0.0;
	int image_height = int(image_width / 16.0 * 9.0);

	cam = camera(lookfrom, lookat, vup, 40, double(image_width)/image_height, aperture, dist_to_focus);
}
```

---

## Step 3: GPU Port (OptiX)

**File**: `gpu/optix/optix_programs.cu`

```cuda
// Add MaterialType::Microfacet to enum (line ~12)
enum class MaterialType : int {
	Lambertian = 0,
	Metal = 1,
	Dielectric = 2,
	DiffuseLight = 3,
	Microfacet = 4  // NEW
};

// Add to MaterialData struct (line ~20)
struct MaterialData {
	MaterialType type;
	float3 albedo;
	float fuzz_or_roughness;  // Renamed to support both
	float ior_or_metallic;    // Renamed to support both
	float3 emission;
};

// Add GGX functions (lines ~72-120)
__device__ __forceinline__ float ggx_distribution(float roughness, float NdotH) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH2 = NdotH * NdotH;
	float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
	denom = M_PI * denom * denom;
	return a2 / denom;
}

__device__ __forceinline__ float ggx_geometry_schlick(float roughness, float NdotV) {
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;
	return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ __forceinline__ float ggx_geometry_smith(float roughness, float NdotV, float NdotL) {
	return ggx_geometry_schlick(roughness, NdotV) * ggx_geometry_schlick(roughness, NdotL);
}

__device__ __forceinline__ float3 fresnel_schlick(float cosTheta, const float3& F0) {
	float p = powf(1.0f - cosTheta, 5.0f);
	return F0 + (make_float3(1.0f) - F0) * p;
}

// In __closesthit__sphere(), add case for Microfacet (line ~400+)
case MaterialType::Microfacet: {
	float roughness = mat.fuzz_or_roughness;
	float metallic = mat.ior_or_metallic;

	float3 V = normalize(-ray_dir);
	float3 N = normal;

	// Sample microfacet normal (simplified; use GGX importance sampling)
	float3 H = random_unit_vector(seed); // TODO: proper GGX sampling
	float3 L = reflect(-V, H);

	if (dot(L, N) <= 0.0f) {
		// Below surface, terminate
		prd.attenuation = make_float3(0.0f);
		return;
	}

	float NdotL = fmaxf(dot(N, L), 0.0f);
	float NdotV = fmaxf(dot(N, V), 0.0f);
	float NdotH = fmaxf(dot(N, H), 0.0f);
	float VdotH = fmaxf(dot(V, H), 0.0f);

	// F0 (lerp between dielectric 0.04 and albedo for metals)
	float3 F0 = (1.0f - metallic) * make_float3(0.04f) + metallic * mat.albedo;

	// Cook-Torrance
	float D = ggx_distribution(roughness, NdotH);
	float G = ggx_geometry_smith(roughness, NdotV, NdotL);
	float3 F = fresnel_schlick(VdotH, F0);

	float3 specular = (D * G * F) / fmaxf(4.0f * NdotV * NdotL, 0.001f);
	float3 kD = (make_float3(1.0f) - F) * (1.0f - metallic);
	float3 diffuse = kD * mat.albedo / M_PI;

	prd.attenuation = diffuse + specular;
	prd.origin = hit_point;
	prd.direction = L;
	break;
}
```

---

## Expected Result

You should see 4 gold spheres with varying glossiness:
- **Left**: Mirror-like (roughness = 0.0)
- **Middle-left**: Glossy plastic (roughness = 0.3)
- **Middle-right**: Brushed metal (roughness = 0.7)
- **Right**: Nearly diffuse (roughness = 1.0)

---

## Next Steps

1. **Improve sampling**: Replace `random_unit_vector(seed)` with proper GGX importance sampling
2. **Add TBN matrix**: For correct normal mapping
3. **Implement MIS**: Combine BRDF + light sampling for faster convergence
4. **Texture support**: Allow roughness/metallic maps

---

## References

- pbrt-v4: `src/pbrt/util/scattering.h` (GGX distribution)
- Karis 2013: "Real Shading in Unreal Engine 4" (Epic Games SIGGRAPH course)
- Burley 2012: "Physically-Based Shading at Disney" (SIGGRAPH course notes)
