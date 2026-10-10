// metal_poc_material_ids.metal - the names of the `materialType` values, shared by the host (#included by the .mm/.h files) and the shader
// (the first file of the concatenated shader source, see metal_poc_shader_files.h).
//
// `materialType` is a plain uint32 in every material struct (TriangleMaterial, SphereMaterial, ...), and the shader and the loader used to
// compare and assign it with bare numbers (`materialType == 25u`), which only the long comment above `TriangleMaterial` explained. These
// are macros, not an enum, because the same text has to be valid C++ and MSL and survive being pasted into a runtime-compiled string.
// The values are part of the host/shader contract: change one here and both sides change together; never reuse a number for something else.
//
// This is Metal's own numbering. The OptiX backend has its own `MaterialType` enum (gpu/optix/optix_types.h) with different values - they
// are separate contracts, and the CPU/OptiX scene is mapped onto either by the pbrt loaders, never by number.
#ifndef METAL_POC_MATERIAL_IDS
#define METAL_POC_MATERIAL_IDS

#define METAL_MAT_LAMBERTIAN                 0u    // diffuse; `color` is the albedo
#define METAL_MAT_MIRROR                     1u    // perfect specular, tinted by `color`
#define METAL_MAT_DIELECTRIC                 2u    // smooth glass; `color` is a Beer-Lambert absorption coefficient
#define METAL_MAT_TEXTURED_LAMBERTIAN        3u    // diffuse with the `earthTexture` image as albedo
#define METAL_MAT_ROUGH_CONDUCTOR            4u    // GGX conductor; `color` is the normal-incidence reflectance
#define METAL_MAT_ROUGH_DIELECTRIC           5u    // frosted glass (VNDF-perturbed reflect/refract)
#define METAL_MAT_CHECKER_LAMBERTIAN         6u    // 2D checkerboard albedo, tile B a fixed fraction of A
#define METAL_MAT_BUMP_LAMBERTIAN            7u    // diffuse with a procedurally bump-mapped shading normal
#define METAL_MAT_CLEARCOAT                  8u    // smooth dielectric coat over a Lambertian base
#define METAL_MAT_CHECKER_ROUGH_CONDUCTOR    9u    // GGX conductor whose roughness comes from a UV checker (spheres only)
#define METAL_MAT_PATTERNED_EMISSIVE         10u   // area-light surface with a checkerboard emission pattern
#define METAL_MAT_THIN_DIELECTRIC            11u   // zero-thickness glass slab (pbrt ThinDielectricBxDF)
#define METAL_MAT_DIFFUSE_TRANSMISSION       12u   // diffuse reflection + diffuse transmission (conductorK.y selects an image variant)
#define METAL_MAT_OREN_NAYAR                 13u   // rough diffuse (Oren-Nayar)
#define METAL_MAT_VELVET                     14u   // velvet-like retroreflective diffuse
#define METAL_MAT_IMAGE_EMISSIVE             15u   // area-light surface whose emission is an image
#define METAL_MAT_CHECKER3D_LAMBERTIAN       16u   // 3D world-space checkerboard albedo
#define METAL_MAT_MARBLE_LAMBERTIAN          17u   // Perlin-noise marble albedo
#define METAL_MAT_NORMALIZED_FRESNEL         18u   // pbrt NormalizedFresnelBxDF
#define METAL_MAT_COATED_DIFFUSE             19u   // pbrt LayeredBxDF: rough dielectric coat over a diffuse base
#define METAL_MAT_COATED_CONDUCTOR           20u   // pbrt LayeredBxDF: rough dielectric coat over a GGX conductor
#define METAL_MAT_NORMAL_MAPPED_LAMBERTIAN   21u   // diffuse with a checker-driven tangent-space normal map
#define METAL_MAT_DISPERSIVE_DIELECTRIC      22u   // smooth glass with wavelength-dependent IOR
#define METAL_MAT_DISPERSIVE_ROUGH_DIELECTRIC 23u  // frosted glass with wavelength-dependent IOR
#define METAL_MAT_PRINCIPLED                 24u   // Disney-style principled BSDF
#define METAL_MAT_TEXTURE_FAMILY             25u   // procedural/image textures on a Diffuse; the variant is picked by `conductorK.y`
#define METAL_MAT_IMAGE_LAMBERTIAN           26u   // pbrt Diffuse whose reflectance is an image map
#define METAL_MAT_IMAGE_CLEARCOAT            27u   // pbrt CoatedDiffuse whose reflectance is an image map
#define METAL_MAT_MEDIUM_HOMOGENEOUS         28u   // sphere/cylinder whose interior is a homogeneous participating medium
#define METAL_MAT_MEDIUM_HETEROGENEOUS       29u   // sphere whose interior is a procedural (cloud) medium
#define METAL_MAT_MEDIUM_RGB_GRID            30u   // sphere whose interior is a per-voxel RGB grid medium
#define METAL_MAT_HAIR                       31u   // Marschner/Chiang fiber scattering
#define METAL_MAT_MEASURED                   32u   // tabulated (Dupuy-Jakob) measured BRDF
#define METAL_MAT_SUBSURFACE                 34u   // pbrt SubsurfaceMaterial: dielectric interface + tabulated BSSRDF (table offset in conductorEta.x)
#define METAL_MAT_MIX                        33u   // pbrt MixMaterial: two full materials in the shared float buffer (bumpOffset), roughness = P(second)

#endif
