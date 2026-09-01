/// @file optix_renderer_scene.cpp
/// @brief OptiX Renderer Implementation -- scene/SBT build.
/// @details Split out of optix_renderer.cpp (see optix_renderer_init.cpp and
///          optix_renderer_render.cpp for the other two thirds, and
///          optix_renderer.h for the shared class declaration). This part
///          owns uploading scene geometry/materials/lights to the device,
///          building the GAS/IAS acceleration structures, and packing the
///          Shader Binding Table that ties geometry to hit-group programs.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"
#include "optix_math_helpers.h"
#include "../../src/shared/bilinear_patch.h"  // blp_area - alias-table power for GpuLightKind::BilinearPatch
#include "optix_disk_cylinder_helpers.h"  // dc_area_disk/dc_area_cylinder - alias-table power for GpuLightKind::Disk/Cylinder
#include "../../src/shared/bvh_light_sampler2.h"  // BVHLightSampler2 - host-side light-BVH tree builder (see d_lightBvhNodes_'s own comment, optix_renderer.h)
#include <cuda.h>
#include <iostream>
#include <cstring>       // memcpy, for instance transform packing
#include <type_traits>   // remove_pointer_t, for the light-flag width assert

bool OptiXRenderer::buildScene(
	const std::vector<SphereData>& spheres,
	const std::vector<QuadData>& quads,
	const std::vector<MaterialData>& materials,
	const std::vector<int>& lightIndices,
	const std::vector<GpuLightKind>& lightKinds,
	const std::vector<PunctualLightGPU>& punctualLights,
	const std::vector<BilinearPatchData>& bilinearPatches,
	const std::vector<TriangleData>& triangles,
	const std::vector<DiskData>& disks,
	const std::vector<CylinderData>& cylinders,
	const std::vector<GpuLensElement>& lensElements,
	const std::vector<GpuExitPupilBounds>& exitPupilBounds,
	const std::vector<TextureData>& textures,
	const std::vector<unsigned char>& texturePixels,
	const std::vector<CloudMedium<float>>& cloudMediums,
	const std::vector<GpuRgbGridMedium>& rgbGridMediums,
	const std::vector<float>& rgbGridData,
	const std::vector<GpuGridMedium>& gridMediums,
	const std::vector<float>& gridData,
	const std::vector<GpuBssrdfTable>& bssrdfTables,
	const std::vector<float>& bssrdfRhoSamples,
	const std::vector<float>& bssrdfRadiusSamples,
	const std::vector<float>& bssrdfProfile,
	const std::vector<float>& bssrdfProfileCdf,
	const std::vector<GpuMeasuredTable>& measuredTables,
	const std::vector<float>& measuredParamValues,
	const std::vector<float>& measuredData,
	const std::vector<float>& measuredMcdf,
	const std::vector<float>& measuredCcdf,
	const std::vector<float>& skyImagePixels,
	const std::vector<float>& skyMarginalCdf,
	const std::vector<float>& skyMarginalFunc,
	float skyMarginalFuncInt,
	const std::vector<float>& skyConditionalCdf,
	const std::vector<float>& skyConditionalFunc,
	const std::vector<float>& skyConditionalFuncInt,
	int skyWidth, int skyHeight, float skyScale,
	const std::vector<float>& portalRectifiedImage,
	const std::vector<float>& portalDistFunc,
	const std::vector<double>& portalSatSum,
	int portalWidth, int portalHeight, float portalScale,
	float3 portalFrameX, float3 portalFrameY, float3 portalFrameZ,
	float3 portalP0, float3 portalP2
) {
	// Store material data on device
	numMaterials_ = static_cast<unsigned int>(materials.size());
	size_t materialSize = materials.size() * sizeof(MaterialData);

	if (d_materials_) {
		cudaFree(reinterpret_cast<void*>(d_materials_));
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materials_), materialSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_materials_),
		materials.data(),
		materialSize,
		cudaMemcpyHostToDevice
	));

	std::cout << "[OptiX] Uploaded " << materials.size() << " materials to GPU\n";

	// Store texture metadata + shared pixel buffer on device. Both are
	// legitimately empty for most scenes (no textures at all) - guard the
	// malloc/memcpy rather than relying on cudaMalloc(0)'s behavior, same
	// caution already taken for bilinearPatches/triangles below.
	numTextures_ = static_cast<unsigned int>(textures.size());
	size_t textureSize = textures.size() * sizeof(TextureData);

	if (d_textures_) {
		cudaFree(reinterpret_cast<void*>(d_textures_));
		d_textures_ = 0;
	}
	if (!textures.empty()) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_textures_), textureSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_textures_),
			textures.data(),
			textureSize,
			cudaMemcpyHostToDevice
		));
	}

	if (d_texturePixels_) {
		cudaFree(reinterpret_cast<void*>(d_texturePixels_));
		d_texturePixels_ = 0;
	}
	if (!texturePixels.empty()) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texturePixels_), texturePixels.size()));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_texturePixels_),
			texturePixels.data(),
			texturePixels.size(),
			cudaMemcpyHostToDevice
		));
	}

	if (!textures.empty())
		std::cout << "[OptiX] Uploaded " << textures.size() << " textures (" << texturePixels.size() << " pixel bytes) to GPU\n";

	// Store sphere data on device. Like the triangle array below, this holds
	// the scene's own world-space spheres followed by every instance
	// definition's object-space ones, because the device indexes a single flat
	// array whichever GAS a hit came from. sceneSphereCount_ marks the
	// boundary; only the leading world-space run takes part in the scene's own
	// acceleration structure and light list.
	sceneSphereCount_ = static_cast<unsigned int>(spheres.size());
	std::vector<SphereData> allSpheres = spheres;
	allSpheres.insert(allSpheres.end(), instanceSpheres_.begin(), instanceSpheres_.end());
	numSpheres_ = static_cast<unsigned int>(allSpheres.size());
	size_t sphereSize = allSpheres.size() * sizeof(SphereData);

	if (d_spheres_) {
		cudaFree(reinterpret_cast<void*>(d_spheres_));
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_spheres_), sphereSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_spheres_),
		allSpheres.data(),
		sphereSize,
		cudaMemcpyHostToDevice
	));

	std::cout << "[OptiX] Uploaded " << spheres.size() << " spheres to GPU";
	if (!instanceSpheres_.empty())
		std::cout << " (+" << instanceSpheres_.size() << " in instance definitions)";
	std::cout << "\n";

	// Store quad data on device
	numQuads_ = static_cast<unsigned int>(quads.size());
	size_t quadSize = quads.size() * sizeof(QuadData);

	if (d_quads_) {
		cudaFree(reinterpret_cast<void*>(d_quads_));
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_quads_), quadSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_quads_),
		quads.data(),
		quadSize,
		cudaMemcpyHostToDevice
	));

	std::cout << "[OptiX] Uploaded " << quads.size() << " quads to GPU\n";

	// Store bilinear patch data on device
	numBilinearPatches_ = static_cast<unsigned int>(bilinearPatches.size());
	size_t bilinearPatchSize = bilinearPatches.size() * sizeof(BilinearPatchData);

	if (d_bilinearPatches_) {
		cudaFree(reinterpret_cast<void*>(d_bilinearPatches_));
		d_bilinearPatches_ = 0;
	}

	if (numBilinearPatches_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bilinearPatches_), bilinearPatchSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_bilinearPatches_),
			bilinearPatches.data(),
			bilinearPatchSize,
			cudaMemcpyHostToDevice
		));
	}

	std::cout << "[OptiX] Uploaded " << bilinearPatches.size() << " bilinear patches to GPU\n";

	// Store disk data on device
	numDisks_ = static_cast<unsigned int>(disks.size());
	size_t diskSize = disks.size() * sizeof(DiskData);

	if (d_disks_) {
		cudaFree(reinterpret_cast<void*>(d_disks_));
		d_disks_ = 0;
	}

	if (numDisks_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_disks_), diskSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_disks_),
			disks.data(),
			diskSize,
			cudaMemcpyHostToDevice
		));
	}

	std::cout << "[OptiX] Uploaded " << disks.size() << " disks to GPU\n";

	// Store cylinder data on device
	numCylinders_ = static_cast<unsigned int>(cylinders.size());
	size_t cylinderSize = cylinders.size() * sizeof(CylinderData);

	if (d_cylinders_) {
		cudaFree(reinterpret_cast<void*>(d_cylinders_));
		d_cylinders_ = 0;
	}

	if (numCylinders_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cylinders_), cylinderSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_cylinders_),
			cylinders.data(),
			cylinderSize,
			cudaMemcpyHostToDevice
		));
	}

	std::cout << "[OptiX] Uploaded " << cylinders.size() << " cylinders to GPU\n";

	// Store triangle data on device. This single array holds BOTH the
	// scene's own world-space triangles AND every instance definition's
	// object-space triangles (instanceTriangles_, supplied out of band via
	// setInstanceData() - see its comment), appended right after them.
	// Device shading code indexes params.triangles[triBase + primIdx]
	// regardless of which GAS a hit came from, so both regions must live in
	// the one array the device sees. sceneTriangleCount_ records where the
	// instanced region starts; a placement's base offset is then
	// sceneTriangleCount_ + its group's triangleBase (see the IAS/base-table
	// build below).
	sceneTriangleCount_ = static_cast<unsigned int>(triangles.size());
	std::vector<TriangleData> allTriangles = triangles;
	allTriangles.insert(allTriangles.end(), instanceTriangles_.begin(), instanceTriangles_.end());
	numTriangles_ = static_cast<unsigned int>(allTriangles.size());
	size_t triangleSize = allTriangles.size() * sizeof(TriangleData);

	if (d_triangles_) {
		cudaFree(reinterpret_cast<void*>(d_triangles_));
		d_triangles_ = 0;
	}

	if (numTriangles_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles_), triangleSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_triangles_),
			allTriangles.data(),
			triangleSize,
			cudaMemcpyHostToDevice
		));
	}

	std::cout << "[OptiX] Uploaded " << triangles.size() << " triangles to GPU";
	if (!instanceGroups_.empty())
		std::cout << " (+" << instanceTriangles_.size() << " in " << instanceGroups_.size()
			<< " instance definition(s), " << instancePlacements_.size() << " placement(s))";
	std::cout << "\n";

	// Store the RealisticCamera's host-precomputed (focus-adjusted) lens
	// table and exit-pupil bounds table on device. Both come from
	// scene_builder.cpp directly instantiating a host-side
	// RealisticCamera<float> - see optix_types.h's GpuLensElement/
	// GpuExitPupilBounds and render()'s camera-pointer-injection comment.
	numLensElements_ = static_cast<unsigned int>(lensElements.size());
	size_t lensElementSize = lensElements.size() * sizeof(GpuLensElement);

	if (d_lensElements_) {
		cudaFree(reinterpret_cast<void*>(d_lensElements_));
		d_lensElements_ = 0;
	}

	if (numLensElements_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lensElements_), lensElementSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_lensElements_),
			lensElements.data(),
			lensElementSize,
			cudaMemcpyHostToDevice
		));
	}

	numExitPupilBounds_ = static_cast<unsigned int>(exitPupilBounds.size());
	size_t exitPupilBoundsSize = exitPupilBounds.size() * sizeof(GpuExitPupilBounds);

	if (d_exitPupilBounds_) {
		cudaFree(reinterpret_cast<void*>(d_exitPupilBounds_));
		d_exitPupilBounds_ = 0;
	}

	if (numExitPupilBounds_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exitPupilBounds_), exitPupilBoundsSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_exitPupilBounds_),
			exitPupilBounds.data(),
			exitPupilBoundsSize,
			cudaMemcpyHostToDevice
		));
	}

	if (numLensElements_ > 0)
		std::cout << "[OptiX] Uploaded " << lensElements.size() << " lens elements, "
			<< exitPupilBounds.size() << " exit-pupil bounds to GPU\n";

	// Heterogeneous cloud media (MaterialType::CloudMedium) - CloudMedium<float>
	// is uploaded as-is (see optix_types.h's cloud_medium.h include comment),
	// same pattern as the lens table above.
	numCloudMediums_ = static_cast<unsigned int>(cloudMediums.size());
	size_t cloudMediumSize = cloudMediums.size() * sizeof(CloudMedium<float>);

	if (d_cloudMediums_) {
		cudaFree(reinterpret_cast<void*>(d_cloudMediums_));
		d_cloudMediums_ = 0;
	}

	if (numCloudMediums_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cloudMediums_), cloudMediumSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_cloudMediums_),
			cloudMediums.data(),
			cloudMediumSize,
			cudaMemcpyHostToDevice
		));
		std::cout << "[OptiX] Uploaded " << cloudMediums.size() << " cloud media to GPU\n";
	}

	// Heterogeneous RGB grid media (MaterialType::RgbGridMedium) - metadata
	// table plus the shared flat voxel-data buffer it slices into (see
	// GpuRgbGridMedium::dataOffset), same two-array pattern as the
	// RealisticCamera lens table (GpuLensElement metadata isn't itself this
	// two-tier, but CloudMedium above and this both mirror that upload
	// approach: cudaMalloc/cudaMemcpy, freeing any prior allocation first).
	numRgbGridMediums_ = static_cast<unsigned int>(rgbGridMediums.size());
	size_t rgbGridMediumSize = rgbGridMediums.size() * sizeof(GpuRgbGridMedium);

	if (d_rgbGridMediums_) {
		cudaFree(reinterpret_cast<void*>(d_rgbGridMediums_));
		d_rgbGridMediums_ = 0;
	}

	if (numRgbGridMediums_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgbGridMediums_), rgbGridMediumSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_rgbGridMediums_),
			rgbGridMediums.data(),
			rgbGridMediumSize,
			cudaMemcpyHostToDevice
		));
	}

	rgbGridDataCount_ = static_cast<unsigned int>(rgbGridData.size());
	size_t rgbGridDataSize = rgbGridData.size() * sizeof(float);

	if (d_rgbGridData_) {
		cudaFree(reinterpret_cast<void*>(d_rgbGridData_));
		d_rgbGridData_ = 0;
	}

	if (rgbGridDataCount_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgbGridData_), rgbGridDataSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_rgbGridData_),
			rgbGridData.data(),
			rgbGridDataSize,
			cudaMemcpyHostToDevice
		));
		std::cout << "[OptiX] Uploaded " << rgbGridMediums.size() << " RGB grid media ("
			<< rgbGridData.size() << " voxel floats) to GPU\n";
	}

	// Heterogeneous single-channel grid media (MaterialType::GridMedium) -
	// same two-array upload pattern as RGB grid media just above.
	numGridMediums_ = static_cast<unsigned int>(gridMediums.size());
	size_t gridMediumSize = gridMediums.size() * sizeof(GpuGridMedium);

	if (d_gridMediums_) {
		cudaFree(reinterpret_cast<void*>(d_gridMediums_));
		d_gridMediums_ = 0;
	}

	if (numGridMediums_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gridMediums_), gridMediumSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_gridMediums_),
			gridMediums.data(),
			gridMediumSize,
			cudaMemcpyHostToDevice
		));
	}

	gridDataCount_ = static_cast<unsigned int>(gridData.size());
	size_t gridDataSize = gridData.size() * sizeof(float);

	if (d_gridData_) {
		cudaFree(reinterpret_cast<void*>(d_gridData_));
		d_gridData_ = 0;
	}

	if (gridDataCount_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gridData_), gridDataSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_gridData_),
			gridData.data(),
			gridDataSize,
			cudaMemcpyHostToDevice
		));
		std::cout << "[OptiX] Uploaded " << gridMediums.size() << " grid media ("
			<< gridData.size() << " voxel floats) to GPU\n";
	}

	// Tabulated BSSRDF tables (MaterialType::Subsurface, recursive backend
	// only, Phase 1 - see optix_types.h's GpuBssrdfTable comment). Same
	// upload shape as the RGB grid media above: one small metadata array
	// (GpuBssrdfTable) plus four shared flat float buffers it slices into.
	numBssrdfTables_ = static_cast<unsigned int>(bssrdfTables.size());
	size_t bssrdfTableSize = bssrdfTables.size() * sizeof(GpuBssrdfTable);

	if (d_bssrdfTables_) { cudaFree(reinterpret_cast<void*>(d_bssrdfTables_)); d_bssrdfTables_ = 0; }
	if (numBssrdfTables_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bssrdfTables_), bssrdfTableSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_bssrdfTables_),
			bssrdfTables.data(),
			bssrdfTableSize,
			cudaMemcpyHostToDevice
		));
	}

	const auto uploadBssrdfFloats = [&](const std::vector<float>& src, CUdeviceptr& dst) {
		if (dst) { cudaFree(reinterpret_cast<void*>(dst)); dst = 0; }
		if (src.empty()) return;
		const size_t bytes = src.size() * sizeof(float);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), bytes));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src.data(), bytes, cudaMemcpyHostToDevice));
	};
	uploadBssrdfFloats(bssrdfRhoSamples, d_bssrdfRhoSamples_);
	uploadBssrdfFloats(bssrdfRadiusSamples, d_bssrdfRadiusSamples_);
	uploadBssrdfFloats(bssrdfProfile, d_bssrdfProfile_);
	uploadBssrdfFloats(bssrdfProfileCdf, d_bssrdfProfileCdf_);

	if (numBssrdfTables_ > 0)
		std::cout << "[OptiX] Uploaded " << bssrdfTables.size() << " BSSRDF table(s) ("
			<< bssrdfProfile.size() << " profile floats) to GPU\n";

	// Real tabulated measured-BRDF tables (MaterialType::Measured, both GPU
	// backends - see optix_types.h's GpuMeasuredTable comment). Same upload
	// shape as the BSSRDF tables just above: one small metadata array
	// (GpuMeasuredTable, itself 5 GpuPL2DTable sub-tables) plus four shared
	// flat float buffers it slices into.
	numMeasuredTables_ = static_cast<unsigned int>(measuredTables.size());
	size_t measuredTableSize = measuredTables.size() * sizeof(GpuMeasuredTable);

	if (d_measuredTables_) { cudaFree(reinterpret_cast<void*>(d_measuredTables_)); d_measuredTables_ = 0; }
	if (numMeasuredTables_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_measuredTables_), measuredTableSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_measuredTables_),
			measuredTables.data(),
			measuredTableSize,
			cudaMemcpyHostToDevice
		));
	}

	const auto uploadMeasuredFloats = [&](const std::vector<float>& src, CUdeviceptr& dst) {
		if (dst) { cudaFree(reinterpret_cast<void*>(dst)); dst = 0; }
		if (src.empty()) return;
		const size_t bytes = src.size() * sizeof(float);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), bytes));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src.data(), bytes, cudaMemcpyHostToDevice));
	};
	uploadMeasuredFloats(measuredParamValues, d_measuredParamValues_);
	uploadMeasuredFloats(measuredData, d_measuredData_);
	uploadMeasuredFloats(measuredMcdf, d_measuredMcdf_);
	uploadMeasuredFloats(measuredCcdf, d_measuredCcdf_);

	if (numMeasuredTables_ > 0)
		std::cout << "[OptiX] Uploaded " << measuredTables.size() << " measured-BRDF table(s) ("
			<< measuredData.size() << " data floats) to GPU\n";

	// Shared by both the sky distribution and portal-light uploads just
	// below: free any previously-uploaded buffer, then malloc+memcpy the new
	// one (skipped entirely for an empty source, leaving dst null - matches
	// GpuSkyDistribution::height<=0/GpuPortalLight::height<=0's own "absent"
	// convention). Templated (not float-only) so it also serves
	// portalSatSum's double precision (SummedAreaTable's own comment on why
	// that one stays double, not narrowed to float like every other GPU
	// buffer here).
	const auto uploadGpuBuf = [&](const auto& src, CUdeviceptr& dst) {
		using ElemT = typename std::remove_reference<decltype(src)>::type::value_type;
		if (dst) { cudaFree(reinterpret_cast<void*>(dst)); dst = 0; }
		if (src.empty()) return;
		const size_t bytes = src.size() * sizeof(ElemT);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), bytes));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src.data(), bytes, cudaMemcpyHostToDevice));
	};

	// Real importance-sampled HDR sky distribution (LightSource "infinite"
	// with an image - see optix_types.h's GpuSkyDistribution comment). Same
	// upload shape as the BSSRDF/measured-BRDF tables just above, minus the
	// per-table metadata array (a scene has at most one infinite light, so
	// there is nothing to dedup/index - just the flat buffers themselves,
	// referenced later by GpuSkyDistribution's own pointer fields).
	skyWidth_ = skyWidth;
	skyHeight_ = skyHeight;
	skyScale_ = skyScale;
	skyMarginalFuncInt_ = skyMarginalFuncInt;

	uploadGpuBuf(skyImagePixels, d_skyImagePixels_);
	uploadGpuBuf(skyMarginalCdf, d_skyMarginalCdf_);
	uploadGpuBuf(skyMarginalFunc, d_skyMarginalFunc_);
	uploadGpuBuf(skyConditionalCdf, d_skyConditionalCdf_);
	uploadGpuBuf(skyConditionalFunc, d_skyConditionalFunc_);
	uploadGpuBuf(skyConditionalFuncInt, d_skyConditionalFuncInt_);

	if (skyHeight_ > 0)
		std::cout << "[OptiX] Uploaded real HDR sky distribution (" << skyWidth_ << "x" << skyHeight_
			<< ", " << skyImagePixels.size() << " pixel floats) to GPU\n";

	// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's own
	// comment (optix_types.h). Mutually exclusive with the sky distribution
	// just above (matches CPU) - same "flat buffers, no per-table metadata"
	// upload shape.
	portalWidth_ = portalWidth;
	portalHeight_ = portalHeight;
	portalScale_ = portalScale;
	portalFrameX_ = portalFrameX; portalFrameY_ = portalFrameY; portalFrameZ_ = portalFrameZ;
	portalP0_ = portalP0; portalP2_ = portalP2;

	uploadGpuBuf(portalRectifiedImage, d_portalRectifiedImage_);
	uploadGpuBuf(portalDistFunc, d_portalDistFunc_);
	uploadGpuBuf(portalSatSum, d_portalSatSum_);

	if (portalHeight_ > 0)
		std::cout << "[OptiX] Uploaded real portal infinite light (" << portalWidth_ << "x" << portalHeight_
			<< ", " << portalRectifiedImage.size() << " rectified-image floats) to GPU\n";

	// Store light data on device for MIS
	numLights_ = static_cast<unsigned int>(lightIndices.size());

	if (numLights_ > 0) {
		// Upload light indices
		size_t lightIndexSize = lightIndices.size() * sizeof(int);
		if (d_lightIndices_) {
			cudaFree(reinterpret_cast<void*>(d_lightIndices_));
		}
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lightIndices_), lightIndexSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_lightIndices_),
			lightIndices.data(),
			lightIndexSize,
			cudaMemcpyHostToDevice
		));

		// Upload the light kinds verbatim - no repacking, because the host
		// vector's element type IS the type the device reads (GpuLightKind is
		// an int-backed enum class). That identity is the point: this used to
		// pack a std::vector<int> that the device then read through a bool*,
		// advancing one byte per light, so only light 0 ever landed on its own
		// entry. The assert below pins the two together, since the failure is
		// invisible in any scene with fewer than two lights and every built-in
		// scene here has one.
		static_assert(
			sizeof(std::remove_reference_t<decltype(lightKinds)>::value_type) ==
				sizeof(std::remove_pointer_t<decltype(LaunchParams::lightKinds)>),
			"light-kind upload width must match what the device reads; see "
			"GpuLightKind in optix_types.h");

		size_t lightKindSize = lightKinds.size() * sizeof(GpuLightKind);
		if (d_lightKinds_) {
			cudaFree(reinterpret_cast<void*>(d_lightKinds_));
		}
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lightKinds_), lightKindSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_lightKinds_),
			lightKinds.data(),
			lightKindSize,
			cudaMemcpyHostToDevice
		));

		std::cout << "[OptiX] Uploaded " << numLights_ << " light sources for MIS\n";

		// Build power-weighted alias table (pbrt-v4 PowerLightSampler / Vose method)
		// phi_i = area * luminance(emission) * pi  (matches CPU power_light_sampler.h)
		std::vector<float> powers(numLights_);
		// Light BVH (pbrt-v4 §12.6, docs/FEATURE_INVENTORY.md's "no light BVH
		// on GPU" entry - see OptiXRenderer::d_lightBvhNodes_'s own comment)
		// - per-light world-space AABB + dominant emission direction/cone,
		// computed alongside the power estimate above since both need the
		// same per-kind geometry. Flat single-normal shapes (Quad/Triangle/
		// BilinearPatch) get a real, tight direction bound (cosTheta_o=1: the
		// normal never varies across a planar shape); curved shapes (Sphere/
		// Disk/Cylinder) get the conservative omnidirectional default
		// (cosTheta_o=-1, w arbitrary) rather than a real per-point normal
		// cone - CPU's own light_bounds_for() (bvh_light_sampler.h) makes the
		// identical simplification for any shape it doesn't special-case.
		// Only the SPATIAL/power partitioning benefit is lost for these, not
		// correctness - a conservative bound can only under-prune, never
		// wrongly zero out a light that should contribute.
		auto applyAffine = [](const float m[12], float x, float y, float z, float out[3]) {
			out[0] = m[0]*x + m[1]*y + m[2]*z  + m[3];
			out[1] = m[4]*x + m[5]*y + m[6]*z  + m[7];
			out[2] = m[8]*x + m[9]*y + m[10]*z + m[11];
		};
		std::vector<LightBounds> lightBoundsArr(numLights_);
		for (unsigned int i = 0; i < numLights_; ++i) {
			int prim_idx = lightIndices[i];
			float3 emission = make_float3(0.f, 0.f, 0.f);
			float area = 1.0f;
			bool twoSided = false;
			float bMin[3] = {0,0,0}, bMax[3] = {0,0,0};
			float wx = 0.f, wy = 1.f, wz = 0.f, cosThetaO = -1.0f;
			if (lightKinds[i] == GpuLightKind::Sphere) {
				const SphereData& s = spheres[prim_idx];
				const MaterialData& m = materials[s.materialIdx];
				emission = m.emission;
				twoSided = m.twoSided;
				// A ClippedSphere's true emitting area is the visible cap/
				// wedge, not the full sphere s.radius still carries (that
				// field is deliberately the baked full-sphere radius, kept
				// only for the NEE-sampling cone approximation - see
				// SphereData's own comment). Using the full area here would
				// over-select a small clipped cap in this power-weighted
				// alias table far more often than its real contribution
				// warrants - matches pbrt-v4's own Sphere::Area() (phiMax *
				// r * (zMax-zMin), src/shared/shapes.h's SphereShape<T>::
				// area()), reducing to the ordinary 4*pi*r^2 sphere area
				// when phiMax=2*pi and zMax-zMin=2*r (an unclipped sphere).
				area = (s.shapeKind == GpuMediumShapeKind::ClippedSphere)
					? s.phiMax * s.radiusLocal * (s.zMax - s.zMin)
					: 4.0f * 3.14159265f * s.radius * s.radius;  // surface area of sphere
				bMin[0] = s.center.x - s.radius; bMax[0] = s.center.x + s.radius;
				bMin[1] = s.center.y - s.radius; bMax[1] = s.center.y + s.radius;
				bMin[2] = s.center.z - s.radius; bMax[2] = s.center.z + s.radius;
			} else if (lightKinds[i] == GpuLightKind::Triangle) {
				// Indexes `triangles`, not `quads` - and note this is the
				// SCENE's triangle array, which is what lightIndices was built
				// against; instanced triangles are never emissive (flatten()
				// bakes emitters per placement), so no base offset applies.
				const TriangleData& t = triangles[prim_idx];
				const MaterialData& m = materials[t.materialIdx];
				emission = m.emission;
				twoSided = m.twoSided;
				// Half the parallelogram the two edges span.
				const float3 e1 = t.p1 - t.p0;
				const float3 e2 = t.p2 - t.p0;
				float3 n = cross(e1, e2);
				area = 0.5f * length(n);
				if (area > 1e-12f) { n = n / length(n); wx = n.x; wy = n.y; wz = n.z; cosThetaO = 1.0f; }
				bMin[0] = fminf(t.p0.x, fminf(t.p1.x, t.p2.x)); bMax[0] = fmaxf(t.p0.x, fmaxf(t.p1.x, t.p2.x));
				bMin[1] = fminf(t.p0.y, fminf(t.p1.y, t.p2.y)); bMax[1] = fmaxf(t.p0.y, fmaxf(t.p1.y, t.p2.y));
				bMin[2] = fminf(t.p0.z, fminf(t.p1.z, t.p2.z)); bMax[2] = fmaxf(t.p0.z, fmaxf(t.p1.z, t.p2.z));
			} else if (lightKinds[i] == GpuLightKind::BilinearPatch) {
				// Indexes `bilinearPatches`, not `quads` - this branch must
				// stay explicit (not fall into the trailing else below) or
				// prim_idx gets reinterpreted against the wrong array, which
				// for a scene with fewer quads than bilinear patches (e.g.
				// sportscar-area-lights.pbrt has zero quads and 5 bilinear-
				// patch lights) is an out-of-bounds read, not just a wrong
				// number.
				const BilinearPatchData& bp = bilinearPatches[prim_idx];
				const MaterialData& m = materials[bp.materialIdx];
				emission = m.emission;
				twoSided = m.twoSided;
				const float p00[3] = {bp.p00.x, bp.p00.y, bp.p00.z};
				const float p10[3] = {bp.p10.x, bp.p10.y, bp.p10.z};
				const float p01[3] = {bp.p01.x, bp.p01.y, bp.p01.z};
				const float p11[3] = {bp.p11.x, bp.p11.y, bp.p11.z};
				area = blp_area(p00, p10, p01, p11);
				// Not necessarily planar in general (see BilinearPatchData's
				// own comment) - a real per-point normal cone would need
				// integrating over the surface, out of scope here; use the
				// same conservative omnidirectional default as Sphere/Disk/
				// Cylinder rather than a single (possibly wrong for a curved
				// patch) corner normal.
				bMin[0] = fminf(fminf(bp.p00.x,bp.p10.x), fminf(bp.p01.x,bp.p11.x));
				bMax[0] = fmaxf(fmaxf(bp.p00.x,bp.p10.x), fmaxf(bp.p01.x,bp.p11.x));
				bMin[1] = fminf(fminf(bp.p00.y,bp.p10.y), fminf(bp.p01.y,bp.p11.y));
				bMax[1] = fmaxf(fmaxf(bp.p00.y,bp.p10.y), fmaxf(bp.p01.y,bp.p11.y));
				bMin[2] = fminf(fminf(bp.p00.z,bp.p10.z), fminf(bp.p01.z,bp.p11.z));
				bMax[2] = fmaxf(fmaxf(bp.p00.z,bp.p10.z), fmaxf(bp.p01.z,bp.p11.z));
			} else if (lightKinds[i] == GpuLightKind::Disk) {
				// Indexes `disks`, not `quads` - same "must stay explicit"
				// reasoning as the BilinearPatch branch above. dc_area_disk()
				// (optix_disk_cylinder_helpers.h) is tagged __host__
				// __device__ specifically so this plain host C++ file can
				// call it directly instead of hand-copying the formula.
				const DiskData& d = disks[prim_idx];
				const MaterialData& m = materials[d.materialIdx];
				emission = m.emission;
				twoSided = m.twoSided;
				area = dc_area_disk(d);
				// Not baked to world space (see DiskData's own comment) - a
				// conservative object-space square around the outer radius,
				// transformed to world by o2w, at 4 corners of the disk's own
				// z=height plane.
				bMin[0]=bMin[1]=bMin[2]= 1e30f; bMax[0]=bMax[1]=bMax[2]=-1e30f;
				for (int cx = -1; cx <= 1; cx += 2) for (int cy = -1; cy <= 1; cy += 2) {
					float p[3];
					applyAffine(d.o2w, cx*d.radius, cy*d.radius, d.height, p);
					for (int c = 0; c < 3; ++c) { bMin[c] = std::min(bMin[c], p[c]); bMax[c] = std::max(bMax[c], p[c]); }
				}
			} else if (lightKinds[i] == GpuLightKind::Cylinder) {
				const CylinderData& c = cylinders[prim_idx];
				const MaterialData& m = materials[c.materialIdx];
				emission = m.emission;
				twoSided = m.twoSided;
				area = dc_area_cylinder(c);
				// Same "not baked to world space, transform a conservative
				// object-space box by o2w" approach as Disk above - 8 corners
				// of the object-space [-r,r]x[-r,r]x[zMin,zMax] box.
				bMin[0]=bMin[1]=bMin[2]= 1e30f; bMax[0]=bMax[1]=bMax[2]=-1e30f;
				for (int cx = -1; cx <= 1; cx += 2) for (int cy = -1; cy <= 1; cy += 2) {
					float pLo[3], pHi[3];
					applyAffine(c.o2w, cx*c.radius, cy*c.radius, c.zMin, pLo);
					applyAffine(c.o2w, cx*c.radius, cy*c.radius, c.zMax, pHi);
					for (int cc = 0; cc < 3; ++cc) {
						bMin[cc] = std::min(bMin[cc], std::min(pLo[cc], pHi[cc]));
						bMax[cc] = std::max(bMax[cc], std::max(pLo[cc], pHi[cc]));
					}
				}
			} else {
				const QuadData& q = quads[prim_idx];
				const MaterialData& m = materials[q.materialIdx];
				emission = m.emission;
				twoSided = m.twoSided;
				// area = |u x v|
				float3 cr = make_float3(
					q.u.y*q.v.z - q.u.z*q.v.y,
					q.u.z*q.v.x - q.u.x*q.v.z,
					q.u.x*q.v.y - q.u.y*q.v.x);
				area = sqrtf(cr.x*cr.x + cr.y*cr.y + cr.z*cr.z);
				wx = q.normal.x; wy = q.normal.y; wz = q.normal.z; cosThetaO = 1.0f;
				const float3 corners[4] = { q.Q, q.Q+q.u, q.Q+q.v, q.Q+q.u+q.v };
				bMin[0]=bMin[1]=bMin[2]= 1e30f; bMax[0]=bMax[1]=bMax[2]=-1e30f;
				for (const float3& p : corners) {
					bMin[0]=std::min(bMin[0],p.x); bMax[0]=std::max(bMax[0],p.x);
					bMin[1]=std::min(bMin[1],p.y); bMax[1]=std::max(bMax[1],p.y);
					bMin[2]=std::min(bMin[2],p.z); bMax[2]=std::max(bMax[2],p.z);
				}
			}
			float lum = 0.2126f*emission.x + 0.7152f*emission.y + 0.0722f*emission.z;
			powers[i] = area * lum * 3.14159265f;  // phi = area * Le * pi
			if (twoSided) powers[i] *= 2.0f;  // emits from both faces - pbrt-v4 doubles phi to match
			if (powers[i] <= 0.f) powers[i] = 1e-6f;  // geometry-only target
			// cosTheta_e=0 (Lambertian pi/2 cutoff) for every kind - matches
			// CPU's own light_bounds_for() default and every emissive
			// material this loader supports (no non-Lambertian area-light
			// emission profile exists here on either backend).
			lightBoundsArr[i] = LightBounds(bMin[0],bMin[1],bMin[2], bMax[0],bMax[1],bMax[2],
											 wx, wy, wz, powers[i], cosThetaO, 0.0f, twoSided);
		}

		// Vose alias method
		float total = 0.f;
		for (float p : powers) total += p;
		std::vector<GpuAliasEntry> table(numLights_);
		for (unsigned int i = 0; i < numLights_; ++i) {
			table[i].pdf = powers[i] / total;
			table[i].q   = powers[i] / total * float(numLights_);
			table[i].alias = (int)i;
		}
		std::vector<int> small_idx, large_idx;
		for (unsigned int i = 0; i < numLights_; ++i) {
			if (table[i].q < 1.f) small_idx.push_back(i);
			else                  large_idx.push_back(i);
		}
		while (!small_idx.empty() && !large_idx.empty()) {
			int s = small_idx.back(); small_idx.pop_back();
			int l = large_idx.back(); large_idx.pop_back();
			table[s].alias = l;
			table[l].q -= (1.f - table[s].q);
			if (table[l].q < 1.f) small_idx.push_back(l);
			else                  large_idx.push_back(l);
		}
		// Residuals: floating-point rounding may leave items in either list; set q=1 (pbrt-v4 pattern)
		while (!large_idx.empty()) { int l = large_idx.back(); large_idx.pop_back(); table[l].q = 1.f; table[l].alias = l; }
		while (!small_idx.empty()) { int s = small_idx.back(); small_idx.pop_back(); table[s].q = 1.f; table[s].alias = s; }

		size_t aliasTableSize = table.size() * sizeof(GpuAliasEntry);
		if (d_aliasTable_) cudaFree(reinterpret_cast<void*>(d_aliasTable_));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aliasTable_), aliasTableSize));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aliasTable_),
			table.data(), aliasTableSize, cudaMemcpyHostToDevice));
		std::cout << "[OptiX] Uploaded alias table (" << numLights_ << " entries) for power-weighted sampling\n";

		// Light BVH (see OptiXRenderer::d_lightBvhNodes_'s own comment) -
		// built host-side from lightBoundsArr above via the previously-dead-
		// code BVHLightSampler2 (src/shared/bvh_light_sampler2.h), then its
		// flat node/bit-trail arrays are uploaded for gpu_light_bvh_sample()/
		// gpu_light_bvh_pmf() (optix_device_helpers.h) to traverse. A
		// zero-node tree (e.g. every light had phi<=0, which the power loop
		// above already floors away from - so this is only reachable if
		// numLights_ were 0, already excluded by this else-branch) leaves
		// lightBvhNodeCount_ at 0, which every NEE call site already treats
		// as "fall back to the alias table" - no separate empty-tree guard
		// needed here beyond what BVHLightSampler2::Empty() already implies.
		BVHLightSampler2 lightBvh(lightBoundsArr.data(), static_cast<int>(numLights_));
		lightBvhNodeCount_ = lightBvh.NodeCount();
		if (d_lightBvhNodes_) { cudaFree(reinterpret_cast<void*>(d_lightBvhNodes_)); d_lightBvhNodes_ = 0; }
		if (d_lightBvhBitTrail_) { cudaFree(reinterpret_cast<void*>(d_lightBvhBitTrail_)); d_lightBvhBitTrail_ = 0; }
		if (lightBvhNodeCount_ > 0) {
			lightBvhAllBMinX_ = lightBvh.AllBMinX(); lightBvhAllBMinY_ = lightBvh.AllBMinY(); lightBvhAllBMinZ_ = lightBvh.AllBMinZ();
			lightBvhAllBMaxX_ = lightBvh.AllBMaxX(); lightBvhAllBMaxY_ = lightBvh.AllBMaxY(); lightBvhAllBMaxZ_ = lightBvh.AllBMaxZ();

			size_t nodesSize = static_cast<size_t>(lightBvhNodeCount_) * sizeof(LightBVHNode);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lightBvhNodes_), nodesSize));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_lightBvhNodes_),
				lightBvh.Nodes(), nodesSize, cudaMemcpyHostToDevice));

			std::vector<uint32_t> bitTrails = lightBvh.BitTrails(static_cast<int>(numLights_));
			size_t bitTrailSize = bitTrails.size() * sizeof(uint32_t);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lightBvhBitTrail_), bitTrailSize));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_lightBvhBitTrail_),
				bitTrails.data(), bitTrailSize, cudaMemcpyHostToDevice));

			std::cout << "[OptiX] Built light BVH (" << lightBvhNodeCount_ << " nodes) for spatial+power sampling\n";
		}
	} else {
		// No lights in scene
		if (d_lightIndices_) {
			cudaFree(reinterpret_cast<void*>(d_lightIndices_));
			d_lightIndices_ = 0;
		}
		if (d_lightKinds_) {
			cudaFree(reinterpret_cast<void*>(d_lightKinds_));
			d_lightKinds_ = 0;
		}
		if (d_aliasTable_) {
			cudaFree(reinterpret_cast<void*>(d_aliasTable_));
			d_aliasTable_ = 0;
		}
		if (d_lightBvhNodes_) { cudaFree(reinterpret_cast<void*>(d_lightBvhNodes_)); d_lightBvhNodes_ = 0; }
		if (d_lightBvhBitTrail_) { cudaFree(reinterpret_cast<void*>(d_lightBvhBitTrail_)); d_lightBvhBitTrail_ = 0; }
		lightBvhNodeCount_ = 0;
		std::cout << "[OptiX] No emissive lights in scene\n";
	}

	// Store punctual (point/spot/distant) lights on device - separate from
	// the area-light arrays above, evaluated deterministically every hit
	// rather than selected via the alias table (see optix_device_helpers.h
	// eval_punctual_light / add_punctual_lights_lambertian).
	numPunctualLights_ = static_cast<unsigned int>(punctualLights.size());
	if (d_punctualLights_) {
		cudaFree(reinterpret_cast<void*>(d_punctualLights_));
		d_punctualLights_ = 0;
	}
	if (numPunctualLights_ > 0) {
		size_t punctualSize = punctualLights.size() * sizeof(PunctualLightGPU);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_punctualLights_), punctualSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_punctualLights_),
			punctualLights.data(),
			punctualSize,
			cudaMemcpyHostToDevice
		));
		std::cout << "[OptiX] Uploaded " << numPunctualLights_ << " punctual (point/spot/distant) lights\n";
	}

	// Build acceleration structure for custom primitives
	// We'll use AABB (axis-aligned bounding box) custom primitives

	if (spheres.empty() && quads.empty() && bilinearPatches.empty() && triangles.empty() &&
		disks.empty() && cylinders.empty()) {
		std::cerr << "[OptiX] Error: Scene contains no geometry" << std::endl;
		return false;
	}

	// Triangles are excluded from this count - they don't go through the
	// AABB-based custom-primitive path below (see the comment further down).
	size_t totalAabbGeoms = spheres.size() + quads.size() + bilinearPatches.size();
	std::vector<OptixAabb> aabbs;
	aabbs.reserve(totalAabbGeoms);

	// Build AABBs for spheres (and, for shapeKind==Box entries, real boxes -
	// see GpuMediumShapeKind's comment in optix_types.h. A box's own bounds
	// ARE its tight AABB already, unlike a sphere's center+-radius bound.)
	for (const auto& s : spheres) {
		OptixAabb aabb;
		if (s.shapeKind == GpuMediumShapeKind::Box) {
			aabb.minX = s.boxMin.x;
			aabb.minY = s.boxMin.y;
			aabb.minZ = s.boxMin.z;
			aabb.maxX = s.boxMax.x;
			aabb.maxY = s.boxMax.y;
			aabb.maxZ = s.boxMax.z;
		} else if (s.shapeKind == GpuMediumShapeKind::ClippedSphere) {
			// Object-space box [-radiusLocal,radiusLocal] x [-radiusLocal,
			// radiusLocal] x [zMin,zMax] - a conservative bound on the
			// clipped surface (the same one CPU's own
			// affine_transform::transformed_bbox(o2w_, -radius,radius,
			// -radius,radius,z_min,z_max) computes, sphere_clipped_
			// hittable.h) - transformed to world space by the 8 corners,
			// since o2w may rotate/shear/non-uniformly-scale, unlike the
			// plain-sphere case above where center+-radius is exact.
			// Looser than a tight per-clip-region bound would be (no
			// correctness cost - __intersection__sphere's own zMin/zMax/
			// phiMax rejection is what actually enforces the real, tighter
			// clipped shape; a loose AABB only means testing a few more
			// candidate rays than strictly necessary).
			const float r = s.radiusLocal;
			float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
			float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
			for (int c = 0; c < 8; ++c) {
				const float ox = (c & 1) ? r : -r;
				const float oy = (c & 2) ? r : -r;
				const float oz = (c & 4) ? s.zMax : s.zMin;
				const float wx = s.o2w[0]*ox + s.o2w[1]*oy + s.o2w[2]*oz  + s.o2w[3];
				const float wy = s.o2w[4]*ox + s.o2w[5]*oy + s.o2w[6]*oz  + s.o2w[7];
				const float wz = s.o2w[8]*ox + s.o2w[9]*oy + s.o2w[10]*oz + s.o2w[11];
				minX = fminf(minX, wx); maxX = fmaxf(maxX, wx);
				minY = fminf(minY, wy); maxY = fmaxf(maxY, wy);
				minZ = fminf(minZ, wz); maxZ = fmaxf(maxZ, wz);
			}
			aabb.minX = minX; aabb.minY = minY; aabb.minZ = minZ;
			aabb.maxX = maxX; aabb.maxY = maxY; aabb.maxZ = maxZ;
		} else {
			aabb.minX = s.center.x - s.radius;
			aabb.minY = s.center.y - s.radius;
			aabb.minZ = s.center.z - s.radius;
			aabb.maxX = s.center.x + s.radius;
			aabb.maxY = s.center.y + s.radius;
			aabb.maxZ = s.center.z + s.radius;
		}
		aabbs.push_back(aabb);
	}

	// Build AABBs for quads
	for (const auto& q : quads) {
		// Quad corners: Q, Q+u, Q+v, Q+u+v
		float3 corners[4] = {
			q.Q,
			make_float3(q.Q.x + q.u.x, q.Q.y + q.u.y, q.Q.z + q.u.z),
			make_float3(q.Q.x + q.v.x, q.Q.y + q.v.y, q.Q.z + q.v.z),
			make_float3(q.Q.x + q.u.x + q.v.x, q.Q.y + q.u.y + q.v.y, q.Q.z + q.u.z + q.v.z)
		};

		OptixAabb aabb;
		aabb.minX = fminf(fminf(corners[0].x, corners[1].x), fminf(corners[2].x, corners[3].x));
		aabb.minY = fminf(fminf(corners[0].y, corners[1].y), fminf(corners[2].y, corners[3].y));
		aabb.minZ = fminf(fminf(corners[0].z, corners[1].z), fminf(corners[2].z, corners[3].z));
		aabb.maxX = fmaxf(fmaxf(corners[0].x, corners[1].x), fmaxf(corners[2].x, corners[3].x));
		aabb.maxY = fmaxf(fmaxf(corners[0].y, corners[1].y), fmaxf(corners[2].y, corners[3].y));
		aabb.maxZ = fmaxf(fmaxf(corners[0].z, corners[1].z), fmaxf(corners[2].z, corners[3].z));
		aabbs.push_back(aabb);
	}

	// Build AABBs for bilinear patches. The patch's surface (a bilinear/convex
	// combination of the 4 corners for (u,v) in [0,1]^2) always lies within
	// the convex hull of its corners, so a tight min/max of the 4 raw corners
	// is already a valid conservative bound - no epsilon slop needed (matches
	// bilinear_patch_hittable's CPU bounding_box, minus its +-0.01 slop).
	for (const auto& p : bilinearPatches) {
		OptixAabb aabb;
		aabb.minX = fminf(fminf(p.p00.x, p.p10.x), fminf(p.p01.x, p.p11.x));
		aabb.minY = fminf(fminf(p.p00.y, p.p10.y), fminf(p.p01.y, p.p11.y));
		aabb.minZ = fminf(fminf(p.p00.z, p.p10.z), fminf(p.p01.z, p.p11.z));
		aabb.maxX = fmaxf(fmaxf(p.p00.x, p.p10.x), fmaxf(p.p01.x, p.p11.x));
		aabb.maxY = fmaxf(fmaxf(p.p00.y, p.p10.y), fmaxf(p.p01.y, p.p11.y));
		aabb.maxZ = fmaxf(fmaxf(p.p00.z, p.p10.z), fmaxf(p.p01.z, p.p11.z));
		aabbs.push_back(aabb);
	}

	// Triangles are NOT included in this combined AABB buffer - they use
	// OptiX's built-in hardware triangle geometry (OPTIX_BUILD_INPUT_TYPE_
	// TRIANGLES, see the vertex-buffer upload and triBuildInput below),
	// which needs a vertex buffer, not host-computed AABBs (OptiX derives
	// its own bounds internally from the vertex data during the accel
	// build). This matches pbrt-v4's own GPU renderer, which also gives
	// triangles OptiX's native path while keeping quadrics/bilinear-patches
	// as custom AABB primitives like `totalAabbGeoms`'s other three types here.

	// Motion blur: true if any sphere's ray-time-t=1 position differs from
	// its t=0 position (see SphereData::center1's doc comment). Detected
	// here, not passed in as a parameter, so any future scene with moving
	// spheres gets motion support automatically just by setting center1.
	sceneHasMotion_ = false;
	for (const auto& s : spheres) {
		if (s.center1.x != s.center.x || s.center1.y != s.center.y || s.center1.z != s.center.z) {
			sceneHasMotion_ = true;
			break;
		}
	}

	// A second set of AABBs at ray-time t=1, only built when the scene
	// actually has motion. Quad/bilinear-patch/triangle geometry never
	// moves, so their t=1 AABB is identical to their t=0 one - motion keys
	// apply per accel-structure build (shared across every build input in
	// it, since OptiX requires all build inputs in one accelBuild() call to
	// use the same key count), not per build-input, so every build input in
	// a motion-enabled GAS must supply 2 keys even if only spheres move.
	std::vector<OptixAabb> aabbsKey1;
	if (sceneHasMotion_) {
		aabbsKey1.reserve(totalAabbGeoms);
		for (size_t sphIdx = 0; sphIdx < spheres.size(); ++sphIdx) {
			const auto& s = spheres[sphIdx];
			OptixAabb aabb;
			if (s.shapeKind == GpuMediumShapeKind::Box) {
				// No scene combines motion blur with a box medium boundary -
				// same (static) bounds at both motion keys, matching the t=0
				// loop above.
				aabb.minX = s.boxMin.x;
				aabb.minY = s.boxMin.y;
				aabb.minZ = s.boxMin.z;
				aabb.maxX = s.boxMax.x;
				aabb.maxY = s.boxMax.y;
				aabb.maxZ = s.boxMax.z;
			} else if (s.shapeKind == GpuMediumShapeKind::ClippedSphere) {
				// No scene combines motion blur with a clipped sphere either
				// (GpuMediumShapeKind::ClippedSphere's own comment) - static,
				// same bounds at both motion keys, same reasoning as Box
				// above. Unlike Box (which re-derives its trivial bound
				// inline), this reuses the t=0 loop's already-computed 8-
				// corner transform via `aabbs[sphIdx]` (both loops iterate
				// `spheres` in the same order) rather than redoing that
				// transform a second time for a value that can't differ.
				aabb = aabbs[sphIdx];
			} else {
				aabb.minX = s.center1.x - s.radius;
				aabb.minY = s.center1.y - s.radius;
				aabb.minZ = s.center1.z - s.radius;
				aabb.maxX = s.center1.x + s.radius;
				aabb.maxY = s.center1.y + s.radius;
				aabb.maxZ = s.center1.z + s.radius;
			}
			aabbsKey1.push_back(aabb);
		}
		// Static primitives: duplicate the t=0 AABBs already computed above
		// (at the same relative offsets in `aabbs`, right after the spheres).
		aabbsKey1.insert(aabbsKey1.end(), aabbs.begin() + spheres.size(), aabbs.end());
	}

	// Upload AABBs to device
	CUdeviceptr d_aabb;
	size_t aabbSize = aabbs.size() * sizeof(OptixAabb);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb), aabbSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_aabb),
		aabbs.data(),
		aabbSize,
		cudaMemcpyHostToDevice
	));

	// Key-1 (t=1) buffer - only allocated/uploaded when the scene has
	// motion. Left at 0 otherwise; every use of it below is either gated on
	// sceneHasMotion_ or never dereferenced by OptiX when numKeys<2 (see the
	// per-build-input aabbBuffers arrays), so a "0 + offset" placeholder
	// value in the unused case is inert - CUdeviceptr is an integer handle,
	// not a real pointer, so this arithmetic is well-defined either way.
	CUdeviceptr d_aabbKey1 = 0;
	if (sceneHasMotion_) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabbKey1), aabbSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_aabbKey1),
			aabbsKey1.data(),
			aabbSize,
			cudaMemcpyHostToDevice
		));
	}

	// Build input for sphere geometry. OptiX validation rejects a non-null
	// aabbBuffers when numPrimitives==0 ("numPrimitives is zero, but
	// aabbBuffers is non-null") even though it would never be dereferenced -
	// null out the pointer itself for empty types rather than just leaving
	// numPrimitives at 0.
	CUdeviceptr d_sphere_aabb_keys[2] = { d_aabb, d_aabbKey1 };
	std::vector<uint32_t> sphere_flags(spheres.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput sphereBuildInput = {};
	sphereBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	sphereBuildInput.customPrimitiveArray.aabbBuffers = spheres.empty() ? nullptr : d_sphere_aabb_keys;
	sphereBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(spheres.size());
	sphereBuildInput.customPrimitiveArray.flags = sphere_flags.data();
	sphereBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all spheres
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for quad geometry
	CUdeviceptr d_quad_aabb_keys[2] = {
		d_aabb + (spheres.size() * sizeof(OptixAabb)),
		d_aabbKey1 + (spheres.size() * sizeof(OptixAabb))
	};
	std::vector<uint32_t> quad_flags(quads.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput quadBuildInput = {};
	quadBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	quadBuildInput.customPrimitiveArray.aabbBuffers = quads.empty() ? nullptr : d_quad_aabb_keys;
	quadBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(quads.size());
	quadBuildInput.customPrimitiveArray.flags = quad_flags.data();
	quadBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all quads
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for bilinear patch geometry
	CUdeviceptr d_blp_aabb_keys[2] = {
		d_aabb + ((spheres.size() + quads.size()) * sizeof(OptixAabb)),
		d_aabbKey1 + ((spheres.size() + quads.size()) * sizeof(OptixAabb))
	};
	std::vector<uint32_t> blp_flags(bilinearPatches.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput blpBuildInput = {};
	blpBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	blpBuildInput.customPrimitiveArray.aabbBuffers = bilinearPatches.empty() ? nullptr : d_blp_aabb_keys;
	blpBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(bilinearPatches.size());
	blpBuildInput.customPrimitiveArray.flags = blp_flags.data();
	blpBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all bilinear patches
	blpBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	blpBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	blpBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for triangle geometry - OptiX's built-in triangle type,
	// not a custom AABB primitive (see the comment above the AABB loops).
	// Vertex buffer is a flat triangle soup (3 vertices per TriangleData,
	// in order, no index buffer) - primitive i's 3 vertices are exactly
	// triangles[i].p0/p1/p2, so optixGetPrimitiveIndex() in the closest-hit/
	// any-hit programs keeps indexing params.triangles[primIdx] unchanged.
	CUdeviceptr d_triVertices = 0;
	std::vector<float3> triVertsHost;
	if (!triangles.empty()) {
		triVertsHost.reserve(triangles.size() * 3);
		for (const auto& t : triangles) {
			triVertsHost.push_back(t.p0);
			triVertsHost.push_back(t.p1);
			triVertsHost.push_back(t.p2);
		}
		size_t triVertsSize = triVertsHost.size() * sizeof(float3);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triVertices), triVertsSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_triVertices),
			triVertsHost.data(),
			triVertsSize,
			cudaMemcpyHostToDevice
		));
	}
	// Triangles get their own GAS (see the "two GASes + one IAS" comment
	// below) built with motionOptions.numKeys=0 always, since triangles
	// never move - so unlike the AABB types above, no duplicate-for-key-1
	// buffer is needed here, just a single vertex buffer.
	CUdeviceptr d_tri_vertex_keys[1] = { d_triVertices };
	// flags[] for a triangle build input is indexed per SBT record (here:
	// just 1, since numSbtRecords=1), NOT per primitive like the AABB
	// custom-primitive arrays above - a single entry is correct.
	std::vector<uint32_t> tri_flags(1, OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput triBuildInput = {};
	triBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triBuildInput.triangleArray.vertexBuffers = triangles.empty() ? nullptr : d_tri_vertex_keys;
	triBuildInput.triangleArray.numVertices = static_cast<unsigned int>(triangles.size() * 3);
	triBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triBuildInput.triangleArray.indexBuffer = 0;  // implicit: every 3 vertices form one triangle
	triBuildInput.triangleArray.flags = tri_flags.data();
	triBuildInput.triangleArray.numSbtRecords = 1;  // Single hit group for all triangles
	triBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	// Two GASes + one IAS: OptiX forbids mixing OPTIX_BUILD_INPUT_TYPE_
	// TRIANGLES and OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES build inputs in
	// a single GAS (confirmed via NVIDIA's own forums - "you cannot mix
	// different primitive types in a single GAS"), so triangles get their
	// own GAS, spheres/quads/bilinear-patches keep sharing the other one
	// (same conditional-inclusion, same relative [sphere, quad, bilinear
	// patch] order as before), and a top-level IAS with 2 static-identity
	// instances combines them into the single traversable
	// trace_shadow_ray()/the radiance loop both already use. Each
	// instance's sbtOffset shifts that whole child GAS's hit-group records
	// to its own region of the flat SBT array buildSBT() builds below -
	// custom-prim types keep the exact same indices they always had
	// (instance 0's sbtOffset=0), triangles land right after them (instance
	// 1's sbtOffset = however many hit records the custom-prim types
	// occupy) - so buildSBT()'s own record layout doesn't change at all,
	// only how the offset into it is supplied (per-instance now, instead of
	// via build-input position within one shared GAS).
	std::vector<OptixBuildInput> customBuildInputVec;
	if (!spheres.empty()) customBuildInputVec.push_back(sphereBuildInput);
	if (!quads.empty()) customBuildInputVec.push_back(quadBuildInput);
	if (!bilinearPatches.empty()) customBuildInputVec.push_back(blpBuildInput);

	OptixAccelBuildOptions customAccelOptions = {};
	customAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	customAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	// numKeys<2 means "no motion" (OptiX treats 0 and 1 identically) - only
	// scenes with moving spheres (sceneHasMotion_) pay for real motion keys.
	customAccelOptions.motionOptions.numKeys = sceneHasMotion_ ? 2 : 0;
	customAccelOptions.motionOptions.timeBegin = 0.0f;
	customAccelOptions.motionOptions.timeEnd = 1.0f;
	customAccelOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;

	// g_renderer (optix_interface.cpp) is a process-lifetime singleton reused
	// across scene switches, not reconstructed per scene - so gasCustomHandle_/
	// gasTriHandle_ are members that can carry a stale value from whichever
	// PREVIOUS scene actually had that geometry type, if the CURRENT scene
	// doesn't. Reset both unconditionally before the conditional build blocks
	// below: without this, e.g. switching from a triangle-mesh scene to one
	// with none would leave gasTriHandle_ pointing at a GAS whose device
	// memory this same function already frees a few lines down (d_gasTri_),
	// and the IAS build after would wire that dangling handle into a live
	// instance - undefined behavior at trace time, not something that fails
	// loudly here.
	gasCustomHandle_ = 0;
	gasTriHandle_ = 0;
	gasDiskCylinderHandle_ = 0;

	CUdeviceptr d_gasCustomOutput = 0;
	if (!customBuildInputVec.empty()) {
		OptixAccelBufferSizes customBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &customAccelOptions, customBuildInputVec.data(),
			static_cast<unsigned int>(customBuildInputVec.size()), &customBufferSizes
		));

		CUdeviceptr d_customTemp;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_customTemp), customBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasCustomOutput), customBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &customAccelOptions, customBuildInputVec.data(),
			static_cast<unsigned int>(customBuildInputVec.size()),
			d_customTemp, customBufferSizes.tempSizeInBytes,
			d_gasCustomOutput, customBufferSizes.outputSizeInBytes,
			&gasCustomHandle_, nullptr, 0
		));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_customTemp));
	}
	cudaFree(reinterpret_cast<void*>(d_aabb));
	cudaFree(reinterpret_cast<void*>(d_aabbKey1));  // cudaFree(0) is a documented no-op when motion wasn't used

	// Triangle GAS - always static (numKeys=0): triangles never move
	// regardless of sceneHasMotion_ (that flag only tracks moving spheres).
	OptixAccelBuildOptions triAccelOptions = {};
	triAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	triAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	triAccelOptions.motionOptions.numKeys = 0;

	CUdeviceptr d_gasTriOutput = 0;
	if (!triangles.empty()) {
		OptixAccelBufferSizes triBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &triAccelOptions, &triBuildInput, 1, &triBufferSizes
		));

		CUdeviceptr d_triTemp;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triTemp), triBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasTriOutput), triBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &triAccelOptions, &triBuildInput, 1,
			d_triTemp, triBufferSizes.tempSizeInBytes,
			d_gasTriOutput, triBufferSizes.outputSizeInBytes,
			&gasTriHandle_, nullptr, 0
		));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_triTemp));
	}
	cudaFree(reinterpret_cast<void*>(d_triVertices));  // cudaFree(0) is a no-op when there were no triangles

	// Disk/Cylinder: their OWN GAS, deliberately separate from both the
	// shared custom-prim GAS (spheres/quads/bilinear-patches) and the
	// triangle GAS - see gasDiskCylinderHandle_'s own comment (optix_renderer.h)
	// for why. World-space AABBs are computed corner-by-corner from each
	// primitive's own stored o2w transform (same technique as disk_cylinder_
	// hittable.h's CPU-side transformed_bbox() - a naive transform of the
	// object-space box's own min/max would clip the geometry the moment a
	// rotation is involved).
	const auto diskCylinderWorldAabb = [](const float o2w[12],
										   float xlo, float xhi, float ylo, float yhi,
										   float zlo, float zhi) -> OptixAabb {
		OptixAabb box{};
		float lox = 0, loy = 0, loz = 0, hix = 0, hiy = 0, hiz = 0;
		bool first = true;
		for (int corner = 0; corner < 8; ++corner) {
			const float x = (corner & 1) ? xhi : xlo;
			const float y = (corner & 2) ? yhi : ylo;
			const float z = (corner & 4) ? zhi : zlo;
			const float wx = o2w[0] * x + o2w[1] * y + o2w[2]  * z + o2w[3];
			const float wy = o2w[4] * x + o2w[5] * y + o2w[6]  * z + o2w[7];
			const float wz = o2w[8] * x + o2w[9] * y + o2w[10] * z + o2w[11];
			if (first) { lox = hix = wx; loy = hiy = wy; loz = hiz = wz; first = false; continue; }
			lox = fminf(lox, wx); hix = fmaxf(hix, wx);
			loy = fminf(loy, wy); hiy = fmaxf(hiy, wy);
			loz = fminf(loz, wz); hiz = fmaxf(hiz, wz);
		}
		box.minX = lox; box.minY = loy; box.minZ = loz;
		box.maxX = hix; box.maxY = hiy; box.maxZ = hiz;
		return box;
	};

	std::vector<OptixAabb> diskCylinderAabbs;
	diskCylinderAabbs.reserve(disks.size() + cylinders.size());
	for (const auto& d : disks) {
		diskCylinderAabbs.push_back(diskCylinderWorldAabb(
			d.o2w, -d.radius, d.radius, -d.radius, d.radius, d.height, d.height));
	}
	for (const auto& c : cylinders) {
		diskCylinderAabbs.push_back(diskCylinderWorldAabb(
			c.o2w, -c.radius, c.radius, -c.radius, c.radius, c.zMin, c.zMax));
	}

	CUdeviceptr d_diskCylinderAabb = 0;
	CUdeviceptr d_gasDiskCylinderOutput = 0;
	if (!diskCylinderAabbs.empty()) {
		size_t dcAabbSize = diskCylinderAabbs.size() * sizeof(OptixAabb);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_diskCylinderAabb), dcAabbSize));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_diskCylinderAabb), diskCylinderAabbs.data(),
			dcAabbSize, cudaMemcpyHostToDevice));

		CUdeviceptr d_disk_aabb_keys[1] = { d_diskCylinderAabb };
		std::vector<uint32_t> diskFlags(disks.size(), OPTIX_GEOMETRY_FLAG_NONE);
		OptixBuildInput diskBuildInput = {};
		diskBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		diskBuildInput.customPrimitiveArray.aabbBuffers = disks.empty() ? nullptr : d_disk_aabb_keys;
		diskBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(disks.size());
		diskBuildInput.customPrimitiveArray.flags = diskFlags.data();
		diskBuildInput.customPrimitiveArray.numSbtRecords = 1;
		diskBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		diskBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
		diskBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

		CUdeviceptr d_cylinder_aabb_keys[1] = { d_diskCylinderAabb + (disks.size() * sizeof(OptixAabb)) };
		std::vector<uint32_t> cylinderFlags(cylinders.size(), OPTIX_GEOMETRY_FLAG_NONE);
		OptixBuildInput cylinderBuildInput = {};
		cylinderBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		cylinderBuildInput.customPrimitiveArray.aabbBuffers = cylinders.empty() ? nullptr : d_cylinder_aabb_keys;
		cylinderBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(cylinders.size());
		cylinderBuildInput.customPrimitiveArray.flags = cylinderFlags.data();
		cylinderBuildInput.customPrimitiveArray.numSbtRecords = 1;
		cylinderBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		cylinderBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
		cylinderBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

		// No gaps for an absent type, matching the shared custom-prim GAS's
		// own rule (see numCustomPrimSbtRecords' comment further down).
		std::vector<OptixBuildInput> diskCylinderBuildInputVec;
		if (!disks.empty()) diskCylinderBuildInputVec.push_back(diskBuildInput);
		if (!cylinders.empty()) diskCylinderBuildInputVec.push_back(cylinderBuildInput);

		// triAccelOptions, not customAccelOptions: neither shape supports
		// motion blur, matching every other static (numKeys=0) GAS here.
		OptixAccelBufferSizes dcBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &triAccelOptions, diskCylinderBuildInputVec.data(),
			static_cast<unsigned int>(diskCylinderBuildInputVec.size()), &dcBufferSizes));

		CUdeviceptr d_dcTemp;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dcTemp), dcBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasDiskCylinderOutput), dcBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &triAccelOptions, diskCylinderBuildInputVec.data(),
			static_cast<unsigned int>(diskCylinderBuildInputVec.size()),
			d_dcTemp, dcBufferSizes.tempSizeInBytes,
			d_gasDiskCylinderOutput, dcBufferSizes.outputSizeInBytes,
			&gasDiskCylinderHandle_, nullptr, 0
		));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_dcTemp));
	}
	cudaFree(reinterpret_cast<void*>(d_diskCylinderAabb));  // cudaFree(0) is a no-op when there were none

	// ---- object instancing: one GAS per instance DEFINITION -----------------
	// Built once here regardless of how many times it's placed; each IAS
	// instance added below just points at the same GAS with its own
	// transform - that sharing is the entire point of instancing over
	// baking. g_renderer (optix_interface.cpp) is a process-lifetime
	// singleton, so these are member vectors that must be rebuilt (and their
	// old device memory freed) every buildScene() call, same reasoning as
	// gasCustomHandle_/gasTriHandle_ above.
	for (CUdeviceptr p : d_gasGroupTri_) if (p) cudaFree(reinterpret_cast<void*>(p));
	for (CUdeviceptr p : d_gasGroupSphere_) if (p) cudaFree(reinterpret_cast<void*>(p));
	d_gasGroupTri_.assign(instanceGroups_.size(), 0);
	gasGroupTriHandles_.assign(instanceGroups_.size(), 0);
	d_gasGroupSphere_.assign(instanceGroups_.size(), 0);
	gasGroupSphereHandles_.assign(instanceGroups_.size(), 0);

	for (std::size_t g = 0; g < instanceGroups_.size(); ++g) {
		const SceneData::InstanceGroupGPU& grp = instanceGroups_[g];
		if (grp.triangleCount <= 0) continue;

		std::vector<float3> groupVertsHost;
		groupVertsHost.reserve(static_cast<std::size_t>(grp.triangleCount) * 3);
		for (int i = 0; i < grp.triangleCount; ++i) {
			const TriangleData& t = instanceTriangles_[static_cast<std::size_t>(grp.triangleBase + i)];
			groupVertsHost.push_back(t.p0);
			groupVertsHost.push_back(t.p1);
			groupVertsHost.push_back(t.p2);
		}
		CUdeviceptr d_groupVerts = 0;
		size_t groupVertsSize = groupVertsHost.size() * sizeof(float3);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_groupVerts), groupVertsSize));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_groupVerts), groupVertsHost.data(),
			groupVertsSize, cudaMemcpyHostToDevice));

		CUdeviceptr d_groupVertKeys[1] = { d_groupVerts };
		std::vector<uint32_t> groupFlags(1, OPTIX_GEOMETRY_FLAG_NONE);
		OptixBuildInput groupBuildInput = {};
		groupBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		groupBuildInput.triangleArray.vertexBuffers = d_groupVertKeys;
		groupBuildInput.triangleArray.numVertices = static_cast<unsigned int>(groupVertsHost.size());
		groupBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		groupBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
		groupBuildInput.triangleArray.indexBuffer = 0;
		groupBuildInput.triangleArray.flags = groupFlags.data();
		groupBuildInput.triangleArray.numSbtRecords = 1;
		groupBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
		groupBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
		groupBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

		OptixAccelBufferSizes groupBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &triAccelOptions, &groupBuildInput, 1, &groupBufferSizes));

		CUdeviceptr d_groupTemp = 0, d_groupOutput = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_groupTemp), groupBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_groupOutput), groupBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &triAccelOptions, &groupBuildInput, 1,
			d_groupTemp, groupBufferSizes.tempSizeInBytes,
			d_groupOutput, groupBufferSizes.outputSizeInBytes,
			&gasGroupTriHandles_[g], nullptr, 0));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_groupTemp));
		cudaFree(reinterpret_cast<void*>(d_groupVerts));

		d_gasGroupTri_[g] = d_groupOutput;
	}

	// Spheres in a definition need a SECOND GAS of their own: they are custom
	// AABB primitives, which OptiX refuses to put in the same structure as the
	// native triangles above. Their AABBs are computed in the definition's
	// object space and stay there - the placement's transform is applied by
	// traversal, which is what lets a non-uniform scale produce an ellipsoid
	// instead of a resized sphere.
	for (std::size_t g = 0; g < instanceGroups_.size(); ++g) {
		const SceneData::InstanceGroupGPU& grp = instanceGroups_[g];
		if (grp.sphereCount <= 0) continue;

		std::vector<OptixAabb> groupAabbs;
		groupAabbs.reserve(static_cast<std::size_t>(grp.sphereCount));
		for (int i = 0; i < grp.sphereCount; ++i) {
			const SphereData& s = instanceSpheres_[static_cast<std::size_t>(grp.sphereBase + i)];
			OptixAabb aabb;
			aabb.minX = s.center.x - s.radius;
			aabb.minY = s.center.y - s.radius;
			aabb.minZ = s.center.z - s.radius;
			aabb.maxX = s.center.x + s.radius;
			aabb.maxY = s.center.y + s.radius;
			aabb.maxZ = s.center.z + s.radius;
			groupAabbs.push_back(aabb);
		}

		CUdeviceptr d_groupAabb = 0;
		size_t groupAabbSize = groupAabbs.size() * sizeof(OptixAabb);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_groupAabb), groupAabbSize));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_groupAabb), groupAabbs.data(),
			groupAabbSize, cudaMemcpyHostToDevice));

		CUdeviceptr d_groupAabbKeys[1] = { d_groupAabb };
		std::vector<uint32_t> groupSphereFlags(groupAabbs.size(), OPTIX_GEOMETRY_FLAG_NONE);
		OptixBuildInput groupSphereInput = {};
		groupSphereInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		groupSphereInput.customPrimitiveArray.aabbBuffers = d_groupAabbKeys;
		groupSphereInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(groupAabbs.size());
		groupSphereInput.customPrimitiveArray.flags = groupSphereFlags.data();
		groupSphereInput.customPrimitiveArray.numSbtRecords = 1;
		groupSphereInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		groupSphereInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
		groupSphereInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

		// triAccelOptions rather than customAccelOptions: that one carries the
		// scene's motion keys, and an instance definition never moves (a pbrt
		// ObjectInstance has one static transform). numKeys=0 is what this
		// wants regardless of sceneHasMotion_.
		OptixAccelBufferSizes groupBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &triAccelOptions, &groupSphereInput, 1, &groupBufferSizes));

		CUdeviceptr d_groupTemp = 0, d_groupOutput = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_groupTemp), groupBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_groupOutput), groupBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &triAccelOptions, &groupSphereInput, 1,
			d_groupTemp, groupBufferSizes.tempSizeInBytes,
			d_groupOutput, groupBufferSizes.outputSizeInBytes,
			&gasGroupSphereHandles_[g], nullptr, 0));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_groupTemp));
		cudaFree(reinterpret_cast<void*>(d_groupAabb));

		d_gasGroupSphere_[g] = d_groupOutput;
	}

	if (d_gasCustom_) cudaFree(reinterpret_cast<void*>(d_gasCustom_));
	d_gasCustom_ = d_gasCustomOutput;
	if (d_gasTri_) cudaFree(reinterpret_cast<void*>(d_gasTri_));
	d_gasTri_ = d_gasTriOutput;
	if (d_gasDiskCylinder_) cudaFree(reinterpret_cast<void*>(d_gasDiskCylinder_));
	d_gasDiskCylinder_ = d_gasDiskCylinderOutput;

	// Top-level IAS: one static-identity instance per non-empty child GAS,
	// then one further instance per instanced-geometry placement.
	//
	// SBT LAYOUT AND WHY INSTANCED GEOMETRY GETS ITS OWN RECORDS
	// A child GAS's hit record is found at
	//     instance.sbtOffset + build_input_index * RAY_TYPE_COUNT + ray_type,
	// and the scene's custom-primitive region is packed with no gaps for
	// absent types, so a build input's index there depends on which OTHER
	// types the scene happens to contain. An instance GAS has exactly one
	// build input (index 0), so pointing it into that packed region would
	// only work by coincidence - and for spheres, which sort FIRST among the
	// custom types, it would go wrong the moment a scene has instanced
	// spheres but no world-space ones. So buildSBT() appends a dedicated
	// record pair per instanced geometry type after everything else, and
	// these offsets address those. See buildSBT()'s own parameters.
	//
	// RAY_TYPE_COUNT (not a literal 2) - each present geometry type now
	// contributes a TRIPLE of records (radiance/shadow/probe - see
	// optix_types.h's RAY_TYPE_PROBE comment), not a pair, since the probe
	// ray type was added for MaterialType::Subsurface's GPU probe walk.
	const bool haveInstancedTriangles = !instanceTriangles_.empty();
	const bool haveInstancedSpheres = !instanceSpheres_.empty();
	const int numCustomPrimSbtRecords = RAY_TYPE_COUNT * (int)(!spheres.empty() + !quads.empty() + !bilinearPatches.empty());
	const int sceneTriSbtOffset = numCustomPrimSbtRecords;
	const int instTriSbtOffset = numCustomPrimSbtRecords + (triangles.empty() ? 0 : RAY_TYPE_COUNT);
	const int instSphereSbtOffset = instTriSbtOffset + (haveInstancedTriangles ? RAY_TYPE_COUNT : 0);
	// Disk/Cylinder's own dedicated GAS gets its own instance, appended after
	// every other region (including instanced geometry) - see
	// gasDiskCylinderHandle_'s own comment (optix_renderer.h) for why this
	// can never shift any other type's offset regardless of whether the
	// scene has any disks/cylinders at all. Disk comes first (build_input
	// index 0), cylinder second, with no gap when one of the two is absent -
	// same "packed, no gaps for absent types" rule as every other region.
	const int diskCylinderSbtOffset =
		instSphereSbtOffset + (haveInstancedSpheres ? RAY_TYPE_COUNT : 0);

	static const float kIdentity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
	std::vector<OptixInstance> instances;
	// instancePrimBase, built in lockstep with `instances` so that entry i
	// describes instance id i - the id is assigned as the push position
	// rather than a fixed number, because how many entries exist before a
	// placement now depends on which child GASes the scene has. -1 means
	// "already world space, base 0" (see LaunchParams::instancePrimBase).
	std::vector<int> baseTable;
	const auto addInstance = [&](OptixTraversableHandle handle, const float* xform,
								 int sbtOffset, int primBase) {
		OptixInstance inst{};
		memcpy(inst.transform, xform, sizeof(inst.transform));
		inst.instanceId = static_cast<unsigned int>(instances.size());
		inst.sbtOffset = static_cast<unsigned int>(sbtOffset);
		inst.visibilityMask = 255;
		inst.flags = OPTIX_INSTANCE_FLAG_NONE;
		inst.traversableHandle = handle;
		instances.push_back(inst);
		baseTable.push_back(primBase);
	};

	if (gasCustomHandle_) addInstance(gasCustomHandle_, kIdentity, 0, -1);
	if (gasTriHandle_) addInstance(gasTriHandle_, kIdentity, sceneTriSbtOffset, -1);
	if (gasDiskCylinderHandle_) addInstance(gasDiskCylinderHandle_, kIdentity, diskCylinderSbtOffset, -1);

	// One instance per PLACEMENT, not per group: the same group GAS is
	// referenced by every placement of it, each with its own transform, and
	// that sharing is the entire point of instancing over baking. A group
	// holding both triangles and spheres contributes TWO entries per
	// placement, since its geometry had to be split across two GASes.
	for (const SceneData::InstancePlacementGPU& placement : instancePlacements_) {
		if (placement.group < 0 ||
			static_cast<std::size_t>(placement.group) >= instanceGroups_.size()) continue;
		const std::size_t g = static_cast<std::size_t>(placement.group);
		const SceneData::InstanceGroupGPU& grp = instanceGroups_[g];

		if (gasGroupTriHandles_[g]) {
			addInstance(gasGroupTriHandles_[g], placement.transform, instTriSbtOffset,
						static_cast<int>(sceneTriangleCount_) + grp.triangleBase);
		}
		if (gasGroupSphereHandles_[g]) {
			addInstance(gasGroupSphereHandles_[g], placement.transform, instSphereSbtOffset,
						static_cast<int>(sceneSphereCount_) + grp.sphereBase);
		}
	}

	CUdeviceptr d_instances;
	size_t instancesSize = instances.size() * sizeof(OptixInstance);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instancesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), instances.data(), instancesSize, cudaMemcpyHostToDevice));

	OptixBuildInput iasBuildInput = {};
	iasBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	iasBuildInput.instanceArray.instances = d_instances;
	iasBuildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

	OptixAccelBuildOptions iasAccelOptions = {};
	iasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	iasAccelOptions.motionOptions.numKeys = 0;  // Instance transforms are static; motion lives inside the custom-prim GAS

	OptixAccelBufferSizes iasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		context_, &iasAccelOptions, &iasBuildInput, 1, &iasBufferSizes
	));

	CUdeviceptr d_iasTemp;
	CUdeviceptr d_iasOutput;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iasTemp), iasBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iasOutput), iasBufferSizes.outputSizeInBytes));

	OPTIX_CHECK(optixAccelBuild(
		context_, stream_, &iasAccelOptions, &iasBuildInput, 1,
		d_iasTemp, iasBufferSizes.tempSizeInBytes,
		d_iasOutput, iasBufferSizes.outputSizeInBytes,
		&gasHandle_, nullptr, 0
	));
	CUDA_CHECK(cudaStreamSynchronize(stream_));
	cudaFree(reinterpret_cast<void*>(d_iasTemp));
	cudaFree(reinterpret_cast<void*>(d_instances));

	if (d_gas_) cudaFree(reinterpret_cast<void*>(d_gas_));
	d_gas_ = d_iasOutput;

	// Upload the base table assembled alongside the instance array above.
	// Rebuilt every call for the same reason as the GASes; left null (no
	// upload at all) for scenes with no placements, so params.instancePrimBase
	// stays nullptr and every hit program takes its cheaper non-instanced path
	// without even a table read.
	if (d_instanceBase_) { cudaFree(reinterpret_cast<void*>(d_instanceBase_)); d_instanceBase_ = 0; }
	if (!instancePlacements_.empty() && !baseTable.empty()) {
		size_t baseTableSize = baseTable.size() * sizeof(int);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instanceBase_), baseTableSize));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instanceBase_), baseTable.data(),
			baseTableSize, cudaMemcpyHostToDevice));
	}

	std::cout << "[OptiX] Built acceleration structure: "
		<< spheres.size() << " spheres, "
		<< quads.size() << " quads, "
		<< bilinearPatches.size() << " bilinear patches, "
		<< disks.size() << " disks, "
		<< cylinders.size() << " cylinders, "
		<< triangles.size() << " triangles\n";

	// Build Shader Binding Table (SBT). The scene-only geometry decides the
	// packed record region; instanced geometry gets appended pairs of its own,
	// which the instTri/instSphere offsets computed above address; disks/
	// cylinders get their own trailing region the same way (diskCylinderSbtOffset).
	if (!buildSBT(spheres, quads, bilinearPatches, triangles,
				  haveInstancedTriangles, haveInstancedSpheres, disks, cylinders)) {
		std::cerr << "Failed to build SBT\n";
		return false;
	}

	return true;
}

bool OptiXRenderer::buildSBT(
	const std::vector<SphereData>& spheres,
	const std::vector<QuadData>& quads,
	const std::vector<BilinearPatchData>& bilinearPatches,
	const std::vector<TriangleData>& triangles,
	bool haveInstancedTriangles,
	bool haveInstancedSpheres,
	const std::vector<DiskData>& disks,
	const std::vector<CylinderData>& cylinders
) {
	// Raygen record
	RaygenRecord raygenRecord;
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &raygenRecord));
	raygenRecord.data = 0;  // No data

	if (d_raygenRecord_) cudaFree(reinterpret_cast<void*>(d_raygenRecord_));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord_), sizeof(RaygenRecord)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_raygenRecord_),
		&raygenRecord,
		sizeof(RaygenRecord),
		cudaMemcpyHostToDevice
	));

	// Miss records (radiance + shadow + probe) - indexed by missSBTIndex at
	// each optixTrace() call site, which passes RAY_TYPE_RADIANCE/SHADOW/
	// PROBE directly (0/1/2), so this array's order must match that enum
	// exactly (optix_types.h).
	std::vector<MissRecord> missRecords(RAY_TYPE_COUNT);

	// Radiance miss
	OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &missRecords[RAY_TYPE_RADIANCE]));
	missRecords[RAY_TYPE_RADIANCE].data = 0;

	// Shadow miss
	OPTIX_CHECK(optixSbtRecordPackHeader(shadowMissPG_, &missRecords[RAY_TYPE_SHADOW]));
	missRecords[RAY_TYPE_SHADOW].data = 0;

	// Probe miss (no-op - see optix_probe_hit.h's __miss__probe())
	OPTIX_CHECK(optixSbtRecordPackHeader(probeMissPG_, &missRecords[RAY_TYPE_PROBE]));
	missRecords[RAY_TYPE_PROBE].data = 0;

	if (d_missRecord_) cudaFree(reinterpret_cast<void*>(d_missRecord_));
	size_t missRecordSize = missRecords.size() * sizeof(MissRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord_), missRecordSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_missRecord_),
		missRecords.data(),
		missRecordSize,
		cudaMemcpyHostToDevice
	));

	// Hit group records - radiance + shadow + probe for each geometry type
	// PRESENT in the scene. OptiX SBT layout: index = (build_input_index *
	// RAY_TYPE_COUNT) + ray_type_index, where build_input_index is the
	// geometry type's POSITION among the non-empty build inputs in
	// buildScene() (empty types are omitted from that array entirely - see
	// its comment) - so this must emit exactly one (radiance, shadow,
	// probe) record TRIPLE per present type, in the same relative [sphere,
	// quad, bilinear patch, triangle] order and ray-type order (matching
	// RAY_TYPE_RADIANCE/SHADOW/PROBE = 0/1/2), with no gaps for absent
	// types.
	std::vector<HitGroupRecord> hitGroupRecords;
	if (!spheres.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!quads.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupQuadPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupQuadPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupQuadPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!bilinearPatches.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupBilinearPatchPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupBilinearPatchPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupBilinearPatchPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!triangles.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}

	// Instanced geometry's own records, appended AFTER the packed scene region
	// above so adding them cannot shift any existing build input's index. The
	// programs are the same ones the scene's records already name - these
	// records are duplicates whose only job is to give a per-definition GAS a
	// sbtOffset it can reach without depending on which geometry types the
	// scene itself happens to have. buildScene() computes the matching offsets
	// from the same two flags; the order here (triangles, then spheres) is
	// what those computations assume.
	if (haveInstancedTriangles) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (haveInstancedSpheres) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}

	// Disk/Cylinder's own trailing region (see gasDiskCylinderHandle_'s and
	// diskCylinderSbtOffset's own comments in buildScene()) - appended last,
	// same "packed, no gaps for absent types" rule as everything above, in
	// the same [disk, cylinder] order their shared GAS's build inputs use.
	if (!disks.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupDiskPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupDiskPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupDiskPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!cylinders.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupCylinderPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupCylinderPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(probeHitgroupCylinderPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}

	if (d_hitgroupRecords_) cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));
	size_t hitRecordSize = hitGroupRecords.size() * sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecords_), hitRecordSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_hitgroupRecords_),
		hitGroupRecords.data(),
		hitRecordSize,
		cudaMemcpyHostToDevice
	));

	numHitRecords_ = hitGroupRecords.size();

	// Configure SBT
	sbt_.raygenRecord = d_raygenRecord_;
	sbt_.missRecordBase = d_missRecord_;
	sbt_.missRecordStrideInBytes = sizeof(MissRecord);
	sbt_.missRecordCount = static_cast<unsigned int>(missRecords.size());  // radiance + shadow + probe
	sbt_.hitgroupRecordBase = d_hitgroupRecords_;
	sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
	sbt_.hitgroupRecordCount = static_cast<unsigned int>(hitGroupRecords.size());

	std::cout << "[OptiX] Built SBT: " << missRecords.size() << " miss records (radiance + shadow + probe), "
		<< hitGroupRecords.size() << " hit records (one radiance+shadow+probe triple per present geometry type"
		<< (disks.empty() && cylinders.empty() ? "" : ", including disk/cylinder's own trailing region") << ")\n";
	return true;
}

