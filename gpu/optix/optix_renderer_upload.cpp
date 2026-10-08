// optix_renderer_upload.cpp -- OptiXRenderer: uploading scene data to the device (materials, textures, lights, media, tables) and the probe grid.
// A pure split of optix_renderer_scene.cpp; nothing here changed.

#include "optix_renderer_scene_helpers.h"

// ============================================================================
// buildProbeGrid -- world-space irradiance probe cache (Live Preview only,
// gpu/optix/probe_grid_types.h and this project's own plan). See
// optix_renderer.h's own buildProbeGrid()/probeGridMeta_ comments.
// ============================================================================
void OptiXRenderer::buildProbeGrid(
	const std::vector<SphereData>& spheres, const std::vector<QuadData>& quads,
	const std::vector<BilinearPatchData>& bilinearPatches, const std::vector<TriangleData>& triangles,
	const std::vector<DiskData>& disks, const std::vector<CylinderData>& cylinders)
{
	float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
	float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
	auto fold = [&](float x, float y, float z) {
		minX = fminf(minX, x); minY = fminf(minY, y); minZ = fminf(minZ, z);
		maxX = fmaxf(maxX, x); maxY = fmaxf(maxY, y); maxZ = fmaxf(maxZ, z);
	};
	for (const auto& s : spheres) {
		// Box-shape medium boundaries leave center/radius unused/zero (see
		// SphereData's own comment, optix_types.h) - fold their real boxMin/
		// boxMax instead, same distinction the GAS-build AABB loop further
		// down this file already makes. ClippedSphere entries DO populate a
		// real, conservative full-sphere center/radius (optix_types.h's own
		// comment), so they fall through to the plain sphere fold below
		// unlike Box.
		if (s.shapeKind == GpuMediumShapeKind::Box) {
			fold(s.boxMin.x, s.boxMin.y, s.boxMin.z);
			fold(s.boxMax.x, s.boxMax.y, s.boxMax.z);
			continue;
		}
		fold(s.center.x - s.radius, s.center.y - s.radius, s.center.z - s.radius);
		fold(s.center.x + s.radius, s.center.y + s.radius, s.center.z + s.radius);
	}
	for (const auto& q : quads) {
		fold(q.Q.x, q.Q.y, q.Q.z);
		fold(q.Q.x + q.u.x, q.Q.y + q.u.y, q.Q.z + q.u.z);
		fold(q.Q.x + q.v.x, q.Q.y + q.v.y, q.Q.z + q.v.z);
		fold(q.Q.x + q.u.x + q.v.x, q.Q.y + q.u.y + q.v.y, q.Q.z + q.u.z + q.v.z);
	}
	for (const auto& p : bilinearPatches) {
		fold(p.p00.x, p.p00.y, p.p00.z); fold(p.p01.x, p.p01.y, p.p01.z);
		fold(p.p10.x, p.p10.y, p.p10.z); fold(p.p11.x, p.p11.y, p.p11.z);
	}
	for (const auto& t : triangles) {
		fold(t.p0.x, t.p0.y, t.p0.z); fold(t.p1.x, t.p1.y, t.p1.z); fold(t.p2.x, t.p2.y, t.p2.z);
	}
	// Disks/cylinders: exact corner-transformed world AABB, same helper the
	// GAS-build AABB loop further down this file uses for these two shapes
	// (wf_disk_cylinder_world_aabb, this file's own file-scope helper) -
	// tight even for a disk/cylinder rotated far from axis-aligned, unlike a
	// radius-only isotropic margin.
	for (const auto& d : disks) {
		const OptixAabb box = wf_disk_cylinder_world_aabb(d.o2w, -d.radius, d.radius, -d.radius, d.radius, d.height, d.height);
		fold(box.minX, box.minY, box.minZ);
		fold(box.maxX, box.maxY, box.maxZ);
	}
	for (const auto& c : cylinders) {
		const OptixAabb box = wf_disk_cylinder_world_aabb(c.o2w, -c.radius, c.radius, -c.radius, c.radius, c.zMin, c.zMax);
		fold(box.minX, box.minY, box.minZ);
		fold(box.maxX, box.maxY, box.maxZ);
	}

	if (d_probeGrid_) { cudaFree(reinterpret_cast<void*>(d_probeGrid_)); d_probeGrid_ = 0; }
	// Path guiding's own per-probe histogram array - freed here in lockstep
	// with d_probeGrid_ (see its own comment, optix_renderer.h) since both
	// are sized probeGridMeta_.totalProbes and rebuilt together below.
	if (d_guidingHistograms_) { cudaFree(reinterpret_cast<void*>(d_guidingHistograms_)); d_guidingHistograms_ = 0; }
	probeGridMeta_ = GpuProbeGridMeta{};

	if (minX > maxX) {
		// No geometry contributed a bound - shouldn't happen (buildScene()
		// already requires non-empty geometry before this point), but
		// degrade to "no probe grid" rather than build one from an inverted
		// box - wf_finish_material_scatter's own totalProbes<=0 gate already
		// treats this as a safe no-op everywhere else.
		std::cout << "[OptiX] Probe cache: no scene bounds, skipping probe grid\n";
		return;
	}

	const float dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
	const float diag = std::sqrt(dx * dx + dy * dy + dz * dz);
	// spacing = sceneDiagonal/20, clamped - see this project's own plan.
	float spacing = diag / 20.0f;
	if (!(spacing > 0.0f) || std::isnan(spacing)) spacing = 1.0f;
	spacing = std::max(spacing, 0.05f);

	const int kMaxProbesPerAxis = 32;  // hard cap: 32^3 = 32768 probes max
	const int dimX = std::min(kMaxProbesPerAxis, std::max(1, static_cast<int>(std::ceil(dx / spacing)) + 1));
	const int dimY = std::min(kMaxProbesPerAxis, std::max(1, static_cast<int>(std::ceil(dy / spacing)) + 1));
	const int dimZ = std::min(kMaxProbesPerAxis, std::max(1, static_cast<int>(std::ceil(dz / spacing)) + 1));

	probeGridMeta_.gridMin = make_float3(minX, minY, minZ);
	probeGridMeta_.cellSize = make_float3(spacing, spacing, spacing);
	probeGridMeta_.dims = make_int3(dimX, dimY, dimZ);
	probeGridMeta_.totalProbes = dimX * dimY * dimZ;

	// Zeroed on upload - GpuProbe's own default member initializers already
	// give every field its correct "never updated" state (numRaysEverTraced
	// ==0), so a plain value-initialized vector is a fully valid initial grid.
	std::vector<GpuProbe> zeroed(static_cast<size_t>(probeGridMeta_.totalProbes));
	const size_t bytes = zeroed.size() * sizeof(GpuProbe);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeGrid_), bytes));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_probeGrid_), zeroed.data(), bytes, cudaMemcpyHostToDevice));

	// Path guiding's own per-probe histogram array (gpu/optix/wavefront_
	// guiding.h) - built here, alongside d_probeGrid_ above, NOT gated on
	// pathGuidingEnabled_ (cheap, matches d_probeGrid_'s own "always built,
	// usage gated separately" precedent - see enablePathGuiding()'s own
	// comment). GpuGuidingHistogram's own default member initializers
	// already give every field its correct "never updated" state
	// (numSamplesEverAdded==0), same zeroed-vector-upload pattern as above.
	std::vector<GpuGuidingHistogram> zeroedHistograms(static_cast<size_t>(probeGridMeta_.totalProbes));
	const size_t histogramBytes = zeroedHistograms.size() * sizeof(GpuGuidingHistogram);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_guidingHistograms_), histogramBytes));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_guidingHistograms_), zeroedHistograms.data(), histogramBytes, cudaMemcpyHostToDevice));

	std::cout << "[OptiX] Built probe cache grid (" << dimX << "x" << dimY << "x" << dimZ
			  << " = " << probeGridMeta_.totalProbes << " probes, spacing=" << spacing << ")\n";
}

// ============================================================================
// buildScene() section helpers -- each uploads one self-contained slice of
// scene data to the device, touching only its own matching d_*_/num*_
// members (see each one's own declaration comment, optix_renderer.h).
// Extracted out of buildScene() itself purely to keep that function's own
// top-to-bottom flow (geometry -> lights -> accel structures -> SBT)
// readable; call order/semantics are unchanged from when this was inline.
// ============================================================================

void OptiXRenderer::uploadMaterials(const std::vector<MaterialData>& materials) {
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
}

void OptiXRenderer::uploadTextures(const std::vector<TextureData>& textures,
									const std::vector<unsigned char>& texturePixels) {
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
}

void OptiXRenderer::uploadQuads(const std::vector<QuadData>& quads) {
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
}

void OptiXRenderer::uploadBilinearPatches(const std::vector<BilinearPatchData>& bilinearPatches) {
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
}

void OptiXRenderer::uploadDisks(const std::vector<DiskData>& disks) {
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
}

void OptiXRenderer::uploadCylinders(const std::vector<CylinderData>& cylinders) {
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
}

void OptiXRenderer::uploadLensTables(const std::vector<GpuLensElement>& lensElements,
									  const std::vector<GpuExitPupilBounds>& exitPupilBounds) {
	// Both come from scene_builder.cpp directly instantiating a host-side
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
}

void OptiXRenderer::uploadCloudMedia(const std::vector<CloudMedium<float>>& cloudMediums) {
	// CloudMedium<float> is uploaded as-is (see optix_types.h's
	// cloud_medium.h include comment).
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
}

void OptiXRenderer::uploadRgbGridMedia(const std::vector<GpuRgbGridMedium>& rgbGridMediums,
										const std::vector<float>& rgbGridData) {
	// Metadata table plus the shared flat voxel-data buffer it slices into
	// (see GpuRgbGridMedium::dataOffset).
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
}

void OptiXRenderer::uploadGridMedia(const std::vector<GpuGridMedium>& gridMediums,
									 const std::vector<float>& gridData) {
	// Same two-array upload pattern as RGB grid media (uploadRgbGridMedia).
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
}

void OptiXRenderer::uploadBssrdfTables(const std::vector<GpuBssrdfTable>& bssrdfTables,
										const std::vector<float>& bssrdfRhoSamples,
										const std::vector<float>& bssrdfRadiusSamples,
										const std::vector<float>& bssrdfProfile,
										const std::vector<float>& bssrdfProfileCdf) {
	// Tabulated BSSRDF tables (MaterialType::Subsurface, recursive backend
	// only, Phase 1 - see optix_types.h's GpuBssrdfTable comment): one small
	// metadata array (GpuBssrdfTable) plus four shared flat float buffers it
	// slices into.
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
}

void OptiXRenderer::uploadMeasuredTables(const std::vector<GpuMeasuredTable>& measuredTables,
										  const std::vector<float>& measuredParamValues,
										  const std::vector<float>& measuredData,
										  const std::vector<float>& measuredMcdf,
										  const std::vector<float>& measuredCcdf) {
	// Real tabulated measured-BRDF tables (MaterialType::Measured, both GPU
	// backends - see optix_types.h's GpuMeasuredTable comment). Same upload
	// shape as the BSSRDF tables (uploadBssrdfTables): one small metadata
	// array (GpuMeasuredTable, itself 5 GpuPL2DTable sub-tables) plus four
	// shared flat float buffers it slices into.
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
}

void OptiXRenderer::uploadSkyLight(const std::vector<float>& skyImagePixels,
									const std::vector<float>& skyMarginalCdf,
									const std::vector<float>& skyMarginalFunc,
									float skyMarginalFuncInt,
									const std::vector<float>& skyConditionalCdf,
									const std::vector<float>& skyConditionalFunc,
									const std::vector<float>& skyConditionalFuncInt,
									int skyWidth, int skyHeight, float skyScale) {
	// Real importance-sampled HDR sky distribution (LightSource "infinite"
	// with an image - see optix_types.h's GpuSkyDistribution comment). Same
	// upload shape as the BSSRDF/measured-BRDF tables, minus the per-table
	// metadata array (a scene has at most one infinite light, so there is
	// nothing to dedup/index - just the flat buffers themselves, referenced
	// later by GpuSkyDistribution's own pointer fields). wf_upload_gpu_buf
	// is the file-scope helper shared with uploadPortalLight() below.
	skyWidth_ = skyWidth;
	skyHeight_ = skyHeight;
	skyScale_ = skyScale;
	skyMarginalFuncInt_ = skyMarginalFuncInt;

	wf_upload_gpu_buf(skyImagePixels, d_skyImagePixels_);
	wf_upload_gpu_buf(skyMarginalCdf, d_skyMarginalCdf_);
	wf_upload_gpu_buf(skyMarginalFunc, d_skyMarginalFunc_);
	wf_upload_gpu_buf(skyConditionalCdf, d_skyConditionalCdf_);
	wf_upload_gpu_buf(skyConditionalFunc, d_skyConditionalFunc_);
	wf_upload_gpu_buf(skyConditionalFuncInt, d_skyConditionalFuncInt_);

	if (skyHeight_ > 0)
		std::cout << "[OptiX] Uploaded real HDR sky distribution (" << skyWidth_ << "x" << skyHeight_
			<< ", " << skyImagePixels.size() << " pixel floats) to GPU\n";
}

void OptiXRenderer::uploadPortalLight(const std::vector<float>& portalRectifiedImage,
									   const std::vector<float>& portalDistFunc,
									   const std::vector<double>& portalSatSum,
									   int portalWidth, int portalHeight, float portalScale,
									   float3 portalFrameX, float3 portalFrameY, float3 portalFrameZ,
									   float3 portalP0, float3 portalP2) {
	// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's own
	// comment (optix_types.h). Mutually exclusive with uploadSkyLight()
	// above (matches CPU) - same "flat buffers, no per-table metadata"
	// upload shape.
	portalWidth_ = portalWidth;
	portalHeight_ = portalHeight;
	portalScale_ = portalScale;
	portalFrameX_ = portalFrameX; portalFrameY_ = portalFrameY; portalFrameZ_ = portalFrameZ;
	portalP0_ = portalP0; portalP2_ = portalP2;

	wf_upload_gpu_buf(portalRectifiedImage, d_portalRectifiedImage_);
	wf_upload_gpu_buf(portalDistFunc, d_portalDistFunc_);
	wf_upload_gpu_buf(portalSatSum, d_portalSatSum_);

	if (portalHeight_ > 0)
		std::cout << "[OptiX] Uploaded real portal infinite light (" << portalWidth_ << "x" << portalHeight_
			<< ", " << portalRectifiedImage.size() << " rectified-image floats) to GPU\n";
}

void OptiXRenderer::uploadPunctualLights(const std::vector<PunctualLightGPU>& punctualLights) {
	// Separate from the area-light arrays uploaded elsewhere - punctual
	// lights are evaluated deterministically every hit rather than selected
	// via the alias table (see optix_device_helpers.h eval_punctual_light /
	// add_punctual_lights_lambertian).
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
}
