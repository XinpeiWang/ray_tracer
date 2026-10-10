// metal_poc_pbrt_nanovdb.mm
// pbrt "nanovdb" media (MakeNamedMedium "nanovdb", scene E9) on Metal. The file is read and baked into a dense density grid on the host by src/shared/nanovdb_dense.h (the step the
// CPU renderer and both OptiX backends use), then drawn by the per-voxel RGB grid medium (materialType 30, shadeRgbGridMediumSphere) the "rgbgrid" medium already has, so no NanoVDB
// code runs on the device. Pure scattering with sigma_s = luminance(medium sigma_s) * density, like the CPU's bake and OptiX's grid. The grid has no per-voxel emission, so a
// "temperaturename" grid is dropped with a warning (the density is rendered).

#include "metal_poc_app.h"
#include "../../src/shared/nanovdb_dense.h"

#include <cstdio>
#include <map>
#include <memory>

namespace {

// One read per (file, grid): a Live Preview rebuilds its scene whenever the picture size changes.
std::shared_ptr<const nanovdb_dense::Grid> cachedGrid(const std::string& file, const std::string& gridName) {
    static std::map<std::string, std::shared_ptr<const nanovdb_dense::Grid>> cache;
    const std::string key = file + "\n" + gridName;
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, std::make_shared<const nanovdb_dense::Grid>(nanovdb_dense::readGrid(file, gridName, std::string()))).first;
    return it->second;
}

// A grid of more voxels than this is not uploaded (three floats a voxel in the shared buffer): the medium is left invisible and the scene says so.
constexpr size_t kMaxVoxels = 64u * 1024u * 1024u;

}  // namespace

// `cOff` is bboxCenter - sceneOffset / sceneScale (the same offset the rgbgrid branch derives), so the world->medium map composes with the Metal scene's rescale and recentre.
// Fills `mat` and returns true when the grid was usable; otherwise leaves `mat` alone and returns false.
bool MetalPocApp::mapNanovdbMedium(const pbrt_flatten::Medium& nm, float sceneScale, float cOffX, float cOffY, float cOffZ, TriangleMaterial& mat) {
    const std::shared_ptr<const nanovdb_dense::Grid> grid = cachedGrid(nm.nanovdbFilename, nm.nanovdbGridName);
    nanovdb_dense::Placement placement;
    if (!grid || !grid->ok() || !nanovdb_dense::placeInWorld(*grid, nm.nanovdbXform, placement)) return false;
    const size_t voxels = (size_t)grid->nx * grid->ny * grid->nz;
    if (voxels > kMaxVoxels) {
        fprintf(stderr, "loadPbrtScene: nanovdb grid \"%s\" has %zu voxels (more than %zu): not uploaded, the medium renders as empty\n", nm.nanovdbFilename.c_str(), voxels, kMaxVoxels);
        return false;
    }
    if (!nm.nanovdbTemperatureGridName.empty())
        fprintf(stderr, "loadPbrtScene: nanovdb medium names a temperature grid \"%s\": the Metal grid medium has no per-voxel emission, the density is rendered without the glow\n",
                nm.nanovdbTemperatureGridName.c_str());

    const float invScale = 1.0f / sceneScale;
    GpuRgbGridMedium g{};
    for (int i = 0; i < 3; ++i) { g.boundsMin[i] = 0.0f; g.boundsMax[i] = 1.0f; }   // the grid's own unit index cube: medium space is [0,1]^3
    // world (pbrt) -> medium is placement.toMedium*; the Metal scene's point is pbrt p = (pMetal - sceneOffset)/sceneScale + bboxCenter, so (as for rgbgrid) the matrix takes the
    // 1/sceneScale and the translation gains M * cOff.
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) g.worldToMediumMat[r * 3 + c] = (float)placement.toMediumMat[r * 3 + c] * invScale;
        g.worldToMediumTranslate[r] = (float)placement.toMediumTranslate[r]
            + (float)(placement.toMediumMat[r * 3 + 0] * cOffX + placement.toMediumMat[r * 3 + 1] * cOffY + placement.toMediumMat[r * 3 + 2] * cOffZ);
    }
    g.nx = grid->nx; g.ny = grid->ny; g.nz = grid->nz;
    g.dataOffset = (int)rgbGridData.size();
    g.sigmaScale = invScale;
    g.phaseG = (float)nm.g;
    // sigma_s grids R, G, B (all three the same: the medium is grey) = density * luminance(sigma_s); no absorption (pure scattering), no emission.
    const float sigmaS = (float)(0.2126 * nm.sigma_s[0] + 0.7152 * nm.sigma_s[1] + 0.0722 * nm.sigma_s[2]);
    float maxDensity = 0.0f;
    for (float d : grid->density) maxDensity = std::max(maxDensity, d);
    for (int channel = 0; channel < 3; ++channel)
        for (float d : grid->density) rgbGridData.push_back(d * sigmaS);
    g.saDataOffset = -1;
    g.sigmaAConst = 0.0f;
    g.leDataOffset = -1;
    g.sigmaMaj = maxDensity * sigmaS * invScale * 1.01f;
    const int gridIdx = (int)rgbGridMediums.size();
    rgbGridMediums.push_back(g);
    mat = TriangleMaterial{};
    mat.materialType = METAL_MAT_MEDIUM_RGB_GRID;
    mat.lightId = -1;
    mat.conductorEta = PackedFloat3{(float)gridIdx, 0.0f, 0.0f};
    return true;
}
