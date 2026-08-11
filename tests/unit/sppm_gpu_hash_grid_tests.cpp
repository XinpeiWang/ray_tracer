// sppm_gpu_hash_grid_tests.cpp
// Isolated host-driven verification of sppm_hash_grid_insert_kernel (GPU
// SPPM sub-phase 1c) -- no OptiX pipeline/SBT/traversal involved, just the
// plain CUDA kernel in sppm_kernels.cu via its sppm_launch_hash_grid_insert
// host wrapper (sppm_launch.cu). De-risks the hash grid -- the piece with
// no existing precedent anywhere in this codebase -- before it becomes
// load-bearing in the sub-phase 1d photon pass.
#include "gtest/gtest.h"
#include "optix_types.h"
#include "sppm_types.h"
#include <cuda_runtime.h>
#include <unordered_set>
#include <vector>

extern "C" void sppm_launch_hash_grid_insert(
	const SPPMPixelGPU* d_pixels, int numPixels,
	SPPMHashGridParams gridParams,
	int* d_cellHead, int* d_nodeNext, int* d_nodePixel,
	cudaStream_t stream);

namespace {

// Host-side reproduction of sppm_to_grid/sppm_hash_bucket (sppm_types.h),
// used only to compute the EXPECTED bucket for each synthetic point
// independently of the kernel under test.
void ExpectedGridCoords(const SPPMHashGridParams& gp, float wx, float wy, float wz, int out[3]) {
	float pg[3] = {
		(wx - gp.gridMin.x) / (gp.gridMax.x - gp.gridMin.x),
		(wy - gp.gridMin.y) / (gp.gridMax.y - gp.gridMin.y),
		(wz - gp.gridMin.z) / (gp.gridMax.z - gp.gridMin.z)
	};
	for (int d = 0; d < 3; ++d) {
		out[d] = static_cast<int>(gp.gridRes[d] * pg[d]);
		if (out[d] < 0) out[d] = 0;
		if (out[d] >= gp.gridRes[d]) out[d] = gp.gridRes[d] - 1;
	}
}

unsigned int ExpectedHashPoint3i(int x, int y, int z) {
	unsigned int a = static_cast<unsigned int>(x) * 2654435769u;
	unsigned int b = static_cast<unsigned int>(y) * 805459861u;
	unsigned int c = static_cast<unsigned int>(z) * 3674653429u;
	return a ^ b ^ c;
}

int ExpectedBucket(const SPPMHashGridParams& gp, float wx, float wy, float wz) {
	int gi[3];
	ExpectedGridCoords(gp, wx, wy, wz, gi);
	return static_cast<int>(ExpectedHashPoint3i(gi[0], gi[1], gi[2]) % static_cast<unsigned int>(gp.hashSize));
}

SPPMPixelGPU MakeVisiblePoint(float x, float y, float z, float radius) {
	SPPMPixelGPU px{};
	px.radius = radius;
	px.vp_p = make_float3(x, y, z);
	px.vp_valid = true;
	return px;
}

// Reads back cellHead/nodeNext/nodePixel and, for each bucket, walks its
// linked list into a plain vector of pixel indices -- the GPU analog of
// CPU's forward_list traversal, done host-side after copy-back.
std::vector<std::vector<int>> ReadBackBuckets(
	const std::vector<int>& cellHead, const std::vector<int>& nodeNext, const std::vector<int>& nodePixel) {
	std::vector<std::vector<int>> buckets(cellHead.size());
	for (size_t h = 0; h < cellHead.size(); ++h) {
		int slot = cellHead[h];
		while (slot != -1) {
			buckets[h].push_back(nodePixel[slot]);
			slot = nodeNext[slot];
		}
	}
	return buckets;
}

class SppmGpuHashGridTest : public ::testing::Test {
  protected:
	SPPMHashGridParams MakeGridParams(int hashSize, int res) const {
		SPPMHashGridParams gp{};
		gp.gridMin = make_float3(-10.0f, -10.0f, -10.0f);
		gp.gridMax = make_float3(10.0f, 10.0f, 10.0f);
		gp.gridRes[0] = gp.gridRes[1] = gp.gridRes[2] = res;
		gp.cellSize = make_float3(20.0f / res, 20.0f / res, 20.0f / res);
		gp.hashSize = hashSize;
		return gp;
	}

	// Uploads pixels, runs the kernel, copies results back. Node pool is
	// sized numPixels*kSPPMMaxCellsPerPoint to match production usage
	// (sppm_path_tracer.cpp).
	void RunInsert(const std::vector<SPPMPixelGPU>& pixels, const SPPMHashGridParams& gp,
	               std::vector<int>* cellHead, std::vector<int>* nodeNext, std::vector<int>* nodePixel) {
		const int numPixels = static_cast<int>(pixels.size());
		const int poolSize = numPixels * kSPPMMaxCellsPerPoint;

		SPPMPixelGPU* d_pixels = nullptr;
		int* d_cellHead = nullptr;
		int* d_nodeNext = nullptr;
		int* d_nodePixel = nullptr;

		ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_pixels), numPixels * sizeof(SPPMPixelGPU)), cudaSuccess);
		ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_cellHead), gp.hashSize * sizeof(int)), cudaSuccess);
		ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_nodeNext), poolSize * sizeof(int)), cudaSuccess);
		ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_nodePixel), poolSize * sizeof(int)), cudaSuccess);

		ASSERT_EQ(cudaMemcpy(d_pixels, pixels.data(), numPixels * sizeof(SPPMPixelGPU), cudaMemcpyHostToDevice),
		          cudaSuccess);
		ASSERT_EQ(cudaMemset(d_cellHead, 0xFF, gp.hashSize * sizeof(int)), cudaSuccess); // -1 fill
		ASSERT_EQ(cudaMemset(d_nodeNext, 0xFF, poolSize * sizeof(int)), cudaSuccess);
		ASSERT_EQ(cudaMemset(d_nodePixel, 0xFF, poolSize * sizeof(int)), cudaSuccess);

		sppm_launch_hash_grid_insert(d_pixels, numPixels, gp, d_cellHead, d_nodeNext, d_nodePixel, nullptr);
		ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

		cellHead->resize(gp.hashSize);
		nodeNext->resize(poolSize);
		nodePixel->resize(poolSize);
		ASSERT_EQ(cudaMemcpy(cellHead->data(), d_cellHead, gp.hashSize * sizeof(int), cudaMemcpyDeviceToHost),
		          cudaSuccess);
		ASSERT_EQ(cudaMemcpy(nodeNext->data(), d_nodeNext, poolSize * sizeof(int), cudaMemcpyDeviceToHost),
		          cudaSuccess);
		ASSERT_EQ(cudaMemcpy(nodePixel->data(), d_nodePixel, poolSize * sizeof(int), cudaMemcpyDeviceToHost),
		          cudaSuccess);

		cudaFree(d_pixels);
		cudaFree(d_cellHead);
		cudaFree(d_nodeNext);
		cudaFree(d_nodePixel);
	}
};

TEST_F(SppmGpuHashGridTest, SinglePointLandsInExpectedBucket) {
	// Coordinates deliberately offset from any grid-cell boundary (cellSize
	// here is 1.0 world unit): a point sitting exactly on a boundary would
	// have its radius-expanded AABB straddle into a neighbor cell even for
	// a tiny radius, which is correct kernel behavior but would make this
	// single-bucket assertion spuriously fail on a case the kernel is
	// actually handling right -- see LargeRadiusRegistersMultipleCells for
	// that behavior tested on purpose instead.
	SPPMHashGridParams gp = MakeGridParams(101, 20);
	std::vector<SPPMPixelGPU> pixels = {MakeVisiblePoint(3.37f, -1.62f, 5.13f, 0.01f)};

	std::vector<int> cellHead, nodeNext, nodePixel;
	RunInsert(pixels, gp, &cellHead, &nodeNext, &nodePixel);

	int expected = ExpectedBucket(gp, 3.37f, -1.62f, 5.13f);
	auto buckets = ReadBackBuckets(cellHead, nodeNext, nodePixel);

	ASSERT_EQ(buckets[expected].size(), 1u);
	EXPECT_EQ(buckets[expected][0], 0);

	for (size_t h = 0; h < buckets.size(); ++h) {
		if (static_cast<int>(h) != expected) EXPECT_TRUE(buckets[h].empty());
	}
}

TEST_F(SppmGpuHashGridTest, InvalidPixelsAreSkipped) {
	SPPMHashGridParams gp = MakeGridParams(101, 20);
	SPPMPixelGPU valid = MakeVisiblePoint(0.37f, 0.21f, 0.44f, 0.01f); // mid-cell, see above
	SPPMPixelGPU invalid = MakeVisiblePoint(1.37f, 1.21f, 1.44f, 0.01f);
	invalid.vp_valid = false;
	std::vector<SPPMPixelGPU> pixels = {valid, invalid};

	std::vector<int> cellHead, nodeNext, nodePixel;
	RunInsert(pixels, gp, &cellHead, &nodeNext, &nodePixel);
	auto buckets = ReadBackBuckets(cellHead, nodeNext, nodePixel);

	int total = 0;
	for (auto& b : buckets) total += static_cast<int>(b.size());
	ASSERT_EQ(total, 1);

	int expected = ExpectedBucket(gp, 0.37f, 0.21f, 0.44f);
	ASSERT_EQ(buckets[expected].size(), 1u);
	EXPECT_EQ(buckets[expected][0], 0);
}

// A radius large enough to span multiple grid cells should register the
// point into every one of them (up to kSPPMMaxCellsPerPoint), not just the
// bucket its center point falls in -- this is the entire reason the search
// AABB (pMin/pMax) exists rather than a single-cell hash lookup.
TEST_F(SppmGpuHashGridTest, LargeRadiusRegistersMultipleCells) {
	SPPMHashGridParams gp = MakeGridParams(211, 20); // cellSize = 1.0 world unit
	std::vector<SPPMPixelGPU> pixels = {MakeVisiblePoint(0.0f, 0.0f, 0.0f, 1.4f)};

	std::vector<int> cellHead, nodeNext, nodePixel;
	RunInsert(pixels, gp, &cellHead, &nodeNext, &nodePixel);
	auto buckets = ReadBackBuckets(cellHead, nodeNext, nodePixel);

	std::unordered_set<int> nonEmpty;
	int totalEntries = 0;
	for (size_t h = 0; h < buckets.size(); ++h) {
		if (!buckets[h].empty()) nonEmpty.insert(static_cast<int>(h));
		totalEntries += static_cast<int>(buckets[h].size());
		for (int pixelIdx : buckets[h]) EXPECT_EQ(pixelIdx, 0);
	}

	// radius 1.4 / cellSize 1.0 spans a 3x3x3 = 27-cell AABB, capped at
	// kSPPMMaxCellsPerPoint (8) insertions -- exercising the documented cap.
	EXPECT_EQ(totalEntries, kSPPMMaxCellsPerPoint);
	EXPECT_GT(nonEmpty.size(), 1u);
}

TEST_F(SppmGpuHashGridTest, MultiplePointsInSameCellShareBucketWithoutLosingEntries) {
	SPPMHashGridParams gp = MakeGridParams(101, 4); // coarse grid: cellSize = 5.0
	std::vector<SPPMPixelGPU> pixels = {
		MakeVisiblePoint(0.1f, 0.1f, 0.1f, 0.01f),
		MakeVisiblePoint(0.2f, 0.2f, 0.2f, 0.01f),
		MakeVisiblePoint(0.3f, 0.3f, 0.3f, 0.01f),
	};

	std::vector<int> cellHead, nodeNext, nodePixel;
	RunInsert(pixels, gp, &cellHead, &nodeNext, &nodePixel);
	auto buckets = ReadBackBuckets(cellHead, nodeNext, nodePixel);

	int expected = ExpectedBucket(gp, 0.1f, 0.1f, 0.1f);
	ASSERT_EQ(buckets[expected].size(), 3u);

	std::unordered_set<int> seen(buckets[expected].begin(), buckets[expected].end());
	EXPECT_EQ(seen.size(), 3u); // atomicExch prepend must not drop concurrent inserts
	EXPECT_TRUE(seen.count(0) && seen.count(1) && seen.count(2));
}

TEST_F(SppmGpuHashGridTest, EmptyPixelListIsANoOp) {
	SPPMHashGridParams gp = MakeGridParams(101, 20);
	std::vector<SPPMPixelGPU> pixels; // empty

	std::vector<int> cellHead, nodeNext, nodePixel;
	RunInsert(pixels, gp, &cellHead, &nodeNext, &nodePixel);
	auto buckets = ReadBackBuckets(cellHead, nodeNext, nodePixel);

	for (auto& b : buckets) EXPECT_TRUE(b.empty());
}

} // namespace
