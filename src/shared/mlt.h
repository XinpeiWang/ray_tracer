#pragma once
// ---------------------------------------------------------------------------
// mlt.h -- Metropolis Light Transport (pbrt-v4 port)
//
// Ports pbrt-v4:
//   samplers.h / samplers.cpp  -- MLTSampler + PrimarySample
//   cpu/integrators.cpp        -- MLTIntegrator::L, MLTIntegrator::Render
//
// Components:
//   mlt_detail::ErfInv(u)           -- inverse error function (pbrt-v4 math.h)
//   mlt_detail::SampleNormal(u,s)   -- Gaussian sample via erf-inv
//   MLTSampler<T>                   -- primary-sample-space Markov sampler
//   MLTPathResult<T>                -- output of one path evaluation
//   MLTEvalPath<T,Scene>()          -- evaluate one BDPT (s,t) strategy
//                                      (mirrors MLTIntegrator::L)
//   MLTBootstrap<T,Scene>()         -- warm-up: compute b = E[f]
//   MLTRenderLoop<T,Scene>()        -- full Metropolis render loop
//
// Design rules (same as bdpt.h):
//   - Header-only, no virtual functions, no heap allocation in hot paths
//   - Template T: float or double
//   - Template Scene: same duck-typed scene concept as bdpt.h
//   - Threading: single-threaded reference; callers wrap in parallel loops
//   - Output: per-splat callback instead of film API
//
// Scene concept extensions required beyond bdpt.h:
//   // Sample pixel position uniformly in [0,1)^2 (for camera subpath start)
//   void SamplePixel2D(T u1, T u2, T& px, T& py) const;
//   // Convert (px,py) in [0,1)^2 to a camera ray
//   bool PixelToRay(T px, T py, T ray_o[3], T ray_d[3], T cam_n[3]) const;
//   // Luminance of a spectral RGB triple (scalar importance measure)
//   T Luminance(T r, T g, T b) const;
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "rng.h"
#include "bdpt.h"
#include "reservoir_sampler.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstdint>
#include <cassert>
#include <limits>

// ===========================================================================
// Section 1 -- Math helpers: ErfInv and SampleNormal
// ===========================================================================

namespace mlt_detail {

// ErfInv: inverse error function.
// Mirrors pbrt-v4 util/math.h ErfInv() (Winitzki polynomial approximation).
// Reference: https://stackoverflow.com/a/49743348
CPU_GPU float ErfInvF(float a) {
	float p;
	float t = std::log(std::max(1.f - a*a, 1e-38f));
	if (std::abs(t) > 6.125f) {
		p = 3.03697567e-10f;
		p = p*t + 2.93243101e-8f;
		p = p*t + 1.22150334e-6f;
		p = p*t + 2.84108955e-5f;
		p = p*t + 3.93552968e-4f;
		p = p*t + 3.02698812e-3f;
		p = p*t + 4.83185798e-3f;
		p = p*t - 2.64646143e-1f;
		p = p*t + 8.40016484e-1f;
	} else {
		p = 5.43877832e-9f;
		p = p*t + 1.43286059e-7f;
		p = p*t + 1.22775396e-6f;
		p = p*t + 1.12962631e-7f;
		p = p*t - 5.61531961e-5f;
		p = p*t - 1.47697705e-4f;
		p = p*t + 2.31468701e-3f;
		p = p*t + 1.15392562e-2f;
		p = p*t - 2.32015476e-1f;
		p = p*t + 8.86226892e-1f;
	}
	return a * p;
}

CPU_GPU double ErfInvD(double a) {
	return (double)ErfInvF((float)a);
}

template<typename T> CPU_GPU T ErfInv(T a);
template<> CPU_GPU float  ErfInv<float> (float  a) { return ErfInvF(a); }
template<> CPU_GPU double ErfInv<double>(double a) { return ErfInvD(a); }

// SampleNormal: sample N(0, sigma) via inverse CDF.
// pbrt-v4: SampleNormal(u, mu=0, sigma) = mu + sqrt(2)*sigma*ErfInv(2u-1)
template<typename T>
CPU_GPU T SampleNormal(T u, T sigma) {
	return T(1.41421356237309504880) * sigma * ErfInv<T>(T(2)*u - T(1));
}

} // namespace mlt_detail

// ===========================================================================
// Section 2 -- MLTSampler<T>
// ===========================================================================
// Mirrors pbrt-v4 MLTSampler (samplers.h lines 634-760, samplers.cpp 321-396).
// Primary-sample-space Markov sampler with:
//   - Large steps: draw fresh uniform samples (probability largeStepProb)
//   - Small steps: Gaussian perturbation (std = sigma * sqrt(nSmall))
//   - Per-stream indexing: dimension = streamIndex + streamCount * sampleIndex
//   - Accept/Reject via per-sample backup/restore

template<typename T>
class MLTSampler {
public:
	// Stream indices (pbrt-v4 MLTIntegrator constants)
	static constexpr int kCameraStream     = 0;
	static constexpr int kLightStream      = 1;
	static constexpr int kConnectionStream = 2;
	static constexpr int kStreamCount      = 3;

	// PrimarySample: one dimension in primary-sample space with backup
	struct PrimarySample {
		T       value                    = T(0);
		int64_t lastModificationIter     = 0;
		T       valueBackup              = T(0);
		int64_t modifyBackup             = 0;

		void Backup()  { valueBackup = value; modifyBackup = lastModificationIter; }
		void Restore() { value = valueBackup; lastModificationIter = modifyBackup; }
	};

	// ctor mirrors pbrt-v4: MLTSampler(mutationsPerPixel, rngSequenceIndex, sigma,
	//                                  largeStepProbability, streamCount)
	MLTSampler(int mutationsPerPixel, uint64_t rngSeed,
			   T sigma, T largeStepProbability, int streamCount = kStreamCount)
		: mutationsPerPixel_(mutationsPerPixel)
		, rng_(rngSeed)
		, sigma_(sigma)
		, largeStepProb_(largeStepProbability)
		, streamCount_(streamCount)
		, streamIndex_(0)
		, sampleIndex_(0)
		, currentIteration_(0)
		, largeStep_(true)
		, lastLargeStepIter_(0)
	{}

	// StartIteration: begin a new proposal (pbrt-v4 StartIteration)
	void StartIteration() {
		++currentIteration_;
		largeStep_ = (rng_.template Uniform<float>() < (float)largeStepProb_);
	}

	// Accept: commit current proposal (pbrt-v4 Accept)
	void Accept() {
		if (largeStep_)
			lastLargeStepIter_ = currentIteration_;
	}

	// Reject: restore all samples modified this iteration (pbrt-v4 Reject)
	void Reject() {
		for (auto& xi : X_)
			if (xi.lastModificationIter == currentIteration_)
				xi.Restore();
		--currentIteration_;
	}

	// StartStream: set active stream (pbrt-v4 StartStream)
	void StartStream(int index) {
		assert(index < streamCount_);
		streamIndex_ = index;
		sampleIndex_ = 0;
	}

	// GetNextIndex: dimension index for current sample (pbrt-v4 GetNextIndex)
	int GetNextIndex() { return streamIndex_ + streamCount_ * sampleIndex_++; }

	// Get1D: return next sample from the primary-sample vector
	T Get1D() {
		int idx = GetNextIndex();
		EnsureReady(idx);
		return X_[idx].value;
	}

	// Get2D / GetPixel2D
	void Get2D(T& u1, T& u2) { u1 = Get1D(); u2 = Get1D(); }
	void GetPixel2D(T& u1, T& u2) { Get2D(u1, u2); }

	int MutationsPerPixel() const { return mutationsPerPixel_; }
	int64_t CurrentIteration() const { return currentIteration_; }
	bool IsLargeStep() const { return largeStep_; }

	// Access to raw primary-sample vector (for tests)
	const std::vector<PrimarySample>& X() const { return X_; }

private:
	void EnsureReady(int index) {
		if (index >= (int)X_.size())
			X_.resize(index + 1);
		PrimarySample& xi = X_[index];

		// Reset if a large step happened since last touch
		if (xi.lastModificationIter < lastLargeStepIter_) {
			xi.value = (T)rng_.template Uniform<float>();
			xi.lastModificationIter = lastLargeStepIter_;
		}

		xi.Backup();

		if (largeStep_) {
			xi.value = (T)rng_.template Uniform<float>();
		} else {
			int64_t nSmall = currentIteration_ - xi.lastModificationIter;
			T effSigma = sigma_ * std::sqrt((T)nSmall);
			T delta = mlt_detail::SampleNormal<T>((T)rng_.template Uniform<float>(), effSigma);
			xi.value += delta;
			// Wrap into [0,1) -- mirrors pbrt-v4: x -= floor(x)
			xi.value -= std::floor(xi.value);
		}
		xi.lastModificationIter = currentIteration_;
	}

	int     mutationsPerPixel_;
	RNG     rng_;
	T       sigma_;
	T       largeStepProb_;
	int     streamCount_;
	int     streamIndex_;
	int     sampleIndex_;
	int64_t currentIteration_;
	bool    largeStep_;
	int64_t lastLargeStepIter_;
	std::vector<PrimarySample> X_;
};

// ===========================================================================
// Section 3 -- MLTPathResult<T>
// ===========================================================================

template<typename T>
struct MLTPathResult {
	T L[3];       // RGB radiance estimate
	T px, py;     // film/pixel position in [0,1)^2
	int s, t;     // BDPT strategy used
	bool valid;
};

// ===========================================================================
// Section 4 -- MLTEvalPath: evaluate one BDPT (s,t) strategy via MLTSampler
// ===========================================================================
// Mirrors pbrt-v4 MLTIntegrator::L().
// The MLTSampler drives all random decisions via its primary-sample vector;
// the Scene provides the same query API as bdpt.h.
//
// depth = s + t - 2  (path length from 0)
// nStrategies = depth + 2  (or 1 for depth == 0)
// s is chosen uniformly from 0..nStrategies-1; t = nStrategies - s.

template<typename T, typename Scene>
MLTPathResult<T> MLTEvalPath(MLTSampler<T>& sampler, int depth,
							  const Scene& scene, int maxDepth) {
	MLTPathResult<T> result{};
	result.valid = false;
	result.L[0] = result.L[1] = result.L[2] = T(0);

	// pbrt-v4: sampler.StartStream(cameraStreamIndex) is called ONCE at the
	// top, then the same stream is used for strategy selection, pixel sampling,
	// and camera subpath generation (no repeated StartStream).
	sampler.StartStream(MLTSampler<T>::kCameraStream);

	// Determine strategy (s,t) for this depth
	// pbrt-v4: if depth==0, s=0,t=2,nStrategies=1; else sample s uniformly.
	int nStrategies, s, t;
	if (depth == 0) {
		nStrategies = 1; s = 0; t = 2;
	} else {
		nStrategies = depth + 2;
		T uStrategy = sampler.Get1D();   // camera stream dim 0
		s = std::min((int)(uStrategy * nStrategies), nStrategies - 1);
		t = nStrategies - s;
	}
	result.s = s; result.t = t;

	// Sample pixel position -- continues in camera stream (dim 0 for depth==0,
	// or dims 1+2 for depth>0, matching pbrt-v4's GetPixel2D call order).
	T px, py;
	sampler.GetPixel2D(px, py);
	result.px = px; result.py = py;

	// Build camera ray from pixel position
	T cam_p[3], cam_n[3], ray_d[3];
	if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n))
		return result;

	// Allocate subpath storage
	std::vector<BDPTVertex<T>> cameraVerts(t > 0 ? t : 1);
	std::vector<BDPTVertex<T>> lightVerts(s > 0 ? s : 1);

	// MLTSceneAdapter: wraps the scene and routes RandFloat() through
	// the currently active MLTSampler stream, exactly as pbrt-v4's
	// GenerateCameraSubpath / GenerateLightSubpath call sampler.Get1D()/Get2D().
	struct MLTSceneAdapter {
		const Scene&    scene;
		MLTSampler<T>&  sampler;

		bool Intersect(const T org[3], const T dir[3], T t_max, BDPTHit<T>& hit) const {
			return scene.Intersect(org, dir, t_max, hit);
		}
		bool Unoccluded(const T p0[3], const T p1[3]) const {
			return scene.Unoccluded(p0, p1);
		}
		bool SampleLight(T u, const T ref_p[3], BDPTLightSample<T>& ls) const {
			return scene.SampleLight(u, ref_p, ls);
		}
		bool SampleLightLe(T u1d, const T u2a[2], const T u2b[2], BDPTLightLeSample<T>& les) const {
			return scene.SampleLightLe(u1d, u2a, u2b, les);
		}
		T LightPMF(int id) const { return scene.LightPMF(id); }
		void LightPDFLe(int id, const T* p, const T* n, const T* w,
						T& pdfPos, T& pdfDir) const {
			scene.LightPDFLe(id, p, n, w, pdfPos, pdfDir);
		}
		void CameraPDFWe(const T* o, const T* d, T& pdfPos, T& pdfDir) const {
			scene.CameraPDFWe(o, d, pdfPos, pdfDir);
		}
		bool CameraSampleWi(const T ref_p[3], const T* u2,
							T wi[3], T& pdf, T& importance, T pRaster[2]) const {
			return scene.CameraSampleWi(ref_p, u2, wi, pdf, importance, pRaster);
		}
		void SceneBoundingSphere(T center[3], T& radius) const {
			scene.SceneBoundingSphere(center, radius);
		}
		void InfiniteLightLe(const T d[3], T out[3]) const {
			scene.InfiniteLightLe(d, out);
		}
		float InfiniteLightDensity(const T d[3]) const {
			return scene.InfiniteLightDensity(d);
		}
		void BSDFf(int id, const T wo[3], const T wi[3], const T n[3], T out[3]) const {
			scene.BSDFf(id, wo, wi, n, out);
		}
		bool BSDFSampleF(int id, const T wo[3], const T n[3], T u1, T u2,
						 T new_dir[3], T f_val[3], T& pdf, bool& is_specular) const {
			return scene.BSDFSampleF(id, wo, n, u1, u2, new_dir, f_val, pdf, is_specular);
		}
		T BSDFPdf(int id, const T wo[3], const T wi[3], const T n[3]) const {
			return scene.BSDFPdf(id, wo, wi, n);
		}
		// Route all random decisions through the current MLTSampler stream
		T RandFloat() const { return sampler.Get1D(); }
	} adapter{scene, sampler};

	// Camera subpath -- continues in camera stream (no StartStream reset here,
	// matching pbrt-v4 which uses the same stream throughout L())
	int nCamera = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d,
											 t, adapter, cameraVerts.data());
	if (nCamera != t) return result;

	// Light subpath -- pbrt-v4: sampler.StartStream(lightStreamIndex)
	sampler.StartStream(MLTSampler<T>::kLightStream);
	int nLight = BDPTGenerateLightSubpath(s, adapter, lightVerts.data());
	if (nLight != s) return result;

	// Connect -- pbrt-v4: sampler.StartStream(connectionStreamIndex)
	sampler.StartStream(MLTSampler<T>::kConnectionStream);
	T L[3];
	BDPTConnect(lightVerts.data(), cameraVerts.data(), s, t, adapter, L);

	// Scale by nStrategies (pbrt-v4: return ConnectBDPT(...) * nStrategies)
	result.L[0] = L[0] * (T)nStrategies;
	result.L[1] = L[1] * (T)nStrategies;
	result.L[2] = L[2] * (T)nStrategies;
	result.valid = true;
	return result;
}

// ===========================================================================
// Section 5 -- MLTBootstrap: compute normalization constant b
// ===========================================================================
// Mirrors pbrt-v4 bootstrap phase in MLTIntegrator::Render.
// Fills bootstrapWeights[i * (maxDepth+1) + depth] = c(L_i).
// Returns b = (maxDepth+1) / nBootstrapSamples * sum(weights).

template<typename T, typename Scene>
T MLTBootstrap(int nBootstrap, int maxDepth, T sigma, T largeStepProb,
			   const Scene& scene,
			   std::vector<T>& bootstrapWeights) {
	int nBootstrapSamples = nBootstrap * (maxDepth + 1);
	bootstrapWeights.assign(nBootstrapSamples, T(0));

	for (int i = 0; i < nBootstrap; ++i) {
		for (int depth = 0; depth <= maxDepth; ++depth) {
			int rngIndex = i * (maxDepth + 1) + depth;
			MLTSampler<T> sampler(1, (uint64_t)rngIndex, sigma, largeStepProb);
			MLTPathResult<T> r = MLTEvalPath(sampler, depth, scene, maxDepth);
			bootstrapWeights[rngIndex] = (T)scene.Luminance(r.L[0], r.L[1], r.L[2]);
		}
	}

	T sum = T(0);
	for (T w : bootstrapWeights) sum += w;
	// b = (maxDepth+1) / nBootstrapSamples * sum
	return (T)(maxDepth + 1) / (T)nBootstrapSamples * sum;
}

// ===========================================================================
// Section 6 -- MLTRenderLoop
// ===========================================================================
// Mirrors pbrt-v4 MLTIntegrator::Render chain loop (single-threaded).
//
// Parameters:
//   nBootstrap        -- number of bootstrap samples per depth
//   nMutations        -- total Metropolis mutations to perform
//   maxDepth          -- maximum BDPT path depth
//   sigma             -- small-step Gaussian std dev (pbrt default: 0.01)
//   largeStepProb     -- probability of a large step (pbrt default: 0.3)
//   scene             -- scene query object
//   splatCallback     -- called for each (px, py, weight*L) contribution
//                        signature: void(T px, T py, T Lr, T Lg, T Lb)
//
// Normalization: the callback receives values pre-scaled by b/nMutations so
// the caller can simply accumulate them into a floating-point framebuffer.

template<typename T, typename Scene, typename SplatFn>
void MLTRenderLoop(int nBootstrap, int64_t nMutations, int maxDepth,
				   T sigma, T largeStepProb,
				   const Scene& scene,
				   SplatFn splatCallback) {
	// Bootstrap phase
	std::vector<T> bootstrapWeights;
	T b = MLTBootstrap(nBootstrap, maxDepth, sigma, largeStepProb,
					   scene, bootstrapWeights);

	if (b == T(0)) return; // all-black scene

	// Build alias table over bootstrap weights
	// reservoir_sampler.h AliasTable::sample(double u) -> int index
	// We use the local AliasTable directly.
	AliasTable aliasTable(std::vector<double>(bootstrapWeights.begin(),
											  bootstrapWeights.end()));

	T invNorm = b / (T)nMutations; // per-splat scale factor

	// Single Markov chain (callers can run multiple chains in parallel)
	RNG rng(12345u);
	int bootstrapIndex = aliasTable.sample(rng.Uniform<double>());
	int depth = bootstrapIndex % (maxDepth + 1);

	MLTSampler<T> sampler(1, (uint64_t)bootstrapIndex, sigma, largeStepProb);
	MLTPathResult<T> current = MLTEvalPath(sampler, depth, scene, maxDepth);
	T cCurrent = (T)scene.Luminance(current.L[0], current.L[1], current.L[2]);

	for (int64_t j = 0; j < nMutations; ++j) {
		sampler.StartIteration();

		MLTPathResult<T> proposed = MLTEvalPath(sampler, depth, scene, maxDepth);
		T cProposed = (T)scene.Luminance(proposed.L[0], proposed.L[1], proposed.L[2]);

		// Acceptance probability (pbrt-v4: accept = min(1, cProposed/cCurrent))
		T accept = (cCurrent > T(0)) ? std::min(T(1), cProposed / std::max(cCurrent, std::numeric_limits<T>::min())) : T(1);

		// Splat both current and proposed (MH unbiased estimator)
		if (accept > T(0) && proposed.valid) {
			splatCallback(proposed.px, proposed.py,
						  proposed.L[0] * accept / std::max(cProposed, T(1e-30)) * invNorm,
						  proposed.L[1] * accept / std::max(cProposed, T(1e-30)) * invNorm,
						  proposed.L[2] * accept / std::max(cProposed, T(1e-30)) * invNorm);
		}
		if (current.valid && cCurrent > T(0)) {
			splatCallback(current.px, current.py,
						  current.L[0] * (T(1) - accept) / cCurrent * invNorm,
						  current.L[1] * (T(1) - accept) / cCurrent * invNorm,
						  current.L[2] * (T(1) - accept) / cCurrent * invNorm);
		}

		// Accept or reject
		if ((T)rng.Uniform<float>() < accept) {
			current = proposed;
			cCurrent = cProposed;
			sampler.Accept();
		} else {
			sampler.Reject();
		}
	}
}
