#pragma once
//==============================================================================================
// bdpt_adapter_entrypoints.h -- render-loop drivers for BDPTSceneAdapter
// (bdpt_adapter.h, which #includes this file at its own end - never include
// this file directly, BDPTSceneAdapter must already be defined). One entry
// point per integrator that uses BDPTSceneAdapter as its Scene: BDPT, MLT,
// RandomWalk, AO, SimplePath, SimpleVolPath, LightPath.
//
// RandomWalk/AO/SimplePath/SimpleVolPath share one row-parallel "one sample
// per pixel per shutter sample, average" shape (parallel_render_tile_loop()
// below) - a code-review pass on the commit that first added the four found
// them hand-duplicating that ~35-line thread-pool/crop-rect/average
// skeleton verbatim, factored out here. BDPT/MLT/LightPath each have a
// genuinely different shape (BDPT and LightPath both splat into arbitrary
// pixels via a shared SplatFilm; MLT is a Markov-chain mutation loop, not a
// per-pixel one) and keep their own drivers - see each one's own comment.
//==============================================================================================

// ---------------------------------------------------------------------------
// SplatFilm -- LightPathTrace's Film concept (a single Splat(px,py,L) method),
// also used by bdpt_render_with_adapter() below for BDPT's own t==1 "light
// tracing" strategy contributions. Splats land at essentially arbitrary
// pixels (a light path can connect to the camera from anywhere it happens
// to walk to), unlike every other driver's own-pixel-only accumulation, so
// this needs real cross-thread synchronization -- one std::mutex per pixel,
// the same granularity sppm_adapter.h's own photon pass already uses for
// its per-pixel splat accumulation (see its pixel_mutexes vector) --
// coarser (one mutex for the whole image) would serialize every worker
// thread through a single lock; finer isn't meaningful (a pixel is the
// atomic unit of accumulation here).
// ---------------------------------------------------------------------------
class SplatFilm {
  public:
	// cropX0/X1/Y0/Y1: pbrt-v4 Film "cropwindow"/"pixelbounds", already
	// resolved to concrete pixel bounds by the caller (camera::initialize()'s
	// crop_x0/x1/y0/y1) - see bdpt_render_with_adapter()'s/
	// lightpath_render_with_adapter()'s own comment on why gating splats
	// (rather than skipping trace generation, which BDPT's t==1/LightPath
	// have no per-pixel loop to skip) is this class's own share of honoring
	// a crop request. Defaulted to "-1 = unset" (full frame), matching
	// camera.h's own sentinel convention, so every pre-existing call site
	// (tests, any caller with no crop request) compiles and behaves
	// unchanged.
	// buf_/mutexes_ are sized to the CROP rect, not the full width*height -
	// a code-review pass found them still full-frame-sized despite the crop
	// bounds being available right here, wasting the "hundreds of MB at 4K+"
	// this class's own Splat()/ToRGB() comments already warn about even for
	// a crop covering a small fraction of the frame (exactly the case a
	// crop rect is normally used for - fast iteration on a small region).
	SplatFilm(int width, int height,
	          int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1)
		: width_(width), height_(height),
		  cropX0_(cropX0), cropX1_(cropX1 < 0 ? width : cropX1),
		  cropY0_(cropY0), cropY1_(cropY1 < 0 ? height : cropY1),
		  buf_(static_cast<size_t>(cropX1_ - cropX0_) * (cropY1_ - cropY0_) * 3, 0.0),
		  mutexes_(static_cast<size_t>(cropX1_ - cropX0_) * (cropY1_ - cropY0_)) {}

	void Splat(double px, double py, const double L[3]) {
		int ix = static_cast<int>(px);
		int iy = static_cast<int>(py);
		if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) return;
		if (!in_crop_rect(ix, iy, cropX0_, cropX1_, cropY0_, cropY1_)) return;
		// Crop-local index - buf_/mutexes_ are sized to the crop rect only,
		// so a full-frame (ix,iy) must be rebased against (cropX0_,cropY0_)
		// before indexing into them.
		size_t pidx = static_cast<size_t>(iy - cropY0_) * (cropX1_ - cropX0_) + (ix - cropX0_);
		std::lock_guard<std::mutex> lg(mutexes_[pidx]);
		for (int c = 0; c < 3; ++c) {
			double v = L[c];
			if (!std::isfinite(v) || v < 0.0) v = 0.0;
			buf_[pidx * 3 + c] += v;
		}
	}

	// Adds this film's splats (converted to RGB via /norm) into `out_rgb`, a
	// FULL-FRAME width*height*3 buffer already sized/zero-initialized by the
	// caller - NOT a same-shape return the way this used to work before
	// buf_ became crop-sized (see this class's own constructor comment);
	// each crop-local bucket is rebased back to its full-frame pixel offset
	// on write, since the caller's own out_rgb still spans the whole image
	// (a cropped-out pixel there just never receives a += from here).
	//
	// norm: total samples PER PIXEL across the whole image (spp) -- NOT
	// divided by pixel count again, since each splat already lands at a
	// specific pixel; this mirrors pbrt-v4's own LightPathIntegrator film
	// reconstruction (splat weight accumulates raw, normalized once by the
	// image's total sample count at the end).
	void AddToRGB(std::vector<double>& out_rgb, double norm) const {
		const int cropWidth = cropX1_ - cropX0_;
		const int cropHeight = cropY1_ - cropY0_;
		for (int ly = 0; ly < cropHeight; ++ly) {
			for (int lx = 0; lx < cropWidth; ++lx) {
				const size_t localIdx = (static_cast<size_t>(ly) * cropWidth + lx) * 3;
				const size_t fullIdx = (static_cast<size_t>(cropY0_ + ly) * width_ + (cropX0_ + lx)) * 3;
				for (int c = 0; c < 3; ++c) out_rgb[fullIdx + c] += buf_[localIdx + c] / norm;
			}
		}
	}

  private:
	int width_, height_;
	int cropX0_, cropX1_, cropY0_, cropY1_;
	std::vector<double> buf_;
	std::vector<std::mutex> mutexes_;
};

// ===========================================================================
// bdpt_render_with_adapter -- row-parallel BDPT render loop
// ===========================================================================
// bdpt.h has no driver of its own (see this integration's task description:
// "There is NO existing multithreaded pixel-loop driver for it") -- this is
// that driver, mirroring sppm_camera_pass_with_sky()'s row-steal threading
// pattern (atomic row index, not per-pixel locks) as closely as BDPT's
// per-sample (not per-iteration) shape allows: for each pixel, for each
// sample, generate a camera ray via the adapter's PixelToRay(), call
// BDPTLi(), accumulate, average.
//
// cameraVerts/lightVerts are allocated ONCE per worker thread (sized to
// BDPTLi's own documented minimum: (maxDepth+2) and (maxDepth+1)
// respectively) and reused across every pixel/sample that thread handles,
// avoiding a heap allocation per sample.
// t==1 contributions land at a DIFFERENT pixel than whichever camera ray
// produced the current sample (see CameraSampleWi()'s own comment for the
// normalized-vs-raster unit split), so they can't be added to `sum` below
// like every other strategy -- they're splatted into a SplatFilm instead
// (same mechanism lightpath_render_with_adapter() below already uses for
// LightPath's own pure light-tracing splats) and merged additively into
// out_rgb once every worker thread has joined.
inline void bdpt_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height,
                                      int spp, int maxDepth, std::vector<double>& out_rgb,
                                      int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	if (cropX1 < 0) cropX1 = width;
	if (cropY1 < 0) cropY1 = height;

	// SplatFilm (one std::mutex + 3 doubles per pixel - hundreds of MB at
	// 4K+) is constructed lazily, on the first t==1 light-tracing
	// contribution that actually occurs, instead of unconditionally at the
	// top of every --bdpt render: many renders (e.g. --bdpt-max-depth 0,
	// where t==1 can structurally never fire, or any scene where no
	// light-subpath vertex ever has an unoccluded camera connection) would
	// otherwise pay this allocation for zero benefit. std::call_once
	// guards the one-time construction race across worker threads.
	std::optional<SplatFilm> film;
	std::once_flag film_once;
	auto ensure_film = [&]() -> SplatFilm& {
		std::call_once(film_once, [&]() { film.emplace(width, height, cropX0, cropX1, cropY0, cropY1); });
		return *film;
	};

	unsigned int nthreads = determine_render_thread_count();
	std::atomic<int> next_row(0);

	auto worker = [&]() {
		std::vector<BDPTVertex<double>> cameraVerts(static_cast<size_t>(maxDepth) + 2);
		std::vector<BDPTVertex<double>> lightVerts(static_cast<size_t>(maxDepth) + 1);
		auto splat = [&](double px01, double py01, double Lr, double Lg, double Lb) {
			double L[3] = { Lr, Lg, Lb };
			ensure_film().Splat(px01 * width, py01 * height, L);
		};

		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;

			for (int ix = 0; ix < width; ++ix) {
				// Film "cropwindow"/"pixelbounds" - unlike every OTHER
				// per-pixel driver in this file, BDPT's own t==1 light-
				// tracing splats (the `splat` lambda above) are generated as
				// a SIDE EFFECT of this same per-pixel/per-sample loop and
				// normalized by SplatFilm::ToRGB() against a FIXED `spp`
				// (see that function's own comment: the total sample count
				// the whole image's splat buffer is implicitly assumed to
				// have been fed by, independent of how many pixels actually
				// ran). Skipping this loop body entirely for an out-of-crop
				// pixel would shrink that total sample count in proportion
				// to the crop's own area while `spp` stays fixed, silently
				// DARKENING every t==1 contribution inside the crop rather
				// than merely rendering a smaller region - a real
				// correctness bug, not just a missed optimization (found by
				// code review). MLT's/LightPath's own splat mechanisms
				// don't have this problem because they already preserve
				// their full sample/mutation budget unconditionally and
				// gate only the splat DESTINATION (see their own crop-check
				// comments) - so BDPT does the same here: every pixel still
				// runs its full `spp` budget (feeding `splat` at the same
				// rate as an uncropped render), and only the FINAL
				// out_rgb write below is skipped for an out-of-crop pixel
				// (SplatFilm's own crop gate already keeps t==1 energy from
				// ever landing outside the crop rect either).
				const bool inCrop = in_crop_rect(ix, iy, cropX0, cropX1, cropY0, cropY1);
				double sum[3] = { 0.0, 0.0, 0.0 };
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;

					double cam_p[3], cam_n[3], ray_d[3];
					if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) continue;

					double L[3];
					BDPTLi<double>(cam_p, cam_n, ray_d, maxDepth, scene,
					                cameraVerts.data(), lightVerts.data(), L, splat);

					if (!inCrop) continue;   // still ran BDPTLi (feeds `splat`), just don't keep its own t>=2 estimate
					for (int c = 0; c < 3; ++c) {
						double v = L[c];
						if (!std::isfinite(v) || v < 0.0) v = 0.0;   // firefly/NaN guard, matches cpu_interface's path tracer
						sum[c] += v;
					}
				}
				if (!inCrop) continue;
				int idx = (iy * width + ix) * 3;
				out_rgb[idx + 0] = sum[0] / spp;
				out_rgb[idx + 1] = sum[1] / spp;
				out_rgb[idx + 2] = sum[2] / spp;
			}
		}
	};

	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker);
	for (auto& th : threads) th.join();

	if (film) {
		film->AddToRGB(out_rgb, static_cast<double>(spp));
	}
}

// Replicates MLTRenderLoop()'s own post-bootstrap Markov-chain loop
// (mlt.h) verbatim, EXCEPT that `depth` and `b` (this chain's normalization
// constant) are supplied by the caller instead of being drawn from a
// shared, power-weighted alias table over every (bootstrap sample, depth)
// pair. See mlt_render_with_adapter()'s own comment for why that mattered
// enough to duplicate this ~20-line loop body rather than reuse
// MLTRenderLoop() directly -- everything below this comment is otherwise a
// line-for-line mirror of mlt.h's own mutation loop, kept that way
// deliberately so a future mlt.h change is easy to notice and re-apply here.
//
// nMutationsRun vs nMutationsNormalize: when several independent chains
// share ONE depth (chains_per_depth[d] > 1 in mlt_render_with_adapter()),
// each chain only RUNS its own fraction of that depth's total mutation
// budget (nMutationsRun), but must normalize its splats as though it were
// responsible for the FULL depth budget (nMutationsNormalize = that
// depth's total across all its chains combined). Using nMutationsRun for
// BOTH (the naive choice) would make each chain's own splat sum
// independently reconstruct the full b_depth[d] on its own (the same "sum
// of splats over a full chain == b" identity MLTRenderLoop's own invNorm
// relies on) -- summing chains_per_depth[d] of those together would then
// over-count that depth's contribution by exactly chains_per_depth[d].
// Normalizing by the shared depth-level total instead makes each chain
// contribute only its proportional share, so summing them reconstructs
// b_depth[d] exactly once, matching mlt_render_with_adapter()'s own
// "sum across chains and depths" combination step.
template<typename Scene, typename SplatFn>
inline void mlt_run_depth_chain(int depth, double b, int64_t nMutationsRun, int64_t nMutationsNormalize,
                                 int maxDepth, double sigma, double largeStepProb,
                                 const Scene& scene, SplatFn splatCallback, uint64_t chainSeed) {
	if (nMutationsRun <= 0 || nMutationsNormalize <= 0 || b <= 0.0) return;
	double invNorm = b / (double)nMutationsNormalize;

	// mlt.h's chainSeed doc comment explains the splitmix64-style combine
	// below (bootstrapIndex there plays the same "must not let two chains
	// collide" role chainSeed alone plays here, since depth is no longer
	// drawn from anything random).
	uint64_t samplerSeed = (uint64_t)(depth + 1) * 0x9E3779B97F4A7C15ull + chainSeed;
	MLTSampler<double> sampler(1, samplerSeed, sigma, largeStepProb);
	MLTPathResult<double> current = MLTEvalPath(sampler, depth, scene, maxDepth);
	double cCurrent = scene.Luminance(current.L[0], current.L[1], current.L[2]);

	RNG rng(chainSeed);
	for (int64_t j = 0; j < nMutationsRun; ++j) {
		sampler.StartIteration();

		MLTPathResult<double> proposed = MLTEvalPath(sampler, depth, scene, maxDepth);
		double cProposed = scene.Luminance(proposed.L[0], proposed.L[1], proposed.L[2]);

		double accept = (cCurrent > 0.0)
			? std::min(1.0, cProposed / std::max(cCurrent, std::numeric_limits<double>::min()))
			: 1.0;

		if (accept > 0.0 && proposed.valid) {
			splatCallback(proposed.px, proposed.py,
			              proposed.L[0] * accept / std::max(cProposed, 1e-30) * invNorm,
			              proposed.L[1] * accept / std::max(cProposed, 1e-30) * invNorm,
			              proposed.L[2] * accept / std::max(cProposed, 1e-30) * invNorm);
		}
		if (current.valid && cCurrent > 0.0) {
			splatCallback(current.px, current.py,
			              current.L[0] * (1.0 - accept) / cCurrent * invNorm,
			              current.L[1] * (1.0 - accept) / cCurrent * invNorm,
			              current.L[2] * (1.0 - accept) / cCurrent * invNorm);
		}

		if (rng.Uniform<float>() < accept) {
			current = proposed;
			cCurrent = cProposed;
			sampler.Accept();
		} else {
			sampler.Reject();
		}
	}
}

// ===========================================================================
// mlt_render_with_adapter -- multi-chain, depth-stratified MLT render loop
// ===========================================================================
// See this file's own header comment for why "just call MLTRenderLoop() N
// times on N threads" doesn't actually parallelize anything without
// mlt.h's chainSeed parameter -- but that alone turned out NOT to be
// enough for a usable image, which is the real reason this function looks
// nothing like a thin loop around MLTRenderLoop() anymore:
//
// MLTRenderLoop() (and MLTBootstrap() underneath it) builds ONE alias table
// over every (bootstrap sample, depth) pair combined, and each independent
// chain draws its OWN fixed depth from that single shared table, weighted
// by luminance. On scene A1 this measurably starves every depth except 0:
// a depth-0 path (camera ray hits the light directly, no bounces) has huge
// per-sample luminance (raw, unattenuated light emission) but is only ever
// nonzero for the handful of pixels where the light is directly visible --
// diagnostic bootstrap run on A1 (nBootstrap=2000, maxDepth=5) measured
// depth 0's max bootstrap weight at ~15 against ~0.2-0.5 for every other
// depth, with only 14/2000 depth-0 samples nonzero at all. A shared alias
// table dominated by those few huge outliers hands nearly every chain a
// depth-0 seed, so nearly every chain then spends its ENTIRE mutation
// budget exploring the tiny screen region where the light is directly
// visible -- every other depth (all of the actual indirect/diffuse
// transport that makes up most of a Cornell box image) gets starved,
// producing a near-black image with a handful of bright pixels. Verified
// directly: an early version of this driver (a thin MLTRenderLoop()
// wrapper, no stratification) rendered A1 at mean brightness 0.02/255 with
// 99.2% of pixels exactly zero, on a render whose (working) BDPT
// counterpart on the same scene/sample budget averaged 47/255.
//
// The fix -- explicit stratification by depth, a standard MLT technique --
// runs MLTBootstrap() exactly ONCE (shared across every chain, not
// per-chain like a naive parallel-MLTRenderLoop driver would), derives
// each depth's own average bootstrap weight b_depth[d] directly from that
// single bootstrap pass (mathematically exact: mlt.h's own combined
// b = (maxDepth+1)/nBootstrapSamples * sum(all weights) is provably equal
// to sum_d(b_depth[d]) -- both reduce to (1/nBootstrap) * sum over all
// (i,d) of weight[i,d] -- so summing every depth's own independently
// normalized chain output reproduces the SAME total image energy the
// combined-table approach targets, just with guaranteed per-depth
// coverage instead of luck), then guarantees at least one dedicated chain
// per depth (more if nthreads > maxDepth+1, split round-robin), each chain
// running mlt_run_depth_chain() above with that depth's own b_depth[d] and
// an even share of the mutation budget.
//
// Per-thread buffers are SUMMED (not averaged) at the end: each chain's
// own splats are already normalized (via b_depth[d]/mutations_for_that_chain)
// to represent its stratum's own share of the image's total energy, so the
// full image is the sum of all strata, not their average -- averaging
// here (this file's earlier draft's mistake) would divide the image's
// brightness by nChains for no reason.
inline void mlt_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height,
                                     int nBootstrap, int64_t nMutations, int maxDepth,
                                     double sigma, double largeStepProb,
                                     std::vector<double>& out_rgb,
                                     int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	if (cropX1 < 0) cropX1 = width;
	if (cropY1 < 0) cropY1 = height;
	unsigned int nthreads = determine_render_thread_count();
	if (nthreads < 1) nthreads = 1;

	// Shared bootstrap phase -- computed once, not per-chain.
	std::vector<double> bootstrapWeights;
	double b_total = MLTBootstrap<double>(nBootstrap, maxDepth, sigma, largeStepProb, scene, bootstrapWeights);
	if (b_total <= 0.0) return;   // all-black scene, matches MLTRenderLoop()'s own early-out

	int nDepths = maxDepth + 1;
	std::vector<double> b_depth(nDepths, 0.0);
	for (int d = 0; d < nDepths; ++d) {
		double sum = 0.0;
		for (int i = 0; i < nBootstrap; ++i) sum += bootstrapWeights[(size_t)i * nDepths + d];
		b_depth[d] = sum / nBootstrap;
	}

	// At least one chain per depth, more if there are threads to spare.
	int nChains = (int)nthreads;
	if (nChains < nDepths) nChains = nDepths;

	std::vector<int> chain_depth(nChains);
	std::vector<int> chains_per_depth(nDepths, 0);
	for (int c = 0; c < nChains; ++c) {
		int d = c % nDepths;
		chain_depth[c] = d;
		++chains_per_depth[d];
	}
	// Floor to 1: nMutations/nDepths truncates to 0 whenever nMutations <
	// nDepths (e.g. --mlt-mutations 5 --mlt-max-depth 10), which previously
	// made every chain below skip its `mutations_for_chain <= 0` check
	// entirely -- a genuinely empty run, not caught by any validation, that
	// silently wrote out a fully black image with cpu_render_main_mlt still
	// returning SUCCESS. A caller asking for fewer mutations than depths is
	// asking for a noisier-than-useful render, not for one that renders
	// nothing -- so this guarantees at least SOME real sampling happens per
	// depth instead of a silent no-op.
	int64_t mutations_per_depth = nMutations / nDepths;
	if (mutations_per_depth < 1) mutations_per_depth = 1;

	std::vector<std::vector<double>> thread_buffers(
		nthreads, std::vector<double>(static_cast<size_t>(width) * height * 3, 0.0));
	std::atomic<int> next_chain(0);

	auto worker = [&](unsigned int tid) {
		std::vector<double>& buf = thread_buffers[tid];
		auto splat = [&](double px, double py, double Lr, double Lg, double Lb) {
			if (!std::isfinite(Lr) || !std::isfinite(Lg) || !std::isfinite(Lb)) return;
			int ix = static_cast<int>(px * width);
			int iy = static_cast<int>(py * height);
			if (ix < 0) ix = 0; if (ix >= width)  ix = width - 1;
			if (iy < 0) iy = 0; if (iy >= height) iy = height - 1;
			// Film "cropwindow"/"pixelbounds" - MLT has no per-pixel loop to
			// skip (the Markov chain visits essentially arbitrary (px,py)
			// via its own mutation strategy, not a pixel iteration), so
			// gating the splat itself is this integrator's only handle on a
			// crop request. Each accepted splat still carries its own
			// correct invNorm weight regardless of how many OTHER splats
			// were dropped here, so no renormalization is needed - see
			// mlt_run_depth_chain()'s own comment for that weight's
			// derivation.
			if (!in_crop_rect(ix, iy, cropX0, cropX1, cropY0, cropY1)) return;
			int idx = (iy * width + ix) * 3;
			buf[idx + 0] += Lr; buf[idx + 1] += Lg; buf[idx + 2] += Lb;
		};
		while (true) {
			int c = next_chain.fetch_add(1);
			if (c >= nChains) break;
			int depth = chain_depth[c];
			double b_d = b_depth[depth];
			if (b_d <= 0.0) continue;   // this depth's stratum is genuinely empty (e.g. maxDepth deeper than any light-bounce path reaches)
			// Floor to 1: mutations_per_depth/chains_per_depth[depth] can
			// still truncate to 0 even after mutations_per_depth's own floor
			// above, whenever a depth has more chains assigned to it than its
			// mutation budget (e.g. a modest --mlt-mutations on a
			// many-thread machine, where nChains == nthreads spreads several
			// chains per depth) -- every chain sharing that depth would
			// independently compute the same 0 and skip, so that whole
			// depth's stratum would silently contribute nothing to the
			// image, same failure mode as the mutations_per_depth==0 case
			// above just one level down.
			int64_t mutations_for_chain = mutations_per_depth / chains_per_depth[depth];
			if (mutations_for_chain < 1) mutations_for_chain = 1;

			// splitmix64-style per-chain seed -- see mlt.h's chainSeed doc
			// comment (same reasoning: two chains must never collide, and 0
			// is avoided as a degenerate PCG32 seed elsewhere in rng.h).
			uint64_t seed = 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(c) + 1) + 1u;
			// nMutationsNormalize must equal the REAL total mutations this
			// depth's chains will actually run combined, not the theoretical
			// mutations_per_depth target -- when the mutations_for_chain
			// floor above kicks in, mutations_for_chain*chains_per_depth[depth]
			// can exceed mutations_per_depth, and every chain sharing this
			// depth independently computes this same product (same b_d,
			// same chains_per_depth[depth]), so they all agree on one
			// consistent normalizer without needing shared/atomic state.
			// See mlt_run_depth_chain()'s own doc comment for why this must
			// be the WHOLE depth's total, not mutations_for_chain alone (that
			// would over-count this depth's contribution by
			// chains_per_depth[depth]).
			const int64_t mutations_normalize =
				mutations_for_chain * static_cast<int64_t>(chains_per_depth[depth]);
			mlt_run_depth_chain(depth, b_d, mutations_for_chain, mutations_normalize, maxDepth, sigma, largeStepProb,
			                     scene, splat, seed);
		}
	};

	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
	for (auto& th : threads) th.join();

	// Density reconstruction: (px,py) are normalized [0,1)^2 coordinates
	// (BDPTSceneAdapter::PixelToRay()'s own convention -- see its doc
	// comment), so b's implicit domain measure is the UNIT SQUARE, not
	// width*height discrete pixels. A splat's accumulated luminance at one
	// pixel estimates that pixel's own share of a probability DENSITY over
	// the continuous [0,1)^2 domain, not a value already scaled for
	// display -- reconstructing the actual per-pixel radiance needs
	// multiplying by the number of pixels (each pixel occupies
	// 1/(width*height) of that unit square, and the density->image
	// conversion divides by that pixel's own area). This is the same role
	// pbrt-v4's own `1/mutationsPerPixel` final-image scale plays in its
	// MLTIntegrator::Render() (mutationsPerPixel = totalMutations/nPixels,
	// so multiplying by 1/mutationsPerPixel is multiplying by
	// nPixels/totalMutations -- the nPixels factor is the same
	// density-reconstruction step, and the /totalMutations half is already
	// folded into invNorm=b/nMutations above). Confirmed empirically: an
	// earlier draft of this function without this factor rendered scene A1
	// at mean brightness ~0.02/255 (nearly pure black); with it, brightness
	// lands in the same ballpark as a BDPT render of the same scene/budget
	// (see this integration's own verification numbers).
	double density_scale = static_cast<double>(width) * static_cast<double>(height);
	for (size_t i = 0; i < out_rgb.size(); ++i) {
		double sum = 0.0;
		for (unsigned int t = 0; t < nthreads; ++t) sum += thread_buffers[t][i];
		out_rgb[i] = sum * density_scale;   // SUM across strata (not average), then density-reconstruct -- see this function's own comment
	}
}

// Writes a flat RGB double buffer to a P3 PPM file, applying the same tone
// mapping / sRGB encoding as sppm_adapter.h's sppm_write_ppm() (color.h's
// write_color()) so BDPT/MLT output looks consistent with every other
// render this codebase produces. Duplicated rather than reused from
// sppm_adapter.h -- see this file's own header comment on why this adapter
// avoids depending on that header at all (the AliasTable collision).
inline void bdpt_write_ppm(const std::string& path, int width, int height,
                            const std::vector<double>& rgb) {
	std::ofstream out(path);
	out << "P3\n" << width << ' ' << height << "\n255\n";
	for (int i = 0; i < width * height; ++i) {
		color c(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2]);
		write_color(out, c);
	}
}

// EXR counterpart to bdpt_write_ppm() -- see sppm_adapter.h's sppm_write_exr()
// for why this exists (--bdpt/--mlt --output *.exr must not silently fall
// through to bdpt_write_ppm() and produce a PPM mislabeled as EXR).
inline bool bdpt_write_exr(const std::string& path, int width, int height,
                            const std::vector<double>& rgb, std::string& error) {
	std::vector<float> pixels(rgb.size());
	for (size_t i = 0; i < rgb.size(); ++i) pixels[i] = static_cast<float>(rgb[i]);
	return write_exr_image(path, pixels.data(), width, height, error);
}

// ===========================================================================
// Round 6 Phase 2 -- render-loop drivers for RandomWalk/AO/SimplePath/
// SimpleVolPath/LightPath, reusing BDPTSceneAdapter (extended above) as
// their Scene. RandomWalk/AO/SimplePath/SimpleVolPath mirror
// bdpt_render_with_adapter()'s own row-parallel per-pixel/per-sample loop
// exactly (generate a camera ray via PixelToRay(), call the integrator,
// average); LightPath is shaped completely differently (it SPLATS into
// arbitrary pixels rather than returning one pixel's own radiance), so it
// gets its own driver and its own tiny Film type below.
// ===========================================================================

// Shared row-parallel "one L(px,py) sample per shutter sample, average per
// pixel" render loop. RandomWalk/AO/SimplePath/SimpleVolPath below all
// reduced to exactly this shape, differing only in the per-sample
// integrator call and its own extra parameters - a code-review pass found
// them hand-duplicating this same ~35-line thread-pool/crop-rect/average
// skeleton four times. LightPath is NOT one of these: it splats into
// arbitrary pixels rather than returning one pixel's own radiance (see its
// own doc comment below), a genuinely different shape this helper doesn't
// try to cover.
//
// perSampleLi(px, py, out_L) is called once per shutter sample per pixel
// with film-space [0,1) coordinates; it must return false to skip that
// sample (matching every caller's own pre-existing "PixelToRay missed the
// scene" skip - the sample still counts toward the `spp` divisor, exactly
// as it did before this helper existed, just contributes zero) or true
// with a finite, non-negative out_L (NaN/negative values are clamped to
// zero here regardless, same guard every caller already had inline).
template <typename PerSampleLi>
inline void parallel_render_tile_loop(int width, int height, int spp, std::vector<double>& out_rgb,
                                       int cropX0, int cropX1, int cropY0, int cropY1,
                                       PerSampleLi&& perSampleLi) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	if (cropX1 < 0) cropX1 = width;
	if (cropY1 < 0) cropY1 = height;
	unsigned int nthreads = determine_render_thread_count();
	std::atomic<int> next_row(0);
	auto worker = [&]() {
		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;
			if (iy < cropY0 || iy >= cropY1) continue;
			for (int ix = 0; ix < width; ++ix) {
				if (!in_crop_rect(ix, iy, cropX0, cropX1, cropY0, cropY1)) continue;
				double sum[3] = {0.0, 0.0, 0.0};
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;
					double L[3] = {0.0, 0.0, 0.0};
					if (!perSampleLi(px, py, L)) continue;
					for (int c = 0; c < 3; ++c) {
						double v = L[c];
						if (!std::isfinite(v) || v < 0.0) v = 0.0;
						sum[c] += v;
					}
				}
				int idx = (iy * width + ix) * 3;
				out_rgb[idx + 0] = sum[0] / spp;
				out_rgb[idx + 1] = sum[1] / spp;
				out_rgb[idx + 2] = sum[2] / spp;
			}
		}
	};
	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker);
	for (auto& th : threads) th.join();
}

inline void randomwalk_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height,
                                            int spp, int maxDepth, std::vector<double>& out_rgb,
                                            int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
	parallel_render_tile_loop(width, height, spp, out_rgb, cropX0, cropX1, cropY0, cropY1,
		[&](double px, double py, double L[3]) {
			double cam_p[3], cam_n[3], ray_d[3];
			if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) return false;
			RandomWalkLi<double>(cam_p, ray_d, scene, maxDepth, rand2d, L);
			return true;
		});
}

inline void ao_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                    double maxDist, bool cosSample, double illumScale, const double illumRgb[3],
                                    std::vector<double>& out_rgb,
                                    int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
	parallel_render_tile_loop(width, height, spp, out_rgb, cropX0, cropX1, cropY0, cropY1,
		[&](double px, double py, double L[3]) {
			double cam_p[3], cam_n[3], ray_d[3];
			if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) return false;
			AOLi<double>(cam_p, ray_d, scene, maxDist, cosSample, illumScale, illumRgb, rand2d, L);
			return true;
		});
}

inline void simplepath_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                            int maxDepth, bool sampleLights, bool sampleBsdf,
                                            std::vector<double>& out_rgb,
                                            int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
	auto rand1d = []() { return random_double(); };
	parallel_render_tile_loop(width, height, spp, out_rgb, cropX0, cropX1, cropY0, cropY1,
		[&](double px, double py, double L[3]) {
			double cam_p[3], cam_n[3], ray_d[3];
			if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) return false;
			SimplePathLi<double>(cam_p, ray_d, scene, maxDepth, sampleLights, sampleBsdf, rand2d, rand1d, L);
			return true;
		});
}

inline void simplevolpath_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                               int maxDepth, std::vector<double>& out_rgb,
                                               int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	parallel_render_tile_loop(width, height, spp, out_rgb, cropX0, cropX1, cropY0, cropY1,
		[&](double px, double py, double L[3]) {
			double cam_p[3], cam_n[3], ray_d[3];
			if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) return false;
			SimpleVolPathLi<double>(cam_p, ray_d, scene, maxDepth, L);
			return true;
		});
}

inline void lightpath_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                           int maxDepth, std::vector<double>& out_rgb,
                                           int cropX0 = 0, int cropX1 = -1, int cropY0 = 0, int cropY1 = -1) {
	// Film "cropwindow"/"pixelbounds" - LightPath has no per-pixel loop
	// either (see mlt_render_with_adapter()'s splat lambda's own comment for
	// the identical reasoning) - SplatFilm's own crop gate (see its
	// constructor's comment) is this integrator's only handle on a crop
	// request. total_paths below stays spp*width*height either way (not
	// shrunk to the crop's own pixel count) - simpler, and every path this
	// budget doesn't spend on a crop-rect pixel would otherwise have been
	// wasted on an out-of-crop one anyway.
	SplatFilm film(width, height, cropX0, cropX1, cropY0, cropY1);
	unsigned int nthreads = determine_render_thread_count();
	long long total_paths = static_cast<long long>(spp) * width * height;
	std::atomic<long long> next_path(0);

	auto worker = [&]() {
		auto rand1d = []() { return random_double(); };
		auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
		while (true) {
			long long idx = next_path.fetch_add(1);
			if (idx >= total_paths) break;
			LightPathTrace<double>(scene, film, maxDepth, rand1d, rand2d);
		}
	};
	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker);
	for (auto& th : threads) th.join();

	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	film.AddToRGB(out_rgb, static_cast<double>(spp));
}
