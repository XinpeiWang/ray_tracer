#pragma once
//==============================================================================================
// bdpt_adapter.h -- Bridges src/shared/bdpt.h's (and src/shared/mlt.h's,
// which shares the same Scene concept) duck-typed Scene concept to this
// renderer's real hittable/material/camera types, the same way
// sppm_adapter.h already does for src/shared/sppm.h.
//
// Relationship to sppm_adapter.h / SPPMSceneAdapter
// --------------------------------------------------
// This is a NEW, separate adapter class (BDPTSceneAdapter) rather than an
// extension of SPPMSceneAdapter, for two reasons discovered while working
// through what BDPT/MLT actually need beyond SPPM:
//
//   1. Scene concept size: bdpt.h/mlt.h's Scene concept is a strict
//      superset of sppm.h's (see bdpt.h's own file-header comment) --
//      SPPMSceneAdapter implements exactly the sppm.h subset (Intersect,
//      BSDFSampleF, BSDFf, DirectLight, SampleLightLe, PixelToRay,
//      RandFloat) and explicitly documents that the rest (SampleLight,
//      LightPMF, LightPDFLe, CameraPDFWe, CameraSampleWi,
//      SceneBoundingSphere, InfiniteLightLe, InfiniteLightDensity,
//      Unoccluded, BSDFPdf) exist only for parity, "not needed here". BDPT
//      genuinely needs essentially all of them (full bidirectional MIS
//      needs light-origin pdfs, reverse pdfs, and a real BSDF pdf query
//      that SPPM's single-strategy NEE never does), so bolting them onto
//      SPPMSceneAdapter would double that class's size for a concept it
//      was deliberately scoped to not need.
//
//   2. Per-hit context lifetime: SPPMSceneAdapter's shading-context storage
//      (durable_ctx_ / transient_ctx_) is built around SPPM's own access
//      pattern -- at most ONE "in flight" hit per thread at a time (a
//      camera-pass hit is either resampled immediately via BSDFSampleF, or
//      promoted to a single per-pixel durable slot consumed much later by
//      the photon pass). BDPT's RandomWalk builds an entire SUBPATH of up
//      to (maxDepth+2) vertices, ALL of whose bsdf_ids must stay valid
//      simultaneously through camera-subpath generation, light-subpath
//      generation, and every (s,t) connection/MIS-weight evaluation that
//      follows -- multiple hits genuinely alive at once, not one. That
//      needs a different storage shape (a small per-thread ring buffer,
//      see ctx_pool_ below), not SPPMSceneAdapter's single-slot design.
//
// What IS reused: bsdf_bridge.h's Layer 1 (SPPMShadingContext,
// sppm_is_delta_material, sppm_resolve_material, sppm_bsdf_f,
// sppm_bsdf_sample_f, sppm_bsdf_pdf) -- the actual material-system bridge,
// which is scene-model-specific, not SPPM-specific, and is used here
// completely unchanged. This adapter's Intersect()/BSDFf()/BSDFSampleF()/
// BSDFPdf() are thin wrappers around those same free functions, just
// against a differently-shaped context table. Light sampling (SampleLight/
// SampleLightLe/LightPMF/LightPDFLe) and camera-ray generation
// (CameraPDFWe/PixelToRay) mirror SPPMSceneAdapter's own constructor/
// SampleLightLe/PixelToRay logic closely, adapted to the (s,t)-connection
// math BDPT needs instead of SPPM's own DirectLight() NEE call.
//
// Scope (v2 -- area + point/spot/distant + sky, still mirrors SPPM's own
// incremental scoping precedent for what's left out)
// --------------------------------------------------------------
// - Area lights, point/spot/distant punctual lights, AND a real sky/
//   infinite light (cam.sky) all share ONE unified, power-weighted light
//   distribution (unifiedAlias_ below) that SampleLight/SampleLightLe/
//   LightPMF/LightPDFLe all draw from consistently -- required for BDPT's
//   MIS math to stay internally consistent (see this class's own
//   SampleLight()/SampleLightLe() comments for the combined-index-space
//   and per-kind-pdf derivation). Delta lights (point/spot/distant) are
//   marked is_delta on their BDPTLightSample/BDPTLightLeSample; the sky is
//   marked is_infinite instead, mirroring bdpt.h's own BDPTVertex::
//   IsDeltaLight()/IsInfiniteLight() split -- both correctly restrict
//   longer (s>1) BDPT connections through that vertex the same way
//   pbrt-v4's own Vertex::IsConnectible() does.
//   Still NOT wired in: goniometric and projection lights (both delta-
//   position, image-modulated) -- punctual_light_objects.h's own
//   goniometric_light_obj/projection_light_obj have no SampleLe (light-
//   subpath-emission) implementation yet, unlike point/spot (spot already
//   had one, see SpotLightData::sample_le) or distant/sky (straightforward
//   disk-sampling around the scene's bounding sphere, added here). A scene
//   whose ONLY light is goniometric or projection still renders black
//   under --bdpt/--mlt, same honestly-scoped limitation as before, just a
//   smaller residual set. cam.background is still honored as a flat
//   ambient backdrop for camera rays that escape a sky-less scene (see
//   InfiniteLightLe below).
// - CameraSampleWi() implements the t==1 "light tracing" strategy for
//   real: connects a light-subpath vertex directly to the camera (or, when
//   defocus_angle>0, to a sampled lens point) via the same importance math
//   SampleCameraConnection() (LightPath's own camera connection) already
//   used, shared through cameraConnectionCore().
//   bdpt_render_with_adapter() below owns a SplatFilm (mirroring
//   lightpath_render_with_adapter()'s own use of it) that BDPTLi()'s
//   splat callback writes t==1 contributions into; mlt_render_with_adapter()
//   reuses its own pre-existing splat lambda via MLTEvalPath() now passing
//   a real pRaster through to BDPTConnect(). Caustics and other paths only
//   reachable by directly connecting a light-subpath vertex to the camera
//   (never touched by any camera-subpath random walk) are no longer
//   dropped from BDPT/MLT output.
// - CameraPDFWe() assumes the default perspective camera model (vfov +
//   focus_dist, matching camera.h's own initialize()) -- alt camera models
//   (alt_ortho_cam/alt_spherical_cam/alt_realistic_cam) would get a wrong
//   forward-pdf and are unverified under BDPT/MLT, mirroring SPPM's own
//   "verified end-to-end on scene 11 only" scoping honesty.
//
// Two render-loop drivers below, mirroring sppm_adapter.h's
// sppm_render_with_adapter() as closely as each algorithm's own shape
// allows:
//   - bdpt_render_with_adapter(): straightforward row-parallel loop calling
//     bdpt.h's BDPTLi() once per sample (bdpt.h has no driver of its own at
//     all -- see this integration's task description).
//   - mlt_render_with_adapter(): mlt.h's own MLTRenderLoop() is a complete
//     single-chain driver (bootstrap + Metropolis mutation loop), but two
//     things beyond simple threading turned out to matter on a real scene:
//     (1) its entropy sources used to be fully hardcoded (see mlt.h's own
//     updated doc comment on the chainSeed parameter this integration
//     added) -- calling it N times unmodified would have run N
//     bit-identical redundant copies of the SAME chain, not N independent
//     ones; (2) even with distinct seeds, letting each independent chain
//     draw its own (bootstrap sample, depth) pair from ONE shared,
//     luminance-weighted alias table measurably starves every depth except
//     whichever has the single largest-magnitude outlier (on scene A1,
//     depth 0 -- direct camera-to-light hits, huge per-sample luminance but
//     tiny screen coverage), producing a near-black image. This driver
//     therefore runs MLTBootstrap() once (shared, not per-chain), derives
//     each depth's own normalization constant directly from that one pass,
//     and explicitly stratifies chains across depths (at least one
//     dedicated chain per depth) using mlt_run_depth_chain() -- a
//     line-for-line duplicate of MLTRenderLoop()'s own post-bootstrap loop
//     body, parametrized by an externally-supplied depth/b instead of an
//     internally-drawn one. See mlt_render_with_adapter()'s own doc comment
//     for the full derivation (why summing, not averaging, per-chain
//     buffers is the mathematically correct combination here) and the
//     measured before/after numbers.
//==============================================================================================

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "aabb.h"
#include "material.h"
#include "onb.h"
#include "quad.h"
#include "sphere.h"
#include "camera.h"
#include "color.h"
#include "shadow_ray.h"
#include "thread_count.h"        // determine_render_thread_count()
#include "bsdf_bridge.h"         // Layer 1 -- SPPMShadingContext + BSDF bridge (shared with sppm_adapter.h)
#include "../shared/bdpt.h"      // BDPTHit, BDPTVertex, BDPTLi, ...
#include "../shared/mlt.h"       // MLTRenderLoop (pulls in reservoir_sampler.h's AliasTable)
#include "../shared/exr_writer.h"
#include "../shared/utility_integrators.h"  // RandomWalkLi, AOLi
#include "../shared/simple_path.h"          // SimplePathLi
#include "../shared/simple_vol_path.h"      // SimpleVolPathLi, VolPathMediumProps
#include "../shared/light_path.h"           // LightPathTrace, LightEmissionSample, CameraConnection
#include "../shared/sampling_helpers.h"     // SampleUniformDiskConcentric -- distant/sky emission-position sampling

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>

// ===========================================================================
// BDPTSceneAdapter
// ===========================================================================
class BDPTSceneAdapter {
  public:
	// world: flat scene geometry (a scene_registry build_world() closure's
	//        result directly -- NOT BVH-wrapped, matching SPPMSceneAdapter's
	//        own precedent; hittable_list::hit() is a fine linear scan at
	//        Cornell-box scale).
	// cam:   camera used for image_width/image_height/vfov/focus_dist/
	//        lookfrom/lookat/get_ray()/background; caller must have already
	//        called cam.initialize().
	//
	// Unlike SPPMSceneAdapter's 3-arg constructor (world, nee_lights, cam),
	// there is no separate nee_lights parameter here: bdpt.h's Scene
	// concept has no notion of a caller-supplied NEE-biasing hittable_list
	// the way sppm.h's DirectLight() used one (SPPMSceneAdapter's
	// nee_lights_ could include non-emissive importance-sampling targets,
	// e.g. a glass sphere, alongside real lights) -- SampleLight/
	// SampleLightLe/LightPMF/LightPDFLe all have to agree on ONE light
	// distribution for BDPT's MIS math to be internally consistent, so this
	// adapter builds and owns that single distribution itself, scanning
	// `world` for real emitters exactly like SPPMSceneAdapter's own
	// constructor does.
	explicit BDPTSceneAdapter(const hittable_list& world, const camera& cam)
		: world_(world), cam_(cam)
	{
		// Camera basis/viewport constants (see this class's own field
		// comment) - derived once here instead of per-call inside
		// cameraConnectionCore(). Mirrors camera.h's own initialize()
		// formulas exactly (viewport_width/height from vfov/focus_dist/
		// aspect, the u/v/w basis from lookfrom/lookat/vup).
		camForward_ = unit_vector(cam_.lookat - cam_.lookfrom);
		camW_       = -camForward_;
		camRight_   = unit_vector(cross(cam_.vup, camW_));
		camUp_      = cross(camW_, camRight_);
		{
			double theta = degrees_to_radians(cam_.vfov);
			double h = std::tan(theta / 2.0);
			viewportHeight_ = 2.0 * h * cam_.focus_dist;
			viewportWidth_  = viewportHeight_ * (double(cam_.image_width) / double(cam_.image_height));
			camA_ = viewportWidth_ * viewportHeight_;
			camD_ = cam_.focus_dist;
		}
		if (cam_.defocus_angle > 0.0)
			defocusRadius_ = cam_.focus_dist * std::tan(degrees_to_radians(cam_.defocus_angle / 2.0));

		std::vector<double> weights;
		for (auto& obj : world_.objects) {
			AreaLightSample probe;
			if (!obj->sample_area(0.5, 0.5, probe)) continue;   // not an area-samplable shape
			shared_ptr<material> m = hittable_material(obj);
			auto dl = std::dynamic_pointer_cast<diffuse_light>(m);
			if (!dl) continue;   // real emitters only
			double area = (probe.pdf_pos > 0.0) ? 1.0 / probe.pdf_pos : 0.0;
			color e = dl->get_texture()->value(probe.u, probe.v, probe.p);
			double lum = 0.2126*e.x() + 0.7152*e.y() + 0.0722*e.z();
			double power = area * lum * pi;   // pbrt-v4 / power_light_sampler.h convention: phi = area*Le*pi
			if (dl->is_two_sided()) power *= 2.0;   // emits from both faces -- pbrt-v4 doubles phi to match
			emitters_.push_back(obj);
			emitter_dl_.push_back(dl);
			emitter_pdf_pos_.push_back(probe.pdf_pos);
			weights.push_back(std::max(power, 1e-9));
		}
		nEmitters_ = (int)emitters_.size();
		// Area-only table: unchanged from before, still exactly what
		// SampleLightEmission() (LightPathTrace's own emission sample -
		// still area-light-only, see its own comment) draws from.
		emitter_alias_ = AliasTable(weights.empty() ? std::vector<double>{1.0} : weights);

		// Scene bounding sphere, needed to place a well-defined (if
		// artificial) "position" for a delta-direction/infinite light's
		// NEE sample (see SampleLight()'s own comment) and as the emission
		// disk radius for distant/sky light-subpath sampling (see
		// SampleLightLe()). Computed once here rather than per-sample -
		// same box-based formula as SceneBoundingSphere() below, just
		// cached.
		{
			aabb box = world_.bounding_box();
			sceneCenter_[0] = (box.x.min + box.x.max) * 0.5;
			sceneCenter_[1] = (box.y.min + box.y.max) * 0.5;
			sceneCenter_[2] = (box.z.min + box.z.max) * 0.5;
			double dx = box.x.max - sceneCenter_[0], dy = box.y.max - sceneCenter_[1], dz = box.z.max - sceneCenter_[2];
			sceneRadius_ = std::sqrt(dx*dx + dy*dy + dz*dz);
			if (!(sceneRadius_ > 0.0)) sceneRadius_ = 1.0;
		}

		// Unified distribution: area lights (weights already collected
		// above) followed by point/spot/distant punctual lights, followed
		// by the sky (if present) - see SampleLight()'s own comment for
		// the resulting combined index space. Goniometric/projection
		// punctual lights are deliberately NOT included here (see class
		// Scope comment).
		punctBase_ = nEmitters_;
		int nPoint = 0, nSpot = 0, nDistant = 0;
		if (cam_.punct_lights) {
			nPoint    = (int)cam_.punct_lights->points.size();
			nSpot     = (int)cam_.punct_lights->spots.size();
			nDistant  = (int)cam_.punct_lights->distants.size();
			for (const auto& L : cam_.punct_lights->points)   weights.push_back(std::max(L.power(), 1e-9));
			for (const auto& L : cam_.punct_lights->spots)    weights.push_back(std::max(L.power(), 1e-9));
			for (const auto& L : cam_.punct_lights->distants) weights.push_back(std::max(L.power(), 1e-9));
		}
		spotBase_ = punctBase_ + nPoint;
		distBase_ = spotBase_ + nSpot;
		skyIdx_   = distBase_ + nDistant;

		// Sky/infinite light power: only affects how OFTEN the sky gets
		// chosen relative to other lights, not correctness - any positive
		// finite weight keeps the estimator unbiased, this just keeps it
		// well-behaved (a bright HDR sky competes fairly with a bright area
		// light instead of being drowned out or dominating by
		// construction). See sky_light::power_estimate()'s own comment for
		// the derivation - deterministic and reuses the sky's own
		// already-built importance distribution, no fresh sampling needed.
		hasSky_ = static_cast<bool>(cam_.sky);
		if (hasSky_) {
			weights.push_back(std::max(cam_.sky->power_estimate(), 1e-9));
			nTotal_ = skyIdx_ + 1;
		} else {
			nTotal_ = skyIdx_;
		}
		unifiedAlias_ = AliasTable(weights.empty() ? std::vector<double>{1.0} : weights);
	}

	// Total light count across every kind this adapter samples from (area +
	// point/spot/distant + sky - see class Scope comment) - zero means
	// SampleLight/SampleLightLe will unconditionally return false (see
	// their own `if (nTotal_ == 0) return false;` checks below), so BDPT/
	// MLT/SimplePath/LightPath will render a flat/black image with no
	// error. Exposed so callers (bdpt_render_core()/mlt_render_core()/etc.)
	// can warn about that up front instead of leaving it silent.
	int EmitterCount() const { return nTotal_; }

	// Area lights only - SampleLightEmission()/LightSurfaceLe() (LightPath's
	// own emission sample, still area-light-only, see that method's own
	// comment) draw from emitters_/emitter_alias_ exclusively, not the
	// unified distribution above, so a caller that only exercises THAT path
	// (lightpath_render_core) needs this narrower count instead of
	// EmitterCount() - a scene lit only by a point/spot/distant/sky light
	// still can't drive LightPathTrace, even though it now can drive
	// BDPT/MLT/SimplePath via EmitterCount()/SampleLight() above.
	int AreaEmitterCount() const { return nEmitters_; }

	// ------------------------------------------------------------------
	// Intersect / BSDFf / BSDFSampleF / BSDFPdf
	// ------------------------------------------------------------------
	//
	// bsdf_id space: [0, nEmitters_) is reserved for LIGHT identity (the
	// same numbering SampleLight/SampleLightLe hand out as light_id), and
	// every REAL hit's bsdf_id is nEmitters_ + (a slot in ctx_pool_ below)
	// -- see LightPMF/LightPDFLe's own comment for why these two numbering
	// spaces have to coexist in a single `int id` parameter at all (bdpt.h
	// itself reuses a Surface vertex's bsdf_id AS its light identifier when
	// that surface turns out to be emissive -- see BDPTVertex::PDFLight()/
	// PDFLightOrigin()'s `int lid = ... si.bsdf_id` line).
	bool Intersect(const double org[3], const double dir[3], double t_max,
	               BDPTHit<double>& hit) const {
		ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]));
		hit_record rec;
		if (!world_.hit(r, interval(0.001, t_max), rec)) return false;

		// Emission read from the raw (possibly mix_material) hit, same as
		// SPPMSceneAdapter::Intersect() -- see its own comment on why
		// mix_material's emitted() should be evaluated before resolution.
		color Le = rec.mat ? rec.mat->emitted(r, rec, rec.u, rec.v, rec.p) : color(0, 0, 0);

		shared_ptr<material> resolved_mat = rec.mat
			? sppm_resolve_material(rec.mat, rec.u, rec.v, rec.p)
			: rec.mat;

		int pool_idx = push_context(SPPMShadingContext{ rec.p, rec.normal, rec.u, rec.v, resolved_mat });

		for (int c = 0; c < 3; ++c) hit.p[c] = rec.p[c];
		for (int c = 0; c < 3; ++c) hit.geo_n[c] = rec.normal[c];       // no separate geometric normal in this codebase
		for (int c = 0; c < 3; ++c) hit.shading_n[c] = rec.normal[c];
		vec3 wo = unit_vector(-r.direction());
		hit.wo[0] = wo.x(); hit.wo[1] = wo.y(); hit.wo[2] = wo.z();
		hit.uv[0] = rec.u; hit.uv[1] = rec.v;
		hit.area_Le[0] = Le.x(); hit.area_Le[1] = Le.y(); hit.area_Le[2] = Le.z();
		hit.t_hit = rec.t;
		hit.is_medium_boundary = resolved_mat ? sppm_is_medium_boundary(resolved_mat.get()) : false;
		hit.is_delta_bsdf = resolved_mat ? sppm_is_delta_material(resolved_mat.get(), rec) : false;
		hit.bsdf_id = nEmitters_ + pool_idx;
		hit.light_id = -1;   // unused by bdpt.h's own BDPTVertex construction (see class comment)
		return true;
	}

	void BSDFf(int id, const double wo[3], const double wi[3], const double n[3], double out[3]) const {
		const SPPMShadingContext* ctx = context_for(id);
		if (!ctx) { out[0]=out[1]=out[2]=0.0; return; }
		sppm_bsdf_f(*ctx, wo, wi, n, out);
	}

	bool BSDFSampleF(int id, const double wo[3], const double n[3], double u1, double u2,
	                  double new_dir[3], double f_val[3], double& pdf, bool& is_specular) const {
		const SPPMShadingContext* ctx = context_for(id);
		if (!ctx) return false;
		return sppm_bsdf_sample_f(*ctx, wo, n, u1, u2, new_dir, f_val, pdf, is_specular);
	}

	double BSDFPdf(int id, const double wo[3], const double wi[3], const double n[3]) const {
		const SPPMShadingContext* ctx = context_for(id);
		if (!ctx) return 0.0;
		return sppm_bsdf_pdf(*ctx, wo, wi, n);
	}

	// ------------------------------------------------------------------
	// Extensions for RandomWalk/AO/SimplePath/SimpleVolPath/LightPath
	// ------------------------------------------------------------------
	// Round 6 Phase 2: the 5 debug/reference integrators in utility_
	// integrators.h/simple_path.h/simple_vol_path.h/light_path.h each need a
	// subset of this same duck-typed Scene concept, mostly already satisfied
	// by the BDPT/MLT members above (Intersect/BSDFf/BSDFSampleF/
	// SampleLight/Unoccluded/InfiniteLightLe). These methods fill the
	// remaining gaps, reusing this adapter's existing emitter list/context
	// pool rather than a second adapter class.

	// SurfaceLe: RandomWalkLi/SimplePathLi's own emission query, at a hit
	// already produced by Intersect() above -- which, like
	// SPPMSceneAdapter's, computes area_Le eagerly rather than lazily, so
	// this is a direct copy (wo unused, matching this codebase's one-sided
	// diffuse_light convention -- see named-material-and-texture.pbrt's own
	// "twosided parsed nowhere" comment for the same convention elsewhere).
	void SurfaceLe(const BDPTHit<double>& hit, const double* /*wo*/, double out[3]) const {
		out[0] = hit.area_Le[0]; out[1] = hit.area_Le[1]; out[2] = hit.area_Le[2];
	}

	// True for a real interface_material hit (Intersect() classifies it via
	// sppm_is_medium_boundary() above) - light_path.h's own skip predicate,
	// consulted the same way bdpt.h's BDPTRandomWalk consults
	// hit.is_medium_boundary directly. Looks the classification back up from
	// the shading-context pool since this only gets a bsdf_id, not the hit
	// itself - same lookup BSDFf()/BSDFSampleF()/BSDFPdf() already do.
	bool BSDFIsNull(int bsdf_id) const {
		const SPPMShadingContext* ctx = context_for(bsdf_id);
		return ctx && sppm_is_medium_boundary(ctx->mat.get());
	}

	// Offsets along the geometric normal, SIGNED by which side `dir` exits
	// on -- unlike the unit tests' own SpawnRay mocks (which always offset
	// along +geo_n, fine for their reflection-only synthetic scenes), a real
	// dielectric sphere (scene A1's glass sphere) can hand SpawnRay a `dir`
	// that transmits through the surface, and offsetting into the wrong
	// side would immediately self-intersect the same surface again. Mirrors
	// pbrt-v4's own OffsetRayOrigin() convention (offset on the side the new
	// ray direction actually points to).
	void SpawnRay(const BDPTHit<double>& hit, const double dir[3],
	              double new_o[3], double new_d[3]) const {
		constexpr double kEps = 1e-3;
		double side = (dir[0]*hit.geo_n[0] + dir[1]*hit.geo_n[1] + dir[2]*hit.geo_n[2] >= 0.0) ? 1.0 : -1.0;
		new_o[0] = hit.p[0] + side * kEps * hit.geo_n[0];
		new_o[1] = hit.p[1] + side * kEps * hit.geo_n[1];
		new_o[2] = hit.p[2] + side * kEps * hit.geo_n[2];
		new_d[0] = dir[0]; new_d[1] = dir[1]; new_d[2] = dir[2];
	}

	// AOLi's shadow-ray-with-a-distance-cap -- same shadow_ray_hit()
	// transmittance-aware occlusion test as Unoccluded() below, just bounded
	// by max_dist along dir instead of a fixed endpoint.
	bool UnoccludedWithin(const double p[3], const double dir[3], double max_dist) const {
		if (max_dist <= 0.002) return true;
		ray r(point3(p[0], p[1], p[2]), vec3(dir[0], dir[1], dir[2]));
		hit_record rec;
		return !shadow_ray_hit(world_, r, rec, max_dist - 0.002);
	}

	// SimplePathLi's uniform-sphere-vs-hemisphere fallback selector -- true
	// for the dielectric family (the only materials in this codebase's
	// BSDF bridge that transmit as well as reflect), same class set
	// bsdf_bridge.h's sppm_is_delta_material() already tests against.
	bool BSDFIsReflectiveAndTransmissive(int id) const {
		const SPPMShadingContext* ctx = context_for(id);
		if (!ctx || !ctx->mat) return false;
		const material* m = ctx->mat.get();
		return dynamic_cast<const dielectric*>(m) != nullptr ||
		       dynamic_cast<const rough_dielectric*>(m) != nullptr ||
		       dynamic_cast<const thin_dielectric*>(m) != nullptr;
	}

	// SimpleVolPathLi's medium hooks -- no participating-media scenes are in
	// this adapter's scope (see BSDFIsNull's own comment above), so
	// HasMedium() unconditionally false makes SampleTMaj/SamplePhase dead
	// code by construction: SimpleVolPathLi only calls them inside its own
	// `if (scene.HasMedium(...))` guard. --simplevolpath is consequently
	// reachable and correct on ordinary solid-geometry scenes (it reduces
	// to pbrt-v4's own SimpleVolPathIntegrator behaviour on a medium-free
	// scene: it terminates at the first surface hit, adding that surface's
	// area_Le and nothing else -- SimpleVolPathIntegrator has no surface
	// BSDF support at all, matching the upstream reference exactly, not a
	// simplification introduced here). Full participating-media wiring
	// (reusing constant_medium.h/cloud_medium.h/rgb_grid_medium.h's own
	// delta-tracking, the way camera.h's main path tracer does) is future
	// work -- see launcher_args.h's --simplevolpath help text.
	bool HasMedium(const double* /*org*/, const double* /*dir*/) const { return false; }
	double SampleTMaj(const double* /*org*/, const double* /*dir*/, double /*t_max*/, double /*u_maj*/,
	                   const std::function<bool(const double p[3], const VolPathMediumProps<double>& mp,
	                                             double sigma_maj, double T_maj)>& /*callback*/) const {
		return 1.0;   // unreachable: HasMedium() is always false
	}
	bool SamplePhase(const double* /*wo*/, double /*g*/, double /*u1*/, double /*u2*/,
	                  double* /*wi*/, double& /*pdf*/) const {
		return false;   // unreachable: HasMedium() is always false
	}
	double PhaseP(const double* /*wo*/, const double* /*wi*/, double /*g*/) const {
		return 0.0;   // unreachable: HasMedium() is always false
	}

	// LightPathTrace's importance-transport BSDF hooks. Radiance and
	// importance transport only differ for non-symmetric scattering (real
	// refraction through a dielectric interface picks up an eta^2 factor --
	// see pbrt-v4's own BxDF::Sample_f(..., TransportMode::Importance)
	// documentation) -- this adapter does not apply that correction, so
	// --lightpath output through the dielectric family is an approximation
	// (exact for every purely-reflective material this codebase's BSDF
	// bridge supports). Documented, not hidden -- see launcher_args.h's
	// --lightpath help text, same "debug/reference integrator, not
	// radiometrically exact everywhere" framing as --ao/--randomwalk.
	void BSDFfImportance(int id, const double wo[3], const double wi[3], const double n[3], double out[3]) const {
		BSDFf(id, wo, wi, n, out);
	}
	bool BSDFSampleFImportance(int id, const double wo[3], const double n[3], double u1, double u2,
	                            double new_dir[3], double f_val[3], double& pdf) const {
		bool is_specular = false;
		return BSDFSampleF(id, wo, n, u1, u2, new_dir, f_val, pdf, is_specular);
	}

	// LightPathTrace's own light-emission sample -- same power-weighted-
	// emitter / uniform-area-position / cosine-weighted-direction scheme as
	// SampleLightLe() above (LightEmissionSample<T> is a differently-shaped
	// struct than BDPTLightLeSample<T>, so this can't just forward), plus a
	// filled-in BDPTHit for LightSurfaceLe()/the has_surface direct-
	// connection path below to read back from.
	bool SampleLightEmission(double /*u_light*/, double u0a, double u0b, double u1a, double u1b,
	                          LightEmissionSample<double>& les) const {
		if (nEmitters_ == 0) return false;
		int idx = emitter_alias_.sample(random_double());
		const auto& light = emitters_[idx];

		AreaLightSample as;
		if (!light->sample_area(u0a, u0b, as)) return false;

		// Two-sided lights emit from either face with equal probability
		// (pbrt-v4's own DiffuseAreaLight::SampleLe convention) -- halving
		// pdf_dir below accounts for that extra binary choice, keeping the
		// direction density correctly normalized over the full sphere
		// instead of just the +as.n hemisphere.
		const bool two_sided = emitter_dl_[idx]->is_two_sided();
		vec3 n_emit = as.n;
		double side_pdf = 1.0;
		if (two_sided) {
			if (random_double() < 0.5) n_emit = -as.n;
			side_pdf = 0.5;
		}
		onb uvw(n_emit);
		vec3 dir = uvw.transform(random_cosine_direction());
		(void)u1a; (void)u1b;   // this codebase's random_cosine_direction() draws its own randomness (see SampleLightLe's own note)
		double cos_theta = dot(dir, n_emit);
		if (cos_theta <= 0.0) return false;

		color Le = emitter_dl_[idx]->get_texture()->value(as.u, as.v, as.p);

		les.ray_o[0]=as.p.x(); les.ray_o[1]=as.p.y(); les.ray_o[2]=as.p.z();
		les.ray_d[0]=dir.x();  les.ray_d[1]=dir.y();  les.ray_d[2]=dir.z();
		les.Le[0]=Le.x(); les.Le[1]=Le.y(); les.Le[2]=Le.z();
		les.pdf_pos = as.pdf_pos * emitter_alias_.pmf(idx);
		les.pdf_dir = side_pdf * cos_theta / pi;
		les.p_light = emitter_alias_.pmf(idx);
		les.abs_cos_theta = cos_theta;
		les.has_surface = true;

		BDPTHit<double>& sh = les.surface_hit;
		sh.p[0]=as.p.x(); sh.p[1]=as.p.y(); sh.p[2]=as.p.z();
		sh.geo_n[0]=as.n.x(); sh.geo_n[1]=as.n.y(); sh.geo_n[2]=as.n.z();
		sh.shading_n[0]=sh.geo_n[0]; sh.shading_n[1]=sh.geo_n[1]; sh.shading_n[2]=sh.geo_n[2];
		sh.wo[0]=sh.wo[1]=sh.wo[2]=0.0;
		sh.uv[0]=as.u; sh.uv[1]=as.v;
		sh.area_Le[0]=Le.x(); sh.area_Le[1]=Le.y(); sh.area_Le[2]=Le.z();
		sh.t_hit = 0.0;
		sh.is_medium_boundary = false;
		sh.is_delta_bsdf = false;
		sh.bsdf_id = -1;      // never read: LightSurfaceLe() below reads area_Le directly, not a material
		sh.light_id = idx;
		les.light_id = idx;
		return true;
	}

	// A light surface's emitted radiance toward the camera -- already
	// computed by SampleLightEmission() above into surface_hit.area_Le
	// (this codebase's diffuse_light emission has no view-angle dependence,
	// matching SurfaceLe()'s own "wo unused" convention).
	void LightSurfaceLe(const BDPTHit<double>& light_hit, const double* /*wi_to_camera*/, double out[3]) const {
		out[0] = light_hit.area_Le[0]; out[1] = light_hit.area_Le[1]; out[2] = light_hit.area_Le[2];
	}

	// pbrt-v4's light.PDF_Li(pLens, -wi) -- the solid-angle density of
	// sampling THIS light's direction from p_lens, i.e. exactly what
	// SampleLight() above already computes inline as
	// `light->pdf_value(P, wi) * emitter_alias_.pmf(idx)`; factored out
	// here so LightPathTrace's direct area-light-to-camera splat can query
	// it independently of drawing a new sample.
	double LightPdfLi(int light_id, const double* p_lens, const double* neg_wi) const {
		if (light_id < 0 || light_id >= nEmitters_) return 0.0;
		point3 P(p_lens[0], p_lens[1], p_lens[2]);
		vec3 wi(neg_wi[0], neg_wi[1], neg_wi[2]);
		return emitters_[light_id]->pdf_value(P, wi) * emitter_alias_.pmf(light_id);
	}

	// Core camera importance-sampling math, shared by SampleCameraConnection()
	// (LightPath's own camera connection, raster-pixel-index output) and
	// CameraSampleWi() below (BDPT/MLT's t==1 light-tracing connection,
	// normalized [0,1) output) - both need the identical computation,
	// differing only in what units the resulting pixel position is
	// reported in (each caller's own established convention, matching its
	// own splat destination's expectations).
	//
	// Basis/viewport constants (camForward_/camW_/camRight_/camUp_/
	// viewportWidth_/viewportHeight_/camA_/camD_/defocusRadius_) are cached
	// members, computed once in the constructor from cam_'s fields rather
	// than re-derived here on every call - see this class's own field
	// comment. Re-derives camera.h's own initialize() formulas (pixel00_loc's
	// mapping from raster to viewport-plane position) algebraically inverted
	// (viewport position -> raster position instead of raster -> ray)
	// rather than duplicating camera.h's private pixel00_loc_ member
	// directly.
	//
	// Lens sampling (defocus_angle>0): samples a point on the lens disk from
	// (u1,u2) via the same sqrt/polar uniform-disk mapping camera.h's own
	// random_in_unit_disk() achieves via rejection sampling (both give a
	// uniform sample over the unit disk; a closed-form mapping is used here
	// instead of rejection sampling so a FIXED (u1,u2) pair - required for
	// MLT's primary-sample-space reproducibility, see bdpt.h's own
	// scene.RandFloat() call sites - always produces exactly one lens
	// point). Wi/pdf use the standard thin-lens importance response
	// We = D^2/(A*cos^4(theta)) and the matching solid-angle pdf
	// dist^2/(lensArea*cos(theta)); however We/pdf's lensArea factor always
	// cancels in the only place a caller uses them together
	// (importance/pdf, see bdpt.h's Le_cam), so it's deliberately omitted
	// from both formulas below rather than introduced and cancelled by the
	// caller - the formulas are therefore identical to the pinhole case
	// (lensArea=1), correct for a lens camera too, not just simpler.
	bool cameraConnectionCore(const double p[3], double u1, double u2, double& px01, double& py01,
	                           double wi[3], double& We, double& pdf, double& dist_out) const {
		point3 p_lens = cam_.center;
		if (defocusRadius_ > 0.0) {
			double r = std::sqrt(u1);
			double theta_disk = 2.0 * pi * u2;
			double lx = r * std::cos(theta_disk);
			double ly = r * std::sin(theta_disk);
			p_lens = cam_.center + (lx * defocusRadius_) * camRight_ + (ly * defocusRadius_) * camUp_;
		}

		vec3 to_cam = p_lens - point3(p[0], p[1], p[2]);
		double dist = to_cam.length();
		if (dist < 1e-9) return false;
		vec3 w = to_cam / dist;

		vec3 lens_to_p = point3(p[0], p[1], p[2]) - p_lens;
		double t_forward = dot(lens_to_p, camForward_);
		double cosTheta = t_forward / dist;   // = dot(lens_to_p/dist, camForward_)
		if (cosTheta <= 0.0) return false;   // hit point is behind the camera

		if (camA_ <= 0.0 || camD_ <= 0.0) return false;

		// Focus plane is a fixed property of the camera (independent of
		// which lens point was sampled) - anchored at cam_.center, not
		// p_lens, matching camera.h's own viewport_upper_left derivation.
		vec3 plane_point = p_lens + lens_to_p * (camD_ / t_forward);
		vec3 viewport_center = cam_.center - camD_ * camW_;
		vec3 offset = plane_point - viewport_center;
		double su = dot(offset, camRight_);
		double sv = dot(offset, camUp_);

		double px = 0.5 + su / viewportWidth_;
		double py = 0.5 - sv / viewportHeight_;
		if (px < 0.0 || px >= 1.0 || py < 0.0 || py >= 1.0) return false;   // off-screen

		double cos4 = cosTheta * cosTheta * cosTheta * cosTheta;
		double We_ = (camD_ * camD_) / (camA_ * cos4);
		double pdf_ = (dist * dist) / cosTheta;
		if (!(We_ > 0.0) || !(pdf_ > 0.0)) return false;

		px01 = px; py01 = py;
		wi[0] = w.x(); wi[1] = w.y(); wi[2] = w.z();
		We = We_; pdf = pdf_;
		dist_out = dist;
		return true;
	}

	// LightPathTrace's camera-connection sample.
	bool SampleCameraConnection(const BDPTHit<double>& hit, double u1, double u2,
	                             CameraConnection<double>& cc) const {
		double px01, py01, wi[3], We, pdf, dist;
		if (!cameraConnectionCore(hit.p, u1, u2, px01, py01, wi, We, pdf, dist)) return false;
		// p_lens = hit.p + dist*wi (exact by construction of wi/dist above)
		// rather than always cam_.center - the sampled lens point genuinely
		// varies per-call once defocus_angle>0.
		cc.p_lens[0] = hit.p[0] + dist * wi[0];
		cc.p_lens[1] = hit.p[1] + dist * wi[1];
		cc.p_lens[2] = hit.p[2] + dist * wi[2];
		cc.p_raster[0] = px01 * cam_.image_width;
		cc.p_raster[1] = py01 * cam_.image_height;
		cc.Wi[0] = cc.Wi[1] = cc.Wi[2] = We;
		cc.wi[0] = wi[0]; cc.wi[1] = wi[1]; cc.wi[2] = wi[2];
		cc.pdf = pdf;
		return true;
	}

	// ------------------------------------------------------------------
	// Visibility
	// ------------------------------------------------------------------
	// Uses shadow_ray_hit() (shadow_ray.h) rather than a raw world_.hit(),
	// same reasoning as SPPMSceneAdapter::DirectLight()'s shadow rays: a
	// naive occlusion test would treat a glass surface between p0 and p1 as
	// full occlusion, which is wrong (see shadow_ray.h's own file comment
	// for the measured brightness-gap bug this fixed elsewhere in the
	// codebase).
	bool Unoccluded(const double p0[3], const double p1[3]) const {
		vec3 d(p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]);
		double dist = d.length();
		if (dist < 1e-9) return true;
		vec3 dir = d / dist;
		double t_max = dist - 0.002;
		if (t_max <= 0.001) return true;
		ray r(point3(p0[0], p0[1], p0[2]), dir);
		hit_record rec;
		return !shadow_ray_hit(world_, r, rec, t_max);
	}

	// ------------------------------------------------------------------
	// Light sampling (area lights only -- see class comment)
	// ------------------------------------------------------------------

	// Direct-illumination NEE sample for ConnectBDPT's s==1 strategy.
	// `u` is unused (see SampleLightLe's own note below on why this
	// codebase's material/light interfaces don't thread caller-supplied
	// randomness -- same limitation, same reasoning, applies here).
	//
	// Unified index space (matches unifiedAlias_'s own internal numbering):
	//   [0, nEmitters_)                 area lights (unchanged from before)
	//   [nEmitters_, spotBase_)         point lights
	//   [spotBase_, distBase_)          spot lights
	//   [distBase_, skyIdx_)            distant lights
	//   skyIdx_ (only if hasSky_)       the sky/infinite light
	// ls.light_id/les.light_id are NOT this raw index directly for anything
	// past an area light - toLightId() below offsets it clear of
	// Intersect()'s own bsdf_id space (which also starts at nEmitters_), so
	// resolve_emitter_index() can tell "a punctual/sky light was chosen"
	// apart from "this bsdf_id happens to belong to an emissive surface"
	// unambiguously. See toLightId()'s own comment.
	bool SampleLight(double /*u*/, const double ref_p[3], BDPTLightSample<double>& ls) const {
		if (nTotal_ == 0) return false;
		int idx = unifiedAlias_.sample(random_double());
		double pmf = unifiedAlias_.pmf(idx);
		point3 P(ref_p[0], ref_p[1], ref_p[2]);

		if (idx < nEmitters_) {
			const auto& light = emitters_[idx];

			AreaLightSample as;
			if (!light->sample_area(random_double(), random_double(), as)) return false;

			vec3 to_light = as.p - P;
			double dist = to_light.length();
			if (dist < 1e-9) return false;
			vec3 wi = to_light / dist;

			// quad/sphere's own pdf_value() intersects from either side (no
			// backface culling), so a one-sided light needs an explicit check
			// here to match diffuse_light::emitted()'s own front-face gate --
			// otherwise NEE would light points sitting behind a one-sided
			// emitter's declared normal. Two-sided lights need no such gate.
			if (!emitter_dl_[idx]->is_two_sided() && dot(wi, as.n) >= 0.0) return false;

			// light->pdf_value() already performs the area->solid-angle
			// Jacobian conversion for THIS one shape (quad.h/sphere.h's own
			// pdf_value() implementations) -- multiplying by the (unified)
			// light-selection PMF gives the full combined solid-angle
			// density, reusing tested code instead of re-deriving the
			// area/solid-angle conversion here.
			double pdf_solid_angle = light->pdf_value(P, wi) * pmf;
			if (pdf_solid_angle <= 0.0) return false;

			color Le = emitter_dl_[idx]->get_texture()->value(as.u, as.v, as.p);

			ls.p_light[0]=as.p.x(); ls.p_light[1]=as.p.y(); ls.p_light[2]=as.p.z();
			ls.n_light[0]=as.n.x(); ls.n_light[1]=as.n.y(); ls.n_light[2]=as.n.z();
			ls.L[0]=Le.x(); ls.L[1]=Le.y(); ls.L[2]=Le.z();
			ls.pdf = pdf_solid_angle;
			ls.wi[0]=wi.x(); ls.wi[1]=wi.y(); ls.wi[2]=wi.z();
			ls.is_delta = false;
			ls.is_infinite = false;
			ls.light_id = idx;
			return true;
		}

		// Punctual/sky: none of these have a real, finite extended surface,
		// so a delta light's p_light is placed exactly at its own true
		// position (point/spot - reconstructed from wi*dist, both exact),
		// and a delta-DIRECTION/infinite light's p_light is placed far
		// enough past `P` (2x the scene's bounding-sphere diameter, plus
		// P's own offset from the sphere's center so this clears regardless
		// of where P sits) that Unoccluded()'s shadow ray still correctly
		// clears all real scene geometry before "arriving" - this value is
		// otherwise unused (a delta/infinite light vertex's own pdf math
		// never reads distance-to-p_light, only ls.wi/ls.pdf; see
		// LightPDFLe's own comment).
		double farDist = 2.0 * sceneRadius_
			+ (P - point3(sceneCenter_[0], sceneCenter_[1], sceneCenter_[2])).length();

		if (idx < spotBase_) {                                    // point
			const point_light_obj& L = cam_.punct_lights->points[idx - punctBase_];
			PunctualLiSample s = L.sample_direct(P);
			if (!(s.Li.x()>0.0 || s.Li.y()>0.0 || s.Li.z()>0.0)) return false;
			point3 pl = P + s.wi * s.t_max;
			ls.p_light[0]=pl.x(); ls.p_light[1]=pl.y(); ls.p_light[2]=pl.z();
			ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.0;
			ls.L[0]=s.Li.x(); ls.L[1]=s.Li.y(); ls.L[2]=s.Li.z();
			ls.pdf = pmf;   // delta position: within-light density is 1
			ls.wi[0]=s.wi.x(); ls.wi[1]=s.wi.y(); ls.wi[2]=s.wi.z();
			ls.is_delta = true; ls.is_infinite = false; ls.light_id = toLightId(idx);
			return true;
		}
		if (idx < distBase_) {                                    // spot
			const spot_light_obj& L = cam_.punct_lights->spots[idx - spotBase_];
			PunctualLiSample s = L.sample_direct(P);
			if (!(s.Li.x()>0.0 || s.Li.y()>0.0 || s.Li.z()>0.0)) return false;
			point3 pl = P + s.wi * s.t_max;
			ls.p_light[0]=pl.x(); ls.p_light[1]=pl.y(); ls.p_light[2]=pl.z();
			ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.0;
			ls.L[0]=s.Li.x(); ls.L[1]=s.Li.y(); ls.L[2]=s.Li.z();
			ls.pdf = pmf;
			ls.wi[0]=s.wi.x(); ls.wi[1]=s.wi.y(); ls.wi[2]=s.wi.z();
			ls.is_delta = true; ls.is_infinite = false; ls.light_id = toLightId(idx);
			return true;
		}
		if (idx < skyIdx_) {                                      // distant
			const distant_light_obj& L = cam_.punct_lights->distants[idx - distBase_];
			PunctualLiSample s = L.sample_direct(P);   // t_max = infinity
			point3 pl = P + s.wi * farDist;
			ls.p_light[0]=pl.x(); ls.p_light[1]=pl.y(); ls.p_light[2]=pl.z();
			ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.0;
			ls.L[0]=s.Li.x(); ls.L[1]=s.Li.y(); ls.L[2]=s.Li.z();
			ls.pdf = pmf;
			ls.wi[0]=s.wi.x(); ls.wi[1]=s.wi.y(); ls.wi[2]=s.wi.z();
			ls.is_delta = true; ls.is_infinite = false; ls.light_id = toLightId(idx);
			return true;
		}
		// Sky (idx == skyIdx_, only reachable when hasSky_) -- NOT a delta
		// light: sample_Le() draws a real, continuous direction from the
		// image's own importance distribution, exactly like the main path
		// tracer's own NEE-toward-sky strategy (camera.h's "Strategy A-2").
		SkyLiSample sm = cam_.sky->sample_Le();
		if (sm.pdf <= 0.0) return false;
		color Le = cam_.sky->Le(sm.direction);   // already unit-length (sample_Le()'s own guarantee)
		if (!(Le.x()>0.0 || Le.y()>0.0 || Le.z()>0.0)) return false;
		point3 pl = P + sm.direction * farDist;
		ls.p_light[0]=pl.x(); ls.p_light[1]=pl.y(); ls.p_light[2]=pl.z();
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.0;
		ls.L[0]=Le.x(); ls.L[1]=Le.y(); ls.L[2]=Le.z();
		ls.pdf = sm.pdf * pmf;   // real solid-angle density, combined w/ light choice
		ls.wi[0]=sm.direction.x(); ls.wi[1]=sm.direction.y(); ls.wi[2]=sm.direction.z();
		ls.is_delta = false; ls.is_infinite = true; ls.light_id = toLightId(idx);
		return true;
	}

	// Light subpath emission. Area-light case is near-identical to
	// SPPMSceneAdapter's own SampleLightLe (same power-weighted-emitter,
	// uniform-area-position, cosine-weighted-direction scheme), duplicated
	// rather than shared for the same reason SampleLight()'s own area-light
	// branch is (see this file's header comment on why this adapter
	// deliberately does not depend on sppm_adapter.h at all). Punctual/sky
	// cases are new (see class Scope comment) - each places the emitted
	// ray's origin exactly on the light for point/spot (real, delta
	// positions) or on a disk covering the scene's bounding sphere,
	// perpendicular to the emission direction, for distant/sky (mirrors
	// pbrt-v4's own DistantLight::SampleLe/ImageInfiniteLight::SampleLe -
	// the standard technique for launching a light-subpath ray from a
	// light with no finite position).
	bool SampleLightLe(double /*u1*/, const double /*u2a*/[2], const double /*u2b*/[2],
	                    BDPTLightLeSample<double>& les) const {
		if (nTotal_ == 0) return false;
		int idx = unifiedAlias_.sample(random_double());
		double pmf = unifiedAlias_.pmf(idx);

		if (idx < nEmitters_) {
			const auto& light = emitters_[idx];

			AreaLightSample as;
			if (!light->sample_area(random_double(), random_double(), as)) return false;

			// Two-sided lights emit from either face with equal probability --
			// see SampleLightEmission()'s own comment on the matching side_pdf
			// halving above.
			const bool two_sided = emitter_dl_[idx]->is_two_sided();
			vec3 n_emit = as.n;
			double side_pdf = 1.0;
			if (two_sided) {
				if (random_double() < 0.5) n_emit = -as.n;
				side_pdf = 0.5;
			}
			onb uvw(n_emit);
			vec3 dir = uvw.transform(random_cosine_direction());
			double cos_theta = dot(dir, n_emit);
			if (cos_theta <= 0.0) return false;   // degenerate onb edge case

			color Le = emitter_dl_[idx]->get_texture()->value(as.u, as.v, as.p);

			les.ray_o[0]=as.p.x(); les.ray_o[1]=as.p.y(); les.ray_o[2]=as.p.z();
			les.ray_d[0]=dir.x();  les.ray_d[1]=dir.y();  les.ray_d[2]=dir.z();
			les.p_on_light[0]=as.p.x(); les.p_on_light[1]=as.p.y(); les.p_on_light[2]=as.p.z();
			les.n_on_light[0]=as.n.x(); les.n_on_light[1]=as.n.y(); les.n_on_light[2]=as.n.z();
			les.L[0]=Le.x(); les.L[1]=Le.y(); les.L[2]=Le.z();
			les.pdf_pos = as.pdf_pos * pmf;   // combined light-choice * position pdf
			les.pdf_dir = side_pdf * cos_theta / pi;   // cosine_pdf's own value() formula, halved for two-sided
			les.abs_cos_theta = cos_theta;
			les.is_on_surface = true;
			les.is_infinite = false;
			les.is_delta_dir = false;
			les.light_id = idx;
			return true;
		}

		if (idx < spotBase_) {                                    // point
			const point_light_obj& L = cam_.punct_lights->points[idx - punctBase_];
			vec3 dir = random_unit_vector();   // isotropic: uniform sphere, pdf = 1/(4*pi)
			color Le = L.intensity();
			point3 p = L.position();
			les.ray_o[0]=p.x(); les.ray_o[1]=p.y(); les.ray_o[2]=p.z();
			les.ray_d[0]=dir.x(); les.ray_d[1]=dir.y(); les.ray_d[2]=dir.z();
			les.p_on_light[0]=p.x(); les.p_on_light[1]=p.y(); les.p_on_light[2]=p.z();
			les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.0;
			les.L[0]=Le.x(); les.L[1]=Le.y(); les.L[2]=Le.z();
			les.pdf_pos = pmf;   // delta position
			les.pdf_dir = 1.0 / (4.0 * pi);
			les.abs_cos_theta = 1.0;   // no surface normal at a point source to weight against - pbrt-v4's own convention
			les.is_on_surface = true;
			les.is_infinite = false;
			les.is_delta_dir = true;
			les.light_id = toLightId(idx);
			return true;
		}
		if (idx < distBase_) {                                    // spot
			const spot_light_obj& L = cam_.punct_lights->spots[idx - spotBase_];
			vec3 dir; double pdf_dir;
			L.sample_le(random_double(), random_double(), random_double(), dir, pdf_dir);
			if (pdf_dir <= 0.0) return false;
			color Le = L.peak_intensity() * L.falloff(dir);
			point3 p = L.position();
			les.ray_o[0]=p.x(); les.ray_o[1]=p.y(); les.ray_o[2]=p.z();
			les.ray_d[0]=dir.x(); les.ray_d[1]=dir.y(); les.ray_d[2]=dir.z();
			les.p_on_light[0]=p.x(); les.p_on_light[1]=p.y(); les.p_on_light[2]=p.z();
			les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.0;
			les.L[0]=Le.x(); les.L[1]=Le.y(); les.L[2]=Le.z();
			les.pdf_pos = pmf;
			les.pdf_dir = pdf_dir;
			les.abs_cos_theta = 1.0;
			les.is_on_surface = true;
			les.is_infinite = false;
			les.is_delta_dir = true;
			les.light_id = toLightId(idx);
			return true;
		}
		if (idx < skyIdx_) {                                      // distant
			const distant_light_obj& L = cam_.punct_lights->distants[idx - distBase_];
			vec3 wiToLight = L.direction();
			point3 origin = diskEmissionOrigin(wiToLight);
			vec3 rayDir = -wiToLight;   // photon travels the opposite way from "toward the light"
			color Le = L.radiance();
			les.ray_o[0]=origin.x(); les.ray_o[1]=origin.y(); les.ray_o[2]=origin.z();
			les.ray_d[0]=rayDir.x(); les.ray_d[1]=rayDir.y(); les.ray_d[2]=rayDir.z();
			les.p_on_light[0]=origin.x(); les.p_on_light[1]=origin.y(); les.p_on_light[2]=origin.z();
			les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.0;
			les.L[0]=Le.x(); les.L[1]=Le.y(); les.L[2]=Le.z();
			les.pdf_pos = pmf / (pi * sceneRadius_ * sceneRadius_);   // uniform disk, combined w/ light choice
			les.pdf_dir = 1.0;   // delta direction
			les.abs_cos_theta = 1.0;   // disk is perpendicular to rayDir by construction
			les.is_on_surface = true;
			les.is_infinite = false;
			les.is_delta_dir = true;
			les.light_id = toLightId(idx);
			return true;
		}
		// Sky - same disk technique as distant, but the direction is drawn
		// from the image's own importance distribution (sample_Le() below)
		// instead of being fixed, and is_on_surface=false routes
		// BDPTGenerateLightSubpath to MakeLightInfinite instead of
		// MakeLightSurface (see that function's own dispatch).
		SkyLiSample sm = cam_.sky->sample_Le();
		if (sm.pdf <= 0.0) return false;
		vec3 w = sm.direction;        // "arrival" convention, same as SampleLight()'s own sky branch - already unit-length (sample_Le()'s own guarantee)
		point3 origin = diskEmissionOrigin(w);
		vec3 rayDir = -w;             // photon travel direction
		color Le = cam_.sky->Le(w);
		les.ray_o[0]=origin.x(); les.ray_o[1]=origin.y(); les.ray_o[2]=origin.z();
		les.ray_d[0]=rayDir.x(); les.ray_d[1]=rayDir.y(); les.ray_d[2]=rayDir.z();
		les.p_on_light[0]=les.p_on_light[1]=les.p_on_light[2]=0.0;
		les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.0;
		les.L[0]=Le.x(); les.L[1]=Le.y(); les.L[2]=Le.z();
		les.pdf_pos = pmf / (pi * sceneRadius_ * sceneRadius_);
		les.pdf_dir = sm.pdf;   // raw solid-angle density at w - matches SampleLight()'s own sky pdf convention
		les.abs_cos_theta = 1.0;
		les.is_on_surface = false;
		les.is_infinite = true;
		les.is_delta_dir = false;
		les.light_id = toLightId(idx);
		return true;
	}

	// id may be EITHER a unified light index (from SampleLight/
	// SampleLightLe's own light_id, offset via toLightId() for anything
	// past an area light) OR a bsdf_id (from a Surface vertex whose hit
	// happened to be emissive -- see this class's own header comment on why
	// bdpt.h reuses si.bsdf_id as a light identifier for that case).
	// resolve_emitter_index() disambiguates and returns the resolved index
	// into unifiedAlias_'s own 0..nTotal_-1 numbering either way.
	double LightPMF(int id) const {
		int k = resolve_emitter_index(id);
		return (k >= 0) ? unifiedAlias_.pmf(k) : 0.0;
	}

	// k (already resolved to unifiedAlias_'s own numbering by
	// resolve_emitter_index()) determines which light kind's pdf formula
	// applies:
	//   - area  (k < nEmitters_): p/n identify the specific surface point,
	//     pdf_pos is that shape's own precomputed 1/area, pdf_dir is the
	//     cosine-weighted emission density.
	//   - point/spot (delta position): pdf_pos = 1 (a delta has no real
	//     positional density; SampleLight/SampleLightLe's own combined pdfs
	//     already fold the light-choice pmf into the field that DOES carry
	//     real information for that light instead - see their own
	//     comments), pdf_dir = the real emission-direction density (spot's
	//     own cone-importance-sampled pdf_le(); point's isotropic 1/(4*pi)).
	//   - distant/sky (delta direction / infinite): pdf_pos = the disk-
	//     sampling density 1/(pi*r^2) (this IS real spatial information -
	//     see SampleLightLe's own comment on why distant/sky's roles are
	//     the mirror image of point/spot's), pdf_dir = 1 for distant
	//     (delta), or the sky's own real solid-angle density for sky.
	// p is only used by the area case (unused parameter name kept for
	// documentation - not renamed to avoid an unused-parameter warning
	// diverging from every other kind's own signature).
	void LightPDFLe(int id, const double* /*p*/, const double* n, const double* w,
	                 double& pdfPos, double& pdfDir) const {
		int k = resolve_emitter_index(id);
		if (k < 0) { pdfPos = 0.0; pdfDir = 0.0; return; }
		if (k < nEmitters_) {
			pdfPos = emitter_pdf_pos_[k];
			double cosTheta = n ? (w[0]*n[0] + w[1]*n[1] + w[2]*n[2]) : 1.0;
			pdfDir = (cosTheta > 0.0) ? cosTheta / pi : 0.0;
			return;
		}
		if (k < spotBase_) {                    // point
			pdfPos = 1.0;
			pdfDir = 1.0 / (4.0 * pi);
			return;
		}
		if (k < distBase_) {                    // spot
			pdfPos = 1.0;
			pdfDir = cam_.punct_lights->spots[k - spotBase_].pdf_le(vec3(w[0], w[1], w[2]));
			return;
		}
		// distant or sky - identical positional density (both use the
		// same bounding-sphere disk), differ only in directional density.
		pdfPos = 1.0 / (pi * sceneRadius_ * sceneRadius_);
		pdfDir = (k < skyIdx_) ? 1.0 : cam_.sky->pdf_Li(vec3(w[0], w[1], w[2]));
	}

	// ------------------------------------------------------------------
	// Camera
	// ------------------------------------------------------------------

	// pbrt-v4 PerspectiveCamera::PDF_We(), adapted to camera.h's own
	// screen-window formula (initialize(): viewport_height = 2*tan(vfov/2)
	// *focus_dist, viewport_width = viewport_height*aspect) rather than
	// reading camera.h's private pixel_delta_u/v members -- recomputes the
	// same two numbers from cam_'s public fields (vfov/focus_dist/
	// image_width/image_height), which is exactly what initialize()
	// itself does internally, so this stays byte-for-byte consistent with
	// whatever screen camera.h's get_ray() actually generates rays through.
	// pdfPos is a placeholder (1.0): traced through bdpt.h's own code, it
	// is read by BDPTGenerateCameraSubpath/BDPTVertex::PDF() and then
	// immediately discarded in both call sites -- see this class's own
	// note in the file header on CameraPDFWe's alt-camera-model caveat.
	void CameraPDFWe(const double* /*ray_o*/, const double* ray_d,
	                  double& pdfPos, double& pdfDir) const {
		vec3 forward = unit_vector(cam_.lookat - cam_.lookfrom);
		vec3 d = unit_vector(vec3(ray_d[0], ray_d[1], ray_d[2]));
		double cosTheta = dot(d, forward);
		if (cosTheta <= 0.0) { pdfPos = 0.0; pdfDir = 0.0; return; }

		double theta = degrees_to_radians(cam_.vfov);
		double h = std::tan(theta / 2.0);
		double viewport_height = 2.0 * h * cam_.focus_dist;
		double viewport_width  = viewport_height * (double(cam_.image_width) / double(cam_.image_height));
		double A = viewport_width * viewport_height;
		double D = cam_.focus_dist;
		if (A <= 0.0 || D <= 0.0) { pdfPos = 0.0; pdfDir = 0.0; return; }

		// pdfDir is the solid-angle density of ray DIRECTIONS conditioned on
		// a fixed ray origin - once the origin is fixed (whether that's the
		// single pinhole point or one particular sampled point on the
		// defocus disk), the geometric relationship between that point and
		// the image plane is exactly the same pinhole cos^3(theta)/(A*D^2)
		// form (pbrt-v4's own PerspectiveCamera::PDF_We() likewise leaves
		// this formula's shape unchanged for a thin-lens camera - only
		// pdfPos below picks up a lens-area factor). No defocus-angle
		// correction belongs here.
		double cos3 = cosTheta * cosTheta * cosTheta;
		pdfDir = (D * D) / (A * cos3);

		// pdfPos: with no lens (defocus_angle<=0, camera.h's get_ray() always
		// uses the single fixed pinhole point), the camera position is a
		// delta distribution - 1.0 is the conventional "point mass" pdfPos
		// pbrt-v4 itself uses for a pinhole PerspectiveCamera. With a lens
		// (defocus_angle>0), get_ray() instead samples ray_origin uniformly
		// from a disk of radius focus_dist*tan(defocus_angle/2) (see
		// camera.h's initialize()/defocus_disk_sample()), so pdfPos is
		// really 1/lensArea, not 1.0.
		//
		// This is currently DEAD for BDPT/MLT's actual output: every call
		// site in src/shared/bdpt.h reads pdfPos into a discarded/unused
		// local (see this class's own file-header comment) - only pdfDir
		// above ever reaches a real MIS weight. Computed correctly anyway
		// (rather than left as the pinhole placeholder) so this value is
		// right by construction if a future change ever does start using
		// it. CameraSampleWi's own t==1 light-tracing strategy (see
		// cameraConnectionCore()) now DOES support lens sampling, but
		// deliberately never reads pdfPos either - its lensArea factor
		// always cancels in the only place it would be used
		// (importance/pdf), so introducing it there would just be
		// cancelled straight back out, not a missing correction.
		if (cam_.defocus_angle > 0.0) {
			const double lens_radius = cam_.focus_dist * std::tan(degrees_to_radians(cam_.defocus_angle / 2.0));
			const double lens_area = pi * lens_radius * lens_radius;
			pdfPos = (lens_area > 0.0) ? (1.0 / lens_area) : 1.0;
		} else {
			pdfPos = 1.0;
		}
	}

	// t==1 "light tracing" strategy: connects a light-subpath vertex
	// directly to the camera (or, when defocus_angle>0, to a sampled lens
	// point - see cameraConnectionCore()'s own comment). Shares its core
	// importance math with SampleCameraConnection() above but reports
	// pRaster in NORMALIZED [0,1) image coordinates, not raster pixel-index
	// units -- this is the convention mlt_render_with_adapter()'s own
	// pre-existing splat lambda expects (it does its own `px*width`/
	// `py*height` conversion), and BDPT's caller (bdpt_render_with_adapter())
	// converts to raster units itself before handing the result to
	// SplatFilm, so a single normalized contract here satisfies both
	// callers without either one adapting to the other's convention.
	//
	// Returns `dist` (the distance from ref_p to the sampled camera point)
	// rather than the camera point itself: the caller already has ref_p and
	// the returned wi, and p_cam = ref_p + dist*wi is an exact vector
	// identity regardless of camera model (pinhole or lens) - narrower than
	// widening this duck-typed interface with a 3-vector every implementer
	// (including test scenes that never exercise lens sampling) would have
	// to carry.
	bool CameraSampleWi(const double* ref_p, const double* u2,
	                     double* wi, double& pdf, double& importance,
	                     double* pRaster, double& dist) const {
		double px01, py01, We;
		double u1v = u2 ? u2[0] : 0.5;
		double u2v = u2 ? u2[1] : 0.5;
		if (!cameraConnectionCore(ref_p, u1v, u2v, px01, py01, wi, We, pdf, dist)) {
			pdf = 0.0; importance = 0.0;
			return false;
		}
		importance = We;
		pRaster[0] = px01;
		pRaster[1] = py01;
		return true;
	}

	void SceneBoundingSphere(double center[3], double& radius) const {
		center[0] = sceneCenter_[0]; center[1] = sceneCenter_[1]; center[2] = sceneCenter_[2];
		radius = sceneRadius_;
	}

	// A real sky (cam_.sky) returns its actual per-direction radiance,
	// exactly matching the main path tracer's own miss handling
	// (camera.h's `sky->Le(unit_vector(current_ray.direction()))`) - `dir`
	// here is the escaping camera ray's own direction (BDPTVertex::Le()'s
	// call convention, see its own comment: "neg_w" there is the direction
	// FROM the light TOWARD the previous vertex, which for a camera ray
	// escaping to infinity is exactly the ray's own travel direction).
	// Falls back to cam_.background when there's no sky, matching what a
	// plain camera ray miss already renders as everywhere else in this
	// codebase - for a black-background, sky-less scene this is exactly
	// zero, i.e. a no-op.
	void InfiniteLightLe(const double* dir, double out[3]) const {
		if (hasSky_) {
			color Le = cam_.sky->Le(unit_vector(vec3(dir[0], dir[1], dir[2])));
			out[0] = Le.x(); out[1] = Le.y(); out[2] = Le.z();
			return;
		}
		out[0] = cam_.background.x(); out[1] = cam_.background.y(); out[2] = cam_.background.z();
	}

	// Every caller (bdpt.h's PDFLightOrigin() and BDPTGenerateLightSubpath())
	// now passes the "arrival" convention (direction FROM the one real
	// vertex a sky vertex was captured relative to, TOWARD the sky) -
	// BDPTEndpointData::dir's own comment (bdpt.h) documents this, and both
	// call sites derive it consistently (BDPTGenerateLightSubpath negates
	// les.ray_d, its own emission-direction sample, before storing it). A
	// flat (sky-less) backdrop has no associated sampling strategy
	// (SampleLightLe never emits an is_infinite=true sample when !hasSky_),
	// so its origin-pdf contribution to any OTHER strategy's MIS weight is
	// correctly zero.
	double InfiniteLightDensity(const double* dir) const {
		if (!hasSky_) return 0.0;
		double pmf = unifiedAlias_.pmf(skyIdx_);
		return cam_.sky->pdf_Li(vec3(dir[0], dir[1], dir[2])) * pmf;
	}

	double RandFloat() const { return random_double(); }

	// px,py in [0,1) -- pixel-center-normalized, matching
	// SPPMSceneAdapter::PixelToRay's own convention. cam_n is set to the
	// camera's forward direction, NOT a placeholder: bdpt.h's
	// BDPTVertex::IsOnSurface() treats a non-zero ei.n as "this endpoint is
	// on a surface" and applies a cosine factor accordingly, which is
	// exactly pbrt-v4's own EndpointInteraction convention for a
	// perspective camera (its interaction normal IS the viewing direction)
	// -- unlike SPPMSceneAdapter::PixelToRay, where cam_n is genuinely
	// unused padding (sppm.h's own SPPMCameraPass never reads it), this
	// value has real, load-bearing meaning for BDPT's ConvertDensity()/PDF()
	// math.
	bool PixelToRay(double px, double py, double cam_p[3], double ray_d[3], double cam_n[3]) const {
		int i = static_cast<int>(px * cam_.image_width);
		int j = static_cast<int>(py * cam_.image_height);
		if (i < 0) i = 0;
		if (i >= cam_.image_width) i = cam_.image_width - 1;
		if (j < 0) j = 0;
		if (j >= cam_.image_height) j = cam_.image_height - 1;

		ray r = cam_.get_ray(i, j, 0, 0, vec3(0.0, 0.0, 0.0));
		cam_p[0] = r.origin().x(); cam_p[1] = r.origin().y(); cam_p[2] = r.origin().z();
		vec3 d = unit_vector(r.direction());
		ray_d[0] = d.x(); ray_d[1] = d.y(); ray_d[2] = d.z();
		vec3 fwd = unit_vector(cam_.lookat - cam_.lookfrom);
		cam_n[0] = fwd.x(); cam_n[1] = fwd.y(); cam_n[2] = fwd.z();
		return true;
	}

	// mlt.h's own required extension beyond bdpt.h's concept (its file
	// header: "Luminance of a spectral RGB triple (scalar importance
	// measure)") -- standard Rec. 709 relative luminance, matching
	// SPPMSceneAdapter's own emitter-power weighting formula and
	// power_light_sampler.h's for consistency across this codebase.
	double Luminance(double r, double g, double b) const {
		return 0.2126*r + 0.7152*g + 0.0722*b;
	}

  private:
	const hittable_list& world_;
	const camera&         cam_;

	std::vector<shared_ptr<hittable>>      emitters_;
	std::vector<shared_ptr<diffuse_light>> emitter_dl_;
	std::vector<double>                    emitter_pdf_pos_;  // 1/area per emitter, parallel to emitters_
	AliasTable                             emitter_alias_;    // area lights only - SampleLightEmission()'s own table
	int                                    nEmitters_ = 0;

	// Unified area+point+spot+distant+sky distribution and the index-range
	// bookkeeping SampleLight()/SampleLightLe()/LightPDFLe() dispatch on -
	// see SampleLight()'s own comment for the layout. sceneCenter_/
	// sceneRadius_ are the scene's bounding sphere, computed once in the
	// constructor (see its own comment).
	AliasTable unifiedAlias_;
	int        punctBase_ = 0, spotBase_ = 0, distBase_ = 0, skyIdx_ = 0, nTotal_ = 0;
	bool       hasSky_ = false;
	double     sceneCenter_[3] = {0.0, 0.0, 0.0};
	double     sceneRadius_ = 1.0;

	// Camera basis/viewport constants used by cameraConnectionCore() below -
	// derived once here (constructor) from cam_'s fields (vfov/focus_dist/
	// image dimensions/lookfrom/lookat/vup/defocus_angle), all of which are
	// fixed for the adapter's lifetime, rather than re-derived (std::tan +
	// several vector ops) on every camera-connection call. defocusRadius_
	// is 0 when defocus_angle<=0 (pinhole - cameraConnectionCore treats
	// that as "no lens sampling").
	vec3   camForward_, camW_, camRight_, camUp_;
	double viewportWidth_ = 0.0, viewportHeight_ = 0.0, camA_ = 0.0, camD_ = 0.0;
	double defocusRadius_ = 0.0;

	// Fixed-capacity, thread-local ring buffer of shading contexts. Unlike
	// SPPMSceneAdapter's durable_ctx_/transient_ctx_ split, BDPT/MLT need
	// MULTIPLE contexts simultaneously alive per thread (a whole subpath's
	// worth, both camera and light side, all read again during connection/
	// MIS-weight evaluation) -- but only ever the ones from the path
	// CURRENTLY being evaluated: bdpt.h's BDPTLi() and mlt.h's MLTEvalPath()
	// each run start-to-finish, single-threaded, entirely synchronously
	// before returning (no async/deferred use of a hit's bsdf_id survives
	// past the call that produced it), and each touches at most
	// 2*(maxDepth+2) hits. Sizing the ring well above that (kCtxPoolCapacity)
	// makes wraparound-during-one-path-evaluation practically impossible for
	// any sane maxDepth, while keeping memory BOUNDED regardless of how many
	// samples/mutations the caller runs -- unlike a naive ever-growing
	// std::vector, which would be the natural first design (and was this
	// file's first draft) but leaks unboundedly across e.g. an MLT chain's
	// millions of mutations, since nothing outside this class ever gets a
	// natural "start of a new path" hook to clear it at (mlt.h's
	// MLTRenderLoop is a closed-box loop with no such callback). No
	// explicit reset method is needed as a result -- contrast
	// SPPMSceneAdapter's BeginIteration(), which exists only for interface
	// parity and does nothing either, for a different reason.
	static constexpr size_t kCtxPoolCapacity = 1u << 16;  // 65536 slots/thread
	static inline thread_local std::vector<SPPMShadingContext> ctx_pool_;

	// The numeric offset toLightId()/resolve_emitter_index() use to keep a
	// punctual/sky light_id from ever colliding with a real bsdf_id (which
	// occupies [nEmitters_, nEmitters_+kCtxPoolCapacity)) - a SEPARATE,
	// independently-sized constant from kCtxPoolCapacity above, even though
	// it happens to reuse the same value today: kCtxPoolCapacity is sized
	// purely for the ring buffer's own wraparound-avoidance (see its own
	// comment) and could legitimately shrink or grow for that reason alone
	// in the future, with no relation to how many light_ids need disjoint
	// numbering. Keeping them as two named constants (with the static_assert
	// below enforcing the one relationship that actually matters - this
	// offset must never be smaller than the bsdf_id range) means a future
	// change to one can't silently break the other by coincidence.
	static constexpr int kLightIdOffset = 1 << 16;
	static_assert(kLightIdOffset >= (int)kCtxPoolCapacity,
	              "kLightIdOffset must stay >= kCtxPoolCapacity, or a real "
	              "bsdf_id could be numerically reinterpreted as a light_id");
	static inline thread_local uint64_t ctx_write_pos_ = 0;

	int push_context(SPPMShadingContext ctx) const {
		if (ctx_pool_.size() < kCtxPoolCapacity) ctx_pool_.resize(kCtxPoolCapacity);
		size_t slot = static_cast<size_t>(ctx_write_pos_ % kCtxPoolCapacity);
		ctx_pool_[slot] = std::move(ctx);
		++ctx_write_pos_;
		return static_cast<int>(slot);
	}

	const SPPMShadingContext* context_for(int id) const {
		int pool_idx = id - nEmitters_;
		if (pool_idx < 0 || pool_idx >= (int)ctx_pool_.size()) return nullptr;
		return &ctx_pool_[pool_idx];
	}

	// Real bsdf_ids (Intersect()'s hit.bsdf_id = nEmitters_ + pool_idx)
	// occupy [nEmitters_, nEmitters_+kCtxPoolCapacity). A punctual/sky
	// light's own light_id needs to live OUTSIDE that whole range so
	// resolve_emitter_index() below can tell the two apart by numeric range
	// alone, with no ambiguity regardless of how full ctx_pool_ currently
	// is - offsetting by kLightIdOffset (see its own comment for why this
	// is a dedicated constant, not kCtxPoolCapacity reused directly) is
	// what guarantees that. Called by SampleLight()/SampleLightLe() for
	// every kind past an area light; area lights keep using their raw
	// unifiedAlias_ index directly (identity here) since that numbering
	// was already established, pre-existing contract before this class
	// supported anything else.
	int toLightId(int aliasIdx) const {
		return (aliasIdx < nEmitters_) ? aliasIdx : (aliasIdx + kLightIdOffset);
	}

	// Samples a point on a disk of radius sceneRadius_, centered on the
	// scene's bounding sphere and displaced by sceneRadius_ along axisDir,
	// with the disk's own plane perpendicular to axisDir - the standard
	// technique for launching a light-subpath ray from a light with no
	// finite position (mirrors pbrt-v4's DistantLight::SampleLe/
	// ImageInfiniteLight::SampleLe). Shared by SampleLightLe()'s distant
	// and sky branches, which otherwise differ only in how axisDir/the
	// emitted radiance are obtained.
	point3 diskEmissionOrigin(const vec3& axisDir) const {
		onb frame(axisDir);
		double dx, dy;
		SampleUniformDiskConcentric(random_double(), random_double(), dx, dy);
		point3 center(sceneCenter_[0], sceneCenter_[1], sceneCenter_[2]);
		return center + sceneRadius_ * axisDir + sceneRadius_ * frame.transform(vec3(dx, dy, 0.0));
	}

	// Disambiguates the numbering spaces LightPMF/LightPDFLe can be called
	// with -- an area-light index (< nEmitters_, used as-is), a punctual/
	// sky light_id (>= nEmitters_+kLightIdOffset, see toLightId() above -
	// unwrapped back to unifiedAlias_'s own numbering), or a bsdf_id (from
	// a Surface vertex whose hit happened to be emissive -- see this
	// class's own header comment on why bdpt.h reuses si.bsdf_id as a light
	// identifier for that case). Returns the resolved index into
	// unifiedAlias_'s own 0..nTotal_-1 numbering, or -1.
	int resolve_emitter_index(int id) const {
		if (id >= 0 && id < nEmitters_) return id;
		if (id >= nEmitters_ + kLightIdOffset) {
			int k = id - kLightIdOffset;
			return (k < nTotal_) ? k : -1;
		}
		const SPPMShadingContext* ctx = context_for(id);
		if (!ctx || !ctx->mat) return -1;
		const material* m = ctx->mat.get();
		for (int k = 0; k < nEmitters_; ++k)
			if (emitter_dl_[k].get() == m) return k;
		return -1;
	}

	// Only quad/sphere currently expose get_material() (both predate this
	// file); mirrors SPPMSceneAdapter's own identically-named private
	// helper (duplicated rather than shared -- see this file's own header
	// comment on why this adapter avoids depending on sppm_adapter.h).
	static shared_ptr<material> hittable_material(const shared_ptr<hittable>& h) {
		if (auto q = std::dynamic_pointer_cast<quad>(h)) return q->get_material();
		if (auto s = std::dynamic_pointer_cast<sphere>(h)) return s->get_material();
		return nullptr;
	}
};

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
	SplatFilm(int width, int height)
		: width_(width), height_(height),
		  buf_(static_cast<size_t>(width) * height * 3, 0.0),
		  mutexes_(static_cast<size_t>(width) * height) {}

	void Splat(double px, double py, const double L[3]) {
		int ix = static_cast<int>(px);
		int iy = static_cast<int>(py);
		if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) return;
		size_t pidx = static_cast<size_t>(iy) * width_ + ix;
		std::lock_guard<std::mutex> lg(mutexes_[pidx]);
		for (int c = 0; c < 3; ++c) {
			double v = L[c];
			if (!std::isfinite(v) || v < 0.0) v = 0.0;
			buf_[pidx * 3 + c] += v;
		}
	}

	// norm: total samples PER PIXEL across the whole image (spp) -- NOT
	// divided by pixel count again, since each splat already lands at a
	// specific pixel; this mirrors pbrt-v4's own LightPathIntegrator film
	// reconstruction (splat weight accumulates raw, normalized once by the
	// image's total sample count at the end).
	void ToRGB(std::vector<double>& out_rgb, double norm) const {
		out_rgb.resize(buf_.size());
		for (size_t i = 0; i < buf_.size(); ++i) out_rgb[i] = buf_[i] / norm;
	}

  private:
	int width_, height_;
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
                                      int spp, int maxDepth, std::vector<double>& out_rgb) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);

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
		std::call_once(film_once, [&]() { film.emplace(width, height); });
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
				double sum[3] = { 0.0, 0.0, 0.0 };
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;

					double cam_p[3], cam_n[3], ray_d[3];
					if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) continue;

					double L[3];
					BDPTLi<double>(cam_p, cam_n, ray_d, maxDepth, scene,
					                cameraVerts.data(), lightVerts.data(), L, splat);

					for (int c = 0; c < 3; ++c) {
						double v = L[c];
						if (!std::isfinite(v) || v < 0.0) v = 0.0;   // firefly/NaN guard, matches cpu_interface's path tracer
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

	if (film) {
		std::vector<double> splat_rgb;
		film->ToRGB(splat_rgb, static_cast<double>(spp));
		for (size_t i = 0; i < out_rgb.size(); ++i) out_rgb[i] += splat_rgb[i];
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
                                     std::vector<double>& out_rgb) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
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

inline void randomwalk_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height,
                                            int spp, int maxDepth, std::vector<double>& out_rgb) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	unsigned int nthreads = determine_render_thread_count();
	std::atomic<int> next_row(0);
	auto worker = [&]() {
		auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;
			for (int ix = 0; ix < width; ++ix) {
				double sum[3] = {0.0, 0.0, 0.0};
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;
					double cam_p[3], cam_n[3], ray_d[3];
					if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) continue;
					double L[3];
					RandomWalkLi<double>(cam_p, ray_d, scene, maxDepth, rand2d, L);
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

inline void ao_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                    double maxDist, bool cosSample, double illumScale, const double illumRgb[3],
                                    std::vector<double>& out_rgb) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	unsigned int nthreads = determine_render_thread_count();
	std::atomic<int> next_row(0);
	auto worker = [&]() {
		auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;
			for (int ix = 0; ix < width; ++ix) {
				double sum[3] = {0.0, 0.0, 0.0};
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;
					double cam_p[3], cam_n[3], ray_d[3];
					if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) continue;
					double L[3];
					AOLi<double>(cam_p, ray_d, scene, maxDist, cosSample, illumScale, illumRgb, rand2d, L);
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

inline void simplepath_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                            int maxDepth, bool sampleLights, bool sampleBsdf,
                                            std::vector<double>& out_rgb) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	unsigned int nthreads = determine_render_thread_count();
	std::atomic<int> next_row(0);
	auto worker = [&]() {
		auto rand2d = []() { return std::pair<double, double>(random_double(), random_double()); };
		auto rand1d = []() { return random_double(); };
		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;
			for (int ix = 0; ix < width; ++ix) {
				double sum[3] = {0.0, 0.0, 0.0};
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;
					double cam_p[3], cam_n[3], ray_d[3];
					if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) continue;
					double L[3] = {0.0, 0.0, 0.0};
					SimplePathLi<double>(cam_p, ray_d, scene, maxDepth, sampleLights, sampleBsdf,
					                      rand2d, rand1d, L);
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

inline void simplevolpath_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                               int maxDepth, std::vector<double>& out_rgb) {
	out_rgb.assign(static_cast<size_t>(width) * height * 3, 0.0);
	unsigned int nthreads = determine_render_thread_count();
	std::atomic<int> next_row(0);
	auto worker = [&]() {
		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;
			for (int ix = 0; ix < width; ++ix) {
				double sum[3] = {0.0, 0.0, 0.0};
				for (int s = 0; s < spp; ++s) {
					double px = (ix + random_double()) / width;
					double py = (iy + random_double()) / height;
					double cam_p[3], cam_n[3], ray_d[3];
					if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n)) continue;
					double L[3] = {0.0, 0.0, 0.0};
					SimpleVolPathLi<double>(cam_p, ray_d, scene, maxDepth, L);
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

inline void lightpath_render_with_adapter(const BDPTSceneAdapter& scene, int width, int height, int spp,
                                           int maxDepth, std::vector<double>& out_rgb) {
	SplatFilm film(width, height);
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

	film.ToRGB(out_rgb, static_cast<double>(spp));
}
