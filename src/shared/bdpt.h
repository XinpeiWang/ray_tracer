#pragma once
// ---------------------------------------------------------------------------
// bdpt.h -- Bidirectional Path Tracing helpers (pbrt-v4 port)
//
// Ports pbrt-v4 cpu/integrators.cpp BDPT subsystem:
//   - BDPTVertexType enum (Camera, Light, Surface)
//   - ScopedAssign<T>   RAII temporary-value helper used by MISWeight
//   - BDPTVertex        path vertex with endpoint/surface payloads
//   - G()               geometry coupling term with visibility
//   - InfiniteLightDensity()
//   - MISWeight()       power-heuristic multi-strategy weight
//   - RandomWalk()      generic subpath tracer (camera or light)
//   - GenerateCameraSubpath() / GenerateLightSubpath()
//   - ConnectBDPT()     bidirectional connection for one (s,t) strategy
//   - BDPTLi()          top-level estimator (sums all strategies)
//
// Design rules (same as ratio_tracking.h / measured_bxdf.h):
//   - Header-only, no virtual functions, no heap allocation in hot paths
//   - Template parameter T: float or double
//   - Template parameter Scene: caller-supplied scene query interface
//     (see BDPTSceneConcept below for the required API)
//   - Medium (volume) vertices are deferred; only surface + endpoint
//     vertices are handled here, matching the initial pbrt-v4 BDPT scope.
//
// Scene concept (duck-typed):
//   struct Scene {
//     // Intersect a ray; returns false if missed.
//     // On hit fills: hit_p[3], geo_n[3], shading_n[3], uv[2],
//     //               area_light_Le[3] (zero if none), bsdf (see below),
//     //               is_medium_boundary, t_hit.
//     bool Intersect(const T org[3], const T dir[3], T t_max,
//                    BDPTHit<T>& hit) const;
//
//     // Shadow ray: returns true if p0->p1 is unoccluded.
//     bool Unoccluded(const T p0[3], const T p1[3]) const;
//
//     // Sample a light for direct illumination.
//     // Returns false if no lights. Fills ls.
//     bool SampleLight(T u, const T ref_p[3],
//                      BDPTLightSample<T>& ls) const;
//
//     // Sample a light ray for the light subpath (SampleLe).
//     bool SampleLightLe(T u1d, const T u2a[2], const T u2b[2],
//                        BDPTLightLeSample<T>& les) const;
//
//     // PDF for choosing a particular light (PMF).
//     T LightPMF(int light_id) const;
//
//     // Camera: sample direction from film position.
//     //   pdfPos, pdfDir are output.
//     void CameraPDFWe(const T ray_o[3], const T ray_d[3],
//                      T& pdfPos, T& pdfDir) const;
//
//     // Camera importance sampling from a reference point.
//     bool CameraSampleWi(const T ref_p[3], const T u2[2],
//                         T wi[3], T& pdf, T& importance,
//                         T pRaster[2]) const;
//
//     // Scene bounding sphere for infinite-light PDF.
//     void SceneBoundingSphere(T center[3], T& radius) const;
//
//     // Infinite lights: iterate with for(auto& il : scene.InfiniteLights())
//     // Each il provides il.PDF_Li(dir[3]) and il.PMF().
//   };
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "scalar_math.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <optional>
#include <array>
#include <cstring>

// ===========================================================================
// Section 1 -- Enumerations and basic data structures
// ===========================================================================

enum class BDPTVertexType { Camera, Light, Surface };

// BDPTHit<T>: scene intersection result passed back to bdpt.h
template<typename T>
struct BDPTHit {
	T p[3];          // hit position
	T geo_n[3];      // geometric normal (unit)
	T shading_n[3];  // shading normal (unit)
	T wo[3];         // outgoing direction at hit (towards previous vertex)
	T uv[2];
	T area_Le[3];    // emitted radiance if area light (zero otherwise)
	T t_hit;
	bool is_medium_boundary; // skip over medium boundaries (no BSDF)
	bool is_delta_bsdf;      // true for perfect specular/transmissive
	// For BSDF evaluation the caller stores opaque per-hit data;
	// we call back through the Scene trait methods.
	int bsdf_id;    // opaque index used by Scene::BSDFf / BSDFSampleF / BSDFPdf
	int light_id;   // -1 if not an area light; otherwise light index for PDF queries
};

// BDPTLightSample<T>: result of Scene::SampleLight (direct illumination)
template<typename T>
struct BDPTLightSample {
	T p_light[3];    // sampled point on the light
	T n_light[3];    // normal at that point (zero for point lights)
	T L[3];          // emitted radiance / intensity
	T pdf;           // solid-angle or position pdf
	T wi[3];         // direction from ref_p towards light
	bool is_delta;   // true for point/directional lights
	int light_id;
};

// BDPTLightLeSample<T>: result of Scene::SampleLightLe (light subpath)
template<typename T>
struct BDPTLightLeSample {
	T ray_o[3];   // ray origin
	T ray_d[3];   // ray direction (unit)
	T p_on_light[3];
	T n_on_light[3];  // zero for point lights
	T L[3];
	T pdf_pos;
	T pdf_dir;
	T abs_cos_theta; // |dot(ray_d, n_on_light)|
	bool is_on_surface; // true if light is an area light (has n_on_light)
	bool is_infinite;   // true for environment lights
	bool is_delta_dir;  // true for directional lights
	int light_id;
};

// ===========================================================================
// Section 2 -- ScopedAssign<T>: RAII temporary mutation (pbrt-v4 ScopedAssignment)
// ===========================================================================

template<typename T>
class ScopedAssign {
public:
	ScopedAssign() : target_(nullptr) {}
	ScopedAssign(T* target, T value) : target_(target), backup_(*target) {
		*target = value;
	}
	~ScopedAssign() {
		if (target_) *target_ = backup_;
	}
	ScopedAssign(const ScopedAssign&) = delete;
	ScopedAssign& operator=(const ScopedAssign&) = delete;
	ScopedAssign& operator=(ScopedAssign&& o) noexcept {
		if (target_) *target_ = backup_;
		target_ = o.target_; backup_ = o.backup_; o.target_ = nullptr;
		return *this;
	}
	ScopedAssign(ScopedAssign&& o) noexcept : target_(o.target_), backup_(o.backup_) {
		o.target_ = nullptr;
	}
private:
	T* target_;
	T  backup_;
};

// ===========================================================================
// Section 3 -- BDPTVertex
// ===========================================================================

namespace bdpt_detail {

template<typename T>
inline T dot3(const T a[3], const T b[3]) {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
template<typename T>
inline T len2_3(const T a[3]) { return a[0]*a[0]+a[1]*a[1]+a[2]*a[2]; }
template<typename T>
inline T len_3(const T a[3]) { return std::sqrt(len2_3(a)); }
template<typename T>
inline void norm3(T a[3]) {
	T inv = T(1) / len_3(a);
	a[0]*=inv; a[1]*=inv; a[2]*=inv;
}
template<typename T>
inline T absdot3(const T a[3], const T b[3]) {
	return std::abs(dot3(a,b));
}

} // namespace bdpt_detail

// Surface interaction data stored in a surface vertex
template<typename T>
struct BDPTSurfaceData {
	T p[3];          // position
	T geo_n[3];      // geometric normal
	T shading_n[3];  // shading normal
	T wo[3];         // outgoing direction (towards camera/previous)
	T area_Le[3];    // emitted radiance (zero if not an area light)
	bool is_delta_bsdf;
	int bsdf_id;     // opaque, passed back to scene for f/sample/pdf calls
};

// Endpoint data: camera or light endpoint
template<typename T>
struct BDPTEndpointData {
	T p[3];
	T n[3];          // surface normal (zero for non-surface endpoints)
	T Le[3];         // emitted radiance (light endpoints)
	T pdf_pos;       // spatial pdf used when this was created
	bool is_infinite; // true for environment map lights
	bool is_delta;    // true for point/directional lights
	int light_id;    // -1 for camera
};

template<typename T>
struct BDPTVertex {
	BDPTVertexType type;
	T beta[3];       // path throughput up to and including this vertex
	T pdfFwd;        // forward area pdf
	T pdfRev;        // reverse area pdf
	bool delta;      // true if this vertex is delta-distributional

	// Payload: either endpoint or surface
	BDPTEndpointData<T> ei;  // Camera or Light
	BDPTSurfaceData<T>  si;  // Surface

	// ---------- Constructors ----------
	BDPTVertex() : type(BDPTVertexType::Camera), pdfFwd(T(0)), pdfRev(T(0)), delta(false) {
		std::memset(beta, 0, sizeof(beta));
		std::memset(&ei, 0, sizeof(ei));
		std::memset(&si, 0, sizeof(si));
	}

	// Factory: camera vertex
	static BDPTVertex MakeCamera(const T p[3], const T n[3], const T beta_in[3], T pdfDir) {
		BDPTVertex v;
		v.type = BDPTVertexType::Camera;
		v.beta[0]=beta_in[0]; v.beta[1]=beta_in[1]; v.beta[2]=beta_in[2];
		v.pdfFwd = pdfDir;
		v.ei.p[0]=p[0]; v.ei.p[1]=p[1]; v.ei.p[2]=p[2];
		v.ei.n[0]=n[0]; v.ei.n[1]=n[1]; v.ei.n[2]=n[2];
		v.ei.light_id = -1;
		return v;
	}

	// Factory: light endpoint from area light surface sample
	static BDPTVertex MakeLightSurface(const T p[3], const T n[3], const T Le[3],
										T pdf_pos, int light_id, bool is_delta) {
		BDPTVertex v;
		v.type = BDPTVertexType::Light;
		v.beta[0]=Le[0]; v.beta[1]=Le[1]; v.beta[2]=Le[2];
		v.pdfFwd = pdf_pos;
		v.ei.p[0]=p[0]; v.ei.p[1]=p[1]; v.ei.p[2]=p[2];
		v.ei.n[0]=n[0]; v.ei.n[1]=n[1]; v.ei.n[2]=n[2];
		v.ei.Le[0]=Le[0]; v.ei.Le[1]=Le[1]; v.ei.Le[2]=Le[2];
		v.ei.pdf_pos = pdf_pos;
		v.ei.is_delta = is_delta;
		v.ei.light_id = light_id;
		return v;
	}

	// Factory: light endpoint from environment light (infinite, non-surface)
	static BDPTVertex MakeLightInfinite(const T Le[3], T pdf_pos,
										 int light_id) {
		BDPTVertex v;
		v.type = BDPTVertexType::Light;
		v.beta[0]=Le[0]; v.beta[1]=Le[1]; v.beta[2]=Le[2];
		v.pdfFwd = pdf_pos;
		v.ei.Le[0]=Le[0]; v.ei.Le[1]=Le[1]; v.ei.Le[2]=Le[2];
		v.ei.pdf_pos = pdf_pos;
		v.ei.is_infinite = true;
		v.ei.light_id = light_id;
		std::memset(v.ei.p, 0, sizeof(v.ei.p));
		std::memset(v.ei.n, 0, sizeof(v.ei.n));
		return v;
	}

	// Factory: surface vertex
	static BDPTVertex MakeSurface(const BDPTHit<T>& hit,
								   const T beta_in[3], T pdfFwd_in,
								   const BDPTVertex& prev) {
		BDPTVertex v;
		v.type = BDPTVertexType::Surface;
		v.beta[0]=beta_in[0]; v.beta[1]=beta_in[1]; v.beta[2]=beta_in[2];
		// Convert directional pdf to area pdf
		T dp[3] = { hit.p[0]-prev.p()[0], hit.p[1]-prev.p()[1], hit.p[2]-prev.p()[2] };
		T dist2 = bdpt_detail::len2_3(dp);
		T pdfA = pdfFwd_in;
		if (dist2 > T(0)) {
			T inv_dist2 = T(1) / dist2;
			T dn[3] = { dp[0], dp[1], dp[2] };
			bdpt_detail::norm3(dn);
			pdfA *= bdpt_detail::absdot3(hit.geo_n, dn) * inv_dist2;
		}
		v.pdfFwd = pdfA;
		std::memcpy(v.si.p,        hit.p,        3*sizeof(T));
		std::memcpy(v.si.geo_n,    hit.geo_n,    3*sizeof(T));
		std::memcpy(v.si.shading_n,hit.shading_n,3*sizeof(T));
		std::memcpy(v.si.wo,       hit.wo,        3*sizeof(T));
		std::memcpy(v.si.area_Le,  hit.area_Le,   3*sizeof(T));
		v.si.is_delta_bsdf = hit.is_delta_bsdf;
		v.si.bsdf_id = hit.bsdf_id;
		return v;
	}

	// ---------- Accessors ----------

	const T* p() const {
		if (type == BDPTVertexType::Surface) return si.p;
		return ei.p;
	}
	const T* ng() const {
		if (type == BDPTVertexType::Surface) return si.geo_n;
		return ei.n;
	}
	const T* ns() const {
		if (type == BDPTVertexType::Surface) return si.shading_n;
		return ei.n;
	}

	bool IsOnSurface() const {
		const T* n = ng();
		return n[0]*n[0]+n[1]*n[1]+n[2]*n[2] > T(0.5);
	}

	bool IsLight() const {
		return type == BDPTVertexType::Light ||
			   (type == BDPTVertexType::Surface &&
				(si.area_Le[0]!=T(0)||si.area_Le[1]!=T(0)||si.area_Le[2]!=T(0)));
	}

	bool IsDeltaLight() const {
		return type == BDPTVertexType::Light && ei.is_delta;
	}

	bool IsInfiniteLight() const {
		return type == BDPTVertexType::Light && ei.is_infinite;
	}

	bool IsConnectible() const {
		switch (type) {
		case BDPTVertexType::Camera:  return true;
		case BDPTVertexType::Light:   return !ei.is_delta; // not delta-direction
		case BDPTVertexType::Surface: return !si.is_delta_bsdf;
		}
		return false;
	}

	// ---------- Le: emitted radiance towards direction of vertex 'next' ----------

	// scene provides InfiniteLightLe(dir, lambda) for env lights
	template<typename Scene>
	void Le(const BDPTVertex& next, const Scene& scene, T out[3]) const {
		if (!IsLight()) { out[0]=out[1]=out[2]=T(0); return; }
		T w[3] = { next.p()[0]-p()[0], next.p()[1]-p()[1], next.p()[2]-p()[2] };
		if (bdpt_detail::len2_3(w) == T(0)) { out[0]=out[1]=out[2]=T(0); return; }
		bdpt_detail::norm3(w);
		if (IsInfiniteLight()) {
			T neg_w[3] = { -w[0], -w[1], -w[2] };
			scene.InfiniteLightLe(neg_w, out);
		} else if (type == BDPTVertexType::Surface) {
			// area light on surface: L only if w is on same side as geo_n
			if (bdpt_detail::dot3(w, si.geo_n) > T(0)) {
				out[0]=si.area_Le[0]; out[1]=si.area_Le[1]; out[2]=si.area_Le[2];
			} else { out[0]=out[1]=out[2]=T(0); }
		} else {
			out[0]=ei.Le[0]; out[1]=ei.Le[1]; out[2]=ei.Le[2];
		}
	}

	// ---------- f: BSDF / phase evaluation toward 'next' ----------
	template<typename Scene>
	void f(const BDPTVertex& next, const Scene& scene, T out[3]) const {
		T wn[3] = { next.p()[0]-p()[0], next.p()[1]-p()[1], next.p()[2]-p()[2] };
		if (bdpt_detail::len2_3(wn) == T(0)) { out[0]=out[1]=out[2]=T(0); return; }
		bdpt_detail::norm3(wn);
		if (type == BDPTVertexType::Surface) {
			scene.BSDFf(si.bsdf_id, si.wo, wn, si.shading_n, out);
		} else {
			out[0]=out[1]=out[2]=T(0);
		}
	}

	// ---------- ConvertDensity: directional -> area pdf ----------
	T ConvertDensity(T pdf_dir, const BDPTVertex& next) const {
		if (next.IsInfiniteLight()) return pdf_dir;
		T w[3] = { next.p()[0]-p()[0], next.p()[1]-p()[1], next.p()[2]-p()[2] };
		T dist2 = bdpt_detail::len2_3(w);
		if (dist2 == T(0)) return T(0);
		T inv_dist2 = T(1) / dist2;
		T dn[3] = { w[0], w[1], w[2] };
		bdpt_detail::norm3(dn);
		if (next.IsOnSurface())
			pdf_dir *= bdpt_detail::absdot3(next.ng(), dn);
		return pdf_dir * inv_dist2;
	}

	// ---------- PDF: area pdf of sampling direction toward 'next' ----------
	template<typename Scene>
	T PDF(const BDPTVertex* prev, const BDPTVertex& next,
		  const Scene& scene) const {
		if (type == BDPTVertexType::Light)
			return PDFLight(next, scene);

		T wn[3] = { next.p()[0]-p()[0], next.p()[1]-p()[1], next.p()[2]-p()[2] };
		if (bdpt_detail::len2_3(wn) == T(0)) return T(0);
		bdpt_detail::norm3(wn);

		T wp[3] = {T(0),T(0),T(0)};
		if (prev) {
			wp[0]=prev->p()[0]-p()[0]; wp[1]=prev->p()[1]-p()[1]; wp[2]=prev->p()[2]-p()[2];
			if (bdpt_detail::len2_3(wp) == T(0)) return T(0);
			bdpt_detail::norm3(wp);
		}

		T pdf_dir = T(0);
		if (type == BDPTVertexType::Camera) {
			T dummy;
			scene.CameraPDFWe(p(), wn, dummy, pdf_dir);
		} else if (type == BDPTVertexType::Surface) {
			pdf_dir = scene.BSDFPdf(si.bsdf_id, wp, wn, si.shading_n);
		}
		return ConvertDensity(pdf_dir, next);
	}

	// ---------- PDFLight ----------
	template<typename Scene>
	T PDFLight(const BDPTVertex& v, const Scene& scene) const {
		T w[3] = { v.p()[0]-p()[0], v.p()[1]-p()[1], v.p()[2]-p()[2] };
		T inv_dist2 = T(1) / bdpt_detail::len2_3(w);
		T wn[3] = { w[0], w[1], w[2] };
		bdpt_detail::norm3(wn);
		T pdf;
		if (IsInfiniteLight()) {
			// planar sampling density for env lights
			T center[3]; T radius;
			scene.SceneBoundingSphere(center, radius);
			pdf = T(1) / (T(3.14159265358979323846) * radius * radius);
		} else {
			T pdf_pos, pdf_dir;
			int lid = (type==BDPTVertexType::Light) ? ei.light_id : si.bsdf_id; // area light
			scene.LightPDFLe(lid, IsOnSurface() ? p() : nullptr,
							 IsOnSurface() ? ng() : nullptr,
							 wn, pdf_pos, pdf_dir);
			pdf = pdf_dir * inv_dist2;
		}
		if (v.IsOnSurface())
			pdf *= bdpt_detail::absdot3(v.ng(), wn);
		return pdf;
	}

	// ---------- PDFLightOrigin ----------
	template<typename Scene>
	T PDFLightOrigin(const BDPTVertex& v, const Scene& scene) const {
		T w[3] = { v.p()[0]-p()[0], v.p()[1]-p()[1], v.p()[2]-p()[2] };
		if (bdpt_detail::len2_3(w) == T(0)) return T(0);
		bdpt_detail::norm3(w);
		if (IsInfiniteLight()) {
			return scene.InfiniteLightDensity(w);
		} else {
			int lid = (type==BDPTVertexType::Light) ? ei.light_id : si.bsdf_id;
			T pdf_pos, pdf_dir;
			scene.LightPDFLe(lid, IsOnSurface() ? p() : nullptr,
							 IsOnSurface() ? ng() : nullptr,
							 w, pdf_pos, pdf_dir);
			T pmf = scene.LightPMF(lid);
			return pdf_pos * pmf;
		}
	}
};

// ===========================================================================
// Section 4 -- G, InfiniteLightDensity, MISWeight
// ===========================================================================

// G: geometry coupling term (without transmittance -- use Unoccluded separately)
template<typename T>
T BDPTGeometryTerm(const BDPTVertex<T>& v0, const BDPTVertex<T>& v1) {
	T d[3] = { v0.p()[0]-v1.p()[0], v0.p()[1]-v1.p()[1], v0.p()[2]-v1.p()[2] };
	T dist2 = bdpt_detail::len2_3(d);
	if (dist2 == T(0)) return T(0);
	T g = T(1) / dist2;
	T dn[3] = { d[0], d[1], d[2] };
	bdpt_detail::norm3(dn);
	if (v0.IsOnSurface()) g *= bdpt_detail::absdot3(v0.ns(), dn);
	if (v1.IsOnSurface()) g *= bdpt_detail::absdot3(v1.ns(), dn);
	return g;
}

// MISWeight: balanced power-heuristic weight across all (s,t) strategies.
// Mirrors pbrt-v4 MISWeight() with ScopedAssignment pattern.
// splatScale: film area / pixel bounds area normalisation (default 1).
template<typename T, typename Scene>
T BDPTMISWeight(BDPTVertex<T>* lightVerts, BDPTVertex<T>* cameraVerts,
				BDPTVertex<T>& sampled, int s, int t,
				const Scene& scene, T splatScale = T(1)) {
	if (s + t == 2) return T(1);
	T sumRi = T(0);

	auto remap0 = [](T f) -> T { return f != T(0) ? f : T(1); };

	BDPTVertex<T>* qs      = s > 0 ? &lightVerts[s-1]  : nullptr;
	BDPTVertex<T>* pt      = t > 0 ? &cameraVerts[t-1] : nullptr;
	BDPTVertex<T>* qsMinus = s > 1 ? &lightVerts[s-2]  : nullptr;
	BDPTVertex<T>* ptMinus = t > 1 ? &cameraVerts[t-2] : nullptr;

	// Temporarily replace the connection vertices with the sampled vertex
	ScopedAssign<BDPTVertex<T>> a1;
	if (s == 1) a1 = ScopedAssign<BDPTVertex<T>>(qs, sampled);
	else if (t == 1) a1 = ScopedAssign<BDPTVertex<T>>(pt, sampled);

	// Mark connection vertices as non-degenerate for this weight computation
	ScopedAssign<bool> a2, a3;
	if (pt) a2 = ScopedAssign<bool>(&pt->delta, false);
	if (qs) a3 = ScopedAssign<bool>(&qs->delta, false);

	// Update reverse pdfs at t-1 and t-2
	ScopedAssign<T> a4;
	if (pt) {
		T revPdf = (s > 0)
			? qs->template PDF<Scene>(qsMinus, *pt, scene)
			: pt->template PDFLightOrigin<Scene>(*ptMinus, scene);
		a4 = ScopedAssign<T>(&pt->pdfRev, revPdf);
	}
	ScopedAssign<T> a5;
	if (ptMinus) {
		T revPdf = (s > 0)
			? pt->template PDF<Scene>(qs, *ptMinus, scene)
			: pt->template PDFLight<Scene>(*ptMinus, scene);
		a5 = ScopedAssign<T>(&ptMinus->pdfRev, revPdf);
	}

	// Update reverse pdfs at s-1 and s-2
	ScopedAssign<T> a6;
	if (qs) {
		a6 = ScopedAssign<T>(&qs->pdfRev,
							 pt->template PDF<Scene>(ptMinus, *qs, scene));
	}
	ScopedAssign<T> a7;
	if (qsMinus) {
		a7 = ScopedAssign<T>(&qsMinus->pdfRev,
							 qs->template PDF<Scene>(pt, *qsMinus, scene));
	}

	// Accumulate camera subpath ratios
	T ri = T(1);
	for (int i = t - 1; i > 0; --i) {
		ri *= remap0(cameraVerts[i].pdfRev) / remap0(cameraVerts[i].pdfFwd);
		if (i == 1) ri /= splatScale;
		if (!cameraVerts[i].delta && !cameraVerts[i-1].delta)
			sumRi += ri;
	}

	// Accumulate light subpath ratios
	ri = T(1);
	for (int i = s - 1; i >= 0; --i) {
		ri *= remap0(lightVerts[i].pdfRev) / remap0(lightVerts[i].pdfFwd);
		bool deltaLight = (i > 0) ? lightVerts[i-1].delta : lightVerts[0].IsDeltaLight();
		if (!lightVerts[i].delta && !deltaLight)
			sumRi += ri;
	}

	if (t == 1) sumRi /= splatScale;
	return T(1) / (T(1) + sumRi);
}

// ===========================================================================
// Section 5 -- RandomWalk, GenerateCameraSubpath, GenerateLightSubpath
// ===========================================================================

// RandomWalk: traces a ray starting from the given position/direction and
// fills in 'path' vertices.  Returns the number of vertices added.
// 'mode': true = camera (radiance) transport, false = light (importance)
//
// Scene additional required API for RandomWalk:
//   bool Intersect(org, dir, t_max, BDPTHit<T>&)
//   BSDFf(bsdf_id, wo, wi, shading_n, out)
//   BSDFSampleF(bsdf_id, wo, shading_n, u1, u2, BxDFSampleResult<T>&) -> bool
//   BSDFPdf(bsdf_id, wo, wi, shading_n) -> T
template<typename T, typename Scene>
int BDPTRandomWalk(const T ray_o[3], const T ray_d[3],
				   T beta_r, T beta_g, T beta_b,
				   T pdfFwd,
				   int maxDepth,
				   bool camera_mode, // true = radiance, false = importance
				   BDPTVertex<T>* path,
				   const Scene& scene) {
	if (maxDepth == 0) return 0;

	T org[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };
	T beta[3] = { beta_r, beta_g, beta_b };

	int bounces = 0;
	bool anyNonSpecular = false;

	while (true) {
		BDPTHit<T> hit{};
		bool found = scene.Intersect(org, dir, T(1e30), hit);

		if (!found) {
			// Escaped -- add environment light vertex on camera paths
			if (camera_mode) {
				// Create a virtual infinite-light vertex using the escape direction
				T Le[3] = {T(0), T(0), T(0)};
				scene.InfiniteLightLe(dir, Le);
				path[bounces] = BDPTVertex<T>::MakeLightInfinite(Le, pdfFwd, -1);
				path[bounces].beta[0] = beta[0];
				path[bounces].beta[1] = beta[1];
				path[bounces].beta[2] = beta[2];
				++bounces;
			}
			break;
		}

		// Skip medium boundaries (e.g. glass boundary without shading)
		if (hit.is_medium_boundary) {
			// advance ray through boundary
			org[0] = hit.p[0] + dir[0]*T(1e-4);
			org[1] = hit.p[1] + dir[1]*T(1e-4);
			org[2] = hit.p[2] + dir[2]*T(1e-4);
			continue;
		}

		// Create surface vertex
		BDPTVertex<T>& prev = (bounces == 0) ? path[-1] : path[bounces-1];
		// For the first bounce prev is the endpoint stored at path[-1]... 
		// Actually caller sets path to start after the endpoint, so prev is valid.
		BDPTVertex<T> vertex = BDPTVertex<T>::MakeSurface(hit, beta, pdfFwd, prev);
		path[bounces] = vertex;
		++bounces;

		if (bounces >= maxDepth) break;

		// Sample BSDF for next direction
		T wo[3] = { hit.wo[0], hit.wo[1], hit.wo[2] };
		T u1 = scene.RandFloat();
		T u2 = scene.RandFloat();

		// Use the scene to sample the BSDF
		T new_dir[3];
		T f_val[3];
		T pdf_bsdf;
		bool is_specular;
		bool sampled = scene.BSDFSampleF(hit.bsdf_id, wo, hit.shading_n, u1, u2,
										  new_dir, f_val, pdf_bsdf, is_specular);
		if (!sampled || pdf_bsdf == T(0)) break;

		T cos_theta = std::abs(
			new_dir[0]*hit.shading_n[0] + new_dir[1]*hit.shading_n[1] + new_dir[2]*hit.shading_n[2]);

		// Update throughput
		beta[0] *= f_val[0] * cos_theta / pdf_bsdf;
		beta[1] *= f_val[1] * cos_theta / pdf_bsdf;
		beta[2] *= f_val[2] * cos_theta / pdf_bsdf;

		if (beta[0] == T(0) && beta[1] == T(0) && beta[2] == T(0)) break;

		// Reverse pdf for the previous vertex
		T pdfRev = is_specular ? T(0) :
			scene.BSDFPdf(hit.bsdf_id, new_dir, wo, hit.shading_n);

		path[bounces-1].delta = is_specular;
		if (is_specular) {
			pdfFwd = T(0);
			pdfRev = T(0);
		} else {
			pdfFwd = pdf_bsdf;
			anyNonSpecular = true;
		}

		// Convert reverse pdf to area measure and store on previous vertex
		if (bounces >= 2) {
			path[bounces-2].pdfRev =
				path[bounces-1].ConvertDensity(pdfRev, path[bounces-2]);
		}

		// Spawn next ray
		org[0]=hit.p[0]; org[1]=hit.p[1]; org[2]=hit.p[2];
		dir[0]=new_dir[0]; dir[1]=new_dir[1]; dir[2]=new_dir[2];
	}
	return bounces;
}

// GenerateCameraSubpath: path[0] = camera vertex, path[1..] = surface vertices
template<typename T, typename Scene>
int BDPTGenerateCameraSubpath(const T cam_p[3], const T cam_n[3],
							   const T ray_d[3],
							   int maxDepth,
							   const Scene& scene,
							   BDPTVertex<T>* path) {
	if (maxDepth == 0) return 0;

	T beta[3] = { T(1), T(1), T(1) };
	T pdfPos, pdfDir;
	scene.CameraPDFWe(cam_p, ray_d, pdfPos, pdfDir);
	path[0] = BDPTVertex<T>::MakeCamera(cam_p, cam_n, beta, pdfDir);

	int nWalk = BDPTRandomWalk(cam_p, ray_d,
							   T(1), T(1), T(1), pdfDir,
							   maxDepth - 1, true, path + 1, scene);
	return nWalk + 1;
}

// GenerateLightSubpath: path[0] = light vertex, path[1..] = surface vertices
template<typename T, typename Scene>
int BDPTGenerateLightSubpath(int maxDepth, const Scene& scene,
							  BDPTVertex<T>* path) {
	if (maxDepth == 0) return 0;

	BDPTLightLeSample<T> les{};
	T u1d  = scene.RandFloat();
	T u2a[2] = { scene.RandFloat(), scene.RandFloat() };
	T u2b[2] = { scene.RandFloat(), scene.RandFloat() };
	if (!scene.SampleLightLe(u1d, u2a, u2b, les)) return 0;
	if (les.pdf_pos == T(0) || les.pdf_dir == T(0)) return 0;

	T p_l = les.pdf_pos; // combined: light choice PMF * position pdf

	if (les.is_on_surface) {
		path[0] = BDPTVertex<T>::MakeLightSurface(les.p_on_light, les.n_on_light,
												   les.L, p_l, les.light_id,
												   les.is_delta_dir);
	} else {
		path[0] = BDPTVertex<T>::MakeLightInfinite(les.L, p_l, les.light_id);
	}

	T cos_theta = les.abs_cos_theta;
	T beta[3] = {
		les.L[0] * cos_theta / (p_l * les.pdf_dir),
		les.L[1] * cos_theta / (p_l * les.pdf_dir),
		les.L[2] * cos_theta / (p_l * les.pdf_dir),
	};

	int nWalk = BDPTRandomWalk(les.ray_o, les.ray_d,
							   beta[0], beta[1], beta[2],
							   les.pdf_dir, maxDepth - 1,
							   false, path + 1, scene);

	// Fix pdfFwd for infinite-area light: first surface vertex uses positional pdf
	if (path[0].IsInfiniteLight() && nWalk > 0) {
		path[1].pdfFwd = les.pdf_pos;
		if (path[1].IsOnSurface()) {
			T dn[3] = { les.ray_d[0], les.ray_d[1], les.ray_d[2] };
			path[1].pdfFwd *= bdpt_detail::absdot3(path[1].ng(), dn);
		}
		// pdfFwd for the infinite-light vertex uses InfiniteLightDensity
		path[0].pdfFwd = scene.InfiniteLightDensity(les.ray_d);
	}
	return nWalk + 1;
}

// ===========================================================================
// Section 6 -- ConnectBDPT
// ===========================================================================

// ConnectBDPT: evaluate the contribution from connecting lightVerts[0..s-1]
// to cameraVerts[0..t-1] and compute the MIS weight.
// Returns the spectral contribution in L[3].
// For t==1 strategies (light tracing), also returns a raster position.
template<typename T, typename Scene>
void BDPTConnect(BDPTVertex<T>* lightVerts, BDPTVertex<T>* cameraVerts,
				 int s, int t, const Scene& scene,
				 T L[3], T* pRaster = nullptr, T* misWeightOut = nullptr) {
	L[0] = L[1] = L[2] = T(0);
	if (pRaster) { pRaster[0] = pRaster[1] = T(0); }

	BDPTVertex<T> sampled;

	// Strategy s=0: camera path reaches a light source directly
	if (s == 0) {
		const BDPTVertex<T>& pt = cameraVerts[t-1];
		if (pt.IsLight()) {
			BDPTVertex<T> dummy;
			pt.template Le<Scene>(cameraVerts[t-2], scene, L);
			// weight beta
			L[0] *= pt.beta[0];
			L[1] *= pt.beta[1];
			L[2] *= pt.beta[2];
		}
	}
	// Strategy s=1: connect a light sample to the camera subpath endpoint
	else if (s == 1) {
		const BDPTVertex<T>& pt = cameraVerts[t-1];
		if (pt.IsConnectible()) {
			BDPTLightSample<T> ls{};
			T uLight = scene.RandFloat();
			T u2[2]  = { scene.RandFloat(), scene.RandFloat() };
			if (scene.SampleLight(uLight, pt.p(), ls) &&
				ls.pdf > T(0) && (ls.L[0]||ls.L[1]||ls.L[2])) {
				// Create light vertex
				T Le_scaled[3] = { ls.L[0]/(ls.pdf), ls.L[1]/(ls.pdf), ls.L[2]/(ls.pdf) };
				sampled = BDPTVertex<T>::MakeLightSurface(ls.p_light, ls.n_light,
														  ls.L, ls.pdf, ls.light_id, ls.is_delta);
				sampled.beta[0]=Le_scaled[0]; sampled.beta[1]=Le_scaled[1]; sampled.beta[2]=Le_scaled[2];
				sampled.pdfFwd = sampled.template PDFLightOrigin<Scene>(pt, scene);

				// Evaluate f at pt toward sampled
				T f_pt[3];
				pt.template f<Scene>(sampled, scene, f_pt);

				T cos_pt = bdpt_detail::absdot3(ls.wi, pt.ns());

				L[0] = pt.beta[0] * f_pt[0] * Le_scaled[0] * cos_pt;
				L[1] = pt.beta[1] * f_pt[1] * Le_scaled[1] * cos_pt;
				L[2] = pt.beta[2] * f_pt[2] * Le_scaled[2] * cos_pt;

				if ((L[0]||L[1]||L[2]) &&
					!scene.Unoccluded(pt.p(), ls.p_light)) {
					L[0] = L[1] = L[2] = T(0);
				}
			}
		}
	}
	// Strategy t=1: connect a light subpath vertex to the camera
	else if (t == 1) {
		// Light tracing: importance-sample the camera
		const BDPTVertex<T>& qs = lightVerts[s-1];
		if (qs.IsConnectible()) {
			T wi[3], importance, raster[2];
			T pdf_we;
			if (scene.CameraSampleWi(qs.p(), nullptr, wi, pdf_we, importance, raster) &&
				pdf_we > T(0) && importance > T(0)) {
				T cam_n[3]; T unused;
				scene.CameraPDFWe(qs.p(), wi, unused, pdf_we);
				T Le_cam[3] = { importance/pdf_we, importance/pdf_we, importance/pdf_we };
				sampled = BDPTVertex<T>::MakeCamera(qs.p(), cam_n, Le_cam, pdf_we);

				T f_qs[3];
				qs.template f<Scene>(sampled, scene, f_qs);

				T cos_qs = bdpt_detail::absdot3(wi, qs.ns());
				L[0] = qs.beta[0] * f_qs[0] * Le_cam[0] * cos_qs;
				L[1] = qs.beta[1] * f_qs[1] * Le_cam[1] * cos_qs;
				L[2] = qs.beta[2] * f_qs[2] * Le_cam[2] * cos_qs;

				if (pRaster) { pRaster[0]=raster[0]; pRaster[1]=raster[1]; }

				if ((L[0]||L[1]||L[2]) &&
					!scene.Unoccluded(qs.p(), sampled.p())) {
					L[0]=L[1]=L[2]=T(0);
				}
			}
		}
	}
	// General case: connect two subpaths
	else {
		const BDPTVertex<T>& qs = lightVerts[s-1];
		const BDPTVertex<T>& pt = cameraVerts[t-1];
		if (qs.IsConnectible() && pt.IsConnectible()) {
			T f_qs[3], f_pt[3];
			qs.template f<Scene>(pt, scene, f_qs);
			pt.template f<Scene>(qs, scene, f_pt);

			T g = BDPTGeometryTerm(qs, pt);

			L[0] = qs.beta[0] * f_qs[0] * f_pt[0] * pt.beta[0] * g;
			L[1] = qs.beta[1] * f_qs[1] * f_pt[1] * pt.beta[1] * g;
			L[2] = qs.beta[2] * f_qs[2] * f_pt[2] * pt.beta[2] * g;

			if ((L[0]||L[1]||L[2]) &&
				!scene.Unoccluded(qs.p(), pt.p())) {
				L[0]=L[1]=L[2]=T(0);
			}
		}
	}

	bool nonzero = L[0]||L[1]||L[2];
	T misWeight = nonzero
		? BDPTMISWeight(lightVerts, cameraVerts, sampled, s, t, scene)
		: T(0);
	L[0] *= misWeight; L[1] *= misWeight; L[2] *= misWeight;

	if (misWeightOut) *misWeightOut = misWeight;
}

// ===========================================================================
// Section 7 -- BDPTLi: top-level Li estimator
// ===========================================================================

// BDPTLi: estimate incident radiance for a camera ray using BDPT.
// path memory: caller allocates at least (maxDepth+2) + (maxDepth+1) BDPTVertex.
template<typename T, typename Scene>
void BDPTLi(const T cam_p[3], const T cam_n[3], const T ray_d[3],
			 int maxDepth, const Scene& scene,
			 BDPTVertex<T>* cameraVerts, // (maxDepth+2) elements
			 BDPTVertex<T>* lightVerts,  // (maxDepth+1) elements
			 T L_out[3]) {
	L_out[0] = L_out[1] = L_out[2] = T(0);

	int nCamera = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d,
											 maxDepth + 2, scene, cameraVerts);
	int nLight  = BDPTGenerateLightSubpath(maxDepth + 1, scene, lightVerts);

	for (int t = 1; t <= nCamera; ++t) {
		for (int s = 0; s <= nLight; ++s) {
			int depth = t + s - 2;
			if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;

			T Lpath[3];
			BDPTConnect(lightVerts, cameraVerts, s, t, scene, Lpath);

			if (t != 1) {
				L_out[0] += Lpath[0];
				L_out[1] += Lpath[1];
				L_out[2] += Lpath[2];
			}
			// t==1 (light tracing) contributions go to film splat,
			// not returned here. Callers needing film splat should call
			// BDPTConnect directly with a pRaster output.
		}
	}
}
