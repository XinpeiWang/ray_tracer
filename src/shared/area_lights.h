#pragma once
// ---------------------------------------------------------------------------
// area_lights.h -- Shared CPU/GPU area light primitives.
//
// Ports pbrt-v4 src/pbrt/lights.h (DiffuseAreaLight, UniformInfiniteLight)
// to the local template/scalar style (no Transform/Allocator/SampledSpectrum).
//
// Provides two light types:
//
//   DiffuseAreaLight<T, ShapeT>
//     Diffusely emitting surface attached to any shape from shapes.h.
//     Emits constant RGB radiance Le (optionally two-sided).
//     Directly wraps ShapeT::sample_from / pdf_from for importance sampling.
//     Reference: pbrt-v4 DiffuseAreaLight (lights.h / lights.cpp)
//
//   UniformInfiniteLight<T>
//     Constant-radiance environment (sky dome). Samples uniformly over the
//     sphere; supports allowIncompletePDF to skip the light in BSDF-sampling
//     MIS strategies that handle the infinite case separately.
//     Reference: pbrt-v4 UniformInfiniteLight (lights.h / lights.cpp)
//
// Supporting data types:
//   LightSampleContext<T>  -- shading point, geometric + shading normal
//   LightLiSample<T>       -- result of SampleLi: (Lr,Lg,Lb), wi, pdf,
//                             sampled point on light
//
// Design rules (same as shapes.h / punctual_lights.h):
//   - Header-only, no heap allocation, no virtual functions
//   - Template parameter T: double on CPU, float on GPU
//   - CPU_GPU macro: __host__ __device__ under NVCC, inline otherwise
//
// Dependencies: shapes.h (SamplingContext, ShapeSample),
//               sampling_sphere_cone.h (SampleUniformSphere, UniformSpherePDF)
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "shapes.h"              // SamplingContext<T>, ShapeSample<T>
#include "sampling_sphere_cone.h" // SampleUniformSphere, UniformSpherePDF

#include <cmath>
#include <optional>
#include <limits>

// ===========================================================================
// LightSampleContext<T>
// ===========================================================================
// Shading point passed to SampleLi / PDF_Li.
// Mirrors pbrt-v4 LightSampleContext (lights.h).
//
// Fields:
//   (px,py,pz)   -- world-space shading point
//   (nx,ny,nz)   -- geometric surface normal (zero for medium interactions)
//   (nsx,nsy,nsz)-- shading normal (may differ from geometric if bump-mapped)
// ===========================================================================

template<typename T>
struct LightSampleContext {
	T px, py, pz;     // shading point position
	T nx, ny, nz;     // geometric normal (0 if from a medium)
	T nsx, nsy, nsz;  // shading normal (0 if from a medium)

	// Construct from position only (medium interaction)
	CPU_GPU static LightSampleContext from_point(T x, T y, T z) {
		return LightSampleContext{x,y,z, T(0),T(0),T(0), T(0),T(0),T(0)};
	}

	// Construct from surface interaction
	CPU_GPU static LightSampleContext from_surface(T x, T y, T z,
													T gnx, T gny, T gnz,
													T snx, T sny, T snz) {
		return LightSampleContext{x,y,z, gnx,gny,gnz, snx,sny,snz};
	}

	// Convert to SamplingContext for shapes.h
	CPU_GPU SamplingContext<T> to_sampling_ctx() const {
		return SamplingContext<T>{px, py, pz, nx, ny, nz};
	}
};


// ===========================================================================
// LightLiSample<T>
// ===========================================================================
// Result of SampleLi.
// Mirrors pbrt-v4 LightLiSample (lights.h).
//
// Fields:
//   (Lr,Lg,Lb)       -- incident radiance RGB at the shading point
//   (wi_x,wi_y,wi_z) -- unit direction toward the light sample
//   pdf              -- solid-angle PDF of the sampled direction
//   (lx,ly,lz)       -- sampled point on the light surface (or on the
//                       virtual sphere at sceneRadius for infinite lights)
// ===========================================================================

template<typename T>
struct LightLiSample {
	T Lr, Lg, Lb;        // incident radiance
	T wi_x, wi_y, wi_z;  // direction toward light (unit vector)
	T pdf;               // solid-angle PDF
	T lx, ly, lz;        // sampled point on light (world space)
};


// ===========================================================================
// DiffuseAreaLight<T, ShapeT>
// ===========================================================================
//
// Uniformly emitting diffuse area light wrapping a shape from shapes.h.
// Uses the shape's solid-angle sampling (sample_from / pdf_from) for
// importance sampling, exactly as pbrt-v4 DiffuseAreaLight::SampleLi does.
//
// Template parameters:
//   T      -- scalar type (double on CPU, float on GPU)
//   ShapeT -- one of SphereShape<T>, DiskShape<T>, TriangleShape<T>
//
// Usage:
//   auto shape = SphereShape<double>::make(0,0,0, 1.0);
//   DiffuseAreaLight<double, SphereShape<double>> light;
//   light.shape    = shape;
//   light.Lr = light.Lg = light.Lb = 10.0;   // RGB radiance
//   light.scale    = 1.0;
//   light.two_sided = false;
//
// Reference: pbrt-v4 DiffuseAreaLight (lights.h / lights.cpp)
// ===========================================================================

template<typename T, typename ShapeT>
struct DiffuseAreaLight {
	ShapeT shape;           // the emitting shape
	T Lr, Lg, Lb;           // emitted radiance (constant RGB)
	T scale;                // additional scale factor
	bool two_sided;         // emit from both sides of the surface?

	// -----------------------------------------------------------------------
	// L() -- Emitted radiance at a surface point toward direction w
	//
	// Returns (Lr,Lg,Lb)*scale if the point is on the emitting side,
	// (0,0,0) otherwise.
	// Reference: pbrt-v4 DiffuseAreaLight::L()
	// -----------------------------------------------------------------------
	CPU_GPU void L(T nx, T ny, T nz,       // surface normal at hit point
				   T wx, T wy, T wz,       // outgoing direction (toward viewer)
				   T& out_r, T& out_g, T& out_b) const
	{
		// Check for emitting side: n·w > 0 (or two_sided)
		T cos_theta = nx*wx + ny*wy + nz*wz;
		if (!two_sided && cos_theta < T(0)) {
			out_r = out_g = out_b = T(0);
			return;
		}
		out_r = Lr * scale;
		out_g = Lg * scale;
		out_b = Lb * scale;
	}

	// -----------------------------------------------------------------------
	// sample_li() -- Importance-sample an incident direction from ctx
	//
	// Delegates to the shape's solid-angle sampler (sample_from).
	// Returns empty optional if the sample is degenerate.
	// Reference: pbrt-v4 DiffuseAreaLight::SampleLi()
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<LightLiSample<T>>
	sample_li(const LightSampleContext<T>& ctx, T u0, T u1) const
	{
		SamplingContext<T> shape_ctx = ctx.to_sampling_ctx();
		ShapeSample<T> ss = shape.sample_from(shape_ctx, u0, u1);

		if (ss.pdf == T(0)) return {};

		// Direction from shading point toward sampled light point
		T wix = ss.px - ctx.px;
		T wiy = ss.py - ctx.py;
		T wiz = ss.pz - ctx.pz;
		T wi_len2 = wix*wix + wiy*wiy + wiz*wiz;
		if (wi_len2 == T(0)) return {};
		T wi_len = std::sqrt(wi_len2);
		wix /= wi_len; wiy /= wi_len; wiz /= wi_len;

		// Evaluate emitted radiance toward -wi (viewer side)
		T out_r, out_g, out_b;
		L(ss.nx, ss.ny, ss.nz, -wix, -wiy, -wiz, out_r, out_g, out_b);
		if (out_r == T(0) && out_g == T(0) && out_b == T(0)) return {};

		LightLiSample<T> result;
		result.Lr = out_r; result.Lg = out_g; result.Lb = out_b;
		result.wi_x = wix; result.wi_y = wiy; result.wi_z = wiz;
		result.pdf  = ss.pdf;
		result.lx   = ss.px; result.ly = ss.py; result.lz = ss.pz;
		return result;
	}

	// -----------------------------------------------------------------------
	// pdf_li() -- Solid-angle PDF for a given direction wi from ctx
	//
	// Delegates to the shape's pdf_from.
	// Reference: pbrt-v4 DiffuseAreaLight::PDF_Li()
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_li(const LightSampleContext<T>& ctx,
					 T wi_x, T wi_y, T wi_z) const
	{
		SamplingContext<T> shape_ctx = ctx.to_sampling_ctx();
		return shape.pdf_from(shape_ctx, wi_x, wi_y, wi_z);
	}

	// -----------------------------------------------------------------------
	// power() -- Approximate total emitted power (used by light samplers)
	//
	// phi = pi * (1 or 2) * area * Le
	// Reference: pbrt-v4 DiffuseAreaLight::Phi()
	//   = Pi * (twoSided ? 2 : 1) * area * L
	// -----------------------------------------------------------------------
	CPU_GPU void power(T& pr, T& pg, T& pb) const {
		const T pi = T(3.14159265358979323846);
		T factor = pi * (two_sided ? T(2) : T(1)) * shape.area();
		pr = factor * Lr * scale;
		pg = factor * Lg * scale;
		pb = factor * Lb * scale;
	}
};


// ===========================================================================
// UniformInfiniteLight<T>
// ===========================================================================
//
// Constant-radiance environment light (uniform sky dome).
// Samples directions uniformly over the full sphere.
//
// Usage:
//   UniformInfiniteLight<double> sky;
//   sky.Lr = sky.Lg = sky.Lb = 1.0;
//   sky.scale        = 1.0;
//   sky.scene_center_x = sky.scene_center_y = sky.scene_center_z = 0.0;
//   sky.scene_radius = 100.0;
//
// Reference: pbrt-v4 UniformInfiniteLight (lights.h / lights.cpp)
// ===========================================================================

template<typename T>
struct UniformInfiniteLight {
	T Lr, Lg, Lb;              // emitted radiance (constant RGB)
	T scale;                   // additional scale factor
	T scene_center_x, scene_center_y, scene_center_z;
	T scene_radius;            // bounding sphere radius of scene

	// -----------------------------------------------------------------------
	// Le() -- Radiance for a ray that escapes the scene
	// Reference: pbrt-v4 UniformInfiniteLight::Le()
	// -----------------------------------------------------------------------
	CPU_GPU void Le(T& out_r, T& out_g, T& out_b) const {
		out_r = Lr * scale;
		out_g = Lg * scale;
		out_b = Lb * scale;
	}

	// -----------------------------------------------------------------------
	// sample_li() -- Sample an incident direction from the environment
	//
	// When allowIncompletePDF is true, returns empty (the light is handled
	// by the BSDF-sampling path instead).
	// Reference: pbrt-v4 UniformInfiniteLight::SampleLi()
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<LightLiSample<T>>
	sample_li(const LightSampleContext<T>& ctx,
			  T u0, T u1,
			  bool allow_incomplete_pdf = false) const
	{
		if (allow_incomplete_pdf) return {};

		// Sample a uniform direction on the sphere
		T wi_x, wi_y, wi_z;
		SampleUniformSphere(u0, u1, wi_x, wi_y, wi_z);
		T pdf = UniformSpherePDF<T>();

		// Virtual light point: project outward to scene bounding sphere
		T lx = ctx.px + wi_x * (T(2) * scene_radius);
		T ly = ctx.py + wi_y * (T(2) * scene_radius);
		T lz = ctx.pz + wi_z * (T(2) * scene_radius);

		LightLiSample<T> result;
		result.Lr = Lr * scale;
		result.Lg = Lg * scale;
		result.Lb = Lb * scale;
		result.wi_x = wi_x; result.wi_y = wi_y; result.wi_z = wi_z;
		result.pdf  = pdf;
		result.lx   = lx; result.ly = ly; result.lz = lz;
		return result;
	}

	// -----------------------------------------------------------------------
	// pdf_li() -- PDF for a given direction wi
	// Reference: pbrt-v4 UniformInfiniteLight::PDF_Li()
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_li(T /*wi_x*/, T /*wi_y*/, T /*wi_z*/,
					 bool allow_incomplete_pdf = false) const
	{
		if (allow_incomplete_pdf) return T(0);
		return UniformSpherePDF<T>();
	}

	// -----------------------------------------------------------------------
	// power() -- Approximate total emitted power
	// phi = 4 * pi^2 * r^2 * Le
	// Reference: pbrt-v4 UniformInfiniteLight::Phi()
	//   = 4 * Pi * Pi * Sqr(sceneRadius) * scale * Lemit->Sample(lambda)
	// -----------------------------------------------------------------------
	CPU_GPU void power(T& pr, T& pg, T& pb) const {
		const T pi = T(3.14159265358979323846);
		T factor = T(4) * pi * pi * scene_radius * scene_radius * scale;
		pr = factor * Lr;
		pg = factor * Lg;
		pb = factor * Lb;
	}
};
