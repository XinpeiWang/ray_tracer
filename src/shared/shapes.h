#pragma once
// ---------------------------------------------------------------------------
// shapes.h -- Shared CPU/GPU shape intersection and sampling primitives.
//
// Ports pbrt-v4 src/pbrt/shapes.h/.cpp for three fundamental shapes:
//   SphereShape<T>   -- full sphere (or z-clipped) with quadric intersection
//   DiskShape<T>     -- flat disk aligned to XY plane (z = height)
//   TriangleShape<T> -- Watertight triangle (Woop/Igehy style, pbrt-v4)
//
// Each shape exposes:
//   intersect(ray_ox,oy,oz, rd_dx,dy,dz, tMin, tMax)
//       -> optional<ShapeHit<T>>
//   area()                      -> T
//   sample(u0, u1)              -> ShapeSample<T>   (area-uniform)
//   pdf_area()                  -> T   (= 1/area())
//   sample_from(ctx, u0, u1)    -> ShapeSample<T>   (solid-angle from point)
//   pdf_from(ctx, wi_dx,wy,wz)  -> T   (solid-angle PDF)
//
// SamplingContext<T> holds the shading point used for solid-angle sampling.
//
// Design rules (same as bxdfs.h / sampling.h):
//   - No virtual functions, no heap allocation
//   - Template parameter T: double on CPU, float on GPU
//   - CPU_GPU macro: __host__ __device__ under NVCC, inline otherwise
//   - Coordinate convention: right-handed, y-up (matches local codebase)
//
// Reference: pbrt-v4 src/pbrt/shapes.h, shapes.cpp
//            Wald et al. 2014 "Watertight Ray/Triangle Intersection"
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"
#include "scalar_math.h"

#include "sampling_sphere_cone.h"   // SampleUniformSphere, SampleUniformDiskConcentric,
								// SampleUniformCone, UniformConePDF, etc.
#include "shading_frame.h"          // ShadingFrame<T>
#include "splines.h"                // CubicBezierControlPoints, EvaluateCubicBezierD, SubdivideCubicBezier
#include "interval_vec.h"           // Point3fi, OffsetRayOrigin, SpawnRay

#include <cmath>
#include <optional>
#include <algorithm>
#include <limits>

#include "shape_common.h"
#include "shape_sphere.h"
#include "shape_disk.h"
#include "shape_cylinder.h"
#include "shape_cone.h"
#include "shape_paraboloid.h"
#include "shape_triangle.h"
