// optix_programs.cu -- OptiX device programs (top-level).
// Split into logical units; this file includes them all.

#include "../../src/shared/cpu_gpu.h"  // kMaxMediumBoundaryCrossings, used by optix_raygen.h
#include "optix_disk_cylinder_helpers.h"
#include "optix_device_helpers.h"
#include "optix_intersection_sphere.h"
#include "optix_intersection_quad.h"
#include "optix_intersection_bilinear_patch.h"
#include "optix_intersection_disk_cylinder.h"
#include "optix_intersection_triangle.h"
#include "optix_anyhit_shadow.h"
#include "optix_miss.h"
// RAY_TYPE_PROBE closest-hit/miss programs (recursive backend only, Phase 1
// BSSRDF) - see optix_types.h's RAY_TYPE_PROBE comment. After the
// intersection headers above: reuses their __intersection__<type>()
// programs unchanged (only closest-hit differs per ray type), and needs
// TriangleData/SphereData/etc. already visible via optix_device_helpers.h's
// own includes.
#include "optix_probe_hit.h"
#include "optix_raygen.h"

