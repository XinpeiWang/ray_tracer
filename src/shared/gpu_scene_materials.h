#pragma once
// gpu_scene_materials.h -- the material decisions the CPU builder and both GPU scene loaders make the same way. GPU scene stage 2 (docs/GPU_SCENE_COMMON.md).
//
// Less is shared here than the proposal first hoped, on purpose. Looked at line by line, most of what the loaders do with a material is either packing (where the bytes go, per
// backend) or a decision the backends make differently by design: mix materials (OptiX picks per hit, Metal per triangle at load time) and a chromatic medium (OptiX carries
// three extinctions, Metal reduces to one scalar plus a flag, with its own thresholds). Those stay per backend. What is left, and was copied three times, is below.
// Standard library only; templated on the float type so the CPU (double) and the GPU loaders (float) each keep their own arithmetic.

#include <cmath>

namespace gpu_scene_materials {

// A conductor given only a "reflectance" (no eta/k, no named metal): eta = 1 and k solved from the normal-incidence reflectance r = ((eta-1)^2 + k^2) / ((eta+1)^2 + k^2), which at
// eta = 1 reduces to k = 2 sqrt(r) / sqrt(1 - r). r is clamped to [0, 0.9999] and the denominator floored at 1e-4 so a perfect mirror stays finite. Used for plain Conductor and
// for CoatedConductor's base, on the CPU, OptiX and Metal. (The device-side conductor shader in materials.h solves the same relation per hit with its own clamp; that one is
// not this function.)
template <class T>
inline T reflectanceToConductorK(T r) {
	r = r < T(0) ? T(0) : (r > T(0.9999) ? T(0.9999) : r);
	const T d = T(1) - r;
	return T(2) * std::sqrt(r) / std::sqrt(d > T(1e-4) ? d : T(1e-4));
}

}  // namespace gpu_scene_materials
