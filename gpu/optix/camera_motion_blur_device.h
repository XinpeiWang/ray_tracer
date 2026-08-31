#pragma once
// camera_motion_blur_device.h -- shared per-ray camera-motion-blur interpolation
//
// Used by both GPU backends (optix_device_helpers.h for the recursive
// backend, wavefront_kernels.cu for the wavefront backend). Unlike most
// device helpers in this codebase, these two functions take no dependency
// on either backend's own __constant__ state - everything they need comes
// in as plain parameters - so there is no reason to duplicate them per-
// backend the way generate_primary_ray()/wf_generate_primary_ray()
// themselves must be (those two genuinely differ: the recursive backend
// reads its GpuCameraParams off the file's own __constant__ params global,
// while the wavefront kernel receives it as an explicit parameter, having
// no access to that global at all).
//
// Splits what used to be one "slerp + apply" function into two steps so a
// caller applying the SAME interpolated rotation to two different vectors
// (a primary ray's local origin and local direction) only pays for the
// slerp + quaternion-to-matrix conversion once, not once per vector.

struct GpuAnimRotMat {
	float m[3][3];  // pbrt-v4's left-handed "mInv" layout - see gpu_camera_anim_apply()
};

// Interpolates two shortest-arc-aligned rotation quaternions (x,y,z,w) at
// shutter fraction dt in [0,1] and converts the result to a 3x3 rotation
// matrix. Float-only mirror of AnimatedTransform::Interpolate's rotation
// term + at_quat_to_mat() (src/shared/animated_transform.h) - see
// build_gpu_animated_camera_params() (scene_builder.cpp) for where animR0/
// animR1 are decomposed, host-side, by the real double-precision class.
__device__ __forceinline__ GpuAnimRotMat gpu_camera_anim_rotation(
	const float4& animR0, const float4& animR1, float dt
) {
	float cosTheta = animR0.x*animR1.x + animR0.y*animR1.y + animR0.z*animR1.z + animR0.w*animR1.w;
	float qx, qy, qz, qw;
	if (cosTheta > 0.9995f) {
		qx = (1.0f-dt)*animR0.x + dt*animR1.x;
		qy = (1.0f-dt)*animR0.y + dt*animR1.y;
		qz = (1.0f-dt)*animR0.z + dt*animR1.z;
		qw = (1.0f-dt)*animR0.w + dt*animR1.w;
		float len = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);
		if (len > 0.0f) { qx /= len; qy /= len; qz /= len; qw /= len; }
	} else {
		float theta = acosf(fmaxf(-1.0f, fminf(1.0f, cosTheta)));
		float thetaP = theta * dt;
		float tx = animR1.x - cosTheta*animR0.x;
		float ty = animR1.y - cosTheta*animR0.y;
		float tz = animR1.z - cosTheta*animR0.z;
		float tw = animR1.w - cosTheta*animR0.w;
		float tlen = sqrtf(tx*tx + ty*ty + tz*tz + tw*tw);
		if (tlen > 0.0f) { tx /= tlen; ty /= tlen; tz /= tlen; tw /= tlen; }
		float cp = cosf(thetaP), sp = sinf(thetaP);
		qx = cp*animR0.x + sp*tx;
		qy = cp*animR0.y + sp*ty;
		qz = cp*animR0.z + sp*tz;
		qw = cp*animR0.w + sp*tw;
	}

	// Quaternion -> 3x3 rotation, pbrt-v4's left-handed convention (forward
	// matrix = transpose of the "standard" mInv below) - mirrors
	// at_quat_to_mat()'s exact derivation (animated_transform.h).
	float xx=qx*qx, yy=qy*qy, zz=qz*qz;
	float xy=qx*qy, xz=qx*qz, yz=qy*qz;
	float wx=qx*qw, wy=qy*qw, wz=qz*qw;
	GpuAnimRotMat rot;
	rot.m[0][0]=1.0f-2.0f*(yy+zz); rot.m[0][1]=2.0f*(xy+wz);      rot.m[0][2]=2.0f*(xz-wy);
	rot.m[1][0]=2.0f*(xy-wz);      rot.m[1][1]=1.0f-2.0f*(xx+zz); rot.m[1][2]=2.0f*(yz+wx);
	rot.m[2][0]=2.0f*(xz+wy);      rot.m[2][1]=2.0f*(yz-wx);      rot.m[2][2]=1.0f-2.0f*(xx+yy);
	return rot;
}

// Applies an already-interpolated rotation (from gpu_camera_anim_rotation())
// to a local-space point (isPoint=true, lerped translation between animT0/
// animT1 added) or vector (isPoint=false, translation-free). Splitting the
// matrix build from its application is what lets a caller transforming both
// a ray's origin (a point) and direction (a vector) with the SAME
// interpolated rotation pay for the slerp/quaternion-to-matrix conversion
// only once, not once per vector - see this header's own top comment.
//
// Callers transforming a direction (isPoint=false) still need to normalize
// the result themselves: a rotation preserves vector length, but the LOCAL
// direction fed in here is not unit length to begin with (it's `pixel_sample
// - lens_origin`, not a normalized ray direction) - this function only
// rotates it.
__device__ __forceinline__ float3 gpu_camera_anim_apply(
	const GpuAnimRotMat& rot, const float3& v, bool isPoint,
	const float3& animT0, const float3& animT1, float dt
) {
	float3 out;
	out.x = rot.m[0][0]*v.x + rot.m[1][0]*v.y + rot.m[2][0]*v.z;
	out.y = rot.m[0][1]*v.x + rot.m[1][1]*v.y + rot.m[2][1]*v.z;
	out.z = rot.m[0][2]*v.x + rot.m[1][2]*v.y + rot.m[2][2]*v.z;
	if (isPoint) {
		out.x += (1.0f-dt)*animT0.x + dt*animT1.x;
		out.y += (1.0f-dt)*animT0.y + dt*animT1.y;
		out.z += (1.0f-dt)*animT0.z + dt*animT1.z;
	}
	return out;
}
