#pragma once
// ---------------------------------------------------------------------------
// animated_transform.h -- Keyframe transform interpolation
//
// Mirrors pbrt-v4 AnimatedTransform (src/pbrt/util/transform.h/.cpp).
//
// Stores two endpoint 4x4 transforms with associated times and interpolates
// between them at query time using independent TRS (Translation / Rotation /
// Scale) decomposition:
//
//   Translation : linear lerp of the translation vectors
//   Rotation    : quaternion slerp (shortest-arc path)
//   Scale       : element-wise linear lerp of the 3x3 scale matrix
//
// The decomposition follows the polar-decomposition approach from pbrt-v4:
//   - Extract T from the last column of the matrix
//   - Iteratively compute R via R_next = (R + (R^-T)) / 2 until convergence
//   - S = R^-1 * M   (M = original without translation)
//
// Public API (matches pbrt-v4 naming):
//   AnimatedTransform(start, t0, end, t1)
//   bool  IsAnimated()      const
//   bool  HasScale()        const
//   Mat44 Interpolate(t)    const  -> interpolated 4x4 matrix at time t
//   void  apply_point (p, t, out)  -> transform a point
//   void  apply_vector(v, t, out)  -> transform a direction vector
//   void  apply_normal(n, t, out)  -> transform a surface normal
//   void  apply_ray   (o,d,t,out_o,out_d) -> transform a ray
//   void  apply_inverse_point(p,t,out)
//   void  apply_inverse_vector(v,t,out)
//
// Design rules:
//   - Self-contained; no external math library required
//   - CPU_GPU tagged for CUDA compatibility
//   - T=double for CPU; callers may instantiate with float on GPU
//
// Reference: pbrt-v4 src/pbrt/util/transform.h/.cpp
//            pbrt-v4 src/pbrt/util/vecmath.h  (Quaternion, Slerp)
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

// ===========================================================================
// Mat44  -- minimal 4x4 matrix (row-major, same layout as pbrt-v4 SquareMatrix<4>)
// ===========================================================================
struct AT_Mat44 {
	double m[4][4];

	CPU_GPU AT_Mat44() {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				m[i][j] = (i == j) ? 1.0 : 0.0;
	}

	CPU_GPU AT_Mat44(const double src[4][4]) {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				m[i][j] = src[i][j];
	}

	CPU_GPU double operator[](int i) const = delete; // use [i][j]
};

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------
CPU_GPU AT_Mat44 at_identity() { return AT_Mat44{}; }

CPU_GPU AT_Mat44 at_mul(const AT_Mat44& a, const AT_Mat44& b) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) {
			r.m[i][j] = 0.0;
			for (int k = 0; k < 4; ++k)
				r.m[i][j] += a.m[i][k] * b.m[k][j];
		}
	return r;
}

CPU_GPU AT_Mat44 at_add(const AT_Mat44& a, const AT_Mat44& b) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = a.m[i][j] + b.m[i][j];
	return r;
}

CPU_GPU AT_Mat44 at_scale_mat(const AT_Mat44& a, double s) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = a.m[i][j] * s;
	return r;
}

CPU_GPU AT_Mat44 at_lerp_mat(double t, const AT_Mat44& a, const AT_Mat44& b) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = (1.0 - t) * a.m[i][j] + t * b.m[i][j];
	return r;
}

CPU_GPU AT_Mat44 at_transpose(const AT_Mat44& a) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = a.m[j][i];
	return r;
}

// 4x4 matrix inversion via Gauss-Jordan.
// pbrt-v4: InvertOrExit (util/vecmath.h)
CPU_GPU bool at_invert(const AT_Mat44& src, AT_Mat44& inv) {
	double a[4][8];
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) a[i][j] = src.m[i][j];
		for (int j = 0; j < 4; ++j) a[i][j + 4] = (i == j) ? 1.0 : 0.0;
	}
	for (int col = 0; col < 4; ++col) {
		// Find pivot
		int pivot = col;
		for (int row = col + 1; row < 4; ++row)
			if (std::fabs(a[row][col]) > std::fabs(a[pivot][col])) pivot = row;
		if (pivot != col)
			for (int j = 0; j < 8; ++j) { double tmp = a[col][j]; a[col][j] = a[pivot][j]; a[pivot][j] = tmp; }
		if (a[col][col] == 0.0) return false;
		double inv_piv = 1.0 / a[col][col];
		for (int j = 0; j < 8; ++j) a[col][j] *= inv_piv;
		for (int row = 0; row < 4; ++row) {
			if (row == col) continue;
			double factor = a[row][col];
			for (int j = 0; j < 8; ++j) a[row][j] -= factor * a[col][j];
		}
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			inv.m[i][j] = a[i][j + 4];
	return true;
}

// Build a translation matrix.
// pbrt-v4: Translate(Vector3f delta)
CPU_GPU AT_Mat44 at_translate(double tx, double ty, double tz) {
	AT_Mat44 r;
	r.m[0][3] = tx; r.m[1][3] = ty; r.m[2][3] = tz;
	return r;
}

// ===========================================================================
// AT_Quat -- unit quaternion for rotation (w + xi + yj + zk)
// pbrt-v4: Quaternion (util/vecmath.h)
// ===========================================================================
struct AT_Quat {
	double x, y, z, w;
	CPU_GPU AT_Quat() : x(0), y(0), z(0), w(1) {}
	CPU_GPU AT_Quat(double x_, double y_, double z_, double w_)
		: x(x_), y(y_), z(z_), w(w_) {}
};

CPU_GPU double at_dot(const AT_Quat& a, const AT_Quat& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

CPU_GPU AT_Quat at_normalize(const AT_Quat& q) {
	double len = std::sqrt(at_dot(q, q));
	if (len == 0.0) return AT_Quat(0, 0, 0, 1);
	return AT_Quat(q.x / len, q.y / len, q.z / len, q.w / len);
}

CPU_GPU AT_Quat operator-(const AT_Quat& a) {
	return AT_Quat(-a.x, -a.y, -a.z, -a.w);
}

CPU_GPU AT_Quat operator+(const AT_Quat& a, const AT_Quat& b) {
	return AT_Quat(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

CPU_GPU AT_Quat operator*(double s, const AT_Quat& q) {
	return AT_Quat(s * q.x, s * q.y, s * q.z, s * q.w);
}

CPU_GPU AT_Quat operator-(const AT_Quat& a, const AT_Quat& b) {
	return AT_Quat(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Spherical linear interpolation (shortest arc).
// pbrt-v4: Slerp(Float t, Quaternion q1, Quaternion q2)
CPU_GPU AT_Quat at_slerp(double t, const AT_Quat& q1, const AT_Quat& q2) {
	double cosTheta = at_dot(q1, q2);
	if (cosTheta > 0.9995) {
		// Quaternions nearly parallel: linear interpolation
		return at_normalize(
			AT_Quat((1.0 - t) * q1.x + t * q2.x,
					(1.0 - t) * q1.y + t * q2.y,
					(1.0 - t) * q1.z + t * q2.z,
					(1.0 - t) * q1.w + t * q2.w));
	}
	double theta = std::acos(std::max(-1.0, std::min(1.0, cosTheta)));
	double thetaP = theta * t;
	// q_perp = normalize(q2 - q1 * cosTheta)
	AT_Quat tmp = q2 - (cosTheta * q1);
	tmp = at_normalize(tmp);
	return std::cos(thetaP) * q1 + std::sin(thetaP) * tmp;
}

// Convert a unit quaternion to a rotation 4x4 matrix.
// pbrt-v4: Transform(Quaternion q)
CPU_GPU AT_Mat44 at_quat_to_mat(const AT_Quat& q) {
	double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
	double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
	double wx = q.x * q.w, wy = q.y * q.w, wz = q.z * q.w;

	// pbrt-v4 uses left-handed convention: m = Transpose(mInv)
	// mInv = rotation matrix from quaternion
	double mInv[4][4] = {
		{ 1 - 2*(yy+zz),  2*(xy+wz),     2*(xz-wy),    0 },
		{ 2*(xy-wz),      1 - 2*(xx+zz), 2*(yz+wx),    0 },
		{ 2*(xz+wy),      2*(yz-wx),     1 - 2*(xx+yy),0 },
		{ 0,              0,             0,             1 }
	};
	// m = Transpose(mInv)
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = mInv[j][i];
	return r;
}

// Extract a unit quaternion from a rotation matrix.
// Based on Shepperd's method.
// pbrt-v4: Quaternion(Transform t) -- constructs from the m matrix
CPU_GPU AT_Quat at_mat_to_quat(const AT_Mat44& rot) {
	// rot.m is the forward rotation matrix
	// pbrt-v4 stores: m = rotation_matrix, mInv = Transpose(m)
	// So mInv[i][j] = m[j][i] = rot.m[j][i]
	// The quaternion constructor in pbrt-v4 works on mInv:
	//   mInv[0][0] = 1-2(yy+zz), mInv[0][1] = 2(xy+wz), ...
	// which equals rot.m[0][0], rot.m[1][0], ...
	// Use rot.m directly in Shepperd order
	const double* M = &rot.m[0][0]; // row-major
	// M[row*4+col]
	double trace = M[0*4+0] + M[1*4+1] + M[2*4+2];
	AT_Quat q;
	if (trace > 0) {
		double s = 0.5 / std::sqrt(trace + 1.0);
		q.w = 0.25 / s;
		q.x = (M[2*4+1] - M[1*4+2]) * s;
		q.y = (M[0*4+2] - M[2*4+0]) * s;
		q.z = (M[1*4+0] - M[0*4+1]) * s;
	} else if (M[0*4+0] > M[1*4+1] && M[0*4+0] > M[2*4+2]) {
		double s = 2.0 * std::sqrt(1.0 + M[0*4+0] - M[1*4+1] - M[2*4+2]);
		q.w = (M[2*4+1] - M[1*4+2]) / s;
		q.x = 0.25 * s;
		q.y = (M[0*4+1] + M[1*4+0]) / s;
		q.z = (M[0*4+2] + M[2*4+0]) / s;
	} else if (M[1*4+1] > M[2*4+2]) {
		double s = 2.0 * std::sqrt(1.0 + M[1*4+1] - M[0*4+0] - M[2*4+2]);
		q.w = (M[0*4+2] - M[2*4+0]) / s;
		q.x = (M[0*4+1] + M[1*4+0]) / s;
		q.y = 0.25 * s;
		q.z = (M[1*4+2] + M[2*4+1]) / s;
	} else {
		double s = 2.0 * std::sqrt(1.0 + M[2*4+2] - M[0*4+0] - M[1*4+1]);
		q.w = (M[1*4+0] - M[0*4+1]) / s;
		q.x = (M[0*4+2] + M[2*4+0]) / s;
		q.y = (M[1*4+2] + M[2*4+1]) / s;
		q.z = 0.25 * s;
	}
	return at_normalize(q);
}

// ===========================================================================
// TRS decomposition
//
// Extracts (T, R, S) from a 4x4 affine matrix such that M = T * R * S.
//
// pbrt-v4: Transform::Decompose (util/transform.cpp)
//   - T  = translation from last column
//   - R  = rotation via polar decomposition (iterative convergence)
//   - S  = R^-1 * M_no_translation
//
// Outputs:
//   T_out[3]        -- translation vector
//   R_out           -- rotation as AT_Mat44 (upper-left 3x3 is the rotation)
//   S_out           -- scale as AT_Mat44
// ===========================================================================
CPU_GPU void at_decompose(const AT_Mat44& mat,
								  double T_out[3],
								  AT_Mat44& R_out,
								  AT_Mat44& S_out)
{
	// --- Extract translation ---
	T_out[0] = mat.m[0][3];
	T_out[1] = mat.m[1][3];
	T_out[2] = mat.m[2][3];

	// --- Remove translation column ---
	AT_Mat44 M = mat;
	for (int i = 0; i < 3; ++i) { M.m[i][3] = 0.0; M.m[3][i] = 0.0; }
	M.m[3][3] = 1.0;

	// --- Polar decomposition for rotation (pbrt-v4: iterative R = (R + R^{-T})/2) ---
	R_out = M;
	int count = 0;
	double norm;
	do {
		AT_Mat44 Rit;
		at_invert(at_transpose(R_out), Rit);
		AT_Mat44 Rnext = at_scale_mat(at_add(R_out, Rit), 0.5);

		norm = 0.0;
		for (int i = 0; i < 3; ++i) {
			double n = std::fabs(R_out.m[i][0] - Rnext.m[i][0])
					 + std::fabs(R_out.m[i][1] - Rnext.m[i][1])
					 + std::fabs(R_out.m[i][2] - Rnext.m[i][2]);
			norm = std::max(norm, n);
		}
		R_out = Rnext;
	} while (++count < 100 && norm > 1e-4);

	// --- Scale: S = R^-1 * M ---
	AT_Mat44 R_inv;
	at_invert(R_out, R_inv);
	S_out = at_mul(R_inv, M);
}

// ===========================================================================
// AnimatedTransform
//
// pbrt-v4: AnimatedTransform (util/transform.h)
//
// Stores start/end 4x4 transforms with times, decomposes them into TRS,
// and interpolates at query time.
// ===========================================================================
struct AnimatedTransform {

	// -----------------------------------------------------------------------
	// Stored endpoint transforms (pbrt-v4: public members)
	AT_Mat44 startTransform;
	AT_Mat44 endTransform;
	double   startTime = 0.0;
	double   endTime   = 1.0;

	// -----------------------------------------------------------------------
	// Decomposed TRS for each endpoint (private in pbrt-v4)
	double   T[2][3];      // translation at t=0 and t=1
	AT_Quat  R[2];         // rotation quaternions
	AT_Mat44 S[2];         // scale matrices
	bool     actuallyAnimated = false;

	// -----------------------------------------------------------------------
	// Default constructor: identity, t=[0,1]
	AnimatedTransform() {
		startTransform = at_identity();
		endTransform   = at_identity();
		actuallyAnimated = false;
		T[0][0] = T[0][1] = T[0][2] = 0.0;
		T[1][0] = T[1][1] = T[1][2] = 0.0;
		R[0] = AT_Quat(0, 0, 0, 1);
		R[1] = AT_Quat(0, 0, 0, 1);
		S[0] = at_identity();
		S[1] = at_identity();
	}

	// -----------------------------------------------------------------------
	// Construct from two endpoint transforms + times.
	// pbrt-v4: AnimatedTransform::AnimatedTransform(...)
	AnimatedTransform(const AT_Mat44& start, double t0,
					  const AT_Mat44& end,   double t1)
		: startTransform(start), endTransform(end)
		, startTime(t0), endTime(t1)
	{
		// Detect if actually animated (compare matrices)
		actuallyAnimated = false;
		for (int i = 0; i < 4 && !actuallyAnimated; ++i)
			for (int j = 0; j < 4 && !actuallyAnimated; ++j)
				if (start.m[i][j] != end.m[i][j]) actuallyAnimated = true;

		if (!actuallyAnimated) {
			// Still decompose start for consistency
			AT_Mat44 R_mat;
			at_decompose(start, T[0], R_mat, S[0]);
			R[0] = at_mat_to_quat(R_mat);
			T[1][0] = T[0][0]; T[1][1] = T[0][1]; T[1][2] = T[0][2];
			R[1] = R[0];
			S[1] = S[0];
			return;
		}

		// Decompose both endpoints
		AT_Mat44 R_mat;
		at_decompose(start, T[0], R_mat, S[0]);
		R[0] = at_mat_to_quat(R_mat);

		at_decompose(end, T[1], R_mat, S[1]);
		R[1] = at_mat_to_quat(R_mat);

		// Flip R[1] if needed to select shortest path
		// pbrt-v4: if (Dot(R[0], R[1]) < 0) R[1] = -R[1]
		if (at_dot(R[0], R[1]) < 0.0)
			R[1] = -R[1];
	}

	// Convenience constructor from a single (static) transform.
	explicit AnimatedTransform(const AT_Mat44& m)
		: AnimatedTransform(m, 0.0, m, 1.0) {}

	// -----------------------------------------------------------------------
	// IsAnimated
	// pbrt-v4: bool IsAnimated() const { return actuallyAnimated; }
	CPU_GPU bool IsAnimated() const { return actuallyAnimated; }

	// -----------------------------------------------------------------------
	// HasScale
	// pbrt-v4: bool HasScale() const { ... }
	// Returns true if either endpoint has a non-unit-scale component.
	CPU_GPU bool HasScale() const {
		auto has_scale = [](const AT_Mat44& m) -> bool {
			// Check if any of the three column vectors (upper 3x3) deviate from unit length
			for (int col = 0; col < 3; ++col) {
				double len2 = m.m[0][col]*m.m[0][col]
							+ m.m[1][col]*m.m[1][col]
							+ m.m[2][col]*m.m[2][col];
				if (std::fabs(len2 - 1.0) > 1e-3) return true;
			}
			return false;
		};
		return has_scale(startTransform) || has_scale(endTransform);
	}

	// -----------------------------------------------------------------------
	// Interpolate
	//
	// Returns the interpolated 4x4 matrix at the given time.
	// pbrt-v4: Transform AnimatedTransform::Interpolate(Float time) const
	//
	// Algorithm:
	//   dt = (time - startTime) / (endTime - startTime)
	//   trans = lerp(dt, T[0], T[1])
	//   rotate = slerp(dt, R[0], R[1])
	//   scale = lerp(dt, S[0], S[1])
	//   return Translate(trans) * RotationMatrix(rotate) * scale
	// -----------------------------------------------------------------------
	CPU_GPU AT_Mat44 Interpolate(double time) const {
		if (!actuallyAnimated || time <= startTime) return startTransform;
		if (time >= endTime)                        return endTransform;

		double dt = (time - startTime) / (endTime - startTime);

		// Interpolate translation
		double tx = (1.0 - dt) * T[0][0] + dt * T[1][0];
		double ty = (1.0 - dt) * T[0][1] + dt * T[1][1];
		double tz = (1.0 - dt) * T[0][2] + dt * T[1][2];
		AT_Mat44 trans_mat = at_translate(tx, ty, tz);

		// Interpolate rotation
		AT_Quat rot = at_slerp(dt, R[0], R[1]);
		AT_Mat44 rot_mat = at_quat_to_mat(rot);

		// Interpolate scale
		AT_Mat44 scale_mat = at_lerp_mat(dt, S[0], S[1]);

		// Compose: T * R * S
		return at_mul(trans_mat, at_mul(rot_mat, scale_mat));
	}

	// -----------------------------------------------------------------------
	// apply_point
	// Transform a 3D point (homogeneous divide if w != 1).
	// pbrt-v4: Point3f AnimatedTransform::operator()(Point3f p, Float time)
	// -----------------------------------------------------------------------
	CPU_GPU void apply_point(const double in[3], double time, double out[3]) const {
		AT_Mat44 mat = Interpolate(time);
		double x = in[0], y = in[1], z = in[2];
		double ox = mat.m[0][0]*x + mat.m[0][1]*y + mat.m[0][2]*z + mat.m[0][3];
		double oy = mat.m[1][0]*x + mat.m[1][1]*y + mat.m[1][2]*z + mat.m[1][3];
		double oz = mat.m[2][0]*x + mat.m[2][1]*y + mat.m[2][2]*z + mat.m[2][3];
		double ow = mat.m[3][0]*x + mat.m[3][1]*y + mat.m[3][2]*z + mat.m[3][3];
		if (ow == 1.0) { out[0] = ox; out[1] = oy; out[2] = oz; }
		else           { out[0] = ox/ow; out[1] = oy/ow; out[2] = oz/ow; }
	}

	// -----------------------------------------------------------------------
	// apply_vector
	// Transform a direction vector (no translation applied).
	// pbrt-v4: Vector3f AnimatedTransform::operator()(Vector3f v, Float time)
	// -----------------------------------------------------------------------
	CPU_GPU void apply_vector(const double in[3], double time, double out[3]) const {
		AT_Mat44 mat = Interpolate(time);
		double x = in[0], y = in[1], z = in[2];
		out[0] = mat.m[0][0]*x + mat.m[0][1]*y + mat.m[0][2]*z;
		out[1] = mat.m[1][0]*x + mat.m[1][1]*y + mat.m[1][2]*z;
		out[2] = mat.m[2][0]*x + mat.m[2][1]*y + mat.m[2][2]*z;
	}

	// -----------------------------------------------------------------------
	// apply_normal
	// Transform a surface normal (multiply by inverse-transpose).
	// pbrt-v4: Normal3f AnimatedTransform::operator()(Normal3f n, Float time)
	// -----------------------------------------------------------------------
	CPU_GPU void apply_normal(const double in[3], double time, double out[3]) const {
		AT_Mat44 mat = Interpolate(time);
		AT_Mat44 inv;
		at_invert(mat, inv);
		// Normal transforms by inverse-transpose (= inv transposed applied to n)
		double x = in[0], y = in[1], z = in[2];
		out[0] = inv.m[0][0]*x + inv.m[1][0]*y + inv.m[2][0]*z;
		out[1] = inv.m[0][1]*x + inv.m[1][1]*y + inv.m[2][1]*z;
		out[2] = inv.m[0][2]*x + inv.m[1][2]*y + inv.m[2][2]*z;
	}

	// -----------------------------------------------------------------------
	// apply_ray
	// Transform a ray (origin as point, direction as vector).
	// pbrt-v4: Ray AnimatedTransform::operator()(const Ray& r, ...) const
	// -----------------------------------------------------------------------
	CPU_GPU void apply_ray(const double orig[3], const double dir[3],
							double time,
							double out_orig[3], double out_dir[3]) const {
		apply_point (orig, time, out_orig);
		apply_vector(dir,  time, out_dir);
	}

	// -----------------------------------------------------------------------
	// apply_inverse_point
	// pbrt-v4: Point3f AnimatedTransform::ApplyInverse(Point3f p, Float time)
	// -----------------------------------------------------------------------
	CPU_GPU void apply_inverse_point(const double in[3], double time, double out[3]) const {
		AT_Mat44 mat = Interpolate(time);
		AT_Mat44 inv;
		at_invert(mat, inv);
		double x = in[0], y = in[1], z = in[2];
		double ox = inv.m[0][0]*x + inv.m[0][1]*y + inv.m[0][2]*z + inv.m[0][3];
		double oy = inv.m[1][0]*x + inv.m[1][1]*y + inv.m[1][2]*z + inv.m[1][3];
		double oz = inv.m[2][0]*x + inv.m[2][1]*y + inv.m[2][2]*z + inv.m[2][3];
		double ow = inv.m[3][0]*x + inv.m[3][1]*y + inv.m[3][2]*z + inv.m[3][3];
		if (ow == 1.0) { out[0] = ox; out[1] = oy; out[2] = oz; }
		else           { out[0] = ox/ow; out[1] = oy/ow; out[2] = oz/ow; }
	}

	// -----------------------------------------------------------------------
	// apply_inverse_vector
	// pbrt-v4: Vector3f AnimatedTransform::ApplyInverse(Vector3f v, Float time)
	// -----------------------------------------------------------------------
	CPU_GPU void apply_inverse_vector(const double in[3], double time, double out[3]) const {
		AT_Mat44 mat = Interpolate(time);
		AT_Mat44 inv;
		at_invert(mat, inv);
		double x = in[0], y = in[1], z = in[2];
		out[0] = inv.m[0][0]*x + inv.m[0][1]*y + inv.m[0][2]*z;
		out[1] = inv.m[1][0]*x + inv.m[1][1]*y + inv.m[1][2]*z;
		out[2] = inv.m[2][0]*x + inv.m[2][1]*y + inv.m[2][2]*z;
	}
};
