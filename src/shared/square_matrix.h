#pragma once
// ---------------------------------------------------------------------------
// square_matrix.h -- Generic NÃƒÆ’Ã¢â‚¬â€N square matrix
//
// Mirrors pbrt-v4 SquareMatrix<N> (util/math.h, Apache-2.0), adapted for
// double-precision CPU-only use.
//
// Components:
//   SquareMatrix<N>          -- NÃƒÆ’Ã¢â‚¬â€N matrix (identity default ctor)
//   Transpose<N>             -- generic transpose
//   Determinant<1,2,3,4>     -- specializations using DifferenceOfProducts
//   Inverse<3>, Inverse<4>   -- exact cofactor / Laplace expansion
//   operator*(M,M)           -- N=4 uses InnerProduct; generic N uses FMA loop
//   Mul<Tresult>(M, v)       -- matrix-vector product
//   LinearLeastSquares<N>    -- normal-equation least-squares solver
//
// Dependencies:
//   compensated_float.h      -- DifferenceOfProducts, InnerProduct, FMA
//
// Design rules:
//   - double precision throughout
//   - std::optional replaces pstd::optional
//   - operator[] returns double* / const double* (no pstd::span)
//   - No pbrt Allocator, no CUDA deps
//
// References:
//   pbrt-v4 src/pbrt/util/math.h  (Apache-2.0)
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "compensated_float.h"
#include <optional>
#include <cstring>
#include <cmath>
#include <cassert>

// ===========================================================================
// SquareMatrix<N>
// ===========================================================================
template <int N>
class SquareMatrix {
  public:
	// ---- Factories ----------------------------------------------------------

	CPU_GPU static SquareMatrix Zero() {
		SquareMatrix m;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				m.m[i][j] = 0.0;
		return m;
	}

	// ---- Constructors -------------------------------------------------------

	// Default: identity matrix
	CPU_GPU SquareMatrix() {
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				m[i][j] = (i == j) ? 1.0 : 0.0;
	}

	// From flat 2-D array
	CPU_GPU SquareMatrix(const double mat[N][N]) {
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				m[i][j] = mat[i][j];
	}

	// From flat 1-D array (row-major)
	CPU_GPU SquareMatrix(const double* t, int count) {
		assert(count == N * N);
		for (int i = 0; i < N * N; ++i)
			m[i / N][i % N] = t[i];
	}

	// Variadic element constructor: SquareMatrix<2>(a,b,c,d)
	// (requires exactly N*N arguments)
	template <typename... Args>
	CPU_GPU SquareMatrix(double v, Args... args) {
		static_assert(1 + sizeof...(Args) == N * N,
					  "Incorrect number of values provided to SquareMatrix constructor");
		init(m, 0, 0, v, args...);
	}

	// Diagonal matrix factory
	template <typename... Args>
	CPU_GPU static SquareMatrix Diag(double v, Args... args) {
		static_assert(1 + sizeof...(Args) == N,
					  "Incorrect number of values for SquareMatrix::Diag");
		SquareMatrix result;
		initDiag(result.m, 0, v, args...);
		return result;
	}

	// ---- Element access -----------------------------------------------------

	CPU_GPU const double* operator[](int i) const { return m[i]; }
	CPU_GPU double*       operator[](int i)       { return m[i]; }

	// ---- Arithmetic ---------------------------------------------------------

	CPU_GPU SquareMatrix operator+(const SquareMatrix& o) const {
		SquareMatrix r = *this;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				r.m[i][j] += o.m[i][j];
		return r;
	}

	CPU_GPU SquareMatrix operator*(double s) const {
		SquareMatrix r = *this;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				r.m[i][j] *= s;
		return r;
	}

	CPU_GPU SquareMatrix operator/(double s) const {
		SquareMatrix r = *this;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				r.m[i][j] /= s;
		return r;
	}

	// ---- Comparison ---------------------------------------------------------

	CPU_GPU bool operator==(const SquareMatrix& o) const {
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				if (m[i][j] != o.m[i][j]) return false;
		return true;
	}

	CPU_GPU bool operator!=(const SquareMatrix& o) const { return !(*this == o); }

	CPU_GPU bool operator<(const SquareMatrix& o) const {
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j) {
				if (m[i][j] < o.m[i][j]) return true;
				if (m[i][j] > o.m[i][j]) return false;
			}
		return false;
	}

	// ---- Predicates ---------------------------------------------------------

	CPU_GPU bool IsIdentity() const {
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				if (i == j ? m[i][j] != 1.0 : m[i][j] != 0.0)
					return false;
		return true;
	}

	// ---- Internal storage ---------------------------------------------------
	double m[N][N];

  private:
	// Recursive variadic-constructor helpers (pbrt-v4 style)
	template <int Rows>
	CPU_GPU static void init(double (&mat)[Rows][Rows], int i, int j) {}

	template <int Rows, typename... Args>
	CPU_GPU static void init(double (&mat)[Rows][Rows], int i, int j,
							  double v, Args... args) {
		mat[i][j] = v;
		int ni = i + (j + 1) / Rows;
		int nj = (j + 1) % Rows;
		init(mat, ni, nj, args...);
	}

	template <int Rows>
	CPU_GPU static void initDiag(double (&mat)[Rows][Rows], int i) {}

	template <int Rows, typename... Args>
	CPU_GPU static void initDiag(double (&mat)[Rows][Rows], int i,
								  double v, Args... args) {
		mat[i][i] = v;
		initDiag(mat, i + 1, args...);
	}
};

// ===========================================================================
// Scalar-on-left multiply
// ===========================================================================
template <int N>
CPU_GPU SquareMatrix<N> operator*(double s, const SquareMatrix<N>& m) {
	return m * s;
}

// ===========================================================================
// Transpose
// ===========================================================================
template <int N>
CPU_GPU SquareMatrix<N> Transpose(const SquareMatrix<N>& m) {
	SquareMatrix<N> r;
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			r[i][j] = m[j][i];
	return r;
}

// ===========================================================================
// Determinant specializations (exact via DifferenceOfProducts)
// ===========================================================================
template <int N>
CPU_GPU double Determinant(const SquareMatrix<N>& m);

template <>
inline double Determinant(const SquareMatrix<1>& m) {
	return m[0][0];
}

template <>
inline double Determinant(const SquareMatrix<2>& m) {
	return DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);
}

template <>
inline double Determinant(const SquareMatrix<3>& m) {
	double minor12 = DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
	double minor02 = DifferenceOfProducts(m[1][0], m[2][2], m[1][2], m[2][0]);
	double minor01 = DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
	return std::fma(m[0][2], minor01,
					DifferenceOfProducts(m[0][0], minor12, m[0][1], minor02));
}

template <>
inline double Determinant(const SquareMatrix<4>& m) {
	double s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
	double s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
	double s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);
	double s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
	double s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
	double s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);
	double c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
	double c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
	double c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);
	double c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
	double c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
	double c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);
	// Matches pbrt-v4: three DifferenceOfProducts pairs, not a flat InnerProduct
	return (DifferenceOfProducts(s0, c5, s1, c4) +
			DifferenceOfProducts(s2, c3, -s3, c2) +
			DifferenceOfProducts(s5, c0, s4, c1));
}

// ===========================================================================
// Inverse specializations
// ===========================================================================
template <int N>
CPU_GPU std::optional<SquareMatrix<N>> Inverse(const SquareMatrix<N>&);

template <>
inline std::optional<SquareMatrix<3>> Inverse(const SquareMatrix<3>& m) {
	double det = Determinant(m);
	if (det == 0.0) return {};
	double invDet = 1.0 / det;

	SquareMatrix<3> r;
	r[0][0] = invDet * DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
	r[1][0] = invDet * DifferenceOfProducts(m[1][2], m[2][0], m[1][0], m[2][2]);
	r[2][0] = invDet * DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
	r[0][1] = invDet * DifferenceOfProducts(m[0][2], m[2][1], m[0][1], m[2][2]);
	r[1][1] = invDet * DifferenceOfProducts(m[0][0], m[2][2], m[0][2], m[2][0]);
	r[2][1] = invDet * DifferenceOfProducts(m[0][1], m[2][0], m[0][0], m[2][1]);
	r[0][2] = invDet * DifferenceOfProducts(m[0][1], m[1][2], m[0][2], m[1][1]);
	r[1][2] = invDet * DifferenceOfProducts(m[0][2], m[1][0], m[0][0], m[1][2]);
	r[2][2] = invDet * DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);
	return r;
}

template <>
inline std::optional<SquareMatrix<4>> Inverse(const SquareMatrix<4>& m) {
	// Laplace expansion via sub-2x2 determinants (Eberly / Google Ion method)
	double s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
	double s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
	double s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);
	double s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
	double s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
	double s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

	double c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
	double c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
	double c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);
	double c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
	double c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
	double c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

	double det = InnerProduct(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
	if (det == 0.0) return {};
	double s = 1.0 / det;

	double inv[4][4] = {
		{ s * InnerProduct( m[1][1], c5,  m[1][3], c3, -m[1][2], c4),
		  s * InnerProduct(-m[0][1], c5,  m[0][2], c4, -m[0][3], c3),
		  s * InnerProduct( m[3][1], s5,  m[3][3], s3, -m[3][2], s4),
		  s * InnerProduct(-m[2][1], s5,  m[2][2], s4, -m[2][3], s3) },

		{ s * InnerProduct(-m[1][0], c5,  m[1][2], c2, -m[1][3], c1),
		  s * InnerProduct( m[0][0], c5,  m[0][3], c1, -m[0][2], c2),
		  s * InnerProduct(-m[3][0], s5,  m[3][2], s2, -m[3][3], s1),
		  s * InnerProduct( m[2][0], s5,  m[2][3], s1, -m[2][2], s2) },

		{ s * InnerProduct( m[1][0], c4,  m[1][3], c0, -m[1][1], c2),
		  s * InnerProduct(-m[0][0], c4,  m[0][1], c2, -m[0][3], c0),
		  s * InnerProduct( m[3][0], s4,  m[3][3], s0, -m[3][1], s2),
		  s * InnerProduct(-m[2][0], s4,  m[2][1], s2, -m[2][3], s0) },

		{ s * InnerProduct(-m[1][0], c3,  m[1][1], c1, -m[1][2], c0),
		  s * InnerProduct( m[0][0], c3,  m[0][2], c0, -m[0][1], c1),
		  s * InnerProduct(-m[3][0], s3,  m[3][1], s1, -m[3][2], s0),
		  s * InnerProduct( m[2][0], s3,  m[2][2], s0, -m[2][1], s1) }
	};
	return SquareMatrix<4>(inv);
}

// ===========================================================================
// Matrix-matrix multiplication
// Generic definition comes first so that the explicit specialization (N=4)
// is valid per-standard and accepted by MSVC.
// N=4: uses InnerProduct (4-term compensated dot product) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â matches pbrt-v4.
// Generic N: FMA loop (pbrt-v4 does not specialize N=3).
// ===========================================================================

template <int N>
CPU_GPU SquareMatrix<N> operator*(const SquareMatrix<N>& m1, const SquareMatrix<N>& m2) {
	SquareMatrix<N> r = SquareMatrix<N>::Zero();
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			for (int k = 0; k < N; ++k)
				r[i][j] = std::fma(m1[i][k], m2[k][j], r[i][j]);
	return r;
}

template <>
inline SquareMatrix<4> operator*(const SquareMatrix<4>& m1, const SquareMatrix<4>& m2) {
	SquareMatrix<4> r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r[i][j] = InnerProduct(m1[i][0], m2[0][j],
								   m1[i][1], m2[1][j],
								   m1[i][2], m2[2][j],
								   m1[i][3], m2[3][j]);
	return r;
}

// ===========================================================================
// Matrix-vector product: Mul<Tresult>(M, v)
// T must support operator[](int) and be constructible.
// ===========================================================================
template <typename Tresult, int N, typename T>
CPU_GPU Tresult Mul(const SquareMatrix<N>& m, const T& v) {
	Tresult result{};
	for (int i = 0; i < N; ++i) {
		double sum = 0.0;
		for (int j = 0; j < N; ++j)
			sum += m[i][j] * static_cast<double>(v[j]);
		using ElemType = std::remove_reference_t<decltype(result[i])>;
		result[i] = static_cast<ElemType>(sum);
	}
	return result;
}

// Matrix-vector via operator* (returns T directly)
template <int N, typename T>
CPU_GPU T operator*(const SquareMatrix<N>& m, const T& v) {
	return Mul<T>(m, v);
}

// ===========================================================================
// LinearLeastSquares<N>
// Solves A^T A x = A^T B using normal equations.
// A: rowsÃƒÆ’Ã¢â‚¬â€N,  B: rowsÃƒÆ’Ã¢â‚¬â€N  (each column of B is a right-hand side)
// Returns the NÃƒÆ’Ã¢â‚¬â€N matrix X such that A X ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  B.
// pbrt-v4: pstd::optional<SquareMatrix<N>> LinearLeastSquares(...)
// ===========================================================================
template <int N>
std::optional<SquareMatrix<N>> LinearLeastSquares(const double A[][N],
												   const double B[][N],
												   int rows) {
	SquareMatrix<N> AtA = SquareMatrix<N>::Zero();
	SquareMatrix<N> AtB = SquareMatrix<N>::Zero();

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			for (int r = 0; r < rows; ++r) {
				AtA[i][j] += A[r][i] * A[r][j];
				AtB[i][j] += A[r][i] * B[r][j];
			}

	auto AtAi = Inverse(AtA);
	if (!AtAi) return {};
	return Transpose(*AtAi * AtB);
}
