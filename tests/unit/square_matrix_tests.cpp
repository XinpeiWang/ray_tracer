// square_matrix_tests.cpp
// Unit tests for SquareMatrix<N> ported from pbrt-v4.
// Covers: construction, arithmetic, Transpose, Determinant, Inverse, operator*, Mul, LinearLeastSquares.

#include "../../src/shared/square_matrix.h"
#include <gtest/gtest.h>
#include <cmath>
#include <array>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, DefaultIsIdentity) {
	SquareMatrix<3> I;
	EXPECT_TRUE(I.IsIdentity());
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_DOUBLE_EQ(I[i][j], i == j ? 1.0 : 0.0);
}

TEST(SquareMatrixTest, ZeroFactory) {
	SquareMatrix<4> Z = SquareMatrix<4>::Zero();
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			EXPECT_DOUBLE_EQ(Z[i][j], 0.0);
	EXPECT_FALSE(Z.IsIdentity());
}

TEST(SquareMatrixTest, VariadicConstructor3x3) {
	SquareMatrix<3> m(1, 2, 3,
					  4, 5, 6,
					  7, 8, 9);
	EXPECT_DOUBLE_EQ(m[0][0], 1.0);  EXPECT_DOUBLE_EQ(m[0][1], 2.0);  EXPECT_DOUBLE_EQ(m[0][2], 3.0);
	EXPECT_DOUBLE_EQ(m[1][0], 4.0);  EXPECT_DOUBLE_EQ(m[1][1], 5.0);  EXPECT_DOUBLE_EQ(m[1][2], 6.0);
	EXPECT_DOUBLE_EQ(m[2][0], 7.0);  EXPECT_DOUBLE_EQ(m[2][1], 8.0);  EXPECT_DOUBLE_EQ(m[2][2], 9.0);
}

TEST(SquareMatrixTest, FlatArrayConstructor) {
	double data[] = {1, 0, 0, 0,
					 0, 2, 0, 0,
					 0, 0, 3, 0,
					 0, 0, 0, 4};
	SquareMatrix<4> m(data, 16);
	EXPECT_DOUBLE_EQ(m[1][1], 2.0);
	EXPECT_DOUBLE_EQ(m[3][3], 4.0);
	EXPECT_DOUBLE_EQ(m[0][1], 0.0);
}

TEST(SquareMatrixTest, DiagFactory) {
	auto D = SquareMatrix<3>::Diag(2.0, 3.0, 5.0);
	EXPECT_DOUBLE_EQ(D[0][0], 2.0);
	EXPECT_DOUBLE_EQ(D[1][1], 3.0);
	EXPECT_DOUBLE_EQ(D[2][2], 5.0);
	EXPECT_DOUBLE_EQ(D[0][1], 0.0);
	EXPECT_DOUBLE_EQ(D[1][0], 0.0);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, ScalarMultiply) {
	SquareMatrix<2> m(1, 2, 3, 4);
	auto r = m * 2.0;
	EXPECT_DOUBLE_EQ(r[0][0], 2.0);
	EXPECT_DOUBLE_EQ(r[1][1], 8.0);
	// scalar-on-left
	auto r2 = 3.0 * m;
	EXPECT_DOUBLE_EQ(r2[0][1], 6.0);
}

TEST(SquareMatrixTest, MatrixAdd) {
	SquareMatrix<2> a(1, 2, 3, 4);
	SquareMatrix<2> b(5, 6, 7, 8);
	auto c = a + b;
	EXPECT_DOUBLE_EQ(c[0][0], 6.0);
	EXPECT_DOUBLE_EQ(c[1][1], 12.0);
}

TEST(SquareMatrixTest, EqualityAndInequality) {
	SquareMatrix<2> a(1, 0, 0, 1);
	SquareMatrix<2> b;
	EXPECT_EQ(a, b);
	SquareMatrix<2> c(2, 0, 0, 1);
	EXPECT_NE(a, c);
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, Transpose3x3) {
	SquareMatrix<3> m(1, 2, 3,
					  4, 5, 6,
					  7, 8, 9);
	auto t = Transpose(m);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_DOUBLE_EQ(t[i][j], m[j][i]);
}

TEST(SquareMatrixTest, TransposeIdentityIsIdentity) {
	SquareMatrix<4> I;
	EXPECT_EQ(Transpose(I), I);
}

// ---------------------------------------------------------------------------
// Determinant
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, Det1x1) {
	SquareMatrix<1> m(5.0);
	EXPECT_DOUBLE_EQ(Determinant(m), 5.0);
}

TEST(SquareMatrixTest, Det2x2) {
	SquareMatrix<2> m(3, 8, 4, 6);
	// 3*6 - 8*4 = 18 - 32 = -14
	EXPECT_DOUBLE_EQ(Determinant(m), -14.0);
}

TEST(SquareMatrixTest, Det3x3) {
	SquareMatrix<3> m(6, 1, 1,
					  4, -2, 5,
					  2, 8, 7);
	// expected: 6*((-2)*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
	//         = 6*(-54) - 1*(18) + 1*(36) = -324 - 18 + 36 = -306
	EXPECT_DOUBLE_EQ(Determinant(m), -306.0);
}

TEST(SquareMatrixTest, DetIdentity) {
	EXPECT_DOUBLE_EQ(Determinant(SquareMatrix<3>()), 1.0);
	EXPECT_DOUBLE_EQ(Determinant(SquareMatrix<4>()), 1.0);
}

// ---------------------------------------------------------------------------
// Inverse
// ---------------------------------------------------------------------------
static double matNorm(const SquareMatrix<3>& a) {
	double s = 0;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			s += a[i][j] * a[i][j];
	return std::sqrt(s);
}
static double matNorm(const SquareMatrix<4>& a) {
	double s = 0;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			s += a[i][j] * a[i][j];
	return std::sqrt(s);
}

TEST(SquareMatrixTest, Inverse3x3RoundTrip) {
	SquareMatrix<3> m(1, 2, 3,
					  0, 1, 4,
					  5, 6, 0);
	auto inv = Inverse(m);
	ASSERT_TRUE(inv.has_value());
	auto prod = m * (*inv);
	SquareMatrix<3> I;
	double err = 0;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			err = std::max(err, std::abs(prod[i][j] - I[i][j]));
	EXPECT_LT(err, 1e-12);
}

TEST(SquareMatrixTest, Inverse4x4RoundTrip) {
	SquareMatrix<4> m(5,  7,  6,  5,
					  7, 10,  8,  7,
					  6,  8, 10,  9,
					  5,  7,  9, 10);
	auto inv = Inverse(m);
	ASSERT_TRUE(inv.has_value());
	auto prod = m * (*inv);
	SquareMatrix<4> I;
	double err = 0;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			err = std::max(err, std::abs(prod[i][j] - I[i][j]));
	EXPECT_LT(err, 1e-10);
}

TEST(SquareMatrixTest, InverseSingular3x3) {
	SquareMatrix<3> m(1, 2, 3,
					  4, 5, 6,
					  7, 8, 9);   // singular: row 3 = row 1 + row 2 - const
	auto inv = Inverse(m);
	EXPECT_FALSE(inv.has_value());
}

TEST(SquareMatrixTest, InverseSingular4x4) {
	SquareMatrix<4> m = SquareMatrix<4>::Zero();
	auto inv = Inverse(m);
	EXPECT_FALSE(inv.has_value());
}

TEST(SquareMatrixTest, InverseIdentityIsIdentity) {
	auto inv = Inverse(SquareMatrix<3>());
	ASSERT_TRUE(inv.has_value());
	EXPECT_TRUE(inv->IsIdentity());
}

// ---------------------------------------------------------------------------
// Matrix-matrix multiply
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, Multiply3x3IdentityNoop) {
	SquareMatrix<3> m(1, 2, 3,
					  4, 5, 6,
					  7, 8, 9);
	SquareMatrix<3> I;
	EXPECT_EQ(m * I, m);
	EXPECT_EQ(I * m, m);
}

TEST(SquareMatrixTest, Multiply4x4IdentityNoop) {
	SquareMatrix<4> m(1, 2, 0, 0,
					  3, 4, 0, 0,
					  0, 0, 5, 6,
					  0, 0, 7, 8);
	SquareMatrix<4> I;
	EXPECT_EQ(m * I, m);
}

TEST(SquareMatrixTest, Multiply3x3Known) {
	SquareMatrix<3> a(1, 2, 3,
					  4, 5, 6,
					  7, 8, 9);
	SquareMatrix<3> b(9, 8, 7,
					  6, 5, 4,
					  3, 2, 1);
	auto c = a * b;
	// row 0: [1*9+2*6+3*3, 1*8+2*5+3*2, 1*7+2*4+3*1] = [30, 24, 18]
	EXPECT_DOUBLE_EQ(c[0][0], 30.0);
	EXPECT_DOUBLE_EQ(c[0][1], 24.0);
	EXPECT_DOUBLE_EQ(c[0][2], 18.0);
	// row 2: [7*9+8*6+9*3, 7*8+8*5+9*2, 7*7+8*4+9*1] = [138, 114, 90]
	EXPECT_DOUBLE_EQ(c[2][0], 138.0);
	EXPECT_DOUBLE_EQ(c[2][1], 114.0);
	EXPECT_DOUBLE_EQ(c[2][2], 90.0);
}

// ---------------------------------------------------------------------------
// Matrix-vector multiply
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, MatVecMul3) {
	SquareMatrix<3> m(1, 0, 0,
					  0, 2, 0,
					  0, 0, 3);
	std::array<double, 3> v = {1.0, 2.0, 3.0};
	auto r = Mul<std::array<double, 3>>(m, v);
	EXPECT_DOUBLE_EQ(r[0], 1.0);
	EXPECT_DOUBLE_EQ(r[1], 4.0);
	EXPECT_DOUBLE_EQ(r[2], 9.0);
}

// ---------------------------------------------------------------------------
// LinearLeastSquares
// ---------------------------------------------------------------------------
TEST(SquareMatrixTest, LinearLeastSquaresDiagonalIdentity) {
	// If A = I (identity rows) and B = I, solution is I.
	double A[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
	double B[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
	auto sol = LinearLeastSquares<3>(A, B, 3);
	ASSERT_TRUE(sol.has_value());
	double err = 0;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			err = std::max(err, std::abs((*sol)[i][j] - (i == j ? 1.0 : 0.0)));
	EXPECT_LT(err, 1e-12);
}
