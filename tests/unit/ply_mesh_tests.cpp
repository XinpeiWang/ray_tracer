/**
 * @file ply_mesh_tests.cpp
 * @brief Unit tests for the PLY mesh reader
 *
 * The reader's real risk is not rejecting bad files - it is silently
 * misreading good ones. A PLY may carry per-vertex colour, confidence values,
 * or entire extra elements, in any order, and consuming those by the wrong
 * number of bytes does not fail: it desynchronises the stream and produces a
 * mesh of plausible-looking garbage. So the skipping cases get as much
 * attention here as the parsing ones, and the binary tests assert exact
 * coordinates rather than just "it loaded".
 */

#include <gtest/gtest.h>

#include "ply_mesh.h"

#include <cstdint>
#include <cstring>
#include <string>

using namespace ply_mesh;

namespace {

// Appends a value in host (little-endian) byte order.
template <typename T>
void put(std::string &s, T v) {
	char buf[sizeof(T)];
	std::memcpy(buf, &v, sizeof(T));
	s.append(buf, sizeof(T));
}

// Appends a value byte-reversed, for the big-endian tests.
template <typename T>
void putBE(std::string &s, T v) {
	char buf[sizeof(T)];
	std::memcpy(buf, &v, sizeof(T));
	for (std::size_t i = 0; i < sizeof(T) / 2; ++i)
		std::swap(buf[i], buf[sizeof(T) - 1 - i]);
	s.append(buf, sizeof(T));
}

const char *kAsciiTriangle =
	"ply\n"
	"format ascii 1.0\n"
	"comment made by a test\n"
	"element vertex 3\n"
	"property float x\n"
	"property float y\n"
	"property float z\n"
	"element face 1\n"
	"property list uchar int vertex_indices\n"
	"end_header\n"
	"0 0 0\n"
	"1 0 0\n"
	"0 1 0\n"
	"3 0 1 2\n";

} // namespace

// ===========================================================================
// ASCII
// ===========================================================================

TEST(PlyAsciiTest, ReadsATriangle) {
	const LoadResult r = parse(kAsciiTriangle);
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
	EXPECT_FLOAT_EQ(r.mesh.positions[3], 1.0f);   // second vertex x
	EXPECT_FLOAT_EQ(r.mesh.positions[7], 1.0f);   // third vertex y
	EXPECT_EQ(r.mesh.indices[0], 0);
	EXPECT_EQ(r.mesh.indices[2], 2);
	EXPECT_TRUE(r.mesh.normals.empty());
	EXPECT_TRUE(r.mesh.uvs.empty());
}

TEST(PlyAsciiTest, PicksUpNormalsAndTextureCoordinates) {
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 1\n"
		"property float x\nproperty float y\nproperty float z\n"
		"property float nx\nproperty float ny\nproperty float nz\n"
		"property float s\nproperty float t\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"end_header\n"
		"1 2 3  0 0 1  0.25 0.75\n"
		"3 0 0 0\n");
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.mesh.normals.size(), 3u);
	ASSERT_EQ(r.mesh.uvs.size(), 2u);
	EXPECT_FLOAT_EQ(r.mesh.normals[2], 1.0f);
	EXPECT_FLOAT_EQ(r.mesh.uvs[0], 0.25f);
	EXPECT_FLOAT_EQ(r.mesh.uvs[1], 0.75f);
}

TEST(PlyAsciiTest, AcceptsTheUvSpellingAsWellAsSt) {
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 1\n"
		"property float x\nproperty float y\nproperty float z\n"
		"property float u\nproperty float v\n"
		"element face 1\nproperty list uchar int vertex_indices\n"
		"end_header\n"
		"0 0 0  0.5 0.5\n"
		"3 0 0 0\n");
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.mesh.uvs.size(), 2u);
	EXPECT_FLOAT_EQ(r.mesh.uvs[0], 0.5f);
}

TEST(PlyAsciiTest, PolygonsAreFanTriangulated) {
	// Exporters emit quads even when the scene is nominally triangular.
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 4\n"
		"property float x\nproperty float y\nproperty float z\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"end_header\n"
		"0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
		"4 0 1 2 3\n");
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.mesh.triangleCount(), 2u) << "a quad becomes two triangles";
	EXPECT_EQ(r.mesh.indices[0], 0);
	EXPECT_EQ(r.mesh.indices[1], 1);
	EXPECT_EQ(r.mesh.indices[2], 2);
	EXPECT_EQ(r.mesh.indices[3], 0);
	EXPECT_EQ(r.mesh.indices[4], 2);
	EXPECT_EQ(r.mesh.indices[5], 3);
}

// ===========================================================================
// Properties and elements we do not care about
// ===========================================================================

TEST(PlySkipTest, UnknownVertexPropertiesDoNotShiftTheCoordinates) {
	// Colour between the position and the normal. If the reader assumed a
	// layout instead of consuming by declared width, the normal would silently
	// pick up the colour bytes.
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 1\n"
		"property float x\nproperty float y\nproperty float z\n"
		"property uchar red\nproperty uchar green\nproperty uchar blue\n"
		"property float nx\nproperty float ny\nproperty float nz\n"
		"element face 1\nproperty list uchar int vertex_indices\n"
		"end_header\n"
		"7 8 9  255 128 0  0 1 0\n"
		"3 0 0 0\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FLOAT_EQ(r.mesh.positions[0], 7.0f);
	EXPECT_FLOAT_EQ(r.mesh.positions[2], 9.0f);
	ASSERT_EQ(r.mesh.normals.size(), 3u);
	EXPECT_FLOAT_EQ(r.mesh.normals[1], 1.0f) << "normal must not absorb the colour bytes";
}

TEST(PlySkipTest, AnEntireUnknownElementIsConsumedExactly) {
	// A whole element between vertex and face. Getting its size wrong would
	// make the face data decode as noise.
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 3\n"
		"property float x\nproperty float y\nproperty float z\n"
		"element weirdstuff 2\n"
		"property int a\nproperty float b\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"end_header\n"
		"0 0 0\n1 0 0\n0 1 0\n"
		"11 1.5\n22 2.5\n"
		"3 0 1 2\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
	ASSERT_EQ(r.mesh.triangleCount(), 1u);
	EXPECT_EQ(r.mesh.indices[2], 2) << "face data must survive the skipped element";
}

TEST(PlySkipTest, UnknownListPropertyOnFacesIsConsumed) {
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 3\n"
		"property float x\nproperty float y\nproperty float z\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"property list uchar float texcoord\n"
		"end_header\n"
		"0 0 0\n1 0 0\n0 1 0\n"
		"3 0 1 2  6 0 0 1 0 0 1\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
}

// ===========================================================================
// Binary
// ===========================================================================

TEST(PlyBinaryTest, LittleEndianRoundTripsExactCoordinates) {
	std::string s =
		"ply\nformat binary_little_endian 1.0\n"
		"element vertex 3\n"
		"property float x\nproperty float y\nproperty float z\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"end_header\n";
	put<float>(s, 0.0f); put<float>(s, 0.0f); put<float>(s, 0.0f);
	put<float>(s, 1.5f); put<float>(s, 0.0f); put<float>(s, 0.0f);
	put<float>(s, 0.0f); put<float>(s, 2.5f); put<float>(s, 0.0f);
	put<std::uint8_t>(s, 3);
	put<std::int32_t>(s, 0); put<std::int32_t>(s, 1); put<std::int32_t>(s, 2);

	const LoadResult r = parse(s);
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
	EXPECT_FLOAT_EQ(r.mesh.positions[3], 1.5f);
	EXPECT_FLOAT_EQ(r.mesh.positions[7], 2.5f);
	EXPECT_EQ(r.mesh.indices[1], 1);
}

TEST(PlyBinaryTest, BigEndianIsByteSwappedNotMisread) {
	std::string s =
		"ply\nformat binary_big_endian 1.0\n"
		"element vertex 3\n"
		"property float x\nproperty float y\nproperty float z\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"end_header\n";
	putBE<float>(s, 0.0f); putBE<float>(s, 0.0f); putBE<float>(s, 0.0f);
	putBE<float>(s, 1.5f); putBE<float>(s, 0.0f); putBE<float>(s, 0.0f);
	putBE<float>(s, 0.0f); putBE<float>(s, 2.5f); putBE<float>(s, 0.0f);
	putBE<std::uint8_t>(s, 3);
	putBE<std::int32_t>(s, 0); putBE<std::int32_t>(s, 1); putBE<std::int32_t>(s, 2);

	const LoadResult r = parse(s);
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FLOAT_EQ(r.mesh.positions[3], 1.5f);
	EXPECT_FLOAT_EQ(r.mesh.positions[7], 2.5f);
	EXPECT_EQ(r.mesh.indices[2], 2);
}

TEST(PlyBinaryTest, MixedWidthPropertiesAreConsumedByTheirDeclaredSize) {
	// double positions with a uchar and a short in between - the widths must
	// come from the header, not from an assumption that everything is float.
	std::string s =
		"ply\nformat binary_little_endian 1.0\n"
		"element vertex 1\n"
		"property double x\nproperty double y\nproperty double z\n"
		"property uchar flag\nproperty short id\n"
		"property float nx\nproperty float ny\nproperty float nz\n"
		"element face 1\n"
		"property list uchar int vertex_indices\n"
		"end_header\n";
	put<double>(s, 10.0); put<double>(s, 20.0); put<double>(s, 30.0);
	put<std::uint8_t>(s, 7); put<std::int16_t>(s, -9);
	put<float>(s, 0.0f); put<float>(s, 0.0f); put<float>(s, 1.0f);
	put<std::uint8_t>(s, 3);
	put<std::int32_t>(s, 0); put<std::int32_t>(s, 0); put<std::int32_t>(s, 0);

	const LoadResult r = parse(s);
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FLOAT_EQ(r.mesh.positions[0], 10.0f);
	EXPECT_FLOAT_EQ(r.mesh.positions[2], 30.0f);
	ASSERT_EQ(r.mesh.normals.size(), 3u);
	EXPECT_FLOAT_EQ(r.mesh.normals[2], 1.0f);
}

TEST(PlyBinaryTest, CrLfAfterEndHeaderDoesNotEatAVertexByte) {
	// Windows-written headers end "end_header\r\n". Mishandling that shifts
	// every subsequent byte by one.
	std::string s =
		"ply\r\nformat binary_little_endian 1.0\r\n"
		"element vertex 1\r\n"
		"property float x\r\nproperty float y\r\nproperty float z\r\n"
		"element face 1\r\n"
		"property list uchar int vertex_indices\r\n"
		"end_header\r\n";
	put<float>(s, 4.0f); put<float>(s, 5.0f); put<float>(s, 6.0f);
	put<std::uint8_t>(s, 3);
	put<std::int32_t>(s, 0); put<std::int32_t>(s, 0); put<std::int32_t>(s, 0);

	const LoadResult r = parse(s);
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FLOAT_EQ(r.mesh.positions[0], 4.0f);
	EXPECT_FLOAT_EQ(r.mesh.positions[2], 6.0f);
}

// ===========================================================================
// Malformed input
// ===========================================================================

TEST(PlyErrorTest, RejectsAFileThatIsNotPly) {
	const LoadResult r = parse("OFF\n3 1 0\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("magic"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, RejectsAMissingEndHeader) {
	const LoadResult r = parse("ply\nformat ascii 1.0\nelement vertex 1\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("end_header"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, RejectsAnUnsupportedFormat) {
	const LoadResult r = parse("ply\nformat something_else 1.0\nend_header\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("unsupported"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, RejectsAnUnknownPropertyType) {
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 1\nproperty quadruple x\n"
		"end_header\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("quadruple"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, TruncatedBinaryDataIsReportedNotSilentlyShort) {
	std::string s =
		"ply\nformat binary_little_endian 1.0\n"
		"element vertex 4\n"
		"property float x\nproperty float y\nproperty float z\n"
		"end_header\n";
	put<float>(s, 1.0f);   // one float where twelve were promised
	const LoadResult r = parse(s);
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("truncated"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, OutOfRangeFaceIndexIsRejectedAtLoadTime) {
	// Left alone this indexes past the vertex array far away from the cause.
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element vertex 3\n"
		"property float x\nproperty float y\nproperty float z\n"
		"element face 1\nproperty list uchar int vertex_indices\n"
		"end_header\n"
		"0 0 0\n1 0 0\n0 1 0\n"
		"3 0 1 9\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("9"), std::string::npos) << r.error;
	EXPECT_NE(r.error.find("outside"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, AVertexlessFileIsRejected) {
	const LoadResult r = parse(
		"ply\nformat ascii 1.0\n"
		"element face 0\nproperty list uchar int vertex_indices\n"
		"end_header\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("no vertices"), std::string::npos) << r.error;
}

TEST(PlyErrorTest, MissingFileIsNamedInTheError) {
	const LoadResult r = loadFile("definitely/not/here.ply");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("definitely/not/here.ply"), std::string::npos) << r.error;
}
