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
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
// Wavefront OBJ (the non-standard extension that lets `Shape "plymesh"` name a
// .obj directly - see ply_mesh.h's OBJ SUPPORT note).
// ---------------------------------------------------------------------------

TEST(ObjTest, ATriangleLoadsAsIs) {
	const LoadResult r = parseObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
	ASSERT_EQ(r.mesh.indices.size(), 3u);
	EXPECT_EQ(r.mesh.indices[0], 0);
	EXPECT_EQ(r.mesh.indices[1], 1);
	EXPECT_EQ(r.mesh.indices[2], 2);
	EXPECT_TRUE(r.mesh.uvs.empty());
	EXPECT_TRUE(r.mesh.normals.empty());
}

TEST(ObjTest, AQuadIsFanTriangulated) {
	const LoadResult r = parseObj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
	ASSERT_TRUE(r.ok) << r.error;
	const std::vector<int> expected = {0, 1, 2, 0, 2, 3};
	EXPECT_EQ(r.mesh.indices, expected);
}

TEST(ObjTest, NormalOnlyReferencesLoadPerVertexNormals) {
	// `v//vn` - the shape of nearly every position+normal export. No vt means no
	// uvs; the normals are per-vertex, so positions are expanded only as far as
	// distinct (position, normal) pairs require - here each corner uses a
	// different position, so 3 vertices.
	const LoadResult r = parseObj(
		"v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
	EXPECT_TRUE(r.mesh.uvs.empty());
	ASSERT_EQ(r.mesh.normals.size(), 9u);
	EXPECT_FLOAT_EQ(r.mesh.normals[2], 1.0f);   // vertex 0 -> vn 1 = (0,0,1)
	EXPECT_FLOAT_EQ(r.mesh.normals[4], 1.0f);   // vertex 1 -> vn 2 = (0,1,0)
	EXPECT_FLOAT_EQ(r.mesh.normals[6], 1.0f);   // vertex 2 -> vn 3 = (1,0,0)
}

TEST(ObjTest, ASharedPositionWithDifferentNormalsIsSplitButWithTheSameNormalIsShared) {
	// Hard edge: positions 2 and 3 are used by both triangles with DIFFERENT
	// normals, so each is duplicated (3 + 3 distinct (position,normal) pairs).
	const LoadResult hard = parseObj(
		"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvn 0 0 1\nvn 0 0 -1\n"
		"f 1//1 2//1 3//1\nf 2//2 4//2 3//2\n");
	ASSERT_TRUE(hard.ok) << hard.error;
	EXPECT_EQ(hard.mesh.vertexCount(), 6u);
	// Smooth edge: same normal on both sides -> the shared positions stay shared.
	const LoadResult smooth = parseObj(
		"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvn 0 0 1\n"
		"f 1//1 2//1 3//1\nf 2//1 4//1 3//1\n");
	ASSERT_TRUE(smooth.ok) << smooth.error;
	EXPECT_EQ(smooth.mesh.vertexCount(), 4u);
	EXPECT_EQ(smooth.mesh.normals.size(), 12u);
}

TEST(ObjTest, NoNormalReferencesLeavesNormalsEmptyAndPositionsShared) {
	const LoadResult r = parseObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1 2 3\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
	EXPECT_TRUE(r.mesh.normals.empty());
}

TEST(ObjTest, TextureCoordinatesBecomePerVertexByDuplicatingSharedPositions) {
	// Two triangles share positions 1 and 3 but give them DIFFERENT uvs, which a
	// per-vertex mesh can only represent by duplicating those positions.
	const LoadResult r = parseObj(
		"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
		"vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\nvt 0.5 0.5\n"
		"f 1/1 2/2 3/3\n"
		"f 2/2 4/4 3/5\n");
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.mesh.indices.size(), 6u);
	EXPECT_EQ(r.mesh.uvs.size(), r.mesh.vertexCount() * 2);
	// Position 3 is used with uv 3 and uv 5 -> two distinct vertices.
	EXPECT_NE(r.mesh.indices[2], r.mesh.indices[5]);
	// Position 2 is used twice with the SAME uv -> shared.
	EXPECT_EQ(r.mesh.indices[1], r.mesh.indices[3]);
	EXPECT_EQ(r.mesh.vertexCount(), 5u);
	const int dup = r.mesh.indices[5];
	EXPECT_FLOAT_EQ(r.mesh.uvs[dup * 2 + 0], 0.5f);
	EXPECT_FLOAT_EQ(r.mesh.uvs[dup * 2 + 1], 0.5f);
	EXPECT_FLOAT_EQ(r.mesh.positions[dup * 3 + 0], 0.0f);
	EXPECT_FLOAT_EQ(r.mesh.positions[dup * 3 + 1], 1.0f);
}

TEST(ObjTest, ACornerWithNoNormalGetsItsFaceNormalNotPlusZ) {
	// Mixed file: face 1 carries normals (all +X, deliberately not +Z), face 2 has none and lies in the
	// XY plane, so its fallback must be +Z from the geometry; a placeholder of "+Z in object space"
	// would be indistinguishable, hence the second face is wound to face -Z.
	const LoadResult r = parseObj(R"(v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
v 1 0 1
v 0 1 1
vn 1 0 0
f 1//1 2//1 3//1
f 4 6 5
)");
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.mesh.normals.size(), r.mesh.positions.size());
	// corners 3..5 belong to the normal-less face (cross of (0,1,0)x(1,0,0) = -Z)
	EXPECT_NEAR(r.mesh.normals[3 * 3 + 2], -1.0f, 1e-6f);
	EXPECT_NEAR(r.mesh.normals[4 * 3 + 2], -1.0f, 1e-6f);
	EXPECT_NEAR(r.mesh.normals[5 * 3 + 2], -1.0f, 1e-6f);
	// the authored face keeps its normal
	EXPECT_FLOAT_EQ(r.mesh.normals[0], 1.0f);
}

TEST(ObjTest, NegativeIndicesCountBackFromTheEnd) {
	const LoadResult r = parseObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n");
	ASSERT_TRUE(r.ok) << r.error;
	const std::vector<int> expected = {0, 1, 2};
	EXPECT_EQ(r.mesh.indices, expected);
}

TEST(ObjTest, CommentsGroupsAndMaterialRecordsAreIgnored) {
	const LoadResult r = parseObj(
		"# comment\nmtllib x.mtl\no thing\ng part\ns off\nusemtl m\n"
		"v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n");   // CRLF too
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
}

TEST(ObjTest, OutOfRangeFaceIndexIsRejectedWithTheLineNamed) {
	const LoadResult r = parseObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("line 4"), std::string::npos) << r.error;
	EXPECT_NE(r.error.find("9"), std::string::npos) << r.error;
}

TEST(ObjTest, AVertexlessFileIsRejected) {
	const LoadResult r = parseObj("# nothing here\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("no vertices"), std::string::npos) << r.error;
}

TEST(ObjTest, LoadFileChoosesTheObjReaderByExtension) {
	const std::string path = "ply_mesh_tests_tmp_cube_face.OBJ";   // upper-case on purpose
	{
		std::ofstream out(path, std::ios::binary);
		out << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
	}
	const LoadResult r = loadFile(path);
	std::remove(path.c_str());
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
}

// ---- "file.obj#material" group loading ---------------------------------------

namespace {
// Two quads sharing an edge, split across two materials, plus one stray face
// before any usemtl: 3 groups ("" / red / blue), each compacted on its own.
const char *kGroupedObj =
	"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 2 1 0\nv 5 5 5\n"
	"vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
	"f 7 7 7\n"
	"usemtl red\nf 1/1 2/2 3/3 4/4\n"
	"usemtl blue\nf 2/2 5/1 6/3 3/4\n"
	"usemtl red\nf 1/1 2/2 3/3\n";

LoadResult loadGroup(const char *obj, const std::string &group) {
	const std::string path = "ply_mesh_tests_tmp_groups.obj";
	{
		std::ofstream out(path, std::ios::binary);
		out << obj;
	}
	LoadResult r = loadFile(path + "#" + group);
	clearObjGroupCache();
	std::remove(path.c_str());
	return r;
}
} // namespace

TEST(ObjGroupTest, EachMaterialGetsOnlyItsOwnFacesAndVertices) {
	std::vector<ObjGroup> groups;
	std::string error;
	ASSERT_TRUE(parseObjGroups(kGroupedObj, /*splitByMaterial=*/true, groups, error)) << error;
	ASSERT_EQ(groups.size(), 3u);
	EXPECT_EQ(groups[0].name, "");
	EXPECT_EQ(groups[0].mesh.triangleCount(), 1u);
	EXPECT_EQ(groups[1].name, "red");
	EXPECT_EQ(groups[1].mesh.triangleCount(), 3u);   // the quad's two + the later triangle
	EXPECT_EQ(groups[1].mesh.vertexCount(), 4u);     // corners 1..4, shared across both red usemtl runs
	EXPECT_EQ(groups[1].mesh.uvs.size(), 8u);
	EXPECT_EQ(groups[2].name, "blue");
	EXPECT_EQ(groups[2].mesh.triangleCount(), 2u);
	EXPECT_EQ(groups[2].mesh.vertexCount(), 4u);
}

TEST(ObjGroupTest, ThePlainReaderStillIgnoresUsemtl) {
	const LoadResult r = parseObj(kGroupedObj);
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u + 3u + 2u);
}

TEST(ObjGroupTest, HashSuffixSelectsAMaterial) {
	const LoadResult blue = loadGroup(kGroupedObj, "blue");
	ASSERT_TRUE(blue.ok) << blue.error;
	EXPECT_EQ(blue.mesh.triangleCount(), 2u);
	const LoadResult stray = loadGroup(kGroupedObj, "");
	ASSERT_TRUE(stray.ok) << stray.error;
	EXPECT_EQ(stray.mesh.triangleCount(), 1u);
}

TEST(ObjGroupTest, UnknownMaterialIsAnErrorNamingIt) {
	const LoadResult r = loadGroup(kGroupedObj, "green");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("green"), std::string::npos) << r.error;
}

TEST(ObjGroupTest, AGroupCanBeLoadedTwiceWhileTheCacheIsLive) {
	const std::string path = "ply_mesh_tests_tmp_groups_twice.obj";
	{
		std::ofstream out(path, std::ios::binary);
		out << kGroupedObj;
	}
	const LoadResult a = loadFile(path + "#red");
	const LoadResult b = loadFile(path + "#red");   // CPU then GPU build of one scene
	clearObjGroupCache();
	std::remove(path.c_str());
	ASSERT_TRUE(a.ok) << a.error;
	ASSERT_TRUE(b.ok) << b.error;
	EXPECT_EQ(a.mesh.indices, b.mesh.indices);
	EXPECT_EQ(a.mesh.positions, b.mesh.positions);
}

TEST(ObjGroupTest, TheCacheSurvivesUntilTheLastLeaseIsReleased) {
	// Two scene loads overlapping (CPU and GPU builds of one scene): the first to finish
	// must not empty the cache under the second.
	const std::string path = "ply_mesh_tests_tmp_groups_lease.obj";
	{
		std::ofstream out(path, std::ios::binary);
		out << kGroupedObj;
	}
	{
		ObjGroupCacheLease outer;
		{
			ObjGroupCacheLease inner;
			ASSERT_TRUE(loadFile(path + "#red").ok);
		}
		std::remove(path.c_str());   // only the cache can serve this now
		const LoadResult stillServed = loadFile(path + "#blue");
		EXPECT_TRUE(stillServed.ok) << stillServed.error;
	}
	const LoadResult cleared = loadFile(path + "#blue");
	EXPECT_FALSE(cleared.ok) << "the last lease should have freed the cache";
}

