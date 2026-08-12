/**
 * @file gzip_inflate_tests.cpp
 * @brief Unit tests for gzip decompression of scene geometry
 *
 * The fixtures are real gzip streams produced by an independent compressor
 * (.NET's GZipStream), not bytes this code round-trips through itself. A
 * decoder tested only against its own encoder agrees with itself and with
 * nothing else; the whole point here is to read files other tools wrote.
 */

#include <gtest/gtest.h>

#include "gzip_inflate.h"
#include "ply_mesh.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// A gzipped ASCII PLY - one triangle - compressed by .NET's GZipStream.
const unsigned char kGzippedPly[] = {
	0x1F,0x8B,0x08,0x00,0x00,0x00,0x00,0x00,0x04,0x00,0x5D,0x8D,0xE1,0x0A,0xC3,0x20,
	0x0C,0x84,0xFF,0xE7,0x29,0xEE,0x09,0x86,0xAE,0x0F,0x54,0x82,0x46,0x1A,0xB0,0x2A,
	0xEA,0x46,0xDD,0xD3,0xCF,0xF5,0x47,0x07,0xE5,0x20,0x77,0x7C,0x09,0xB9,0x12,0x07,
	0x85,0x5C,0x77,0xEE,0xE0,0xE6,0x54,0x61,0x1F,0x86,0x24,0xCA,0x2E,0xA9,0xE3,0x2D,
	0xB5,0xCB,0x81,0x85,0x4A,0xCD,0x65,0xE6,0x81,0x10,0xF3,0xBC,0x3C,0xEE,0x60,0xDC,
	0xC1,0xE7,0xFA,0x11,0xD8,0x09,0xEC,0x7F,0x1F,0xB5,0x75,0xBC,0xDC,0xC6,0x15,0x7A,
	0x55,0xAC,0x9A,0xBC,0x3A,0x69,0x24,0xC9,0xAF,0x9B,0xB0,0x97,0x4A,0x06,0x53,0x64,
	0xCF,0x69,0x30,0x9D,0x16,0xFC,0xFC,0x49,0x5F,0xB1,0x74,0x78,0xF9,0xB4,0x00,0x00,
	0x00
};

std::string gzippedPly() {
	return std::string(reinterpret_cast<const char *>(kGzippedPly), sizeof(kGzippedPly));
}

} // namespace

TEST(GzipTest, DecompressesAStreamWrittenByAnotherTool) {
	std::string out, error;
	ASSERT_TRUE(gzip::inflate(gzippedPly(), out, error)) << error;
	EXPECT_EQ(out.substr(0, 3), "ply");
	EXPECT_NE(out.find("end_header"), std::string::npos);
	EXPECT_NE(out.find("3 0 1 2"), std::string::npos);
}

TEST(GzipTest, RecognisesGzipByItsMagicBytesNotItsName) {
	EXPECT_TRUE(gzip::looksGzipped(gzippedPly()));
	EXPECT_FALSE(gzip::looksGzipped("ply\nformat ascii 1.0\n"));
	EXPECT_FALSE(gzip::looksGzipped(""));
	EXPECT_FALSE(gzip::looksGzipped("\x1F"));       // one byte of the magic
}

TEST(GzipTest, PlainDataIsRefusedRatherThanMisread) {
	std::string out, error;
	EXPECT_FALSE(gzip::inflate("ply\nformat ascii 1.0\n", out, error));
	EXPECT_FALSE(error.empty());
}

TEST(GzipTest, ATruncatedStreamFailsInsteadOfReturningPartialData) {
	// A partially decompressed mesh is worse than none: it renders, so the
	// corruption shows up as geometry rather than as an error.
	std::string half = gzippedPly().substr(0, sizeof(kGzippedPly) / 2);
	std::string out, error;
	EXPECT_FALSE(gzip::inflate(half, out, error));
	EXPECT_TRUE(out.empty());
}

TEST(GzipTest, AHeaderThatClaimsMoreFieldsThanItHasIsRejected) {
	// FEXTRA set with nothing following it. Reading the length field off the
	// end of the buffer is the classic way a decoder like this crashes.
	std::string bad = gzippedPly();
	bad[3] = 0x04;                                   // FLG = FEXTRA
	std::string out, error;
	EXPECT_FALSE(gzip::inflate(bad, out, error));
}

TEST(GzipTest, AnUnsupportedCompressionMethodIsNamedNotGuessedAt) {
	std::string bad = gzippedPly();
	bad[2] = 0x09;                                   // CM must be 8 (deflate)
	std::string out, error;
	EXPECT_FALSE(gzip::inflate(bad, out, error));
	EXPECT_NE(error.find("compression method"), std::string::npos) << error;
}

TEST(GzipTest, AHugeClaimedSizeDoesNotCauseAHugeAllocation) {
	// This is the test that matters most in this file, and it was written
	// after the truncation test above drove the process to 1.99 GB and had to
	// be killed. A gzip trailer claims the uncompressed size; on a truncated
	// stream those bytes are just compressed data, and reading them as a
	// length asks for an allocation of whatever they happen to say.
	//
	// The bound is the compressed size times DEFLATE's maximum expansion
	// ratio. A 129-byte input cannot honestly claim four gigabytes.
	std::string lying = gzippedPly();
	lying[lying.size() - 4] = static_cast<char>(0xFF);
	lying[lying.size() - 3] = static_cast<char>(0xFF);
	lying[lying.size() - 2] = static_cast<char>(0xFF);
	lying[lying.size() - 1] = static_cast<char>(0xFF);   // claims ~4 GB

	std::string out, error;
	// Either outcome is acceptable - what is not acceptable is exhausting
	// memory on the way to it.
	gzip::inflate(lying, out, error);
	EXPECT_LT(out.size(), 1u << 20)
		<< "a 129-byte stream produced " << out.size() << " bytes";
}

TEST(GzipTest, ALyingSizeHintDoesNotBreakTheResult) {
	// The trailer's uncompressed-size field is used to size the first
	// allocation. It comes from the file, so a wrong value must cost
	// performance and nothing else.
	std::string lying = gzippedPly();
	lying[lying.size() - 4] = 0x01;                  // ISIZE = something small
	lying[lying.size() - 3] = 0x00;
	lying[lying.size() - 2] = 0x00;
	lying[lying.size() - 1] = 0x00;

	std::string out, error;
	ASSERT_TRUE(gzip::inflate(lying, out, error)) << error;
	EXPECT_NE(out.find("end_header"), std::string::npos)
		<< "a bad size hint truncated the output";
}

namespace {

class GzipPlyFile : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		dir_ = std::string(tmp ? tmp : ".") + "/gzip_ply_tests/";
		std::error_code ec;
		std::filesystem::create_directories(dir_, ec);
	}
	void TearDown() override {
		std::error_code ec;
		std::filesystem::remove_all(dir_, ec);
	}
	void writeBytes(const std::string &name, const std::string &bytes) {
		std::ofstream out(dir_ + name, std::ios::binary);
		out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
	}
	std::string dir_;
};

} // namespace

TEST_F(GzipPlyFile, LoadFileReadsAGzippedMeshTransparently) {
	writeBytes("tri.ply.gz", gzippedPly());
	const ply_mesh::LoadResult r = ply_mesh::loadFile(dir_ + "tri.ply.gz");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
	EXPECT_EQ(r.mesh.vertexCount(), 3u);
}

TEST_F(GzipPlyFile, AGzippedFileNamedPlyIsStillRead) {
	// Sniffing content rather than the extension is what makes this work, and
	// scenes in the wild do exactly this.
	writeBytes("tri.ply", gzippedPly());
	const ply_mesh::LoadResult r = ply_mesh::loadFile(dir_ + "tri.ply");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
}

TEST_F(GzipPlyFile, APlainPlyStillLoadsUnchanged) {
	writeBytes("plain.ply",
			   "ply\nformat ascii 1.0\n"
			   "element vertex 3\n"
			   "property float x\nproperty float y\nproperty float z\n"
			   "element face 1\nproperty list uchar int vertex_indices\n"
			   "end_header\n"
			   "0 0 0\n1 0 0\n0 1 0\n"
			   "3 0 1 2\n");
	const ply_mesh::LoadResult r = ply_mesh::loadFile(dir_ + "plain.ply");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.mesh.triangleCount(), 1u);
}

TEST_F(GzipPlyFile, ACorruptGzipReportsThePathAlongsideTheReason) {
	std::string corrupt = gzippedPly();
	corrupt[40] ^= 0xFF;                             // damage the deflate data
	writeBytes("broken.ply.gz", corrupt);
	const ply_mesh::LoadResult r = ply_mesh::loadFile(dir_ + "broken.ply.gz");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("broken.ply.gz"), std::string::npos) << r.error;
}
