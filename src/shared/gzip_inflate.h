#pragma once
// gzip_inflate.h -- decompresses gzip (RFC 1952) data.
//
// WHY THIS EXISTS
// ---------------
// Nearly all the PLY geometry in pbrt's published scene collection ships
// gzipped - ganesha, dragon, sssdragon and the rest are all `.ply.gz`. Without
// this, loading any of them requires decompressing by hand first, which makes
// "point the renderer at a scene folder" a lie.
//
// NO NEW DEPENDENCY
// -----------------
// stb_image.h is already vendored here and already linked into every binary
// that could want this (its implementation lives in stb_image_impl.cpp, which
// is compiled into cpu_renderer.lib, which the tests, the launcher and the
// metadata DLL all link). PNG is a deflate stream, so stb carries a complete
// inflate implementation and exposes it - all that is missing is the gzip
// wrapper around it, which is a header, a deflate stream, and a trailer.
//
// Declarations only: including stb_image.h without STB_IMAGE_IMPLEMENTATION
// gives prototypes, exactly as rtw_stb_image.h and scene_builder.cpp do.

#include <cstdint>
#include <cstdlib>
#include <string>

#include "../external/stb_image.h"

namespace gzip {

// The two magic bytes every gzip stream starts with. Sniffing content rather
// than trusting the extension: a scene is free to name a compressed file .ply,
// and some do.
inline bool looksGzipped(const std::string &bytes) {
	return bytes.size() >= 2 &&
		   static_cast<unsigned char>(bytes[0]) == 0x1F &&
		   static_cast<unsigned char>(bytes[1]) == 0x8B;
}

// Decompresses `in` into `out`. Returns false and sets `error` on malformed
// input rather than throwing or returning a truncated result - a half-read
// mesh is worse than a refusal, because it renders.
inline bool inflate(const std::string &in, std::string &out, std::string &error) {
	out.clear();
	if (!looksGzipped(in)) { error = "not a gzip stream"; return false; }
	if (in.size() < 18) { error = "gzip stream is too short to be valid"; return false; }

	const unsigned char *p = reinterpret_cast<const unsigned char *>(in.data());
	if (p[2] != 8) { error = "gzip uses an unsupported compression method"; return false; }

	const unsigned char flags = p[3];
	std::size_t at = 10;                     // magic(2) CM(1) FLG(1) MTIME(4) XFL(1) OS(1)

	// The optional header fields, in the order RFC 1952 defines them. Each is
	// skipped rather than used; only their lengths matter.
	if (flags & 0x04) {                      // FEXTRA: 2-byte length then data
		if (at + 2 > in.size()) { error = "gzip header is truncated (FEXTRA)"; return false; }
		const std::size_t xlen = static_cast<std::size_t>(p[at]) |
								 (static_cast<std::size_t>(p[at + 1]) << 8);
		at += 2 + xlen;
	}
	if (flags & 0x08) {                      // FNAME: NUL-terminated
		while (at < in.size() && p[at] != 0) ++at;
		++at;
	}
	if (flags & 0x10) {                      // FCOMMENT: NUL-terminated
		while (at < in.size() && p[at] != 0) ++at;
		++at;
	}
	if (flags & 0x02) at += 2;               // FHCRC

	if (at + 8 > in.size()) { error = "gzip header is truncated"; return false; }

	const std::size_t deflateBytes = in.size() - at - 8;
	if (deflateBytes > static_cast<std::size_t>(INT32_MAX)) {
		error = "gzip stream is larger than this decoder can address";
		return false;
	}

	// The last four bytes are the uncompressed size mod 2^32, and using it to
	// size the first allocation turns a 130 MB mesh from a dozen
	// reallocate-and-copy rounds into one or two.
	//
	// It is a number from an untrusted file and it must be treated as one. On
	// a TRUNCATED stream those four bytes are not the trailer at all - they
	// are whatever compressed data happened to land at the end - and reading
	// them as a length asks for an allocation of that size. A test that
	// truncates a stream duly sent this to 1.99 GB before it was bounded.
	//
	// DEFLATE cannot expand data by more than about 1032x, so the compressed
	// size is a real upper bound on what any honest header could claim, and a
	// fixed ceiling caps the rest. Both are only hints: stb grows the buffer
	// when the file turns out to be bigger, so being wrong costs one memcpy.
	const unsigned char *tail = p + in.size() - 4;
	std::uint64_t isize = static_cast<std::uint64_t>(tail[0]) |
						  (static_cast<std::uint64_t>(tail[1]) << 8) |
						  (static_cast<std::uint64_t>(tail[2]) << 16) |
						  (static_cast<std::uint64_t>(tail[3]) << 24);

	constexpr std::uint64_t kMaxDeflateRatio = 1032;
	constexpr std::uint64_t kInitialCeiling = 256ull << 20;
	const std::uint64_t plausible =
		static_cast<std::uint64_t>(deflateBytes) * kMaxDeflateRatio + 64;
	if (isize == 0 || isize > plausible) isize = plausible;
	if (isize > kInitialCeiling) isize = kInitialCeiling;
	if (isize < 1024) isize = 1024;

	// DECODING INTO A FIXED BUFFER, NOT A GROWING ONE
	// -----------------------------------------------
	// stb's malloc-based inflate does not terminate on truncated input. Past
	// the end of the buffer its byte reader returns zeros rather than
	// signalling exhaustion, and a run of zero bits decodes as an endless
	// sequence of literals, so the output buffer doubles until the process
	// dies. A test that truncates a stream hung here and had to be killed at
	// 1.99 GB; bounding the size hint alone did not fix it, because the growth
	// happens inside the decoder.
	//
	// The fixed-buffer variant has the stopping condition the malloc one
	// lacks: it fails when the output does not fit. That turns "runs until the
	// machine gives up" into "returns an error", which is what a corrupt file
	// should do.
	const auto tryDecode = [&](std::uint64_t capacity) -> bool {
		std::string buf(static_cast<std::size_t>(capacity), '\0');
		const int n = stbi_zlib_decode_noheader_buffer(
			&buf[0], static_cast<int>(capacity),
			in.data() + at, static_cast<int>(deflateBytes));
		if (n < 0) return false;
		buf.resize(static_cast<std::size_t>(n));
		out.swap(buf);
		return true;
	};

	if (tryDecode(isize)) return true;

	// The trailer's size can be wrong on a file that is otherwise intact, so
	// one retry at the ratio bound distinguishes "the hint was low" from "the
	// data is corrupt" - without ever exceeding the ceiling.
	const std::uint64_t retry = (plausible > kInitialCeiling) ? kInitialCeiling : plausible;
	if (retry > isize && tryDecode(retry)) return true;

	error = "gzip stream could not be decompressed (truncated or corrupt)";
	return false;
}

} // namespace gzip
