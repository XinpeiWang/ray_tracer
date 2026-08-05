#pragma once
// ---------------------------------------------------------------------------
// pbrt_hash.h -- Hashing utilities
//
// Mirrors pbrt-v4 util/hash.h (Apache-2.0), adapted for CPU-only use.
//
// Components:
//   MurmurHash64A(key, len, seed)  -- MurmurHash2 64-bit finalizer
//   MixBits(v)                     -- fast 64-bit avalanche mixer
//   HashBuffer<T>(ptr, n, seed)    -- hash a typed array
//   Hash<Args...>(args...)         -- hash arbitrary value pack by-value
//   HashFloat<Args...>(args...)    -- Hash -> float in [0, 1)
//
// Design rules:
//   - No pbrt includes; only <cstdint> and <cstring>
//   - CPU_GPU macro preserved for future CUDA use
//   - std::memcpy used for safe type-punning (UB-free)
//
// References:
//   pbrt-v4 src/pbrt/util/hash.h  (Apache-2.0)
//   https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
//   http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <cstdint>
#include <cstring>

// ===========================================================================
// MurmurHash64A
// 64-bit MurmurHash2 (Austin Appleby).  Processes 8 bytes at a time.
// ===========================================================================
CPU_GPU uint64_t MurmurHash64A(const unsigned char* key, size_t len,
								uint64_t seed) {
	const uint64_t m = 0xc6a4a7935bd1e995ull;
	const int r = 47;

	uint64_t h = seed ^ (len * m);

	const unsigned char* end = key + 8 * (len / 8);

	while (key != end) {
		uint64_t k;
		std::memcpy(&k, key, sizeof(uint64_t));
		key += 8;

		k *= m;
		k ^= k >> r;
		k *= m;

		h ^= k;
		h *= m;
	}

	switch (len & 7) {
	case 7: h ^= uint64_t(key[6]) << 48; [[fallthrough]];
	case 6: h ^= uint64_t(key[5]) << 40; [[fallthrough]];
	case 5: h ^= uint64_t(key[4]) << 32; [[fallthrough]];
	case 4: h ^= uint64_t(key[3]) << 24; [[fallthrough]];
	case 3: h ^= uint64_t(key[2]) << 16; [[fallthrough]];
	case 2: h ^= uint64_t(key[1]) <<  8; [[fallthrough]];
	case 1: h ^= uint64_t(key[0]);
			h *= m;
	}

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	return h;
}

// ===========================================================================
// MixBits
// Fast 64-bit avalanche mixer (Murmur3 finalizer variant).
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
// ===========================================================================
CPU_GPU uint64_t MixBits(uint64_t v) {
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185ull;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44dull;
	v ^= (v >> 33);
	return v;
}

// ===========================================================================
// HashBuffer<T>
// Hash nElements contiguous objects of type T.
// ===========================================================================
template <typename T>
CPU_GPU uint64_t HashBuffer(const T* ptr, size_t nElements, uint64_t seed = 0) {
	return MurmurHash64A(reinterpret_cast<const unsigned char*>(ptr),
						 nElements * sizeof(T), seed);
}

// ===========================================================================
// Hash<Args...>
// Hash an arbitrary pack of trivially-copyable values.
// Each argument is memcpy'd into a stack buffer; the whole buffer is then
// passed to MurmurHash64A.  Matches pbrt-v4 behaviour exactly.
// ===========================================================================
namespace detail {

// Base case: nothing to copy.
CPU_GPU void hash_recursive_copy(char*) {}

// Copy v into buf, then recurse on remaining args.
template <typename T, typename... Args>
CPU_GPU void hash_recursive_copy(char* buf, T v, Args... args) {
	std::memcpy(buf, &v, sizeof(T));
	hash_recursive_copy(buf + sizeof(T), args...);
}

} // namespace detail

template <typename... Args>
CPU_GPU uint64_t Hash(Args... args) {
	// Total byte size of all arguments (C++17 fold expression).
	constexpr size_t sz = (sizeof(Args) + ... + 0);
	constexpr size_t n  = (sz + 7) / 8;   // number of uint64_t words needed
	uint64_t buf[n == 0 ? 1 : n];          // avoid zero-length array
	detail::hash_recursive_copy(reinterpret_cast<char*>(buf), args...);
	return MurmurHash64A(reinterpret_cast<const unsigned char*>(buf), sz, 0);
}

// ===========================================================================
// HashFloat<Args...>
// Maps Hash result to a float in [0, 1) using the low 32 bits.
// Matches pbrt-v4: uint32_t(Hash(args...)) * 0x1p-32f
// ===========================================================================
template <typename... Args>
CPU_GPU float HashFloat(Args... args) {
	return static_cast<uint32_t>(Hash(args...)) * 0x1p-32f;
}
