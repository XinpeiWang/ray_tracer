// Single compilation unit for tinyexr's implementation.
// This file should be compiled exactly once in the entire project - mirrors
// stb_image_impl.cpp's own pattern for the same reason (a header-only
// library whose IMPLEMENTATION macro must only ever be defined in one .cpp).
//
// TINYEXR_USE_MINIZ stays at its default (1): miniz.c/.h are vendored
// alongside tinyexr.h/exr_reader.hh in this same directory and compiled as
// their own translation unit (see miniz.c) - the zlib-compatible inflate
// tinyexr needs to decompress EXR's ZIP-compressed scanlines.
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
