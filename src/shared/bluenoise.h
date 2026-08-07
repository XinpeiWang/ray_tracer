#pragma once
//==============================================================================
// bluenoise.h -- Blue-noise lookup table and query function (pbrt-v4 §8.x)
//
// pbrt-v4 reference: src/pbrt/util/bluenoise.h
//                    src/pbrt/util/bluenoise.cpp
//
// The table contains 48 pre-computed 128×128 blue-noise textures derived from
// the HDR reference images at http://momentsingraphics.de/?p=127.
// Each texel is a uint16_t in [0, 65535]; blue_noise() maps it to [0, 1).
//
// Usage:
//   float bn = blue_noise(dimension, pixel_x, pixel_y);
//   // Returns a value in [0,1) with blue-noise spectral properties.
//   // Tiles seamlessly at every 128 pixels; wrap happens automatically.
//   // Use different `dimension` values for independent 1-D blue-noise samples.
//==============================================================================

#include "../data/bluenoise_data.h"   // 48×128×128 uint16_t table

// blue_noise()
//   tableIndex -- which of the 48 independent textures to sample (wraps mod 48)
//   px, py     -- integer pixel coordinates (wraps mod 128)
//   Returns a float in [0, 1).
//
// Direct port of pbrt-v4 BlueNoise(int textureIndex, Point2i p).
inline float blue_noise(int tableIndex, int px, int py) {
	using namespace bluenoise_detail;
	tableIndex = ((tableIndex % NumBlueNoiseTextures) + NumBlueNoiseTextures)
				 % NumBlueNoiseTextures;
	int x = ((px % BlueNoiseResolution) + BlueNoiseResolution) % BlueNoiseResolution;
	int y = ((py % BlueNoiseResolution) + BlueNoiseResolution) % BlueNoiseResolution;
	return BlueNoiseTextures[tableIndex][x][y] / 65535.0f;
}
