// procedural_textures_tests.cpp
// Unit tests for src/shared/procedural_textures.h
//
// procedural_textures.h used to be a much larger port of pbrt-v4's
// non-image procedural textures (Bilerp/Checkerboard/Mix/Scaled/FBm/
// Wrinkled/Windy/Dots/Marble) - trimmed down to just checkerboard_weight_2d
// (real anti-aliased box-filtered checkerboard integration), the one piece
// that wasn't redundant with texture.h's own point-sampled equivalents once
// it was actually wired into the live rendering pipeline
// (uv_checker_texture::value_diff(), texture.h). See procedural_textures.h's
// own file comment for the full history.

#include <gtest/gtest.h>
#include "../../src/shared/procedural_textures.h"

// ===========================================================================
// checkerboard_weight_2d
// ===========================================================================

TEST(ProceduralTextures, CheckerboardWeight2D_SmallFootprint_EvenCell) {
	// s=0.5, t=0.5 is center of cell (0,0) (parity 0) -> weight ~ 0
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	float w = checkerboard_weight_2d<float>(c);
	EXPECT_NEAR(w, 0.f, 0.05f);
}

TEST(ProceduralTextures, CheckerboardWeight2D_SmallFootprint_OddCell) {
	// s=1.5, t=0.5 -> cell (1,0), parity 1 -> weight ~ 1
	TexCoord2D c;
	c.st   = {1.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	float w = checkerboard_weight_2d<float>(c);
	EXPECT_NEAR(w, 1.f, 0.05f);
}

TEST(ProceduralTextures, CheckerboardWeight2D_LargeFootprint_NearHalf) {
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 5.f; c.dsdy = 0.f;
	c.dtdx = 0.f; c.dtdy = 5.f;
	float w = checkerboard_weight_2d<float>(c);
	EXPECT_NEAR(w, 0.5f, 0.1f);
}

TEST(ProceduralTextures, CheckerboardWeight2D_DoubleType) {
	// Same math, T=double - both instantiations are used live (float on
	// GPU-facing code, double on CPU's uv_checker_texture).
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	double w = checkerboard_weight_2d<double>(c);
	EXPECT_NEAR(w, 0.0, 0.05);
}
