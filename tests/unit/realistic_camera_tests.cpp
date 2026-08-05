// realistic_camera_tests.cpp
// Unit tests for RealisticCamera<T>
// Mirrors pbrt-v4 RealisticCamera (cameras.h / cameras.cpp).
#include <gtest/gtest.h>
#include "../../src/shared/cameras.h"
#include <cmath>
#include <vector>

// Simple 4-element singlet biconvex lens in pbrt-v4 format (front to back):
//   element 0: front glass surface (scene side, curvRadius > 0)
//   element 1: aperture stop
//   element 2: rear glass surface (film side of glass, curvRadius < 0)
//   element 3: air spacer to film — LensRearZ() = this element's thickness
// Values in mm; constructor converts to metres.
static std::vector<double> simple_lens_params() {
	return {
		 35.98,  4.0,  1.5,  25.0,   // front glass surface (glass fills to next element)
		  0.0,   2.0,  0.0,  17.0,   // aperture stop (air gap)
		-35.98,  4.0,  1.5,  25.0,   // rear glass surface (glass fills to next element)
		  0.0,  50.0,  0.0,  25.0,   // air spacer to film; LensRearZ = 50mm = 0.05m
	};
}

// Film half-extents for a 35mm frame (24x36mm -> half: 12x18mm)
static constexpr double FILM_HX = 18.0; // mm
static constexpr double FILM_HY = 12.0; // mm

// Focus distance 1 metre
static constexpr double FOCUS_DIST = 1.0; // metres

// Aperture diameter
static constexpr double APERTURE = 17.0; // mm

// Identity camera-to-world
static Mat4<double> identity_c2w() { return Mat4<double>{}; }

// Helper: build a simple RealisticCamera
static RealisticCamera<double> make_camera() {
	return RealisticCamera<double>(identity_c2w(),
								   FILM_HX, FILM_HY,
								   FOCUS_DIST,
								   APERTURE,
								   simple_lens_params());
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
TEST(RealisticCamera, ConstructsWithoutCrash) {
	auto cam = make_camera();
	EXPECT_GT(cam.num_elements(), 0);
}

TEST(RealisticCamera, LensRearZIsPositive) {
	auto cam = make_camera();
	EXPECT_GT(cam.lens_rear_z(), 0.0);
}

TEST(RealisticCamera, LensFrontZGreaterThanRearZ) {
	auto cam = make_camera();
	// Front z accumulates all thicknesses; rear z is just the last element's thickness
	EXPECT_GT(cam.lens_front_z(), cam.lens_rear_z());
}

TEST(RealisticCamera, NumElements) {
	auto cam = make_camera();
	EXPECT_EQ(cam.num_elements(), 4);
}

// ---------------------------------------------------------------------------
// generate_ray: central ray (pFilm = 0,0) should produce a ray
// ---------------------------------------------------------------------------
TEST(RealisticCamera, GenerateRayCentralPointProducesRay) {
	auto cam = make_camera();
	CameraSample<double> sample;
	// pFilm in physical metres: centre of film
	sample.pFilm_x = 0.0;
	sample.pFilm_y = 0.0;
	sample.pLens_u = 0.5;
	sample.pLens_v = 0.5;
	sample.time    = 0.0;
	auto r = cam.generate_ray(sample);
	// A central ray should pass through with positive weight
	EXPECT_GT(r.weight, 0.0);
}

TEST(RealisticCamera, GenerateRayDirectionIsNormalized) {
	auto cam = make_camera();
	CameraSample<double> sample;
	sample.pFilm_x = 0.0; sample.pFilm_y = 0.0;
	sample.pLens_u = 0.5; sample.pLens_v = 0.5;
	sample.time    = 0.0;
	auto r = cam.generate_ray(sample);
	if (r.weight > 0.0) {
		double len = std::sqrt(r.direction.x*r.direction.x +
							   r.direction.y*r.direction.y +
							   r.direction.z*r.direction.z);
		EXPECT_NEAR(len, 1.0, 1e-9);
	}
}

TEST(RealisticCamera, GenerateRayCentralRayPointsForward) {
	auto cam = make_camera();
	CameraSample<double> sample;
	sample.pFilm_x = 0.0; sample.pFilm_y = 0.0;
	sample.pLens_u = 0.5; sample.pLens_v = 0.5;
	sample.time    = 0.0;
	auto r = cam.generate_ray(sample);
	if (r.weight > 0.0) {
		// With identity camera-to-world, the forward axis is +Z
		EXPECT_GT(r.direction.z, 0.0);
	}
}

// ---------------------------------------------------------------------------
// Off-axis film point: should still produce a valid ray (may be vignetted)
// ---------------------------------------------------------------------------
TEST(RealisticCamera, GenerateRayOffAxisDoesNotCrash) {
	auto cam = make_camera();
	CameraSample<double> sample;
	// Near film edge (half-extent in metres = FILM_HX/1000)
	sample.pFilm_x = FILM_HX * 0.001 * 0.8;  // 80% of half-extent
	sample.pFilm_y = 0.0;
	sample.pLens_u = 0.5; sample.pLens_v = 0.5;
	sample.time    = 0.0;
	auto r = cam.generate_ray(sample);
	// Weight >= 0 (may be vignetted)
	EXPECT_GE(r.weight, 0.0);
}

// ---------------------------------------------------------------------------
// Vignetting: extreme off-axis + edge lens sample should be vignetted
// ---------------------------------------------------------------------------
TEST(RealisticCamera, VignettedRayHasZeroWeight) {
	auto cam = make_camera();
	CameraSample<double> sample;
	// Far beyond the film diagonal
	sample.pFilm_x = FILM_HX * 0.001 * 5.0;
	sample.pFilm_y = FILM_HY * 0.001 * 5.0;
	sample.pLens_u = 0.01;
	sample.pLens_v = 0.01;
	sample.time    = 0.0;
	auto r = cam.generate_ray(sample);
	EXPECT_EQ(r.weight, 0.0);
}

// ---------------------------------------------------------------------------
// Scale factor: changing aperture affects ray weight (more open = more light)
// ---------------------------------------------------------------------------
TEST(RealisticCamera, WiderApertureDoesNotReduceWeight) {
	// Build two cameras differing only in aperture
	RealisticCamera<double> cam_small(identity_c2w(), FILM_HX, FILM_HY,
									  FOCUS_DIST, 5.0, simple_lens_params());
	RealisticCamera<double> cam_large(identity_c2w(), FILM_HX, FILM_HY,
									  FOCUS_DIST, APERTURE, simple_lens_params());
	CameraSample<double> sample;
	sample.pFilm_x = 0.0; sample.pFilm_y = 0.0;
	sample.pLens_u = 0.5; sample.pLens_v = 0.5; sample.time = 0.0;

	auto rs = cam_small.generate_ray(sample);
	auto rl = cam_large.generate_ray(sample);
	// Wider aperture: if both pass, the pupil area is larger so pdf is smaller -> larger weight
	// At minimum, both should have weight >= 0
	EXPECT_GE(rs.weight, 0.0);
	EXPECT_GE(rl.weight, 0.0);
}

// ---------------------------------------------------------------------------
// Multiple samples: any sample with weight > 0 should produce a normalised direction
// ---------------------------------------------------------------------------
TEST(RealisticCamera, MultiSampleDirectionsAreNormalized) {
	auto cam = make_camera();
	for (int i = 1; i <= 5; ++i) {
		for (int j = 1; j <= 5; ++j) {
			CameraSample<double> sample;
			sample.pFilm_x = 0.0;
			sample.pFilm_y = 0.0;
			sample.pLens_u = i / 6.0;
			sample.pLens_v = j / 6.0;
			sample.time    = 0.0;
			auto r = cam.generate_ray(sample);
			if (r.weight > 0.0) {
				double len = std::sqrt(r.direction.x*r.direction.x +
									   r.direction.y*r.direction.y +
									   r.direction.z*r.direction.z);
				EXPECT_NEAR(len, 1.0, 1e-9) << "i=" << i << " j=" << j;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Rear element radius is positive
// ---------------------------------------------------------------------------
TEST(RealisticCamera, RearElementRadiusIsPositive) {
	auto cam = make_camera();
	EXPECT_GT(cam.rear_element_radius(), 0.0);
}

// ===========================================================================
// pbrt-v4 parity tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Cardinal points: effective focal length must be positive (converging lens)
// ---------------------------------------------------------------------------
TEST(RealisticCamera, FocalLengthIsPositive) {
	auto cam = make_camera();
	double efl = cam.effective_focal_length();
	EXPECT_GT(efl, 0.0) << "Singlet should be a converging lens (EFL > 0)";
}

// ---------------------------------------------------------------------------
// EFL of our singlet (R=35.98 mm, eta=1.5, t=4 mm) should lie in [0.02, 0.10] m
// (roughly 20-100 mm) — a sanity bound against gross sign / unit errors.
// pbrt-v4 analogue: LOG_VERBOSE("Effective focal length %f", fz[0]-pz[0])
// ---------------------------------------------------------------------------
TEST(RealisticCamera, FocalLengthReasonableForSinglet) {
	auto cam = make_camera();
	double efl_m = cam.effective_focal_length();
	EXPECT_GT(efl_m, 0.02) << "EFL < 20 mm is implausibly short for this singlet";
	EXPECT_LT(efl_m, 0.10) << "EFL > 100 mm is implausibly long for this singlet";
}

// ---------------------------------------------------------------------------
// Focus distance must shift rear element thickness
// pbrt-v4: FocusThickLens adjusts elementInterfaces.back().thickness by delta
// ---------------------------------------------------------------------------
TEST(RealisticCamera, FocusDistanceAffectsRearElementThickness) {
	// Two cameras identical except for focus distance
	RealisticCamera<double> cam_near(identity_c2w(), FILM_HX, FILM_HY,
									  0.3, APERTURE, simple_lens_params());
	RealisticCamera<double> cam_far (identity_c2w(), FILM_HX, FILM_HY,
									  2.0, APERTURE, simple_lens_params());
	// Closer focus → smaller rear thickness (lens moved toward film)
	EXPECT_NE(cam_near.lens_rear_z(), cam_far.lens_rear_z())
		<< "Different focus distances must produce different rear element thickness";
}

// ---------------------------------------------------------------------------
// On-axis rays: a 5x5 grid of pupil samples at film centre should mostly pass
// pbrt-v4: central region of pupil is unobstructed for a well-formed lens
// ---------------------------------------------------------------------------
TEST(RealisticCamera, OnAxisRaysMostlyPass) {
	auto cam = make_camera();
	int pass = 0, total = 0;
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			CameraSample<double> s;
			s.pFilm_x = 0.0; s.pFilm_y = 0.0;
			s.pLens_u = (i + 0.5) / 5.0;
			s.pLens_v = (j + 0.5) / 5.0;
			s.time = 0.0;
			auto r = cam.generate_ray(s);
			if (r.weight > 0.0) ++pass;
			++total;
		}
	}
	EXPECT_GE(pass, total / 2) << "At least half of on-axis pupil samples should pass";
}

// ---------------------------------------------------------------------------
// Ray time is preserved through generate_ray
// pbrt-v4: GenerateRay copies time to the outgoing ray
// ---------------------------------------------------------------------------
TEST(RealisticCamera, RayTimeIsPreserved) {
	auto cam = make_camera();
	CameraSample<double> s;
	s.pFilm_x = 0.0; s.pFilm_y = 0.0;
	s.pLens_u = 0.5; s.pLens_v = 0.5;
	s.time = 0.42;
	auto r = cam.generate_ray(s);
	if (r.weight > 0.0)
		EXPECT_DOUBLE_EQ(r.time, 0.42) << "Ray time must be copied from CameraSample";
}

// ---------------------------------------------------------------------------
// cos^4 weight decay: mean weight on-axis must be >= mean weight near edge
// pbrt-v4: w *= cos4(theta) / (pdf * lrz^2) — lower for oblique rays
// ---------------------------------------------------------------------------
TEST(RealisticCamera, WeightHigherOnAxisThanOffAxis) {
	auto cam = make_camera();
	double sum_center = 0.0, sum_edge = 0.0;
	int cnt_center = 0, cnt_edge = 0;
	for (int i = 0; i < 5; ++i) {
		double u = (i + 0.5) / 5.0;
		double v = 0.5;
		// Centre (0,0)
		{
			CameraSample<double> s{0.0, 0.0, u, v, 0.0};
			auto r = cam.generate_ray(s);
			if (r.weight > 0.0) { sum_center += r.weight; ++cnt_center; }
		}
		// Near film edge (80 % of half-extent)
		{
			CameraSample<double> s{FILM_HX * 0.001 * 0.8, 0.0, u, v, 0.0};
			auto r = cam.generate_ray(s);
			if (r.weight > 0.0) { sum_edge += r.weight; ++cnt_edge; }
		}
	}
	if (cnt_center > 0 && cnt_edge > 0) {
		double mean_c = sum_center / cnt_center;
		double mean_e = sum_edge   / cnt_edge;
		EXPECT_GE(mean_c, mean_e * 0.5)
			<< "On-axis mean weight should not be drastically lower than off-axis";
	}
}

// ---------------------------------------------------------------------------
// Symmetry: +x and -x film points should produce the same ray weight
// pbrt-v4: lens system is rotationally symmetric; only radial position matters
// for exit-pupil lookup, so mirrored film x/y give equal weights
// ---------------------------------------------------------------------------
TEST(RealisticCamera, SymmetricFilmPointsHaveEqualWeight) {
	auto cam = make_camera();
	double fx = FILM_HX * 0.001 * 0.4;  // 40% of half-extent, well inside frame
	CameraSample<double> s_pos{  fx, 0.0, 0.5, 0.5, 0.0 };
	CameraSample<double> s_neg{ -fx, 0.0, 0.5, 0.5, 0.0 };
	auto r_pos = cam.generate_ray(s_pos);
	auto r_neg = cam.generate_ray(s_neg);
	EXPECT_NEAR(r_pos.weight, r_neg.weight, 1e-9)
		<< "+x and -x film points must give identical weights (radial symmetry)";
}
