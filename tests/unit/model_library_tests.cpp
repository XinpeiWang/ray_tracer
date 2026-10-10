// model_library_tests.cpp - src/shared/model_library.h: the model catalog and the rule that sets a library model down on the floor.
#include <gtest/gtest.h>

#include "../../src/shared/model_library.h"

#include <cmath>
#include <filesystem>
#include <set>
#include <string>

TEST(ModelLibraryTest, TheCatalogHasUniqueStemsAndNames) {
	std::set<std::string> stems, names;
	for (const model_library::Entry& e : model_library::catalog()) {
		EXPECT_TRUE(stems.insert(e.stem).second) << "duplicate stem " << e.stem;
		EXPECT_TRUE(names.insert(e.name).second) << "duplicate name " << e.name;
		EXPECT_GT(e.kiloTriangles, 0) << e.stem;
	}
	EXPECT_GE(stems.size(), 20u);
	EXPECT_NE(model_library::findEntry("spot"), nullptr);
	EXPECT_EQ(model_library::findEntry("no-such-model"), nullptr);
}

TEST(ModelLibraryTest, EveryModelThatIsInTheFolderHasAThumbnailAndTheBundledOnesExist) {
	// The tests run from the repository root (ctest sets the working directory); skip quietly when run elsewhere.
	if (!std::filesystem::exists("models/spot.obj")) GTEST_SKIP() << "models/ is not next to the working directory";
	for (const model_library::Entry& e : model_library::catalog()) {
		const bool present = std::filesystem::exists(std::string("models/") + e.stem + ".obj");
		if (e.bundled) EXPECT_TRUE(present) << e.stem << " is flagged as bundled but its file is missing";
		if (present) EXPECT_TRUE(std::filesystem::exists(std::string("models/thumbnails/") + e.stem + ".png")) << e.stem << " has no thumbnail";
	}
}

TEST(ModelLibraryTest, PlacementScalesToTheTargetCentresAndStandsOnTheFloor) {
	// A box 10 x 20 x 40 (largest side 40), lowest point at y = -5, centred at x = 100, z = -30.
	const double lo[3] = {95, -5, -50}, hi[3] = {105, 15, -10};
	const model_library::Placement p = model_library::placeOnFloor(lo, hi, 2.0, 3.0, 1.6);
	EXPECT_DOUBLE_EQ(p.scale, 1.6 / 40.0);
	// After the scale, the box's centre (100, 5, -30) must land on (2, ?, 3) and its lowest point (y = -5) on y = 0.
	EXPECT_NEAR(100.0 * p.scale + p.offset[0], 2.0, 1e-12);
	EXPECT_NEAR(-30.0 * p.scale + p.offset[2], 3.0, 1e-12);
	EXPECT_NEAR(-5.0 * p.scale + p.offset[1], 0.0, 1e-12);
}

TEST(ModelLibraryTest, ADegenerateBoxKeepsScaleOne) {
	const double lo[3] = {1, 1, 1}, hi[3] = {1, 1, 1};
	EXPECT_DOUBLE_EQ(model_library::placeOnFloor(lo, hi, 0, 0).scale, 1.0);
}
