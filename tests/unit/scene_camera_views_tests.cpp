// scene_camera_views_tests.cpp - src/shared/scene_camera_views.h: saved cameras of the Scene Builder (name, apply, update, rename, remove) and their JSON.
#include <gtest/gtest.h>

#include <string>

#include "../../src/shared/scene_camera_views.h"
#include "../../src/shared/scene_doc_diff.h"

using namespace scene_doc;

namespace {

Camera cameraAt(double x, double y, double z) {
	Camera c;
	c.position = {x, y, z};
	c.target = {0.0, 1.0, 0.0};
	return c;
}

}  // namespace

TEST(SceneCameraViews, SavingKeepsTheWholeCameraUnderANumberedName) {
	Document d;
	Camera c = cameraAt(4, 3, 2);
	c.fov = 55;
	c.lensRadius = 0.2;
	c.focusDistance = 5.5;
	EXPECT_EQ(saveCameraView(d, "Front", c), 0);
	EXPECT_EQ(saveCameraView(d, "Front", cameraAt(0, 1, 8)), 1);
	EXPECT_EQ(saveCameraView(d, "  ", cameraAt(0, 1, 9)), 2);
	ASSERT_EQ(d.cameraViews.size(), 3u);
	EXPECT_EQ(d.cameraViews[0].name, "Front");
	EXPECT_EQ(d.cameraViews[1].name, "Front 2");
	EXPECT_EQ(d.cameraViews[2].name, "View") << "a blank name becomes 'View'";
	EXPECT_DOUBLE_EQ(d.cameraViews[0].camera.fov, 55);
	EXPECT_DOUBLE_EQ(d.cameraViews[0].camera.lensRadius, 0.2);
	EXPECT_DOUBLE_EQ(d.cameraViews[0].camera.focusDistance, 5.5);
}

TEST(SceneCameraViews, NumberingContinuesFromTheStemNotAppended) {
	Document d;
	saveCameraView(d, "Front", cameraAt(0, 0, 1));
	saveCameraView(d, "Front 2", cameraAt(0, 0, 2));
	saveCameraView(d, "Front 2", cameraAt(0, 0, 3));
	EXPECT_EQ(d.cameraViews[2].name, "Front 3") << "not 'Front 2 2'";
}

TEST(SceneCameraViews, ApplyingPutsTheSavedCameraOnTheScene) {
	Document d;
	d.camera = cameraAt(0, 2.5, 8);
	saveCameraView(d, "Close", cameraAt(1, 1, 2));
	EXPECT_TRUE(applyCameraView(d, 0));
	EXPECT_DOUBLE_EQ(d.camera.position.x, 1);
	EXPECT_DOUBLE_EQ(d.camera.position.z, 2);
	EXPECT_FALSE(applyCameraView(d, 1));
	EXPECT_FALSE(applyCameraView(d, -1));
}

TEST(SceneCameraViews, UpdateRenameAndRemove) {
	Document d;
	saveCameraView(d, "A", cameraAt(1, 1, 1));
	saveCameraView(d, "B", cameraAt(2, 2, 2));
	EXPECT_TRUE(updateCameraView(d, 0, cameraAt(9, 9, 9)));
	EXPECT_DOUBLE_EQ(d.cameraViews[0].camera.position.x, 9);
	EXPECT_EQ(d.cameraViews[0].name, "A") << "update keeps the name";
	EXPECT_TRUE(renameCameraView(d, 0, "  Top  "));
	EXPECT_EQ(d.cameraViews[0].name, "Top");
	EXPECT_TRUE(renameCameraView(d, 1, "Top"));
	EXPECT_EQ(d.cameraViews[1].name, "Top 2") << "another view has that name";
	EXPECT_TRUE(renameCameraView(d, 0, "Top")) << "a view does not clash with its own name";
	EXPECT_EQ(d.cameraViews[0].name, "Top");
	EXPECT_FALSE(renameCameraView(d, 0, "   ")) << "a blank name is refused";
	EXPECT_EQ(d.cameraViews[0].name, "Top");
	EXPECT_TRUE(removeCameraView(d, 0));
	ASSERT_EQ(d.cameraViews.size(), 1u);
	EXPECT_EQ(d.cameraViews[0].name, "Top 2");
	EXPECT_FALSE(removeCameraView(d, 5));
}

TEST(SceneCameraViews, ThereIsALimit) {
	Document d;
	for (std::size_t i = 0; i < kMaxCameraViews; ++i) EXPECT_GE(saveCameraView(d, "V", cameraAt(0, 0, 1)), 0);
	EXPECT_EQ(saveCameraView(d, "One too many", cameraAt(0, 0, 1)), -1);
	EXPECT_EQ(d.cameraViews.size(), kMaxCameraViews);
}

TEST(SceneCameraViews, LookingFromKeepsTheLensAndFollowsTheFocusOnlyWithoutDepthOfField) {
	Camera base = cameraAt(0, 2.5, 8);
	base.fov = 33;
	base.up = {0, 1, 0};
	const Camera sharp = cameraLookingFrom(base, {0, 1, 6}, {0, 1, 0});
	EXPECT_DOUBLE_EQ(sharp.position.z, 6);
	EXPECT_DOUBLE_EQ(sharp.fov, 33);
	EXPECT_NEAR(sharp.focusDistance, 6.0, 1e-9) << "no depth of field: the focus follows what is looked at";
	base.lensRadius = 0.1;
	base.focusDistance = 3.0;
	const Camera blurry = cameraLookingFrom(base, {0, 1, 6}, {0, 1, 0});
	EXPECT_DOUBLE_EQ(blurry.focusDistance, 3.0) << "depth of field: the focus was set on purpose";
	EXPECT_DOUBLE_EQ(blurry.lensRadius, 0.1);
}

TEST(SceneCameraViews, ViewsSurviveTheJsonCopyAndOldFilesHaveNone) {
	Document d;
	Camera c = cameraAt(4, 3, 2);
	c.fov = 55;
	c.lensRadius = 0.2;
	saveCameraView(d, "Front", c);
	saveCameraView(d, "Back", cameraAt(0, 1, -8));
	Document back;
	std::string error;
	ASSERT_TRUE(fromJson(toJson(d), back, error)) << error;
	ASSERT_EQ(back.cameraViews.size(), 2u);
	EXPECT_EQ(back.cameraViews[0].name, "Front");
	EXPECT_DOUBLE_EQ(back.cameraViews[0].camera.fov, 55);
	EXPECT_DOUBLE_EQ(back.cameraViews[0].camera.lensRadius, 0.2);
	EXPECT_DOUBLE_EQ(back.cameraViews[1].camera.position.z, -8);
	EXPECT_EQ(toJson(back), toJson(d));
	Document plain;
	EXPECT_EQ(toJson(plain).find("cameraViews"), std::string::npos) << "no views: nothing written, so older files and new ones agree";
	Document fromOld;
	ASSERT_TRUE(fromJson(toJson(plain), fromOld, error)) << error;
	EXPECT_TRUE(fromOld.cameraViews.empty());
}

TEST(SceneCameraViews, TheLogSaysWhenAViewIsAddedOrChanged) {
	Document a, b;
	saveCameraView(b, "Front", cameraAt(0, 1, 8));
	EXPECT_NE(describeChange(a, b).find("camera view"), std::string::npos);
	Document c = b;
	updateCameraView(c, 0, cameraAt(5, 1, 8));
	EXPECT_NE(describeChange(b, c).find("Front"), std::string::npos);
}
