// light_bounds_tests.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/light_bounds.h"

static LightBounds MakeUnitLight(float phi = 10.f, bool twoSided = false) {
    return LightBounds(0.f,0.f,0.f,1.f,1.f,1.f,0.f,0.f,1.f,phi,0.f,0.f,twoSided);
}

TEST(LightBounds, DefaultIsZeroPower) { LightBounds lb; EXPECT_EQ(lb.phi, 0.f); }

TEST(LightBounds, ConstructorStoresFields) {
    LightBounds lb = MakeUnitLight(5.f);
    EXPECT_FLOAT_EQ(lb.phi, 5.f);
    EXPECT_FLOAT_EQ(lb.cosTheta_o, 0.f);
    EXPECT_FLOAT_EQ(lb.cosTheta_e, 0.f);
    EXPECT_FALSE(lb.twoSided);
    EXPECT_NEAR(lb.wz, 1.f, 1e-5f);
}

TEST(LightBounds, ConstructorNormalisesW) {
    LightBounds lb(0,0,0,1,1,1,0.f,0.f,3.f,1.f,0.f,0.f,false);
    EXPECT_NEAR(lb.wx, 0.f, 1e-5f);
    EXPECT_NEAR(lb.wy, 0.f, 1e-5f);
    EXPECT_NEAR(lb.wz, 1.f, 1e-5f);
}

TEST(LightBoundsImportance, ZeroPowerIsZero) {
    LightBounds lb;
    EXPECT_EQ(Importance(lb,5.f,5.f,5.f,0.f,0.f,0.f), 0.f);
}

TEST(LightBoundsImportance, PositiveOnLitSide) {
    LightBounds lb = MakeUnitLight(10.f);
    EXPECT_GT(Importance(lb,0.5f,0.5f,5.f,0.f,0.f,0.f), 0.f);
}

TEST(LightBoundsImportance, DecreaseWithDistance) {
    LightBounds lb = MakeUnitLight(10.f);
    float impNear = Importance(lb,0.5f,0.5f,3.f,0.f,0.f,0.f);
    float impFar  = Importance(lb,0.5f,0.5f,10.f,0.f,0.f,0.f);
    EXPECT_GT(impNear, impFar);
}

TEST(LightBoundsImportance, NeverNegative) {
    LightBounds lb = MakeUnitLight(10.f);
    EXPECT_GE(Importance(lb,0.5f,0.5f,5.f,0,0,0),  0.f);
    EXPECT_GE(Importance(lb,0.5f,0.5f,-5.f,0,0,0), 0.f);
    EXPECT_GE(Importance(lb,5.f, 0.5f,5.f,0,0,0),  0.f);
}

TEST(LightBoundsImportance, TwoSidedPositiveBothSides) {
    LightBounds lb = MakeUnitLight(10.f, true);
    EXPECT_GT(Importance(lb,0.5f,0.5f, 5.f,0,0,0), 0.f);
    EXPECT_GT(Importance(lb,0.5f,0.5f,-5.f,0,0,0), 0.f);
}

TEST(LightBoundsImportance, NormalFacingAwayReducesImportance) {
    LightBounds lb = MakeUnitLight(10.f);
    float neg1 = -1.f;
    float impToward = Importance(lb,0.5f,0.5f,5.f,0.f,0.f,1.f);
    float impAway   = Importance(lb,0.5f,0.5f,5.f,0.f,0.f,neg1);
    EXPECT_GE(impToward, impAway);
    EXPECT_GE(impToward, 0.f);
    EXPECT_GE(impAway,   0.f);
}

TEST(LightBoundsUnion, ZeroPowerReturnsOther) {
    LightBounds zero;
    LightBounds lb = MakeUnitLight(7.f);
    EXPECT_FLOAT_EQ(Union(zero,lb).phi, 7.f);
    EXPECT_FLOAT_EQ(Union(lb,zero).phi, 7.f);
}

TEST(LightBoundsUnion, CombinesPhi) {
    LightBounds a = MakeUnitLight(3.f);
    LightBounds b = MakeUnitLight(4.f);
    EXPECT_NEAR(Union(a,b).phi, 7.f, 1e-5f);
}

TEST(LightBoundsUnion, ExpandsAABB) {
    LightBounds a(0,0,0,1,1,1,0,0,1,5.f,0.f,0.f,false);
    LightBounds b(2,2,2,3,3,3,0,0,1,5.f,0.f,0.f,false);
    LightBounds u = Union(a,b);
    EXPECT_FLOAT_EQ(u.bMinX,0.f); EXPECT_FLOAT_EQ(u.bMinY,0.f); EXPECT_FLOAT_EQ(u.bMinZ,0.f);
    EXPECT_FLOAT_EQ(u.bMaxX,3.f); EXPECT_FLOAT_EQ(u.bMaxY,3.f); EXPECT_FLOAT_EQ(u.bMaxZ,3.f);
}

TEST(LightBoundsUnion, TwoSidedIsORed) {
    LightBounds a = MakeUnitLight(1.f, false);
    LightBounds b = MakeUnitLight(1.f, true);
    EXPECT_TRUE(Union(a,b).twoSided);
    EXPECT_FALSE(Union(a,a).twoSided);
}

TEST(LightBoundsUnion, CosThetaEIsMin) {
    LightBounds a(0,0,0,1,1,1,0,0,1,1.f,0.f,0.5f,false);
    LightBounds b(0,0,0,1,1,1,0,0,1,1.f,0.f,0.2f,false);
    EXPECT_NEAR(Union(a,b).cosTheta_e, 0.2f, 1e-5f);
}

TEST(LightBoundsUnion, ImportanceDoesNotUnderestimate) {
    LightBounds a(0,0,0,1,1,1,0,0,1,5.f,0.f,0.f,false);
    LightBounds b(2,0,0,3,1,1,0,0,1,5.f,0.f,0.f,false);
    LightBounds u = Union(a,b);
    float impA = Importance(a,1.5f,0.5f,10.f,0,0,0);
    float impB = Importance(b,1.5f,0.5f,10.f,0,0,0);
    float impU = Importance(u,1.5f,0.5f,10.f,0,0,0);
    EXPECT_GE(impU, impA);
    EXPECT_GE(impU, impB);
}
