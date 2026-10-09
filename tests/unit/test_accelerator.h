// test_accelerator.h - tests that build a pbrt-backed scene through the scene registry need the accelerator choice made first.
//
// launcher/main.cpp calls accelerator_override::set() before any render (scene_registry.h asserts, in a Debug build, that it ran before the first pbrt-backed
// scene is built, so a scene cannot cache a default choice by accident). A test binary has no main() of that kind, so including this header makes the choice once, with
// no override ("" and ""), before the first test - in Debug builds too, where that assert is compiled in. Header-only; the inline variable registers it once however
// many test files include it.
#pragma once

#include <gtest/gtest.h>

#include "../../src/shared/accelerator_override.h"

namespace test_accelerator {

class Environment : public ::testing::Environment {
public:
	void SetUp() override {
		if (!accelerator_override::was_set()) accelerator_override::set({"", ""});
	}
};

inline const bool kRegistered = (::testing::AddGlobalTestEnvironment(new Environment), true);

}  // namespace test_accelerator
