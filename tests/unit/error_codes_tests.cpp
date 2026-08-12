#include <gtest/gtest.h>

#include "../../src/TheRestOfYourLife/error_codes.h"

#include <set>
#include <string>
#include <vector>

// ============================================================================
// Error code contract
// ============================================================================
// The Qt GUI (qt_gui/error_handler.h) presents these codes to the user. It
// used to hold an independent copy of the whole table - its own integer
// literals and its own text - with nothing tying the two together, so a code
// added here would surface in the GUI as "Unknown Error" with no build failure
// and no test to catch it.
//
// The GUI now includes this header, keys its overrides off the enum constants,
// and falls back to get_error_message()/get_troubleshooting_hint() via the
// has_* predicates. These tests pin down the half of that contract the GUI
// depends on. The GUI itself is built by a separate qmake/MinGW project that
// this MSVC test binary cannot link, so its side is enforced at compile time
// instead: naming the constants means a rename or removal here breaks the GUI
// build rather than silently degrading it.
// ============================================================================

namespace {

// Every code the enum defines. Kept explicit rather than derived, so adding a
// code to the enum without considering its user-facing text shows up here.
const std::vector<int>& allErrorCodes() {
	static const std::vector<int> codes = {
		SUCCESS,
		ERR_UNKNOWN, ERR_INVALID_ARGUMENTS, ERR_FILE_NOT_FOUND,
		ERR_FILE_READ_FAILED, ERR_FILE_WRITE_FAILED, ERR_FILE_COPY_FAILED,
		ERR_DIRECTORY_CREATE_FAILED, ERR_INVALID_DIMENSIONS,
		ERR_INVALID_SAMPLE_COUNT, ERR_INVALID_MAX_DEPTH, ERR_INVALID_SCENE_ID,
		ERR_INVALID_CAMERA_POSITION, ERR_OUTPUT_PATH_INVALID,
		ERR_VIDEO_ASSEMBLY_FAILED,

		ERR_CPU_SCENE_BUILD_FAILED, ERR_CPU_SCENE_EMPTY,
		ERR_CPU_CAMERA_INIT_FAILED, ERR_CPU_RENDER_FAILED, ERR_CPU_THREAD_FAILED,
		ERR_CPU_MEMORY_ALLOCATION, ERR_CPU_BVH_BUILD_FAILED,
		ERR_CPU_TEXTURE_LOAD_FAILED, ERR_CPU_LIGHTS_EMPTY,
		ERR_CPU_MATERIAL_INVALID,

		ERR_GPU_NO_DEVICE, ERR_GPU_DEVICE_INIT_FAILED, ERR_GPU_MEMORY_ALLOCATION,
		ERR_GPU_MEMORY_COPY_FAILED, ERR_GPU_KERNEL_LAUNCH_FAILED,
		ERR_GPU_KERNEL_EXECUTION_FAILED, ERR_GPU_SCENE_SERIALIZATION_FAILED,
		ERR_GPU_DEVICE_SYNCHRONIZATION_FAILED, ERR_GPU_OUT_OF_MEMORY,
		ERR_GPU_INVALID_CONFIGURATION, ERR_GPU_TEXTURE_BINDING_FAILED,
		ERR_GPU_UNSUPPORTED_SCENE, ERR_GPU_SCENE_BUILD_FAILED,
		ERR_GPU_RENDER_FAILED, ERR_GPU_EXCEPTION, ERR_GPU_UNKNOWN_ERROR,
	};
	return codes;
}

} // namespace

// Every defined code must have real text, or the GUI shows a placeholder that
// tells the user nothing.
TEST(ErrorCodesTest, EveryDefinedCodeHasARealMessage) {
	for (int code : allErrorCodes()) {
		EXPECT_TRUE(has_error_message(code))
			<< "error code " << code << " has no entry in error_message_table(); "
			<< "the GUI would fall back to a placeholder for it";
		EXPECT_FALSE(get_error_message(code).empty()) << "code " << code;
	}
}

// The placeholder path must stay distinguishable from a real message -
// qt_gui/error_handler.h relies on has_error_message() rather than testing the
// returned string, precisely because the unknown path returns non-empty text.
TEST(ErrorCodesTest, UnknownCodeIsReportedAsUnknownNotEmpty) {
	constexpr int bogus = 31337;
	EXPECT_FALSE(has_error_message(bogus));
	const std::string msg = get_error_message(bogus);
	EXPECT_FALSE(msg.empty())
		<< "an emptiness check would be an unreliable way to detect unknown codes";
	EXPECT_NE(msg.find("31337"), std::string::npos)
		<< "the placeholder should name the code it could not describe";
}

TEST(ErrorCodesTest, UnknownHintIsReportedAsUnknownNotEmpty) {
	constexpr int bogus = 31337;
	EXPECT_FALSE(has_troubleshooting_hint(bogus));
	EXPECT_FALSE(get_troubleshooting_hint(bogus).empty());
}

// has_* must agree with get_* rather than being a second, drifting opinion.
TEST(ErrorCodesTest, PredicatesAgreeWithTables) {
	for (int code : allErrorCodes()) {
		const auto& msgs = error_message_table();
		EXPECT_EQ(has_error_message(code), msgs.find(code) != msgs.end())
			<< "code " << code;

		const auto& hints = troubleshooting_hint_table();
		EXPECT_EQ(has_troubleshooting_hint(code), hints.find(code) != hints.end())
			<< "code " << code;
	}
}

// Codes are grouped by range (1-99 general, 100-199 CPU, 200-299 GPU) and both
// the GUI's category colouring and get_error_category() depend on that split,
// so a code landing in the wrong band would be miscategorised in the UI.
TEST(ErrorCodesTest, CodesStayInTheirDocumentedRanges) {
	EXPECT_EQ(SUCCESS, 0);

	const std::vector<int> general = {
		ERR_UNKNOWN, ERR_INVALID_ARGUMENTS, ERR_FILE_NOT_FOUND,
		ERR_VIDEO_ASSEMBLY_FAILED,
	};
	for (int code : general) {
		EXPECT_GE(code, 1);
		EXPECT_LE(code, 99) << "general code " << code << " escaped the 1-99 band";
	}

	const std::vector<int> cpu = {
		ERR_CPU_SCENE_BUILD_FAILED, ERR_CPU_MATERIAL_INVALID,
	};
	for (int code : cpu) {
		EXPECT_GE(code, 100);
		EXPECT_LE(code, 199) << "CPU code " << code << " escaped the 100-199 band";
	}

	const std::vector<int> gpu = {
		ERR_GPU_NO_DEVICE, ERR_GPU_UNKNOWN_ERROR, ERR_GPU_UNSUPPORTED_SCENE,
	};
	for (int code : gpu) {
		EXPECT_GE(code, 200);
		EXPECT_LE(code, 299) << "GPU code " << code << " escaped the 200-299 band";
	}
}

// Distinct codes must stay distinct: a collision here previously shipped as a
// real bug (a GPU code reusing a CPU code's number), so it is worth pinning.
TEST(ErrorCodesTest, AllCodesAreUnique) {
	std::set<int> seen;
	for (int code : allErrorCodes()) {
		EXPECT_TRUE(seen.insert(code).second)
			<< "duplicate error code value " << code;
	}
}
