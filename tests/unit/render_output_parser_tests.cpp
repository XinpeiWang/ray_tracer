#include <gtest/gtest.h>

#include "../../qt_gui/render_output_parser.h"

#include <string>
#include <vector>

using namespace render_output;

// ============================================================================
// The GUI <-> renderer stdout contract
// ============================================================================
// The Qt GUI learns everything about a running render by scraping
// ray_tracer.exe's stdout. That makes the wording of those std::cout calls a
// real protocol, and it has broken twice before: once when the log classifier
// keyed off decorative emoji, once when a progress regex went stale. Neither
// failure had a test, and neither was loud - the progress bar simply stopped
// moving.
//
// Every literal below was copied from an ACTUAL run of ray_tracer.exe
// (--cpu, 80x80, 4spp, scene 0), not written from memory, so these tests
// describe what the renderer really emits.
// ============================================================================

namespace {

// Verbatim lines from a real CPU render.
const char *kRealScanline      = "Scanlines remaining: 70 ";  // note trailing space
const char *kRealLauncherCpu   = "Launching renderer (CPU mode)...";
const char *kRealCpuInterface  = "[cpu_interface] Building scene 0 (Cornell Box)...";
const char *kRealTech          = "[TECH] Integrator     : Iterative path tracer  (pbrt-v4 PathIntegrator::Li style)";
const char *kRealRenderTime    = "RENDER TIME: 216 ms";
const char *kRealPngSaved      = "\xE2\x9C\x93 PNG saved: \"C:/tmp/fixture.png\"";  // real line starts with U+2713
const char *kRealSettings      = "Using command-line settings: width=80 height=80 spp=4 max_depth=4 scene_id=0 camera=(278,278,-800)";
const char *kRealWritingOutput = "Writing output to: C:/tmp/fixture.ppm";
const char *kRealBanner        = "========================================";

} // namespace

// ----------------------------------------------------------------------------
// Progress
// ----------------------------------------------------------------------------

TEST(RenderOutputParserTest, ParsesProgressFromRealScanlineLine) {
	// 80-tall image, 70 scanlines left => 10/80 done => 12%.
	EXPECT_EQ(parseScanlineProgress(kRealScanline, 80), 12);
}

TEST(RenderOutputParserTest, ScanlineProgressIsMonotonicAcrossARealSweep) {
	int previous = -1;
	for (int remaining = 80; remaining >= 0; remaining -= 10) {
		const std::string line = "Scanlines remaining: " + std::to_string(remaining) + " ";
		const int pct = parseScanlineProgress(line, 80);
		ASSERT_NE(pct, kNoProgress) << "failed to parse: " << line;
		EXPECT_GE(pct, previous) << "progress went backwards at " << line;
		previous = pct;
	}
	EXPECT_EQ(previous, 100);
}

TEST(RenderOutputParserTest, NonProgressLinesReportNoProgress) {
	EXPECT_EQ(parseScanlineProgress(kRealTech, 80), kNoProgress);
	EXPECT_EQ(parseScanlineProgress(kRealRenderTime, 80), kNoProgress);
	EXPECT_EQ(parseScanlineProgress("", 80), kNoProgress);
}

TEST(RenderOutputParserTest, ScanlineProgressSurvivesZeroHeight) {
	// Guards a divide-by-zero rather than asserting a meaningful percentage.
	EXPECT_EQ(parseScanlineProgress(kRealScanline, 0), 0);
	EXPECT_EQ(parseScanlineProgress(kRealScanline, -5), 0);
}

TEST(RenderOutputParserTest, ParsesVideoFrameProgressAndCapsAt95) {
	EXPECT_EQ(parseVideoFrameProgress("[5/60] Rendering frame_0005.ppm"), (5 * 95) / 60);
	// The last frame must not reach 100 - ffmpeg assembly still has to happen.
	EXPECT_EQ(parseVideoFrameProgress("[60/60] Rendering frame_0060.ppm"), 95);
	EXPECT_EQ(parseVideoFrameProgress(kRealScanline), kNoProgress);
}

TEST(RenderOutputParserTest, VideoFrameProgressSurvivesZeroTotal) {
	EXPECT_EQ(parseVideoFrameProgress("[0/0] Rendering frame_0000.ppm"), 0);
}

TEST(RenderOutputParserTest, DetectsVideoAssemblyHandoff) {
	EXPECT_TRUE(isVideoAssemblyStart("=== ASSEMBLING VIDEO WITH FFMPEG ==="));
	EXPECT_FALSE(isVideoAssemblyStart(kRealRenderTime));
}

// ----------------------------------------------------------------------------
// Line splitting
// ----------------------------------------------------------------------------

TEST(RenderOutputParserTest, TreatsCarriageReturnAsALineTerminator) {
	// This is why progress works at all: the renderer emits its counter with a
	// bare '\r'. Waiting for '\n' would stall the bar.
	std::string pending;
	const auto lines = splitOutputLines(
		"Scanlines remaining: 80 \rScanlines remaining: 70 \r", pending);

	ASSERT_EQ(lines.size(), 2u);
	EXPECT_EQ(parseScanlineProgress(lines[0], 80), 0);
	EXPECT_EQ(parseScanlineProgress(lines[1], 80), 12);
	EXPECT_TRUE(pending.empty());
}

TEST(RenderOutputParserTest, HoldsBackAPartialLineUntilTheRestArrives) {
	// A chunk boundary must not split one log line into two - that bug would
	// show up as truncated log entries and missed progress updates.
	std::string pending;
	auto lines = splitOutputLines("Scanlines rema", pending);
	EXPECT_TRUE(lines.empty());
	EXPECT_FALSE(pending.empty());

	lines = splitOutputLines("ining: 40 \r", pending);
	ASSERT_EQ(lines.size(), 1u);
	EXPECT_EQ(parseScanlineProgress(lines[0], 80), 50);
	EXPECT_TRUE(pending.empty());
}

TEST(RenderOutputParserTest, HandlesCrLfWithoutEmittingPhantomLines) {
	std::string pending;
	const auto lines = splitOutputLines("alpha\r\nbeta\r\n", pending);
	// "\r\n" yields one real line plus one empty segment; the GUI drops empty
	// lines, so what matters is that the content lines survive intact.
	std::vector<std::string> nonEmpty;
	for (const auto &l : lines)
		if (!l.empty()) nonEmpty.push_back(l);

	ASSERT_EQ(nonEmpty.size(), 2u);
	EXPECT_EQ(nonEmpty[0], "alpha");
	EXPECT_EQ(nonEmpty[1], "beta");
}

// ----------------------------------------------------------------------------
// Log classification
// ----------------------------------------------------------------------------

TEST(RenderOutputParserTest, ClassifiesRealRendererLines) {
	EXPECT_STREQ(classifyLogLine(kRealTech).label, "TECH");
	EXPECT_STREQ(classifyLogLine(kRealRenderTime).label, "PERF");
	EXPECT_STREQ(classifyLogLine(kRealPngSaved).label, " OK ");
	EXPECT_STREQ(classifyLogLine(kRealWritingOutput).label, "DBG ");
	EXPECT_EQ(classifyLogLine(kRealBanner).style, LineStyle::Banner);
}

// Backend attribution deliberately outranks event type: a line carrying
// "[cpu_interface]" or "[OptiX]" is labelled with its backend even when it is
// also a scene-setup line. Writing this test from real output initially
// assumed the opposite - it is recorded here so the precedence is a decision
// rather than an accident, and so flipping it would fail loudly.
TEST(RenderOutputParserTest, BackendPrefixOutranksEventType) {
	EXPECT_STREQ(classifyLogLine(kRealCpuInterface).label, "CPU ")
		<< "a [cpu_interface] line stays attributed to the CPU backend";
	EXPECT_STREQ(classifyLogLine("[OptiX] Built acceleration structure").label, "GPU ");

	// Scene lines with no backend prefix still get the scene label.
	EXPECT_STREQ(classifyLogLine("Building scene 0 (Cornell Box)...").label, "SCN ");
}

// Regression: the launcher echoes "... spp=4 ..." when reporting its settings,
// and the performance rule matched a bare " spp", so a plain settings line was
// tagged as a performance measurement. Found by feeding this test real output.
TEST(RenderOutputParserTest, SettingsEchoIsNotMistakenForAPerfMeasurement) {
	EXPECT_STREQ(classifyLogLine(kRealSettings).label, "DBG ")
		<< "a settings echo is debug information, not a timing measurement";
	// The measurement form the rule actually targets must still match.
	EXPECT_STREQ(classifyLogLine("Rendered 800x800 @ 100 spp").label, "PERF");
}

TEST(RenderOutputParserTest, LauncherModeLineIsAttributedToItsBackend) {
	EXPECT_STREQ(classifyLogLine(kRealLauncherCpu).label, "CPU ");
	EXPECT_STREQ(classifyLogLine("Launching renderer (GPU mode)...").label, "GPU ");
}

TEST(RenderOutputParserTest, ErrorsKeepTheirSubsystemInsteadOfCollapsingToERR) {
	// Regression guard: every GPU/CPU line containing "error" used to be
	// flattened into a generic ERR tag, losing which backend failed.
	EXPECT_STREQ(classifyLogLine("[OptiX] CUDA error: invalid program counter").label, "GPU ");
	EXPECT_STREQ(classifyLogLine("[cpu_interface] render error occurred").label, "CPU ");
	EXPECT_STREQ(classifyLogLine("Something generic FAILED").label, "ERR ");
}

TEST(RenderOutputParserTest, SuccessIsRecognisedWithoutRelyingOnGlyphs) {
	// The decorative check marks are a hint, not the contract - these must all
	// classify as success on their ASCII words alone.
	EXPECT_STREQ(classifyLogLine("Result: SUCCESS  |  Output: x.png").label, " OK ");
	EXPECT_STREQ(classifyLogLine("PNG saved: x.png").label, " OK ");
	EXPECT_STREQ(classifyLogLine("Render complete!").label, " OK ");
	EXPECT_STREQ(classifyLogLine("[OK] done").label, " OK ");
}

TEST(RenderOutputParserTest, RenderStartBannerDoesNotDependOnItsGlyph) {
	// Previously matched startsWith("<play glyph> RENDER START"); dropping the
	// glyph silently demoted the banner to a plain line.
	const auto withGlyph = classifyLogLine("\xE2\x96\xB6 RENDER START  2026-08-11 18:00:00");
	const auto without   = classifyLogLine("RENDER START  2026-08-11 18:00:00");
	EXPECT_EQ(withGlyph.style, LineStyle::BoldLabeled);
	EXPECT_EQ(without.style, LineStyle::BoldLabeled);
}

TEST(RenderOutputParserTest, ErrorDetailsBannerOutranksTheSeparatorRule) {
	// "=== ERROR DETAILS ===" starts with "===" and would otherwise be styled
	// as a plain grey separator, burying the one banner that flags a failure.
	const auto cat = classifyLogLine("=== ERROR DETAILS ===");
	EXPECT_EQ(cat.style, LineStyle::Banner);
	EXPECT_STREQ(cat.colour, "#FF6B6B") << "the failure banner must not be grey";
}

TEST(RenderOutputParserTest, UnrecognisedLinesFallBackToPlainInfo) {
	const auto cat = classifyLogLine("some line nobody wrote a rule for");
	EXPECT_STREQ(cat.label, "INFO");
	EXPECT_EQ(cat.style, LineStyle::Normal);
}
