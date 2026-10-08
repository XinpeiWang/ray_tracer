// photo_helper_report_tests.cpp - the Qt-free text handling behind the Scene Builder's optional photo helper (src/shared/photo_helper_report.h):
// the helper's progress lines, which facts of the Diagnostics "Photo helper" section the setup script would fix, and the splitter that turns a
// subprocess's bytes into lines (the installer's log and failure message depend on it handling Windows' \r\n and pip's lone \r).
#include <gtest/gtest.h>

#include "../../src/shared/photo_helper_report.h"

using namespace photo_report;

TEST(PhotoHelperProgressTest, ReadsPercentAndMessage) {
	int pct = -1;
	std::string msg;
	ASSERT_TRUE(parseProgressLine("PROGRESS 35 Guessing the 3D shape", pct, msg));
	EXPECT_EQ(pct, 35);
	EXPECT_EQ(msg, "Guessing the 3D shape");
	ASSERT_TRUE(parseProgressLine("PROGRESS 100", pct, msg));
	EXPECT_EQ(pct, 100);
	EXPECT_EQ(msg, "");
	ASSERT_TRUE(parseProgressLine("PROGRESS 250 too much", pct, msg));
	EXPECT_EQ(pct, 100) << "clamped";
}

TEST(PhotoHelperProgressTest, OtherLinesAreNotProgress) {
	int pct = 0;
	std::string msg;
	EXPECT_FALSE(parseProgressLine("ERROR the photo was not found", pct, msg));
	EXPECT_FALSE(parseProgressLine("PROGRESS abc nope", pct, msg));
	EXPECT_FALSE(parseProgressLine("PROGRESS -5 nope", pct, msg));
	EXPECT_FALSE(parseProgressLine("PROGRESS ", pct, msg));
	EXPECT_FALSE(parseProgressLine("  PROGRESS 5 indented", pct, msg));
	EXPECT_FALSE(parseProgressLine("", pct, msg));
}

namespace {
const char* kReport =
    "=== Ray Tracer System Diagnostics ===\n"
    "GPU: NVIDIA GeForce RTX 5080\n"
    "=== Network ===\n"
    "Internet: not available (could not reach host)\n"
    "=== Photo helper (Scene Builder, Add > Object from a photo; optional) ===\n"
    "Helper Script: missing (not found next to the program)\n"
    "Python Environment: present (C:\\env\\python.exe)\n"
    "Python: 3.12.10 (C:\\env\\python.exe)\r\n"
    "PyTorch: missing (not installed; run scripts/setup_photo_to_mesh.ps1)\r\n"
    "Transformers: 5.0.0 not usable (TripoSR's model needs a version below 5)\n"
    "NumPy: 2.5.2 present\n"
    "Graphics Card for PyTorch: not available (a photo runs on the processor)\n"
    "TripoSR Weights: missing (about 1.7 GB, downloaded the first time a photo is converted)\n"
    "Saved Photo Objects: none yet (C:\\data\\photo_meshes; not available to anything)\n";
}  // namespace

TEST(PhotoHelperMissingFactsTest, ListsWhatSetupWouldFixAndNothingElse) {
	const std::vector<std::string> facts = missingFacts(kReport);
	ASSERT_EQ(facts.size(), 3u);
	EXPECT_EQ(facts[0].rfind("PyTorch: missing", 0), 0u);
	EXPECT_EQ(facts[1].rfind("Transformers:", 0), 0u);
	EXPECT_EQ(facts[2].rfind("TripoSR Weights: missing", 0), 0u);
	for (const std::string& f : facts) EXPECT_EQ(f.find('\r'), std::string::npos) << "CRs are trimmed";
}

TEST(PhotoHelperMissingFactsTest, OtherSectionsAreNotRead) {
	// "not available" in the Network section, and a missing GPU line elsewhere, are not the photo helper's.
	const std::string only = "=== Network ===\nInternet: not available (no route)\n=== Ray Tracer System Diagnostics ===\nOptiX: not available\n";
	EXPECT_TRUE(missingFacts(only).empty());
	EXPECT_TRUE(missingFacts("").empty());
}

TEST(PhotoHelperMissingFactsTest, ANotSetUpEnvironmentIsListed) {
	const std::string r =
	    "=== Photo helper (x) ===\nHelper Script: present (a)\nPython Environment: not available (not set up; run scripts\\setup.ps1)\nExpected At: C:\\nowhere\\python.exe\n";
	const std::vector<std::string> facts = missingFacts(r);
	ASSERT_EQ(facts.size(), 1u);
	EXPECT_EQ(facts[0].rfind("Python Environment: not available", 0), 0u);
}

TEST(PhotoHelperMissingFactsTest, AHealthyInstallListsNothing) {
	const std::string r =
	    "=== Photo helper (x) ===\nHelper Script: present (a)\nPython Environment: present (b)\nPyTorch: 2.11.0 present\nGraphics Card for PyTorch: available (RTX)\n"
	    "TripoSR Weights: present (1.7 GB, cached)\nSaved Photo Objects: 2 folders, 30 MB (c)\n";
	EXPECT_TRUE(missingFacts(r).empty());
}

namespace {
std::vector<std::string> lines(const std::vector<LineSplitter::Piece>& ps, bool endsLine) {
	std::vector<std::string> out;
	for (const auto& p : ps)
		if (p.endsLine == endsLine) out.push_back(p.text);
	return out;
}
}  // namespace

TEST(LineSplitterTest, LfLinesAndCrLfLinesBothEndALine) {
	LineSplitter s;
	auto a = s.feed("first\nsecond\r\nthird\n");
	ASSERT_EQ(a.size(), 3u);
	EXPECT_EQ(lines(a, true), (std::vector<std::string>{"first", "second", "third"}));
	EXPECT_TRUE(lines(a, false).empty()) << "a \\r\\n is not a status redraw";
}

TEST(LineSplitterTest, ACrLfSplitAcrossTwoChunksStillMakesOneLine) {
	LineSplitter s;
	auto a = s.feed("ERROR: no network\r");
	EXPECT_TRUE(a.empty()) << "the \\r may be the first half of \\r\\n, so nothing yet";
	auto b = s.feed("\nnext\n");
	ASSERT_EQ(b.size(), 2u);
	EXPECT_EQ(b[0].text, "ERROR: no network");
	EXPECT_TRUE(b[0].endsLine);
	EXPECT_EQ(b[1].text, "next");
}

// Windows PowerShell turns a native command's "\r\n" into "\r\r\n" when its output goes to a pipe: seen with the real installer's python lines.
TEST(LineSplitterTest, ACrCrLfEndsALineToo) {
	LineSplitter s;
	auto a = s.feed("first line\r\r\nERROR: no network\r\r\n");
	EXPECT_EQ(lines(a, true), (std::vector<std::string>{"first line", "ERROR: no network"}));
	EXPECT_TRUE(lines(a, false).empty());
	// ... even when the run of CRs is cut in two by the end of a chunk
	LineSplitter t;
	EXPECT_TRUE(t.feed("cut\r").empty());
	EXPECT_TRUE(t.feed("\r").empty());
	auto b = t.feed("\nnext\n");
	ASSERT_EQ(b.size(), 2u);
	EXPECT_EQ(b[0].text, "cut");
	EXPECT_TRUE(b[0].endsLine);
	EXPECT_EQ(b[1].text, "next");
}

TEST(LineSplitterTest, ALoneCrIsAStatusRedrawNotALine) {
	LineSplitter s;
	auto a = s.feed("Downloading 10%\rDownloading 60%\rDownloading done\n");
	EXPECT_EQ(lines(a, false), (std::vector<std::string>{"Downloading 10%", "Downloading 60%"}));
	EXPECT_EQ(lines(a, true), (std::vector<std::string>{"Downloading done"}));
}

TEST(LineSplitterTest, APartialLineWaitsForItsEndAndFlushReturnsTheRest) {
	LineSplitter s;
	EXPECT_TRUE(s.feed("half a li").empty());
	auto a = s.feed("ne\nand the tail");
	ASSERT_EQ(a.size(), 1u);
	EXPECT_EQ(a[0].text, "half a line");
	EXPECT_EQ(s.flush(), "and the tail");
	EXPECT_EQ(s.flush(), "") << "flush empties it";
}

TEST(LineSplitterTest, BlankLinesAndWhitespaceAreDropped) {
	LineSplitter s;
	auto a = s.feed("\n\r\n   \n  padded  \n");
	ASSERT_EQ(a.size(), 1u);
	EXPECT_EQ(a[0].text, "padded");
}
