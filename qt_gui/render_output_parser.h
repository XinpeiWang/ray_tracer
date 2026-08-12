#ifndef RENDER_OUTPUT_PARSER_H
#define RENDER_OUTPUT_PARSER_H

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <vector>

// ============================================================================
// Parsing ray_tracer.exe's console output
// ============================================================================
// Everything the GUI learns about a running render comes from scraping the
// subprocess's stdout: how far along it is, and what severity each log line
// should be shown at. That made the exact wording of ray_tracer.exe's
// std::cout calls an undocumented protocol - reword one and the progress bar
// silently stops moving, with nothing failing to say so. It has already bitten
// twice: once when the log classifier keyed off decorative emoji, and once
// when a progress regex went stale.
//
// This header exists so that protocol can be TESTED. It is deliberately
// Qt-free and uses std::string/std::regex, because the Qt install here is
// MinGW-only while the gtest binary is built with MSVC - a Qt-based parser
// simply could not be reached from the test suite. The GUI converts each line
// once (QString::toStdString) before calling in; lines are short and arrive at
// most a few hundred times a second, so the conversion is not worth avoiding.
//
// See tests/unit/render_output_parser_tests.cpp.
// ============================================================================
namespace render_output {

// How a classified line should be presented.
enum class LineStyle {
	Normal,       // "HH:mm:ss [LABEL] message" - most lines
	BoldLabeled,  // as Normal but bold (render-start banners)
	Banner,       // bold, no timestamp or label (separators, error banner)
};

struct LogCategory {
	const char *colour = "#D0D0D0";
	const char *label = "INFO";   // ignored when style == Banner
	LineStyle style = LineStyle::Normal;
};

// Sentinel for "this line carried no progress information".
constexpr int kNoProgress = -1;

namespace detail {

inline bool contains(const std::string &haystack, const char *needle) {
	return haystack.find(needle) != std::string::npos;
}

inline bool containsNoCase(const std::string &haystack, const std::string &needle) {
	if (needle.empty()) return true;
	auto it = std::search(haystack.begin(), haystack.end(),
						  needle.begin(), needle.end(),
						  [](unsigned char a, unsigned char b) {
							  return std::tolower(a) == std::tolower(b);
						  });
	return it != haystack.end();
}

inline bool startsWith(const std::string &s, const char *prefix) {
	return s.rfind(prefix, 0) == 0;
}

} // namespace detail

// ----------------------------------------------------------------------------
// Chunking stdout into lines
// ----------------------------------------------------------------------------

// Splits a raw stdout chunk into complete lines, carrying any trailing partial
// line forward in `pending` until the rest of it arrives.
//
// '\r' counts as a terminator, not just '\n'. That is load-bearing: the
// renderer writes its progress counter as "Scanlines remaining: N\r" with no
// newline, so an implementation that waited for '\n' would buffer every
// progress update until the next ordinary line and the bar would move in
// lurches (or, for a render whose last output is progress, not at all).
inline std::vector<std::string> splitOutputLines(const std::string &chunk,
												 std::string &pending) {
	pending += chunk;

	std::vector<std::string> lines;
	std::string current;
	for (char c : pending) {
		if (c == '\r' || c == '\n') {
			lines.push_back(current);
			current.clear();
		} else {
			current.push_back(c);
		}
	}
	// Whatever followed the last terminator is incomplete - hold it back.
	pending = current;
	return lines;
}

// ----------------------------------------------------------------------------
// Progress
// ----------------------------------------------------------------------------

// "Scanlines remaining: N" -> percent complete, or kNoProgress.
// imageHeight is the total scanline count; a non-positive height yields 0
// rather than dividing by zero.
inline int parseScanlineProgress(const std::string &line, int imageHeight) {
	static const std::regex re(R"(Scanlines remaining:\s*(\d+))");
	std::smatch m;
	if (!std::regex_search(line, m, re)) return kNoProgress;

	const int remaining = std::stoi(m[1].str());
	if (imageHeight <= 0) return 0;
	const int completed = imageHeight - remaining;
	int percent = (completed * 100) / imageHeight;
	return std::max(0, std::min(percent, 100));
}

// "[5/60] Rendering frame_..." -> percent complete, or kNoProgress.
// Capped at 95: the remaining headroom belongs to ffmpeg assembly, which
// reports no usable per-frame progress of its own.
inline int parseVideoFrameProgress(const std::string &line) {
	static const std::regex re(R"(\[(\d+)/(\d+)\] Rendering frame_)");
	std::smatch m;
	if (!std::regex_search(line, m, re)) return kNoProgress;

	const int current = std::stoi(m[1].str());
	const int total = std::stoi(m[2].str());
	if (total <= 0) return 0;
	int percent = (current * 95) / total;
	return std::max(0, std::min(percent, 95));
}

// The point where frame rendering ends and ffmpeg assembly begins.
inline bool isVideoAssemblyStart(const std::string &line) {
	return detail::contains(line, "ASSEMBLING VIDEO WITH FFMPEG");
}

// ----------------------------------------------------------------------------
// Log line classification
// ----------------------------------------------------------------------------
// Ordered, first match wins. Matching is on ASCII words rather than the
// decorative glyphs ray_tracer.exe prefixes lines with - those are decoration,
// the words are the contract.

inline LogCategory classifyLogLine(const std::string &line) {
	using namespace detail;

	// Would otherwise match the "===" separator rule below and render as a
	// plain grey line, burying the one banner meant to flag a failure.
	if (contains(line, "ERROR DETAILS"))
		return LogCategory{"#FF6B6B", "", LineStyle::Banner};

	if (startsWith(line, "\xE2\x95\x90") ||   // U+2550 box drawing double horizontal
		startsWith(line, "\xE2\x94\x80") ||   // U+2500 box drawing light horizontal
		startsWith(line, "===") || startsWith(line, "---"))
		return LogCategory{"#555555", "", LineStyle::Banner};

	if (contains(line, "RENDER START") || startsWith(line, "Starting render"))
		return LogCategory{"#74C0FC", "INFO", LineStyle::BoldLabeled};

	if (containsNoCase(line, "error") || contains(line, "FAILED") ||
		containsNoCase(line, "fatal") || contains(line, "ERR_") ||
		startsWith(line, "Result: FAILED")) {
		// Keep the red "this is an error" signal, but preserve which subsystem
		// it came from instead of collapsing every GPU/CPU line that happens
		// to contain "error"/"failed" into a generic ERR tag.
		if (contains(line, "[OptiX]") || contains(line, "[optix]") ||
			contains(line, "OptiX") || containsNoCase(line, "GPU mode"))
			return LogCategory{"#FF6B6B", "GPU "};
		if (contains(line, "[cpu_interface]") || containsNoCase(line, "CPU mode"))
			return LogCategory{"#FF6B6B", "CPU "};
		return LogCategory{"#FF6B6B", "ERR "};
	}

	if (containsNoCase(line, "warning") || contains(line, "WARN") ||
		containsNoCase(line, "Requires external files"))
		return LogCategory{"#FFD700", "WARN"};

	// The check-mark prefixes are an extra hint only; every case they cover is
	// also matched by an ASCII phrase here.
	if (startsWith(line, "Result: SUCCESS") ||
		startsWith(line, "\xE2\x9C\x85") ||   // U+2705 white heavy check mark
		startsWith(line, "\xE2\x9C\x93") ||   // U+2713 check mark
		contains(line, "[OK]") ||
		containsNoCase(line, "Render completed") ||
		containsNoCase(line, "render complete") ||
		containsNoCase(line, "rendered successfully") ||
		containsNoCase(line, "saved successfully") ||
		containsNoCase(line, "PNG saved"))
		return LogCategory{"#51CF66", " OK "};

	if (contains(line, "[OptiX]") || contains(line, "[optix]") ||
		contains(line, "OptiX") || containsNoCase(line, "optix_render") ||
		containsNoCase(line, "GPU mode") || contains(line, "NVIDIA") ||
		contains(line, "RTX") || containsNoCase(line, "Launching renderer (GPU"))
		return LogCategory{"#A9E34B", "GPU "};

	if (contains(line, "[cpu_interface]") || containsNoCase(line, "CPU mode") ||
		containsNoCase(line, "Launching renderer (CPU"))
		return LogCategory{"#74C0FC", "CPU "};

	// " spp" alone is too loose: the launcher echoes its settings as
	// "... height=80 spp=4 ...", which matched and tagged a plain settings
	// line as a performance measurement. Requiring digits-then-space is still
	// not enough - "height=80 spp=4" contains "80 spp" - so the assignment
	// form is excluded explicitly. What the rule actually targets is a
	// reported count like "Rendered 800x800 @ 100 spp".
	static const std::regex sppMeasurement(R"(\d\s+spp(?!=))");
	if (contains(line, "RENDER TIME") || containsNoCase(line, "time=") ||
		contains(line, " ms") || std::regex_search(line, sppMeasurement) ||
		contains(line, "Rendered ") || containsNoCase(line, "Pipeline stat"))
		return LogCategory{"#FFA94D", "PERF"};

	if (containsNoCase(line, "Building scene") || containsNoCase(line, "Uploaded ") ||
		contains(line, "Built ") || containsNoCase(line, "materials to GPU") ||
		containsNoCase(line, "spheres to GPU") || containsNoCase(line, "quads to GPU") ||
		containsNoCase(line, "light sources") ||
		containsNoCase(line, "acceleration struct"))
		return LogCategory{"#CC99FF", "SCN "};

	if (contains(line, "Pipeline") || containsNoCase(line, "Initializ") ||
		containsNoCase(line, "Loaded PTX") || containsNoCase(line, "Created program") ||
		contains(line, "Using GPU:"))
		return LogCategory{"#63E6BE", "INIT"};

	if (startsWith(line, "[TECH]"))
		return LogCategory{"#E599F7", "TECH"};

	if (startsWith(line, "Command:"))
		return LogCategory{"#CCCCCC", "CMD "};

	if (contains(line, "[DEBUG]") || contains(line, "Parsed ") ||
		containsNoCase(line, "Using command-line") ||
		containsNoCase(line, "Writing output to"))
		return LogCategory{"#A8A8A8", "DBG "};

	if (startsWith(line, "Process finished") || startsWith(line, "Result:"))
		return LogCategory{"#D0D0D0", "INFO"};

	return LogCategory{};  // plain INFO
}

} // namespace render_output

#endif // RENDER_OUTPUT_PARSER_H
