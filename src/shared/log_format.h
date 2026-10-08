#pragma once
// log_format.h -- the pieces of the GUI's log file (qt_gui/app_log.cpp) that need no Qt, so they can be unit-tested: one-line entries, size-based rotation
// and reading back the last lines. A log entry is always ONE line ("2026-10-07 21:30:01.123 [INFO ] builder: added Sphere 2 objects"), so grep, tail and a
// bug report's pasted excerpt all work without a viewer.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace log_format {

enum class Level { Debug, Info, Warn, Error };

inline const char* levelName(Level l) {
	switch (l) {
		case Level::Debug: return "DEBUG";
		case Level::Info: return "INFO ";
		case Level::Warn: return "WARN ";
		case Level::Error: return "ERROR";
	}
	return "INFO ";
}

// A message as one line: line breaks become " | " (no empty segments), trailing space goes, and a very long one is cut with a note of how much was dropped.
inline std::string oneLine(const std::string& message, std::size_t maxLength = 4000) {
	std::string out;
	out.reserve(message.size());
	bool pendingBreak = false;
	for (char c : message) {
		if (c == '\n' || c == '\r') { pendingBreak = !out.empty(); continue; }
		if (pendingBreak) { out += " | "; pendingBreak = false; }
		out += c;
	}
	while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
	if (out.size() > maxLength) {
		const std::size_t dropped = out.size() - maxLength;
		out.resize(maxLength);
		out += "... (+" + std::to_string(dropped) + " characters)";
	}
	return out;
}

inline std::string formatLine(const std::string& timestamp, Level level, const std::string& category, const std::string& message) {
	return timestamp + " [" + levelName(level) + "] " + category + ": " + oneLine(message);
}

// When `file` has grown past `maxBytes`, moves it to file.1 (file.1 to file.2, ...), keeping `keep` old ones. Returns true if it rotated.
inline bool rotateIfLarge(const std::filesystem::path& file, std::uintmax_t maxBytes, int keep) {
	namespace fs = std::filesystem;
	std::error_code ec;
	if (!fs::is_regular_file(file, ec) || fs::file_size(file, ec) <= maxBytes) return false;
	auto numbered = [&](int n) { return fs::path(file.string() + "." + std::to_string(n)); };
	fs::remove(numbered(keep), ec);
	for (int n = keep - 1; n >= 1; --n)
		if (fs::exists(numbered(n), ec)) fs::rename(numbered(n), numbered(n + 1), ec);
	fs::rename(file, numbered(1), ec);
	return !ec;
}

// The last `count` lines of a text file (all of them if it is shorter); empty if it cannot be read.
inline std::vector<std::string> tailLines(const std::filesystem::path& file, std::size_t count) {
	std::ifstream in(file, std::ios::binary);
	std::vector<std::string> lines;
	std::string line;
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		lines.push_back(line);
		if (lines.size() > count * 2 + 16) lines.erase(lines.begin(), lines.end() - static_cast<std::ptrdiff_t>(count));
	}
	if (lines.size() > count) lines.erase(lines.begin(), lines.end() - static_cast<std::ptrdiff_t>(count));
	return lines;
}

}  // namespace log_format
