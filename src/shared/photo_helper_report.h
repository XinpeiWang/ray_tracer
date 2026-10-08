// photo_helper_report.h - the text side of the Scene Builder's optional "Object from a photo" helper, with no Qt in it so it can be
// unit-tested: the helper's progress lines, the Diagnostics "Photo helper" section's missing facts, and the splitter that turns a
// subprocess's byte stream into lines. (The Qt code that runs the processes is qt_gui/photo_import.*.)
#pragma once

#include <string>
#include <vector>

namespace photo_report {

inline std::string trim(const std::string& s) {
	const char* ws = " \t\r\n";
	const size_t a = s.find_first_not_of(ws);
	if (a == std::string::npos) return std::string();
	return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

inline bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

inline std::string lower(std::string s) {
	for (char& c : s)
		if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
	return s;
}

// "PROGRESS 35 Guessing the 3D shape" -> percent 35 (clamped to 0..100) and the message. False for any other line.
inline bool parseProgressLine(const std::string& line, int& percent, std::string& message) {
	const std::string tag = "PROGRESS ";
	if (!startsWith(line, tag.c_str())) return false;
	const std::string rest = line.substr(tag.size());
	const size_t space = rest.find(' ');
	const std::string number = space == std::string::npos ? rest : rest.substr(0, space);
	if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos || number.size() > 9) return false;
	const int p = std::stoi(number);
	percent = p > 100 ? 100 : p;
	message = space == std::string::npos ? std::string() : trim(rest.substr(space + 1));
	return true;
}

// The lines of the report's "=== Photo helper" section that the setup script would fix: a missing environment, package, model file or
// the TripoSR code, and a broken install. Not the graphics card, and not a missing helper script (setup does not provide that).
inline std::vector<std::string> missingFacts(const std::string& report) {
	std::vector<std::string> out;
	const size_t at = report.find("=== Photo helper");
	if (at == std::string::npos) return out;
	std::string section = report.substr(at);
	const size_t next = section.find("\n===");  // the next section's banner; this one's own banner is on the first line
	if (next != std::string::npos) section = section.substr(0, next);
	size_t pos = 0;
	while (pos <= section.size()) {
		size_t eol = section.find('\n', pos);
		if (eol == std::string::npos) eol = section.size();
		const std::string line = trim(section.substr(pos, eol - pos));
		pos = eol + 1;
		const size_t colon = line.find(':');
		if (startsWith(line, "===") || colon == std::string::npos) continue;
		const std::string key = line.substr(0, colon);
		if (key == "Helper Script" || key == "Expected At" || key == "Python" || startsWith(key, "Graphics Card") || startsWith(key, "Saved Photo Objects")) continue;
		const std::string low = lower(line);
		if (low.find("missing") != std::string::npos || low.find("not usable") != std::string::npos || low.find("not available") != std::string::npos)
			out.push_back(line);
	}
	return out;
}

// Turns a subprocess's output into lines. A \n, \r\n or \r\r\n ends a line; a lone \r (pip redrawing its progress bar) ends a "status" piece
// that replaces the previous status instead of adding a line. Feed it whatever bytes arrive, in any chunking.
class LineSplitter {
public:
	struct Piece {
		std::string text;    // trimmed, never empty
		bool endsLine;       // true: a finished line; false: a status redraw
	};

	std::vector<Piece> feed(const std::string& chunk) {
		pending_ += chunk;
		std::vector<Piece> out;
		std::string current;
		bool trailingCr = false;
		for (size_t i = 0; i < pending_.size(); ++i) {
			const char c = pending_[i];
			if (c == '\n') {
				push(out, current, true);
			} else if (c == '\r') {
				// A run of CRs: followed by \n it is part of the line ending (\r\n, and the \r\r\n that Windows PowerShell makes of a native
				// command's output); followed by text it is a redraw; at the end of the data it may still become either.
				size_t j = i;
				while (j < pending_.size() && pending_[j] == '\r') ++j;
				if (j == pending_.size()) { trailingCr = true; break; }
				if (pending_[j] != '\n') push(out, current, false);
				i = j - 1;  // (for a \n, the next turn ends the line)
			} else {
				current += c;
			}
		}
		pending_ = current + (trailingCr ? "\r" : "");
		return out;
	}

	// Whatever has not ended a line yet (at the end of the stream), trimmed.
	std::string flush() {
		const std::string rest = trim(pending_);
		pending_.clear();
		return rest;
	}

private:
	static void push(std::vector<Piece>& out, std::string& current, bool endsLine) {
		const std::string text = trim(current);
		if (!text.empty()) out.push_back({text, endsLine});
		current.clear();
	}
	std::string pending_;
};

}  // namespace photo_report
