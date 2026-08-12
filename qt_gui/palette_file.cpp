#include "palette_file.h"

#include <algorithm>
#include <cctype>

namespace palette_file {
namespace {

// Every colour role, paired with where it lives in PaletteData. One table
// drives both the required-key list and the assignment, so a role can't be
// required but never stored, or stored but never required.
struct ColourField {
	const char *key;
	palette_data::Rgb palette_data::PaletteData::*member;
};

const ColourField kColourFields[] = {
	{"surface0", &palette_data::PaletteData::surface0},
	{"surface1", &palette_data::PaletteData::surface1},
	{"surface2", &palette_data::PaletteData::surface2},
	{"surface3", &palette_data::PaletteData::surface3},
	{"textbody", &palette_data::PaletteData::textBody},
	{"textmuted", &palette_data::PaletteData::textMuted},
	{"textdisabled", &palette_data::PaletteData::textDisabled},
	{"accentprimary", &palette_data::PaletteData::accentPrimary},
	{"accentsecondary", &palette_data::PaletteData::accentSecondary},
	{"accentdim", &palette_data::PaletteData::accentDim},
	{"primarytop", &palette_data::PaletteData::primaryTop},
	{"primarybottom", &palette_data::PaletteData::primaryBottom},
	{"primarytophover", &palette_data::PaletteData::primaryTopHover},
	{"primarybottomhover", &palette_data::PaletteData::primaryBottomHover},
	{"border", &palette_data::PaletteData::border},
	{"borderstrong", &palette_data::PaletteData::borderStrong},
	{"hoverrow", &palette_data::PaletteData::hoverRow},
	{"success", &palette_data::PaletteData::success},
	{"error", &palette_data::PaletteData::error},
	{"loginfo", &palette_data::PaletteData::logInfo},
	{"logerror", &palette_data::PaletteData::logError},
	{"logwarning", &palette_data::PaletteData::logWarning},
	{"logsuccess", &palette_data::PaletteData::logSuccess},
	{"loggpu", &palette_data::PaletteData::logGpu},
	{"logcpu", &palette_data::PaletteData::logCpu},
	{"logperformance", &palette_data::PaletteData::logPerformance},
	{"logscene", &palette_data::PaletteData::logScene},
	{"loginit", &palette_data::PaletteData::logInit},
	{"logtechnique", &palette_data::PaletteData::logTechnique},
	{"logcommand", &palette_data::PaletteData::logCommand},
	{"logdebug", &palette_data::PaletteData::logDebug},
	{"logseparator", &palette_data::PaletteData::logSeparator},
};

int hexDigit(char c) {
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

std::string trimmed(const std::string &s) {
	const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
	auto b = std::find_if(s.begin(), s.end(), notSpace);
	auto e = std::find_if(s.rbegin(), s.rend(), notSpace).base();
	return b < e ? std::string(b, e) : std::string();
}

const std::string *find(const std::map<std::string, std::string> &m,
						const std::string &key) {
	auto it = m.find(key);
	return it == m.end() ? nullptr : &it->second;
}

} // namespace

bool parseColour(const std::string &text, palette_data::Rgb &out) {
	std::string s = trimmed(text);
	if (!s.empty() && s.front() == '#') s.erase(s.begin());
	if (s.size() != 6) return false;

	int v[6];
	for (int i = 0; i < 6; ++i) {
		v[i] = hexDigit(s[static_cast<std::size_t>(i)]);
		if (v[i] < 0) return false;
	}
	out.r = static_cast<unsigned char>(v[0] * 16 + v[1]);
	out.g = static_cast<unsigned char>(v[2] * 16 + v[3]);
	out.b = static_cast<unsigned char>(v[4] * 16 + v[5]);
	return true;
}

const std::vector<std::string> &requiredKeys() {
	static const std::vector<std::string> keys = []() {
		std::vector<std::string> k{"id", "name"};
		for (const ColourField &f : kColourFields) k.push_back(f.key);
		return k;
	}();
	return keys;
}

ParseResult parse(const std::map<std::string, std::string> &values,
				  const std::vector<std::string> &reservedIds) {
	ParseResult r;

	const std::string *id = find(values, "id");
	if (!id || trimmed(*id).empty()) {
		r.error = "missing required key 'id'";
		return r;
	}
	r.palette.id = trimmed(*id);

	if (std::find(reservedIds.begin(), reservedIds.end(), r.palette.id)
		!= reservedIds.end()) {
		// Allowing this would hide the built-in rather than replace it: the
		// registry is searched in order and the built-ins come first, so the
		// file would load, appear in the menu, and never be selectable.
		r.error = "id '" + r.palette.id + "' is already used by a built-in theme";
		return r;
	}

	const std::string *name = find(values, "name");
	if (!name || trimmed(*name).empty()) {
		r.error = "missing required key 'name'";
		return r;
	}
	r.palette.name = trimmed(*name);

	for (const ColourField &f : kColourFields) {
		const std::string *raw = find(values, f.key);
		if (!raw) {
			r.error = std::string("missing required colour '") + f.key + "'";
			return r;
		}
		palette_data::Rgb c;
		if (!parseColour(*raw, c)) {
			r.error = std::string("'") + f.key + "' is not a #rrggbb colour: '"
					  + trimmed(*raw) + "'";
			return r;
		}
		r.palette.*(f.member) = c;
	}

	// Optional. origin defaults to something honest rather than empty, because
	// it is shown as the menu entry's status tip.
	const std::string *origin = find(values, "origin");
	r.palette.origin = (origin && !trimmed(*origin).empty())
						   ? trimmed(*origin)
						   : "Custom theme loaded from file";

	r.ok = true;
	return r;
}

} // namespace palette_file
