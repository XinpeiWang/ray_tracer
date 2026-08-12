#ifndef PALETTE_FILE_H
#define PALETTE_FILE_H

#include "palette_data.h"

#include <map>
#include <string>

// ============================================================================
// Reading a palette from a key/value file - deliberately Qt-free
// ============================================================================
// Themes were compile-time only: adding one meant editing palette_data.cpp and
// rebuilding, which is fine for the twelve that ship and useless for anyone who
// just wants their own colours. This parses one into the same PaletteData the
// built-ins use, so a loaded theme is not a second-class citizen - it goes
// through the identical registry, lookup, menu and contrast checks.
//
// Qt-free for the usual reason (see palette_data.h): the MSVC test binary
// cannot include Qt headers, and the interesting failure modes here are all in
// the parsing - a missing key, a malformed colour, an id that collides with a
// built-in. The Qt side does nothing but read the file into a map and hand it
// over, which is the part with nothing to get wrong.
// ============================================================================
namespace palette_file {

// Parses "#rrggbb" (the leading # optional, case-insensitive). Returns false
// and leaves `out` untouched on anything else - a short string, a stray
// character, the empty string.
bool parseColour(const std::string &text, palette_data::Rgb &out);

struct ParseResult {
	bool ok = false;
	std::string error;              // human-readable, names the offending key
	palette_data::PaletteData palette;
};

// Builds a palette from a flat key -> value map (INI keys, lower-cased by the
// caller). Every colour role is required: a partially specified theme would
// otherwise silently get black wherever it forgot something, which is exactly
// the failure the built-in tables are checked against.
//
// `reservedIds` rejects a file trying to shadow a built-in, which would make
// the built-in permanently unreachable - byId() returns whichever comes first.
ParseResult parse(const std::map<std::string, std::string> &values,
				  const std::vector<std::string> &reservedIds);

// The keys parse() requires, in a stable order. Exposed so the documentation
// and the sample file cannot drift from what the parser actually wants.
const std::vector<std::string> &requiredKeys();

} // namespace palette_file

#endif // PALETTE_FILE_H
