/**
 * @file palette_file_tests.cpp
 * @brief Unit tests for parsing a user-supplied .theme file
 *
 * A theme file is user content, so every one of these failure modes is
 * something a real person will hit: a typo'd colour, a forgotten role, an id
 * that happens to match a built-in. Each has to produce a message naming what
 * is wrong rather than a black-looking theme or a silent skip.
 */

#include <gtest/gtest.h>

#include "../../qt_gui/palette_file.h"

#include <map>
#include <string>

using namespace palette_file;
using palette_data::Rgb;

namespace {

// A file with every required key present and valid, which individual tests
// then break in one specific way.
std::map<std::string, std::string> validFile() {
	std::map<std::string, std::string> m;
	m["id"] = "my-theme";
	m["name"] = "My Theme";
	for (const std::string &k : requiredKeys()) {
		if (k == "id" || k == "name") continue;
		m[k] = "#123456";
	}
	return m;
}

} // namespace

// ===========================================================================
// Colour parsing
// ===========================================================================

TEST(ParseColourTest, AcceptsHashPrefixedAndBare) {
	Rgb c;
	ASSERT_TRUE(parseColour("#1a2b3c", c));
	EXPECT_EQ(c.r, 0x1a); EXPECT_EQ(c.g, 0x2b); EXPECT_EQ(c.b, 0x3c);

	ASSERT_TRUE(parseColour("1a2b3c", c));
	EXPECT_EQ(c.r, 0x1a); EXPECT_EQ(c.g, 0x2b); EXPECT_EQ(c.b, 0x3c);
}

TEST(ParseColourTest, IsCaseInsensitiveAndTrimsWhitespace) {
	Rgb lower, upper, padded;
	ASSERT_TRUE(parseColour("#abcdef", lower));
	ASSERT_TRUE(parseColour("#ABCDEF", upper));
	ASSERT_TRUE(parseColour("  #abcdef  ", padded));
	EXPECT_TRUE(lower == upper);
	EXPECT_TRUE(lower == padded);
}

TEST(ParseColourTest, RejectsMalformedInput) {
	Rgb c;
	EXPECT_FALSE(parseColour("", c))          << "empty";
	EXPECT_FALSE(parseColour("#abc", c))      << "three-digit shorthand is not supported";
	EXPECT_FALSE(parseColour("#1234567", c))  << "too long";
	EXPECT_FALSE(parseColour("#12345g", c))   << "non-hex digit";
	EXPECT_FALSE(parseColour("rebeccapurple", c)) << "colour names are not supported";
}

TEST(ParseColourTest, LeavesOutputUntouchedOnFailure) {
	// Callers rely on this: a failed parse must not half-write a colour.
	Rgb c{9, 9, 9};
	EXPECT_FALSE(parseColour("nope", c));
	EXPECT_EQ(c.r, 9); EXPECT_EQ(c.g, 9); EXPECT_EQ(c.b, 9);
}

// ===========================================================================
// Whole-file parsing
// ===========================================================================

TEST(PaletteFileTest, ParsesACompleteFile) {
	auto m = validFile();
	m["surface0"] = "#0a0b0c";
	m["logdebug"] = "#ffeedd";

	const ParseResult r = parse(m, {});
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.palette.id, "my-theme");
	EXPECT_EQ(r.palette.name, "My Theme");
	EXPECT_TRUE(r.palette.surface0 == (Rgb{0x0a, 0x0b, 0x0c}));
	EXPECT_TRUE(r.palette.logDebug == (Rgb{0xff, 0xee, 0xdd}));
}

TEST(PaletteFileTest, OriginDefaultsToSomethingHonest) {
	// origin is shown as the menu entry's status tip, so an empty one would
	// leave a blank tooltip rather than saying where the theme came from.
	const ParseResult r = parse(validFile(), {});
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FALSE(r.palette.origin.empty());
}

TEST(PaletteFileTest, OriginIsUsedWhenGiven) {
	auto m = validFile();
	m["origin"] = "Made by me";
	const ParseResult r = parse(m, {});
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.palette.origin, "Made by me");
}

TEST(PaletteFileTest, RejectsMissingId) {
	auto m = validFile();
	m.erase("id");
	const ParseResult r = parse(m, {});
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("id"), std::string::npos) << "error should name the key";
}

TEST(PaletteFileTest, RejectsMissingName) {
	auto m = validFile();
	m.erase("name");
	const ParseResult r = parse(m, {});
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("name"), std::string::npos);
}

TEST(PaletteFileTest, RejectsAnyMissingColour) {
	// Every role, one at a time - a partial theme would otherwise get black
	// wherever it forgot something, which is the exact failure the built-in
	// tables are checked against.
	for (const std::string &key : requiredKeys()) {
		if (key == "id" || key == "name") continue;
		auto m = validFile();
		m.erase(key);
		const ParseResult r = parse(m, {});
		EXPECT_FALSE(r.ok) << "a file missing '" << key << "' was accepted";
		EXPECT_NE(r.error.find(key), std::string::npos)
			<< "error for missing '" << key << "' does not name it: " << r.error;
	}
}

TEST(PaletteFileTest, RejectsMalformedColourAndNamesTheKey) {
	auto m = validFile();
	m["accentprimary"] = "octarine";
	const ParseResult r = parse(m, {});
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("accentprimary"), std::string::npos) << r.error;
	EXPECT_NE(r.error.find("octarine"), std::string::npos)
		<< "error should quote the offending value: " << r.error;
}

TEST(PaletteFileTest, RejectsIdThatShadowsABuiltIn) {
	// Not merely tidiness: the registry is searched in order with built-ins
	// first, so a shadowing theme would load, appear in the menu, and never be
	// selectable. Failing loudly beats that.
	auto m = validFile();
	m["id"] = "cyberpunk";
	const ParseResult r = parse(m, {"cyberpunk", "dracula"});
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("cyberpunk"), std::string::npos) << r.error;
}

TEST(PaletteFileTest, AllowsAnIdThatIsNotReserved) {
	auto m = validFile();
	m["id"] = "not-taken";
	const ParseResult r = parse(m, {"cyberpunk", "dracula"});
	EXPECT_TRUE(r.ok) << r.error;
}

TEST(PaletteFileTest, RequiredKeysCoversEveryColourRole) {
	// requiredKeys() drives both the docs and the sample file, so it must not
	// drift from the parser. 32 colours + id + name.
	EXPECT_EQ(requiredKeys().size(), 34u);
}
