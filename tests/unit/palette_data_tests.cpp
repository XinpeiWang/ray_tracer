/**
 * @file palette_data_tests.cpp
 * @brief Unit tests for the GUI's colour palettes
 *
 * These exist because until the palette tables were split out of theme.cpp
 * (which is Qt code, and so unreachable from this MSVC-built binary) nothing
 * checked them at all - twelve palettes of thirty-two colours, 384 hand-typed
 * values, entirely unverified.
 *
 * Two bugs had already shipped that a contrast assertion would have caught the
 * moment it was written:
 *   - a completed progress bar drew a dark-olive "100%" on Solarized Light's
 *     dark-olive success fill: technically painted, effectively invisible
 *   - scene warnings used a fixed gold that is unreadable on cream
 * Both were found by eye, late. ContrastTest below is the mechanical version.
 */

#include <gtest/gtest.h>

#include "../../qt_gui/palette_data.h"

#include <set>
#include <string>
#include <vector>

using namespace palette_data;

namespace {

std::vector<const PaletteData *> allPalettes() {
	std::vector<const PaletteData *> v;
	for (std::size_t i = 0; i < builtinCount(); ++i)
		v.push_back(&builtins()[i]);
	return v;
}

// ---------------------------------------------------------------------------
// Schemes that knowingly sit below WCAG AA
// ---------------------------------------------------------------------------
// Solarized and Nord are transcriptions of published palettes, not values
// invented here (see qt_gui/THEMES.md). Both are famously low-contrast by
// design - Solarized in particular picks its greys for a narrow luminance
// span on purpose - and neither reaches 4.5:1 for body text. Nudging the
// numbers until they pass would mean shipping something that says "Solarized"
// but is not, which is a worse outcome than the honest shortfall: a user who
// picks Solarized is asking for Solarized.
//
// So they are held to a lower floor rather than exempted outright. The floor
// is set just under what they measure today, which means it still catches a
// regression - if someone edits one of these and makes it worse, the test
// fails. It just does not demand they be something they are not.
bool isLowContrastByDesign(const std::string &id) {
	return id == "nord" || id == "solarized-dark" || id == "solarized-light";
}

// Every colour field, paired with its name so a failure says which one.
std::vector<std::pair<const char *, Rgb>> fieldsOf(const PaletteData &p) {
	return {
		{"surface0", p.surface0}, {"surface1", p.surface1},
		{"surface2", p.surface2}, {"surface3", p.surface3},
		{"textBody", p.textBody}, {"textMuted", p.textMuted},
		{"textDisabled", p.textDisabled},
		{"accentPrimary", p.accentPrimary}, {"accentSecondary", p.accentSecondary},
		{"accentDim", p.accentDim},
		{"primaryTop", p.primaryTop}, {"primaryBottom", p.primaryBottom},
		{"primaryTopHover", p.primaryTopHover},
		{"primaryBottomHover", p.primaryBottomHover},
		{"border", p.border}, {"borderStrong", p.borderStrong},
		{"hoverRow", p.hoverRow},
		{"success", p.success}, {"error", p.error},
		{"logInfo", p.logInfo}, {"logError", p.logError},
		{"logWarning", p.logWarning}, {"logSuccess", p.logSuccess},
		{"logGpu", p.logGpu}, {"logCpu", p.logCpu},
		{"logPerformance", p.logPerformance}, {"logScene", p.logScene},
		{"logInit", p.logInit}, {"logTechnique", p.logTechnique},
		{"logCommand", p.logCommand}, {"logDebug", p.logDebug},
		{"logSeparator", p.logSeparator},
	};
}

} // namespace

// ===========================================================================
// Structure
// ===========================================================================

TEST(PaletteDataTest, RegistryIsNonEmpty) {
	EXPECT_GT(builtinCount(), 0u);
}

TEST(PaletteDataTest, HasExpectedCount) {
	// Update deliberately when adding a theme, so one appearing or vanishing by
	// accident is loud.
	EXPECT_EQ(builtinCount(), 12u);
}

TEST(PaletteDataTest, IdsAreUniqueAndNonEmpty) {
	// The id is what gets written to QSettings, so a duplicate would make one
	// theme permanently unreachable - byId() returns whichever comes first.
	std::set<std::string> seen;
	for (const PaletteData *p : allPalettes()) {
		ASSERT_NE(p->id, nullptr);
		const std::string id = p->id;
		EXPECT_FALSE(id.empty()) << "empty theme id";
		EXPECT_TRUE(seen.insert(id).second) << "duplicate theme id '" << id << "'";
	}
}

TEST(PaletteDataTest, NamesAndOriginsAreNonEmpty) {
	for (const PaletteData *p : allPalettes()) {
		ASSERT_NE(p->name, nullptr);
		ASSERT_NE(p->origin, nullptr);
		EXPECT_GT(std::string(p->name).size(), 0u) << "theme " << p->id;
		// origin is shown as the menu entry's status tip and records where the
		// numbers came from; an empty one means an unattributed palette.
		EXPECT_GT(std::string(p->origin).size(), 0u) << "theme " << p->id;
	}
}

TEST(PaletteDataTest, EveryColourFieldIsAssigned) {
	// A value-initialised Rgb is {0,0,0}. No palette uses pure black on
	// purpose, so this catches the failure that used to be entirely silent:
	// adding a theme and forgetting a field, which rendered black text on a
	// light scheme with no warning anywhere.
	for (const PaletteData *p : allPalettes()) {
		for (const auto &f : fieldsOf(*p)) {
			EXPECT_FALSE(f.second == kUnset)
				<< "theme '" << p->id << "' leaves " << f.first
				<< " unset (pure black is the sentinel for 'never assigned')";
		}
	}
}

TEST(PaletteDataTest, TiledBackgroundsDeclareAnImage) {
	for (const PaletteData *p : allPalettes()) {
		if (p->backgroundTiled) {
			ASSERT_NE(p->backgroundImage, nullptr) << "theme " << p->id;
			EXPECT_GT(std::string(p->backgroundImage).size(), 0u)
				<< "theme '" << p->id << "' is marked tiled but has no image";
		}
	}
}

// ===========================================================================
// Contrast (WCAG 2.1)
// ===========================================================================

TEST(ContrastTest, KnownRatiosAreCorrect) {
	// Anchor the maths before trusting it on real palettes.
	const Rgb black{0, 0, 0}, white{255, 255, 255};
	EXPECT_NEAR(contrastRatio(black, white), 21.0, 0.01);
	EXPECT_NEAR(contrastRatio(white, white), 1.0, 0.001);
	// Order must not matter.
	EXPECT_NEAR(contrastRatio(black, white), contrastRatio(white, black), 1e-9);
}

TEST(ContrastTest, BodyTextMeetsAaOnEverySurface) {
	// 4.5:1 is WCAG AA for body text. textBody lands on surface0 (the window)
	// and surface1 (panels); both have to pass.
	for (const PaletteData *p : allPalettes()) {
		const double floor = isLowContrastByDesign(p->id) ? 3.5 : 4.5;
		EXPECT_GE(contrastRatio(p->textBody, p->surface0), floor)
			<< "theme '" << p->id << "': body text on surface0";
		EXPECT_GE(contrastRatio(p->textBody, p->surface1), floor)
			<< "theme '" << p->id << "': body text on surface1";
	}
}

TEST(ContrastTest, MutedTextMeetsLargeTextMinimum) {
	// Muted text is secondary information at small sizes but is never the only
	// route to anything, so it is held to 3.0:1 rather than 4.5:1.
	for (const PaletteData *p : allPalettes()) {
		const double floor = isLowContrastByDesign(p->id) ? 2.5 : 3.0;
		EXPECT_GE(contrastRatio(p->textMuted, p->surface0), floor)
			<< "theme '" << p->id << "': muted text on surface0";
		EXPECT_GE(contrastRatio(p->textMuted, p->surface1), floor)
			<< "theme '" << p->id << "': muted text on surface1";
	}
}

TEST(ContrastTest, EveryLogSeverityIsReadableOnTheLogSurface) {
	// The log pane is drawn on surface0 and is the densest coloured surface in
	// the app - twelve severities carrying text, each its own colour. This is
	// where a palette is most likely to have one line quietly unreadable.
	//
	// logSeparator is deliberately absent: it colours the "=====" rules between
	// sections, which are decoration rather than information. WCAG exempts
	// purely decorative content for good reason - a divider forced to 3:1 stops
	// being a divider and starts competing with the text it separates. It gets
	// its own, much weaker check below.
	for (const PaletteData *p : allPalettes()) {
		const double floor = isLowContrastByDesign(p->id) ? 2.8 : 3.0;
		const std::pair<const char *, Rgb> logs[] = {
			{"logInfo", p->logInfo}, {"logError", p->logError},
			{"logWarning", p->logWarning}, {"logSuccess", p->logSuccess},
			{"logGpu", p->logGpu}, {"logCpu", p->logCpu},
			{"logPerformance", p->logPerformance}, {"logScene", p->logScene},
			{"logInit", p->logInit}, {"logTechnique", p->logTechnique},
			{"logCommand", p->logCommand}, {"logDebug", p->logDebug},
		};
		for (const auto &l : logs) {
			EXPECT_GE(contrastRatio(l.second, p->surface0), floor)
				<< "theme '" << p->id << "': " << l.first << " on the log surface";
		}
	}
}

TEST(ContrastTest, SeparatorsArePerceptibleWithoutCompeting) {
	// Decoration, so the requirement is only that it can be seen at all - the
	// failure this guards against is a separator that vanishes into the
	// background entirely, which has happened on a theme whose author copied
	// the surface colour by accident.
	for (const PaletteData *p : allPalettes()) {
		const double r = contrastRatio(p->logSeparator, p->surface0);
		EXPECT_GE(r, 1.35) << "theme '" << p->id << "': separator is invisible";
		EXPECT_LE(r, 4.5) << "theme '" << p->id
						  << "': separator is loud enough to compete with body text";
	}
}

TEST(ContrastTest, ProgressBarOutcomeTextIsReadable) {
	// The regression test for a bug that shipped. A finished bar is filled edge
	// to edge, so its percentage text sits on success/error, not on the trough.
	// mainwindow_style.cpp picks white or surface0 for that text, whichever
	// contrasts better; this asserts the better of the two actually clears the
	// bar rather than merely being the lesser of two bad options.
	const Rgb white{255, 255, 255};
	for (const PaletteData *p : allPalettes()) {
		for (const auto &fill : {std::make_pair("success", p->success),
								 std::make_pair("error", p->error)}) {
			const double best = std::max(contrastRatio(white, fill.second),
										 contrastRatio(p->surface0, fill.second));
			EXPECT_GE(best, 3.0)
				<< "theme '" << p->id << "': no readable text colour on the "
				<< fill.first << " fill";
		}
	}
}

TEST(ContrastTest, PrimaryButtonLabelIsReadableOnItsGradient) {
	// The primary action draws accentPrimary text over a primaryTop ->
	// primaryBottom gradient. Both ends have to work; checking only one would
	// miss a gradient that goes bad halfway.
	for (const PaletteData *p : allPalettes()) {
		const double floor = isLowContrastByDesign(p->id) ? 2.6 : 3.0;
		EXPECT_GE(contrastRatio(p->accentPrimary, p->primaryTop), floor)
			<< "theme '" << p->id << "': primary label on the gradient's top";
		EXPECT_GE(contrastRatio(p->accentPrimary, p->primaryBottom), floor)
			<< "theme '" << p->id << "': primary label on the gradient's bottom";
	}
}
