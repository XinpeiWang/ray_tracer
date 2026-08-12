#ifndef THEME_H
#define THEME_H

#include <QColor>
#include <QString>
#include <QVector>

#include "palette_data.h"
#include "render_output_parser.h"

// ============================================================================
// Themes
// ============================================================================
// A theme is data, not code. Everything the UI paints comes from one of these
// structs, so adding a colour scheme means adding a table entry rather than
// touching stylesheet strings.
//
// The palettes are the published values of well-known developer colour schemes
// (Solarized, Dracula, Nord, Gruvbox, Breeze, ...) rather than invented ones,
// so they land somewhere users already recognise and have been eye-tested by a
// lot more people than this project has. Only the colour VALUES are used - no
// artwork, code or assets from those projects - and each entry records where
// its numbers come from.
//
// Roles are named by what they are FOR, not what they look like, which is what
// lets one stylesheet serve a near-black cyberpunk scheme and a paper-white one
// without branching.
// ============================================================================
namespace theme {

struct Palette {
	QString id;        // stable key written to QSettings
	QString name;      // shown in the menu
	QString origin;    // where the palette comes from, for the About/tooltip

	// Surface ramp, deepest first.
	QColor surface0;   // window / deepest background
	QColor surface1;   // panels, group boxes, inputs
	QColor surface2;   // raised or hovered controls
	QColor surface3;   // pressed / selected

	// Text.
	QColor textBody;
	QColor textMuted;
	QColor textDisabled;

	// Accents. accentPrimary drives the primary action and selection;
	// accentSecondary is the focus/active marker.
	QColor accentPrimary;
	QColor accentSecondary;
	QColor accentDim;      // primary border at rest

	// The primary button's gradient. Explicit rather than derived from the
	// accent: deriving it looked plausible on the purple scheme it was tuned
	// against and wrong on every other one.
	QColor primaryTop;
	QColor primaryBottom;
	QColor primaryTopHover;
	QColor primaryBottomHover;

	// Lines.
	QColor border;         // ordinary 1px chrome
	QColor borderStrong;   // interactive controls
	QColor hoverRow;       // list/menu row under the cursor

	// Outcome colours for the progress bar.
	QColor success;
	QColor error;

	// One colour per log severity. Kept explicit per theme because a palette
	// that reads well on near-black is often unreadable on white - the log is
	// the densest coloured surface in the app.
	QColor logInfo;
	QColor logError;
	QColor logWarning;
	QColor logSuccess;
	QColor logGpu;
	QColor logCpu;
	QColor logPerformance;
	QColor logScene;
	QColor logInit;
	QColor logTechnique;
	QColor logCommand;
	QColor logDebug;
	QColor logSeparator;

	QColor colourFor(render_output::LogSeverity severity) const;

	// Optional decorative motif, painted on the tab pane's background - i.e.
	// the area AROUND the group boxes, never underneath their text. Panels stay
	// fully opaque on purpose: ghosting a pattern through body text and dense
	// log lines costs legibility for very little character. What you get
	// instead is the motif in the margins and the gaps between panels.
	//
	// Opacity is baked into the artwork rather than applied at runtime, because
	// Qt stylesheets have no opacity property for background-image and each of
	// these is drawn for one specific palette anyway.
	QString backgroundImage;              // qrc path; empty = no motif
	bool    backgroundTiled = false;      // true = seamless repeat, false = single placement
	QString backgroundPosition = "bottom right";  // ignored when tiled
};

// All available themes, in menu order. The first entry is the default.
const QVector<Palette> &all();

// Looks a theme up by its settings id; falls back to the default when the id
// is unknown, which is what happens when settings were written by a newer
// build or a theme is removed.
const Palette &byId(const QString &id);

const Palette &defaultPalette();

// Reads *.theme files from <app dir>/themes and the per-user config directory.
// Called once by all(); exposed so the GUI can report what failed to load.
// Never throws and never fails as a whole - a bad file is skipped and appended
// to `problems` (if given), because a theme file is user content and the app
// has to start without it.
QVector<palette_data::PaletteData> loadUserPalettes(QStringList *problems = nullptr);

// Whatever loadUserPalettes() reported the first time the registry was built.
const QStringList &userThemeProblems();

} // namespace theme

#endif // THEME_H
