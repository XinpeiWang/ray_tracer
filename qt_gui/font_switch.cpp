#include "mainwindow.h"
#include "settings_keys.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QFont>
#include <QFontDialog>
#include <QFontInfo>
#include <QMenu>
#include <QMenuBar>
#include <QSettings>
#include <QStatusBar>
#include <QStringList>
#include <QToolTip>
#include <QVector>

// ============================================================================
// Font selection and persistence
// ============================================================================
// Live, like theme switching (theme_switch.cpp) - but qApp->setFont() alone
// is NOT enough, unlike qApp->setPalette(). Confirmed by hand: switching
// fonts after startup silently did nothing visible until applyFont() also
// forced a full stylesheet repolish. That repolish trick used to just
// re-apply the SAME stylesheet string, which turned out to only be half the
// fix: applyTheme()'s own QSS (mainwindow_style.cpp) hardcodes font-size on
// ~25 selectors, so a font choice's point size never reached most of the UI
// even though its family did - Theme and Font looked decoupled but weren't,
// for size specifically. applyTheme() now reads m_activeFontId to scale
// those font-size rules to the active choice, so applyFont() calls
// applyTheme(m_activeTheme) to rebuild the sheet with the new size baked in,
// rather than re-setting an unchanged string that could never reflect it.
//
// No font files are bundled with this app (no .ttf/.otf, no
// QFontDatabase::addApplicationFont), so every choice below is a fallback
// chain of commonly-installed names, the same shape the original
// hardcoded Cyberpunk list already used - first one QFontInfo confirms is
// actually installed wins.
// ============================================================================

namespace {
struct FontChoice {
	const char *id;      // QSettings value; stable, never shown to the user
	const char *name;    // Shown in the Font menu
	QStringList families;  // Fallback chain, first installed wins
	int pointSize;
};

const QVector<FontChoice> &fontChoices() {
	static const QVector<FontChoice> choices = {
		{"cyberpunk", QT_TRANSLATE_NOOP("MainWindow", "Cyberpunk (Default)"),
			{"Orbitron", "Rajdhani", "Exo 2", "Michroma", "Audiowide", "Chakra Petch",
			 "Saira", "Teko", "Electrolize", "Bahnschrift", "Segoe UI", "Arial"}, 11},
		{"system", QT_TRANSLATE_NOOP("MainWindow", "System UI"),
			{"Segoe UI", "Arial"}, 10},
		{"serif", QT_TRANSLATE_NOOP("MainWindow", "Classic Serif"),
			{"Georgia", "Cambria", "Times New Roman", "serif"}, 11},
		{"monospace", QT_TRANSLATE_NOOP("MainWindow", "Monospace"),
			{"Cascadia Code", "Consolas", "Courier New", "monospace"}, 10},
		{"rounded", QT_TRANSLATE_NOOP("MainWindow", "Rounded"),
			{"Segoe UI Variable", "Calibri", "Verdana"}, 10},
		{"compact", QT_TRANSLATE_NOOP("MainWindow", "Compact"),
			{"Tahoma", "Segoe UI", "Arial"}, 9},
		{"largeprint", QT_TRANSLATE_NOOP("MainWindow", "Large Print"),
			{"Segoe UI", "Arial"}, 13},
	};
	return choices;
}

const FontChoice &fontChoiceById(const QString &id) {
	const QVector<FontChoice> &choices = fontChoices();
	for (const FontChoice &f : choices) {
		if (id == QLatin1String(f.id)) return f;
	}
	return choices[0];  // Unknown/corrupt setting - same "fall back to first" rule byId() uses for themes
}
} // namespace

QString MainWindow::loadSavedFontId() {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kFontKey, QStringLiteral("cyberpunk")).toString();
}

void MainWindow::saveFontId(const QString &id) {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kFontKey, id);
}

QString MainWindow::loadSavedCustomFontFamily() {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kFontCustomFamilyKey, QStringLiteral("Segoe UI")).toString();
}

int MainWindow::loadSavedCustomFontSize() {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kFontCustomSizeKey, 10).toInt();
}

void MainWindow::saveCustomFont(const QString &family, int pointSize) {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kFontCustomFamilyKey, family);
	settings.setValue(settings_keys::kFontCustomSizeKey, pointSize);
}

// Exposed to mainwindow_style.cpp so applyTheme() can scale its font-size
// rules without reaching into this file's anonymous namespace - see this
// function's own declaration in mainwindow.h.
int MainWindow::fontPointSizeForId(const QString &id) {
	if (id == QLatin1String("custom")) return loadSavedCustomFontSize();
	return fontChoiceById(id).pointSize;
}

// Walks a FontChoice's fallback chain and applies the first family QFontInfo
// confirms is actually installed - identical logic to what used to live
// inline in applyTheme(), just parameterized instead of hardcoded to one
// choice. Weight stays Normal app-wide (see the original comment this
// preserves): bold is opt-in, on group-box titles and the primary button
// only, not blanket-applied the way an earlier version of this app did.
void MainWindow::applyFont(const QString &id) {
	QFont font;
	QString primaryFamily;
	int pointSize;

	if (id == QLatin1String("custom")) {
		// The "Custom…" choice (createFontMenu()) has no FontChoice fallback
		// chain to walk - QFontDialog already confirmed the family is
		// installed, so it's used directly.
		m_activeFontId = QStringLiteral("custom");
		primaryFamily = loadSavedCustomFontFamily();
		pointSize = loadSavedCustomFontSize();
		font.setFamily(primaryFamily);
	} else {
		const FontChoice &choice = fontChoiceById(id);
		// Resolved id (choice.id), not the raw argument - so a corrupt/unknown
		// saved value settles on the same id the fallback actually applied,
		// keeping this in agreement with createFontMenu()'s checkmark below.
		m_activeFontId = QString::fromUtf8(choice.id);
		pointSize = choice.pointSize;

		primaryFamily = QStringLiteral("Arial");
		bool familySet = false;
		for (const QString &family : choice.families) {
			font.setFamily(family);
			if (QFontInfo(font).family() == family) {
				primaryFamily = family;
				familySet = true;
				break;
			}
		}
		if (!familySet) font.setFamily(primaryFamily);
	}

	// None of the decorative chains above (or a hand-picked custom font) are
	// guaranteed to carry CJK glyphs, so without this, Chinese/Japanese text
	// was left entirely to the OS's own implicit font substitution - usually
	// passable on Windows, but never a choice this app actually made, and not
	// guaranteed at all on other platforms. setFamilies() (not just
	// setFamily()) makes Qt do real per-glyph fallback across this whole
	// list: Latin text still renders from primaryFamily exactly as before,
	// but any CJK glyph Qt can't find there now falls through to a
	// deliberately-picked font instead. The Windows-only names are tried
	// first so a native install still gets its own look; "Noto Sans SC"
	// (bundled via QFontDatabase::addApplicationFont(), see main.cpp) is the
	// guaranteed-present, cross-platform fallback after them. Applies
	// regardless of the active UI language, so e.g. a Chinese scene name
	// still renders well even while the app itself is in English.
	font.setFamilies({primaryFamily,
		QStringLiteral("Microsoft YaHei UI"), QStringLiteral("Microsoft YaHei"),
		QStringLiteral("Yu Gothic UI"), QStringLiteral("Meiryo UI"),
		QStringLiteral("Noto Sans SC")});
	font.setPointSize(pointSize);
	font.setWeight(QFont::Normal);
	qApp->setFont(font);
	// qApp->setFont() alone doesn't reach QToolTip's popup (QTipLabel) - on
	// Windows in particular, the platform theme registers its own font for
	// that class ahead of the generic application default, so the info
	// icons' tooltips silently kept the OS tooltip font regardless of the
	// active Font choice until this explicit override was added.
	QToolTip::setFont(font);

	// applyTheme()'s stylesheet bakes in font-size rules scaled from
	// m_activeFontId (see this file's header comment), so making the new
	// point size visible needs the sheet rebuilt from the current theme, not
	// just reapplied unchanged - the same full unpolish+polish qApp->setFont()
	// alone still can't trigger on its own for already-styled widgets.
	applyTheme(m_activeTheme);
}

void MainWindow::switchFont(const QString &id) {
	applyFont(id);
	saveFontId(id);
	// m_activeFontId is applyFont()'s resolved id (== id, unless id was
	// unknown/corrupt and it fell back to fontChoices()[0]), so re-reading it
	// here keeps this in agreement for "custom" too, which fontChoiceById()
	// can't resolve on its own.
	syncCheckedAction(m_fontActions, m_activeFontId);

	const QString name = (m_activeFontId == QLatin1String("custom"))
		? loadSavedCustomFontFamily()
		: tr(fontChoiceById(m_activeFontId).name);
	statusBar()->showMessage(tr("Font: %1").arg(name), 3000);
}

// Top-level menu, next to Theme - both are "appearance" choices. Same
// exclusive-checkmark QActionGroup pattern as createThemeMenu().
void MainWindow::createFontMenu() {
	QMenu *fontMenu = menuBar()->addMenu(tr("F&ont"));

	auto *group = new QActionGroup(this);
	group->setExclusive(true);

	// Resolved id, not the raw m_startupFontId - see applyFont()'s own
	// comment on why the resolved id is what has to match here too. "custom"
	// already IS resolved (fontChoiceById() only matters for the curated
	// choices), so it's passed through as-is rather than round-tripped
	// through fontChoiceById(), which can't recognize it.
	const QString activeId = (m_startupFontId == QLatin1String("custom"))
		? m_startupFontId
		: QString::fromUtf8(fontChoiceById(m_startupFontId).id);
	for (const FontChoice &choice : fontChoices()) {
		const QString id = QString::fromUtf8(choice.id);
		QAction *action = fontMenu->addAction(tr(choice.name));
		action->setCheckable(true);
		action->setData(id);
		action->setChecked(id == activeId);
		group->addAction(action);
		m_fontActions.push_back(action);

		connect(action, &QAction::triggered, this, [this, id]() {
			switchFont(id);
		});
	}

	fontMenu->addSeparator();
	QAction *customAction = fontMenu->addAction(tr("Custom…"));
	customAction->setCheckable(true);
	customAction->setData(QStringLiteral("custom"));
	customAction->setChecked(activeId == QLatin1String("custom"));
	group->addAction(customAction);
	m_fontActions.push_back(customAction);

	connect(customAction, &QAction::triggered, this, [this]() {
		bool ok = false;
		QFont chosen = QFontDialog::getFont(&ok, qApp->font(), this, tr("Choose Font"));
		if (!ok) {
			// QActionGroup already moved the checkmark to "Custom…" the
			// instant it was clicked, before this dialog's result was known -
			// undo that back to whatever font is still actually active.
			syncCheckedAction(m_fontActions, m_activeFontId);
			return;
		}
		saveCustomFont(chosen.family(), chosen.pointSize());
		switchFont(QStringLiteral("custom"));
	});
}
