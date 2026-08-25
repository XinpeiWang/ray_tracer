#include "mainwindow.h"
#include "settings_keys.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QFont>
#include <QFontInfo>
#include <QMenu>
#include <QMenuBar>
#include <QSettings>
#include <QStatusBar>
#include <QStringList>
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

// Exposed to mainwindow_style.cpp so applyTheme() can scale its font-size
// rules without reaching into this file's anonymous namespace - see this
// function's own declaration in mainwindow.h.
int MainWindow::fontPointSizeForId(const QString &id) {
	return fontChoiceById(id).pointSize;
}

// Walks a FontChoice's fallback chain and applies the first family QFontInfo
// confirms is actually installed - identical logic to what used to live
// inline in applyTheme(), just parameterized instead of hardcoded to one
// choice. Weight stays Normal app-wide (see the original comment this
// preserves): bold is opt-in, on group-box titles and the primary button
// only, not blanket-applied the way an earlier version of this app did.
void MainWindow::applyFont(const QString &id) {
	const FontChoice &choice = fontChoiceById(id);
	// Resolved id (choice.id), not the raw argument - so a corrupt/unknown
	// saved value settles on the same id the fallback actually applied,
	// keeping this in agreement with createFontMenu()'s checkmark below.
	m_activeFontId = QString::fromUtf8(choice.id);

	QFont font;
	bool familySet = false;
	for (const QString &family : choice.families) {
		font.setFamily(family);
		if (QFontInfo(font).family() == family) {
			familySet = true;
			break;
		}
	}
	if (!familySet) font.setFamily(QStringLiteral("Arial"));
	font.setPointSize(choice.pointSize);
	font.setWeight(QFont::Normal);
	qApp->setFont(font);

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

	const FontChoice &choice = fontChoiceById(id);
	syncCheckedAction(m_fontActions, QString::fromUtf8(choice.id));

	statusBar()->showMessage(tr("Font: %1").arg(tr(choice.name)), 3000);
}

// Top-level menu, next to Theme - both are "appearance" choices. Same
// exclusive-checkmark QActionGroup pattern as createThemeMenu().
void MainWindow::createFontMenu() {
	QMenu *fontMenu = menuBar()->addMenu(tr("F&ont"));

	auto *group = new QActionGroup(this);
	group->setExclusive(true);

	// Resolved id, not the raw m_startupFontId - see applyFont()'s own
	// comment on why the resolved id is what has to match here too.
	const QString activeId = QString::fromUtf8(fontChoiceById(m_startupFontId).id);
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
}
