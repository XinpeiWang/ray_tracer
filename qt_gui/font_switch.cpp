#include "mainwindow.h"

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
// re-applies the app's own stylesheet (see its own comment) - a stylesheet-
// styled widget's computed appearance, once polished under the active style
// (Fusion) and this app's global QSS, does not retroactively pick up a
// later application-font change on its own. applyTheme() happened to get
// away with calling qApp->setFont() inline for years because it always ran
// alongside a real qApp->setStyleSheet(...) rebuild in the same function -
// the stylesheet rebuild was silently doing the work, not the font call by
// itself. This file lifts the font logic out into its own independently
// switchable system, so Theme and Font stop being entangled: switching one
// no longer resets the other, the same independence Theme and Language
// already have.
//
// No font files are bundled with this app (no .ttf/.otf, no
// QFontDatabase::addApplicationFont), so every choice below is a fallback
// chain of commonly-installed names, the same shape the original
// hardcoded Cyberpunk list already used - first one QFontInfo confirms is
// actually installed wins.
// ============================================================================

namespace {
// Same organisation/app names as theme_switch.cpp's/language_switch.cpp's
// QSettings location - see language_switch.cpp's own comment on why this is
// duplicated per file rather than shared.
constexpr const char *kSettingsOrg = "RayTracer";
constexpr const char *kSettingsApp = "RayTracerGUI";
constexpr const char *kFontKey = "ui/font";
constexpr const char *kDefaultFontId = "cyberpunk";

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
	QSettings settings(kSettingsOrg, kSettingsApp);
	return settings.value(kFontKey, QString::fromUtf8(kDefaultFontId)).toString();
}

void MainWindow::saveFontId(const QString &id) {
	QSettings settings(kSettingsOrg, kSettingsApp);
	settings.setValue(kFontKey, id);
}

// Walks a FontChoice's fallback chain and applies the first family QFontInfo
// confirms is actually installed - identical logic to what used to live
// inline in applyTheme(), just parameterized instead of hardcoded to one
// choice. Weight stays Normal app-wide (see the original comment this
// preserves): bold is opt-in, on group-box titles and the primary button
// only, not blanket-applied the way an earlier version of this app did.
void MainWindow::applyFont(const QString &id) {
	const FontChoice &choice = fontChoiceById(id);

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

	// qApp->setFont() alone only reaches widgets created AFTER this call -
	// every widget already on screen was already polished by the active
	// style (Fusion, plus this app's own global stylesheet from
	// applyTheme()) using whatever font was current at THAT time, and a
	// stylesheet-driven widget's computed appearance doesn't retroactively
	// invalidate on a later application-font change alone. Re-setting the
	// same stylesheet string is Qt's standard way to force a full
	// unpolish+polish pass across every styled widget, which is what
	// actually makes the new font (not just the new QFont object) show up.
	if (!qApp->styleSheet().isEmpty())
		qApp->setStyleSheet(qApp->styleSheet());
}

void MainWindow::switchFont(const QString &id) {
	applyFont(id);
	saveFontId(id);

	const FontChoice &choice = fontChoiceById(id);
	for (QAction *action : m_fontActions)
		action->setChecked(action->data().toString() == QLatin1String(choice.id));

	statusBar()->showMessage(tr("Font: %1").arg(tr(choice.name)), 3000);
}

// Top-level menu, next to Theme - both are "appearance" choices. Same
// exclusive-checkmark QActionGroup pattern as createThemeMenu().
void MainWindow::createFontMenu() {
	QMenu *fontMenu = menuBar()->addMenu(tr("F&ont"));

	auto *group = new QActionGroup(this);
	group->setExclusive(true);

	const QString activeId = loadSavedFontId();
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
