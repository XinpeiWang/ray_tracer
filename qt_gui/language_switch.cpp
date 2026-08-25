#include "mainwindow.h"

#include <QAction>
#include <QActionGroup>
#include <QCoreApplication>
#include <QMenu>
#include <QMenuBar>
#include <QProcess>
#include <QSettings>
#include <QStatusBar>
#include <QTimer>

// ============================================================================
// Language selection and persistence
// ============================================================================
// A language change still needs a fresh process, same underlying reason as
// before: theme colour is a property re-derivable at any time (see
// MainWindow::restyleThemedWidgets()), but text is not - every widget here
// is built once, in one pass, inside the create*Tab() functions called from
// the constructor, with no Designer-generated retranslateUi() split between
// "build the widget tree" and "set its text" that would let a second pass
// safely re-run just the text-setting half. Retrofitting that split across
// ~1,750 lines of mainwindow_tabs.cpp for true live switching is possible
// but a much larger, separate undertaking.
//
// What changed: the app now does the restart itself. switchLanguage()
// spawns a fresh instance of the same executable (which reads the
// just-saved choice on its own startup, same as any cold start) and quits
// this one - the user picks a language and the window briefly closes and
// reopens already translated, instead of having to close and relaunch it
// by hand.
// ============================================================================

namespace {
// Same organisation/app names as theme_switch.cpp's QSettings location -
// deliberately duplicated rather than shared, since this file's
// loadSavedLanguageCode()/saveLanguageCode() are static (callable from
// main.cpp before any MainWindow exists) and reaching into theme_switch.cpp's
// anonymous-namespace constants isn't possible across translation units.
constexpr const char *kSettingsOrg = "RayTracer";
constexpr const char *kSettingsApp = "RayTracerGUI";
constexpr const char *kLanguageKey = "ui/language";

struct LanguageInfo {
	const char *code;         // QTranslator/.qm filename suffix; "en" = built-in source text, no .qm to load
	const char *nativeName;   // Shown in the language's own script, not translated itself
};

// English first (the source language - no .qm involved), then alphabetical
// by native name. Adding a language: add a row here, add its "raytracer_XX"
// .ts file to RayTracerGUI.pro's TRANSLATIONS, translate it, done - nothing
// else in this file changes.
constexpr LanguageInfo kLanguages[] = {
	{"en",    "English"},
	{"fr",    "Français"},
	{"ja",    "日本語"},
	{"es",    "Español"},
	{"zh_CN", "简体中文"},
};
} // namespace

QString MainWindow::loadSavedLanguageCode() {
	QSettings settings(kSettingsOrg, kSettingsApp);
	return settings.value(kLanguageKey, QStringLiteral("en")).toString();
}

void MainWindow::saveLanguageCode(const QString &code) {
	QSettings settings(kSettingsOrg, kSettingsApp);
	settings.setValue(kLanguageKey, code);
}

void MainWindow::switchLanguage(const QString &code) {
	saveLanguageCode(code);

	QString nativeName = code;
	for (const LanguageInfo &lang : kLanguages) {
		if (code == QLatin1String(lang.code)) { nativeName = QString::fromUtf8(lang.nativeName); break; }
	}

	for (QAction *action : m_languageActions)
		action->setChecked(action->data().toString() == code);

	statusBar()->showMessage(tr("Language set to %1 - restarting...").arg(nativeName), 5000);

	// A short delay rather than relaunching in the same call: lets the
	// status message above and the menu's own checkmark repaint actually
	// hit the screen (and the triggered QAction's own click handling
	// unwind cleanly) before the window disappears, instead of the click
	// seeming to do nothing right up until the app vanishes.
	QTimer::singleShot(400, this, []() {
		QProcess::startDetached(QCoreApplication::applicationFilePath(),
								 QCoreApplication::arguments().mid(1));
		qApp->quit();
	});
}

// Top-level menu, same reasoning as createThemeMenu()'s own comment: a
// first-class preference worth one click to reach, not buried under View.
// "L" is free as a mnemonic alongside File/Render/View/Theme/Help.
void MainWindow::createLanguageMenu() {
	QMenu *languageMenu = menuBar()->addMenu(tr("&Language"));

	auto *group = new QActionGroup(this);
	group->setExclusive(true);

	const QString activeCode = loadSavedLanguageCode();
	for (const LanguageInfo &lang : kLanguages) {
		const QString code = QString::fromUtf8(lang.code);
		QAction *action = languageMenu->addAction(QString::fromUtf8(lang.nativeName));
		action->setCheckable(true);
		action->setData(code);
		action->setChecked(code == activeCode);
		group->addAction(action);
		m_languageActions.push_back(action);

		connect(action, &QAction::triggered, this, [this, code]() {
			switchLanguage(code);
		});
	}
}
