#include "mainwindow.h"
#include "settings_keys.h"

#include <QAction>
#include <QActionGroup>
#include <QCoreApplication>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QSettings>
#include <QStatusBar>
#include <QStringList>
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
// the ~2,400 combined lines of mainwindow_tabs.cpp/mainwindow_tabs_render.cpp/
// mainwindow_tabs_output.cpp (all still build-once create*Tab() functions,
// just split across three files now) for true live switching is possible
// but a much larger, separate undertaking.
//
// What changed: the app now does the restart itself. switchLanguage()
// spawns a fresh instance of the same executable (which reads the
// just-saved choice on its own startup, same as any cold start) and quits
// this one - the user picks a language and the window briefly closes and
// reopens already translated, instead of having to close and relaunch it
// by hand.
//
// Two things that trick has to get right, both fixed here after review:
// quitting is a real interruption if a render is in flight or jobs are
// queued (see onClearQueue()'s own precedent for treating queue loss as
// worth a confirmation - a running render is more destructive than that,
// yet used to get none at all), and the relaunch can fail to spawn, in
// which case quitting anyway would just end the app with nothing to
// replace it.
// ============================================================================

namespace {
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

// "en" for any code not in kLanguages[] - the same thing main.cpp's
// translator.load() effectively falls back to (a missing/unreadable .qm
// fails soft into running with no translator installed, i.e. English source
// text), so the menu's checkmark agrees with what actually renders instead
// of matching nothing at all.
QString resolveLanguageCode(const QString &code) {
	for (const LanguageInfo &lang : kLanguages) {
		if (code == QLatin1String(lang.code)) return code;
	}
	return QStringLiteral("en");
}
} // namespace

QString MainWindow::loadSavedLanguageCode() {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLanguageKey, QStringLiteral("en")).toString();
}

void MainWindow::saveLanguageCode(const QString &code) {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLanguageKey, code);
}

void MainWindow::switchLanguage(const QString &code) {
	// A second click (misclick-then-correct-choice, or a genuine double
	// click) while the first switch's relaunch timer is still pending would
	// otherwise schedule a second QProcess::startDetached() before the first
	// one's qApp->quit() actually unwinds the event loop, spawning two
	// windows from one user action.
	if (m_languageSwitchPending) return;

	// The relaunch is a real interruption if it happens mid-render or with
	// jobs still queued - m_renderQueue is in-memory only (never round-
	// tripped through QSettings) and MainWindow's destructor stops a running
	// render rather than letting it finish. onClearQueue() already treats
	// losing a queue alone as worth a confirmation ("the one destructive,
	// irreversible action in this app with no undo"); this can lose a queue
	// AND an active render, so it gets at least the same courtesy.
	if (m_isRendering || !m_renderQueue.isEmpty()) {
		QStringList consequences;
		if (m_isRendering)
			consequences << tr("stop the current render");
		if (!m_renderQueue.isEmpty())
			consequences << tr("discard %n queued job(s)", "", m_renderQueue.size());
		const auto choice = QMessageBox::question(this, tr("Switch Language"),
			tr("Switching languages restarts the app now, which will %1. Continue?")
				.arg(consequences.join(tr(" and "))),
			QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
		if (choice != QMessageBox::Yes) return;
	}

	m_languageSwitchPending = true;
	saveLanguageCode(code);

	QString nativeName = code;
	for (const LanguageInfo &lang : kLanguages) {
		if (code == QLatin1String(lang.code)) { nativeName = QString::fromUtf8(lang.nativeName); break; }
	}

	syncCheckedAction(m_languageActions, code);

	statusBar()->showMessage(tr("Language set to %1 - restarting...").arg(nativeName), 5000);

	// A short delay rather than relaunching in the same call: lets the
	// status message above and the menu's own checkmark repaint actually
	// hit the screen (and the triggered QAction's own click handling
	// unwind cleanly) before the window disappears, instead of the click
	// seeming to do nothing right up until the app vanishes.
	QTimer::singleShot(400, this, [this]() {
		const bool started = QProcess::startDetached(QCoreApplication::applicationFilePath(),
													   QCoreApplication::arguments().mid(1));
		if (!started) {
			// Quitting anyway here would end the app with nothing to
			// replace it - the exact failure mode the header comment above
			// warns about. Stay open and say so instead.
			m_languageSwitchPending = false;
			statusBar()->showMessage(
				tr("Could not restart automatically - please close and reopen the app to finish switching languages."),
				8000);
			return;
		}
		qApp->quit();
	});
}

// Top-level menu, same reasoning as createThemeMenu()'s own comment: a
// first-class preference worth one click to reach, not buried under View.
// "L" is free as a mnemonic alongside File/Render/View/Theme/Font/Help.
void MainWindow::createLanguageMenu() {
	QMenu *languageMenu = menuBar()->addMenu(tr("&Language"));

	auto *group = new QActionGroup(this);
	group->setExclusive(true);

	const QString activeCode = resolveLanguageCode(m_startupLanguageCode);
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
