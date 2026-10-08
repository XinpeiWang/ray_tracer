#include "mainwindow.h"
#include <QApplication>
#include <QTimer>
#include <QFontDatabase>
#include <QTranslator>
#include <QLibraryInfo>
#include <QDir>
#include <QCoreApplication>
#include <QSettings>
#include <QStandardPaths>
#include <QTemporaryDir>
#include "settings_keys.h"
#include "app_log.h"
#include "ui_logger.h"
#include "wheel_guard.h"
#include "crash_recovery.h"
#include "startup_profile.h"

#include <memory>

int main(int argc, char *argv[]) {
	startup_profile::begin();
	std::unique_ptr<startup_profile::Stage> qtStartup(new startup_profile::Stage("QApplication"));
	QApplication app(argc, argv);
	qtStartup.reset();

	// Self-test mode (RT_GUI_SELFTEST) uses its own settings domain (settings_keys.h); start every run from a clean slate so one run's
	// changes (it enables depth of field, ...) cannot leak into the next.
	if (qEnvironmentVariableIsSet("RT_GUI_SELFTEST")) QSettings(settings_keys::kOrg, settings_keys::kApp).clear();

#ifdef Q_OS_MAC
	// A Finder/Dock launch starts in "/", but the scene registry (scene_metadata.dylib), the Metal renderer and the ray_tracer
	// subprocess all find pbrt_scenes/ and models/ relative to the working directory. From "/" the registry silently got no
	// scenes' cameras (every scene started at (0,0,0) looking down +Z - Live Preview showed a black tile). The bundle puts
	// those folders next to the executable, so start there.
	if (QDir(QCoreApplication::applicationDirPath()).exists("pbrt_scenes")) QDir::setCurrent(QCoreApplication::applicationDirPath());
#endif

	// Guarantees CJK glyph coverage cross-platform regardless of which Font
	// menu choice is active (font_switch.cpp) - must run before MainWindow is
	// constructed, since applyFont() runs during its startup.
	QFontDatabase::addApplicationFont(":/fonts/NotoSansSC-Regular.ttf");

	// Set application info
	app.setApplicationName("Ray Tracer");
	app.setApplicationVersion("2.0");
	app.setOrganizationName("Ray Tracer Project");

	// The log file (app_log.h) and the user-operation logger (ui_logger.h): started once the application's name and version are set, before any window exists.
	AppLog::init();
	UiLogger::install();
	WheelGuard::install();
	{
		QStringList arguments = QCoreApplication::arguments();
		if (arguments.size() > 1) AppLog::info(QStringLiteral("session"), QStringLiteral("arguments: %1").arg(arguments.mid(1).join(QLatin1Char(' '))));
	}

	// Where files the GUI downloads at run time are kept, and where the renderer looks for them after the folder next to
	// the application (see pbrt_asset_check.h). Per-user, so it works when the app itself is read-only (a mounted disk
	// image) or in a protected folder. Exported through the environment so the ray_tracer subprocess and the renderer
	// libraries loaded into this process agree on it; an explicit value (tests, power users) wins.
	// A self-test (RT_GUI_SELFTEST) never touches the real per-user folder: it saves scenes into the scene list and deletes them again, so it gets a
	// throwaway one, removed when the program ends (scripts/gui_selftest.py sets its own).
	QTemporaryDir selfTestAssets;
	if (!qEnvironmentVariableIsSet("RAY_TRACER_USER_ASSETS") && qEnvironmentVariableIsSet("RT_GUI_SELFTEST") && selfTestAssets.isValid())
		qputenv("RAY_TRACER_USER_ASSETS", selfTestAssets.path().toUtf8());
	if (!qEnvironmentVariableIsSet("RAY_TRACER_USER_ASSETS"))
		qputenv("RAY_TRACER_USER_ASSETS", (QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/user_assets")).toUtf8());

	// Language selection is restart-to-apply (see language_switch.cpp's own
	// comment for why) - the translator has to be installed here, before
	// MainWindow builds a single widget, since nothing re-runs the text-
	// setting code afterward. "en" is the source language written directly
	// into every tr() call, so it has no .qm to load - the app just runs
	// with no translator installed, same as before this feature existed.
	// A QTranslator that outlives QApplication::exec() (kept alive on the
	// stack here, not a local inside an if-block) is required - Qt reads it
	// lazily whenever tr() is called, not just once at installTranslator()
	// time - and staying installed for the app's whole lifetime is exactly
	// what a restart-to-apply language choice needs.
	std::unique_ptr<startup_profile::Stage> translatorsStage(new startup_profile::Stage("translators"));
	QTranslator translator;
	// Qt's own strings - the OK/Cancel/Save buttons, QFileDialog, QFontDialog (the
	// "Choose Font" dialog), the right-click Cut/Copy/Paste menu of every text
	// field - live in Qt's qtbase_<code>.qm, not in ours, so without this second
	// translator they stay English next to an otherwise translated UI. Looked
	// for where each platform's deploy tool puts it (windeployqt: translations/
	// beside the exe; macdeployqt: Contents/Resources/translations) and then in
	// the Qt install itself; none found is fine, those strings just stay English.
	QTranslator qtBaseTranslator;
	const QString languageCode = MainWindow::loadSavedLanguageCode();
	if (languageCode != QLatin1String("en")) {
		// "i18n", not "translations" - qmake's CONFIG+=embed_translations
		// (RayTracerGUI.pro) always packages compiled .qm files under an
		// "i18n" resource prefix regardless of where the source .ts files
		// live on disk (qt_gui/translations/), confirmed against the
		// generated release/qmake_qmake_qm_files.qrc.
		if (translator.load(QStringLiteral(":/i18n/raytracer_%1.qm").arg(languageCode)))
			app.installTranslator(&translator);
		// A missing/unreadable .qm silently falls back to English rather than
		// failing to start - the same "fails soft, not loud" choice this
		// codebase's theme loader (palette_file.cpp) makes for a bad .theme file.
		const QString appDir = QCoreApplication::applicationDirPath();
		const QStringList qtTranslationDirs = {appDir + QStringLiteral("/translations"),
			appDir + QStringLiteral("/../Resources/translations"),
			QLibraryInfo::path(QLibraryInfo::TranslationsPath)};
		for (const QString &dir : qtTranslationDirs) {
			if (qtBaseTranslator.load(QStringLiteral("qtbase_%1").arg(languageCode), dir)) {
				app.installTranslator(&qtBaseTranslator);
				break;
			}
		}
	}

	translatorsStage.reset();
	AppLog::info(QStringLiteral("session"), QStringLiteral("language: %1").arg(languageCode));
	{
		startup_profile::Stage crashQuestion("crash question (the user's time, if it was asked)");
		crash_recovery::offerAfterUncleanExit();   // before the window (and the unsaved scene) is loaded
	}
	std::unique_ptr<startup_profile::Stage> windowStage(new startup_profile::Stage("main window"));
	MainWindow window(nullptr, languageCode);
	windowStage.reset();
	{
		startup_profile::Stage show("show");
		window.show();
	}
	// Once the event loop is running the window is up: write the start-up timings to the log.
	QTimer::singleShot(0, &window, []() { startup_profile::ready("window up"); });

	// Opt-in automated smoke test of the real window (see mainwindow_selftest.cpp); a no-op unless RT_GUI_SELFTEST is set.
	if (qEnvironmentVariableIsSet("RT_GUI_SELFTEST")) {
		const QString mode = qEnvironmentVariable("RT_GUI_SELFTEST");
		const QString out = qEnvironmentVariable("RT_GUI_SELFTEST_OUT", "gui_selftest");
		QTimer::singleShot(1500, &window, [&window, mode, out]() { window.runSelfTest(mode, out); });
	}

	return app.exec();
}
