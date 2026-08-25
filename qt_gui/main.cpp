#include "mainwindow.h"
#include <QApplication>
#include <QTranslator>

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);

	// Set application info
	app.setApplicationName("Ray Tracer");
	app.setApplicationVersion("2.0");
	app.setOrganizationName("Ray Tracer Project");

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
	QTranslator translator;
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
	}

	MainWindow window(nullptr, languageCode);
	window.show();

	return app.exec();
}
