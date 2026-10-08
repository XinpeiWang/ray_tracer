#include "crash_recovery.h"

#include "app_log.h"
#include "scene_builder_widget.h"
#include "window_geometry.h"

#include <QApplication>
#include <QDesktopServices>
#include <QFileInfo>
#include <QMessageBox>
#include <QPushButton>
#include <QUrl>

namespace crash_recovery {

QString startFresh() {
	window_geometry::forget();
	window_geometry::forgetSplitters();
	const QString aside = SceneBuilderWidget::setAsideAutosave();
	AppLog::info(QStringLiteral("session"), QStringLiteral("fresh start after an unexpected exit: window layout reset%1")
	                                           .arg(aside.isEmpty() ? QString() : QStringLiteral(", the unsaved scene set aside as %1").arg(aside)));
	return aside;
}

void offerAfterUncleanExit() {
	if (!AppLog::previousSessionEndedUnexpectedly() || qEnvironmentVariableIsSet("RT_GUI_SELFTEST") || qEnvironmentVariableIsSet("RAY_TRACER_NO_CRASH_PROMPT")) return;
	const bool hasScene = SceneBuilderWidget::hasAutosave();
	QMessageBox box(QMessageBox::Warning, QObject::tr("The program did not close properly last time"),
	                QObject::tr("The last session ended unexpectedly: the program crashed or was closed by force. What it was doing is in the log file."));
	box.setInformativeText(hasScene ? QObject::tr("An unsaved Scene Builder scene was found. Starting normally brings it back.\n\n"
	                                              "If the program keeps stopping at start, choose Start fresh: it resets the window layout and sets that scene aside "
	                                              "(it is renamed, not deleted).")
	                                : QObject::tr("If the program keeps stopping at start, choose Start fresh: it resets the window layout."));
	QPushButton *normal = box.addButton(QObject::tr("Start normally"), QMessageBox::AcceptRole);
	QPushButton *fresh = box.addButton(QObject::tr("Start fresh"), QMessageBox::DestructiveRole);
	QPushButton *logs = box.addButton(QObject::tr("Show Log Folder"), QMessageBox::ActionRole);
	box.setDefaultButton(normal);
	for (;;) {
		box.exec();
		if (box.clickedButton() == logs) {
			if (!AppLog::folderPath().isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(AppLog::folderPath()));
			continue;   // the question stays until one of the two real choices is made
		}
		break;
	}
	if (box.clickedButton() == fresh) startFresh();
	else AppLog::info(QStringLiteral("session"), QStringLiteral("after an unexpected exit the user chose to start normally"));
}

}  // namespace crash_recovery
