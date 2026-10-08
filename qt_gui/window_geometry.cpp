#include "window_geometry.h"

#include "app_log.h"
#include "settings_keys.h"
#include "../src/shared/window_placement.h"

#include <QGuiApplication>
#include <QScreen>
#include <QSettings>
#include <QSplitter>
#include <QWidget>

namespace window_geometry {
namespace {

const char *kGeometryKey = "ui/windowGeometry";

QSettings settings() { return QSettings(settings_keys::kOrg, settings_keys::kApp); }

std::vector<window_placement::Rect> screenRects() {
	std::vector<window_placement::Rect> out;
	for (const QScreen *s : QGuiApplication::screens()) {
		const QRect r = s->availableGeometry();
		out.push_back({r.x(), r.y(), r.width(), r.height()});
	}
	return out;
}

}  // namespace

bool restore(QWidget *window) {
	const QByteArray saved = settings().value(kGeometryKey).toByteArray();
	if (saved.isEmpty() || !window->restoreGeometry(saved)) return false;
	const QRect f = window->frameGeometry();
	if (window_placement::reachable({f.x(), f.y(), f.width(), f.height()}, screenRects())) {
		AppLog::info(QStringLiteral("window"), QStringLiteral("restored the saved window: %1 x %2 at %3,%4%5").arg(f.width()).arg(f.height()).arg(f.x()).arg(f.y())
		                                          .arg(window->isMaximized() ? QStringLiteral(" (maximized)") : QString()));
		return true;
	}
	AppLog::warn(QStringLiteral("window"), QStringLiteral("the saved window position (%1 x %2 at %3,%4) is not on any connected screen; using the default").arg(f.width()).arg(f.height()).arg(f.x()).arg(f.y()));
	return false;
}

void save(const QWidget *window) {
	if (window->isFullScreen() || window->isMinimized()) return;   // keep the last normal geometry: restoring a minimized one would hide the window
	QSettings s = settings();
	s.setValue(kGeometryKey, window->saveGeometry());
}

void forget() { QSettings s = settings(); s.remove(kGeometryKey); }

void restoreSplitter(QSplitter *splitter, const char *key) {
	const QByteArray saved = settings().value(QString::fromLatin1(key)).toByteArray();
	if (saved.isEmpty()) return;
	const int panes = splitter->count();
	QSplitter probe;   // a state saved with a different number of panes would be applied wrongly: check on a scratch splitter first
	for (int i = 0; i < panes; ++i) probe.addWidget(new QWidget);
	probe.setOrientation(splitter->orientation());
	if (!probe.restoreState(saved)) return;
	splitter->restoreState(saved);
}

void saveSplitter(const QSplitter *splitter, const char *key) {
	QSettings s = settings();
	s.setValue(QString::fromLatin1(key), splitter->saveState());
}

}  // namespace window_geometry
