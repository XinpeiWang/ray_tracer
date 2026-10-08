#include "wheel_guard.h"

#include <QAbstractSpinBox>
#include <QApplication>
#include <QComboBox>
#include <QSlider>
#include <QWheelEvent>
#include <QWidget>

void WheelGuard::install() {
	if (qApp) qApp->installEventFilter(new WheelGuard(qApp));
}

bool WheelGuard::eventFilter(QObject *watched, QEvent *event) {
	if (event->type() != QEvent::Wheel) return false;
	auto *w = qobject_cast<QWidget *>(watched);
	if (!w) return false;
	const bool guarded = qobject_cast<QAbstractSpinBox *>(w) || qobject_cast<QComboBox *>(w) || qobject_cast<QSlider *>(w);
	if (!guarded || w->hasFocus()) return false;
	// An editable combo box or a spin box has a line edit inside it that holds the focus.
	if (QWidget *focus = QApplication::focusWidget(); focus && w->isAncestorOf(focus)) return false;

	// Not focused: hand the wheel to whatever contains the control (usually a scroll area's viewport), as if this widget had ignored it. Passed up by hand,
	// widget by widget, until one accepts it (Qt only does that itself for the event it was first given).
	auto *wheel = static_cast<QWheelEvent *>(event);
	for (QWidget *parent = w->parentWidget(); parent; parent = parent->parentWidget()) {
		QWheelEvent forwarded(parent->mapFromGlobal(wheel->globalPosition().toPoint()), wheel->globalPosition(), wheel->pixelDelta(), wheel->angleDelta(),
		                      wheel->buttons(), wheel->modifiers(), wheel->phase(), wheel->inverted(), wheel->source());
		forwarded.setAccepted(false);
		QApplication::sendEvent(parent, &forwarded);
		if (forwarded.isAccepted()) break;
	}
	return true;
}
