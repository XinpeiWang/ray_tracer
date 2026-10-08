#ifndef WHEEL_GUARD_H
#define WHEEL_GUARD_H

// Stops the mouse wheel from changing a value by accident. In a scrolling page (the Scene Builder's properties, the Options and Settings tabs) a spin box,
// combo box or slider under the pointer used to take the wheel for itself, so scrolling the page silently changed Radius, Samples or a material. Now such a
// control only reacts to the wheel once it has keyboard focus (a click or a Tab into it); otherwise the wheel scrolls the page behind it, as if the control
// were not there.

#include <QObject>

class WheelGuard : public QObject {
	Q_OBJECT
public:
	explicit WheelGuard(QObject *parent = nullptr) : QObject(parent) {}

	// Installs one on the application (call after QApplication exists).
	static void install();

protected:
	bool eventFilter(QObject *watched, QEvent *event) override;
};

#endif  // WHEEL_GUARD_H
