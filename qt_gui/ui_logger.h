// ui_logger.h - logs what the user does (clicks, choices, typed values, tabs, menus, shortcuts, dialogs) to the application log (app_log.h) without any code
// per control: an event filter on the application that hooks each widget the first time it is polished. UiLogger::install() is called once from main().
#pragma once

#include <QObject>

class QEvent;
class QWidget;

class UiLogger : public QObject {
	Q_OBJECT
public:
	static void install();
	explicit UiLogger(QObject *parent = nullptr) : QObject(parent) {}

protected:
	bool eventFilter(QObject *watched, QEvent *event) override;

private:
	void hook(QWidget *w);
};
