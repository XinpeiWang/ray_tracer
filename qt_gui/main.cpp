#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);

	// Set application info
	app.setApplicationName("Ray Tracer");
	app.setApplicationVersion("2.0");
	app.setOrganizationName("Ray Tracer Project");

	MainWindow window;
	window.show();

	return app.exec();
}
